#!/usr/bin/env python3
"""Train/evaluate the same global leaf head from base, SFT, or mixed-router weights.

Only task content, observed images/state, output requirements, and task-scoped
capability definitions enter the encoder. Solver scores are evaluation-only.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import sys

from model_policy import filter_record, require_current_table
from routing_policy import (
    UNRESOLVED_CLASS_ID, route_prediction,
)


LEAF_FIELDS = (
    "leaf_id", "name_zh", "definition", "requirement_axes", "task_specific_operations",
    "suboperation_scope", "evidence_rule", "exclusions", "when_to_choose",
    "distinguishing_requirements", "assignment_rule", "excludes_competing_requirements",
)
SYSTEM_PROMPT = (
    "Classify the task into one of the supplied capability leaves or the reserved unresolved class. The input track and task "
    "are already fixed. Choose the most specific supplied leaf covering the task's complete "
    "required operations, using only the task, observed content, and output requirements. "
    "Use a compound profile when its additional operations are required. Select unresolved when "
    "the observed task information is insufficient to determine its requirements, or when no supplied "
    "leaf covers all required operations."
)


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def read_split(directory, split, benchmarks=None):
    directory = Path(directory)
    path = directory / f"{split}.jsonl"
    split_file = path.exists()
    if not split_file:
        path = directory / "records.jsonl"
    if not path.exists() and split == "special_test":
        return []
    selected = set(benchmarks or [])
    with path.open() as stream:
        return [row for line in stream if line.strip()
                for row in [filter_record(json.loads(line))]
                if (split_file or row["split"] == split)
                and (not selected or row["benchmark"] in selected)]


def supervision_class(row):
    """An annotated null is supervision; pending/failed annotation is not."""
    candidates = row["candidate_leaf_ids"]
    if not candidates:
        return None
    if row.get("label_status") == "teacher_labeled" and row.get("leaf_id") in candidates:
        return row["leaf_id"]
    if row.get("label_status") == "unresolved_requirements" and row.get("leaf_id") is None:
        return UNRESOLVED_CLASS_ID
    return None


def classification_candidates(row):
    return list(row["candidate_leaf_ids"]) + [UNRESOLVED_CLASS_ID]


class CapabilityCatalog:
    def __init__(self, tree):
        self.scopes = {}
        self.leaves = {}
        for track in tree["tracks"]:
            for task in track["tasks"]:
                self.scopes[(track["track_id"], task["task_id"])] = task
                for leaf in task["capabilities"]:
                    self.leaves[leaf["leaf_id"]] = {key: leaf[key] for key in LEAF_FIELDS if key in leaf}
        self.leaf_ids = list(self.leaves)

    def definition(self, row):
        task = self.scopes[(row["track_id"], row["tree_task_id"])]
        return {
            "track_id": row["track_id"], "task_id": row["tree_task_id"],
            "task_description": task["task_description"],
            "output_requirements": row["output_contract"],
            "candidate_capabilities": [self.leaves[leaf_id] for leaf_id in row["candidate_leaf_ids"]],
            "unresolved_option": {"class_id": UNRESOLVED_CLASS_ID,
                                  "definition": "Available input does not determine a complete supplied capability profile, or the required combination has no supplied leaf."},
        }

    def validate(self, rows):
        for row in rows:
            candidates = row["candidate_leaf_ids"]
            if not candidates:
                continue
            task = self.scopes[(row["track_id"], row["tree_task_id"])]
            allowed = {leaf["leaf_id"] for leaf in task["capabilities"]}
            if len(candidates) != len(set(candidates)) or not set(candidates).issubset(allowed):
                raise ValueError(f"Invalid task-scoped candidates for {row['id']}")
            if row.get("leaf_id") is not None and row["leaf_id"] not in candidates:
                raise ValueError(f"Label outside candidate leaves for {row['id']}")


def load_processor(args):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.processor_path or args.model_path, local_files_only=False)
    template = Path(args.chat_template_path).read_text() if args.chat_template_path else (
        processor.chat_template or processor.tokenizer.chat_template)
    processor.chat_template = template
    processor.tokenizer.chat_template = template
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    return processor


def retained_text(tokenizer, ids, count):
    if count >= len(ids):
        return tokenizer.decode(ids, skip_special_tokens=False)
    left = (count + 1) // 2
    right = count // 2
    return (tokenizer.decode(ids[:left], skip_special_tokens=False) + "\n[Task content cropped]\n"
            + (tokenizer.decode(ids[-right:], skip_special_tokens=False) if right else ""))


def encode_record(row, processor, catalog, args):
    """Keep complete capability definitions and image grids when cropping text."""
    import torch
    from PIL import Image

    tokenizer = processor.tokenizer
    all_paths = row.get("images", [])
    paths = all_paths[:args.max_images] if args.max_images else all_paths
    images = []
    for path in paths:
        path = Path(path)
        path = path if path.is_absolute() else Path(args.data_dir) / path
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    task_text = row["task_text"]
    state_text = row.get("state_text") or ""
    if not isinstance(state_text, str):
        state_text = json.dumps(state_text, ensure_ascii=False)
    task_ids = tokenizer.encode(task_text, add_special_tokens=False)
    state_ids = tokenizer.encode(state_text, add_special_tokens=False)
    task_retained, state_retained = len(task_ids), len(state_ids)
    definitions = json.dumps(catalog.definition(row), ensure_ascii=False, separators=(",", ":"))

    def render(task, state):
        content = []
        for index in range(len(images)):
            content.extend([{"type": "text", "text": f"Observed image {index + 1}: "},
                            {"type": "image"}, {"type": "text", "text": "\n"}])
        observed = f"Task:\n{task}"
        if state:
            observed += f"\n\nObserved state:\n{state}"
        content.append({"type": "text", "text": observed + "\n\nCapability definitions and output requirements:\n" + definitions})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    prefix = render(task_text, state_text)
    options = dict(text=prefix, return_tensors="pt", add_special_tokens=False, return_mm_token_type_ids=True)
    if images:
        limit = min(args.max_image_tokens, args.image_token_budget // len(images))
        factor = processor.image_processor.patch_size * processor.image_processor.merge_size
        options.update(images=images, images_kwargs={"size": {
            "shortest_edge": min(64, limit) * factor**2, "longest_edge": limit * factor**2,
        }})
    encoded = processor(**options)
    grids = encoded.get("image_grid_thw", [])
    image_lengths = [int(grid.prod()) // processor.image_processor.merge_size**2 for grid in grids]
    original_tokens = int(encoded["input_ids"].shape[1])
    while encoded["input_ids"].shape[1] > args.seq_length:
        remove = int(encoded["input_ids"].shape[1]) - args.seq_length + 16
        crop_task = min(task_retained, remove)
        task_retained -= crop_task
        state_retained = max(0, state_retained - (remove - crop_task))
        prefix = render(retained_text(tokenizer, task_ids, task_retained),
                        retained_text(tokenizer, state_ids, state_retained) if state_ids else "")
        parts = prefix.split("<|image_pad|>")
        expanded = parts[0]
        for length, part in zip(image_lengths, parts[1:], strict=True):
            expanded += "<|image_pad|>" * length + part
        tokenized = tokenizer(expanded, add_special_tokens=False, return_tensors="pt")
        encoded["input_ids"] = tokenized["input_ids"]
        encoded["attention_mask"] = tokenized["attention_mask"]
        encoded["mm_token_type_ids"] = torch.zeros_like(encoded["input_ids"])
        encoded["mm_token_type_ids"][encoded["input_ids"] == tokenizer.convert_tokens_to_ids("<|image_pad|>")] = 1
        if not task_retained and not state_retained and encoded["input_ids"].shape[1] > args.seq_length:
            raise ValueError(f"Definitions and images exceed --seq-length for {row['id']}; increase context length.")
    features = {key: encoded[key] for key in (
        "input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw") if key in encoded}
    info = {
        "input_tokens": int(encoded["input_ids"].shape[1]), "original_input_tokens": original_tokens,
        "cropped_task_tokens": len(task_ids) - task_retained,
        "cropped_state_tokens": len(state_ids) - state_retained,
        "images": len(images), "omitted_images": len(all_paths) - len(images),
        "image_tokens": sum(image_lengths),
    }
    return features, info


def autocast(device):
    import torch
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else nullcontext()


def predict_logits(model, row, processor, catalog, args):
    features, info = encode_record(row, processor, catalog, args)
    inputs = {key: value.to(args.device) for key, value in features.items()}
    with autocast(args.device):
        logits = model.scoped_logits(model(**inputs), row["candidate_leaf_ids"])
    return logits, info


def mean(values):
    return sum(values) / len(values) if values else None


def summarize_predictions(predictions):
    annotated = [row for row in predictions if row["supervision_class_id"] is not None]
    labelled = [row for row in annotated if row["supervision_class_id"] != UNRESOLVED_CLASS_ID]
    unresolved = [row for row in annotated if row["supervision_class_id"] == UNRESOLVED_CLASS_ID]
    predicted_unresolved = [row for row in annotated if row["predicted_class_id"] == UNRESOLVED_CLASS_ID]
    learned_scope = [row for row in labelled if row["candidate_count"] > 1]
    covered = [row for row in predictions if row["router_quality"] is not None]
    fixed = [row for row in predictions if row["fixed_quality"] is not None]
    panel = [row for row in predictions if row["oracle_quality"] is not None]
    shared = [row for row in covered if row["fixed_quality"] is not None]
    n = len(predictions)
    return {
        "rows": n, "labelled_rows": len(labelled),
        "annotation_rows": len(annotated),
        "annotation_accuracy": mean([float(row["predicted_class_id"] == row["supervision_class_id"]) for row in annotated]),
        "annotation_cross_entropy": mean([row["annotation_cross_entropy"] for row in annotated]),
        "scope_unresolved_rows": sum(not row["candidate_count"] for row in predictions),
        "predicted_unresolved_rows": sum(row["predicted_class_id"] == UNRESOLVED_CLASS_ID for row in predictions),
        "unresolved_labelled_rows": len(unresolved),
        "unresolved_recall": mean([float(row["predicted_class_id"] == UNRESOLVED_CLASS_ID) for row in unresolved]),
        "unresolved_precision": mean([float(row["supervision_class_id"] == UNRESOLVED_CLASS_ID) for row in predicted_unresolved]),
        "leaf_accuracy": mean([float(row["predicted_leaf_id"] == row["true_leaf_id"]) for row in labelled]),
        "multi_leaf_labelled_rows": len(learned_scope),
        "multi_leaf_accuracy": mean([float(row["predicted_leaf_id"] == row["true_leaf_id"]) for row in learned_scope]),
        "leaf_cross_entropy": mean([row["leaf_cross_entropy"] for row in labelled]),
        "mapped_rows": sum(row["selected_model"] is not None for row in predictions),
        "routing_covered_rows": len(covered), "routing_coverage": len(covered) / n if n else None,
        "routing_quality": mean([row["router_quality"] for row in covered]),
        "panel_rows": len(panel), "panel_oracle_quality": mean([row["oracle_quality"] for row in panel]),
        "same_panel_oracle_quality": mean([row["oracle_quality"] for row in covered]),
        "same_panel_oracle_regret": mean([row["oracle_quality"] - row["router_quality"] for row in covered]),
        "fixed_covered_rows": len(fixed), "fixed_coverage": len(fixed) / n if n else None,
        "fixed_quality": mean([row["fixed_quality"] for row in fixed]),
        "shared_panel_rows": len(shared),
        "shared_panel_router_quality": mean([row["router_quality"] for row in shared]),
        "shared_panel_fixed_quality": mean([row["fixed_quality"] for row in shared]),
        "shared_panel_oracle_quality": mean([row["oracle_quality"] for row in shared]),
        "input_tokens": sum(row["encoding"]["input_tokens"] for row in predictions),
        "cropped_rows": sum(bool(row["encoding"]["cropped_task_tokens"] or row["encoding"]["cropped_state_tokens"]) for row in predictions),
        "omitted_images": sum(row["encoding"]["omitted_images"] for row in predictions),
    }


def evaluate(model, rows, processor, catalog, table, args, path, split):
    import torch
    import torch.nn.functional as F

    model.eval()
    predictions, counts = [], Counter()
    with Path(path).open("w") as stream, torch.inference_mode():
        for row in rows:
            benchmark = row["benchmark"]
            if args.max_eval_records and counts[benchmark] >= args.max_eval_records:
                continue
            candidates = row["candidate_leaf_ids"]
            label = row.get("leaf_id")
            target_class = supervision_class(row)
            if candidates:
                logits, info = predict_logits(model, row, processor, catalog, args)
                classes = classification_candidates(row)
                index = int(logits.argmax(dim=-1).item())
                predicted_class = classes[index]
                predicted_leaf = None if predicted_class == UNRESOLVED_CLASS_ID else predicted_class
                confidence = float(logits.softmax(dim=-1)[0, index].item())
                loss = (float(F.cross_entropy(logits, torch.tensor([classes.index(target_class)], device=args.device)).item())
                        if target_class is not None else None)
                status = "unresolved_requirements" if predicted_leaf is None else "predicted_leaf"
            else:
                predicted_class, predicted_leaf, confidence, loss = None, None, None, None
                status = "scope_unresolved"
                info = {key: 0 for key in ("input_tokens", "original_input_tokens", "cropped_task_tokens",
                                          "cropped_state_tokens", "images", "omitted_images", "image_tokens")}
            prediction = {
                "id": row["id"], "benchmark": benchmark, "split": split,
                "track_id": row["track_id"], "tree_task_id": row["tree_task_id"],
                "candidate_count": len(candidates), "true_leaf_id": label,
                "class_count": len(candidates) + 1 if candidates else 0,
                "label_status": row.get("label_status"), "supervision_class_id": target_class,
                "prediction_status": status, "predicted_class_id": predicted_class,
                "predicted_leaf_id": predicted_leaf, "annotation_cross_entropy": loss,
                "leaf_cross_entropy": loss if target_class not in (None, UNRESOLVED_CLASS_ID) else None,
                "confidence": confidence,
                "encoding": info,
            }
            prediction.update(route_prediction(prediction, row, table))
            stream.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
            predictions.append(prediction)
            counts[benchmark] += 1
            if len(predictions) % 100 == 0:
                print(json.dumps({"event": "evaluation", "split": split, "rows": len(predictions)}), flush=True)
    by_benchmark = defaultdict(list)
    for row in predictions:
        by_benchmark[row["benchmark"]].append(row)
    benchmarks = {key: summarize_predictions(value) for key, value in sorted(by_benchmark.items())}
    return {
        "overall": summarize_predictions(predictions), "by_benchmark": benchmarks,
        "macro_routing_quality": mean([value["routing_quality"] for value in benchmarks.values()
                                       if value["routing_quality"] is not None]),
        "max_records_per_benchmark": args.max_eval_records or None,
        "score_contract": "Unclassified inputs use the fixed benchmark model (selected by observed train mean). Per-run logs retain score coverage diagnostics; final cross-method comparisons exclude missing selected-model outcomes from every column.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--benchmark", action="append", help="Repeat to select benchmarks; default is all.")
    parser.add_argument("--init-checkpoint", help="Load a mixed-run adapter and global leaf head before specialization.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0, help="Limit optimizer updates for a smoke run.")
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--max-eval-records", type=int, default=0, help="Smoke limit PER benchmark, including dev and test.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--effective-batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--max-image-tokens", type=int, default=256)
    parser.add_argument("--image-token-budget", type=int, default=2048)
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--processor-path", help="Official multimodal processor when the SFT folder has no processor files.")
    parser.add_argument("--chat-template-path")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate --init-checkpoint without further training.")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from model import load_model

    started = time.monotonic()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists() or (out / "checkpoint").exists():
        raise ValueError("Use a new output directory for each matrix run.")
    catalog = CapabilityCatalog(json.loads(Path(args.tree).read_text()))
    table = require_current_table(json.loads(Path(args.table).read_text()))
    datasets = {split: read_split(args.data_dir, split, args.benchmark)
                for split in ("train", "dev", "test", "special_test")}
    for rows in datasets.values():
        catalog.validate(rows)
    learnable = [row for row in datasets["train"] if supervision_class(row) is not None]
    train_rows = learnable[:args.max_train_records] if args.max_train_records else learnable
    counts = {
        "train_rows": len(datasets["train"]), "labelled_rows": len(learnable),
        "leaf_labelled_rows": sum(supervision_class(row) != UNRESOLVED_CLASS_ID for row in learnable),
        "unresolved_labelled_rows": sum(supervision_class(row) == UNRESOLVED_CLASS_ID for row in learnable),
        "single_leaf_rows": sum(len(row["candidate_leaf_ids"]) == 1 for row in learnable),
        "learnable_rows": len(learnable),
        "used_train_rows": len(train_rows),
        "masked_rows": len(datasets["train"]) - len(learnable),
        "scope_unresolved_rows": sum(not row["candidate_leaf_ids"] for row in datasets["train"]),
        "by_benchmark": dict(sorted(Counter(row["benchmark"] for row in datasets["train"]).items())),
    }
    planned_steps = args.epochs * math.ceil(len(train_rows) / args.effective_batch)
    if args.max_steps:
        planned_steps = min(planned_steps, args.max_steps)
    if args.eval_only:
        if not args.init_checkpoint:
            raise ValueError("--eval-only requires --init-checkpoint.")
        planned_steps = 0
    plan = {
        "schema_version": "capability-tree-router-matrix-v2", "arguments": vars(args),
        "global_leaf_count": len(catalog.leaf_ids), "training_data": counts,
        "classifier_class_count": len(catalog.leaf_ids) + 1, "unresolved_class_id": UNRESOLVED_CLASS_ID,
        "optimizer_updates": planned_steps, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "input_contract": "Task, observed images/state, output requirements and candidate capability definitions only.",
        "single_leaf_policy": "Every scoped task includes its actual leaves plus the reserved unresolved class; single-leaf tasks are trainable.",
        "supervision_policy": "teacher_labeled leaves and unresolved_requirements nulls are supervised; all other annotation statuses are masked.",
        "unresolved_policy": "The reserved class and rows without a resolved scope retain a null leaf and select the fixed benchmark model (selected by observed train mean). Classifier outputs are unchanged; missing selected-model scores are excluded from the comparison panel.",
        "checkpoint_selection": "Minimum dev annotation cross entropy over leaf and unresolved supervision; latest epoch only when dev has no annotations.",
    }
    dump(out / "plan.json", plan)
    print(json.dumps({"event": "loading", "model_path": args.model_path, "training_data": counts,
                      "global_leaf_count": len(catalog.leaf_ids)}, ensure_ascii=False), flush=True)
    processor = load_processor(args)
    model = load_model(args.model_path, catalog.leaf_ids, args.device, args.init_checkpoint, args.lora_r)
    optimizer = torch.optim.AdamW([
        {"params": [value for value in model.encoder.parameters() if value.requires_grad], "base_lr": args.lr},
        {"params": list(model.leaf_head.parameters()), "base_lr": args.head_lr},
    ], lr=args.lr, weight_decay=0.01)
    checkpoint = out / "checkpoint"
    curves, best_loss, best_epoch, step, seen = [], float("inf"), None, 0, Counter()
    rng = random.Random(args.seed)
    epochs = args.epochs if planned_steps else 1
    for epoch in range(1, epochs + 1):
        ordered = list(train_rows) if planned_steps else []
        rng.shuffle(ordered)
        for offset in range(0, len(ordered), args.effective_batch):
            if step >= planned_steps:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch = ordered[offset:offset + args.effective_batch]
            loss_value = 0.0
            for row in batch:
                logits, info = predict_logits(model, row, processor, catalog, args)
                target = torch.tensor([classification_candidates(row).index(supervision_class(row))], device=args.device)
                loss = F.cross_entropy(logits, target) / len(batch)
                loss.backward()
                loss_value += float(loss.detach().item())
                seen.update(info)
                seen["examples"] += 1
                seen["unresolved_supervised_examples"] += int(supervision_class(row) == UNRESOLVED_CLASS_ID)
                if info["cropped_task_tokens"] or info["cropped_state_tokens"] or info["omitted_images"]:
                    append(out / "input_crops.jsonl", {"id": row["id"], "epoch": epoch, **info})
            step += 1
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            fraction = (step - 0.5) / planned_steps
            factor = fraction / 0.05 if fraction < 0.05 else (
                0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (fraction - 0.05) / 0.95)))
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor
            optimizer.step()
            progress = {"event": "training", "epoch": epoch, "step": step, "steps": planned_steps,
                        "loss": loss_value, "gradient_norm": norm, "lr": args.lr * factor,
                        "head_lr": args.head_lr * factor, **dict(seen)}
            dump(out / "progress.json", progress)
            append(out / "metrics.jsonl", progress)
            print(json.dumps(progress), flush=True)
        dev = evaluate(model, datasets["dev"], processor, catalog, table, args,
                       out / f"predictions-dev-{epoch:02d}.jsonl", "dev")
        dev_loss = dev["overall"]["annotation_cross_entropy"]
        selected = best_epoch is None or dev_loss is None or dev_loss < best_loss
        if selected:
            best_epoch = epoch
            if dev_loss is not None:
                best_loss = dev_loss
            model.save_checkpoint(checkpoint, {"model_path": args.model_path, "epoch": epoch,
                                               "step": step, "dev_annotation_cross_entropy": dev_loss})
            processor.save_pretrained(checkpoint / "processor")
            shutil.copyfile(args.table, checkpoint / "routing_table.json")
            shutil.copyfile(args.tree, checkpoint / "capability_tree.json")
            dump(checkpoint / "training_plan.json", plan)
        curves.append({"epoch": epoch, "step": step, "selected": selected, "dev": dev})
        dump(out / "curve.json", curves)
        print(json.dumps({"event": "epoch_complete", "epoch": epoch, "step": step,
                          "dev_annotation_cross_entropy": dev_loss, "best_epoch": best_epoch}), flush=True)
        if step >= planned_steps:
            break
    model.restore_weights(checkpoint)
    test = evaluate(model, datasets["test"], processor, catalog, table, args,
                    out / "predictions-test.jsonl", "test")
    special = (evaluate(model, datasets["special_test"], processor, catalog, table, args,
                        out / "predictions-special_test.jsonl", "special_test") if datasets["special_test"] else None)
    result = {
        "status": "completed", "schema_version": plan["schema_version"],
        "checkpoint": str(checkpoint.resolve()), "best_epoch": best_epoch,
        "best_dev_annotation_cross_entropy": None if best_loss == float("inf") else best_loss,
        "checkpoint_selection": plan["checkpoint_selection"], "training_data": counts,
        "optimizer_updates": step, "training_totals": dict(seen),
        "training_note": "No supervised leaf or unresolved annotations; initialization preserved." if not train_rows else None,
        "smoke": bool(args.max_steps or args.max_train_records or args.max_eval_records),
        "test": test, "special_test": special, "curve": curves,
        "total_seconds": time.monotonic() - started,
    }
    dump(out / "result.json", result)
    print(json.dumps({"event": "complete", "checkpoint": str(checkpoint), "best_epoch": best_epoch,
                      "test": test["overall"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
