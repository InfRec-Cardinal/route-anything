#!/usr/bin/env python3
"""Train seven independent RoboTwin success probabilities from all observed outcomes."""
import argparse
from collections import Counter
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

SYSTEM_PROMPT = (
    "Read the robot task, its initial camera observations, current robot state, "
    "and required action output. Represent the operations needed to complete the task."
)

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

def read_rows(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]

def global_batches(rows, epoch, seed, batch_size):
    ordered = list(rows)
    random.Random(seed + epoch).shuffle(ordered)
    return [ordered[start:start + batch_size] for start in range(0, len(ordered), batch_size)]

def rank_microbatches(batch, rank, world_size):
    """Equal forward counts; padding has zero loss and never adds an exposure."""
    result = []
    for offset in range(0, len(batch), world_size):
        index = offset + rank
        result.append((batch[index] if index < len(batch) else batch[0], index < len(batch)))
    return result

def mean(values):
    return sum(values) / len(values) if values else None

def load_processor(args):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.processor_path or args.model_path, local_files_only=False)
    template = (Path(args.chat_template_path).read_text() if args.chat_template_path else
                processor.chat_template or processor.tokenizer.chat_template)
    processor.chat_template = template
    processor.tokenizer.chat_template = template
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    return processor

def encode_record(row, processor, args):
    """The identical mode-independent multimodal input used by both routers."""
    from PIL import Image

    images = []
    for path in row["images"]:
        path = Path(path)
        path = path if path.is_absolute() else args.data_dir / path
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    state = row.get("state_text") or ""
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    task_ids = processor.tokenizer.encode(row["task_text"], add_special_tokens=False)
    retained = len(task_ids)
    task_text = row["task_text"]
    contract = json.dumps(row["output_contract"], ensure_ascii=False, separators=(",", ":"))
    while True:
        content = []
        for index in range(len(images)):
            content.extend([{"type": "text", "text": f"Initial camera {index + 1}: "},
                            {"type": "image"}, {"type": "text", "text": "\n"}])
        content.append({"type": "text", "text":
                        f"Robot task:\n{task_text}\n\nInitial robot state:\n{state}\n\nRequired action output:\n{contract}"})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
        prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        options = dict(text=prefix, return_tensors="pt", add_special_tokens=False, return_mm_token_type_ids=True)
        if images:
            limit = min(args.max_image_tokens, args.image_token_budget // len(images))
            factor = processor.image_processor.patch_size * processor.image_processor.merge_size
            options.update(images=images, images_kwargs={"size": {
                "shortest_edge": min(64, limit) * factor**2, "longest_edge": limit * factor**2,
            }})
        encoded = processor(**options)
        tokens = int(encoded["input_ids"].shape[1])
        if tokens <= args.seq_length:
            break
        remove = tokens - args.seq_length + 16
        retained = max(0, retained - remove)
        if not retained:
            raise ValueError(f"Initial state and images exceed sequence length for {row['id']}.")
        left = (retained + 1) // 2
        right = retained // 2
        task_text = (processor.tokenizer.decode(task_ids[:left], skip_special_tokens=False)
                     + "\n[Task text cropped]\n"
                     + (processor.tokenizer.decode(task_ids[-right:], skip_special_tokens=False) if right else ""))
    features = {key: encoded[key] for key in (
        "input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw") if key in encoded}
    return features, {
        "input_tokens": tokens, "images": len(images), "cropped_task_tokens": len(task_ids) - retained,
        "image_tokens": sum(int(grid.prod()) // processor.image_processor.merge_size**2
                            for grid in encoded.get("image_grid_thw", [])),
    }

def forward_row(model, row, processor, args, device):
    import torch

    features, info = encode_record(row, processor, args)
    inputs = {key: value.to(device) for key, value in features.items()}
    cast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with cast:
        logits = model(**inputs)
    return logits, info


def quality_targets(row, model_ids):
    """Every observed solver result is a target; no winner or capability labels."""
    return [row["model_scores"][name] for name in model_ids]


def success_loss(logits, targets):
    import torch.nn.functional as F

    # First average seven candidates within a scene, then average scenes.
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(-1).mean()


def better_dev(dev, best):
    """Prefer routing success, then BCE; exact ties keep the earlier epoch."""
    return (best is None or dev["routing_success_rate"] > best["routing_success_rate"]
            or (dev["routing_success_rate"] == best["routing_success_rate"] and dev["bce"] < best["bce"]))


def prediction_record(row, probabilities, bce, model_ids, fixed_model, encoding):
    chosen = max(range(len(model_ids)), key=lambda index: probabilities[index])
    selected = model_ids[chosen]
    scores = row["model_scores"]
    latencies = row.get("model_latencies", {})
    return {
        "id": row["id"], "group_id": row["group_id"],
        "model_probabilities": dict(zip(model_ids, probabilities)),
        "model_scores": {name: scores[name] for name in model_ids},
        "selected_model": selected, "selected_quality": scores[selected],
        "selected_probability": probabilities[chosen],
        "selected_latency_seconds": latencies.get(selected),
        "fixed_model": fixed_model, "fixed_quality": scores[fixed_model],
        "fixed_latency_seconds": latencies.get(fixed_model),
        "oracle_quality": max(scores[name] for name in model_ids),
        "bce": bce,
        "brier": mean([(probabilities[index] - scores[name]) ** 2 for index, name in enumerate(model_ids)]),
        "encoding": encoding,
    }


def summarize(predictions, model_ids):
    n = len(predictions)
    return {
        "rows": n, "supervised_rows": n, "supervised_candidate_outcomes": n * len(model_ids),
        "routing_success_rate": mean([r["selected_quality"] for r in predictions]),
        "routing_successes": sum(r["selected_quality"] for r in predictions),
        "routing_coverage": 1.0 if n else None,
        "routing_mean_latency_seconds": mean([r["selected_latency_seconds"] for r in predictions
                                              if r["selected_latency_seconds"] is not None]),
        "fixed_model": predictions[0]["fixed_model"] if n else None,
        "fixed_success_rate": mean([r["fixed_quality"] for r in predictions]),
        "fixed_mean_latency_seconds": mean([r["fixed_latency_seconds"] for r in predictions
                                            if r["fixed_latency_seconds"] is not None]),
        "oracle_success_rate": mean([r["oracle_quality"] for r in predictions]),
        "oracle_regret": mean([r["oracle_quality"] - r["selected_quality"] for r in predictions]),
        "bce": mean([r["bce"] for r in predictions]),
        "brier": mean([r["brier"] for r in predictions]),
        "per_model_mean_probability": {
            name: mean([r["model_probabilities"][name] for r in predictions]) for name in model_ids
        },
        "per_model_observed_success_rate": {
            name: mean([r["model_scores"][name] for r in predictions]) for name in model_ids
        },
        "selected_model_counts": dict(Counter(r["selected_model"] for r in predictions)),
        "input_tokens": sum(r["encoding"]["input_tokens"] for r in predictions),
        "cropped_rows": sum(r["encoding"]["cropped_task_tokens"] > 0 for r in predictions),
    }


def evaluate(model, rows, processor, args, device, model_ids, fixed_model, output, rank, world_size):
    import torch
    import torch.distributed as dist

    model.eval()
    local = []
    with torch.inference_mode():
        for index in range(rank, len(rows), world_size):
            row = rows[index]
            logits, info = forward_row(model, row, processor, args, device)
            targets = torch.tensor([quality_targets(row, model_ids)], dtype=torch.float32, device=device)
            bce = float(success_loss(logits, targets).item())
            probabilities = logits.float().sigmoid()[0].cpu().tolist()
            local.append((index, prediction_record(row, probabilities, bce, model_ids, fixed_model, info)))
            if rank == 0 and len(local) % 100 == 0:
                print(json.dumps({"event": "evaluation_progress", "file": Path(output).name,
                                  "rank_zero_rows": len(local), "world_size": world_size}), flush=True)
    parts = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(parts, local)
    else:
        parts[0] = local
    predictions = [row for _, row in sorted((item for part in parts for item in part), key=lambda item: item[0])]
    if rank == 0:
        with Path(output).open("w") as stream:
            for row in predictions:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return summarize(predictions, model_ids)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--processor-path")
    parser.add_argument("--chat-template-path")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--effective-batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--max-image-tokens", type=int, default=256)
    parser.add_argument("--image-token-budget", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--max-eval-records", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from model import load_model

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    started = time.monotonic()
    smoke = bool(args.max_steps or args.max_train_records or args.max_eval_records)
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    catalog = json.loads((args.data_dir / "models.json").read_text())
    model_ids = catalog["model_ids"]
    fixed_model = catalog["fixed_model"]
    datasets = {split: read_rows(args.data_dir / f"{split}.jsonl") for split in ("train", "dev")}
    # Implementation smoke runs never read the test file.
    datasets["test"] = [] if smoke else read_rows(args.data_dir / "test.jsonl")
    train_rows = datasets["train"]
    if args.max_train_records:
        train_rows = train_rows[:args.max_train_records]
    if args.max_eval_records:
        datasets["dev"] = datasets["dev"][:args.max_eval_records]
    # All seven binary outcomes are present in this frozen RoboTwin release.
    for rows in (train_rows, datasets["dev"], datasets["test"]):
        for row in rows:
            values = quality_targets(row, model_ids)
            if any(value not in (0.0, 1.0) for value in values):
                raise ValueError(f"Expected seven observed binary qualities for {row['id']}.")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "result.json").exists() or (args.out / "checkpoint").exists():
        raise ValueError("Choose a new output directory for each training run.")
    planned_steps = args.epochs * math.ceil(len(train_rows) / args.effective_batch)
    if args.max_steps:
        planned_steps = min(planned_steps, args.max_steps)
    plan = {
        "schema_version": "robotwin-router-success-v1",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model_ids": model_ids, "world_size": world_size, "fixed_model": fixed_model,
        "effective_global_batch": args.effective_batch, "planned_optimizer_updates": planned_steps,
        "rows": {split: len(rows) for split, rows in datasets.items()},
        "supervised_train_rows": len(train_rows),
        "supervised_train_candidate_outcomes": len(train_rows) * len(model_ids),
        "input_contract": "Same task text, three initial RGB images, robot state and output contract as the prior experiment; no task IDs, candidate outcomes or labels in the encoder input.",
        "loss": "Seven independent binary logits. Unweighted BCE averaged over candidates within each scene, then over scenes in the global batch. All-success and all-failure scenes included.",
        "routing": "Argmax of seven predicted sigmoid success probabilities; exact ties use frozen model order.",
        "checkpoint_selection": "Highest dev routing success rate, then lowest dev BCE; exact ties retain the earlier epoch. Test evaluated once after selection.",
        "unused_fields": ["target_model_id", "target_selection_reason", "capability_id", "capability_valid"],
    }
    if rank == 0:
        dump(args.out / "plan.json", plan)
        dump(args.out / "progress.json", {"status": "loading", "step": 0, "planned_steps": planned_steps})
        print(json.dumps({"event": "loading", "world_size": world_size, "model_path": args.model_path,
                          "supervised_train_rows": len(train_rows)}), flush=True)
    processor = load_processor(args)
    model = load_model(args.model_path, model_ids, device, args.lora_r)
    if smoke:
        probe_name, probe = next((name, parameter) for name, parameter in model.named_parameters()
                                if parameter.requires_grad and "lora_B" in name)
        initial_lora = probe.detach().cpu().clone()
        initial_head = model.head.weight.detach().cpu().clone()
    wrapper = (DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None,
                                      broadcast_buffers=False) if world_size > 1 else model)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "base_lr": args.lr},
        {"params": list(model.head.parameters()), "base_lr": args.head_lr},
    ], lr=args.lr, weight_decay=0.01)
    warmup = max(1, round(planned_steps * 0.05))
    best_dev, best_epoch, step = None, None, 0
    checkpoint = args.out / "checkpoint"
    curves = []
    for epoch in range(1, args.epochs + 1):
        wrapper.train()
        epoch_loss = torch.zeros(2, dtype=torch.float64, device=device)
        epoch_started = time.monotonic()
        for batch in global_batches(train_rows, epoch, args.seed, args.effective_batch):
            if step >= planned_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            progress = max(0, step + 1 - warmup) / max(1, planned_steps - warmup)
            scale = ((step + 1) / warmup if step + 1 <= warmup else
                     0.1 + 0.9 * (1 + math.cos(math.pi * min(1, progress))) / 2)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * scale
            micros = rank_microbatches(batch, rank, world_size)
            for micro_index, (row, real) in enumerate(micros):
                sync = (wrapper.no_sync() if world_size > 1 and micro_index + 1 < len(micros)
                        else nullcontext())
                with sync:
                    logits, _ = forward_row(wrapper, row, processor, args, device)
                    targets = torch.tensor([quality_targets(row, model_ids)], dtype=torch.float32, device=device)
                    loss = success_loss(logits, targets)
                    weighted = loss * (world_size / len(batch) if real else 0.0)
                    weighted.backward()
                if real:
                    epoch_loss[0] += loss.detach().double()
                    epoch_loss[1] += 1
            norm = torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
            optimizer.step()
            step += 1
            if rank == 0 and (step == 1 or step % 25 == 0):
                event = {
                    "event": "training", "epoch": epoch, "step": step, "planned_steps": planned_steps,
                    "local_mean_bce": float((epoch_loss[0] / epoch_loss[1]).item()),
                    "gradient_norm": float(norm.item()), "lr": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": time.monotonic() - started,
                }
                append(args.out / "metrics.jsonl", event)
                dump(args.out / "progress.json", dict(event, status="training"))
                print(json.dumps(event), flush=True)
        if world_size > 1:
            dist.all_reduce(epoch_loss)
        dev = evaluate(model, datasets["dev"], processor, args, device, model_ids, fixed_model,
                       args.out / f"predictions-dev-epoch{epoch:02d}.jsonl", rank, world_size)
        selected = better_dev(dev, best_dev)
        if selected:
            best_epoch, best_dev = epoch, dev
            if rank == 0:
                model.save_checkpoint(checkpoint, {
                    "model_path": args.model_path, "epoch": epoch, "step": step,
                    "dev_success_rate": dev["routing_success_rate"], "dev_bce": dev["bce"],
                })
                processor.save_pretrained(checkpoint / "processor")
                shutil.copyfile(args.data_dir / "models.json", checkpoint / "models.json")
                dump(checkpoint / "training_plan.json", plan)
        curve = {
            "epoch": epoch, "step": step, "selected": selected,
            "train_bce": float((epoch_loss[0] / epoch_loss[1]).item()),
            "train_examples": int(epoch_loss[1].item()),
            "train_candidate_outcomes": int(epoch_loss[1].item()) * len(model_ids),
            "dev": dev, "seconds": time.monotonic() - epoch_started,
        }
        curves.append(curve)
        if rank == 0:
            dump(args.out / "curve.json", curves)
            dump(args.out / "progress.json", {
                "status": "training" if step < planned_steps else "final_evaluation",
                "epoch": epoch, "step": step, "planned_steps": planned_steps, "best_epoch": best_epoch,
                "dev_success_rate": dev["routing_success_rate"], "dev_bce": dev["bce"],
                "elapsed_seconds": time.monotonic() - started,
            })
            print(json.dumps({"event": "epoch_complete", "epoch": epoch, "step": step,
                              "dev_success_rate": dev["routing_success_rate"], "dev_bce": dev["bce"],
                              "best_epoch": best_epoch}), flush=True)
        if world_size > 1:
            dist.barrier()
        if step >= planned_steps:
            break
    model.restore_weights(checkpoint)
    smoke_validation = None
    if smoke:
        model.eval()
        with torch.inference_mode():
            before, _ = forward_row(model, datasets["dev"][0], processor, args, device)
            reference = before.detach().float().cpu()
            head_change = float((model.head.weight.detach().cpu() - initial_head).abs().max())
            lora_change = float((probe.detach().cpu() - initial_lora).abs().max())
        with torch.no_grad():
            model.head.weight.add_(0.125)
            probe.add_(0.125)
        model.restore_weights(checkpoint)
        with torch.inference_mode():
            reloaded, _ = forward_row(model, datasets["dev"][0], processor, args, device)
            restored = reloaded.detach().float().cpu()
        smoke_validation = {
            "evaluation_split": "dev", "probe_lora_parameter": probe_name,
            "lora_max_abs_update": lora_change, "head_max_abs_update": head_change,
            "lora_updated": lora_change > 0, "head_updated": head_change > 0,
            "checkpoint_reload_max_logit_difference": float((reference - restored).abs().max()),
            "checkpoint_reload_logits_match": bool(torch.allclose(reference, restored, atol=1e-5, rtol=1e-5)),
            "reload_check": "Save trained adapter/head, perturb both tensors, restore, compare the same dev input.",
        }
        if not all(smoke_validation[key] for key in ("lora_updated", "head_updated", "checkpoint_reload_logits_match")):
            raise RuntimeError(f"Smoke parameter/checkpoint validation failed: {smoke_validation}")
    final_split = "dev" if smoke else "test"
    final_metrics = evaluate(model, datasets[final_split], processor, args, device, model_ids, fixed_model,
                             args.out / ("predictions-dev-smoke.jsonl" if smoke else "predictions-test.jsonl"),
                             rank, world_size)
    if rank == 0:
        result = {
            "status": "completed", "schema_version": plan["schema_version"],
            "model_path": args.model_path, "checkpoint": str(checkpoint.resolve()),
            "best_epoch": best_epoch, "best_dev_success_rate": best_dev["routing_success_rate"],
            "best_dev_bce": best_dev["bce"], "checkpoint_selection": plan["checkpoint_selection"],
            "optimizer_updates": step, "world_size": world_size,
            "supervised_train_rows": len(train_rows),
            "supervised_train_candidate_outcomes": len(train_rows) * len(model_ids),
            "smoke": smoke, "smoke_validation": smoke_validation,
            "test": None if smoke else final_metrics, "smoke_dev": final_metrics if smoke else None,
            "curve": curves, "total_seconds": time.monotonic() - started,
        }
        dump(args.out / "result.json", result)
        dump(args.out / "progress.json", {"status": "completed", "step": step,
                                         "planned_steps": planned_steps, "best_epoch": best_epoch,
                                         "elapsed_seconds": time.monotonic() - started})
        print(json.dumps({"event": "completed", "evaluation_split": final_split,
                          "metrics": final_metrics}, ensure_ascii=False), flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
