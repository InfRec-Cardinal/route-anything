"""Prepare portable training plans from already aligned, user-supplied records.

This adapter does not reconstruct the paper's private data, taxonomy or splits.
It preserves the original trainers' task-level streams and hyperparameters.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parent


def load_local(name, path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["outcome", "type"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-sequence", type=int, default=8192)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.resolve()
    dimensions = json.loads(args.taxonomy.read_text())["dimensions"]
    train = [json.loads(line) for line in (data_dir / "train.jsonl").read_text().splitlines() if line.strip()]
    if any(row["split"] != "train" for row in train):
        raise ValueError("train.jsonl must contain training rows only.")
    plan = {"model_path": args.model, "max_sequence": args.max_sequence,
            "lr": 2e-5, "head_lr": 3e-4, "effective_batch": 32, "epochs": 6,
            "seeds": [42, 13, 2026, 7], "dimensions": dimensions,
            **{split + "_path": str(data_dir / (split + ".jsonl")) for split in ("train", "dev", "test")}}
    if (data_dir / "special_test.jsonl").exists():
        plan["special_test_path"] = str(data_dir / "special_test.jsonl")
    if args.mode == "type":
        mapping = load_local("release_type_mapping", ROOT / "type_router" / "mapping.py")
        taxonomy = json.loads(args.taxonomy.read_text())
        pools = {family["id"]: family["model_ids"] for family in taxonomy.get("families", [])} or None
        table = mapping.build_table(train, dimensions, model_pools=pools)
        table_path = args.out.resolve() / "table.json"
        table_path.write_text(json.dumps(table, indent=2) + "\n")
        plan.update(schema_version="task-capability-router-v1", table_path=str(table_path))
    else:
        from transformers import AutoTokenizer
        backend = load_local("release_outcome_train", ROOT / "outcome" / "train.py")
        train = backend.read_rows(data_dir / "train.jsonl")
        plan["dimensions"] = [{"id": "total_quality", "name": "Benchmark total quality",
                               "definition": "Predict the normalized continuous quality defined by the stated benchmark metric."}]
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        eligible = [row for row in train if any(backend.supervised(candidate) for candidate in row["candidates"])]
        token_counts = {row["decision_id"]: sum(len(backend.encoded_pair(row, candidate, tokenizer, plan))
                        for candidate in row["candidates"] if backend.supervised(candidate)) for row in eligible}
        ids = [row["decision_id"] for row in eligible]
        if not ids:
            raise ValueError("No supervised candidate outcomes in the supplied training split.")
        plan.update(schema_version="portable-scalar-plan-v1", train_micro_batch=1, streams={},
                    train_panel_ids=ids, probe_ids=ids[:2], smoke_ids=ids[:4])
        for seed in plan["seeds"]:
            rng = random.Random(seed)
            stream = []
            for epoch in range(plan["epochs"]):
                ordered = list(ids)
                rng.shuffle(ordered)
                stream.extend({"decision_id": identifier, "pair_tokens": token_counts[identifier]}
                              for identifier in ordered)
            path = args.out.resolve() / f"stream-{seed}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in stream))
            plan["streams"][str(seed)] = {"path": str(path)}
    destination = args.out / "plan.json"
    destination.write_text(json.dumps(plan, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
