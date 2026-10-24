"""Generate invented schema examples and check CPU replay, without downloads.

Every record here is synthetic. This is an implementation check, not a paper
reproduction, benchmark result, model training run, or robot rollout.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from matched_evaluate import ASTRA, OPUS, evaluate, fixed_from_train


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = [{"id": f"synthetic-train-{i}", "group_id": f"synthetic-train-{i}",
                "benchmark": "synthetic-benchmark", "split": "train",
                "candidate_models": ["synthetic-strong", "synthetic-weak"],
                "model_scores": {"synthetic-strong": None if i < 11 else .9, "synthetic-weak": .8}}
               for i in range(1122)]
    test = [{"id": f"synthetic-test-{i}", "group_id": f"synthetic-group-{i // 2}",
             "benchmark": "synthetic-benchmark", "split": "test",
             "candidate_models": ["synthetic-strong", "synthetic-weak"],
             "model_scores": {"synthetic-strong": .8, "synthetic-weak": .9 if i % 2 else .5}}
            for i in range(8)]
    predictions = [{"id": row["id"], "method": "synthetic-router", "seed": seed,
                    "selected_model": "synthetic-weak" if index % 2 else "synthetic-strong"}
                   for index, row in enumerate(test) for seed in [42, 13, 2026, 7]]
    records += test
    choices, evidence = fixed_from_train(records)
    assert choices["synthetic-benchmark"] == "synthetic-strong"
    assert evidence["synthetic-benchmark"]["synthetic-strong"]["observed_train_rows"] == 1111
    report = evaluate(records, predictions, repetitions=100)
    assert report["pooled"]["rows"] == 8
    assert abs(report["pooled"]["synthetic-router"]["delta_vs_fixed"] - .05) < 1e-9
    incomplete = predictions[:-1]
    paired = evaluate(records, incomplete, repetitions=100)
    assert paired["pooled"]["rows"] == 7
    rows(args.out / "replay-records.jsonl", records)
    rows(args.out / "replay-predictions.jsonl", predictions)
    dump(args.out / "replay-report.json", report)

    # The second row deliberately has no Opus key or per-row candidate list.
    # Its Astra result must still enter the benchmark's Opus-fallback policy.
    fallback_records = [
        {"id": "synthetic-fallback-train-1", "group_id": "synthetic-fallback-train-1",
         "benchmark": "synthetic-fallback", "split": "train",
         "model_scores": {OPUS: 1.0, ASTRA: 0.0, "synthetic-alternative": .7}},
        {"id": "synthetic-fallback-train-2", "group_id": "synthetic-fallback-train-2",
         "benchmark": "synthetic-fallback", "split": "train",
         "model_scores": {ASTRA: 0.0, "synthetic-alternative": .7}},
        {"id": "synthetic-fallback-test", "group_id": "synthetic-fallback-test",
         "benchmark": "synthetic-fallback", "split": "test",
         "candidate_models": [OPUS, ASTRA, "synthetic-alternative"],
         "model_scores": {ASTRA: .3, "synthetic-alternative": .6}},
    ]
    fallback_predictions = [{"id": "synthetic-fallback-test", "method": "synthetic-router",
                             "seed": 42, "selected_model": OPUS}]
    no_fallback = evaluate(fallback_records, fallback_predictions, repetitions=100)
    with_fallback = evaluate(fallback_records, fallback_predictions, opus_fallback=True, repetitions=100)
    assert no_fallback["fixed_models"]["synthetic-fallback"] == OPUS
    assert with_fallback["fixed_models"]["synthetic-fallback"] == "synthetic-alternative"
    opus_evidence = with_fallback["training_evidence"]["synthetic-fallback"][OPUS]
    assert opus_evidence == {"mean": .5, "observed_train_rows": 2, "fallback_train_rows": 1}
    assert no_fallback["pooled"]["rows"] == 0
    assert with_fallback["pooled"]["rows"] == 1
    assert with_fallback["pooled"]["synthetic-router"]["mean"] == .3
    assert with_fallback["pooled"]["fixed"]["mean"] == .6
    rows(args.out / "fallback-records.jsonl", fallback_records)
    rows(args.out / "fallback-predictions.jsonl", fallback_predictions)
    dump(args.out / "fallback-report.json", with_fallback)

    tree = {"tracks": [{"track_id": "synthetic-text", "tasks": [{"task_id": "synthetic-task",
            "task_description": "Synthetic instruction following",
            "capabilities": [{"leaf_id": "synthetic-leaf", "name_zh": "Synthetic leaf",
                              "definition": "Follow an invented short instruction."}]}]}]}
    taxonomy = {"dimensions": [{"id": "synthetic-type", "name": "Synthetic type",
                                "definition": "Follow an invented short instruction."}]}
    dump(args.out / "tree.json", tree)
    dump(args.out / "taxonomy.json", taxonomy)
    all_tree = []
    for split, count in [("train", 4), ("dev", 2), ("test", 2)]:
        tree_rows, text_rows, robot_rows = [], [], []
        for index in range(count):
            identifier = f"synthetic-{split}-{index}"
            scores = {"synthetic-strong": .9, "synthetic-weak": .7}
            tree_row = {"id": identifier, "task_id": identifier, "group_id": identifier,
                        "benchmark": "synthetic-benchmark", "split": split,
                        "track_id": "synthetic-text", "tree_task_id": "synthetic-task",
                        "task_text": f"Synthetic instruction number {index}: say hello.",
                        "images": [], "state_text": "", "output_contract": {"modalities": ["text"]},
                        "candidate_leaf_ids": ["synthetic-leaf"], "leaf_id": "synthetic-leaf",
                        "label_status": "teacher_labeled", "model_scores": scores,
                        "model_costs": {}, "model_tokens": {}}
            tree_rows.append(tree_row)
            candidates = [{"arm_version_id": name, "model_id": name,
                           "profile": {"model": name, "output_modalities": ["text"]},
                           "quality": value, "quality_valid_mask": True, "supervision_valid": True,
                           "observed_mask": True, "selection_mask": True, "capability_mask": True}
                          for name, value in scores.items()]
            text_rows.append({"decision_id": identifier, "task_id": identifier, "group_id": identifier,
                              "benchmark": "synthetic-benchmark", "metric_id": "synthetic-quality",
                              "scalar_spec_id": "synthetic-unit-interval", "split": split,
                              "input_text": tree_row["task_text"], "output_contract": tree_row["output_contract"],
                              "weights": [1.0], "candidates": candidates,
                              "capability_targets": {"synthetic-type": 1.0},
                              "target_capability_id": "synthetic-type", "capability_supervision_valid": True})
            robot_rows.append({"id": identifier, "group_id": identifier, "split": split,
                               "task_text": "Synthetic robot instruction; no real scene.", "images": [],
                               "state_text": "Synthetic state", "output_contract": {"modalities": ["action"]},
                               "model_scores": {f"synthetic-policy-{i}": float(i == index % 7) for i in range(7)},
                               "model_latencies": {}})
        rows(args.out / "capability_tree" / (split + ".jsonl"), tree_rows)
        rows(args.out / "text" / (split + ".jsonl"), text_rows)
        rows(args.out / "robot" / (split + ".jsonl"), robot_rows)
        all_tree.extend(tree_rows)
    dump(args.out / "robot" / "models.json", {"model_ids": [f"synthetic-policy-{i}" for i in range(7)],
                                               "fixed_model": "synthetic-policy-0"})
    source = Path(__file__).parent / "capability_tree" / "fit_table.py"
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("synthetic_leaf_fit", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    table = module.fit_table(all_tree, tree)
    assert table["leaf_to_model"]["synthetic-leaf"] == "synthetic-strong"
    assert table["benchmark_to_model"]["synthetic-benchmark"] == "synthetic-strong"
    dump(args.out / "tree-table.json", table)
    dump(args.out / "NOTICE.json", {"synthetic": True, "paper_reproduction": False,
                                   "description": "Invented input contracts and CPU checks only; no real data or model calls."})
    print(json.dumps({"status": "passed", "synthetic": True,
                      "checks": ["missing-11 baseline eligibility", "fallback changes fixed-policy fitting", "resolved complete panel", "common panel across all seeds", "paired delta", "leaf least-squares fit"],
                      "output": str(args.out)}))


if __name__ == "__main__":
    main()
