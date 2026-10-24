"""Replay supplied router choices on one shared observed-outcome panel.

This portable evaluator follows the revised paper's baseline and pairing rules.
It does not contain paper tasks, outcomes, or fitted policies.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random


FABLE = "claude-fable-5-1"
OPUS = "claude-opus-5-5"
ASTRA = "gpt-6-astra"


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def fixed_from_train(rows, opus_fallback=False):
    """Fit each fixed policy under the same fallback used during evaluation."""
    training = [row for row in rows if row["split"] == "train"]
    candidates = defaultdict(set)
    for row in training:
        candidates[row["benchmark"]].update(row.get("candidate_models", []))
        candidates[row["benchmark"]].update(row["model_scores"])
    for models in candidates.values():
        models.discard(FABLE)
    by_benchmark = defaultdict(lambda: defaultdict(list))
    fallback_counts = defaultdict(lambda: defaultdict(int))
    for row in training:
        for model in candidates[row["benchmark"]]:
            effective, quality = resolve(model, row["model_scores"], opus_fallback)
            if quality is not None:
                by_benchmark[row["benchmark"]][model].append(quality)
                fallback_counts[row["benchmark"]][model] += effective != model
    choices, evidence = {}, {}
    priority = {OPUS: 0, ASTRA: 1}
    for benchmark, models in by_benchmark.items():
        means = {model: sum(values) / len(values) for model, values in models.items()}
        choices[benchmark] = min(means, key=lambda name: (-means[name], priority.get(name, 2), name))
        evidence[benchmark] = {name: {"mean": means[name], "observed_train_rows": len(values),
                                     "fallback_train_rows": fallback_counts[benchmark][name]}
                               for name, values in models.items()}
    return choices, evidence


def resolve(model, scores, opus_fallback=False):
    model = OPUS if model == FABLE else model
    if opus_fallback and model == OPUS and scores.get(model) is None:
        model = ASTRA
    return model, scores.get(model)


def group_interval(rows, key, repetitions=2000, seed=42):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["benchmark"], row["group_id"])].append(row[key] - row["fixed"])
    if len(groups) < 2:
        return None
    totals = [(sum(values), len(values)) for values in groups.values()]
    rng = random.Random(seed)
    draws = []
    for _ in range(repetitions):
        sample = rng.choices(totals, k=len(totals))
        draws.append(sum(total for total, _ in sample) / sum(count for _, count in sample))
    draws.sort()
    return [draws[int((repetitions - 1) * .025)], draws[int((repetitions - 1) * .975)]]


def evaluate(rows, predictions, opus_fallback=False, complete_panel=True, repetitions=2000):
    fixed, evidence = fixed_from_train(rows, opus_fallback)
    indexed = {(row["method"], str(row["seed"]), row["id"]): row["selected_model"] for row in predictions}
    seeds = defaultdict(set)
    for row in predictions:
        seeds[row["method"]].add(str(row["seed"]))
    if not seeds:
        raise ValueError("Supply at least one method's predictions.")
    kept, missing = [], defaultdict(int)
    test = [row for row in rows if row["split"] == "test"]
    for row in test:
        scores = {model: value for model, value in row["model_scores"].items() if model != FABLE}
        candidates = [name for name in row.get("candidate_models", list(scores)) if name != FABLE]
        resolved = [resolve(name, scores, opus_fallback)[1] for name in candidates]
        if complete_panel and (len(candidates) < 2 or any(value is None for value in resolved)):
            missing["incomplete_candidate_panel"] += 1
            continue
        _, baseline = resolve(fixed.get(row["benchmark"]), scores, opus_fallback)
        selected = {method: [resolve(indexed.get((method, seed, row["id"])), scores, opus_fallback)[1]
                             for seed in sorted(method_seeds)] for method, method_seeds in seeds.items()}
        if baseline is None or any(value is None for values in selected.values() for value in values):
            missing["missing_fixed_or_method_outcome"] += 1
            continue
        observed = [value for value in resolved if value is not None]
        if not observed:
            missing["no_oracle_outcomes"] += 1
            continue
        kept.append({"id": row["id"], "group_id": row["group_id"], "benchmark": row["benchmark"],
                     "fixed": baseline, "oracle": max(observed),
                     **{method: sum(values) / len(values) for method, values in selected.items()}})

    def summarize(panel):
        result = {"rows": len(panel)}
        for method in ["fixed", "oracle", *sorted(seeds)]:
            result[method] = {"mean": sum(row[method] for row in panel) / len(panel) if panel else None}
            if method not in ("fixed", "oracle"):
                result[method]["delta_vs_fixed"] = (sum(row[method] - row["fixed"] for row in panel) / len(panel)
                                                     if panel else None)
                result[method]["group_bootstrap_95ci"] = group_interval(panel, method, repetitions)
        return result

    return {"fixed_models": fixed, "training_evidence": evidence,
            "test_rows": len(test), "excluded": dict(missing), "pooled": summarize(kept),
            "by_benchmark": {benchmark: summarize([row for row in kept if row["benchmark"] == benchmark])
                             for benchmark in sorted({row["benchmark"] for row in test})},
            "methods_seeds": {key: sorted(values) for key, values in seeds.items()},
            "comparison": "All methods and seeds use identical retained inputs; seeds averaged within input before group bootstrap.",
            "interval_scope": "Conditional on supplied predictions; no training-seed uncertainty or multiplicity correction.",
            "complete_candidate_panel": complete_panel, "opus_fallback": opus_fallback}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--opus-fallback", action="store_true", help="Use Astra for missing Opus outcomes during fixed-policy fitting and test replay.")
    parser.add_argument("--allow-partial-candidate-panel", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    report = evaluate(read_rows(args.records), read_rows(args.predictions), args.opus_fallback,
                      not args.allow_partial_candidate_panel, args.bootstrap)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"test_rows": report["test_rows"], "shared_rows": report["pooled"]["rows"]}))


if __name__ == "__main__":
    main()
