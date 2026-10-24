"""Fit and replay the CPU learned-price baselines from paper Tables 11 and 12.

All saved results are aggregate metrics. Prompts, task IDs, per-task scores,
costs, predictions and fitted feature artifacts are never written to output.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from features import tfidf_pair
from scipy import sparse
from data import read_jsonl, load_crb, load_routerbench, split_records
from source_core import route, budget_route, pick_points, strat_weights
from ridge import build_menu, targets, fit_predict, compose, knn_predictions, routellm_predictions


CRB_PRIMARIES = ["Qwen3-Max", "kimi-k2.5", "qwen3.5-plus", "glm-5", "gpt-5.4", "claude-sonnet-4-6", "MiniMax-M2.7"]


def point(quality, spend, baseline_quality, baseline_cost):
    return {"r": float((quality.sum() / spend.sum()) / (baseline_quality.sum() / baseline_cost.sum())),
            "dV": float((quality - baseline_quality).mean()), "cost_mUSD": float(1000 * spend.mean()),
            "mean_score": float(quality.mean())}


def bootstrap(quality, spend, baseline_quality, baseline_cost, weights):
    count = weights.sum(1)
    score, cost = weights @ quality, weights @ spend
    ref_score, ref_cost = weights @ baseline_quality, weights @ baseline_cost
    values = {"r": (score / cost) / (ref_score / ref_cost), "dV": (score - ref_score) / count,
              "cost_mUSD": 1000 * cost / count, "mean_score": score / count}
    result = {name: [float(value[0]), *map(float, np.percentile(value[1:], [2.5, 97.5]))]
              for name, value in values.items()}
    result["compliance"] = float(np.mean(values["dV"][1:] >= -1e-12))
    return result


def sweep(parameters, decide_fit, decide_test, fit_value, fit_cost, test_value, test_cost,
          fit_reference, test_reference, weights):
    recs = []
    for parameter in parameters:
        choice = decide_fit(parameter)
        selection = np.arange(len(choice))
        stats = point(fit_value[selection, choice], fit_cost[selection, choice], *fit_reference)
        recs.append({"parameter": float(parameter), "p_r": stats["r"], "p_dV": stats["dV"],
                     "fit": stats})
    maximum, zero_loss = pick_points(recs)
    output = {}
    for name, index in [("max_fit_efficiency", maximum), ("zero_loss_fit", zero_loss)]:
        record = recs[index]
        choice = decide_test(record["parameter"])
        selection = np.arange(len(choice))
        output[name] = {**record, "test": bootstrap(test_value[selection, choice], test_cost[selection, choice],
                                                   *test_reference, weights)}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["crb", "routerbench"], required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="Normalized JSONL; see SCHEMA.md.")
    inputs.add_argument("--crb-data-dir", type=Path)
    inputs.add_argument("--routerbench-pkl", type=Path)
    parser.add_argument("--prompts", type=Path, help="CRB task_id->prompt JSON or JSONL with task_id,prompt.")
    parser.add_argument("--responses", type=Path, help="CRB task_id->model->response JSON; required for FrugalGPT-style scorer.")
    parser.add_argument("--strip-wrappers", action="store_true", help="Alternate protocol: fit source-agnostic union stripping on fitting prompts only.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--weak", help="RouteLLM-style weak model; required if routellm is requested.")
    parser.add_argument("--models", nargs="+", help="Frozen model order; defaults to the source input order.")
    parser.add_argument("--primary-order", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=["ridge", "knn", "routellm", "frugal"], default=["ridge", "knn"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--limit-fit", type=int, default=0, help="Real-input smoke subset; not a paper reproduction.")
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.strip_wrappers and args.profile != "crb":
        parser.error("--strip-wrappers is the alternate CRB protocol; RouterBench uses its parsed source prompts.")
    if args.crb_data_dir:
        records, models = load_crb(args.crb_data_dir, args.prompts, args.responses)
    elif args.routerbench_pkl:
        records, models = load_routerbench(args.routerbench_pkl)
    else:
        records = read_jsonl(args.input)
        models = list(records[0]["scores"])
    models = args.models or models
    reference = models.index(args.reference)
    seed = 20260925 if args.profile == "crb" else 20260926
    fit, test = split_records(records, args.profile, args.reference, seed)
    if args.limit_fit:
        fit = np.sort(np.random.default_rng(seed).choice(fit, min(args.limit_fit, len(fit)), replace=False))
    if args.limit_test:
        test = np.sort(np.random.default_rng(seed).choice(test, min(args.limit_test, len(test)), replace=False))
    train_rows, test_rows = [records[index] for index in fit], [records[index] for index in test]
    if args.strip_wrappers:
        from collections import defaultdict
        from stripwrap import learn, apply_union
        sources = defaultdict(list)
        for row in train_rows:
            sources[row["source"]].append(row["prompt"])
        _, rule = learn(sources, tau=.2, min_count=3)
        train_rows = [{**row, "prompt": apply_union(row["prompt"], rule)} for row in train_rows]
        test_rows = [{**row, "prompt": apply_union(row["prompt"], rule)} for row in test_rows]
    scores = np.array([[row["scores"][model] for model in models] for row in records], dtype=float)
    costs = np.array([[row["costs"][model] for model in models] for row in records], dtype=float)
    thetas = [1.0, .5] if args.profile == "crb" else [1.0]
    primary_order = args.primary_order
    if primary_order is None and args.profile == "crb" and set(CRB_PRIMARIES) == set(models) - {args.reference}:
        primary_order = CRB_PRIMARIES
    values, spend, actions, names = build_menu(scores, costs, models, args.reference, thetas, primary_order)
    fit_reference = scores[fit, reference], costs[fit, reference]
    test_reference = scores[test, reference], costs[test, reference]
    strata = np.array([row["stratum"] for row in train_rows])
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(fit)), strata))
    test_strata = np.array([row["stratum"] for row in test_rows])
    if args.profile == "routerbench":
        weights = strat_weights(test_strata, args.bootstrap, seed)
    else:
        rng = np.random.default_rng(seed)
        weights = np.vstack([np.ones((1, len(test))), rng.multinomial(len(test), np.full(len(test), 1 / len(test)), size=args.bootstrap)])
    result = {"profile": args.profile, "reference": args.reference, "models": models,
              "seed": seed, "fit_rows": len(fit), "test_rows": len(test), "actions": names,
              "smoke": bool(args.limit_fit or args.limit_test), "evidence_type": "recorded-outcome replay",
              "prompt_preprocessing": ("Alternate protocol: train-fitted source-agnostic union stripping (tau=.2,min_count=3)."
                                       if args.strip_wrappers else "Source protocol: supplied prompts unchanged."),
              "methods": {}}
    word, char = tfidf_pair()
    train_text, test_text = [row["prompt"] for row in train_rows], [row["prompt"] for row in test_rows]
    Xword_fit = word.fit_transform(train_text).tocsr(); Xword_test = word.transform(test_text).tocsr()
    Xfit = sparse.hstack([Xword_fit, char.fit_transform(train_text)]).tocsr() / np.sqrt(2)
    Xtest = sparse.hstack([Xword_test, char.transform(test_text)]).tocsr() / np.sqrt(2)
    if args.profile == "routerbench":
        Xfit, Xtest = Xfit.astype(np.float32), Xtest.astype(np.float32)
    result["feature_count"] = int(Xfit.shape[1])
    estimates = {}
    if set(args.methods) & {"ridge", "knn"}:
        target = targets(scores[fit], costs[fit], actions, args.profile)
        held, predicted, alphas, Kfit, Ktest = fit_predict(Xfit, Xtest, target, folds, args.profile)
        calibration = costs[fit].mean(0) / np.exp(held[:, len(models):2 * len(models)]).mean(0)
        if "ridge" in args.methods:
            estimates["quality-cost"] = (compose(held, calibration, actions, len(models), reference),
                                         compose(predicted, calibration, actions, len(models), reference))
            result["ridge"] = {"alphas": alphas, "cost_calibration": calibration.tolist()}
        if "knn" in args.methods:
            estimates["knn20"] = knn_predictions(Kfit, Ktest, values[fit], spend[fit])
        del Kfit, Ktest
    lambdas = np.logspace(-2, 3, 201) if args.profile == "crb" else np.logspace(0, 5, 251)
    bounds = (-4.0, 7.0) if args.profile == "crb" else (-3.0, 8.0)
    def run_sweep(parameters, decide_fit, decide_test):
        return sweep(parameters, decide_fit, decide_test, values[fit], spend[fit], values[test], spend[test],
                     fit_reference, test_reference, weights)
    for name, (fit_est, test_est) in estimates.items():
        for variant, allowed in [("direct", list(range(len(models)))), ("cascade", list(range(len(actions))))]:
            result["methods"][f"{name}/{variant}/price"] = run_sweep(
                lambdas, lambda value: route(*fit_est, value, allowed), lambda value: route(*test_est, value, allowed))
            result["methods"][f"{name}/{variant}/budget"] = run_sweep(
                np.logspace(-3, 0, 40),
                lambda value: budget_route(*fit_est, value * fit_est[1][:, reference].mean(), allowed, bounds)[0],
                lambda value: budget_route(*test_est, value * test_est[1][:, reference].mean(), allowed, bounds)[0])
    if "routellm" in args.methods:
        if not args.weak:
            raise ValueError("--weak is required for the RouteLLM-style pair.")
        weak = models.index(args.weak)
        features = (Xfit, Xtest) if args.profile == "crb" else (Xword_fit, Xword_test)
        held, predicted, metadata = routellm_predictions(*features, scores[fit], reference, weak, folds)
        thresholds = np.unique(np.concatenate([np.quantile(held, np.linspace(0, 1, 101)), [1.01]]))
        result["methods"]["routellm-style"] = {"classifier": metadata, **run_sweep(
            thresholds, lambda value: np.where(held >= value, reference, weak),
            lambda value: np.where(predicted >= value, reference, weak))}
    if "frugal" in args.methods:
        if args.profile != "crb":
            raise ValueError("The released FrugalGPT-style source is a CRB experiment.")
        from frugal import fit_scorers, fit_chain, chain_apply
        held, predicted, metadata = fit_scorers(train_rows, test_rows, models, scores[fit], folds)
        selected = fit_chain(held, scores[fit], costs[fit], models, reference)
        quality, cost, shares = chain_apply(selected["chain_idx"], selected["taus"], predicted, scores[test], costs[test])
        result["methods"]["frugalgpt-style"] = {"scorers": metadata, "selected": selected,
                                                "test": bootstrap(quality, cost, *test_reference, weights),
                                                "stage_return_shares": shares}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "completed", "fit_rows": len(fit), "test_rows": len(test),
                      "smoke": result["smoke"], "methods": list(result["methods"])}))


if __name__ == "__main__":
    main()
