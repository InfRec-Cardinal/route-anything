"""Fit portfolio decisions on the original fitting splits; evaluate held out."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


CODE_MODELS = [
    "Qwen3-Max", "kimi-k2.5", "qwen3.5-plus", "glm-5",
    "MiniMax-M2.7", "gpt-5.4", "claude-sonnet-4-6",
]
ROUTER_POOL = [
    "mistralai/mixtral-8x7b-chat", "zero-one-ai/Yi-34B-Chat",
    "claude-instant-v1", "gpt-3.5-turbo-1106",
    "WizardLM/WizardLM-13B-V1.2", "meta/llama-2-70b-chat",
]


def router_family(name):
    if name.startswith("mmlu-"):
        return "mmlu"
    if name in {"hellaswag", "winogrande", "arc-challenge", "grade-school-math", "mbpp"}:
        return name
    if name.startswith("mtbench"):
        return "mt-bench"
    return "rag-other"


def load_code(path):
    frame = pd.read_csv(path)
    scores = frame.pivot(index="task_id", columns="model", values="score")
    costs = frame.pivot(index="task_id", columns="model", values="cost_usd")
    metadata = frame.drop_duplicates("task_id").set_index("task_id").loc[scores.index]
    dimensions = metadata.dimension.to_numpy()
    contexts = [("overall", np.ones(len(scores), dtype=bool))]
    contexts += [(name, dimensions == name) for name in sorted(set(dimensions))]
    return {
        "name": "CodeRouterBench", "models": CODE_MODELS,
        "scores": scores[CODE_MODELS].to_numpy(float),
        "costs": costs[CODE_MODELS].to_numpy(float),
        "reference_scores": scores["claude-opus-4-6"].to_numpy(float),
        "reference_costs": costs["claude-opus-4-6"].to_numpy(float),
        "fit": metadata.split.to_numpy() == "probing", "contexts": contexts,
        "pool": list(range(len(CODE_MODELS))), "thresholds": [1.0, 0.5],
        "split": "Released probing / id_test assignments, unchanged.",
    }


def load_router(path):
    frame = pd.read_pickle(path)
    reference = "gpt-4-1106-preview"
    models = [name for name in frame.columns if f"{name}|total_cost" in frame.columns and name != reference]
    evaluations = frame.eval_name.astype(str).to_numpy()
    families = np.array([router_family(name) for name in evaluations])
    reference_scores = frame[reference].to_numpy(float)
    fit = np.zeros(len(frame), dtype=bool)
    strata = np.array([f"{name}|{int(score >= 0.5)}" for name, score in zip(evaluations, reference_scores)])
    generator = np.random.default_rng(20260917)
    for name in sorted(set(strata)):
        members = np.flatnonzero(strata == name)
        fit[generator.permutation(members)[:int(round(0.7 * len(members)))]] = True
    contexts = [("overall", np.ones(len(frame), dtype=bool))]
    contexts += [(name, families == name) for name in sorted(set(families))]
    counts = pd.Series(evaluations).value_counts()
    contexts += [(name, evaluations == name) for name in sorted(counts.index)
                 if name.startswith("mmlu-") and counts[name] >= 100]
    return {
        "name": "RouterBench", "models": models,
        "scores": frame[models].to_numpy(float),
        "costs": frame[[f"{name}|total_cost" for name in models]].to_numpy(float),
        "reference_scores": reference_scores,
        "reference_costs": frame[f"{reference}|total_cost"].to_numpy(float),
        "fit": fit, "contexts": contexts,
        "pool": [models.index(name) for name in ROUTER_POOL], "thresholds": [0.5, 1.0],
        "split": "Original seed 20260917, 70/30 within eval_name x reference success at 0.5.",
    }


def portfolio_tables(dataset, rows, threshold, portfolios):
    scores = dataset["scores"][rows]
    costs = dataset["costs"][rows]
    passing = scores >= threshold
    reference_scores = dataset["reference_scores"][rows]
    reference_costs = dataset["reference_costs"][rows]
    hits, values, resources = [], [], []
    for members in portfolios:
        hit = passing[:, members].any(axis=1)
        selected = np.where(passing[:, members], scores[:, members], 0.0).max(axis=1)
        hits.append(hit)
        values.append(np.where(hit, selected, reference_scores))
        resources.append(costs[:, members].sum(axis=1) + (~hit) * reference_costs)
    return {
        "hit": np.stack(hits, axis=1), "value": np.stack(values, axis=1),
        "resource": np.stack(resources, axis=1), "passing": passing,
        "costs": costs, "reference_scores": reference_scores,
        "reference_costs": reference_costs,
    }


def interval(values):
    return np.percentile(values, [2.5, 97.5]).tolist()


def analyze(dataset, resamples):
    model_count = len(dataset["models"])
    bases = sorted(tuple(members) for size in [1, 2]
                   for members in itertools.combinations(sorted(dataset["pool"]), size))
    portfolios = sorted(set(bases) | {
        tuple(sorted((*members, candidate))) for members in bases
        for candidate in range(model_count) if candidate not in members
    })
    portfolio_index = {members: index for index, members in enumerate(portfolios)}
    generator = np.random.default_rng(20260921)
    sign_rows, selection_rows = [], []
    for context, context_rows in dataset["contexts"]:
        fit_rows = context_rows & dataset["fit"]
        test_rows = context_rows & ~dataset["fit"]
        test_count = int(test_rows.sum())
        weights = generator.multinomial(test_count, np.full(test_count, 1 / test_count), size=resamples).astype(float)
        for threshold in dataset["thresholds"]:
            if dataset["name"] == "RouterBench" and threshold == 1.0:
                all_scores = np.column_stack((dataset["scores"][context_rows], dataset["reference_scores"][context_rows]))
                if np.isin(all_scores, [0, 1]).all():
                    continue
            fitting = portfolio_tables(dataset, fit_rows, threshold, portfolios)
            heldout = portfolio_tables(dataset, test_rows, threshold, portfolios)
            fit_efficiency = fitting["value"].mean(axis=0) / fitting["resource"].mean(axis=0)
            test_efficiency = heldout["value"].mean(axis=0) / heldout["resource"].mean(axis=0)
            bootstrap_efficiency = (weights @ heldout["value"]) / (weights @ heldout["resource"])
            passes = fitting["passing"].sum(axis=0)
            usable = (passes >= 20) & (len(fitting["passing"]) - passes >= 20)
            for members in bases:
                if not usable[list(members)].all():
                    continue
                base_index = portfolio_index[members]
                hit = fitting["hit"][:, base_index]
                candidates = [candidate for candidate in range(model_count)
                              if candidate not in members and usable[candidate]
                              and np.sum(fitting["passing"][:, candidate] & ~hit) >= 10
                              and np.sum(~(fitting["passing"][:, candidate] | hit)) >= 10
                              and hit.sum() >= 10]
                if not candidates:
                    continue
                if dataset["name"] == "CodeRouterBench" and len(candidates) < 3:
                    continue
                expanded = [portfolio_index[tuple(sorted((*members, candidate)))] for candidate in candidates]
                residual = np.array([np.mean(fitting["passing"][:, candidate] & ~hit) for candidate in candidates])
                candidate_cost = fitting["costs"][:, candidates].mean(axis=0)
                efficiency = fit_efficiency[base_index]
                accepted_value = fitting["value"][hit, base_index].mean()
                missed_value = fitting["reference_scores"][~hit].mean()
                fallback_cost = fitting["reference_costs"][~hit].mean()
                weight = accepted_value - missed_value + efficiency * fallback_cost
                marginal = fitting["passing"][:, candidates].mean(axis=0)
                simplified = residual * weight - efficiency * candidate_cost
                paired_value_change = (fitting["value"][:, expanded] - fitting["value"][:, [base_index]]).mean(axis=0)
                paired_cost_change = (fitting["resource"][:, expanded] - fitting["resource"][:, [base_index]]).mean(axis=0)
                paired_score = paired_value_change - efficiency * paired_cost_change
                paired_gain = paired_score / fitting["resource"][:, expanded].mean(axis=0)
                rules = {
                    "simplified_residual": simplified,
                    "paired_score": paired_score,
                    "paired_gain": paired_gain,
                    "independence": (1 - hit.mean()) * marginal * weight - efficiency * candidate_cost,
                    "marginal_quality": marginal,
                }
                gains = test_efficiency[expanded] - test_efficiency[base_index]
                gains_bootstrap = bootstrap_efficiency[:, expanded] - bootstrap_efficiency[:, [base_index]]
                for position, candidate in enumerate(candidates):
                    bounds = interval(gains_bootstrap[:, position])
                    sign_rows.append({
                        "context": context, "threshold": threshold,
                        "portfolio": [dataset["models"][member] for member in members],
                        "candidate": dataset["models"][candidate],
                        "fit_paired_score": float(paired_score[position]),
                        "fit_simplified_score": float(simplified[position]),
                        "heldout_gain": float(gains[position]), "heldout_gain_interval": bounds,
                        "decided": bounds[0] > 0 or bounds[1] < 0,
                        "agreement": {name: bool(np.sign(values[position]) == np.sign(gains[position]))
                                      for name, values in rules.items() if name != "marginal_quality"},
                    })
                if len(candidates) < 3:
                    continue
                choices = {name: int(np.argmax(values)) for name, values in rules.items()}
                choices["heldout_oracle"] = int(np.argmax(gains))
                choice_gains = {name: float(gains[position]) for name, position in choices.items()}
                paired_position = choices["paired_gain"]
                simplified_position = choices["simplified_residual"]
                difference = gains_bootstrap[:, paired_position] - gains_bootstrap[:, simplified_position]
                selection_rows.append({
                    "context": context, "threshold": threshold,
                    "portfolio": [dataset["models"][member] for member in members],
                    "n_candidates": len(candidates),
                    "choices": {name: dataset["models"][candidates[position]] for name, position in choices.items()},
                    "heldout_gains": choice_gains,
                    "heldout_relative_gains": {name: gain / test_efficiency[base_index] for name, gain in choice_gains.items()},
                    "paired_minus_simplified_interval": interval(difference),
                })
        print(f"{dataset['name']}: {context} complete", flush=True)
    summaries = {}
    for threshold in dataset["thresholds"]:
        signs = [row for row in sign_rows if row["context"] != "overall" and row["threshold"] == threshold]
        selections = [row for row in selection_rows if row["context"] != "overall" and row["threshold"] == threshold]
        if not selections:
            continue
        decided = [row for row in signs if row["decided"]]
        summaries[str(threshold)] = {
            "portfolios": len(selections), "sign_cases": len(signs), "decided_sign_cases": len(decided),
            "agreement_all": {name: float(np.mean([row["agreement"][name] for row in signs])) for name in signs[0]["agreement"]},
            "agreement_decided": {name: float(np.mean([row["agreement"][name] for row in decided])) for name in signs[0]["agreement"]},
            "heldout_best_choice_rate": {name: float(np.mean([row["choices"][name] == row["choices"]["heldout_oracle"] for row in selections])) for name in selections[0]["choices"] if name != "heldout_oracle"},
            "mean_relative_heldout_gain": {name: float(np.mean([row["heldout_relative_gains"][name] for row in selections])) for name in selections[0]["choices"]},
            "paired_changes_choice": sum(row["choices"]["paired_gain"] != row["choices"]["simplified_residual"] for row in selections),
            "paired_better_point": sum(row["heldout_gains"]["paired_gain"] > row["heldout_gains"]["simplified_residual"] for row in selections),
            "paired_worse_point": sum(row["heldout_gains"]["paired_gain"] < row["heldout_gains"]["simplified_residual"] for row in selections),
            "paired_better_interval": sum(row["paired_minus_simplified_interval"][0] > 0 for row in selections),
            "paired_worse_interval": sum(row["paired_minus_simplified_interval"][1] < 0 for row in selections),
        }
    return {
        "split": dataset["split"], "fit_tasks": int(dataset["fit"].sum()),
        "heldout_tasks": int((~dataset["fit"]).sum()), "summary": summaries,
        "selection_rows": selection_rows, "sign_rows": sign_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-data", type=Path, default=Path('data/agent-as-a-router/data/coderouterbench/id_results_long.csv'))
    parser.add_argument("--router-data", type=Path, default=Path('./data/routerbench/routerbench_0shot.pkl'))
    parser.add_argument("--resamples", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("paired_portfolio_results.json"))
    parser.add_argument("--summary-output", type=Path, default=Path(__file__).with_name("paired_portfolio_summary.json"))
    arguments = parser.parse_args()
    results = {
        "design": "All candidate filters, scores, and choices use fitting tasks only; realized gains and task-bootstrap intervals use disjoint held-out tasks.",
        "paired_score": "mean(V_new - V_old) - eta_old * mean(D_new - D_old), estimated on the fitting split.",
        "paired_gain": "paired_score / mean(D_new), the fitting-split empirical efficiency change.",
        "inference": "95% percentile paired task bootstrap within each context, fitting choices fixed. Aggregate proportions are descriptive; portfolios, nested contexts, and thresholds share tasks.",
        "resamples": arguments.resamples,
    }
    for dataset in [load_code(arguments.code_data), load_router(arguments.router_data)]:
        results[dataset["name"]] = analyze(dataset, arguments.resamples)
    arguments.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    summary = {
        name: {key: value for key, value in content.items() if key not in {"selection_rows", "sign_rows"}}
        if name in {"CodeRouterBench", "RouterBench"} else content
        for name, content in results.items()
    }
    arguments.summary_output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name in ["CodeRouterBench", "RouterBench"]:
        print(name, json.dumps(results[name]["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

