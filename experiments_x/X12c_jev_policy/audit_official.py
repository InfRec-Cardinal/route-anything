#!/usr/bin/env python3
"""Audit the completed Jev Choice replay and write paper-facing summaries.

The Jev calls are already recorded in ``jev_calls_official.jsonl``.  This
script does not call an API; it joins those choices to the released outcome
matrices, computes paired task-bootstrap intervals, and writes a compact
summary plus a cost--quality figure.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("ACROUTER_DATA", "data/agent-as-a-router/data")) / "coderouterbench"
MODELS = [
    "Qwen3-Max", "kimi-k2.5", "qwen3.5-plus", "MiniMax-M2.7",
    "glm-5", "gpt-5.4", "claude-sonnet-4-6", "claude-opus-4-6",
]
DISPLAY = ["Qwen3-Max", "qwen3.5-plus", "gpt-5.4", "claude-opus-4-6"]
SEED = 20260917
N_BOOT = 1000
JEV_INPUT_PRICE = 0.042e-6


def ci(x: np.ndarray) -> list[float]:
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


def load_calls() -> pd.DataFrame:
    rows = [json.loads(line) for line in (HERE / "jev_calls_official.jsonl").read_text().splitlines() if line.strip()]
    frame = pd.DataFrame({
        "task_id": [str(row["task_id"]) for row in rows],
        "split": [str(row["set"]) for row in rows],
        "choice": [row.get("choice") for row in rows],
        "input_tokens": [int((row.get("usage") or {}).get("input_tokens") or 0) for row in rows],
        "error": [row.get("error") for row in rows],
    })
    if frame.error.notna().any():
        raise RuntimeError("official Jev call file contains an error row")
    if frame.task_id.duplicated().any():
        raise RuntimeError("official Jev call file contains duplicate task ids")
    return frame


def bootstrap_summary(frame: pd.DataFrame, outcomes: pd.DataFrame, split: str) -> dict:
    score_col = "score" if split == "ID" else "resolved"
    cost_col = "cost_usd"
    d = outcomes.copy()
    d["task_id"] = d.task_id.astype(str)
    c = frame[frame.split == split].copy().set_index("task_id")
    d = d[d.task_id.isin(c.index)]
    score = d.pivot(index="task_id", columns="model", values=score_col).loc[c.index]
    cost = d.pivot(index="task_id", columns="model", values=cost_col).loc[c.index]
    c = c.loc[score.index]
    policy_score = np.array([score.loc[t, c.loc[t, "choice"]] for t in score.index], float)
    policy_cost = np.array([cost.loc[t, c.loc[t, "choice"]] for t in score.index], float)
    jev_cost = c.input_tokens.to_numpy(float) * JEV_INPUT_PRICE
    total_cost = policy_cost + jev_cost
    q_score = score["Qwen3-Max"].to_numpy(float)
    q_cost = cost["Qwen3-Max"].to_numpy(float)
    rng = np.random.default_rng(SEED + (0 if split == "ID" else 1))
    idx = rng.integers(0, len(score), size=(N_BOOT, len(score)))
    policy_score_b = policy_score[idx].mean(1)
    policy_cost_b = total_cost[idx].mean(1)
    policy_eta_b = policy_score[idx].sum(1) / total_cost[idx].sum(1)
    delta_score_b = (policy_score - q_score)[idx].mean(1)
    delta_cost_b = (total_cost - q_cost)[idx].mean(1)
    policy_pass_b = (policy_score[idx] >= 1).mean(1)
    out = {
        "n": int(len(score)),
        "calls": int(len(c)),
        "input_tokens": int(c.input_tokens.sum()),
        "jev_cost_mean_usd": float(jev_cost.mean()),
        "jev_cost_total_usd": float(jev_cost.sum()),
        "choices": {str(k): int(v) for k, v in c.choice.value_counts().items()},
        "policy": {
            "score_mean": float(policy_score.mean()),
            "score_ci": ci(policy_score_b),
            "pass_at_1": float((policy_score >= 1).mean()),
            "pass_at_1_ci": ci(policy_pass_b),
            "backend_cost_mean_usd": float(policy_cost.mean()),
            "total_cost_mean_usd": float(total_cost.mean()),
            "total_cost_ci": ci(policy_cost_b),
            "efficiency_score_per_usd": float(policy_score.sum() / total_cost.sum()),
            "efficiency_ci": ci(policy_eta_b),
        },
        "fixed": {},
        "paired_vs_qwen": {
            "score_delta": float((policy_score - q_score).mean()),
            "score_delta_ci": ci(delta_score_b),
            "total_cost_delta_usd": float((total_cost - q_cost).mean()),
            "total_cost_delta_ci": ci(delta_cost_b),
        },
    }
    for model in MODELS:
        s = score[model].to_numpy(float)
        co = cost[model].to_numpy(float)
        sb = s[idx].mean(1)
        out["fixed"][model] = {
            "score_mean": float(s.mean()),
            "score_ci": ci(sb),
            "pass_at_1": float((s >= 1).mean()),
            "cost_mean_usd": float(co.mean()),
            "efficiency_score_per_usd": float(s.sum() / co.sum()),
        }
    return out


def make_figure(summary: dict) -> None:
    (HERE / "figures").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    colors = {"Qwen3-Max": "#3568a8", "qwen3.5-plus": "#5e9f55", "gpt-5.4": "#c77c2b", "claude-opus-4-6": "#9a4d83"}
    for ax, split, title in zip(axes, ("ID", "OOD176"), ("ID test", "OOD176")):
        s = summary[split]
        for model in DISPLAY:
            row = s["fixed"][model]
            ax.scatter(row["cost_mean_usd"], row["score_mean"], s=28, color=colors[model], label=model)
        p = s["policy"]
        ax.scatter(p["total_cost_mean_usd"], p["score_mean"], s=55, marker="*", color="#111111", label="Jev Choice")
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("cost / task (USD, log scale)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("mean released score")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(HERE / "figures" / "fig1_jev_policy.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(summary: dict) -> None:
    idr, ood = summary["ID"], summary["OOD176"]
    lines = [
        "# X12c — Jev pre-execution Choice policy",
        "",
        "This completed live experiment evaluates Jev as an ex-ante model selector. It is separate from X8's post-execution verifier: Jev sees the task and model descriptions, while released backend outcomes are joined offline after the call.",
        "",
        f"The official run used {idr['n']} ID-test tasks and {ood['n']} OOD176 tasks, one eight-way Choice call per task, with zero error rows. Jev input cost uses the documented $0.042 per million input tokens. Paired task bootstrap intervals use {N_BOOT} resamples.",
        "",
        "## Main result",
        "",
        "| split | policy | mean score [95% CI] | pass@1 | total cost / task | score / USD |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, s in (("ID", idr), ("OOD176", ood)):
        p = s["policy"]
        lines.append(f"| {name} | Jev Choice | {p['score_mean']:.4f} [{p['score_ci'][0]:.4f}, {p['score_ci'][1]:.4f}] | {p['pass_at_1']:.3f} | ${p['total_cost_mean_usd']:.6f} | {p['efficiency_score_per_usd']:.1f} |")
        q = s["fixed"]["Qwen3-Max"]
        lines.append(f"| {name} | fixed Qwen3-Max | {q['score_mean']:.4f} [{q['score_ci'][0]:.4f}, {q['score_ci'][1]:.4f}] | {q['pass_at_1']:.3f} | ${q['cost_mean_usd']:.6f} | {q['efficiency_score_per_usd']:.1f} |")
    lines += [
        "",
        "The ID policy chose Qwen3-Max 2,724 times, qwen3.5-plus 193 times, and claude-sonnet-4-6 twice. Its score was 0.0082 below fixed Qwen3-Max (paired 95% interval [-0.0142, -0.0016] in `audit_results.json`) while total cost increased by $0.000137 per task. On OOD176 it resolved 47/176 tasks versus 58/176 for fixed Qwen3-Max; the Jev policy cost $0.075656 per task versus $0.023903.",
        "",
        "The policy therefore supplies a useful negative result for the value-of-information analysis: an ex-ante structured signal can be perfectly parseable and inexpensive in absolute dollars yet still lose to a fixed cheap model once its choices are scored on held-out outcomes. The OOD gap is also a reminder that task-level model descriptions do not provide a substitute for measured complementarity or distribution-matched routing data.",
        "",
        "The two-primary Jev replay in REPORT.md remains auxiliary. It uses the X12b-aligned 600/600 sample and is not merged into the official ID/OOD headline.",
    ]
    (HERE / "summary_for_paper.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    calls = load_calls()
    id_outcomes = pd.read_csv(DATA / "id_results_long.csv")
    id_outcomes = id_outcomes[id_outcomes.split == "id_test"]
    ood_outcomes = pd.read_csv(DATA / "ood176_results_long.csv")
    summary = {
        "experiment_id": "X12c_jev_policy",
        "model": "jev-latest",
        "resolved_model": "jev-1.13.0",
        "seed": SEED,
        "n_boot": N_BOOT,
        "price_input_usd_per_token": JEV_INPUT_PRICE,
        "ID": bootstrap_summary(calls, id_outcomes, "ID"),
        "OOD176": bootstrap_summary(calls, ood_outcomes, "OOD176"),
    }
    (HERE / "audit_results.json").write_text(json.dumps(summary, indent=2))
    make_figure(summary)
    write_summary(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

