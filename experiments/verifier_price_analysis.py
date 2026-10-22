"""Report measured judge tokens and price ceilings from archived outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def analyze(rows, judge_rows):
    reference_quality = float(rows.Y_B.mean())
    reference_cost = float(rows.opus_cost_usd.mean())
    reference_efficiency = reference_quality / reference_cost
    results = []
    for model in ["Qwen3-Max", "kimi-k2.5"]:
        calls = [judge_rows[f"{model}|{task}"] for task in rows.task_id]
        mean_input = float(np.mean([call["prompt_tokens"] for call in calls]))
        mean_output = float(np.mean([call["completion_tokens"] for call in calls]))
        for threshold in [0.5, 1.0]:
            accepted = rows[f"V3_{model}"] >= threshold
            quality = np.where(accepted, rows[f"Y_{model}"], rows.Y_B)
            backend_cost = rows[f"cost_{model}"] + (~accepted) * rows.opus_cost_usd
            mean_quality = float(quality.mean())
            mean_backend_cost = float(backend_cost.mean())
            ceiling = mean_quality / reference_efficiency - mean_backend_cost
            results.append({
                "model": "Kimi-K2.5" if model == "kimi-k2.5" else model,
                "threshold": threshold,
                "tasks": len(rows),
                "mean_judge_input_tokens": mean_input,
                "mean_judge_output_tokens": mean_output,
                "mean_judge_total_tokens": float(np.mean([call["total_tokens"] for call in calls])),
                "mean_final_quality": mean_quality,
                "mean_quality_difference": mean_quality - reference_quality,
                "mean_backend_cost_usd": mean_backend_cost,
                "mean_reference_cost_usd": reference_cost,
                "reference_efficiency_per_usd": reference_efficiency,
                "max_mean_judge_charge_usd_for_efficiency_gain": ceiling,
                "uniform_input_output_price_ceiling_usd_per_million_tokens": ceiling * 1e6 / (mean_input + mean_output),
            })
    return {
        "design": f"Point estimates from {len(rows)} archived id_test tasks; thresholds 0.5 and 1.0 are reported without selecting them on held-out outcomes.",
        "judge": "Grok-4.6",
        "actual_judge_dollar_price": None,
        "price_status": "The archived judge calls provide token counts and estimated relay units, without a substantiated dollar invoice or input/output dollar price.",
        "intervals": "Not recomputed. All reported values are point estimates.",
        "cost_ceiling_formula": "mean(V_policy) / eta_reference - mean(primary_cost + rejected * fallback_cost)",
        "price_frontier": "For input/output prices p_in and p_out in USD per million tokens, efficiency exceeds the reference iff (mean_input_tokens * p_in + mean_output_tokens * p_out) / 1e6 < max_mean_judge_charge_usd_for_efficiency_gain.",
        "interpretation": "The quality difference must satisfy a separately chosen quality floor. A negative cost ceiling means the policy cannot exceed reference efficiency even with a free judge. The uniform-token price ceiling is a scenario, not an observed price.",
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    archive = Path('./experiments_x/X8_operating_point')
    parser.add_argument("--rows", type=Path, default=archive / "rows_id_test.csv")
    parser.add_argument("--judge", type=Path, default=archive / "cache/judge.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("verifier_price_results.json"))
    arguments = parser.parse_args()
    rows = pd.read_csv(arguments.rows)
    judge_rows = json.loads(arguments.judge.read_text())["rows"]
    results = analyze(rows, judge_rows)
    arguments.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    for result in results["results"]:
        print(result["model"], result["threshold"], result["max_mean_judge_charge_usd_for_efficiency_gain"])


if __name__ == "__main__":
    main()

