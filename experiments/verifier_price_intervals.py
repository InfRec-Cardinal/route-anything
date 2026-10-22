"""Paired task-bootstrap price ceilings, holding archived decisions fixed."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def analyze(rows, resamples=2000):
    rng = np.random.default_rng(20260923)
    n = len(rows)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=resamples) / n
    reference_value = rows.Y_B.to_numpy(float)
    reference_cost = rows.opus_cost_usd.to_numpy(float)
    eta_reference = reference_value.mean() / reference_cost.mean()
    eta_boot = (weights @ reference_value) / (weights @ reference_cost)
    results = []
    for model in ['Qwen3-Max', 'kimi-k2.5']:
        for threshold in [0.5, 1.0]:
            accepted = rows[f'V3_{model}'].to_numpy() >= threshold
            value = np.where(accepted, rows[f'Y_{model}'], reference_value)
            cost = rows[f'cost_{model}'].to_numpy() + (~accepted) * reference_cost
            cap = value.mean() / eta_reference - cost.mean()
            cap_boot = (weights @ value) / eta_boot - weights @ cost
            results.append(dict(model=model, threshold=threshold, tasks=n,
                                ceiling_usd=float(cap),
                                interval_usd=np.percentile(cap_boot, [2.5, 97.5]).tolist()))
    return dict(resamples=resamples, seed=20260923, results=results,
                interpretation='Paired percentile task bootstrap; all archived outputs and thresholds fixed. Reference efficiency is recomputed in each resample. Sampling uncertainty only; no actual vendor price inferred.')


if __name__ == '__main__':
    rows = pd.read_csv(Path('./experiments_x/X8_operating_point/rows_id_test.csv'))
    result = analyze(rows)
    Path(__file__).with_name('verifier_price_intervals.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

