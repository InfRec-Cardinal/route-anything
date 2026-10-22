"""Bounded OOD176 cost sensitivity from recorded chain paths; no new calls.

Imputation uses each model's positive-cost old112 mean. These scenarios are
estimated missing costs, not provider invoices. Outcomes and chains are fixed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, default=Path('data/agent-as-a-router'))
    p.add_argument('--original-results', type=Path, default=Path('./experiments_x/X7a_regimes_offline/results.json'))
    p.add_argument('--output', type=Path, default=Path(__file__).with_name('ood_cost_imputation_results.json'))
    args = p.parse_args()
    f = pd.read_csv(args.data_root / 'data/coderouterbench/ood176_results_long.csv')
    c = f.pivot(index='task_id', columns='model', values='cost_usd')
    y = f.pivot(index='task_id', columns='model', values='resolved')
    meta = f.drop_duplicates('task_id').set_index('task_id').loc[c.index]
    trace = pd.read_json(args.data_root / 'outputs/acrouter_ood176/ood_decisions.jsonl', lines=True).set_index('task_id').loc[c.index]
    ref = 'claude-opus-4-6'
    old = meta.source_split == 'old112'
    means = {m: float(c.loc[old & (c[m] > 0), m].mean()) for m in c.columns}
    original_cost, cheap_imputed_cost, all_imputed_cost, missing = [], [], [], []
    for task, row in trace.iterrows():
        released = sum(float(c.loc[task, m]) for m in row.chain_run)
        cheap, full = released, released
        for model in row.chain_run:
            if c.loc[task, model] <= 0:
                full += means[model]
                if model != ref:
                    cheap += means[model]
                missing.append({'task_id': task, 'model': model, 'source_split': meta.loc[task, 'source_split']})
        original_cost.append(released); cheap_imputed_cost.append(cheap); all_imputed_cost.append(full)
    original_cost = np.array(original_cost)
    assert np.max(np.abs(original_cost-trace.cost_usd.to_numpy())) < 1e-4
    value = trace.resolved.to_numpy(float)
    ref_value = y[ref].to_numpy(float)
    ref_cost = c[ref].to_numpy(float)
    ref_imputed = np.where(ref_cost > 0, ref_cost, means[ref])
    rng = np.random.default_rng(20260923)
    n = len(c)
    weights = rng.multinomial(n, np.full(n, 1/n), size=1000) / n
    def metrics(policy_cost, baseline_cost):
        policy_cost = np.asarray(policy_cost)
        ev, ed, bv, bd = value.mean(), policy_cost.mean(), ref_value.mean(), baseline_cost.mean()
        vb, db, bvb, bdb = [weights @ a for a in (value, policy_cost, ref_value, baseline_cost)]
        points = {'score': ev, 'mean_cost_usd': ed, 'reference_mean_cost_usd': bd,
                  'efficiency_ratio': (ev/ed)/(bv/bd), 'quality_change': ev-bv}
        draws = {'score': vb, 'mean_cost_usd': db, 'reference_mean_cost_usd': bdb,
                 'efficiency_ratio': (vb/db)/(bvb/bdb), 'quality_change': vb-bvb}
        return {k: [float(v), *np.percentile(draws[k], [2.5, 97.5]).tolist()] for k, v in points.items()}
    scenarios = {
        'released': metrics(original_cost, ref_cost),
        'cheap168_only_reference_released': metrics(cheap_imputed_cost, ref_cost),
        'all169_reference_released': metrics(all_imputed_cost, ref_cost),
        'all169_reference_also_imputed': metrics(all_imputed_cost, ref_imputed),
    }
    previous = json.loads(args.original_results.read_text())
    row = next(r for r in previous['policies'] if r['dataset'] == 'ood176' and r['name'] == 'R3 ACRouter chain trace (imputed attempt costs)')
    assert np.isclose(scenarios['all169_reference_also_imputed']['mean_cost_usd'][0], row['ED'][0], atol=1e-12)
    assert np.isclose(scenarios['all169_reference_also_imputed']['efficiency_ratio'][0], row['r'][0], atol=1e-12)
    result = {'design': 'Fixed recorded chain paths and outcomes. Missing costs imputed by the same model positive-cost old112 mean. All-zero reference cells receive the same imputation in the matched scenario. No actual invoice is inferred. 1000 paired task-bootstrap intervals at seed 20260923; imputation means fixed, so intervals exclude uncertainty about imputation accuracy.',
              'original_result': 'The prior imputed-cost implementation already replaced missing costs for every model, including Opus and the reference. The published 168 counted cheap attempts only; 169 is the complete chain count.',
              'attempts': int(trace.chain_run.map(len).sum()), 'missing_attempts': len(missing),
              'missing_cheap_attempts': sum(x['model'] != ref for x in missing),
              'missing_reference_attempts_in_chain': sum(x['model'] == ref for x in missing),
              'tasks_with_missing_attempt': len({x['task_id'] for x in missing}),
              'missing_attempt_source_splits': sorted({x['source_split'] for x in missing}),
              'missing_always_Opus_reference_cells': int((ref_cost <= 0).sum()),
              'imputation_means_usd': means, 'scenarios': scenarios,
              'previous_matched_imputation': {k: row[k] for k in ('EV', 'ED', 'r', 'Q')}}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

