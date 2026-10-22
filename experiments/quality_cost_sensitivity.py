"""Compare frozen Appendix-A policies jointly in quality, dollars and seconds.

Keeps original fitted policies fixed when dropping missing/common-invalid tasks.
The cheapest fixed model is selected using probing cost, never test cost.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, default=Path('data/agent-as-a-router'))
    p.add_argument('--theory-root', type=Path, default=Path('.'))
    p.add_argument('--output', type=Path, default=Path(__file__).with_name('quality_cost_sensitivity_results.json'))
    args = p.parse_args()
    frame = pd.read_csv(args.data_root / 'data/coderouterbench/id_results_long.csv')
    S = frame.pivot(index='task_id', columns='model', values='score')
    C = frame.pivot(index='task_id', columns='model', values='cost_usd')
    L = frame.pivot(index='task_id', columns='model', values='latency_ms') / 1000
    meta = frame.drop_duplicates('task_id').set_index('task_id').loc[S.index]
    fit = meta.split.to_numpy() == 'probing'
    test = ~fit
    models = list(S.columns)
    qwen = models.index('Qwen3-Max')
    ref = models.index('claude-opus-4-6')
    y, c, l = S.to_numpy(), C.to_numpy(), L.to_numpy()
    missing = (y[:, qwen] == 0) & (c[:, qwen] == 0)
    common = (c > 0).all(1) & (l > 0).all(1)
    cells = (meta.dimension + '|' + meta.index.str.split('_').str[1]).to_numpy()
    old = json.loads((args.theory_root / 'experiments_x/X9_optimal_routing/results.json').read_text())
    names = [x['name'] for x in old['menu']]
    vals, usd, secs = [], [], []
    for action in old['menu']:
        j = models.index(action['model'])
        if action['kind'] != 'attempt':
            vals.append(y[:, j]); usd.append(c[:, j]); secs.append(l[:, j])
        else:
            h = y[:, j] >= action['theta']
            vals.append(np.where(h, y[:, j], y[:, ref]))
            usd.append(c[:, j] + ~h * c[:, ref]); secs.append(l[:, j] + ~h * l[:, ref])
    V, D, T = np.column_stack(vals), np.column_stack(usd), np.column_stack(secs)
    policies = {}
    def fixed(action):
        j = names.index(action)
        return V[:, j], D[:, j], T[:, j]
    def mixture(mapping):
        w = np.zeros_like(V)
        for state, actions in mapping.items():
            for a, prob in actions.items():
                w[cells == state, names.index(a)] = prob
        return (w * V).sum(1), (w * D).sum(1), (w * T).sum(1)
    for basis in ('usd', 'sec'):
        run = old['runs'][basis + '/cell21']
        mapping = {s: {a: 1.0} for s, a in zip(old['states']['cell21']['names'], run['optimal_rule']['actions'])}
        policies['21_cells_' + basis + '_unconstrained'] = mixture(mapping)
        if basis == 'usd':
            for eps in ('0', '0.01'):
                policies['21_cells_usd_eps=' + eps] = mixture(run['floor'][eps]['g_nonzero'])
    cheapest = models[int(np.argmin(c[fit].mean(0)))]
    policies['cheapest_fixed:' + cheapest] = fixed('direct:' + cheapest)
    policies['best_single_dispatch'] = fixed(old['baselines']['usd']['best_single_direct']['name'])
    policies['best_fixed_cascade'] = fixed(old['baselines']['usd']['best_fixed_cascade']['name'])
    policies['always_Opus'] = fixed('direct:claude-opus-4-6')
    for label, path in {
        'DimensionBest': args.data_root / 'data/id/voter_decisions/dimension_best.jsonl',
        'ACRouter': args.data_root / 'outputs/current/id_decisions.jsonl',
    }.items():
        dec = pd.read_json(path, lines=True).drop_duplicates('task_id').set_index('task_id').reindex(S.index)
        choice = dec.chosen_model.map({m: i for i, m in enumerate(models)}).fillna(ref).to_numpy(int)
        ii = np.arange(len(y))
        policies[label] = y[ii, choice], c[ii, choice], l[ii, choice]
    rng = np.random.default_rng(20260923)
    def summarize(mask):
        n = int(mask.sum())
        weights = rng.multinomial(n, np.full(n, 1/n), size=1000) / n
        reference_y, reference_c, reference_l = y[mask, ref], c[mask, ref], l[mask, ref]
        ref_boot = [weights @ v for v in (reference_y, reference_c, reference_l)]
        lam = float(y[fit, ref].sum() / c[fit, ref].sum())
        rows = {}
        for name, arrays in policies.items():
            vv, dd, tt = [a[mask] for a in arrays]
            vb, db, tb = [weights @ a for a in (vv, dd, tt)]
            point = {'score': float(vv.mean()), 'cost_usd': float(dd.mean()), 'seconds': float(tt.mean()),
                     'quality_change': float((vv-reference_y).mean()),
                     'dollar_saving': float((reference_c-dd).mean()),
                     'dollar_efficiency_ratio': float(vv.sum()/dd.sum()/(reference_y.sum()/reference_c.sum())),
                     'seconds_efficiency_ratio': float(vv.sum()/tt.sum()/(reference_y.sum()/reference_l.sum())),
                     'surplus_fit_reference_price': float((vv-reference_y+lam*(reference_c-dd)).mean())}
            samples = {'score': vb, 'cost_usd': db, 'seconds': tb, 'quality_change': vb-ref_boot[0],
                       'dollar_saving': ref_boot[1]-db,
                       'dollar_efficiency_ratio': (vb/db)/(ref_boot[0]/ref_boot[1]),
                       'seconds_efficiency_ratio': (vb/tb)/(ref_boot[0]/ref_boot[2]),
                       'surplus_fit_reference_price': vb-ref_boot[0]+lam*(ref_boot[1]-db)}
            rows[name] = {key: [v, *np.percentile(samples[key], [2.5, 97.5]).tolist()] for key, v in point.items()}
        return {'test_tasks': n, 'policies': rows}
    result = {'design': 'Exploratory comparison of frozen original probing-fitted policies. All masks apply to every compared policy. Dollar/seconds comparison uses the same positive-cost, positive-latency task intersection across all eight models. Recorded sequential latencies omit orchestration and verifier time. Intervals are 1000 paired task bootstraps; no policy is refitted for these outcome sensitivities.',
              'cheapest_fixed_model_probing': cheapest,
              'released': summarize(test), 'drop_missing_Qwen': summarize(test & ~missing),
              'common_resource_valid': summarize(test & common)}
    all_fixed = {}
    for j, model in enumerate(models):
        all_fixed[model] = {'probing_cost_usd': float(c[fit, j].mean()), 'test_cost_usd': float(c[test, j].mean()),
                            'test_score': float(y[test, j].mean())}
    result['fixed_models'] = all_fixed
    result['fourteen_fixed_cascades_common_resources'] = {}
    keep = test & common
    for a in old['menu']:
        if a['kind'] != 'attempt':
            continue
        j = names.index(a['name'])
        result['fourteen_fixed_cascades_common_resources'][a['name']] = {
            'dollar_ratio': float(V[keep,j].sum()/D[keep,j].sum()/(y[keep,ref].sum()/c[keep,ref].sum())),
            'seconds_ratio': float(V[keep,j].sum()/T[keep,j].sum()/(y[keep,ref].sum()/l[keep,ref].sum()))}
    original = json.loads((args.theory_root / 'experiments_x/X7a_regimes_offline/results.json').read_text())
    result['ood176_existing_cost_comparison'] = [{k: row[k] for k in ('name','EV','ED','r','Q','net_usd')}
        for row in original['policies'] if row['dataset'] == 'ood176' and
        ('DimensionBest' in row['name'] or 'dimension_best' in row['name'] or 'chain trace (imputed' in row['name'])]
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(args.output)
    for key in ('released', 'drop_missing_Qwen', 'common_resource_valid'):
        print(key, result[key]['test_tasks'])
        for name, row in result[key]['policies'].items():
            print(name, 'score/cost_m$/seconds', row['score'][0], row['cost_usd'][0]*1000, row['seconds'][0],
                  'ratios', row['dollar_efficiency_ratio'][0], row['seconds_efficiency_ratio'][0])
    print('OOD', result['ood176_existing_cost_comparison'])


if __name__ == '__main__':
    main()

