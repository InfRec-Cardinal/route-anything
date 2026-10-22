"""Post-review offline reanalysis: cost baselines, policy outcomes, missingness.

Rules were specified after inspection of the original held-out results. Fits use
probing only, but these are exploratory reanalyses, not a fresh confirmation set.
No model calls are made. Context/portfolio averages are descriptive, not
independent replicates. Costs are released token-price estimates.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paired_portfolio_analysis import load_code, load_router, portfolio_tables


def portfolio_analysis(dataset, keep=None):
    keep = np.ones(len(dataset['fit']), bool) if keep is None else keep
    bases = sorted(tuple(x) for n in (1, 2) for x in itertools.combinations(dataset['pool'], n))
    portfolios = sorted(set(bases) | {tuple(sorted((*x, m))) for x in bases
        for m in range(len(dataset['models'])) if m not in x})
    index = {x: j for j, x in enumerate(portfolios)}
    rows = []
    for context, cmask in dataset['contexts']:
        if context == 'overall':
            continue
        fit = cmask & dataset['fit'] & keep
        test = cmask & ~dataset['fit'] & keep
        for theta in dataset['thresholds']:
            if dataset['name'] == 'RouterBench' and theta == 1:
                yy = np.column_stack((dataset['scores'][cmask], dataset['reference_scores'][cmask]))
                if np.isin(yy, [0, 1]).all():
                    continue
            f = portfolio_tables(dataset, fit, theta, portfolios)
            t = portfolio_tables(dataset, test, theta, portfolios)
            fv, fd = f['value'].mean(0), f['resource'].mean(0)
            tv, td = t['value'].mean(0), t['resource'].mean(0)
            feta, teta = fv / fd, tv / td
            passes = f['passing'].sum(0)
            usable = (passes >= 20) & (len(f['passing']) - passes >= 20)
            for base in bases:
                if not usable[list(base)].all():
                    continue
                bi = index[base]
                hit = f['hit'][:, bi]
                candidates = [m for m in range(len(dataset['models'])) if m not in base and usable[m]
                    and np.sum(f['passing'][:, m] & ~hit) >= 10
                    and np.sum(~(f['passing'][:, m] | hit)) >= 10 and hit.sum() >= 10]
                if len(candidates) < 3:
                    continue
                ei = [index[tuple(sorted((*base, m)))] for m in candidates]
                residual = np.array([(f['passing'][:, m] & ~hit).mean() for m in candidates])
                c = f['costs'][:, candidates].mean(0)
                vh = f['value'][hit, bi].mean()
                vm = f['reference_scores'][~hit].mean()
                b = f['reference_costs'][~hit].mean()
                pair = fv[ei] - fv[bi] - feta[bi] * (fd[ei] - fd[bi])
                scores = {
                    'paired': pair / fd[ei],
                    'residual': residual * (vh - vm + feta[bi] * b) - feta[bi] * c,
                    'cost_only': residual * b - c,
                    'marginal_success': f['passing'][:, candidates].mean(0),
                }
                choices = {k: int(np.argmax(s)) for k, s in scores.items()}
                oracle = int(np.argmax(teta[ei]))
                stats = {}
                for rule, pos in choices.items():
                    pi = ei[pos]
                    stats[rule] = {
                        'best': bool(np.isclose(teta[pi], teta[ei[oracle]], rtol=1e-12, atol=1e-12)),
                        'value': float(tv[pi]), 'cost': float(td[pi]),
                        'quality_change': float(tv[pi] - tv[bi]),
                        'cost_saving': float(td[bi] - td[pi]),
                        'efficiency_gain': float(teta[pi] - teta[bi]),
                        'relative_efficiency_gain': float((teta[pi] - teta[bi]) / teta[bi]),
                        'relative_efficiency_regret': float((teta[ei[oracle]] - teta[pi]) / teta[ei[oracle]]),
                        'surplus_at_fit_base_price': float(tv[pi] - tv[bi] + feta[bi] * (td[bi] - td[pi])),
                        'candidate': dataset['models'][candidates[pos]],
                    }
                rows.append({'context': context, 'theta': theta, 'portfolio': [dataset['models'][m] for m in base],
                             'stats': stats, 'n_test': int(test.sum())})
    summary = {}
    for theta in dataset['thresholds']:
        selected = [r for r in rows if r['theta'] == theta]
        if not selected:
            continue
        summary[str(theta)] = {'portfolios': len(selected), 'rules': {}}
        for rule in selected[0]['stats']:
            summary[str(theta)]['rules'][rule] = {k: float(np.mean([r['stats'][rule][k] for r in selected]))
                for k in selected[0]['stats'][rule] if k != 'candidate'}
        for baseline in ('residual', 'cost_only'):
            delta = {}
            for metric in ('value', 'cost', 'surplus_at_fit_base_price'):
                delta[metric] = float(np.mean([r['stats']['paired'][metric] - r['stats'][baseline][metric] for r in selected]))
            delta['changed_choices'] = sum(r['stats']['paired']['candidate'] != r['stats'][baseline]['candidate'] for r in selected)
            summary[str(theta)]['paired_minus_' + baseline] = delta
    examples = sorted(rows, key=lambda r: r['stats']['residual']['relative_efficiency_regret'] -
                      r['stats']['paired']['relative_efficiency_regret'], reverse=True)[:3]
    return {'fit_tasks': int((dataset['fit'] & keep).sum()), 'test_tasks': int((~dataset['fit'] & keep).sum()),
            'summary': summary, 'largest_residual_corrections': examples}


def ci(values):
    return [float(np.mean(values)), *np.percentile(values, [2.5, 97.5]).tolist()]


def one_step(frame, old_rows):
    rows = pd.read_csv(old_rows)
    rows = rows[(rows.beta_fa == 0) & (rows.beta_fr == 0) & rows.part.isin(['dim9', 'cell21'])].copy()
    S = frame.pivot(index='task_id', columns='model', values='score')
    C = frame.pivot(index='task_id', columns='model', values='cost_usd')
    meta = frame.drop_duplicates('task_id').set_index('task_id').loc[S.index]
    meta['cell21'] = meta.dimension + '|' + meta.index.str.split('_').str[1]
    dv, saving = [], []
    for row in rows.itertuples():
        state = meta.dimension if row.part == 'dim9' else meta.cell21
        mask = (meta.split == 'id_test') & (state == row.cell)
        primary = S.loc[mask, row.cheap].to_numpy()
        reference = S.loc[mask, 'claude-opus-4-6'].to_numpy()
        reference_cost = C.loc[mask, 'claude-opus-4-6'].to_numpy()
        accepted = primary >= row.theta
        quality = float(np.mean(accepted * (primary-reference)))
        saving.append(float(np.mean(accepted * reference_cost - C.loc[mask, row.cheap].to_numpy())))
        dv.append(quality)
        assert np.isclose(quality + row.lam * saving[-1], row.dJ_meas, atol=1e-12), (row.part, row.cell)
    rows['delta_quality'] = dv
    rows['dollar_saving'] = saving
    out = {}
    for part, group in rows.groupby('part'):
        truth = group.dJ_meas >= 0
        out[part] = {'comparisons': len(group), 'decided': int(group.meas_decided.sum()), 'rules': {}}
        for name, choices in {'full_value_cost': group.pred_cheap_first,
                              'cost_threshold': group.price_rule_cf,
                              'always_attempt': pd.Series(True, index=group.index)}.items():
            decided = group.meas_decided
            regret = np.where(choices == truth, 0, np.abs(group.dJ_meas))
            out[part]['rules'][name] = {
                'agreement_all': float((choices == truth).mean()),
                'agreement_decided': float((choices[decided] == truth[decided]).mean()),
                'mean_surplus_regret_score_units': float(regret.mean()),
                'mean_money_equivalent_regret': float((regret / group.lam).mean()),
                'attempt_fraction': float(choices.mean()),
                'mean_quality_change': float((choices * group.delta_quality).mean()),
                'mean_dollar_saving': float((choices * group.dollar_saving).mean()),
            }
        probing = meta.split == 'probing'
        price = round(float(S.loc[probing, 'claude-opus-4-6'].mean() /
                            C.loc[probing, 'claude-opus-4-6'].mean()), 3)
        subset = group[group.lam == price]
        out[part]['reference_price_only'] = {'lambda': price, 'comparisons': len(subset), 'rules': {}}
        truth = subset.dJ_meas >= 0
        for name, choices in {'full_value_cost': subset.pred_cheap_first,
                              'cost_threshold': subset.price_rule_cf,
                              'always_attempt': pd.Series(True, index=subset.index)}.items():
            loss = np.where(choices == truth, 0, np.abs(subset.dJ_meas))
            out[part]['reference_price_only']['rules'][name] = {
                'agreement_all': float((choices == truth).mean()),
                'mean_surplus_regret_score_units': float(loss.mean()),
                'mean_money_equivalent_regret': float(loss.mean()/price),
                'mean_quality_change': float((choices * subset.delta_quality).mean()),
                'mean_dollar_saving': float((choices * subset.dollar_saving).mean()),
            }
    return {'protocol': 'Original two primaries, two thresholds and five fixed resource prices; no verifier noise. The price rule was already stored in the original analysis. Loss compares each probing-selected attempt/direct choice with the better held-out action at the stated price. Grid means share tasks and mix resource prices.',
            'summary': out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data/agent-as-a-router'))
    parser.add_argument('--theory-root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('decision_baselines_results.json'))
    args = parser.parse_args()
    code_path = args.data_root / 'data/coderouterbench/id_results_long.csv'
    code = load_code(code_path)
    frame = pd.read_csv(code_path)
    scores = frame.pivot(index='task_id', columns='model', values='score')
    costs = frame.pivot(index='task_id', columns='model', values='cost_usd')
    latency = frame.pivot(index='task_id', columns='model', values='latency_ms')
    missing = (costs['Qwen3-Max'] == 0) & (scores['Qwen3-Max'] == 0)
    valid = (costs > 0).all(axis=1) & (latency > 0).all(axis=1)
    result = {
        'design': 'Exploratory reanalysis designed after original test outcomes were inspected. Rules and candidates are fit on probing only. No new untouched evaluation population is claimed.',
        'portfolio_protocol': 'Same original candidate filters and contexts. Equal-weight means over context-portfolio configurations; overlapping configurations are not independent. Missing-Qwen sensitivity removes the entire affected task from every model, then refits on retained probing tasks. Cost-only score is residual coverage times mean reference cost on base failures minus candidate cost.',
        'missing_tasks': int(missing.sum()), 'missing_fit': int((missing.to_numpy() & code['fit']).sum()),
        'missing_test': int((missing.to_numpy() & ~code['fit']).sum()),
        'common_resource_valid_tasks': int(valid.sum()),
        'CodeRouterBench': portfolio_analysis(code),
        'CodeRouterBench_drop_missing': portfolio_analysis(code, ~missing.to_numpy()),
        'RouterBench': portfolio_analysis(load_router(args.theory_root / 'data/routerbench/routerbench_0shot.pkl')),
        'one_step': one_step(frame, args.theory_root / 'experiments/X3_threshold_voi/a_threshold_cells.csv'),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(args.output)
    for name in ('CodeRouterBench', 'CodeRouterBench_drop_missing', 'RouterBench'):
        print(name, json.dumps(result[name]['summary'], indent=2))
    print(json.dumps(result['one_step'], indent=2))


if __name__ == '__main__':
    main()

