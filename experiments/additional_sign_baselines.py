"""Additional gain-sign controls on the released portfolio evaluation splits.

Reuses the original loaders, split, acceptance thresholds and candidate filters.
Reports point agreement on every eligible held-out comparison; it does not rerun
or select comparisons by bootstrap intervals. This is an exploratory reanalysis
after the original test results were inspected, not an untouched confirmation.

Pass --code-data and --router-data, or set CODEROUTERBENCH_MATRIX and
ROUTERBENCH_MATRIX. Defaults are relative to the working directory.
"""

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np

from paired_portfolio_analysis import load_code, load_router, portfolio_tables


def analyze(dataset):
    model_count = len(dataset['models'])
    bases = sorted(tuple(members) for size in (1, 2)
                   for members in itertools.combinations(sorted(dataset['pool']), size))
    portfolios = sorted(set(bases) | {
        tuple(sorted((*members, candidate))) for members in bases
        for candidate in range(model_count) if candidate not in members
    })
    portfolio_index = {members: index for index, members in enumerate(portfolios)}
    outcomes = {str(threshold): [] for threshold in dataset['thresholds']}
    for context, context_rows in dataset['contexts']:
        if context == 'overall':
            continue
        fit_rows = context_rows & dataset['fit']
        test_rows = context_rows & ~dataset['fit']
        for threshold in dataset['thresholds']:
            if dataset['name'] == 'RouterBench' and threshold == 1.0:
                all_scores = np.column_stack((dataset['scores'][context_rows],
                                              dataset['reference_scores'][context_rows]))
                if np.isin(all_scores, [0, 1]).all():
                    continue
            fitting = portfolio_tables(dataset, fit_rows, threshold, portfolios)
            heldout = portfolio_tables(dataset, test_rows, threshold, portfolios)
            fit_value = fitting['value'].mean(axis=0)
            fit_cost = fitting['resource'].mean(axis=0)
            fit_efficiency = fit_value / fit_cost
            test_efficiency = heldout['value'].mean(axis=0) / heldout['resource'].mean(axis=0)
            passes = fitting['passing'].sum(axis=0)
            usable = (passes >= 20) & (len(fitting['passing']) - passes >= 20)
            for members in bases:
                if not usable[list(members)].all():
                    continue
                base = portfolio_index[members]
                hit = fitting['hit'][:, base]
                candidates = [candidate for candidate in range(model_count)
                              if candidate not in members and usable[candidate]
                              and np.sum(fitting['passing'][:, candidate] & ~hit) >= 10
                              and np.sum(~(fitting['passing'][:, candidate] | hit)) >= 10
                              and hit.sum() >= 10]
                if not candidates or (dataset['name'] == 'CodeRouterBench' and len(candidates) < 3):
                    continue
                expanded = [portfolio_index[tuple(sorted((*members, candidate)))]
                            for candidate in candidates]
                residual = np.array([np.mean(fitting['passing'][:, candidate] & ~hit)
                                     for candidate in candidates])
                candidate_cost = fitting['costs'][:, candidates].mean(axis=0)
                fallback_cost = fitting['reference_costs'][~hit].mean()
                accepted_value = fitting['value'][hit, base].mean()
                missed_value = fitting['reference_scores'][~hit].mean()
                efficiency = fit_efficiency[base]
                weight = accepted_value - missed_value + efficiency * fallback_cost
                marginal = fitting['passing'][:, candidates].mean(axis=0)
                scores = {
                    'paired': fit_value[expanded] - fit_value[base]
                              - efficiency * (fit_cost[expanded] - fit_cost[base]),
                    'residual': residual * weight - efficiency * candidate_cost,
                    'independence': (1 - hit.mean()) * marginal * weight - efficiency * candidate_cost,
                    'cost_only': residual * fallback_cost - candidate_cost,
                    'always_negative': np.full(len(candidates), -1.0),
                }
                truth = np.sign(test_efficiency[expanded] - test_efficiency[base])
                for index in range(len(candidates)):
                    outcomes[str(threshold)].append({name: bool(np.sign(score[index]) == truth[index])
                                                    for name, score in scores.items()})
    summary = {}
    for threshold, rows in outcomes.items():
        if not rows:
            continue
        summary[threshold] = {
            'comparisons': len(rows),
            'correct': {name: sum(row[name] for row in rows) for name in rows[0]},
            'agreement_all': {name: float(np.mean([row[name] for row in rows])) for name in rows[0]},
        }
    return {'fit_tasks': int(dataset['fit'].sum()),
            'heldout_tasks': int((~dataset['fit']).sum()), 'summary': summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-data', type=Path,
                        default=Path(os.environ.get('CODEROUTERBENCH_MATRIX',
                                                    'data/coderouterbench/id_results_long.csv')))
    parser.add_argument('--router-data', type=Path,
                        default=Path(os.environ.get('ROUTERBENCH_MATRIX',
                                                    'data/routerbench/routerbench_0shot.pkl')))
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('additional_sign_baselines.json'))
    args = parser.parse_args()
    result = {
        'design': 'Exploratory point-statistic reanalysis after inspection of original held-out results. Primitive estimates use fitting tasks only. All comparisons are retained without bootstrap filtering. Contexts and portfolios share tasks.',
        'cost_only': 'r_m(S) times the fitting mean reference cost conditional on base-portfolio failure, minus the fitting mean candidate calling cost.',
        'always_negative': 'Predict a negative held-out efficiency change for every addition.',
        'protocol': 'Unchanged original splits, thresholds, contexts, portfolios and candidate filters from paired_portfolio_analysis. RouterBench theta=1 includes only nonbinary contexts. CodeRouterBench retains released missing rows.',
    }
    for dataset in (load_code(args.code_data), load_router(args.router_data)):
        result[dataset['name']] = analyze(dataset)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

