"""Frozen probing-to-test complementarity selection; offline oracle-verifier replay."""
from pathlib import Path
import argparse
import itertools
import json
import numpy as np
import pandas as pd

MODELS = ['Qwen3-Max', 'kimi-k2.5', 'qwen3.5-plus', 'glm-5',
          'MiniMax-M2.7', 'gpt-5.4', 'claude-sonnet-4-6']
FALLBACK = 'claude-opus-4-6'
RULES = ['standalone', 'independence', 'residual', 'full_conditional']


def evaluate(members, passing, costs, fallback_value, fallback_cost):
    hit = passing[:, members].any(axis=1)
    return np.where(hit, 1., fallback_value), costs[:, members].sum(axis=1) + (~hit) * fallback_cost


def scores_for(members, passing, costs, fallback_value, fallback_cost):
    v, d = evaluate(members, passing, costs, fallback_value, fallback_cost)
    eta = v.mean() / d.mean()
    hit = passing[:, members].any(axis=1)
    remaining = ~hit
    # Empty residual set implies zero coverage increment, so the weight is irrelevant.
    weight = 1 - fallback_value[remaining].mean() + eta * fallback_cost[remaining].mean() if remaining.any() else 0.
    result = {}
    for m in range(len(MODELS)):
        if m in members:
            continue
        marginal = passing[:, m].mean()
        residual = np.mean(remaining & passing[:, m])
        independent = remaining.mean() * marginal
        vn, dn = evaluate(members + [m], passing, costs, fallback_value, fallback_cost)
        result[m] = {
            'standalone': marginal,
            'independence': independent * weight - eta * costs[:, m].mean(),
            'residual': residual * weight - eta * costs[:, m].mean(),
            'full_conditional': np.mean(vn - v) - eta * np.mean(dn - d),
            'residual_coverage': residual, 'independent_coverage': independent,
            'candidate_mean_cost_usd': costs[:, m].mean(),
            'fit_eta_old': eta, 'fit_eta_new': vn.mean() / dn.mean(),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to the released CodeRouterBench id_results_long.csv')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'portfolio_stopping_results')
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.data)
    s = raw.pivot(index='task_id', columns='model', values='score')
    c = raw.pivot(index='task_id', columns='model', values='cost_usd').loc[s.index]
    meta = raw.drop_duplicates('task_id').set_index('task_id').loc[s.index]
    assert not s.isna().any().any() and not c.isna().any().any()
    fit = meta['split'].eq('probing').to_numpy()
    test = meta['split'].eq('id_test').to_numpy()
    passing, costs = s[MODELS].to_numpy() >= 1., c[MODELS].to_numpy()
    fv, fc = s[FALLBACK].to_numpy(), c[FALLBACK].to_numpy()
    dims = sorted(meta.dimension.unique())
    test_global_positions = np.flatnonzero(test)
    test_lookup = {int(j): i for i, j in enumerate(test_global_positions)}
    n = int(test.sum())
    test_meta = meta.iloc[test_global_positions]
    # Axes: task, fixed primary. Task remains the resampling unit.
    baseline_v = np.zeros((n, len(MODELS)))
    baseline_d = np.zeros_like(baseline_v)
    oracle_options = {size: [] for size in [2, 3]}
    predicted = {}
    diagnostics = []
    selected_rows, candidate_rows = [], []
    for stop in [False, True]:
        mode = 'positive_stop' if stop else 'forced_size'
        for rule in RULES:
            for size in [2, 3]:
                predicted[(mode, rule, size)] = (np.zeros_like(baseline_v), np.zeros_like(baseline_d))
    for dimension in dims:
        dimmask = meta.dimension.eq(dimension).to_numpy()
        fi = np.flatnonzero(dimmask & fit)
        ti = np.flatnonzero(dimmask & test)
        target = np.array([test_lookup[int(i)] for i in ti])
        fp, fcost, ffv, ffc = passing[fi], costs[fi], fv[fi], fc[fi]
        tp, tcost, tfv, tfc = passing[ti], costs[ti], fv[ti], fc[ti]
        for primary in range(len(MODELS)):
            bv, bd = evaluate([primary], tp, tcost, tfv, tfc)
            baseline_v[target, primary], baseline_d[target, primary] = bv, bd
            base_eta = bv.sum() / bd.sum()
            oracle = {}
            for size in [2, 3]:
                best = None
                all_values, all_costs, all_members = [], [], []
                for extra in itertools.combinations([m for m in range(len(MODELS)) if m != primary], size - 1):
                    members = [primary] + list(extra)
                    ov, od = evaluate(members, tp, tcost, tfv, tfc)
                    eta = ov.sum() / od.sum()
                    all_values.append(ov)
                    all_costs.append(od)
                    all_members.append(members)
                    if best is None or eta > best[0]:
                        best = (eta, members)
                oracle[size] = best
                oracle_options[size].append((target, primary, np.array(all_values), np.array(all_costs), all_members))
            for stop in [False, True]:
                mode = 'positive_stop' if stop else 'forced_size'
                for rule in RULES:
                    members = [primary]
                    stopped = False
                    for size in [2, 3]:
                        options = scores_for(members, fp, fcost, ffv, ffc)
                        chosen = max(options, key=lambda k: options[k][rule])
                        # Standalone has no economic criterion; its stop variant uses
                        # the exact fitted net value of its quality-ranked proposal.
                        stop_score = options[chosen]['full_conditional' if rule == 'standalone' else rule]
                        addition = not stopped and (not stop or stop_score > 0)
                        for candidate, stat in options.items():
                            candidate_rows.append(dict(dimension=dimension, primary=MODELS[primary],
                                mode=mode, rule=rule, target_size=size,
                                current_members='|'.join(MODELS[m] for m in members),
                                candidate=MODELS[candidate], selected=addition and candidate == chosen,
                                **{k: float(v) for k, v in stat.items()}))
                        if addition:
                            members.append(chosen)
                        else:
                            stopped = True
                        v, d = evaluate(members, tp, tcost, tfv, tfc)
                        pv, pd_ = predicted[(mode, rule, size)]
                        pv[target, primary], pd_[target, primary] = v, d
                        eta = v.sum() / d.sum()
                        selected_rows.append(dict(dimension=dimension, primary=MODELS[primary],
                            mode=mode, rule=rule, target_size=size, actual_size=len(members),
                            selected_models='|'.join(MODELS[m] for m in members),
                            proposed_model=MODELS[chosen], added=addition,
                            probing_n=len(fi), test_n=len(ti), quality=v.mean(), mean_cost_usd=d.mean(),
                            efficiency=eta, baseline_efficiency=base_eta,
                            relative_efficiency_gain_pct=100 * (eta / base_eta - 1),
                            best_same_size_test_efficiency=oracle[size][0] if not stop else None,
                            best_same_size_test_models='|'.join(MODELS[m] for m in oracle[size][1]) if not stop else None,
                            oracle_gap_pct=100 * (oracle[size][0] - eta) / base_eta if not stop else None))
        print('Completed:', dimension, flush=True)
    selections = pd.DataFrame(selected_rows)
    selections.to_csv(args.output / 'portfolio_selections.csv', index=False)
    pd.DataFrame(candidate_rows).to_csv(args.output / 'portfolio_candidate_scores.csv', index=False)
    rng = np.random.default_rng(20260923)
    weights = np.zeros((args.bootstrap, n))
    for dim in dims:
        ix = np.flatnonzero(test_meta.dimension.to_numpy() == dim)
        weights[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=args.bootstrap)
    base_v, base_d = baseline_v.mean(axis=1), baseline_d.mean(axis=1)
    base_eta = base_v.sum() / base_d.sum()
    # Exhaustive per-context options plus fractional optimization find the true
    # aggregate ratio upper bound in the same dimension/primary policy class.
    oracle_eta = {}
    for size in [2, 3]:
        lam = base_eta
        while True:
            sv, sd = 0., 0.
            for target, primary, vals, ds, members in oracle_options[size]:
                index = np.argmax(vals.sum(axis=1) - lam * ds.sum(axis=1))
                sv += vals[index].sum()
                sd += ds[index].sum()
            new_lam = sv / sd
            if abs(new_lam - lam) < 1e-10:
                break
            lam = new_lam
        oracle_eta[size] = new_lam
    base_boot = (weights @ base_v) / (weights @ base_d)
    aggregate, bootstrap_gains, task_frames = [], {}, []
    for (mode, rule, size), (v, d) in predicted.items():
        avg_v, avg_d = v.mean(axis=1), d.mean(axis=1)
        eta = avg_v.sum() / avg_d.sum()
        boot_eta = (weights @ avg_v) / (weights @ avg_d)
        gain = 100 * (eta / base_eta - 1)
        boot_gain = 100 * (boot_eta / base_boot - 1)
        bootstrap_gains[(mode, rule, size)] = boot_gain
        lo, hi = np.quantile(boot_gain, [.025, .975])
        contexts = selections.query('mode == @mode and rule == @rule and target_size == @size')
        aggregate.append(dict(mode=mode, rule=rule, size=size, test_tasks=n,
            quality=avg_v.mean(), mean_cost_usd=avg_d.mean(), efficiency=eta,
            baseline_quality=base_v.mean(), baseline_cost_usd=base_d.mean(),
            relative_efficiency_gain_pct=gain, ci_low_pct=lo, ci_high_pct=hi,
            mean_selected_size=np.average(contexts.actual_size, weights=contexts.test_n),
            diagnostic_oracle_relative_gain_pct=100 * (oracle_eta[size] / base_eta - 1) if mode == 'forced_size' else None,
            diagnostic_oracle_gap_pp=100 * (oracle_eta[size] - eta) / base_eta if mode == 'forced_size' else None))
        task_frames.append(pd.DataFrame(dict(task_id=test_meta.index, dimension=test_meta.dimension.to_numpy(),
            mode=mode, rule=rule, size=size, value=avg_v, cost_usd=avg_d,
            baseline_value=base_v, baseline_cost_usd=base_d)))
        for primary, name in enumerate(MODELS):
            peta = v[:, primary].sum() / d[:, primary].sum()
            pbeta = baseline_v[:, primary].sum() / baseline_d[:, primary].sum()
            diagnostics.append(dict(mode=mode, rule=rule, size=size, primary=name,
                quality=v[:, primary].mean(), mean_cost_usd=d[:, primary].mean(),
                relative_efficiency_gain_pct=100 * (peta / pbeta - 1)))
    aggregate = pd.DataFrame(aggregate)
    aggregate.to_csv(args.output / 'portfolio_aggregate.csv', index=False)
    pd.concat(task_frames).to_csv(args.output / 'portfolio_task_outcomes.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(args.output / 'portfolio_by_primary.csv', index=False)
    gaps = []
    for mode in ['forced_size', 'positive_stop']:
        for size in [2, 3]:
            for a, b in [('residual', 'independence'), ('full_conditional', 'residual')]:
                boot_gap = bootstrap_gains[(mode, a, size)] - bootstrap_gains[(mode, b, size)]
                av = aggregate.query('mode == @mode and rule == @a and size == @size').iloc[0]
                bv = aggregate.query('mode == @mode and rule == @b and size == @size').iloc[0]
                lo, hi = np.quantile(boot_gap, [.025, .975])
                gaps.append(dict(mode=mode, size=size, comparison=f'{a}_minus_{b}',
                    gain_gap_pp=av.relative_efficiency_gain_pct - bv.relative_efficiency_gain_pct,
                    ci_low_pp=lo, ci_high_pp=hi))
    pd.DataFrame(gaps).to_csv(args.output / 'portfolio_paired_gaps.csv', index=False)
    zeros = ((raw['cost_usd'] == 0) & (raw['score'] == 0)).sum()
    summary = {
        'source': args.data, 'probing_tasks': int(fit.sum()), 'test_tasks': n,
        'models': MODELS, 'fallback': FALLBACK, 'dimensions': dims, 'threshold': 1,
        'contexts_per_size': len(dims) * len(MODELS), 'bootstrap_resamples': args.bootstrap,
        'zero_cost_zero_score_rows_retained': int(zeros),
        'execution': 'All selected candidates are invoked and charged; accept any score >= 1; otherwise invoke Opus and retain its released score. Oracle verifier with no deployment overhead.',
        'selection': 'Each rule greedily adds a candidate on probing only, starting from the same fixed primary. Forced sizes 2/3; a positive-net stop sensitivity is also exported.',
        'simplified_score': 'r*(1-E[V_F|old failure]+eta_old*E[C_F|old failure])-eta_old*E[C_m]. Independence and residual rules differ ONLY in r=(1-p_old)*p_m versus P(old failure and m success).',
        'full_score': 'E[V_new-V_old]-eta_old*E[D_new-D_old], using paired probing outcomes. All economic scores use the same old-eta incremental net-value criterion; no candidate-specific denominator.',
        'aggregation': 'Within each test task average value and cost equally over the 7 fixed-primary policies, then aggregate over all 2919 tasks and take total value / total cost. Gain is relative to the corresponding singleton-primary+fallback mixture. This describes a uniform mixture of primary policies, not an optimized primary selector.',
        'uncertainty': '95% percentile paired test-task bootstrap, stratified by 9 disjoint dimensions, preserving per-dimension test sizes. Each task shares weights across all primaries, sizes and rules. Probing choices fixed; selection-estimation uncertainty not included.',
        'oracle': 'Per-dimension, fixed-primary exhaustive same-size test-best portfolio is diagnostic only. Never used for selection. Context-level oracle gap is descriptive; overlapping contexts are not independent observations.',
        'oracle_aggregation': 'Aggregate oracle enumerates all same-size portfolios per dimension/primary and optimizes the aggregate total-value/total-cost ratio by fractional programming. This is a test-label diagnostic upper bound in the same policy class; comparisons apply only to forced_size. Context rows separately report context-optimal test efficiency.',
        'caveats': ['Replay only; no real API execution and no measured verifier/router/aggregation overhead.',
                    'Forced addition may decrease efficiency; positive-stop sensitivity separates forced size from deployable stop decisions.',
                    'Source zero cost / zero score rows are retained unchanged; free failures can affect observed costs.',
                    'At size 3 each rule builds on its own selected size-2 portfolio; differences include the accumulated first-step choice.',
                    'Standalone stop sensitivity uses exact probing net benefit to vet its quality-ranked addition.'],
        'aggregate': json.loads(aggregate.to_json(orient='records')), 'paired_gaps': gaps,
    }
    (args.output / 'portfolio_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(aggregate.to_string(index=False), flush=True)
    print(pd.DataFrame(gaps).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

