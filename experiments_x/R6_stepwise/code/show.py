import json, sys
R = json.load(open(sys.argv[1])); basis = sys.argv[2] if len(sys.argv) > 2 else 'flops'
print('pair', R['pair'], R['small'], R['large'], 'N', R['n_tasks'])
print('whole-sample (score, flops_rel, sec_rel):')
for k, v in R['whole_sample'].items(): print(f"  {k:55s} {v['score']:.3f} {v['flops_rel']:.3f} {v['sec_rel']:.3f}")
for f, fam in R['whole_sample_families'].items():
    print(' family', f)
    for k, v in fam.items(): print(f"    {k:>22s} {v['score']:.3f} {v['flops_rel']:.3f} {v['sec_rel']:.3f} n_ok={v['n_ok']}")
for fold, FR in R['folds'].items():
    print('==', fold, basis)
    for k, v in FR[basis].items():
        if v.get('feasible') is False: print(f"  {k:48s} infeasible"); continue
        if 'r' not in v: print(' ', k, v); continue
        extra = f" param={v.get('param')} floor_rate={v.get('floor_rate')}" if v['kind'] == 'selected' else ''
        print(f"  {k:48s} r={v['r'][0]:.3f} [{v['r'][1]:.3f},{v['r'][2]:.3f}] score={v['score'][0]:.3f} dV={v['dV'][0]:+.3f} [{v['dV'][1]:+.3f},{v['dV'][2]:+.3f}] cost={v['cost_ratio'][0]:.3f}{extra}")
if 'live_vs_replay' in R:
    print('live vs replay')
    for k, v in R['live_vs_replay'].items(): print(' ', round(float(k), 3), {a: round(b, 3) for a, b in v.items()})

if 'c2c_transfer_summary' in R:
    print('C2C transfer summary', json.dumps(R['c2c_transfer_summary'], indent=0))
print('complementarity', json.dumps(R.get('complementarity', {}))[:900])
print('max-value oracles', R.get('max_value_oracles'))
for b in ('flops', 'sec'):
    c = R['chord_dominance_whole_sample'][b]; print(b, 'above chord', c['n_above_chord'], '/', c['n_total'])
    for k, v in sorted(c['rows'].items(), key=lambda kv: -kv[1]['gap'])[:6]: print('   ', k, {a: round(x, 3) for a, x in v.items()})
for k, v in R['local_comparisons'].items():
    if 'fold0' in k and ('lam=0.2' in k or 'lam=1.0' in k or 'lam=0.5' in k): print(k, 'dec', v['n_decided'], 'pos', v['n_decided_positive'], 'vc', v['agree_value_cost'], 'const', v['agree_best_constant'])

