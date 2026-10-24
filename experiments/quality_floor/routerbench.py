#!/usr/bin/env python3
"""RouterBench price routes and certification from Appendix C.2.
The original policy fitting, split, price path and certification calculations are retained.
Offline replay only. Obtain RouterBench separately and set ROUTERBENCH_PKL.
"""
import os, json, time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from certification import primitives, phi, ratio_opt, cc_lp, state_envelope, price_path, pvalues, fixed_sequence
T0 = time.time()
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PKL = os.environ.get('ROUTERBENCH_PKL', os.path.join(ROOT, 'data', 'routerbench', 'routerbench_0shot.pkl'))
SEED = 20260926
NBOOT = int(os.environ.get('NBOOT', 1000))
N_HALV = int(os.environ.get('RB_NHALV', 200))
REF = 'gpt-4-1106-preview'
EPS_LIST = [0.0, 0.005, 0.01]
ALPHA_CERT = 0.05
TOL_MASS = 1e-8
REL_TOL = 1e-12
MIX = 10
CERT_METHODS = ['hoeffding', 'emp_bernstein']
POL_KEYS = ['unconstrained'] + [f'eps={e:g}' for e in EPS_LIST]
SMOKE = 0

def log(*a):
    print(f'[{time.time()-T0:7.1f}s]', *a, flush=True)

def ci(v):
    v = np.asarray(v, float); w = v[1:][np.isfinite(v[1:])]
    if len(w) == 0:
        return [float(v[0]), None, None]
    return [float(v[0]), float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))]

def family(name):   # verbatim from the paper's experiments/router_selection_recheck.py
    if name.startswith("mmlu-"):
        return "mmlu"
    if name in {"hellaswag", "winogrande", "arc-challenge", "grade-school-math", "mbpp"}:
        return name
    return "mt-bench" if name.startswith("mtbench") else "rag-other"

def paper_split(names, reference_score, seed):   # verbatim logic of router_selection_recheck.split
    strata = np.array([f"{name}|{int(value >= .5)}" for name, value in zip(names, reference_score)])
    fit = np.zeros(len(strata), bool)
    generator = np.random.default_rng(seed)
    for name in sorted(set(strata)):
        ids = np.flatnonzero(strata == name)
        fit[generator.permutation(ids)[:int(round(.7 * len(ids)))]] = True
    return fit

def build_menu(Y, Cm, thetas):
    """direct:m for all 11 models, then cascade:P@theta for the 10 non-reference primaries (reference on H = 0)."""
    yB, cB = Y[:, iB], Cm[:, iB]
    names, Vc, Dc, callsB, casc, prim, th_of, run = [], [], [], [], [], [], [], []
    for m in MODELS:
        j = col[m]
        names.append('direct:' + m); Vc.append(Y[:, j]); Dc.append(Cm[:, j]); callsB.append(m == REF); casc.append(False)
        prim.append(j); th_of.append(np.nan); run.append(np.ones(len(Y)) if m == REF else np.zeros(len(Y)))
    for th in thetas:
        for m in MODELS:
            if m == REF:
                continue
            j = col[m]; H = (Y[:, j] >= th - 1e-12).astype(float)
            names.append(f'cascade:{m}@{th:g}'); Vc.append(H * Y[:, j] + (1 - H) * yB); Dc.append(Cm[:, j] + (1 - H) * cB)
            callsB.append(True); casc.append(True); prim.append(j); th_of.append(th); run.append(1 - H)
    return dict(names=names, V=np.column_stack(Vc), D=np.column_stack(Dc), callsB=np.array(callsB), casc=np.array(casc),
                prim=np.array(prim), theta=np.array(th_of), RUN=np.column_stack(run), U=len(names),
                A_REF=names.index('direct:' + REF), thetas=list(thetas))

def strat_weights(strata, nboot, seed):
    rng = np.random.default_rng(seed)
    n = len(strata); W = np.zeros((nboot, n))
    for s in sorted(set(strata)):
        ids = np.flatnonzero(strata == s)
        W[:, ids] = rng.multinomial(len(ids), np.full(len(ids), 1.0 / len(ids)), size=nboot)
    return np.vstack([np.ones((1, n)), W])

def make_ctx(menu, rows, W):
    yB, cB = yB_all[rows], cB_all[rows]
    return dict(V=menu['V'][rows], D=menu['D'][rows], yB=yB, cB=cB, W=W, n=len(rows), SB=W @ yB, CB=W @ cB,
                RUN=menu['RUN'][rows], callsB=menu['callsB'])

def evaluate(G, ctx):
    """G: n x U assignment probabilities, or an n-vector of action indices.  Row 0 of every array = point estimate."""
    n = ctx['n']
    if G.ndim == 1:
        ar = np.arange(n); Vt = ctx['V'][ar, G]; Dt = ctx['D'][ar, G]; runs = ctx['RUN'][ar, G]; toB = ctx['callsB'][G].astype(float)
    else:
        Vt = (G * ctx['V']).sum(1); Dt = (G * ctx['D']).sum(1); runs = (G * ctx['RUN']).sum(1); toB = G[:, ctx['callsB']].sum(1)
    W = ctx['W']; m = W.sum(1)
    SV, SD = W @ Vt, W @ Dt
    return dict(r=(SV / SD) / (ctx['SB'] / ctx['CB']), EV=SV / m, ED=SD / m, dV=(SV - ctx['SB']) / m,
                share_to_ref_calling=float(toB.mean()), share_ref_runs=float(runs.mean()))

def summ(st, eps=EPS_LIST):
    out = dict(r=ci(st['r']), cost_mUSD=ci(1000 * st['ED']), dV=ci(st['dV']), EV=ci(st['EV']),
               share_to_ref_calling=st['share_to_ref_calling'], share_ref_runs=st['share_ref_runs'])
    if len(st['dV']) > 1:
        out['compliance'] = {f'{e:g}': float(np.mean(st['dV'][1:] >= -e - 1e-12)) for e in eps}
    return out

def fmt(s):
    return (f"r={s['r'][0]:.3f} [{s['r'][1]:.3f},{s['r'][2]:.3f}] cost={s['cost_mUSD'][0]:.4f} "
            f"dV={s['dV'][0]:+.4f} [{s['dV'][1]:+.4f},{s['dV'][2]:+.4f}] toB={s['share_to_ref_calling']:.3f} runsB={s['share_ref_runs']:.3f}")





def pred_summary(g, pi, v, c):
    return float(pi @ (g * v).sum(1)), float(pi @ (g * c).sum(1))

def state_index(labels):
    names = sorted(set(labels)); m = {s: i for i, s in enumerate(names)}
    return np.array([m[s] for s in labels]), names

def fit_scheme(menu, sfull, X, rows=None, eps_list=EPS_LIST):
    rows = FIT if rows is None else rows
    s = sfull[rows]
    pi, v, c = primitives(s, X, menu['V'][rows], menu['D'][rows])
    A = menu['A_REF']
    vbar = float(pi @ v[:, A]); cbar = float(pi @ c[:, A])
    eta, g0 = ratio_opt(pi, v, c)
    _, prun, _ = primitives(s, X, menu['RUN'][rows], menu['RUN'][rows])
    fam = dict(pi=pi, v=v, c=c, prun=prun, vbar=vbar, cbar=cbar, eta_B=vbar / cbar, eta_star=eta, X=X, pol={'unconstrained': g0},
               info={'unconstrained': dict(eta=eta, kappa=0.0, price=eta, binding=False, n_rand=0)})
    for e in eps_list:
        lp = cc_lp(pi, v, c, vbar - e)
        fam['pol'][f'eps={e:g}'] = lp['g']
        fam['info'][f'eps={e:g}'] = {k: lp[k] for k in ('eta', 'kappa', 'price', 'binding', 'n_rand')}
    return fam

def fitted_stats(fam, g, menu):
    ev, ed = pred_summary(g, fam['pi'], fam['v'], fam['c'])
    return dict(r=(ev / ed) / fam['eta_B'], dV=ev - fam['vbar'], cost_mUSD=1000 * ed,
                share_to_ref_calling=float(fam['pi'] @ g[:, menu['callsB']].sum(1)),
                share_ref_runs=float(fam['pi'] @ (g * fam['prun']).sum(1)))

def top_actions(G, menu, k=4):
    mass = G.sum(0) / len(G)
    return {menu['names'][u]: round(float(mass[u]), 4) for u in np.argsort(-mass)[:k] if mass[u] > 0}

def eval_scheme(fam, sfull, menu, ctx, keys=POL_KEYS):
    out = {}
    for k in keys:
        G = fam['pol'][k][sfull[TE]]
        st = evaluate(G, ctx)
        out[k] = dict(heldout=summ(st), fit=fam['info'][k], fitted=fitted_stats(fam, fam['pol'][k], menu),
                      heldout_top_actions=top_actions(G, menu), _st=st)
    return out





def certify(menu, sfull, X, A_rows, B_rows, eps=0.0, mix=True):
    V, D = menu['V'], menu['D']; U_ = menu['U']; EYE = np.eye(U_)
    pi, v, c = primitives(sfull[A_rows], X, V[A_rows], D[A_rows])
    eta, _ = ratio_opt(pi, v, c)
    vbar = float(pi @ v[:, menu['A_REF']]); cbar = float(pi @ c[:, menu['A_REF']])
    path = price_path(pi, v, c, eta); P = path['P']; K = len(P)
    ratio = path['EV'] / path['ED']
    checks = dict(EV_nonincreasing=bool(np.all(np.diff(path['EV']) <= 1e-12)), ED_nonincreasing=bool(np.all(np.diff(path['ED']) <= 1e-15)),
                  ratio_nondecreasing=bool(np.all(np.diff(ratio) >= -1e-9 * ratio[:-1])), gK_attains_eta_star=bool(abs(ratio[-1] - eta) <= 1e-7 * eta))
    Zk = V[B_rows[None, :], P[:, sfull[B_rows]]] - yB_all[B_rows][None, :]
    seqs = {'': (np.arange(K), np.zeros(K), Zk)}
    if mix:
        kk = np.concatenate([np.repeat(np.arange(K - 1), MIX), [K - 1]]); tt = np.concatenate([np.tile(np.arange(MIX) / MIX, K - 1), [0.0]])
        k2 = np.minimum(kk + 1, K - 1)
        seqs[f'+mix{MIX}'] = (kk, tt, (1 - tt)[:, None] * Zk[kk] + tt[:, None] * Zk[k2])
    sel = {}
    for var, (ks, ts, Zs) in seqs.items():
        pv = pvalues(Zs, eps)
        for m in CERT_METHODS:
            j = fixed_sequence(pv[m])
            if j > 0:
                k, t = int(ks[j - 1]), float(ts[j - 1]); kn = min(k + 1, K - 1)
                G = (1 - t) * EYE[P[k]] + t * EYE[P[kn]]
                ev = (1 - t) * path['EV'][k] + t * path['EV'][kn]; ed = (1 - t) * path['ED'][k] + t * path['ED'][kn]
                qB = float(pv['Qhat'][j - 1]); lam = float((1 - t) * path['lam'][k] + t * path['lam'][kn])
            else:
                k, t = -1, 0.0; G = EYE[np.full(X, menu['A_REF'])]; ev, ed = vbar, cbar; qB = 0.0; lam = None
            sel[m + var] = dict(khat=j, k=k + 1, t=t, K=K, G=G, Qhat_B=qB, lam=lam, fitA_dV=float(ev - vbar), fitA_r=float(ev / ed / (vbar / cbar)),
                                p_at_khat=float(pv[m][j - 1]) if j > 0 else None)
    lp = cc_lp(pi, v, c, vbar - eps)
    sel['uncertified_cclp_refit_on_A'] = dict(G=lp['g'], khat=None, fitA_dV=float(pred_summary(lp['g'], pi, v, c)[0] - vbar))
    eta_B_A = vbar / cbar
    return sel, dict(K=K, eta_star_A=eta, eta_B_A=eta_B_A, checks=checks)

def halves(seed):
    """Split the fitting rows within each eval_name: A = first floor(n/2) of a random permutation, B = the rest."""
    rng = np.random.default_rng(seed)
    A, B = [], []
    for s in sorted(set(NAMES[FIT])):
        idx = FIT[NAMES[FIT] == s]
        perm = rng.permutation(idx); h = len(idx) // 2
        A.append(perm[:h]); B.append(perm[h:])
    return np.sort(np.concatenate(A)), np.sort(np.concatenate(B))

def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    return o

frame = pd.read_pickle(PKL)
MODELS = [m for m in frame.columns if f'{m}|total_cost' in frame.columns]
frame = frame[['prompt', 'eval_name'] + MODELS + [f'{m}|total_cost' for m in MODELS]].copy()   # drop model responses (memory)
if SMOKE:
    frame = frame.sample(n=SMOKE, random_state=0).reset_index(drop=True); NBOOT = 50; N_HALV = 3
col = {m: j for j, m in enumerate(MODELS)}
S = frame[MODELS].to_numpy(float)
C = frame[[f'{m}|total_cost' for m in MODELS]].to_numpy(float)
NAMES = frame.eval_name.astype(str).to_numpy()
FAMS = np.array([family(n) for n in NAMES])
N = len(frame)
assert (SMOKE or N == 36497) and len(MODELS) == 11 and REF in MODELS
assert np.isfinite(S).all() and S.min() >= 0 and S.max() <= 1, 'scores must lie in [0,1]'
assert np.isfinite(C).all() and (C > 0).all(), 'costs must be positive and finite'
iB = col[REF]
yB_all, cB_all = S[:, iB], C[:, iB]
FIT_MASK = paper_split(NAMES, yB_all, SEED)
FIT = np.flatnonzero(FIT_MASK); TE = np.flatnonzero(~FIT_MASK)
assert SMOKE or (len(FIT) == 25542 and len(TE) == 10955)
n_te = len(TE)
MAIN = build_menu(S, C, [1.0])
W_TE = strat_weights(NAMES[TE], NBOOT, SEED)
ONES_TE = np.ones((1, n_te))
CTX = {'theta1': make_ctx(MAIN, TE, W_TE)}
SCHEMES = {}
for name, labels in [('family', FAMS), ('eval_name', NAMES)]:
    index, categories = state_index(labels)
    SCHEMES[name] = dict(full=index, names=categories)
RES = dict(settings=dict(seed=SEED, nboot=NBOOT, n_halvings=N_HALV, alpha=ALPHA_CERT, reference=REF,
                        split='eval_name x reference-score stratum, 70/30', fitting_inputs=len(FIT), test_inputs=len(TE)))
RES['fitted_policies'] = {}
for name, scheme in SCHEMES.items():
    fitted = fit_scheme(MAIN, scheme['full'], len(scheme['names']))
    RES['fitted_policies'][name] = eval_scheme(fitted, scheme['full'], MAIN, CTX['theta1'])

CTX_PT = make_ctx(MAIN, TE, ONES_TE)
CERT = dict(procedure='per halving: fit pi, v, c on A (stratified half of the fitting split), enumerate the J_lambda price path on [0, eta*_A], '
                      'test H_k: Q(g_k) < 0 on B in path order (Hoeffding / empirical Bernstein, alpha = 0.05), select g_khat (reference if none); '
                      'uncertified comparator = CC-LP eps=0 floor refitted on A; all evaluated on the 10,955 held-out prompts',
            n_halvings=N_HALV, schemes={})
t1 = time.time()
for sname in ['family', 'eval_name']:
    sch = SCHEMES[sname]; X = len(sch['names']); sfull = sch['full']
    recs = {m: [] for m in CERT_METHODS + [m + f'+mix{MIX}' for m in CERT_METHODS] + ['uncertified_cclp_refit_on_A']}
    Ks, checks_ok, main_rows = [], 0, {}
    for h in range(N_HALV):
        A_rows, B_rows = halves(SEED + h)
        sel, info = certify(MAIN, sfull, X, A_rows, B_rows, 0.0, mix=True)
        Ks.append(info['K']); checks_ok += int(all(info['checks'].values()))
        for m, s_ in sel.items():
            G = s_['G'][sfull[TE]]
            st = evaluate(G, CTX['theta1'] if h == 0 else CTX_PT)
            recs[m].append(dict(r=float(st['r'][0]), dV=float(st['dV'][0]), cost_mUSD=float(1000 * st['ED'][0]),
                                fallback=bool(s_['khat'] == 0) if s_['khat'] is not None else False, fitA_dV=s_['fitA_dV'],
                                share_ref_runs=st['share_ref_runs'], share_to_ref_calling=st['share_to_ref_calling']))
            if h == 0:
                main_rows[m] = dict(heldout=summ(st, [0.0]), khat=s_['khat'], K=info['K'], Qhat_B=s_.get('Qhat_B'), lam=s_.get('lam'),
                                    p_at_khat=s_.get('p_at_khat'), fitA_dV=s_['fitA_dV'], heldout_top_actions=top_actions(G, MAIN), _st=st)
    summ_h = {}
    for m, L in recs.items():
        r = np.array([d['r'] for d in L]); dv = np.array([d['dV'] for d in L]); cst = np.array([d['cost_mUSD'] for d in L])
        summ_h[m] = dict(median_heldout_r=float(np.median(r)), r_p5_p95=[float(np.percentile(r, 5)), float(np.percentile(r, 95))],
                         mean_heldout_dV=float(dv.mean()), dV_min_max=[float(dv.min()), float(dv.max())],
                         share_heldout_dV_ge_0=float(np.mean(dv >= -1e-12)), median_cost_mUSD=float(np.median(cst)),
                         share_fallback_to_reference=float(np.mean([d['fallback'] for d in L])),
                         median_share_ref_runs=float(np.median([d['share_ref_runs'] for d in L])))
        log(f'certified[{sname}] {m:28s} median r {summ_h[m]["median_heldout_r"]:.2f} [p5 {summ_h[m]["r_p5_p95"][0]:.2f}, p95 {summ_h[m]["r_p5_p95"][1]:.2f}] '
            f'mean dV {summ_h[m]["mean_heldout_dV"]:+.4f} share dV>=0 {summ_h[m]["share_heldout_dV_ge_0"]:.3f} fallback {summ_h[m]["share_fallback_to_reference"]:.3f}')
    for m, row in main_rows.items():
        log(f'certified[{sname}] halving 0 {m:28s} {fmt(row["heldout"])} comp0={row["heldout"]["compliance"]["0"]:.3f} khat={row["khat"]} K={row["K"]}')
    CERT['schemes'][sname] = dict(summary_over_halvings=summ_h, halving0=main_rows, K_mean=float(np.mean(Ks)), K_min=int(np.min(Ks)), K_max=int(np.max(Ks)),
                                  path_checks_pass=f'{checks_ok}/{N_HALV}', per_halving={m: L for m, L in recs.items()})
CERT['seconds'] = time.time() - t1
RES['certified_floor'] = CERT


output = os.environ.get('QUALITY_FLOOR_OUT', os.path.join(ROOT, 'runs', 'quality_floor', 'routerbench.json'))
os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
with open(output, 'w') as handle:
    json.dump(clean(RES), handle, indent=2)
log('Wrote', output)
