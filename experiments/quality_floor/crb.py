#!/usr/bin/env python3
"""Appendix C.2 CodeRouterBench certification: original i.i.d. half-split implementation.
Offline replay; no model calls. NSPLIT=200 and NBOOT=1000 reproduce the paper.
"""
import os, json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import beta as betad

from crb_core import *
OUT = {}
NSPLIT = int(os.environ.get('NSPLIT', NSPLIT))
NB_SPLIT = 3540
SOUND = np.array([u == A_REF or ANAMES[u].endswith('@1') for u in range(U)])
DIRECT = np.array([a.startswith('direct:') for a in ANAMES])
assert all((V_ALL[:, u] - yB_all >= -1e-12).all() for u in np.where(SOUND)[0])   # theta=1 replay is pointwise sound


def split_iid(rng):
    while True:
        perm = rng.permutation(PR); A = np.sort(perm[:NB_SPLIT]); B = np.sort(perm[NB_SPLIT:])
        if (np.bincount(sidx21[A], minlength=X21) > 0).all():
            return A, B


def uppers(lam_lo, eta):
    return np.append(lam_lo[1:], eta)


def lp_floor(path, lam_lo, eta, pi, v, c, eps):
    """Theorem (iii) fitted floor optimum via Prop A(a)(3): g_K if it meets the floor, else the mixture of the
    round-down g_k and g_{k+1} whose fitted value equals vbar - eps. Returns (router, price, w, k)."""
    vb, _ = pol_vd(REF_POL, pi, v, c)
    k = fitted_floor(path, pi, v, c, eps)
    up = uppers(lam_lo, eta)
    if k == len(path):
        return path[-1], float(eta), 1.0, k
    va, _ = pol_vd(path[k - 1], pi, v, c); vn, _ = pol_vd(path[k], pi, v, c)
    w = float((vb - eps - vn) / (va - vn))
    return (path[k - 1], path[k], w), float(up[k - 1]), w, k


def lp_floor_cc(pi, v, c, eps):
    """Charnes-Cooper LP of the fitted floor problem (independent check of Prop A(a)(3))."""
    X, Uu = v.shape; n = X * Uu
    vb, _ = pol_vd(REF_POL, pi, v, c)
    obj = -np.append((pi[:, None] * v).ravel(), 0.0)
    Aeq = [np.append((pi[:, None] * c).ravel(), 0.0)]; beq = [1.0]
    for x in range(X):
        row = np.zeros(n + 1); row[x * Uu:(x + 1) * Uu] = 1.0; row[-1] = -1.0; Aeq.append(row); beq.append(0.0)
    Aub = [np.append(-(pi[:, None] * v).ravel(), vb - eps)]
    r = linprog(obj, A_ub=np.array(Aub), b_ub=[0.0], A_eq=np.array(Aeq), b_eq=beq, bounds=[(0, None)] * (n + 1), method='highs')
    return float(-r.fun)


def refine2(path, lam_lo, eta, extra):
    """Path with mixtures of neighbours (grid 0.9..0.1 plus the exact fitted floor weights in `extra`, a dict
    {pair index k: [w, ...]}), in decreasing fitted value. Items: (router, certified price, label)."""
    up = uppers(lam_lo, eta); out = []
    for k in range(len(path)):
        out.append((path[k], float(up[k]), f'g{k+1}'))
        if k < len(path) - 1:
            ws = sorted(set(np.round(MIX_W, 6)) | set(np.round(extra.get(k, []), 6)), reverse=True)
            out.extend(((path[k], path[k + 1], float(w)), float(up[k]), f'g{k+1}|g{k+2}@{w:.3f}') for w in ws if 0 < w < 1)
    return out


def certify_items(items, rowsB, eps, method, alpha=ALPHA):
    k = 0
    for g, _, _ in items:
        v_, _ = realized(g, rowsB)
        if PFUN[method](v_ - yB_all[rowsB], eps) <= alpha:
            k += 1
        else:
            break
    return k


def cell_q(g, rows):
    if isinstance(g, tuple):
        ga, gb, w = g
        return w * cell_q(ga, rows) + (1 - w) * cell_q(gb, rows)
    s = sidx21[rows]; z = V_ALL[rows, g[s]] - yB_all[rows]
    return np.bincount(s, weights=z, minlength=X21) / np.maximum(np.bincount(s, minlength=X21), 1)


def certified_menus(vA, B, eps, allowed_actions, alpha=ALPHA):
    """Corollary: in each cell, candidate non-sound actions (fitted Delta_A >= -eps, decreasing fitted Delta_A) are
    tested in sequence at level alpha/|X| on the cell's B tasks; sound actions need no test."""
    s = sidx21[B]; allowed = np.zeros_like(vA, bool); allowed[:, SOUND & allowed_actions] = True; n_ok = 0
    for x in range(X21):
        dA = vA[x] - vA[x, A_REF]
        cand = [u for u in np.argsort(-dA, kind='stable') if allowed_actions[u] and not SOUND[u] and dA[u] >= -eps - 1e-12]
        r = B[s == x]
        for u in cand:
            if len(r) >= 2 and p_bernstein(V_ALL[r, u] - yB_all[r], eps) <= alpha / X21:
                allowed[x, u] = True; n_ok += 1
            else:
                break
    return allowed, n_ok


def ratio_router(pi_, v_, c_, allowed):
    vr = np.where(allowed, v_, -1e9)
    return argmax_policy(vr, c_, ratio_opt(pi_, vr, c_))


def same_router(g, h):
    if isinstance(g, tuple) != isinstance(h, tuple):
        return False
    if isinstance(g, tuple):
        return np.array_equal(g[0], h[0]) and np.array_equal(g[1], h[1]) and abs(g[2] - h[2]) < 1e-9
    return np.array_equal(g, h)


# ============================================================ Part 1: path on full probing, lambda_eps, strictness
pi, v, c = primitives(PR, sidx21, X21)
eta, path, lam_lo = price_path(pi, v, c)
VD = np.array([pol_vd(g, pi, v, c) for g in path]); vb_p, cb_p = pol_vd(REF_POL, pi, v, c); etaB_p = vb_p / cb_p
eff = VD[:, 0] / VD[:, 1]
p1 = dict(eta_star=eta, eta_B=etaB_p, K=len(path), bound=1 + X21 * (U - 1),
          strict=dict(EV_decreasing=bool(np.all(np.diff(VD[:, 0]) < 0)), ED_decreasing=bool(np.all(np.diff(VD[:, 1]) < 0)),
                      eta_increasing=bool(np.all(np.diff(eff) > 0)), all_distinct=len({g.tobytes() for g in path}) == len(path)),
          first_ratio=float(eff[0] / etaB_p), last_ratio=float(eff[-1] / etaB_p),
          states_changed_per_step=[int((path[k] != path[k - 1]).sum()) for k in range(1, len(path))], floor={})
for eps in EPS_LIST:
    g, lam, w, k = lp_floor(path, lam_lo, eta, pi, v, c, eps)
    ev, ed = realized(g, PR); fe = float(ev.mean() / ed.mean())
    kappa = fe / lam - 1.0
    p1['floor'][f'eps{eps:g}'] = dict(k_rounddown=k, w_on_gk=w, lambda_eps=lam, kappa=kappa, lambda_over_etaB=lam / etaB_p,
                                      fitted_ratio=fe / etaB_p, cc_lp_ratio=lp_floor_cc(pi, v, c, eps) / etaB_p,
                                      interval_k=[float(lam_lo[k - 1]), float(uppers(lam_lo, eta)[k - 1])],
                                      fitted_dV_gk=float(VD[k - 1, 0] - vb_p), fitted_dV_gk1=float(VD[min(k, len(path) - 1), 0] - vb_p),
                                      rounddown_fitted_ratio=float(eff[k - 1] / etaB_p))
OUT['part1'] = p1
log('Part 1', json.dumps({k_: p1[k_] for k_ in p1 if k_ != 'states_changed_per_step'}, default=float))


# ============================================================ Part 2: certification on unstratified i.i.d. splits
def evaluate(g, boot, lam=None, extra=None):
    e = eval_test(g, boot); qt = cell_q(g, TE)
    e.update(min_cell_q_test=float(qt.min()), n_cells_neg_test=int((qt < -1e-12).sum()))
    if lam is not None:
        e['lam'] = lam; e['lam_over_etaB'] = lam / etaB_p
    if extra:
        e.update(extra)
    return e


def fitted_dv(g, pi_, v_, c_):
    if isinstance(g, tuple):
        ga, gb, w = g
        return w * fitted_dv(ga, pi_, v_, c_) + (1 - w) * fitted_dv(gb, pi_, v_, c_)
    return pol_vd(g, pi_, v_, c_)[0] - pol_vd(REF_POL, pi_, v_, c_)[0]


def one_split(rng, boot):
    A, B = split_iid(rng)
    piA, vA, cA = primitives(A, sidx21, X21)
    etaA, pathA, loA = price_path(piA, vA, cA); upA = uppers(loA, etaA)
    extra = {}
    for eps in EPS_LIST:
        g_, lam_, w_, k_ = lp_floor(pathA, loA, etaA, piA, vA, cA, eps)
        if isinstance(g_, tuple):
            extra.setdefault(k_ - 1, []).append(w_)
    items = refine2(pathA, loA, etaA, extra)
    o = dict(K=len(pathA), rows={})
    for eps in EPS_LIST:
        glp, lamlp, wlp, kf = lp_floor(pathA, loA, etaA, piA, vA, cA, eps)
        grd = pathA[kf - 1]
        o['rows'][f'lpfloor_eps{eps:g}'] = evaluate(glp, boot, lamlp, dict(fitted_dV_A=float(fitted_dv(glp, piA, vA, cA))))
        o['rows'][f'rounddown_eps{eps:g}'] = evaluate(grd, boot, float(upA[kf - 1]))
        for m in ('hoeffding', 'bernstein', 'clt'):
            k, ps = certify(pathA, B, eps, m)
            gsel = pick(pathA, k)
            o['rows'][f'{m}_eps{eps:g}'] = evaluate(gsel, boot, float(upA[k - 1]) if k else None,
                                                    dict(k=k, same_as_rounddown=bool(k == kf), fitted_dV_A=float(fitted_dv(gsel, piA, vA, cA))))
            if m != 'clt':
                kb = bonferroni(ps)
                o['rows'][f'bonferroni_{m}_eps{eps:g}'] = evaluate(pick(pathA, kb), boot, float(upA[kb - 1]) if kb else None, dict(k=kb))
                kr = certify_items(items, B, eps, m)
                gr = REF_POL if kr == 0 else items[kr - 1][0]
                o['rows'][f'refined_{m}_eps{eps:g}'] = evaluate(gr, boot, items[kr - 1][1] if kr else None,
                                                                dict(k=kr, label=items[kr - 1][2] if kr else 'ref', fitted_dV_A=float(fitted_dv(gr, piA, vA, cA)),
                                                                     same_as_rounddown=bool(same_router(gr, grd)),
                                                                     same_as_lpfloor=bool(same_router(gr, glp))))
    # state-robust certification (Corollary): full X9 menu and direct-only (deployable) menu
    for tag, allowed_actions in (('x9', np.ones(U, bool)), ('direct', DIRECT)):
        so = ratio_router(piA, vA, cA, np.broadcast_to(SOUND & allowed_actions, vA.shape))
        o['rows'][f'soundonly_{tag}_eps0'] = evaluate(so, boot)
        for eps in EPS_LIST:
            allowed, n_ok = certified_menus(vA, B, eps, allowed_actions)
            gs = ratio_router(piA, vA, cA, allowed)
            o['rows'][f'staterobust_{tag}_eps{eps:g}'] = evaluate(gs, boot, None, dict(n_certified_nonsound=n_ok,
                                                                n_cells_nonsound_used=int((~SOUND[gs]).sum())))
    vD = np.where(DIRECT[None, :], vA, -1e9)
    etaD, pathD, loD = price_path(piA, vD, cA)
    itemsD = refine2(pathD, loD, etaD, {})
    for eps in EPS_LIST:
        kr = certify_items(itemsD, B, eps, 'bernstein')
        o['rows'][f'refined_bernstein_direct_eps{eps:g}'] = evaluate(REF_POL if kr == 0 else itemsD[kr - 1][0], boot)
    return o


main = one_split(np.random.default_rng(SEED), True)
for key, row in main['rows'].items():
    eps = float(key.split('eps')[1])
    row['compliance'] = float(np.mean(row.pop('boot_dV') >= -eps - 1e-12))
OUT['part2_main_split'] = main
log('Part 2 main split (unstratified, seed', SEED, ') K', main['K'])
for key, r in main['rows'].items():
    log(f"  {key:34s} r={r['r'][0]:6.2f} [{r['r'][1]:.2f},{r['r'][2]:.2f}] cost={r['cost_m']:.3f} dV={r['dV'][0]:+.4f} "
        f"[{r['dV'][1]:+.4f},{r['dV'][2]:+.4f}] compl={r['compliance']:.3f} mincell={r['min_cell_q_test']:+.3f} ncellneg={r['n_cells_neg_test']}"
        + (f" lam={r['lam']:.1f} ({r['lam_over_etaB']:.2f} etaB)" if 'lam' in r else '') + (f" {r.get('label', '')} k={r.get('k', '')}"))

rng = np.random.default_rng(SEED + 1)
agg = {}
for s in range(NSPLIT):
    o = one_split(rng, False)
    for key, r in o['rows'].items():
        agg.setdefault(key, []).append(r)
summ = {}
for key, L in agg.items():
    eps = float(key.split('eps')[1])
    rr = np.array([r['r'] for r in L]); dv = np.array([r['dV'] for r in L]); mc = np.array([r['min_cell_q_test'] for r in L])
    d = dict(share_dV_ge_minus_eps=float(np.mean(dv >= -eps - 1e-12)), r_median=float(np.median(rr)),
             r_p10_p90=[float(np.percentile(rr, 10)), float(np.percentile(rr, 90))], dV_mean=float(dv.mean()),
             share_all_cells_ge_minus_eps=float(np.mean(mc >= -eps - 1e-12)), min_cell_q_median=float(np.median(mc)),
             n_cells_neg_median=float(np.median([r['n_cells_neg_test'] for r in L])))
    if 'lam' in L[0] or any('lam' in r for r in L):
        lams = np.array([r['lam_over_etaB'] for r in L if 'lam' in r]); d['lam_over_etaB_median'] = float(np.median(lams))
        d['lam_over_etaB_p10_p90'] = [float(np.percentile(lams, 10)), float(np.percentile(lams, 90))]
    if 'fitted_dV_A' in L[0]:
        fm = np.array([r['fitted_dV_A'] for r in L]) + eps; d['fitted_margin_median'] = float(np.median(fm))
        d['fitted_margin_p10_p90'] = [float(np.percentile(fm, 10)), float(np.percentile(fm, 90))]
    for f in ('same_as_rounddown', 'same_as_lpfloor'):
        if f in L[0]:
            d[f'share_{f}'] = float(np.mean([r[f] for r in L]))
    if 'k' in L[0]:
        d['share_reference'] = float(np.mean([r['k'] == 0 for r in L]))
    summ[key] = d
# splits in which the round-down overshoots past the efficiency jump (ID Test ratio > 25)
rd = np.array([r['r'] for r in agg['rounddown_eps0']])
summ['rounddown_eps0']['share_ratio_above_25'] = float(np.mean(rd > 25))
OUT['part2_repeated_splits'] = summ
log('Part 2:', NSPLIT, 'unstratified splits')
for key, d in summ.items():
    log(f"  {key:34s} share dV>=-eps {d['share_dV_ge_minus_eps']:.3f} r med {d['r_median']:6.2f} [{d['r_p10_p90'][0]:.2f},{d['r_p10_p90'][1]:.2f}] "
        f"dV mean {d['dV_mean']:+.4f} allcells>=-eps {d['share_all_cells_ge_minus_eps']:.3f} mincell med {d['min_cell_q_median']:+.3f} "
        + ' '.join(f'{k_}={v_}' for k_, v_ in d.items() if k_.startswith('share_same') or k_.startswith('lam_over') or k_ in ('share_reference', 'share_ratio_above_25')))


OUT['settings'] = dict(seed=SEED, nsplit=NSPLIT, nboot=NBOOT, alpha=ALPHA,
                       split='unstratified probing halves, 3540/3540', reference=BASE)
output = os.environ.get('QUALITY_FLOOR_OUT', os.path.join(ROOT, 'runs', 'quality_floor', 'crb.json'))
os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
with open(output, 'w') as handle:
    json.dump(OUT, handle, indent=2, default=float)
log('Wrote', output)
