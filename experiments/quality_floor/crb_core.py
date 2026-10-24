"""CodeRouterBench price paths and fixed-sequence tests used in Appendix C.2.
The 2026-10-24 experiment uses crb.py with unstratified fitting/certification halves.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy.stats import norm
sys.dont_write_bytecode = True

T0 = time.time()
SEED = 20260925
NBOOT = int(os.environ.get('NBOOT', 1000))
NSPLIT = int(os.environ.get('NSPLIT', 200))
ALPHA = 0.05
EPS_LIST = [0.0, 0.005, 0.01]
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.environ.get('ACROUTER_DATA', os.path.join(ROOT, 'data', 'agent-as-a-router', 'data'))
BASE = 'claude-opus-4-6'
PRIMARIES = ['Qwen3-Max', 'kimi-k2.5', 'qwen3.5-plus', 'glm-5', 'gpt-5.4', 'claude-sonnet-4-6', 'MiniMax-M2.7']
THETAS = [1.0, 0.5]


def log(*a):
    print(f'[{time.time()-T0:7.1f}s]', *a, flush=True)


def ci(v):
    v = np.asarray(v, float); w = v[1:][np.isfinite(v[1:])]
    return [float(v[0]), float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))]


# ------------------------------------------------------------ data and the X9 22-action menu (as in X9 / R7_policy)
df = pd.read_csv(f'{DATA}/coderouterbench/id_results_long.csv')
assert df.shape[0] == 79992
S = df.pivot(index='task_id', columns='model', values='score')
C = df.pivot(index='task_id', columns='model', values='cost_usd')
meta = df.drop_duplicates('task_id').set_index('task_id').loc[S.index, ['split', 'dimension']].copy()
meta['bench'] = meta.index.str.split('_').str[1]
meta['cell21'] = meta.dimension + '|' + meta.bench
MODELS = list(S.columns); col = {m: j for j, m in enumerate(MODELS)}
Sv, Cv = S.to_numpy(), C.to_numpy()
prob = (meta.split == 'probing').to_numpy(); idt = (meta.split == 'id_test').to_numpy()
assert prob.sum() == 7080 and idt.sum() == 2919


def build_menu(Y, Cm):
    yB, cB = Y[:, col[BASE]], Cm[:, col[BASE]]
    names, Vc, Dc = [], [], []
    for m in MODELS:
        names.append('direct:' + m); Vc.append(Y[:, col[m]]); Dc.append(Cm[:, col[m]])
    for m in PRIMARIES:
        for th in THETAS:
            H = (Y[:, col[m]] >= th - 1e-12).astype(float)
            names.append(f'attempt:{m}@{th:g}'); Vc.append(H * Y[:, col[m]] + (1 - H) * yB); Dc.append(Cm[:, col[m]] + (1 - H) * cB)
    return names, np.column_stack(Vc), np.column_stack(Dc)


ANAMES, V_ALL, D_ALL = build_menu(Sv, Cv)
U = len(ANAMES); assert U == 22
A_REF = ANAMES.index('direct:' + BASE)
yB_all, cB_all = Sv[:, col[BASE]], Cv[:, col[BASE]]
PR = np.where(prob)[0]; TE = np.where(idt)[0]
names21 = sorted(meta.cell21.unique()); sidx21 = meta.cell21.map({c: i for i, c in enumerate(names21)}).to_numpy()
X21 = len(names21); assert X21 == 21


def primitives(rows, sidx, X):
    s = sidx[rows]
    nx = np.bincount(s, minlength=X).astype(float)
    assert (nx > 0).all()
    v = np.column_stack([np.bincount(s, weights=V_ALL[rows, u], minlength=X) for u in range(U)]) / nx[:, None]
    c = np.column_stack([np.bincount(s, weights=D_ALL[rows, u], minlength=X) for u in range(U)]) / nx[:, None]
    return nx / nx.sum(), v, c


def argmax_policy(v, c, lam):
    return np.argmax(v - lam * c, axis=1)          # deterministic tie-break: lowest action index


def pol_vd(g, pi, v, c):
    ix = np.arange(len(g))
    return float(pi @ v[ix, g]), float(pi @ c[ix, g])


def ratio_opt(pi, v, c):
    """Dinkelbach from the fitted reference efficiency; exact after finitely many steps."""
    vb, cb = pol_vd(np.full(len(pi), A_REF), pi, v, c)
    lam = vb / cb
    for _ in range(200):
        g = argmax_policy(v, c, lam); ev, ed = pol_vd(g, pi, v, c); new = ev / ed
        if new <= lam * (1 + 1e-13):
            return lam
        lam = new
    raise RuntimeError('Dinkelbach did not converge')


def price_path(pi, v, c):
    """Maximizers of the fitted J_lambda on the breakpoint intervals of [0, eta*], in increasing lambda."""
    eta = ratio_opt(pi, v, c)
    bps = []
    for x in range(len(pi)):
        for u in range(U):
            for w in range(u + 1, U):
                dc = c[x, u] - c[x, w]
                if abs(dc) > 1e-15:
                    lam = (v[x, u] - v[x, w]) / dc
                    if 0 < lam < eta:
                        bps.append(lam)
    grid = np.concatenate([[0.0], np.unique(bps), [eta]])
    path, lam_lo = [], []
    for lo, hi in zip(grid[:-1], grid[1:]):
        g = argmax_policy(v, c, 0.5 * (lo + hi))
        if not path or not np.array_equal(g, path[-1]):
            path.append(g); lam_lo.append(lo)
    return eta, path, np.array(lam_lo)


def envelope_breakpoints(v_x, c_x, lo, hi):
    """Number of breakpoints of max_u [v - lam c] in the open interval (lo, hi)."""
    cand = []
    for u in range(U):
        for w in range(u + 1, U):
            dc = c_x[u] - c_x[w]
            if abs(dc) > 1e-15:
                lam = (v_x[u] - v_x[w]) / dc
                if lo < lam < hi:
                    cand.append(lam)
    cand = np.unique(cand)
    n = 0
    for lam in cand:
        d = 1e-9 * max(1.0, lam)
        if np.argmax(v_x - (lam - d) * c_x) != np.argmax(v_x - (lam + d) * c_x):
            n += 1
    return n


# ------------------------------------------------------------ p-values (valid under H0: Q(g) < -eps, Z = V - Y_B in [-1, 1])
def p_hoeffding(z, eps):
    n = len(z); t = max(z.mean() + eps, 0.0)
    return float(np.exp(-n * t * t / 2.0))


def p_bernstein(z, eps):
    """Maurer & Pontil (2009, Thm 4) applied to W = (1 - Z)/2 in [0, 1]:
    with prob >= 1 - d, E Z >= mean Z - 2 sqrt(2 Var_n(W) ln(2/d) / n) - 14 ln(2/d) / (3 (n - 1)).
    p = smallest d whose lower bound reaches -eps (1 if none)."""
    n = len(z); Cq = z.mean() + eps
    if Cq <= 0:
        return 1.0
    w = (1.0 - z) / 2.0; var = w.var(ddof=1)
    A = 14.0 / (3.0 * (n - 1)); Bq = 2.0 * np.sqrt(2.0 * var / n)
    s = (-Bq + np.sqrt(Bq * Bq + 4.0 * A * Cq)) / (2.0 * A)
    return float(min(1.0, 2.0 * np.exp(-s * s)))


def p_clt(z, eps):
    """Asymptotic only (not finite-sample valid): one-sided normal test."""
    se = z.std(ddof=1) / np.sqrt(len(z))
    return float(norm.sf((z.mean() + eps) / se)) if se > 0 else float(z.mean() + eps <= 0)


PFUN = {'hoeffding': p_hoeffding, 'bernstein': p_bernstein, 'clt': p_clt}


def realized(g, rows, sidx=sidx21):
    if isinstance(g, tuple):                      # mixture w*ga + (1-w)*gb: expected per-task value and cost
        ga, gb, w = g
        va, da = realized(ga, rows, sidx); vb, db = realized(gb, rows, sidx)
        return w * va + (1 - w) * vb, w * da + (1 - w) * db
    a = g[sidx[rows]]
    return V_ALL[rows, a], D_ALL[rows, a]


MIX_W = np.round(np.arange(0.9, 0.0, -0.1), 2)


def refine(path):
    """Insert mixtures of consecutive path policies (weights 0.9..0.1 on the higher-value one); order fixed by A."""
    out = []
    for k in range(len(path) - 1):
        out.append(path[k]); out.extend((path[k], path[k + 1], float(w)) for w in MIX_W)
    out.append(path[-1])
    return out


def certify(path, rowsB, eps, method, alpha=ALPHA):
    """Fixed-sequence test along the price path; returns (khat, p-values). khat = 0 means the reference."""
    ps = []
    for g in path:
        v, _ = realized(g, rowsB)
        ps.append(PFUN[method](v - yB_all[rowsB], eps))
    ps = np.array(ps); k = 0
    while k < len(ps) and ps[k] <= alpha:
        k += 1
    return k, ps


def bonferroni(ps, alpha=ALPHA):
    ok = np.where(ps <= alpha / len(ps))[0]
    return int(ok.max()) + 1 if len(ok) else 0


def fitted_floor(path, pi, v, c, eps):
    """Deterministic fitted-floor policy: the largest k with fitted E[V] >= fitted vbar - eps (no certification)."""
    vb, _ = pol_vd(np.full(len(pi), A_REF), pi, v, c)
    k = 0
    for j, g in enumerate(path):
        if pol_vd(g, pi, v, c)[0] >= vb - eps - 1e-12:
            k = j + 1
    return k


REF_POL = np.full(X21, A_REF)
rngB = np.random.default_rng(SEED)
WT = np.vstack([np.ones(len(TE)), rngB.multinomial(len(TE), np.full(len(TE), 1.0 / len(TE)), size=NBOOT)]).astype(float)
YB_T, CB_T = yB_all[TE], cB_all[TE]
WYB, WCB = WT @ YB_T, WT @ CB_T


def eval_test(g, boot=True):
    v, d = realized(g, TE)
    if not boot:
        return dict(r=float((v.sum() / d.sum()) / (YB_T.sum() / CB_T.sum())), dV=float((v - YB_T).mean()), cost_m=float(d.mean() * 1e3))
    Wv, Wd, n = WT @ v, WT @ d, WT.sum(1)
    r = (Wv / Wd) / (WYB / WCB); dV = (Wv - WYB) / n
    return dict(r=ci(r), cost_m=float(d.mean() * 1e3), dV=ci(dV), boot_dV=dV[1:])


def split_AB(rng, frac=0.5):
    A, B = [], []
    for x in range(X21):
        rows = PR[sidx21[PR] == x].copy(); rng.shuffle(rows)
        nA = int(round(frac * len(rows))); A.append(rows[:nA]); B.append(rows[nA:])
    return np.sort(np.concatenate(A)), np.sort(np.concatenate(B))


def pick(path, k):
    return REF_POL if k == 0 else path[k - 1]
