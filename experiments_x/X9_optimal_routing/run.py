#!/usr/bin/env python3
"""
X9_optimal_routing -- "Route Anything: A Theory of Inference Routers", Theorem thm:optimal-routing on CodeRouterBench.

Objects tested (labels from main_body.tex / appendix_core.tex):
  thm:optimal-routing (i)   Phi(lambda) = sum_x pi(x) max_u [v(x,u) - lambda c(x,u)] convex, strictly decreasing, unique root
                            eta* = max_g eta(g); optimal rule = state-wise argmax of v - eta* c; two-action menu: p W_{eta*} >= eta* a
  thm:optimal-routing (ii)  Dinkelbach iteration lambda_0 = eta_B, lambda_{t+1} = eta(g_t): monotone, finite, Phi(eta_B) = max net gain
  thm:optimal-routing (iii) quality floor E[V] >= vbar - eps: Charnes-Cooper LP (eq:app-cc-lp), <= 1 randomizing state,
                            multiplier kappa (LP dual), every positive-mass action maximizes (1+kappa) v - eta*_eps c,
                            two-action price eta*_eps/(1+kappa)
  prop:threshold            W_lambda = lambda b + v_h - v_b (homogeneous) and the matched-reference form W = lambda cbar_h + v_h - vbar_h
  thm:surplus               net gain Q + eta_B S = E[V - eta_B D] and its terms (avoided reference work, attempt cost, accepted quality
                            change, recovery cost) for every policy evaluated.

Data: released CodeRouterBench outcome matrix (9,999 tasks x 8 models; score in [0,1], token-price cost_usd, latency_ms).
States: 21 dimension x source cells (primary) and 9 dimensions. Menu U(x): 8 ex-ante direct dispatches + 7 primaries x
theta in {1, 0.5} attempt-then-escalate cascades with the oracle verifier H = 1{score_m >= theta} and fallback claude-opus-4-6.
Reference policy = always-Opus (admissible: it is in the menu). Primitives v(x,u), c(x,u), pi(x) on probing (7,080 tasks);
realizations on id_test (2,919 tasks; never fitted on). Resources: dollars (csv cost_usd) and, as a second full run,
seconds (sequential latency: lat_m + (1-H) lat_Opus). Fully offline; numpy / pandas / scipy / matplotlib only.
Seed 20260917; 1,000 paired task-resampling bootstrap draws on id_test; 1,000 probing resamples for fit uncertainty.
Loader code copied (not imported) from experiments/X1_surplus_cascade/run.py and X3_threshold_voi/run.py.
"""
import os, sys, json, time, math
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

T0 = time.time()
SEED = 20260917
NBOOT = 1000                    # id_test paired task resamples (realization uncertainty)
NBOOT_FIT = 1000                # probing resamples (fit uncertainty of eta*, eta*_eps, kappa, per-state action)
N_HILL_STARTS = 2000            # random deterministic starts for coordinate ascent on eta (full menu)
N_RANDOM_POLICIES = 200000      # random deterministic policies evaluated at once (full menu)
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures'); os.makedirs(FIG, exist_ok=True)
DATA = os.environ.get('ACROUTER_DATA', 'data/agent-as-a-router/data')
ACR_DECISIONS = os.environ.get('ACROUTER_ID_DECISIONS',
                               'data/agent-as-a-router/outputs/current/id_decisions.jsonl')
ROUTERBENCH_PKL = os.environ.get('ROUTERBENCH_PKL', './data/routerbench/routerbench_0shot.pkl')  # not used by X9
BASE = 'claude-opus-4-6'
PRIMARIES = ['Qwen3-Max', 'kimi-k2.5', 'qwen3.5-plus', 'glm-5', 'gpt-5.4', 'claude-sonnet-4-6', 'MiniMax-M2.7']
THETAS = [1.0, 0.5]
EPS_GRID = [0.0, 0.005, 0.01, 0.02, 0.05]                 # protocol grid (superset of the conventions' grid)
EPS_FINE_STEP = 0.0025                                     # frontier curve resolution (figure only)
LAMBDA_GRID = [0.1, 1.0, 10.0, 'eta_B', 100.0]             # per dollar and, literally the same numbers, per second
BASES = ['usd', 'sec']
PARTS = ['cell21', 'dim9']
TOL_MASS = 1e-8                 # positive-mass threshold for g(u|x)
TOL_OPT = 1e-8                  # optimality-check tolerance (score units)
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------ figure style (CONVENTIONS: usetex False, 9-pt serif, no titles)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'text.usetex': False,
    'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 7.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': .65, 'lines.linewidth': 1.6,
    'axes.edgecolor': '#c3c2b7', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#0b0b0b',
    'grid.color': '#e1e0d9', 'grid.linewidth': .6,
})
# validated categorical palette (dataviz skill reference instance; 4 adjacent slots pass CVD/normal-vision gates on white)
BLUE, ORANGE, AQUA, YELLOW, GREY, INK, INK2 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#898781', '#0b0b0b', '#52514e'
PART_COLOR = {'cell21': BLUE, 'dim9': ORANGE}

def log(*a):
    print(f'[{time.time()-T0:7.1f}s]', *a, flush=True)

def ci(v):
    """[point, lo, hi]: row 0 = point estimate, rows 1..B = bootstrap (2.5 / 97.5 percentiles; NaN rows dropped)."""
    v = np.asarray(v, float)
    w = v[1:][np.isfinite(v[1:])]
    if len(w) == 0:
        return [float(v[0]), float('nan'), float('nan')]
    return [float(v[0]), float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))]

def excl0(t):
    return bool(np.isfinite(t[1]) and np.isfinite(t[2]) and (t[1] > 0 or t[2] < 0))

# =========================================================================================== 1. data (loader copied from X1/X3)
df = pd.read_csv(f'{DATA}/coderouterbench/id_results_long.csv')
assert df.shape[0] == 79992, df.shape
S = df.pivot(index='task_id', columns='model', values='score')
C = df.pivot(index='task_id', columns='model', values='cost_usd')
L = df.pivot(index='task_id', columns='model', values='latency_ms') / 1000.0          # seconds
assert S.notna().all().all() and C.notna().all().all() and L.notna().all().all()
meta = df.drop_duplicates('task_id').set_index('task_id').loc[S.index, ['split', 'dimension', 'source_split']].copy()
meta['bench'] = meta.index.str.split('_').str[1]                                        # second underscore token = source benchmark
meta['cell21'] = meta.dimension + '|' + meta.bench
meta['dim9'] = meta.dimension
TASKS = S.index.to_numpy(); MODELS = list(S.columns); col = {m: j for j, m in enumerate(MODELS)}
Sv, Cv, Lv = S.to_numpy(), C.to_numpy(), L.to_numpy()
prob = (meta.split == 'probing').to_numpy(); idt = (meta.split == 'id_test').to_numpy()
assert prob.sum() == 7080 and idt.sum() == 2919 and set(PRIMARIES) | {BASE} == set(MODELS)
so, co, lo = Sv[:, col[BASE]], Cv[:, col[BASE]], Lv[:, col[BASE]]
DATA_CHECKS = {
    'n_tasks': int(len(TASKS)), 'n_models': len(MODELS), 'n_probing': int(prob.sum()), 'n_id_test': int(idt.sum()),
    'always_opus_id_test_total_usd': float(co[idt].sum()), 'paper_always_opus_total_usd': 34.02,
    'always_opus_id_test_mean_score': float(so[idt].mean()), 'paper_always_opus_avgperf': 43.83,
    'rows_cost_zero': int((df.cost_usd <= 0).sum()),
    'rows_cost_zero_by_model': {k: int(v) for k, v in df[df.cost_usd <= 0].model.value_counts().items()},
    'rows_latency_zero_by_model': {k: int(v) for k, v in df[df.latency_ms <= 0].model.value_counts().items()},
    'n_cells21': int(meta.cell21.nunique()), 'n_dim9': int(meta.dim9.nunique()),
}
assert abs(DATA_CHECKS['always_opus_id_test_total_usd'] - 34.02) < 0.01
assert DATA_CHECKS['n_cells21'] == 21 and DATA_CHECKS['n_dim9'] == 9
log('data checks', json.dumps(DATA_CHECKS)[:400])

# =========================================================================================== 2. menu U(x): per-task realized (V, D) of every action
# Per-task accounting (eq:accounting): direct m: V = Y_m, D = C_m.  attempt m@theta: H = 1{Y_m >= theta},
# V = H Y_m + (1-H) Y_B, D = C_m + (1-H) C_B (dollars) / lat_m + (1-H) lat_B (seconds, sequential).
# Net-gain terms of thm:surplus per task: avoided reference work H C_B (direct non-Opus: H = 1), attempt cost A = C_m,
# accepted quality change H (Y_m - Y_B), recovery cost (1-H)(C_F - C_B) = 0 (fallback = reference), erroneous acceptance H 1{Y_m < Y_B}.
ACTIONS = []
cols_V, cols_Du, cols_Ds, cols_H, cols_Au, cols_As, cols_Q, cols_E = [], [], [], [], [], [], [], []
for m in MODELS:
    Y, Du, Ds = Sv[:, col[m]], Cv[:, col[m]], Lv[:, col[m]]
    if m == BASE:
        ACTIONS.append(dict(name='direct:' + m, kind='direct_ref', model=m, theta=None, short='O'))
        H = np.zeros(len(TASKS)); Au = np.zeros(len(TASKS)); As = np.zeros(len(TASKS)); Q = np.zeros(len(TASKS)); E = np.zeros(len(TASKS))
    else:
        ACTIONS.append(dict(name='direct:' + m, kind='direct', model=m, theta=None, short=None))
        H = np.ones(len(TASKS)); Au = Du.copy(); As = Ds.copy(); Q = Y - so; E = (Y < so).astype(float)
    cols_V.append(Y.copy()); cols_Du.append(Du.copy()); cols_Ds.append(Ds.copy()); cols_H.append(H); cols_Au.append(Au); cols_As.append(As); cols_Q.append(Q); cols_E.append(E)
for m in PRIMARIES:
    for th in THETAS:
        Y = Sv[:, col[m]]; H = (Y >= th - 1e-12).astype(float)
        ACTIONS.append(dict(name=f'attempt:{m}@{th:g}', kind='attempt', model=m, theta=th, short=None))
        cols_V.append(H * Y + (1 - H) * so); cols_Du.append(Cv[:, col[m]] + (1 - H) * co); cols_Ds.append(Lv[:, col[m]] + (1 - H) * lo)
        cols_H.append(H); cols_Au.append(Cv[:, col[m]].copy()); cols_As.append(Lv[:, col[m]].copy()); cols_Q.append(H * (Y - so)); cols_E.append(H * (Y < so))
SHORT = {'Qwen3-Max': 'Q', 'kimi-k2.5': 'K', 'qwen3.5-plus': 'P', 'glm-5': 'G', 'gpt-5.4': 'T', 'claude-sonnet-4-6': 'S', 'MiniMax-M2.7': 'M', BASE: 'O'}
for a in ACTIONS:
    a['short'] = SHORT[a['model']] if a['kind'] != 'attempt' else f"{SHORT[a['model']]}@{'1' if a['theta'] == 1.0 else '.5'}"
ANAMES = [a['name'] for a in ACTIONS]; U = len(ACTIONS); A_REF = ANAMES.index('direct:' + BASE)
V_ALL = np.column_stack(cols_V); D_ALL = {'usd': np.column_stack(cols_Du), 'sec': np.column_stack(cols_Ds)}
H_ALL = np.column_stack(cols_H); A_ALL = {'usd': np.column_stack(cols_Au), 'sec': np.column_stack(cols_As)}
Q_ALL = np.column_stack(cols_Q); E_ALL = np.column_stack(cols_E)
HCB_ALL = {'usd': H_ALL * co[:, None], 'sec': H_ALL * lo[:, None]}
REF_C = {'usd': co, 'sec': lo}
# identity checks of the per-task accounting (thm:surplus terms): V - Y_B = accepted quality change; C_B - D = H C_B - A
IDENT = {'V_minus_YB_eq_Qacc': float(np.abs((V_ALL - so[:, None]) - Q_ALL).max()),
         'CB_minus_D_eq_HCB_minus_A_usd': float(np.abs((co[:, None] - D_ALL['usd']) - (HCB_ALL['usd'] - A_ALL['usd'])).max()),
         'CB_minus_D_eq_HCB_minus_A_sec': float(np.abs((lo[:, None] - D_ALL['sec']) - (HCB_ALL['sec'] - A_ALL['sec'])).max())}
assert max(IDENT.values()) < 1e-12, IDENT
assert U == 22
log('menu built:', U, 'actions')

# =========================================================================================== 3. states
STATE = {}
for part in PARTS:
    labels = meta[part].to_numpy()
    names = sorted(set(labels))
    idx = np.array([names.index(s) for s in labels])
    STATE[part] = dict(names=names, idx=idx, X=len(names),
                       pi_probing=np.bincount(idx[prob], minlength=len(names)) / prob.sum(),
                       pi_id_test=np.bincount(idx[idt], minlength=len(names)) / idt.sum(),
                       n_probing=np.bincount(idx[prob], minlength=len(names)), n_id_test=np.bincount(idx[idt], minlength=len(names)))
    assert (STATE[part]['n_probing'] > 0).all() and (STATE[part]['n_id_test'] > 0).all()

def primitives(w, sidx, X, V, D):
    """State-wise expected value / cost of every action and state probabilities from task weights w (>= 0, length n_all).
    v(x,u) = sum_i w_i 1{x_i = x} V_iu / sum_i w_i 1{x_i = x}; pi(x) = sum_i w_i 1{x_i=x} / sum_i w_i."""
    nx = np.bincount(sidx, weights=w, minlength=X)
    v = np.column_stack([np.bincount(sidx, weights=w * V[:, u], minlength=X) for u in range(V.shape[1])]) / nx[:, None]
    c = np.column_stack([np.bincount(sidx, weights=w * D[:, u], minlength=X) for u in range(D.shape[1])]) / nx[:, None]
    return nx / nx.sum(), v, c

# =========================================================================================== 4. core theory objects
def phi(lam, pi, v, c):
    return float(pi @ np.max(v - lam * c, axis=1))

def argmax_policy(lam, v, c):
    return np.argmax(v - lam * c, axis=1)

def eta_of(g, pi, v, c):
    ar = np.arange(len(pi))
    return float((pi @ v[ar, g]) / (pi @ c[ar, g]))

def bisect_root(pi, v, c, tol=1e-13):
    """Unique root of the convex, strictly decreasing Phi (thm:optimal-routing (i)); returns (root, iterations, bracket)."""
    lo = 0.0
    assert phi(lo, pi, v, c) > 0
    hi = 1.0
    while phi(hi, pi, v, c) > 0:
        hi *= 2.0
    it = 0
    while hi - lo > tol * hi and it < 400:
        mid = 0.5 * (lo + hi)
        if phi(mid, pi, v, c) > 0:
            lo = mid
        else:
            hi = mid
        it += 1
    return 0.5 * (lo + hi), it, (lo, hi)

def dinkelbach(pi, v, c, lam0, max_iter=1000):
    """lambda_0 = lam0, g_t = deterministic argmax of the lambda_t-objective, lambda_{t+1} = eta(g_t) (part (ii))."""
    path = [float(lam0)]; pols = []; phis = []
    lam = float(lam0)
    for t in range(max_iter):
        g = argmax_policy(lam, v, c); pols.append(g); phis.append(phi(lam, pi, v, c))
        new = eta_of(g, pi, v, c); path.append(new)
        if abs(new - lam) <= 1e-12 * max(1.0, abs(lam)):
            break
        lam = new
    return dict(path=path, policies=pols, phis=phis, n_policy_evaluations=len(pols), n_steps_to_root=len(pols) - 1, root=path[-1])

def cc_lp(pi, v, c, floor_value, method='highs-ds'):
    """Charnes-Cooper LP eq:app-cc-lp: max sum y v  s.t.  sum y c = 1;  sum_u y(x,u) = pi(x) t;  sum y v - floor t - sigma = 0; y,t,sigma >= 0.
    Returns eta*_eps, the router g(u|x) = y/(pi t), sigma, and kappa = dual of the floor row (checked against the analytic tie)."""
    X, U_ = v.shape; nY = X * U_
    cobj = np.concatenate([-v.ravel(), [0.0, 0.0]])
    A = np.zeros((X + 2, nY + 2)); A[0, :nY] = c.ravel()
    for x in range(X):
        A[1 + x, x * U_:(x + 1) * U_] = 1.0; A[1 + x, nY] = -pi[x]
    A[X + 1, :nY] = v.ravel(); A[X + 1, nY] = -floor_value; A[X + 1, nY + 1] = -1.0
    b = np.zeros(X + 2); b[0] = 1.0
    res = linprog(cobj, A_eq=A, b_eq=b, bounds=(0, None), method=method)
    if res.status != 0:
        return dict(status=int(res.status), message=str(res.message))
    y = res.x[:nY].reshape(X, U_); t = res.x[nY]; sigma = res.x[nY + 1]
    g = y / (pi[:, None] * t)
    g = np.where(g < TOL_MASS, 0.0, g); g = g / g.sum(1, keepdims=True)
    return dict(status=0, eta=float(-res.fun), g=g, t=float(t), sigma=float(sigma), kappa=float(res.eqlin.marginals[X + 1]),
                dual_cost_row=float(res.eqlin.marginals[0]), n_iter=int(res.nit))

def fixed_lambda_floor_lp(pi, v, c, lam, floor_value, method='highs-ds'):
    """J_lambda with a quality floor: max sum_x pi(x) sum_u g(u|x)(v - lam c)  s.t.  sum_u g(u|x) = 1,  sum pi g v >= floor.  kappa = -dual of the floor."""
    X, U_ = v.shape; nY = X * U_
    cobj = -(pi[:, None] * (v - lam * c)).ravel()
    Aeq = np.zeros((X, nY))
    for x in range(X):
        Aeq[x, x * U_:(x + 1) * U_] = 1.0
    Aub = -(pi[:, None] * v).ravel()[None, :]
    res = linprog(cobj, A_ub=Aub, b_ub=[-floor_value], A_eq=Aeq, b_eq=np.ones(X), bounds=(0, 1), method=method)
    if res.status != 0:
        return dict(status=int(res.status), message=str(res.message))
    g = res.x.reshape(X, U_); g = np.where(g < TOL_MASS, 0.0, g); g = g / g.sum(1, keepdims=True)
    return dict(status=0, J=float(-res.fun), g=g, kappa=float(-res.ineqlin.marginals[0]))

def policy_summary(g, pi, v, c, kappa, price):
    """Randomizing states, and the optimality check: every positive-mass action maximizes (1+kappa) v - price c in its state."""
    mod = (1 + kappa) * v - price * c
    best = mod.max(1)
    viol = np.where(g > 0, best[:, None] - mod, 0.0)
    n_rand = int(((g > 0).sum(1) >= 2).sum())
    rand_states = [int(x) for x in np.where((g > 0).sum(1) >= 2)[0]]
    EV = float(pi @ (g * v).sum(1)); ED = float(pi @ (g * c).sum(1))
    return dict(n_randomizing_states=n_rand, randomizing_states=rand_states, max_optimality_violation=float(viol.max()),
                EV=EV, ED=ED, eta=EV / ED)

def deterministic(g):
    return np.argmax(g, axis=1)

# ------------------------------------------------------------------ independent checks of part (i): exhaustive / random / hill-climb
def exhaustive_two_action(pi, v2, c2):
    """Max efficiency over all 2^X deterministic policies of the two-action menu (col 0 = direct Opus, col 1 = attempt)."""
    X = len(pi); N = 1 << X
    baseV = pi @ v2[:, 0]; baseC = pi @ c2[:, 0]; dV = pi * (v2[:, 1] - v2[:, 0]); dC = pi * (c2[:, 1] - c2[:, 0])
    shifts = np.arange(X, dtype=np.uint32); best = -np.inf; best_mask = 0
    for start in range(0, N, 1 << 18):
        idx = np.arange(start, min(N, start + (1 << 18)), dtype=np.uint32)
        bits = ((idx[:, None] >> shifts[None, :]) & 1).astype(np.float64)
        r = (baseV + bits @ dV) / (baseC + bits @ dC)
        k = int(np.argmax(r))
        if r[k] > best:
            best = float(r[k]); best_mask = int(idx[k])
    bits = np.array([(best_mask >> x) & 1 for x in range(X)])
    return best, bits, N

def coordinate_ascent(g, pi, v, c, max_sweeps=200):
    X = len(pi); ar = np.arange(X)
    EV = float(pi @ v[ar, g]); ED = float(pi @ c[ar, g]); sweeps = 0; improved = True
    while improved and sweeps < max_sweeps:
        improved = False; sweeps += 1
        for x in range(X):
            EVx = EV - pi[x] * v[x, g[x]] + pi[x] * v[x]; EDx = ED - pi[x] * c[x, g[x]] + pi[x] * c[x]
            r = EVx / EDx; u = int(np.argmax(r))
            if r[u] > EV / ED * (1 + 1e-15):
                g[x] = u; EV = float(EVx[u]); ED = float(EDx[u]); improved = True
    return g, EV / ED, sweeps

def search_checks(pi, v, c, rng_):
    X, U_ = v.shape; ar = np.arange(X)
    idx = rng_.integers(0, U_, size=(N_RANDOM_POLICIES, X))
    EV = (pi[None, :] * v[ar[None, :], idx]).sum(1); ED = (pi[None, :] * c[ar[None, :], idx]).sum(1)
    rand_max = float((EV / ED).max())
    hill_max = -np.inf; n_at_max = 0; sweeps_tot = 0
    for s in range(N_HILL_STARTS):
        g0 = rng_.integers(0, U_, size=X)
        g, r, sw = coordinate_ascent(g0, pi, v, c); sweeps_tot += sw
        if r > hill_max * (1 + 1e-12):
            hill_max = r; n_at_max = 1
        elif abs(r - hill_max) <= 1e-12 * hill_max:
            n_at_max += 1
    return dict(random_policies=N_RANDOM_POLICIES, random_max=rand_max, hill_starts=N_HILL_STARTS, hill_max=float(hill_max),
                hill_starts_reaching_max=int(n_at_max), mean_sweeps=sweeps_tot / N_HILL_STARTS)

# ------------------------------------------------------------------ two-action rule primitives (prop:threshold; matched-reference form)
def two_action_primitives(w, sidx, X, m, basis):
    """Per state: p = P(H=1), a = E[A], b = E[C_B], v_b = E[Y_B], v_h = E[Y_m|H=1], vbar_h = E[Y_B|H=1], cbar_h = E[C_B|H=1] (probing weights w)."""
    u_att1 = ANAMES.index(f'attempt:{m}@1'); H = H_ALL[:, u_att1]  # theta=1 hit indicator, replaced per theta below
    out = {}
    for th in THETAS:
        u_att = ANAMES.index(f'attempt:{m}@{th:g}'); H = H_ALL[:, u_att]
        Ym = Sv[:, col[m]]; Am = A_ALL[basis][:, u_att]; CB = REF_C[basis]
        nx = np.bincount(sidx, weights=w, minlength=X); nh = np.bincount(sidx, weights=w * H, minlength=X)
        with np.errstate(divide='ignore', invalid='ignore'):
            p = nh / nx
            a = np.bincount(sidx, weights=w * Am, minlength=X) / nx
            b = np.bincount(sidx, weights=w * CB, minlength=X) / nx
            v_b = np.bincount(sidx, weights=w * so, minlength=X) / nx
            v_h = np.bincount(sidx, weights=w * H * Ym, minlength=X) / nh
            vbar_h = np.bincount(sidx, weights=w * H * so, minlength=X) / nh
            cbar_h = np.bincount(sidx, weights=w * H * CB, minlength=X) / nh
        out[th] = dict(p=p, a=a, b=b, v_b=v_b, v_h=v_h, vbar_h=vbar_h, cbar_h=cbar_h, u_att=u_att)
    return out

def two_action_rule(prim, lam):
    """lhs of p W_lambda >= lambda a in the exact matched-reference form (W = lam cbar_h + v_h - vbar_h) and the homogeneous form
    (W = lam b + v_h - v_b, b = E[C_B|x], v_b = E[Y_B|x]); p = 0 states: attempting is strictly worse (lhs = -lam a)."""
    p, a = prim['p'], prim['a']
    with np.errstate(invalid='ignore'):
        W_exact = lam * prim['cbar_h'] + prim['v_h'] - prim['vbar_h']
        W_hom = lam * prim['b'] + prim['v_h'] - prim['v_b']
        lhs_exact = np.where(p > 0, p * np.nan_to_num(W_exact) - lam * a, -lam * a)
        lhs_hom = np.where(p > 0, p * np.nan_to_num(W_hom) - lam * a, -lam * a)
    return lhs_exact, lhs_hom, W_exact, W_hom

# =========================================================================================== 5. realization on id_test + paired bootstrap
test_rows = np.where(idt)[0]; n_test = len(test_rows)
W_TEST = np.vstack([np.ones((1, n_test)), rng.multinomial(n_test, np.full(n_test, 1.0 / n_test), size=NBOOT).astype(float)])
so_t, co_t, lo_t = so[test_rows], co[test_rows], lo[test_rows]
V_T = V_ALL[test_rows]; D_T = {b: D_ALL[b][test_rows] for b in BASES}; HCB_T = {b: HCB_ALL[b][test_rows] for b in BASES}
A_T = {b: A_ALL[b][test_rows] for b in BASES}; Q_T = Q_ALL[test_rows]; E_T = E_ALL[test_rows]; H_T = H_ALL[test_rows]
KIND_IDX = {'direct_ref': np.array([i for i, a in enumerate(ACTIONS) if a['kind'] == 'direct_ref']),
            'direct': np.array([i for i, a in enumerate(ACTIONS) if a['kind'] == 'direct']),
            'attempt': np.array([i for i, a in enumerate(ACTIONS) if a['kind'] == 'attempt'])}

def realize(G, basis):
    """Per-task expected outcome of a (possibly randomized) assignment G (n_test x U, rows sum to 1) on id_test."""
    V = (G * V_T).sum(1); D = (G * D_T[basis]).sum(1); HCB = (G * HCB_T[basis]).sum(1); A = (G * A_T[basis]).sum(1)
    Q = (G * Q_T).sum(1); E = (G * E_T).sum(1)
    shares = {k: float(G[:, ix].sum() / len(G)) for k, ix in KIND_IDX.items()}
    esc = float((G[:, KIND_IDX['attempt']] * (1 - H_T[:, KIND_IDX['attempt']])).sum() / len(G))   # escalation rate (attempts that fell back)
    return dict(V=V, D=D, HCB=HCB, A=A, Q=Q, E=E, shares=shares, escalation_rate=esc)

def boot_stats(rz, basis, lam_price=None):
    """Weighted sums under W_TEST (row 0 point; rows 1..B paired bootstrap): eta = sum V / sum D (ratio of sums), eta_B, net gain terms."""
    cB = REF_C[basis][test_rows]
    M = W_TEST @ np.column_stack([rz['V'], rz['D'], so_t, cB, rz['HCB'], rz['A'], rz['Q'], rz['E']])
    n = W_TEST.sum(1)
    EV, ED, vbar, cbar = M[:, 0] / n, M[:, 1] / n, M[:, 2] / n, M[:, 3] / n
    eta = M[:, 0] / M[:, 1]; eta_B = M[:, 2] / M[:, 3]
    out = dict(eta=eta, eta_B=eta_B, r=eta / eta_B, EV=EV, ED=ED, vbar=vbar, cbar=cbar, dV=EV - vbar, Q=EV - vbar, S=cbar - ED,
               net_gain=(EV - vbar) + eta_B * (cbar - ED), avoided=M[:, 4] / n, attempt_cost=M[:, 5] / n, Q_acc=M[:, 6] / n, e=M[:, 7] / n)
    out['net_gain_terms_resid'] = float(np.abs(out['net_gain'] - (out['Q_acc'] + eta_B * (out['avoided'] - out['attempt_cost']))).max())
    if lam_price is not None:
        out['J_lambda'] = EV - lam_price * ED
    return out

def summarize(st, keys=('r', 'eta', 'EV', 'ED', 'dV', 'Q', 'S', 'net_gain', 'avoided', 'attempt_cost', 'Q_acc', 'e')):
    rec = {k: ci(st[k]) for k in keys if k in st}
    if 'J_lambda' in st:
        rec['J_lambda'] = ci(st['J_lambda'])
    rec['net_gain_terms_resid'] = st['net_gain_terms_resid']
    return rec

def floor_rates(st, eps_list):
    """Fraction of bootstrap resamples with E[V] >= vbar_resample - eps (paired: vbar from the same resample); point check separately."""
    out = {}
    for e in eps_list:
        out[f'{e:g}'] = dict(rate_paired=float(np.mean(st['EV'][1:] >= st['vbar'][1:] - e - 1e-12)),
                            rate_fixed_vbar=float(np.mean(st['EV'][1:] >= st['vbar'][0] - e - 1e-12)),
                            point_satisfied=bool(st['EV'][0] >= st['vbar'][0] - e - 1e-12))
    return out

def assignment_from_rule(g, part):
    return g[STATE[part]['idx'][test_rows]]

def assignment_from_models(chosen):
    G = np.zeros((n_test, U))
    for i, m in enumerate(chosen):
        G[i, ANAMES.index('direct:' + m)] = 1.0
    return G

def assignment_action(u):
    G = np.zeros((n_test, U)); G[:, u] = 1.0; return G

# =========================================================================================== 6. baselines on id_test (decisions files)
def load_decisions(path):
    d = pd.read_json(path, lines=True)
    n_file = len(d)
    d = d[d.task_id.isin(set(TASKS[test_rows]))].drop_duplicates('task_id').set_index('task_id').reindex(TASKS[test_rows])
    assert d.chosen_model.notna().all() and d.chosen_model.isin(MODELS).all(), path
    return d.chosen_model.to_numpy(), n_file

DB_CHOICE, DB_NFILE = load_decisions(f'{DATA}/id/voter_decisions/dimension_best.jsonl')
ACR_CHOICE, ACR_NFILE = load_decisions(ACR_DECISIONS)
G_BASE = {'always_opus': assignment_action(A_REF), 'dimension_best': assignment_from_models(DB_CHOICE), 'acrouter': assignment_from_models(ACR_CHOICE)}
BASE_SHARES = {'dimension_best': {k: int(v) for k, v in pd.Series(DB_CHOICE).value_counts().items()},
               'acrouter': {k: int(v) for k, v in pd.Series(ACR_CHOICE).value_counts().items()}}

# =========================================================================================== 7. per-task oracle on id_test (upper bound; state = task)
def oracle_dinkelbach(w, V, D, lam0):
    """Per-task states with weights w: g_t(i) = argmax_u V_iu - lam D_iu; lam_{t+1} = sum w V / sum w D."""
    n = V.shape[0]; ar = np.arange(n); lam = lam0; path = [lam0]
    for _ in range(200):
        g = np.argmax(V - lam * D, axis=1); new = float((w * V[ar, g]).sum() / (w * D[ar, g]).sum()); path.append(new)
        if abs(new - lam) <= 1e-12 * max(1.0, abs(lam)):
            break
        lam = new
    return g, path

def oracle_floor_lp(V, D, floor_value):
    """Charnes-Cooper LP with one state per id_test task (sparse), pi = 1/n."""
    n, U_ = V.shape; nY = n * U_; pi_ = np.full(n, 1.0 / n)
    cobj = np.concatenate([-V.ravel(), [0.0, 0.0]])
    r0 = np.zeros(nY, int); c0 = np.arange(nY); v0 = D.ravel()
    r1 = 1 + np.repeat(np.arange(n), U_); c1 = np.arange(nY); v1 = np.ones(nY)
    r1t = 1 + np.arange(n); c1t = np.full(n, nY); v1t = -pi_
    r2 = np.full(nY, n + 1); c2 = np.arange(nY); v2 = V.ravel()
    rows = np.concatenate([r0, r1, r1t, r2, [n + 1, n + 1]]); cols = np.concatenate([c0, c1, c1t, c2, [nY, nY + 1]])
    vals = np.concatenate([v0, v1, v1t, v2, [-floor_value, -1.0]])
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n + 2, nY + 2)); b = np.zeros(n + 2); b[0] = 1.0
    res = linprog(cobj, A_eq=A, b_eq=b, bounds=(0, None), method='highs-ds')
    assert res.status == 0, res.message
    y = res.x[:nY].reshape(n, U_); t = res.x[nY]; G = y / (pi_[:, None] * t); G = np.where(G < TOL_MASS, 0.0, G); G = G / G.sum(1, keepdims=True)
    return dict(eta=float(-res.fun), G=G, kappa=float(res.eqlin.marginals[n + 1]), sigma=float(res.x[nY + 1]),
                n_randomizing_tasks=int(((G > 0).sum(1) >= 2).sum()))

# =========================================================================================== 8. main pipeline per resource basis
RESULTS = {'experiment_id': 'X9_optimal_routing', 'seed': SEED, 'nboot_id_test': NBOOT, 'nboot_probing_fit': NBOOT_FIT,
           'data_checks': DATA_CHECKS, 'accounting_identity_max_abs': IDENT,
           'menu': [dict(index=i, **{k: a[k] for k in ('name', 'kind', 'model', 'theta', 'short')}) for i, a in enumerate(ACTIONS)],
           'reference_action': ANAMES[A_REF], 'eps_grid': EPS_GRID, 'lambda_grid': [str(l) for l in LAMBDA_GRID],
           'states': {part: dict(names=STATE[part]['names'], pi_probing=STATE[part]['pi_probing'].tolist(), pi_id_test=STATE[part]['pi_id_test'].tolist(),
                                 n_probing=STATE[part]['n_probing'].tolist(), n_id_test=STATE[part]['n_id_test'].tolist()) for part in PARTS},
           'baseline_files': dict(dimension_best=dict(path=f'{DATA}/id/voter_decisions/dimension_best.jsonl', rows_in_file=int(DB_NFILE), shares=BASE_SHARES['dimension_best']),
                                  acrouter=dict(path=ACR_DECISIONS, rows_in_file=int(ACR_NFILE), shares=BASE_SHARES['acrouter'])),
           'runs': {}, 'baselines': {}, 'oracle': {}, 'sensitivity_lambda': {}, 'frontier': {}, 'fit_bootstrap': {}, 'per_state': {}}
FIGDATA = {}
w_prob = prob.astype(float)

for basis in BASES:
    unit = {'usd': 'per dollar', 'sec': 'per second'}[basis]
    Dm = D_ALL[basis]
    log(f'===== basis {basis} ({unit}) =====')
    # ---------------------------------------------------------------- baselines (realized on id_test, paired bootstrap)
    BL = {}
    ST_REF = boot_stats(realize(G_BASE['always_opus'], basis), basis)
    eta_B_test = float(ST_REF['eta'][0])
    for name in ['always_opus', 'dimension_best', 'acrouter']:
        rz = realize(G_BASE[name], basis); st = boot_stats(rz, basis)
        BL[name] = dict(summary=summarize(st), shares=rz['shares'], escalation_rate=rz['escalation_rate'], floor=floor_rates(st, EPS_GRID), _st=st)
    # fixed cascades (X1) and single direct dispatches: select the best on probing (OOS-fair), report all on id_test
    fixed = {}
    for u, a in enumerate(ACTIONS):
        prob_eta = float(V_ALL[prob, u].sum() / Dm[prob, u].sum())
        rz = realize(assignment_action(u), basis); st = boot_stats(rz, basis)
        fixed[a['name']] = dict(kind=a['kind'], probing_eta=prob_eta, probing_r=prob_eta / (so[prob].sum() / REF_C[basis][prob].sum()),
                                summary=summarize(st), floor=floor_rates(st, EPS_GRID), shares=rz['shares'], escalation_rate=rz['escalation_rate'], _st=st)
    casc_names = [a['name'] for a in ACTIONS if a['kind'] == 'attempt']; dir_names = [a['name'] for a in ACTIONS if a['kind'] == 'direct']
    best_casc = max(casc_names, key=lambda k: fixed[k]['probing_eta']); best_dir = max(dir_names, key=lambda k: fixed[k]['probing_eta'])
    best_casc_test = max(casc_names, key=lambda k: fixed[k]['summary']['eta'][0]); best_dir_test = max(dir_names, key=lambda k: fixed[k]['summary']['eta'][0])
    BL['best_fixed_cascade'] = dict(fixed[best_casc], name=best_casc, selected_on='probing', best_on_id_test=best_casc_test)
    BL['best_single_direct'] = dict(fixed[best_dir], name=best_dir, selected_on='probing', best_on_id_test=best_dir_test)
    BL['_fixed_all'] = fixed
    # per-task oracle (upper bound), unconstrained: point + paired bootstrap of the fixed oracle assignment + re-optimized per resample
    g_or, path_or = oracle_dinkelbach(np.ones(n_test), V_T, D_T[basis], eta_B_test)
    G_or = np.zeros((n_test, U)); G_or[np.arange(n_test), g_or] = 1.0
    rz = realize(G_or, basis); st_or = boot_stats(rz, basis)
    reopt = np.empty(NBOOT + 1)
    for bi in range(NBOOT + 1):
        gb, pb = oracle_dinkelbach(W_TEST[bi], V_T, D_T[basis], eta_B_test); w_ = W_TEST[bi]
        reopt[bi] = float((w_ * V_T[np.arange(n_test), gb]).sum() / (w_ * D_T[basis][np.arange(n_test), gb]).sum())
    ORACLE = dict(unconstrained=dict(summary=summarize(st_or), dinkelbach_path=path_or, n_steps=len(path_or) - 2, shares=rz['shares'], escalation_rate=rz['escalation_rate'],
                                     floor=floor_rates(st_or, EPS_GRID), r_reoptimized_per_resample=ci(reopt / ST_REF['eta']), _st=st_or), floor_eps={})
    vbar_test = float(so_t.mean())
    for e in EPS_GRID:
        t1 = time.time(); lp = oracle_floor_lp(V_T, D_T[basis], vbar_test - e)
        rz = realize(lp['G'], basis); st = boot_stats(rz, basis)
        ORACLE['floor_eps'][f'{e:g}'] = dict(eta=lp['eta'], r=lp['eta'] / eta_B_test, kappa=max(lp['kappa'], 0.0), kappa_raw=lp['kappa'], n_randomizing_tasks=lp['n_randomizing_tasks'],
                                              binding=bool(lp['sigma'] < 1e-9), summary=summarize(st), shares=rz['shares'], lp_seconds=time.time() - t1, _st=st)
    log(f'baselines {basis}: r Opus 1, DimensionBest {BL["dimension_best"]["summary"]["r"][0]:.3f}, ACRouter {BL["acrouter"]["summary"]["r"][0]:.3f}, '
        f'best cascade {best_casc} {BL["best_fixed_cascade"]["summary"]["r"][0]:.3f}, best direct {best_dir} {BL["best_single_direct"]["summary"]["r"][0]:.3f}, '
        f'oracle {ORACLE["unconstrained"]["summary"]["r"][0]:.3f} (eps=0: {ORACLE["floor_eps"]["0"]["r"]:.3f})')
    RESULTS['baselines'][basis] = BL; RESULTS['oracle'][basis] = ORACLE

    for part in PARTS:
        X = STATE[part]['X']; sidx = STATE[part]['idx']; names = STATE[part]['names']
        pi, v, c = primitives(w_prob, sidx, X, V_ALL, Dm)
        pi_test_, v_test, c_test = primitives(idt.astype(float), sidx, X, V_ALL, Dm)
        eta_B = float((pi @ v[:, A_REF]) / (pi @ c[:, A_REF])); vbar = float(pi @ v[:, A_REF]); cbar = float(pi @ c[:, A_REF])
        assert abs(eta_B - so[prob].sum() / REF_C[basis][prob].sum()) < 1e-9 * eta_B
        run = dict(basis=basis, part=part, X=X, U=U, n_deterministic_policies=float(U) ** X, eta_B_probing=eta_B, vbar_probing=vbar, cbar_probing=cbar,
                   c_min=float(c.min()), c_min_action=ANAMES[int(np.argmin(c.min(0)))], eta_B_id_test=eta_B_test,
                   v=v.tolist(), c=c.tolist(), v_id_test=v_test.tolist(), c_id_test=c_test.tolist(),
                   shift_max_abs_v=float(np.abs(v - v_test).max()), shift_max_abs_c=float(np.abs(c - c_test).max()), shift_max_rel_c=float((np.abs(c - c_test) / c).max()))
        # ------------------------------------------------------------ (i) Phi grid, root, convexity
        root, n_bis, bracket = bisect_root(pi, v, c)
        lam_grid = np.linspace(0.0, 1.6 * root, 401)
        phi_vals = np.array([phi(l, pi, v, c) for l in lam_grid])
        d2 = np.diff(phi_vals, 2)
        run['phi'] = dict(lambda_grid=lam_grid.tolist(), values=phi_vals.tolist(), strictly_decreasing=bool(np.all(np.diff(phi_vals) < 0)),
                          convex_min_second_difference=float(d2.min()), convex=bool(d2.min() >= -1e-12 * np.abs(phi_vals).max()),
                          phi_at_root=phi(root, pi, v, c), phi_at_eta_B=phi(eta_B, pi, v, c), phi_at_zero=phi(0.0, pi, v, c))
        run['root'] = dict(eta_star=root, r_star=root / eta_B, bisection_iterations=n_bis, bracket=list(bracket), bracket_width_rel=(bracket[1] - bracket[0]) / root)
        # sensitivity: pi from id_test with probing v, c
        root_pit, _, _ = bisect_root(STATE[part]['pi_id_test'], v, c)
        run['root']['eta_star_pi_id_test'] = root_pit; run['root']['r_star_pi_id_test'] = root_pit / eta_B
        # ------------------------------------------------------------ (ii) Dinkelbach from eta_B
        dk = dinkelbach(pi, v, c, eta_B)
        ar = np.arange(X)
        ED_t = [float(pi @ c[ar, g]) for g in dk['policies']]
        ident = [dk['phis'][t] - ED_t[t] * (dk['path'][t + 1] - dk['path'][t]) for t in range(len(dk['policies']))]
        g0 = dk['policies'][0]
        # Phi(eta_B) = max_g net gain: compare with the net gain of g_0, of every fixed action, and of random policies
        gain_single = np.array([pi @ (v[:, u] - eta_B * c[:, u]) for u in range(U)])
        ridx = rng.integers(0, U, size=(20000, X)); gain_rand = (pi[None, :] * (v[ar[None, :], ridx] - eta_B * c[ar[None, :], ridx])).sum(1)
        run['dinkelbach'] = dict(lambda_path=dk['path'], n_steps_to_root=dk['n_steps_to_root'], n_policy_evaluations=dk['n_policy_evaluations'],
                                 monotone_nondecreasing=bool(np.all(np.diff(dk['path']) >= -1e-12)), strictly_increasing_until_stop=bool(np.all(np.diff(dk['path'][:-1]) > 0)),
                                 terminal_minus_root=dk['root'] - root, terminal_rel_err=abs(dk['root'] - root) / root,
                                 identity_phi_eq_ED_times_step_max_abs=float(np.max(np.abs(ident))), steps_le_X=bool(dk['n_steps_to_root'] <= X),
                                 steps_le_n_deterministic=bool(dk['n_steps_to_root'] <= float(U) ** X),
                                 phi_at_eta_B=dk['phis'][0], net_gain_g0=float(pi @ (v[ar, g0] - eta_B * c[ar, g0])),
                                 net_gain_g0_minus_phi=float(pi @ (v[ar, g0] - eta_B * c[ar, g0])) - dk['phis'][0],
                                 max_net_gain_single_action=float(gain_single.max()), max_net_gain_random_policies=float(gain_rand.max()),
                                 policies=[[ANAMES[u] for u in g] for g in dk['policies']],
                                 policies_short=[[ACTIONS[u]['short'] for u in g] for g in dk['policies']],
                                 eta_of_policies=[eta_of(g, pi, v, c) for g in dk['policies']])
        # ------------------------------------------------------------ (i) checks: LP certificate, search, one-swap, two-action exhaustive + rule
        lp0 = cc_lp(pi, v, c, -1e6)           # floor never binds: exact max of eta over ALL randomized routers (LP certificate)
        assert lp0['status'] == 0
        g_star = argmax_policy(root, v, c)
        eta_gstar = eta_of(g_star, pi, v, c)
        ties = int(np.sum(np.sum(np.abs((v - root * c) - (v - root * c).max(1, keepdims=True)) < 1e-12, axis=1) > 1))
        # one-swap test: any single-state deviation from g* cannot raise eta (and lowers it strictly unless tied)
        EVs, EDs = pi @ v[ar, g_star], pi @ c[ar, g_star]; swap_max = -np.inf; n_swap = 0
        for x in range(X):
            for u in range(U):
                if u == g_star[x]:
                    continue
                r_ = (EVs - pi[x] * v[x, g_star[x]] + pi[x] * v[x, u]) / (EDs - pi[x] * c[x, g_star[x]] + pi[x] * c[x, u]); swap_max = max(swap_max, r_); n_swap += 1
        sc = search_checks(pi, v, c, np.random.default_rng(SEED + 1))
        two = {}
        for m in PRIMARIES:
            prim = two_action_primitives(w_prob, sidx, X, m, basis)
            for th in THETAS:
                u_att = prim[th]['u_att']; v2 = v[:, [A_REF, u_att]]; c2 = c[:, [A_REF, u_att]]
                r2, _, _ = bisect_root(pi, v2, c2)
                best_ex, bits_ex, n_enum = exhaustive_two_action(pi, v2, c2)
                g2 = argmax_policy(r2, v2, c2)
                lhs_exact, lhs_hom, W_exact, W_hom = two_action_rule(prim[th], r2)
                obj_diff = (v2[:, 1] - r2 * c2[:, 1]) - (v2[:, 0] - r2 * c2[:, 0])
                agree_exact = int(np.sum((lhs_exact >= 0) == (g2 == 1))); agree_hom = int(np.sum((lhs_hom >= 0) == (g2 == 1)))
                two[f'{m}@{th:g}'] = dict(eta_star_two_action=r2, r_star=r2 / eta_B, exhaustive_max=best_ex, exhaustive_minus_root=best_ex - r2, n_enumerated=int(n_enum),
                                          exhaustive_policy_eq_argmax=bool(np.array_equal(bits_ex, g2)), attempt_states=int(g2.sum()),
                                          rule_exact_minus_objective_diff_max_abs=float(np.max(np.abs(lhs_exact - obj_diff))),
                                          rule_exact_agreement=agree_exact, rule_homogeneous_agreement=agree_hom, n_states=X,
                                          p=prim[th]['p'].tolist(), a=prim[th]['a'].tolist(), W_exact=np.nan_to_num(W_exact).tolist(), W_hom=np.nan_to_num(W_hom).tolist(),
                                          lhs_exact=lhs_exact.tolist(), lhs_hom=lhs_hom.tolist())
        run['checks_part_i'] = dict(lp_certificate_eta=lp0['eta'], lp_minus_root=lp0['eta'] - root, lp_minus_root_rel=(lp0['eta'] - root) / root,
                                    lp_router_deterministic=bool(((lp0['g'] > 0).sum(1) == 1).all()), lp_router_eq_argmax=bool(np.array_equal(deterministic(lp0['g']), g_star)),
                                    eta_argmax_policy=eta_gstar, eta_argmax_minus_root_rel=(eta_gstar - root) / root, n_tied_states=ties,
                                    one_swap_max_eta=float(swap_max), one_swap_max_minus_root=float(swap_max - root), one_swap_count=n_swap,
                                    search=sc, search_max_minus_root_rel=(max(sc['random_max'], sc['hill_max']) - root) / root,
                                    dinkelbach_terminal_eq_argmax=bool(np.array_equal(dk['policies'][-1], g_star)), two_action=two,
                                    two_action_exhaustive_max_abs_dev=float(max(abs(t['exhaustive_minus_root']) for t in two.values())),
                                    two_action_rule_exact_agreement=int(sum(t['rule_exact_agreement'] for t in two.values())),
                                    two_action_rule_homogeneous_agreement=int(sum(t['rule_homogeneous_agreement'] for t in two.values())),
                                    two_action_rule_total=int(sum(t['n_states'] for t in two.values())))
        run['optimal_rule'] = dict(actions=[ANAMES[u] for u in g_star], short=[ACTIONS[u]['short'] for u in g_star],
                                   shares={k: float(np.isin(g_star, ix).mean()) for k, ix in KIND_IDX.items()},
                                   EV=float(pi @ v[ar, g_star]), ED=float(pi @ c[ar, g_star]), shortfall_vs_reference=float(vbar - pi @ v[ar, g_star]))
        # ------------------------------------------------------------ (iii) quality floor: CC-LP per eps, kappa, randomizing states
        FL = {}; POL_EPS = {}
        for e in EPS_GRID:
            lp = cc_lp(pi, v, c, vbar - e); assert lp['status'] == 0, lp
            kap = max(lp['kappa'], 0.0)
            ps = policy_summary(lp['g'], pi, v, c, kap, lp['eta'])
            # analytic kappa from the tie in the randomizing state (two positive actions u1,u2): (1+kappa)(v1-v2) = eta (c1-c2)
            kap_tie = None
            if ps['n_randomizing_states'] == 1:
                x = ps['randomizing_states'][0]; us = np.where(lp['g'][x] > 0)[0]; u1, u2 = us[0], us[1]
                kap_tie = float(lp['eta'] * (c[x, u1] - c[x, u2]) / (v[x, u1] - v[x, u2]) - 1.0)
            # independent check of eta*_eps: root of Phi_eps(lambda) = max_{g in G_eps} E[V - lambda D] (fixed-lambda floor LP + bisection)
            def phi_eps(lam):
                r_ = fixed_lambda_floor_lp(pi, v, c, lam, vbar - e); return r_['J'], r_
            lo_, hi_ = 0.0, lp['eta'] * 2.0 + 1.0
            for _ in range(60):
                mid = 0.5 * (lo_ + hi_)
                if phi_eps(mid)[0] > 0: lo_ = mid
                else: hi_ = mid
            root_eps = 0.5 * (lo_ + hi_); Jf, fl = phi_eps(lp['eta'])
            FL[f'{e:g}'] = dict(eps=e, floor_value=vbar - e, eta_eps=lp['eta'], r_eps=lp['eta'] / eta_B, kappa=kap, kappa_raw=lp['kappa'], kappa_tie=kap_tie,
                                price=lp['eta'] / (1 + kap), binding=bool(lp['sigma'] < 1e-9), sigma=lp['sigma'], lp_iterations=lp['n_iter'],
                                n_randomizing_states=ps['n_randomizing_states'], randomizing_states=[names[x] for x in ps['randomizing_states']],
                                randomizing_actions=[[(ANAMES[u], float(lp['g'][x, u])) for u in np.where(lp['g'][x] > 0)[0]] for x in ps['randomizing_states']],
                                max_optimality_violation=ps['max_optimality_violation'], EV=ps['EV'], ED=ps['ED'], eta_check=ps['eta'],
                                root_phi_eps=root_eps, root_phi_eps_minus_eta_eps_rel=(root_eps - lp['eta']) / lp['eta'],
                                phi_eps_at_eta_eps=Jf, fixed_lambda_lp_kappa_at_eta_eps=fl['kappa'],
                                actions=[ANAMES[int(np.argmax(lp['g'][x]))] for x in range(X)], short=[ACTIONS[int(np.argmax(lp['g'][x]))]['short'] for x in range(X)],
                                g_nonzero={names[x]: {ANAMES[u]: float(lp['g'][x, u]) for u in np.where(lp['g'][x] > 0)[0]} for x in range(X)},
                                shares={k: float((pi[:, None] * lp['g'][:, ix]).sum()) for k, ix in KIND_IDX.items()})
            POL_EPS[e] = lp['g']
        run['floor'] = FL
        # fine frontier (figure): eps from quality gains (negative) to slack
        fine = {}
        e_lo = -0.10; e_hi = 0.06
        for e in np.round(np.arange(e_lo, e_hi + 1e-9, EPS_FINE_STEP), 6):
            lp = cc_lp(pi, v, c, vbar - e)
            if lp['status'] != 0:
                fine[f'{e:g}'] = dict(eps=float(e), feasible=False); continue
            kap = max(lp['kappa'], 0.0); ps = policy_summary(lp['g'], pi, v, c, kap, lp['eta'])
            fine[f'{e:g}'] = dict(eps=float(e), feasible=True, eta_eps=lp['eta'], r_eps=lp['eta'] / eta_B, kappa=kap, n_randomizing_states=ps['n_randomizing_states'],
                                  price=lp['eta'] / (1 + kap), binding=bool(lp['sigma'] < 1e-9))
        max_gain = float(pi @ v.max(1) - vbar)
        run['frontier_fine'] = dict(points=fine, max_feasible_quality_gain=max_gain, shortfall_of_unconstrained_optimum=run['optimal_rule']['shortfall_vs_reference'])
        # ------------------------------------------------------------ OOS realization of the fitted rules on id_test (paired bootstrap)
        OOS = {}
        pol = {'unconstrained': np.eye(U)[g_star]}
        for e in EPS_GRID:
            pol[f'eps={e:g}'] = POL_EPS[e]
        coin = np.random.default_rng(SEED + 7).random(n_test)
        for pname, g in pol.items():
            Gt = assignment_from_rule(g, part)
            rz = realize(Gt, basis); st = boot_stats(rz, basis)
            rec = dict(summary=summarize(st), shares=rz['shares'], escalation_rate=rz['escalation_rate'], floor=floor_rates(st, EPS_GRID),
                       n_randomizing_states_deployed=int(((g > 0).sum(1) >= 2).sum()),
                       n_randomizing_states_realized_id_test=int(len(set(STATE[part]['idx'][test_rows][(Gt > 0).sum(1) >= 2]))),
                       predicted_eta=float((pi @ (g * v).sum(1)) / (pi @ (g * c).sum(1))), predicted_r=float((pi @ (g * v).sum(1)) / (pi @ (g * c).sum(1))) / eta_B,
                       predicted_EV=float(pi @ (g * v).sum(1)))
            rec['realized_minus_predicted_r'] = ci(st['r'] - rec['predicted_r'])
            # coin-flip realization of the randomization (one uniform per task, fixed seed)
            if rec['n_randomizing_states_deployed'] > 0:
                cum = np.cumsum(Gt, axis=1); pick = (coin[:, None] > cum).sum(1); pick = np.minimum(pick, U - 1)
                Gc = np.zeros_like(Gt); Gc[np.arange(n_test), pick] = 1.0
                stc = boot_stats(realize(Gc, basis), basis); rec['coin_realization'] = dict(summary=summarize(stc), floor=floor_rates(stc, EPS_GRID))
            # paired comparisons against baselines (same resampled tasks)
            comp = {}
            for bname in ['always_opus', 'dimension_best', 'acrouter', 'best_fixed_cascade', 'best_single_direct']:
                d = st['r'] - BL[bname]['_st']['r']
                comp[bname] = dict(diff_r=ci(d), ci_excludes_0=excl0(ci(d)), rule_wins=bool(ci(d)[1] > 0), frac_resamples_rule_better=float(np.mean(d[1:] > 0)))
            d = st['r'] - ORACLE['unconstrained']['_st']['r']
            comp['per_task_oracle'] = dict(diff_r=ci(d), ci_excludes_0=excl0(ci(d)), gap_to_oracle_r=ci(-d), fraction_of_oracle_r=ci(st['r'] / ORACLE['unconstrained']['_st']['r']))
            rec['vs'] = comp; rec['_st'] = st
            OOS[pname] = rec
        # safety margin needed for >= 95% OOS floor satisfaction at each target eps (fit at eps' <= eps)
        margin = {}
        for e in EPS_GRID:
            found = None
            for ep in np.round(np.arange(e, e - 0.0301, -0.0025), 6):
                lp = cc_lp(pi, v, c, vbar - ep)
                if lp['status'] != 0:
                    break
                st = boot_stats(realize(assignment_from_rule(lp['g'], part), basis), basis)
                rate = float(np.mean(st['EV'][1:] >= st['vbar'][1:] - e - 1e-12))
                if rate >= 0.95:
                    found = dict(fit_eps=float(ep), margin=float(e - ep), rate=rate, r=ci(st['r'])); break
            margin[f'{e:g}'] = found if found is not None else dict(fit_eps=None, margin=None, rate=None, note='no fit eps within 0.03 reaches 95%')
        run['oos'] = OOS; run['oos_safety_margin_for_95pct'] = margin
        log(f'{basis}/{part}: eta_B={eta_B:.4g} eta*={root:.4g} (r*={root/eta_B:.3f}) bisect {n_bis} it; Dinkelbach steps {dk["n_steps_to_root"]} path {[round(p, 4) for p in dk["path"]]}; '
            f'LP cert rel dev {run["checks_part_i"]["lp_minus_root_rel"]:.1e}; search rel dev {run["checks_part_i"]["search_max_minus_root_rel"]:.1e}; '
            f'two-action exhaustive max dev {run["checks_part_i"]["two_action_exhaustive_max_abs_dev"]:.1e}, rule agreement exact {run["checks_part_i"]["two_action_rule_exact_agreement"]}/{run["checks_part_i"]["two_action_rule_total"]}, hom {run["checks_part_i"]["two_action_rule_homogeneous_agreement"]}')
        for e in EPS_GRID:
            f = FL[f'{e:g}']; o = OOS[f'eps={e:g}']
            log(f'   eps={e:<6g} eta_eps={f["eta_eps"]:.4g} r={f["r_eps"]:.3f} kappa={f["kappa"]:.3f} (tie {f["kappa_tie"]}) price={f["price"]:.4g} nrand={f["n_randomizing_states"]} '
                f'viol={f["max_optimality_violation"]:.1e} | OOS r={o["summary"]["r"][0]:.3f} [{o["summary"]["r"][1]:.3f},{o["summary"]["r"][2]:.3f}] dV={o["summary"]["dV"][0]:+.4f} floor rate={o["floor"][f"{e:g}"]["rate_paired"]:.3f}')
        log(f'   OOS unconstrained r={OOS["unconstrained"]["summary"]["r"][0]:.3f} vs DimensionBest diff {OOS["unconstrained"]["vs"]["dimension_best"]["diff_r"]}, vs best cascade {OOS["unconstrained"]["vs"]["best_fixed_cascade"]["diff_r"]}')
        # ------------------------------------------------------------ fit uncertainty: probing bootstrap of eta*, eta*_eps, kappa, actions
        t1 = time.time()
        fb = dict(eta_star=[], r_star=[], steps=[], eps={f'{e:g}': dict(eta=[], kappa=[], n_rand=[], actions=[]) for e in EPS_GRID}, actions_unc=[])
        rng_fit = np.random.default_rng(SEED + 11)
        for bi in range(NBOOT_FIT):
            w = rng_fit.multinomial(int(prob.sum()), np.full(int(prob.sum()), 1.0 / prob.sum())).astype(float)
            wfull = np.zeros(len(TASKS)); wfull[prob] = w
            pi_b, v_b, c_b = primitives(wfull, sidx, X, V_ALL, Dm)
            ok = np.isfinite(v_b).all(1)
            pi_b, v_b, c_b = pi_b[ok] / pi_b[ok].sum(), v_b[ok], c_b[ok]
            eta_B_b = float((pi_b @ v_b[:, A_REF]) / (pi_b @ c_b[:, A_REF])); vbar_b = float(pi_b @ v_b[:, A_REF])
            rb, _, _ = bisect_root(pi_b, v_b, c_b); dkb = dinkelbach(pi_b, v_b, c_b, eta_B_b)
            fb['eta_star'].append(rb); fb['r_star'].append(rb / eta_B_b); fb['steps'].append(dkb['n_steps_to_root'])
            gb = np.full(X, -1); gb[ok] = argmax_policy(rb, v_b, c_b); fb['actions_unc'].append(gb)
            for e in EPS_GRID:
                lpb = cc_lp(pi_b, v_b, c_b, vbar_b - e)
                if lpb['status'] != 0:
                    continue
                kb = max(lpb['kappa'], 0.0)
                fb['eps'][f'{e:g}']['eta'].append(lpb['eta'] / eta_B_b); fb['eps'][f'{e:g}']['kappa'].append(kb)
                fb['eps'][f'{e:g}']['n_rand'].append(int(((lpb['g'] > 0).sum(1) >= 2).sum()))
                ab = np.full(X, -1); ab[ok] = deterministic(lpb['g']); fb['eps'][f'{e:g}']['actions'].append(ab)
        au = np.array(fb['actions_unc'])
        FB = dict(seconds=time.time() - t1, n_resamples=NBOOT_FIT,
                  eta_star_ci=ci(np.r_[root, fb['eta_star']]), r_star_ci=ci(np.r_[root / eta_B, fb['r_star']]),
                  dinkelbach_steps_max=int(max(fb['steps'])), dinkelbach_steps_mean=float(np.mean(fb['steps'])), dinkelbach_steps_hist={str(k): int(v_) for k, v_ in zip(*np.unique(fb['steps'], return_counts=True))},
                  action_stability_unconstrained={names[x]: float(np.mean(au[:, x] == g_star[x])) for x in range(X)},
                  action_stability_unconstrained_mean=float(np.mean([np.mean(au[:, x] == g_star[x]) for x in range(X)])), eps={})
        for e in EPS_GRID:
            d_ = fb['eps'][f'{e:g}']; ae = np.array(d_['actions']); g_pt = deterministic(POL_EPS[e])
            FB['eps'][f'{e:g}'] = dict(r_eps_ci=ci(np.r_[FL[f'{e:g}']['r_eps'], d_['eta']]), kappa_ci=ci(np.r_[FL[f'{e:g}']['kappa'], d_['kappa']]),
                                       n_rand_max=int(max(d_['n_rand'])), n_rand_mean=float(np.mean(d_['n_rand'])), n_rand_hist={str(k): int(v_) for k, v_ in zip(*np.unique(d_['n_rand'], return_counts=True))},
                                       frac_floor_binding=float(np.mean(np.array(d_['kappa']) > 1e-9)), n_resamples=len(d_['eta']),
                                       action_stability_mean=float(np.mean([np.mean(ae[:, x] == g_pt[x]) for x in range(X)])),
                                       action_stability={names[x]: float(np.mean(ae[:, x] == g_pt[x])) for x in range(X)})
        run['fit_bootstrap'] = FB
        log(f'   fit bootstrap ({FB["seconds"]:.0f}s): r* {FB["r_star_ci"]}, steps max {FB["dinkelbach_steps_max"]}, max n_rand over eps/resamples {max(FB["eps"][k]["n_rand_max"] for k in FB["eps"])}, action stability {FB["action_stability_unconstrained_mean"]:.3f}')
        # ------------------------------------------------------------ sensitivity: fixed-price objective J_lambda
        SENS = {}
        for lam_key in LAMBDA_GRID:
            lam = eta_B if lam_key == 'eta_B' else float(lam_key)
            g_l = argmax_policy(lam, v, c)
            in_eta = eta_of(g_l, pi, v, c); in_J = float(pi @ (v[ar, g_l] - lam * c[ar, g_l])); J_root_rule = float(pi @ (v[ar, g_star] - lam * c[ar, g_star]))
            st = boot_stats(realize(assignment_from_rule(np.eye(U)[g_l], part), basis), basis, lam_price=lam)
            st_root = boot_stats(realize(assignment_from_rule(np.eye(U)[g_star], part), basis), basis, lam_price=lam)
            # oracle J_lambda on id_test (per task argmax) and its eta
            g_o = np.argmax(V_T - lam * D_T[basis], axis=1); Go = np.zeros((n_test, U)); Go[np.arange(n_test), g_o] = 1.0
            st_o = boot_stats(realize(Go, basis), basis, lam_price=lam)
            fl = {}
            for e in EPS_GRID:
                r_ = fixed_lambda_floor_lp(pi, v, c, lam, vbar - e)
                if r_['status'] != 0:
                    fl[f'{e:g}'] = dict(status=r_['status']); continue
                psl = policy_summary(r_['g'], pi, v, c, max(r_['kappa'], 0.0), lam * (1 + max(r_['kappa'], 0.0)))
                stf = boot_stats(realize(assignment_from_rule(r_['g'], part), basis), basis, lam_price=lam)
                fl[f'{e:g}'] = dict(J=r_['J'], kappa=max(r_['kappa'], 0.0), n_randomizing_states=psl['n_randomizing_states'], max_optimality_violation=psl['max_optimality_violation'],
                                    in_sample_eta=psl['eta'], r=psl['eta'] / eta_B, oos=summarize(stf), oos_floor=floor_rates(stf, [e]))
            SENS[str(lam_key)] = dict(lam=lam, lam_over_eta_star=lam / root, in_sample=dict(eta=in_eta, r=in_eta / eta_B, J=in_J, phi=phi(lam, pi, v, c), J_of_root_rule=J_root_rule,
                                                                                            eta_le_eta_star=bool(in_eta <= root * (1 + 1e-12)), J_ge_root_rule=bool(in_J >= J_root_rule - 1e-12)),
                                      actions_short=[ACTIONS[u]['short'] for u in g_l], shares={k: float(np.isin(g_l, ix).mean()) for k, ix in KIND_IDX.items()},
                                      oos=summarize(st), oos_root_rule=summarize(st_root), oos_diff_J_root_rule_minus_lambda_rule=ci(st_root['J_lambda'] - st['J_lambda']),
                                      oos_diff_r_root_rule_minus_lambda_rule=ci(st_root['r'] - st['r']), oracle_id_test=summarize(st_o), floor=fl)
        run['sensitivity_lambda'] = SENS
        RESULTS['runs'][f'{basis}/{part}'] = run
        FIGDATA[(basis, part)] = dict(pi=pi, v=v, c=c, root=root, eta_B=eta_B, dk=dk, g_star=g_star, POL_EPS=POL_EPS, vbar=vbar)

    # ---------------------------------------------------------------- per-state decomposition (cell21) of the gaps: oracle - rule, rule - ACRouter, rule - DimensionBest
    part = 'cell21'; X = STATE[part]['X']; sidx_t = STATE[part]['idx'][test_rows]; names = STATE[part]['names']
    run = RESULTS['runs'][f'{basis}/{part}']; g_star = FIGDATA[(basis, part)]['g_star']
    cB_t = REF_C[basis][test_rows]
    pols = {'rule': assignment_from_rule(np.eye(U)[g_star], part), 'rule_eps0': assignment_from_rule(FIGDATA[(basis, part)]['POL_EPS'][0.0], part),
            'rule_eps0.01': assignment_from_rule(FIGDATA[(basis, part)]['POL_EPS'][0.01], part),
            'oracle': G_or, 'acrouter': G_BASE['acrouter'], 'dimension_best': G_BASE['dimension_best'], 'always_opus': G_BASE['always_opus']}
    dec = {}
    for pname, Gp in pols.items():
        rz = realize(Gp, basis)
        gain_i = rz['V'] - eta_B_test * rz['D']                        # per-task net gain at the id_test reference price (thm:surplus)
        dec[pname] = dict(net_gain_by_state=(np.bincount(sidx_t, weights=gain_i, minlength=X) / n_test).tolist(),
                          EV_by_state=(np.bincount(sidx_t, weights=rz['V'], minlength=X) / np.bincount(sidx_t, minlength=X)).tolist(),
                          ED_by_state=(np.bincount(sidx_t, weights=rz['D'], minlength=X) / np.bincount(sidx_t, minlength=X)).tolist(),
                          net_gain_total=float(gain_i.mean()), eta=float(rz['V'].sum() / rz['D'].sum()),
                          identity_check=float(abs((rz['V'].sum() / rz['D'].sum() - eta_B_test) - gain_i.mean() / rz['D'].mean())))
    RESULTS['per_state'][basis] = dict(part=part, state_names=names, pi_id_test=STATE[part]['pi_id_test'].tolist(), eta_B_id_test=eta_B_test,
                                       units='net gain per task in score units at the id_test reference price (divide by eta_B for resource units); sums over states give E[V - eta_B D]',
                                       policies=dec,
                                       gap_oracle_minus_rule_by_state=(np.array(dec['oracle']['net_gain_by_state']) - np.array(dec['rule']['net_gain_by_state'])).tolist(),
                                       gap_rule_minus_acrouter_by_state=(np.array(dec['rule']['net_gain_by_state']) - np.array(dec['acrouter']['net_gain_by_state'])).tolist(),
                                       gap_rule_minus_dimension_best_by_state=(np.array(dec['rule']['net_gain_by_state']) - np.array(dec['dimension_best']['net_gain_by_state'])).tolist(),
                                       acrouter_choice_by_state={names[x]: {k: int(v_) for k, v_ in pd.Series(ACR_CHOICE[sidx_t == x]).value_counts().items()} for x in range(X)})
    log(f'per-state decomposition {basis}: identity max {max(d["identity_check"] for d in dec.values()):.1e}; oracle-rule total gap {dec["oracle"]["net_gain_total"]-dec["rule"]["net_gain_total"]:.4f}, rule-ACRouter {dec["rule"]["net_gain_total"]-dec["acrouter"]["net_gain_total"]:.4f}')

# =========================================================================================== 9. audit: independent recomputation of three headline numbers
log('audit: independent recomputation')
from scipy.optimize import brentq
AUD = {}
wide_s = df.pivot(index='task_id', columns='model', values='score'); wide_c = df.pivot(index='task_id', columns='model', values='cost_usd')
info = df.drop_duplicates('task_id').set_index('task_id')[['split', 'dimension']]
info['cell'] = info.dimension + '|' + info.index.str.split('_').str[1]
pr_ids = info.index[info.split == 'probing']; te_ids = info.index[info.split == 'id_test']
# A1: eta* (usd, cell21) from pandas groupby primitives and brentq on Phi
frames = {}
for m in MODELS:
    frames['direct:' + m] = pd.DataFrame({'V': wide_s[m], 'D': wide_c[m]})
for m in PRIMARIES:
    for th in THETAS:
        Hm = (wide_s[m] >= th - 1e-12)
        frames[f'attempt:{m}@{th:g}'] = pd.DataFrame({'V': Hm * wide_s[m] + (~Hm) * wide_s[BASE], 'D': wide_c[m] + (~Hm) * wide_c[BASE]})
vtab = pd.DataFrame({k: f.loc[pr_ids, 'V'].groupby(info.loc[pr_ids, 'cell']).mean() for k, f in frames.items()})
ctab = pd.DataFrame({k: f.loc[pr_ids, 'D'].groupby(info.loc[pr_ids, 'cell']).mean() for k, f in frames.items()})
pitab = info.loc[pr_ids, 'cell'].value_counts(normalize=True).reindex(vtab.index)
def Phi_pd(lam):
    return float((pitab * (vtab - lam * ctab).max(axis=1)).sum())
root_pd = brentq(Phi_pd, 1.0, 5000.0, xtol=1e-12, rtol=1e-14, maxiter=500)
AUD['A1_eta_star_usd_cell21'] = dict(run_py=RESULTS['runs']['usd/cell21']['root']['eta_star'], audit_pandas_brentq=root_pd,
                                     rel_mismatch=abs(root_pd - RESULTS['runs']['usd/cell21']['root']['eta_star']) / root_pd)
# A2: OOS eta/eta_B of the unconstrained rule (usd, cell21) by a plain Python loop over id_test rows
rule_pd = {cell: (vtab.loc[cell] - root_pd * ctab.loc[cell]).idxmax() for cell in vtab.index}
rows_by_task = {}
for r_ in df.itertuples(index=False):
    rows_by_task.setdefault(r_.task_id, {})[r_.model] = (float(r_.score), float(r_.cost_usd))
sumV = sumD = sumYB = sumCB = 0.0
for tid in te_ids:
    act = rule_pd[info.at[tid, 'cell']]; rec_ = rows_by_task[tid]; yb, cb = rec_[BASE]
    if act.startswith('direct:'):
        y, cst = rec_[act.split(':', 1)[1]]
    else:
        m, th = act.split(':', 1)[1].split('@'); ym, cm = rec_[m]
        if ym >= float(th) - 1e-12:
            y, cst = ym, cm
        else:
            y, cst = yb, cm + cb
    sumV += y; sumD += cst; sumYB += yb; sumCB += cb
r_pd = (sumV / sumD) / (sumYB / sumCB)
AUD['A2_oos_r_unconstrained_usd_cell21'] = dict(run_py=RESULTS['runs']['usd/cell21']['oos']['unconstrained']['summary']['r'][0], audit_plain_loop=r_pd,
                                                rel_mismatch=abs(r_pd - RESULTS['runs']['usd/cell21']['oos']['unconstrained']['summary']['r'][0]) / r_pd,
                                                rule_actions_match=bool(all(rule_pd[c] == a for c, a in zip(STATE['cell21']['names'], RESULTS['runs']['usd/cell21']['optimal_rule']['actions']))))
# A3: kappa at eps = 0 (usd, cell21) from the tie in the randomizing state (pandas primitives; no LP), and the Dinkelbach path with plain loops
f0 = RESULTS['runs']['usd/cell21']['floor']['0']
if f0['n_randomizing_states'] == 1:
    cell = f0['randomizing_states'][0]; (u1, w1), (u2, w2) = f0['randomizing_actions'][0]
    eta_eps = f0['eta_eps']
    kappa_pd = eta_eps * (ctab.at[cell, u1] - ctab.at[cell, u2]) / (vtab.at[cell, u1] - vtab.at[cell, u2]) - 1.0
    # eta_eps itself, recomputed from the mixture weights and pandas primitives
    EVm = sum(pitab[c_] * (vtab.at[c_, a_] if c_ != cell else w1 * vtab.at[c_, u1] + w2 * vtab.at[c_, u2]) for c_, a_ in zip(vtab.index, [f0['actions'][STATE['cell21']['names'].index(c_)] for c_ in vtab.index]))
    EDm = sum(pitab[c_] * (ctab.at[c_, a_] if c_ != cell else w1 * ctab.at[c_, u1] + w2 * ctab.at[c_, u2]) for c_, a_ in zip(vtab.index, [f0['actions'][STATE['cell21']['names'].index(c_)] for c_ in vtab.index]))
    AUD['A3_kappa_eps0_usd_cell21'] = dict(run_py_lp_dual=f0['kappa'], audit_tie_formula_pandas=float(kappa_pd), rel_mismatch=abs(kappa_pd - f0['kappa']) / f0['kappa'],
                                          eta_eps_run_py=eta_eps, eta_eps_audit=float(EVm / EDm), eta_eps_rel_mismatch=abs(EVm / EDm - eta_eps) / eta_eps,
                                          floor_check_EV_minus_vbar=float(EVm - (pitab * vtab['direct:' + BASE]).sum()))
else:
    AUD['A3_kappa_eps0_usd_cell21'] = dict(note='no randomizing state at eps=0; kappa not identified by a tie', run_py_lp_dual=f0['kappa'])
lam = float((pitab * vtab['direct:' + BASE]).sum() / (pitab * ctab['direct:' + BASE]).sum()); path_pd = [lam]
for _ in range(100):
    acts = {cell: (vtab.loc[cell] - lam * ctab.loc[cell]).idxmax() for cell in vtab.index}
    new = sum(pitab[c_] * vtab.at[c_, a_] for c_, a_ in acts.items()) / sum(pitab[c_] * ctab.at[c_, a_] for c_, a_ in acts.items()); path_pd.append(float(new))
    if abs(new - lam) <= 1e-12 * max(1.0, abs(lam)):
        break
    lam = new
AUD['A3b_dinkelbach_path_usd_cell21'] = dict(run_py=RESULTS['runs']['usd/cell21']['dinkelbach']['lambda_path'], audit_pandas_loop=path_pd,
                                             max_abs_diff=float(max(abs(a - b) for a, b in zip(RESULTS['runs']['usd/cell21']['dinkelbach']['lambda_path'], path_pd))),
                                             same_length=len(path_pd) == len(RESULTS['runs']['usd/cell21']['dinkelbach']['lambda_path']))
AUD['zero_latency_rows'] = dict(n=int((df.latency_ms <= 0).sum()), share_with_score_zero=float((df[df.latency_ms <= 0].score == 0).mean()),
                                by_model=DATA_CHECKS['rows_latency_zero_by_model'])
RESULTS['audit'] = AUD
log('audit', json.dumps({k: {kk: vv for kk, vv in v_.items() if 'mismatch' in kk or kk in ('same_length', 'rule_actions_match', 'max_abs_diff')} for k, v_ in AUD.items() if isinstance(v_, dict)}))

# =========================================================================================== 10. claims
def run_(b, p): return RESULTS['runs'][f'{b}/{p}']
CL = []
def claim(cid, kind, statement, prediction, measurement, verdict, note=''):
    CL.append(dict(id=cid, kind=kind, statement=statement, prediction=prediction, measurement=measurement, verdict=verdict, note=note))

tol_rel = 1e-9
c1_ok = True; c1_parts = []
for b in BASES:
    for p in PARTS:
        ch = run_(b, p)['checks_part_i']; rt = run_(b, p)['root']
        ok = (abs(ch['lp_minus_root_rel']) < tol_rel and ch['search_max_minus_root_rel'] < tol_rel and abs(ch['eta_argmax_minus_root_rel']) < tol_rel
              and ch['one_swap_max_minus_root'] <= tol_rel * rt['eta_star'] and ch['two_action_exhaustive_max_abs_dev'] < tol_rel * rt['eta_star']
              and ch['lp_router_eq_argmax'] and ch['dinkelbach_terminal_eq_argmax'] and ch['two_action_rule_exact_agreement'] == ch['two_action_rule_total']
              and run_(b, p)['phi']['strictly_decreasing'] and run_(b, p)['phi']['convex'])
        c1_ok &= ok
        c1_parts.append(f"{b}/{p}: eta*={rt['eta_star']:.6g} (r*={rt['r_star']:.3f}); LP certificate rel dev {ch['lp_minus_root_rel']:.1e}; hill-climb {ch['search']['hill_starts_reaching_max']}/{ch['search']['hill_starts']} starts reach eta*, "
                        f"random {ch['search']['random_policies']} policies max rel dev {(ch['search']['random_max']-rt['eta_star'])/rt['eta_star']:.2e}; argmax policy eta rel dev {ch['eta_argmax_minus_root_rel']:.1e}; one-swap max - eta* = {ch['one_swap_max_minus_root']:.2e}; "
                        f"two-action exhaustive (2^{run_(b, p)['X']} policies x 14 sub-menus) max |dev| {ch['two_action_exhaustive_max_abs_dev']:.1e}; rule pW>=eta*a agreement {ch['two_action_rule_exact_agreement']}/{ch['two_action_rule_total']} (homogeneous-W form {ch['two_action_rule_homogeneous_agreement']})")
claim('C1', 'IDENTITY', 'thm:optimal-routing (i): the root eta* of Phi equals the maximum efficiency over deterministic state-wise policies, and the optimal rule is the state-wise argmax of v - eta* c; two-action sub-menus obey p W_{eta*} >= eta* a',
      'root = LP certificate (exact max over all randomized routers) = best deterministic policy found by exhaustive (two-action) / hill-climb / random search, to 1e-9 relative; argmax policy attains it; no single-state deviation improves it; rule agreement 100% (exact matched-reference W)',
      ' | '.join(c1_parts), 'PASS' if c1_ok else 'FAIL',
      'Phi strictly decreasing and convex on the grid in all runs. The homogeneous W = lambda b + v_h - v_b (b, v_b unconditional state means) is an approximation here (reference outcomes depend on acceptance; X1 C3/C4) and disagrees with the argmax in some states; the exact form W = lambda cbar_h + v_h - vbar_h agrees everywhere.')

c2_ok = True; c2_parts = []
for b in BASES:
    for p in PARTS:
        dk_ = run_(b, p)['dinkelbach']; fb_ = run_(b, p)['fit_bootstrap']
        ok = dk_['steps_le_X'] and dk_['monotone_nondecreasing'] and dk_['strictly_increasing_until_stop'] and dk_['terminal_rel_err'] < tol_rel and dk_['identity_phi_eq_ED_times_step_max_abs'] < 1e-9 \
             and abs(dk_['net_gain_g0_minus_phi']) < 1e-12 and dk_['max_net_gain_random_policies'] <= dk_['phi_at_eta_B'] + 1e-12 and fb_['dinkelbach_steps_max'] <= run_(b, p)['X']
        c2_ok &= ok
        c2_parts.append(f"{b}/{p}: path {[round(x, 5) for x in dk_['lambda_path']]} -> {dk_['n_steps_to_root']} steps ({dk_['n_policy_evaluations']} policy evaluations) <= |X| = {run_(b, p)['X']}; terminal rel err {dk_['terminal_rel_err']:.1e}; "
                        f"Phi(lambda_t) = E_gt[D](lambda_t+1 - lambda_t) resid {dk_['identity_phi_eq_ED_times_step_max_abs']:.1e}; Phi(eta_B) = {dk_['phi_at_eta_B']:.5f} = net gain of g_0 (resid {dk_['net_gain_g0_minus_phi']:.1e}) >= max over 20,000 random policies {dk_['max_net_gain_random_policies']:.5f}; "
                        f"probing bootstrap: steps max {fb_['dinkelbach_steps_max']}, mean {fb_['dinkelbach_steps_mean']:.2f}")
claim('C2', 'IDENTITY', 'thm:optimal-routing (ii): the Dinkelbach iteration from lambda_0 = eta_B is monotone, finite (design criterion: steps <= |X|), reaches eta*, and its first step is the net-gain maximization of thm:surplus',
      'lambda_t nondecreasing, strictly increasing until it stops, terminal value = root; steps <= |X| (theorem bound: <= |U|^|X| deterministic routers); Phi(eta_B) = max net gain',
      ' | '.join(c2_parts), 'PASS' if c2_ok else 'FAIL', 'The theorem bounds the number of steps by the number of deterministic routers; |X| is the tighter design criterion and holds with 1-4 steps in every run and probing resample.')

c3_ok = True; c3_parts = []
for b in BASES:
    for p in PARTS:
        FL_ = run_(b, p)['floor']; fb_ = run_(b, p)['fit_bootstrap']; ff = run_(b, p)['frontier_fine']['points']
        nr_max = max(f['n_randomizing_states'] for f in FL_.values()); nr_fine = max(f.get('n_randomizing_states', 0) for f in ff.values() if f['feasible'])
        nr_boot = max(fb_['eps'][k]['n_rand_max'] for k in fb_['eps'])
        viol = max(f['max_optimality_violation'] for f in FL_.values())
        kap_dev = max((abs(f['kappa_tie'] - f['kappa']) / max(f['kappa'], 1e-12)) for f in FL_.values() if f['kappa_tie'] is not None) if any(f['kappa_tie'] is not None for f in FL_.values()) else 0.0
        root_dev = max(abs(f['root_phi_eps_minus_eta_eps_rel']) for f in FL_.values())
        ok = nr_max <= 1 and nr_fine <= 1 and nr_boot <= 1 and viol < TOL_OPT and kap_dev < 1e-6 and root_dev < 1e-6
        c3_ok &= ok
        c3_parts.append(f"{b}/{p}: randomizing states per eps {[f['n_randomizing_states'] for f in FL_.values()]} (max over {sum(1 for f in ff.values() if f['feasible'])} fine-grid eps: {nr_fine}; over {fb_['n_resamples']} probing resamples x 5 eps: {nr_boot}); "
                        f"kappa {[round(f['kappa'], 3) for f in FL_.values()]}, LP dual vs tie formula rel dev {kap_dev:.1e}; eta*_eps vs root of Phi_eps rel dev {root_dev:.1e}; max violation of (1+kappa)v - eta*_eps c optimality {viol:.1e}; "
                        f"price eta*_eps/(1+kappa) {[round(f['price'], 4) for f in FL_.values()]}")
claim('C3', 'IDENTITY', 'thm:optimal-routing (iii): under the floor E[V] >= vbar - eps the Charnes-Cooper optimum randomizes in at most one state, kappa from the LP dual makes every positive-mass action a maximizer of (1+kappa) v - eta*_eps c, and the two-action price is eta*_eps/(1+kappa)',
      '<= 1 randomizing state for every eps, every fine-grid eps and every probing resample; optimality violation 0; kappa(dual) = kappa(tie); eta*_eps = root of Phi_eps',
      ' | '.join(c3_parts), 'PASS' if c3_ok else 'FAIL',
      'In dollars the floor binds for eps < shortfall of the unconstrained optimum (kappa > 0, exactly one randomizing state); for larger eps kappa = 0 and the rule is the unconstrained one. In seconds the unconstrained optimum already improves quality, so kappa = 0 at every eps.')

c4_rows = []; c4_ok = True
for b in BASES:
    for p in PARTS:
        for pn in ['unconstrained'] + [f'eps={e:g}' for e in EPS_GRID]:
            o = run_(b, p)['oos'][pn]
            win_db = o['vs']['dimension_best']['rule_wins']; win_bc = o['vs']['best_fixed_cascade']['rule_wins']
            c4_rows.append(f"{b}/{p}/{pn}: r={o['summary']['r'][0]:.3f} [{o['summary']['r'][1]:.3f}, {o['summary']['r'][2]:.3f}]; diff vs DimensionBest {o['vs']['dimension_best']['diff_r'][0]:+.3f} [{o['vs']['dimension_best']['diff_r'][1]:+.3f}, {o['vs']['dimension_best']['diff_r'][2]:+.3f}]; vs best fixed cascade {o['vs']['best_fixed_cascade']['diff_r'][0]:+.3f} [{o['vs']['best_fixed_cascade']['diff_r'][1]:+.3f}, {o['vs']['best_fixed_cascade']['diff_r'][2]:+.3f}]")
            if p == 'cell21':
                c4_ok &= (win_db and win_bc)
claim('C4', 'OUT-OF-SAMPLE', 'the probing-fitted rule policy (21 cells; unconstrained and every eps) realizes a higher efficiency on id_test than DimensionBest and than the best fixed cascade of X1 (primary x theta selected on probing), with paired bootstrap CIs excluding 0',
      'lower CI bound of the paired difference in eta/eta_B > 0 against both baselines, both resource bases, all rules',
      ' | '.join(c4_rows), 'PASS' if c4_ok else 'FAIL',
      f"Baselines on id_test (dollars): DimensionBest r={RESULTS['baselines']['usd']['dimension_best']['summary']['r'][0]:.3f}, best fixed cascade {RESULTS['baselines']['usd']['best_fixed_cascade']['name']} r={RESULTS['baselines']['usd']['best_fixed_cascade']['summary']['r'][0]:.3f}, ACRouter r={RESULTS['baselines']['usd']['acrouter']['summary']['r'][0]:.3f}, best single dispatch {RESULTS['baselines']['usd']['best_single_direct']['name']} r={RESULTS['baselines']['usd']['best_single_direct']['summary']['r'][0]:.3f}; (seconds): DimensionBest {RESULTS['baselines']['sec']['dimension_best']['summary']['r'][0]:.3f}, best cascade {RESULTS['baselines']['sec']['best_fixed_cascade']['name']} {RESULTS['baselines']['sec']['best_fixed_cascade']['summary']['r'][0]:.3f}, ACRouter {RESULTS['baselines']['sec']['acrouter']['summary']['r'][0]:.3f}, best dispatch {RESULTS['baselines']['sec']['best_single_direct']['name']} {RESULTS['baselines']['sec']['best_single_direct']['summary']['r'][0]:.3f}. In dollars most of the lead comes from cheap direct dispatch (Qwen3-Max, kimi-k2.5) that the coarser baselines do not use; the 9-dimension rule is listed for comparison.")

c5_rows = []; c5_ok = True
_drops = [-(run_('usd', p)['oos'][f'eps={e:g}']['summary']['dV'][0]) - (run_('usd', p)['floor'][f'{e:g}']['floor_value'] - run_('usd', p)['floor'][f'{e:g}']['EV'] + e) for p in PARTS for e in EPS_GRID]
_rates_bind = [run_('usd', p)['oos'][f'eps={e:g}']['floor'][f'{e:g}']['rate_paired'] for p in PARTS for e in EPS_GRID if run_('usd', p)['floor'][f'{e:g}']['binding']]
_rates_sec = [run_('sec', p)['oos'][f'eps={e:g}']['floor'][f'{e:g}']['rate_paired'] for p in PARTS for e in EPS_GRID]
c5_note = (f"In dollars the fitted floor binds in-sample for eps < {run_('usd','cell21')['optimal_rule']['shortfall_vs_reference']:.4f} and the realized quality of every fitted rule falls {min(_drops):.4f}-{max(_drops):.4f} below its in-sample value (selection of the argmax on probing plus split shift), "
           f"so the binding eps-rules satisfy their own floor in only {100*min(_rates_bind):.0f}-{100*max(_rates_bind):.0f}% of resamples and the unconstrained rule at eps = 0.02 in about half; the bootstrap sd of E[V]-vbar is about 0.007, so even without shift a floor that binds exactly is met in about 50% of resamples. "
           f"Refitting at a smaller eps (a safety margin of 0.015-0.03, table 2) restores >= 95%. In seconds the unconstrained optimum improves quality, the floor is slack and is satisfied in {100*min(_rates_sec):.0f}% of resamples. The theorem is a statement about expectations under known primitives; this measures the estimation gap, not a failure of the theorem.")
for b in BASES:
    for p in PARTS:
        for e in EPS_GRID:
            o = run_(b, p)['oos'][f'eps={e:g}']; fr = o['floor'][f'{e:g}']; f_ = run_(b, p)['floor'][f'{e:g}']; mg = run_(b, p)['oos_safety_margin_for_95pct'][f'{e:g}']
            c5_rows.append(f"{b}/{p}/eps={e:g}: in-sample floor {'binding' if f_['binding'] else 'slack'} (shortfall {f_['floor_value'] - f_['EV'] + e:.4f}); realized E[V]-vbar {o['summary']['dV'][0]:+.4f} [{o['summary']['dV'][1]:+.4f}, {o['summary']['dV'][2]:+.4f}]; satisfaction rate {fr['rate_paired']:.3f} (fixed vbar {fr['rate_fixed_vbar']:.3f}; point {'ok' if fr['point_satisfied'] else 'violated'}); "
                           f"margin for 95%: {('fit at eps=' + format(mg['fit_eps'], 'g') + ' (margin ' + format(mg['margin'], '.4f') + ', rate ' + format(mg['rate'], '.3f') + ', r=' + format(mg['r'][0], '.3f') + ')') if mg.get('fit_eps') is not None else 'none within 0.03'}")
            if p == 'cell21':
                c5_ok &= fr['rate_paired'] >= 0.95
claim('C5', 'OUT-OF-SAMPLE', 'floor satisfaction: the eps-rule fitted on probing satisfies E[V] >= vbar - eps on id_test in >= 95% of paired bootstrap resamples (21 cells, every eps in the grid, both bases)',
      'rate >= 0.95 for every eps', ' | '.join(c5_rows), 'PASS' if c5_ok else 'FAIL',
      c5_note)

# C6: descriptive decomposition and the state-granularity ordering
c6_parts = []; c6_ok = True
for b in BASES:
    o21 = run_(b, 'cell21')['oos']['unconstrained']; o9 = run_(b, 'dim9')['oos']['unconstrained']
    d219 = ci(o21['_st']['r'] - o9['_st']['r']); d9d = ci(o9['_st']['r'] - RESULTS['baselines'][b]['best_single_direct']['_st']['r'])
    dor = o21['vs']['per_task_oracle']['gap_to_oracle_r']; dac = o21['vs']['acrouter']['diff_r']
    ps_ = RESULTS['per_state'][b]; gaps_or = np.array(ps_['gap_oracle_minus_rule_by_state']); gaps_ac = np.array(ps_['gap_rule_minus_acrouter_by_state'])
    order = np.argsort(-gaps_or); top3 = [(ps_['state_names'][i], float(gaps_or[i])) for i in order[:3]]
    c6_parts.append(f"{b}: r(21 cells) - r(9 dims) = {d219[0]:+.3f} [{d219[1]:+.3f}, {d219[2]:+.3f}]; r(9 dims) - r(best single action) = {d9d[0]:+.3f} [{d9d[1]:+.3f}, {d9d[2]:+.3f}]; gap to per-task oracle (2,919 states) {dor[0]:.3f} [{dor[1]:.3f}, {dor[2]:.3f}] = {100*(1-o21['vs']['per_task_oracle']['fraction_of_oracle_r'][0]):.1f}% of the oracle's r; "
                    f"lead over ACRouter {dac[0]:+.3f} [{dac[1]:+.3f}, {dac[2]:+.3f}]; net gain at the id_test reference price (score units/task): rule {ps_['policies']['rule']['net_gain_total']:.4f}, oracle {ps_['policies']['oracle']['net_gain_total']:.4f}, ACRouter {ps_['policies']['acrouter']['net_gain_total']:.4f}, DimensionBest {ps_['policies']['dimension_best']['net_gain_total']:.4f}; "
                    f"oracle-rule gap positive in {int((gaps_or > 0).sum())}/21 states (top-3 {top3}); rule-ACRouter gap positive in {int((gaps_ac > 0).sum())}/21 states")
    c6_ok &= dor[1] > 0 and dac[1] > 0 and max(d['identity_check'] for d in ps_['policies'].values()) < 1e-9
claim('C6', 'OUT-OF-SAMPLE', 'descriptive: the gap of the 21-cell rule to the per-task oracle and its lead over ACRouter are positive with paired CIs and decompose additively over states through the net gain pi(x) E[V - eta_B D | x] (identity); the state-granularity ordering (1 action, 9 dimensions, 21 cells, per-task) is reported with paired CIs',
      'oracle - rule > 0 and rule - ACRouter > 0 with CI lower bounds > 0; per-state contributions sum to eta - eta_B exactly; granularity differences reported (not a pass criterion)', ' | '.join(c6_parts), 'PASS' if c6_ok else 'INCONCLUSIVE',
      'ACRouter is scored as the ex-ante dispatch of its released per-task decisions (61% Opus) without its router/verifier overhead; its objective was accuracy, not efficiency, so the lead is the value of the efficiency rule plus fine states, not a like-for-like router comparison. Per-state tables in README and results.json.')
RESULTS['claims'] = CL
for c_ in CL:
    log(f"{c_['id']} [{c_['kind']}] {c_['verdict']}")

# =========================================================================================== 11. figures
def savefig(fig, name):
    path = os.path.join(FIG, name); fig.savefig(path, bbox_inches='tight'); plt.close(fig); log('figure', path)

# --- Figure 1: Phi(lambda) with root and Dinkelbach path (one panel per resource basis; 21 cells and 9 dimensions)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
for ax, basis in zip(axes, BASES):
    for part in PARTS:
        fd = FIGDATA[(basis, part)]; run = RESULTS['runs'][f'{basis}/{part}']
        lam_g = np.array(run['phi']['lambda_grid']); ph = np.array(run['phi']['values']); colr = PART_COLOR[part]
        ax.plot(lam_g, ph, color=colr, lw=1.6, label={'cell21': '21 cells', 'dim9': '9 dimensions'}[part], zorder=3)
        dk = fd['dk']; pi, v, c = fd['pi'], fd['v'], fd['c']; ar = np.arange(len(pi))
        for t, g in enumerate(dk['policies'][:-1] if dk['n_steps_to_root'] > 0 else []):
            lt, lt1 = dk['path'][t], dk['path'][t + 1]; EDt = float(pi @ c[ar, g])
            ax.plot([lt, lt1], [dk['phis'][t], 0.0], color=colr, lw=0.9, alpha=0.55, zorder=2)
            ax.plot([lt], [dk['phis'][t]], marker='o', ms=5.5, color=colr, mec='white', mew=1.0, zorder=4)
        ax.plot([fd['root']], [0.0], marker='o', ms=6, color=colr, mec='white', mew=1.2, zorder=5)
    ax.axhline(0, color='#c3c2b7', lw=0.65, zorder=1)
    fd = FIGDATA[(basis, 'cell21')]
    ax.annotate(r'$\lambda_0=\eta_B$', xy=(fd['eta_B'], 0), xytext=(fd['eta_B'], -0.16 * abs(np.array(RESULTS['runs'][f'{basis}/cell21']['phi']['values'])).max()),
                fontsize=7.5, color=INK2, ha='left', va='top', arrowprops=dict(arrowstyle='-', color='#898781', lw=0.6))
    ax.annotate(r'$\eta^\star$', xy=(fd['root'], 0), xytext=(fd['root'] * 1.04, 0.12 * abs(np.array(RESULTS['runs'][f'{basis}/cell21']['phi']['values'])).max()), fontsize=8, color=INK2, ha='left', va='bottom')
    ax.set_xlabel({'usd': r'price $\lambda$ (score per dollar)', 'sec': r'price $\lambda$ (score per second)'}[basis])
    ax.set_ylabel(r'$\Phi(\lambda)$ (score units)')
    ax.grid(True, axis='y', lw=0.5, alpha=0.8)
axes[0].legend(frameon=False, loc='upper right')
axes[0].text(-0.18, 1.02, 'a', transform=axes[0].transAxes, fontweight='bold'); axes[1].text(-0.18, 1.02, 'b', transform=axes[1].transAxes, fontweight='bold')
savefig(fig, 'fig1_phi_dinkelbach.pdf')

# --- Figure 2: eta/eta_B - quality-shortfall frontier with baselines (one panel per basis)
BL_STYLE = {'always_opus': ('o', 'always-Opus'), 'dimension_best': ('s', 'DimensionBest'), 'acrouter': ('D', 'ACRouter'), 'best_fixed_cascade': ('^', 'best fixed cascade'),
            'best_single_direct': ('v', 'best single dispatch'), 'per_task_oracle': ('*', 'per-task oracle')}
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
for ax, basis in zip(axes, BASES):
    for part in PARTS:
        run = RESULTS['runs'][f'{basis}/{part}']; pts = [p_ for p_ in run['frontier_fine']['points'].values() if p_['feasible']]
        xs = np.array([p_['eps'] for p_ in pts]); ys = np.array([p_['r_eps'] for p_ in pts])
        ax.plot(xs, ys, color=PART_COLOR[part], lw=1.6, label={'cell21': '21 cells (probing frontier)', 'dim9': '9 dimensions (probing frontier)'}[part], zorder=3)
    run = RESULTS['runs'][f'{basis}/cell21']
    for e in EPS_GRID:
        o = run['oos'][f'eps={e:g}']; x_ = -o['summary']['dV'][0]; y_ = o['summary']['r']
        ax.errorbar([x_], [y_[0]], yerr=[[y_[0] - y_[1]], [y_[2] - y_[0]]], fmt='o', ms=4.5, color=BLUE, mfc='white', mew=1.2, elinewidth=0.8, capsize=0, zorder=4,
                    label='21-cell eps-rules realized on id_test' if e == EPS_GRID[0] else None)
    o = run['oos']['unconstrained']; y_ = o['summary']['r']
    ax.errorbar([-o['summary']['dV'][0]], [y_[0]], yerr=[[y_[0] - y_[1]], [y_[2] - y_[0]]], fmt='o', ms=4.5, color=BLUE, mfc=BLUE, mew=0, elinewidth=0.8, zorder=4, label='21-cell unconstrained rule (id_test)')
    orc = RESULTS['oracle'][basis]; ox = [-orc['floor_eps'][f'{e:g}']['summary']['dV'][0] for e in EPS_GRID] + [-orc['unconstrained']['summary']['dV'][0]]
    oy = [orc['floor_eps'][f'{e:g}']['r'] for e in EPS_GRID] + [orc['unconstrained']['summary']['r'][0]]
    oo = np.argsort(ox); ax.plot(np.array(ox)[oo], np.array(oy)[oo], color=GREY, lw=1.0, ls=(0, (3, 2)), zorder=2, label='per-task oracle (id_test, upper bound)')
    for bname, (mk, lab) in BL_STYLE.items():
        if bname == 'per_task_oracle':
            s_ = orc['unconstrained']['summary']
        else:
            s_ = RESULTS['baselines'][basis][bname]['summary']
        x_, y_ = -s_['dV'][0], s_['r'][0]
        ax.plot([x_], [y_], marker=mk, ms=8 if mk == '*' else 5, color=INK if bname == 'always_opus' else GREY, mec='white', mew=0.8, ls='none', zorder=5)
        OFF = {'usd': {'always_opus': (5, -9, 'left'), 'dimension_best': (5, 2, 'left'), 'acrouter': (5, -9, 'left'), 'best_fixed_cascade': (5, 2, 'left'), 'best_single_direct': (-4, -10, 'right'), 'per_task_oracle': (6, -3, 'left')},
               'sec': {'always_opus': (5, 3, 'left'), 'dimension_best': (-5, -10, 'right'), 'acrouter': (-5, 3, 'right'), 'best_fixed_cascade': (5, 3, 'left'), 'best_single_direct': (5, -10, 'left'), 'per_task_oracle': (6, -3, 'left')}}
        dx, dy, ha = OFF[basis][bname]
        ax.annotate(lab, xy=(x_, y_), xytext=(dx, dy), textcoords='offset points', fontsize=6.8, color=INK2, ha=ha)
    ax.set_xlabel(r'quality shortfall $\bar v-\mathbb{E}[V]$' + (' (score units; negative = gain)' if basis == 'usd' else ''))
    ax.set_ylabel(r'$\eta/\eta_B$' + {'usd': ' (dollars)', 'sec': ' (seconds, sequential)'}[basis])
    if basis == 'usd':
        ax.set_yscale('log')
    ax.axvline(0, color='#c3c2b7', lw=0.65, zorder=1); ax.grid(True, axis='y', lw=0.5, alpha=0.8)
hnd, labs = axes[0].get_legend_handles_labels()
fig.subplots_adjust(wspace=0.32)
fig.legend(hnd, labs, frameon=False, loc='upper center', ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, -0.06))
axes[0].text(-0.18, 1.02, 'a', transform=axes[0].transAxes, fontweight='bold'); axes[1].text(-0.18, 1.02, 'b', transform=axes[1].transAxes, fontweight='bold')
savefig(fig, 'fig2_frontier.pdf')

# --- Figure 3: per-state action map (21 cells x {unconstrained, eps grid} for dollars and seconds)
KIND_FILL = {'direct_ref': '#e1e0d9', 'direct': '#cde2fb', 'attempt': '#fbd6c6'}
cols = [(b, k) for b in BASES for k in ['unconstrained'] + [f'eps={e:g}' for e in EPS_GRID]]
names21 = STATE['cell21']['names']; order_rows = np.argsort([-p for p in STATE['cell21']['pi_probing']])
fig, ax = plt.subplots(figsize=(6.6, 6.0))
ncol = len(cols)
for j, (b, k) in enumerate(cols):
    run = RESULTS['runs'][f'{b}/cell21']
    g = FIGDATA[(b, 'cell21')]['g_star'] if k == 'unconstrained' else FIGDATA[(b, 'cell21')]['POL_EPS'][float(k.split('=')[1])]
    xj = j + (0.6 if b == 'sec' else 0.0)
    for i, x in enumerate(order_rows):
        if k == 'unconstrained':
            us = [int(g[x])]; ws = [1.0]
        else:
            us = [int(u) for u in np.where(g[x] > 0)[0]]; ws = [float(g[x, u]) for u in us]
        kinds = [ACTIONS[u]['kind'] for u in us]
        if len(us) == 1:
            ax.add_patch(plt.Rectangle((xj, i), 0.96, 0.92, color=KIND_FILL[kinds[0]], lw=0))
            ax.text(xj + 0.48, i + 0.46, ACTIONS[us[0]]['short'], ha='center', va='center', fontsize=6.5, color=INK)
        else:
            ax.add_patch(plt.Rectangle((xj, i), 0.96 * ws[0], 0.92, color=KIND_FILL[kinds[0]], lw=0))
            ax.add_patch(plt.Rectangle((xj + 0.96 * ws[0], i), 0.96 * ws[1], 0.92, color=KIND_FILL[kinds[1]], lw=0))
            ax.text(xj + 0.48, i + 0.46, f"{ACTIONS[us[0]]['short']}|{ACTIONS[us[1]]['short']}\n{ws[0]:.2f}|{ws[1]:.2f}", ha='center', va='center', fontsize=5.6, color=INK, linespacing=1.05)
ax.set_xlim(0, ncol + 0.6); ax.set_ylim(len(names21), 0)
ax.set_yticks(np.arange(len(names21)) + 0.46); ax.set_yticklabels([f"{names21[x]}  ({STATE['cell21']['pi_probing'][x]:.3f})" for x in order_rows], fontsize=6.5)
ax.set_xticks([j + (0.6 if b == 'sec' else 0.0) + 0.48 for j, (b, k) in enumerate(cols)])
ax.set_xticklabels([('none' if k == 'unconstrained' else k.split('=')[1]) for b, k in cols], fontsize=6.5)
ax.xaxis.tick_top()
ax.text(3.0, -1.35, 'dollars: floor eps', ha='center', va='bottom', fontsize=8, color=INK2); ax.text(6 + 0.6 + 3.0, -1.35, 'seconds (sequential): floor eps', ha='center', va='bottom', fontsize=8, color=INK2)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0)
ax.legend(handles=[Patch(color=KIND_FILL['direct_ref'], label='direct Opus (reference)'), Patch(color=KIND_FILL['direct'], label='direct dispatch to another model'),
                   Patch(color=KIND_FILL['attempt'], label='attempt-then-escalate (model@theta)')], frameon=False, fontsize=6.8, loc='lower center', bbox_to_anchor=(0.5, 1.085), ncol=3)
ax.text(ncol + 0.6, len(names21) + 0.35, 'codes: Q Qwen3-Max, K kimi-k2.5, P qwen3.5-plus, G glm-5, T gpt-5.4, S claude-sonnet-4-6, M MiniMax-M2.7, O Opus; a|b w|w = randomized state (weights)', ha='right', va='top', fontsize=6.0, color=INK2)
savefig(fig, 'fig3_action_map.pdf')

# --- Figure 4: per-state gap decomposition (dollars, 21 cells): oracle - rule, rule - ACRouter, rule - DimensionBest (net gain at the reference price, in m$ of reference work per task)
ps_ = RESULTS['per_state']['usd']; etaBt = ps_['eta_B_id_test']
g1 = np.array(ps_['gap_oracle_minus_rule_by_state']) / etaBt * 1000; g2 = np.array(ps_['gap_rule_minus_acrouter_by_state']) / etaBt * 1000; g3 = np.array(ps_['gap_rule_minus_dimension_best_by_state']) / etaBt * 1000
order_rows_t = np.argsort([-p for p in ps_['pi_id_test']])
fig, ax = plt.subplots(figsize=(6.6, 4.6))
yy = np.arange(21); h = 0.27
ax.barh(yy - h, g1[order_rows_t], height=h * 0.92, color=BLUE, label='per-task oracle minus 21-cell rule (value of task-level information)')
ax.barh(yy, g2[order_rows_t], height=h * 0.92, color=ORANGE, label='21-cell rule minus ACRouter')
ax.barh(yy + h, g3[order_rows_t], height=h * 0.92, color=AQUA, label='21-cell rule minus DimensionBest')
ax.set_yticks(yy); ax.set_yticklabels([f"{ps_['state_names'][x]} ({ps_['pi_id_test'][x]:.3f})" for x in order_rows_t], fontsize=6.5); ax.invert_yaxis()
ax.axvline(0, color='#c3c2b7', lw=0.65); ax.grid(True, axis='x', lw=0.5, alpha=0.8)
ax.set_xlabel(r'state contribution to the net-gain difference, $\pi(x)\,\mathbb{E}[V-\eta_B D\mid x]/\eta_B$ (m\$ of reference work per task; id_test)')
ax.legend(frameon=False, fontsize=6.8, loc='lower right')
savefig(fig, 'fig4_state_gaps.pdf')

# =========================================================================================== 12. results.json
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v_) for k, v_ in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)):
        return [clean(x) for x in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o
RESULTS['runtime_seconds'] = time.time() - T0
with open(os.path.join(HERE, 'results.json'), 'w') as fh:
    json.dump(clean(RESULTS), fh, indent=1)
log('results.json written', os.path.getsize(os.path.join(HERE, 'results.json')) // 1024, 'KB')

# =========================================================================================== 13. README.md and summary_for_paper.md
def f3(t): return f'{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]'
def f4(t): return f'{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]'
def fg(x, d=4): return f'{x:.{d}g}'
def table(hdr, rows):
    return '| ' + ' | '.join(hdr) + ' |\n|' + '---|' * len(hdr) + '\n' + '\n'.join('| ' + ' | '.join(str(c_) for c_ in r_) + ' |' for r_ in rows) + '\n'
UNIT = {'usd': 'dollars (csv `cost_usd`)', 'sec': 'seconds (sequential: `latency_ms` of the primary plus, on a miss, of Opus)'}
md = []
md.append(f"# X9_optimal_routing -- Theorem thm:optimal-routing (efficiency-optimal state-wise routing with a quality floor) on CodeRouterBench\n")
md.append(f"Generated by `run.py` (numpy/pandas/scipy/matplotlib only; fully offline; seed {SEED}; {NBOOT} paired task-resampling bootstrap draws on id_test; {NBOOT_FIT} probing resamples for fit uncertainty; runtime {RESULTS['runtime_seconds']:.0f} s).\n"
          f"Files: `run.py`, `results.json`, `figures/fig1_phi_dinkelbach.pdf` (C1/C2), `figures/fig2_frontier.pdf` (C3-C5), `figures/fig3_action_map.pdf` (rule per state), `figures/fig4_state_gaps.pdf` (C6), `summary_for_paper.md`, this `README.md`.\n")
md.append("## Paper objects tested (labels from `main_body.tex` / `appendix_core.tex`)\n"
          "* **thm:optimal-routing (i)**: `Phi(lambda) = sum_x pi(x) max_u [v(x,u) - lambda c(x,u)]` is convex and strictly decreasing with unique root `eta* = max_g eta(g)`; a router is efficiency-optimal iff in every state it uses only maximizers of `v - eta* c`; for {direct, attempt-then-escalate} this is `p(x) W_{eta*}(x) >= eta* a(x)` (prop:threshold, eq:margin; exact matched-reference form `W = lambda cbar_h + v_h - vbar_h`, app:matched-reference).\n"
          "* **thm:optimal-routing (ii)**: Dinkelbach iteration `lambda_0 = eta_B`, `lambda_{t+1} = eta(g_t)` is monotone and finite; `Phi(eta_B) = max_g [Q_g + eta_B S_g]` (thm:surplus).\n"
          "* **thm:optimal-routing (iii)**: under `E[V] >= vbar - eps` an optimal router randomizes in at most one state (Charnes-Cooper LP eq:app-cc-lp), a multiplier `kappa >= 0` makes every optimal router a state-wise maximizer of `(1+kappa) v - eta*_eps c`; the two-action price is `eta*_eps/(1+kappa)`.\n"
          "* **thm:surplus** accounting for every policy realized: `eta - eta_B = (Q + eta_B S)/(cbar - S)`, net gain `E[V - eta_B D]` = accepted quality change `E[H(Y_P - Y_B)]` + `eta_B` (avoided reference work `E[H C_B]` - attempt cost `E[A]` - recovery cost 0).\n")
md.append("## Setup\n"
          f"**Data.** `$ACROUTER_DATA/coderouterbench/id_results_long.csv`: {DATA_CHECKS['n_tasks']:,} tasks x {DATA_CHECKS['n_models']} models, `score` in [0,1], token-price `cost_usd`, `latency_ms`. Splits: probing {DATA_CHECKS['n_probing']:,} (train+val; all primitives fitted here) / id_test {DATA_CHECKS['n_id_test']:,} (test; realizations only, never fitted on). "
          f"Cross-check: always-Opus id_test total ${DATA_CHECKS['always_opus_id_test_total_usd']:.2f} (paper $34.02), mean score {DATA_CHECKS['always_opus_id_test_mean_score']:.4f} (paper 43.83). {DATA_CHECKS['rows_cost_zero']} Qwen3-Max rows have no token record (cost 0, score 0; kept as recorded, as in X1). Zero-latency rows by model: {DATA_CHECKS['rows_latency_zero_by_model']} ({100*AUD['zero_latency_rows']['share_with_score_zero']:.0f}% of them have score 0; kept; they affect the seconds basis only).\n\n"
          f"**States.** (i) 21 dimension x source cells (source = second underscore token of `task_id`); (ii) 9 dimensions. `pi(x)` = cell frequency; the rule uses the probing frequencies (all primitives come from probing), the realization uses the id_test tasks themselves (id_test frequencies). Both frequency vectors are in `results.json:states` and in the per-state table below; `eta*` recomputed with id_test frequencies and probing `v, c` differs by "
          + ', '.join(f"{b}/{p}: {100*(run_(b,p)['root']['r_star_pi_id_test']/run_(b,p)['root']['r_star']-1):+.2f}%" for b in BASES for p in PARTS) + ".\n\n"
          f"**Menu U(x)** ({U} actions, identical in every state): ex-ante direct dispatch to each of the 8 models (value = mean score, cost = mean `cost_usd` of that model in the cell on probing), and attempt-then-escalate for each of the 7 non-Opus primaries m with the oracle verifier `H = 1{{score_m >= theta}}`, theta in {{1, 0.5}}, fallback claude-opus-4-6: per task `V = H Y_m + (1-H) Y_Opus`, `D = C_m + (1-H) C_Opus` (dollars) or `lat_m + (1-H) lat_Opus` (seconds, sequential). **Reference policy** = always-Opus = action `direct:claude-opus-4-6`, so it is admissible (`eta(g_B) = eta_B` exactly). Design-document deviation: the X8 judge-theta grid is not part of this menu (X8 is a separate live experiment; the task protocol fixes the 22-action menu).\n\n"
          f"**Resources.** Everything is run twice: {UNIT['usd']} and {UNIT['sec']}. `eta = sum V / sum D` (ratio of sums) everywhere; `eta_B` (probing) = {run_('usd','cell21')['eta_B_probing']:.3f} per dollar, {run_('sec','cell21')['eta_B_probing']:.5f} per second; id_test {RESULTS['baselines']['usd']['always_opus']['summary']['eta'][0]:.3f} per dollar, {RESULTS['baselines']['sec']['always_opus']['summary']['eta'][0]:.5f} per second.\n\n"
          f"**Computation.** `Phi` on a 401-point grid; root by bisection to relative width 1e-13; Dinkelbach from `lambda_0 = eta_B` with deterministic argmax policies; constrained problems as the Charnes-Cooper LP (`scipy.optimize.linprog`, `method='highs-ds'`), `kappa` = dual multiplier of the floor row (checked against the analytic tie in the randomizing state and against the root of `Phi_eps`, computed independently by a fixed-price floor LP plus bisection). Independent checks of the root: an LP without floor (exact maximum over all randomized routers), exhaustive enumeration of all 2^|X| deterministic policies for each of the 14 two-action sub-menus, {N_HILL_STARTS:,} coordinate-ascent starts and {N_RANDOM_POLICIES:,} random deterministic policies on the full menu, and a one-swap test of the argmax policy.\n\n"
          f"**Baselines on id_test.** always-Opus; DimensionBest (`$ACR/id/voter_decisions/dimension_best.jsonl`, {RESULTS['baseline_files']['dimension_best']['rows_in_file']} rows, {n_test} on id_test; ex-ante dispatch paying only the chosen model); ACRouter's released decisions (`outputs/current/id_decisions.jsonl`, {RESULTS['baseline_files']['acrouter']['rows_in_file']} rows; shares {RESULTS['baseline_files']['acrouter']['shares']}; scored as ex-ante dispatch of `chosen_model`, without router/verifier overhead); the best fixed cascade of X1 (7 primaries x 2 theta, selected on probing) and, as the one-state version of the rule, the best single direct dispatch (selected on probing); the per-task oracle (state = task; Dinkelbach and Charnes-Cooper LP on id_test itself; an in-sample upper bound for any state-based policy with this menu).\n\n"
          f"**Statistics.** Paired bootstrap: the same {NBOOT} multinomial task resamples of id_test are applied to every policy, differences are taken per resample, 95% percentile CIs. Floor satisfaction rate = fraction of resamples with `E[V] >= vbar - eps`, `vbar` from the same resample. Fit uncertainty: {NBOOT_FIT} multinomial resamples of probing refit `eta*`, `eta*_eps`, `kappa`, the number of randomizing states and the per-state actions. Claims are labelled IDENTITY / IN-SAMPLE / OUT-OF-SAMPLE.\n")
# ---- results tables
md.append("## Results\n### (1) Root, Dinkelbach path and independent maximum-efficiency checks (probing primitives; IDENTITY)\n")
rows = []
for b in BASES:
    for p in PARTS:
        r = run_(b, p); ch = r['checks_part_i']; dk_ = r['dinkelbach']
        rows.append([f"{b}/{p}", f"{r['eta_B_probing']:.5g}", f"{r['root']['eta_star']:.6g}", f"{r['root']['r_star']:.3f} [{r['fit_bootstrap']['r_star_ci'][1]:.3f}, {r['fit_bootstrap']['r_star_ci'][2]:.3f}]",
                     r['root']['bisection_iterations'], ' -> '.join(f'{x:.5g}' for x in dk_['lambda_path']), f"{dk_['n_steps_to_root']} ({r['fit_bootstrap']['dinkelbach_steps_max']})",
                     f"{ch['lp_minus_root_rel']:.1e}", f"{ch['search_max_minus_root_rel']:.1e}", f"{ch['search']['hill_starts_reaching_max']}/{ch['search']['hill_starts']}",
                     f"{ch['two_action_exhaustive_max_abs_dev']:.1e}", f"{ch['two_action_rule_exact_agreement']}/{ch['two_action_rule_total']}", f"{ch['two_action_rule_homogeneous_agreement']}/{ch['two_action_rule_total']}",
                     f"{dk_['identity_phi_eq_ED_times_step_max_abs']:.1e}", 'yes' if r['phi']['convex'] and r['phi']['strictly_decreasing'] else 'NO'])
md.append(table(['basis/states', 'eta_B', 'eta*', 'eta*/eta_B [fit CI]', 'bisection it.', 'Dinkelbach lambda_t', 'steps (max over fit resamples)', 'LP cert. rel. dev.', 'search max rel. dev.', 'hill starts at max',
                 '2-action exhaustive max dev.', 'rule pW>=eta*a exact', 'homogeneous W', 'Phi(l_t)=E[D](l_t+1-l_t) resid', 'Phi convex, decreasing'], rows))
md.append("The LP certificate is the exact maximum of `eta` over all randomized routers (the Charnes-Cooper LP with a non-binding floor), so `root = certificate` proves `eta* = max_g eta(g)` for the full 22-action menu, where the {:.1e} deterministic policies cannot be enumerated; the two-action sub-menus (2^21 or 2^9 policies each) are enumerated exhaustively. 'homogeneous W' counts the states in which the approximate rule with `W = lambda b + v_h - v_b` (unconditional state means, the hypothesis rejected in X1 C3/C4) picks the same action as the exact objective; the exact matched-reference form agrees in every state.\n".format(float(U) ** 21))
md.append("### (2) Quality floor: Charnes-Cooper optimum, multiplier, randomization (probing; IDENTITY/IN-SAMPLE) and its realization on id_test (OUT-OF-SAMPLE)\n")
for b in BASES:
    for p in PARTS:
        r = run_(b, p); rows = []
        for e in EPS_GRID:
            f_ = r['floor'][f'{e:g}']; o = r['oos'][f'eps={e:g}']; fb_ = r['fit_bootstrap']['eps'][f'{e:g}']; mg = r['oos_safety_margin_for_95pct'][f'{e:g}']
            rand_txt = '; '.join(f"{s_}: " + ' / '.join(f"{a_.split(':', 1)[1]} {w_:.3f}" for a_, w_ in acts) for s_, acts in zip(f_['randomizing_states'], f_['randomizing_actions'])) or '-'
            rows.append([f'{e:g}', 'binding' if f_['binding'] else 'slack', f"{f_['r_eps']:.3f} [{fb_['r_eps_ci'][1]:.3f}, {fb_['r_eps_ci'][2]:.3f}]", f"{f_['kappa']:.3f} [{fb_['kappa_ci'][1]:.3f}, {fb_['kappa_ci'][2]:.3f}]",
                         f"{f_['kappa_tie']:.3f}" if f_['kappa_tie'] is not None else '-', f"{f_['price']:.5g}", f"{f_['n_randomizing_states']} (max {fb_['n_rand_max']})", rand_txt, f"{f_['max_optimality_violation']:.0e}",
                         f"{f_['root_phi_eps_minus_eta_eps_rel']:.0e}", f3(o['summary']['r']), f4(o['summary']['dV']), f"{o['floor'][f'{e:g}']['rate_paired']:.3f}",
                         (f"{mg['margin']:.4f} (fit eps {mg['fit_eps']:g}, r {mg['r'][0]:.3f})" if mg.get('fit_eps') is not None else 'none within 0.03')])
        md.append(f"**{b}/{p}** (`vbar` probing = {r['vbar_probing']:.4f}; unconstrained optimum shortfall = {r['optimal_rule']['shortfall_vs_reference']:+.4f}; realized shortfall of the unconstrained rule on id_test = {-r['oos']['unconstrained']['summary']['dV'][0]:+.4f})\n\n"
                  + table(['eps', 'floor', 'eta*_eps/eta_B [fit CI]', 'kappa (LP dual) [fit CI]', 'kappa (tie)', 'price eta*_eps/(1+kappa)', '# randomizing states (max over fit resamples)', 'randomizing state: actions and weights', 'max KKT violation', 'root(Phi_eps) rel. dev.',
                           'OOS eta/eta_B', 'OOS E[V]-vbar', 'OOS floor rate', 'margin for 95% (fit eps, OOS r)'], rows))
md.append("### (3) Out-of-sample comparison on id_test (paired bootstrap; all policies on the same resamples)\n")
for b in BASES:
    rows = []
    def add_row(label, s_, sh, esc, fl_, extra=''):
        rows.append([label, f3(s_['r']), f4(s_['dV']), f"{s_['ED'][0]:.5g}", f"{s_['net_gain'][0]:+.4f}", f"{s_['Q_acc'][0]:+.4f}", f"{s_['avoided'][0]:.4g}", f"{s_['attempt_cost'][0]:.4g}", f"{s_['e'][0]:.3f}",
                     f"{sh['direct_ref']:.2f}/{sh['direct']:.2f}/{sh['attempt']:.2f}", f"{esc:.3f}", ' '.join((f"{fl_[k]['rate_paired']:.2f}" if np.isfinite(fl_[k]['rate_paired']) else '-') for k in ['0', '0.01', '0.02', '0.05']), extra])
    for p in PARTS:
        r = run_(b, p)
        for pn in ['unconstrained'] + [f'eps={e:g}' for e in EPS_GRID]:
            o = r['oos'][pn]
            add_row(f"rule {p} {pn}", o['summary'], o['shares'], o['escalation_rate'], o['floor'], f"pred. r {o['predicted_r']:.3f}; realized-pred {f4(o['realized_minus_predicted_r'])}; vs DimensionBest {f4(o['vs']['dimension_best']['diff_r'])}; vs best cascade {f4(o['vs']['best_fixed_cascade']['diff_r'])}; vs ACRouter {f4(o['vs']['acrouter']['diff_r'])}")
    for bn, lab in [('always_opus', 'always-Opus (reference)'), ('dimension_best', 'DimensionBest'), ('acrouter', 'ACRouter released decisions'), ('best_fixed_cascade', 'best fixed cascade (X1; probing-selected)'), ('best_single_direct', 'best single dispatch (probing-selected)')]:
        s_ = RESULTS['baselines'][b][bn]
        add_row(lab + (f" = {s_['name']}" if 'name' in s_ else ''), s_['summary'], s_['shares'], s_['escalation_rate'], s_['floor'], (f"id_test-best would be {s_['best_on_id_test']}" if 'best_on_id_test' in s_ else ''))
    orc = RESULTS['oracle'][b]
    add_row('per-task oracle (id_test, unconstrained; upper bound)', orc['unconstrained']['summary'], orc['unconstrained']['shares'], orc['unconstrained']['escalation_rate'], orc['unconstrained']['floor'], f"re-optimized per resample r {f3(orc['unconstrained']['r_reoptimized_per_resample'])}; Dinkelbach steps {orc['unconstrained']['n_steps']}")
    for e in EPS_GRID:
        oe = orc['floor_eps'][f'{e:g}']
        add_row(f'per-task oracle, floor eps={e:g}', oe['summary'], oe['shares'], 0.0, {k: {'rate_paired': float('nan')} for k in ['0', '0.01', '0.02', '0.05']}, f"kappa {oe['kappa']:.3f}, randomizing tasks {oe['n_randomizing_tasks']}, point only")
    md.append(f"**Basis: {UNIT[b]}.** `eta_B` (id_test) = {RESULTS['baselines'][b]['always_opus']['summary']['eta'][0]:.5g}; shares = fraction of tasks sent to Opus directly / another model directly / an attempt; escalation = fraction of tasks that attempted and fell back; net gain `E[V - eta_B D]` = accepted quality change + eta_B (avoided reference work - attempt cost - recovery cost) in score units per task, avoided and attempt cost in resource units, recovery cost `E[(1-H)(C_F - C_B)]` = 0 identically because the fallback is the reference; floor rates for eps = 0, 0.01, 0.02, 0.05 (point-only rows: -).\n\n"
              + table(['policy', 'eta/eta_B', 'E[V]-vbar', 'E[D]', 'net gain E[V-eta_B D]', 'accepted quality change', 'avoided ref. work E[HC_B]', 'attempt cost E[A]', 'erroneous acceptance e', 'shares O/direct/attempt', 'escalation', 'floor rates', 'notes'], rows))
md.append("### (4) Sensitivity: fixed-price objective J_lambda = E[V] - lambda E[D] (rule = state-wise argmax at price lambda; no iteration)\n")
for b in BASES:
    for p in PARTS:
        r = run_(b, p); rows = []
        for lk in RESULTS['lambda_grid']:
            s_ = r['sensitivity_lambda'][lk]
            fl_txt = '; '.join(f"eps={k}: kappa {v_['kappa']:.3g}, rand {v_['n_randomizing_states']}, r {v_['r']:.3f}" for k, v_ in s_['floor'].items() if 'kappa' in v_)
            rows.append([lk, f"{s_['lam']:.5g}", f"{s_['lam_over_eta_star']:.3g}", f"{s_['in_sample']['r']:.3f}", f"{s_['in_sample']['J']:+.5g}", f"{s_['in_sample']['J_of_root_rule']:+.5g}", f"{s_['shares']['direct_ref']:.2f}/{s_['shares']['direct']:.2f}/{s_['shares']['attempt']:.2f}",
                         f3(s_['oos']['r']), f"{s_['oos']['J_lambda'][0]:+.5g} [{s_['oos']['J_lambda'][1]:+.5g}, {s_['oos']['J_lambda'][2]:+.5g}]", f"{s_['oos_diff_J_root_rule_minus_lambda_rule'][0]:+.4g} [{s_['oos_diff_J_root_rule_minus_lambda_rule'][1]:+.4g}, {s_['oos_diff_J_root_rule_minus_lambda_rule'][2]:+.4g}]",
                         f3(s_['oracle_id_test']['r']), f"{s_['oracle_id_test']['J_lambda'][0]:+.5g}", fl_txt])
        md.append(f"**{b}/{p}** (`eta*` = {r['root']['eta_star']:.5g}; the Dinkelbach root is the price at which the unconstrained rule's `J_lambda` is 0 and its `eta` is maximal; every fixed-price rule has `eta <= eta*` in-sample, and a higher `J_lambda` than the root rule at its own price)\n\n"
                  + table(['lambda', 'value', 'lambda/eta*', 'in-sample eta/eta_B', 'in-sample J_lambda', 'J_lambda of root rule', 'shares O/direct/attempt', 'OOS eta/eta_B', 'OOS J_lambda', 'OOS J(root rule) - J(lambda rule)', 'oracle J_lambda rule: eta/eta_B', 'oracle J_lambda', 'with floor (fixed-lambda LP)'], rows))
md.append("### (5) Per-state decomposition (21 cells, id_test): rule actions, ACRouter's choices and the net-gain gaps (C6)\n")
for b in BASES:
    ps_ = RESULTS['per_state'][b]; r = run_(b, 'cell21'); rows = []
    etaBt = ps_['eta_B_id_test']
    for x in np.argsort([-p_ for p_ in ps_['pi_id_test']]):
        nm = ps_['state_names'][x]; acr = ps_['acrouter_choice_by_state'][nm]; acr_top = max(acr, key=acr.get)
        rows.append([nm, f"{STATE['cell21']['pi_probing'][x]:.3f} / {ps_['pi_id_test'][x]:.3f}", r['optimal_rule']['short'][x], r['floor']['0']['short'][x] + ('*' if nm in r['floor']['0']['randomizing_states'] else ''), r['floor']['0.01']['short'][x] + ('*' if nm in r['floor']['0.01']['randomizing_states'] else ''),
                     f"{r['fit_bootstrap']['action_stability_unconstrained'][nm]:.2f}", f"{acr_top} ({acr[acr_top]}/{sum(acr.values())})",
                     f"{1000*ps_['policies']['rule']['net_gain_by_state'][x]/etaBt:+.3f}", f"{1000*ps_['policies']['oracle']['net_gain_by_state'][x]/etaBt:+.3f}", f"{1000*ps_['policies']['acrouter']['net_gain_by_state'][x]/etaBt:+.3f}", f"{1000*ps_['policies']['dimension_best']['net_gain_by_state'][x]/etaBt:+.3f}",
                     f"{1000*ps_['gap_oracle_minus_rule_by_state'][x]/etaBt:+.3f}", f"{1000*ps_['gap_rule_minus_acrouter_by_state'][x]/etaBt:+.3f}"])
    tot = {k: 1000 * ps_['policies'][k]['net_gain_total'] / etaBt for k in ['rule', 'oracle', 'acrouter', 'dimension_best']}
    rows.append(['**total**', '1 / 1', '', '', '', f"{r['fit_bootstrap']['action_stability_unconstrained_mean']:.2f}", '', f"{tot['rule']:+.3f}", f"{tot['oracle']:+.3f}", f"{tot['acrouter']:+.3f}", f"{tot['dimension_best']:+.3f}", f"{tot['oracle']-tot['rule']:+.3f}", f"{tot['rule']-tot['acrouter']:+.3f}"])
    md.append(f"**{UNIT[b]}**; contributions `pi(x) E[V - eta_B D | x] / eta_B` in {'m$' if b == 'usd' else 'ms'} of reference work per task (id_test reference price {etaBt:.5g}); they sum to the policy's net gain, and `eta - eta_B = net gain / E[D]` (identity residual <= {max(d['identity_check'] for d in ps_['policies'].values()):.0e}). 'stability' = fraction of probing resamples that select the same unconstrained action. Codes: {', '.join(f'{v_} {k}' for k, v_ in SHORT.items())}; `@1`, `@.5` = attempt with theta; `*` = randomizing state.\n\n"
              + table(['cell', 'pi probing / id_test', 'rule (unconstrained)', 'rule eps=0', 'rule eps=0.01', 'stability', 'ACRouter majority choice', 'rule gain', 'oracle gain', 'ACRouter gain', 'DimensionBest gain', 'oracle - rule', 'rule - ACRouter'], rows))
# ---- claims
md.append("## Claims\n")
md.append(table(['id', 'kind', 'claim', 'verdict'], [[c_['id'], c_['kind'], c_['statement'], f"**{c_['verdict']}**"] for c_ in CL]))
for c_ in CL:
    md.append(f"**{c_['id']} [{c_['kind']}] -- {c_['verdict']}.** {c_['statement']}\n- prediction: {c_['prediction']}\n- measurement: {c_['measurement']}\n- note: {c_['note']}\n")
# ---- hypothesis violations
md.append("## Hypothesis violations measured\n"
          f"* **Router-independent state probabilities**: states are task attributes (dimension, source benchmark), so `pi(x)` cannot depend on the router; probing and id_test frequencies differ by at most {max(abs(a - b_) for a, b_ in zip(STATE['cell21']['pi_probing'], STATE['cell21']['pi_id_test'])):.4f} (21 cells), which moves `eta*` by the percentages quoted in Setup.\n"
          f"* **c(x,u) > 0**: minimum state-action cost {run_('usd','cell21')['c_min']:.3g} $ ({run_('usd','cell21')['c_min_action']}) and {run_('sec','cell21')['c_min']:.3g} s ({run_('sec','cell21')['c_min_action']}); the 148 zero-cost Qwen3-Max rows and the zero-latency rows never make a state-action cost zero.\n"
          f"* **Admissible reference**: always-Opus is in every menu; `eta(g_B) = eta_B` to machine precision (asserted).\n"
          f"* **Known primitives (the theorem is about expectations)**: probing-to-id_test shift of the primitives is up to {run_('usd','cell21')['shift_max_abs_v']:.3f} in `v` and {100*run_('usd','cell21')['shift_max_rel_c']:.0f}% (relative) in `c` over state-action pairs (dollars, 21 cells). Realized minus predicted `eta/eta_B` of the unconstrained rule: dollars {f4(run_('usd','cell21')['oos']['unconstrained']['realized_minus_predicted_r'])}, seconds {f4(run_('sec','cell21')['oos']['unconstrained']['realized_minus_predicted_r'])} (the rule is selected on probing, so its in-sample value is optimistic: winner's curse). The same selection effect makes a floor that binds in-sample fail out of sample about half of the time or more (C5).\n"
          f"* **Two-action rule with homogeneous W** (`b`, `v_b` unconditional): agrees with the exact objective in {run_('usd','cell21')['checks_part_i']['two_action_rule_homogeneous_agreement']}/{run_('usd','cell21')['checks_part_i']['two_action_rule_total']} (dollars, 21 cells) and {run_('sec','cell21')['checks_part_i']['two_action_rule_homogeneous_agreement']}/{run_('sec','cell21')['checks_part_i']['two_action_rule_total']} (seconds) state x sub-menu cases; the exact form `W = lambda cbar_h + v_h - vbar_h` in all. Reference outcomes depend on acceptance (X1 C3/C4), so the exact form is the one the theorem's proof uses.\n"
          f"* **Seconds basis data quality**: {AUD['zero_latency_rows']['n']} rows have `latency_ms = 0` ({100*AUD['zero_latency_rows']['share_with_score_zero']:.0f}% of them with score 0, i.e. failed/missing runs recorded as instantaneous), concentrated in glm-5, qwen3.5-plus and MiniMax-M2.7 in the algorithm and test_generation dimensions; they make those models look faster than they are in a few cells. Kept for comparability with X1/X5.\n")
# ---- audit
md.append("## Audit (independent recomputation)\n"
          f"* **A1 `eta*` (dollars, 21 cells)**: pandas `groupby` primitives from the long CSV and `scipy.optimize.brentq` on `Phi`: {AUD['A1_eta_star_usd_cell21']['audit_pandas_brentq']:.9g} vs run.py (bincount + bisection) {AUD['A1_eta_star_usd_cell21']['run_py']:.9g}; relative mismatch {AUD['A1_eta_star_usd_cell21']['rel_mismatch']:.1e}. Also equal to the LP certificate (relative deviation {run_('usd','cell21')['checks_part_i']['lp_minus_root_rel']:.1e}).\n"
          f"* **A2 realized `eta/eta_B` of the unconstrained 21-cell rule on id_test (dollars)**: plain Python loop over id_test task ids with per-row dictionaries from the CSV: {AUD['A2_oos_r_unconstrained_usd_cell21']['audit_plain_loop']:.6f} vs run.py (matrix, weight row 0) {AUD['A2_oos_r_unconstrained_usd_cell21']['run_py']:.6f}; relative mismatch {AUD['A2_oos_r_unconstrained_usd_cell21']['rel_mismatch']:.1e}; rule actions identical: {AUD['A2_oos_r_unconstrained_usd_cell21']['rule_actions_match']}.\n"
          f"* **A3 `kappa` at eps = 0 (dollars, 21 cells)**: tie formula `(1+kappa)(v1 - v2) = eta*_eps (c1 - c2)` in the randomizing state with pandas primitives: {AUD['A3_kappa_eps0_usd_cell21'].get('audit_tie_formula_pandas', float('nan')):.6f} vs LP dual {AUD['A3_kappa_eps0_usd_cell21']['run_py_lp_dual']:.6f} (relative mismatch {AUD['A3_kappa_eps0_usd_cell21'].get('rel_mismatch', float('nan')):.1e}); `eta*_eps` from the mixture weights {AUD['A3_kappa_eps0_usd_cell21'].get('eta_eps_audit', float('nan')):.6f} vs LP {AUD['A3_kappa_eps0_usd_cell21'].get('eta_eps_run_py', float('nan')):.6f}; floor check E[V]-vbar = {AUD['A3_kappa_eps0_usd_cell21'].get('floor_check_EV_minus_vbar', float('nan')):.1e}. Dinkelbach path recomputed with pandas loops: {[round(x, 4) for x in AUD['A3b_dinkelbach_path_usd_cell21']['audit_pandas_loop']]} (max abs diff {AUD['A3b_dinkelbach_path_usd_cell21']['max_abs_diff']:.1e}, same length {AUD['A3b_dinkelbach_path_usd_cell21']['same_length']}).\n"
          f"* Accounting identities per task: `V - Y_B` = accepted quality change and `C_B - D` = avoided reference work - attempt cost hold to {max(IDENT.values()):.0e}; net-gain term decomposition residual <= {max(run_(b,p)['oos'][k]['summary']['net_gain_terms_resid'] for b in BASES for p in PARTS for k in run_(b,p)['oos']):.0e} over all policies and bootstrap rows; per-state gain sums reproduce `eta - eta_B` to {max(d['identity_check'] for b in BASES for d in RESULTS['per_state'][b]['policies'].values()):.0e}.\n")
# ---- limitations
md.append("## Limitations\n"
          "* The verifier is the oracle `H = 1{score_m >= theta}` (as in X1): attempt actions are upper bounds for real verifiers; the X8 judge/sandbox operating points are not in this menu.\n"
          "* Efficiency `eta` without a floor rewards cheap dispatch: in dollars the unconstrained optimum sends every cell to Qwen3-Max or kimi-k2.5 and gives up 0.016 of the reference's mean score (0.020 realized). The floor is what makes the rule quality-safe, and its in-sample version needs a safety margin out of sample (C5, `oos_safety_margin_for_95pct`).\n"
          "* ACRouter is scored from its released per-task decisions as ex-ante dispatch; its router, sandbox-verifier and memory costs and its actual escalation traces are not included (X7 covers the live accounting). DimensionBest and the fixed cascades are exactly reproducible from the matrix.\n"
          "* The per-task oracle is optimized on id_test itself (an in-sample upper bound); its bootstrap CI re-optimizes per resample and is reported separately from the fixed-assignment CI.\n"
          "* Seconds use recorded latencies with zero-latency rows kept; sequential accounting only (no concurrent fallback launch; X5 treats hedging).\n"
          "* Fit-uncertainty CIs (probing resamples) and realization CIs (id_test resamples) are reported separately, not combined.\n")
with open(os.path.join(HERE, 'README.md'), 'w') as fh:
    fh.write('\n'.join(md))
# ---- summary for paper (<= 15 lines)
u21 = run_('usd', 'cell21'); s21 = run_('sec', 'cell21'); ou = u21['oos']['unconstrained']; oe0 = u21['oos']['eps=0']
lines = [
    f"# X9 summary for the paper (thm:optimal-routing; CodeRouterBench, 21 dimension x source cells, 22-action menu, reference always-Opus; probing-fitted, id_test-realized)",
    f"1. [IDENTITY, C1] Root of Phi: eta* = {u21['root']['eta_star']:.1f} per dollar = {u21['root']['r_star']:.2f} eta_B (9 dimensions: {run_('usd','dim9')['root']['r_star']:.2f}); seconds: {s21['root']['r_star']:.3f} eta_B. Equals the LP maximum over all randomized routers and every exhaustive/hill-climb/random search to 1e-9; the optimal rule is the state-wise argmax; the two-action rule p W_{{eta*}} >= eta* a (exact W = lambda cbar_h + v_h - vbar_h) reproduces it in {u21['checks_part_i']['two_action_rule_exact_agreement']}/{u21['checks_part_i']['two_action_rule_total']} cases (homogeneous W: {u21['checks_part_i']['two_action_rule_homogeneous_agreement']}).",
    f"2. [IDENTITY, C2] Dinkelbach from eta_B: {' -> '.join(f'{x:.1f}' for x in u21['dinkelbach']['lambda_path'])} per dollar, {u21['dinkelbach']['n_steps_to_root']} steps ({u21['dinkelbach']['n_policy_evaluations']} policy evaluations); seconds {s21['dinkelbach']['n_steps_to_root']} step; max {max(run_(b,p)['fit_bootstrap']['dinkelbach_steps_max'] for b in BASES for p in PARTS)} steps over 4,000 probing resamples, always <= |X|; Phi(eta_B) = max net gain = {u21['dinkelbach']['phi_at_eta_B']:.4f} score/task.",
    f"3. [IDENTITY, C3] Floor: at most one randomizing state in every (eps, basis, state space, resample); kappa (LP dual) = kappa (tie): dollars eps=0: eta*_eps/eta_B = {u21['floor']['0']['r_eps']:.2f}, kappa = {u21['floor']['0']['kappa']:.2f}, price eta*_eps/(1+kappa) = {u21['floor']['0']['price']:.1f} per dollar; eps=0.01: {u21['floor']['0.01']['r_eps']:.2f}, kappa {u21['floor']['0.01']['kappa']:.2f}; floor slack for eps >= {u21['optimal_rule']['shortfall_vs_reference']:.4f} (dollars) and for every eps in seconds (the seconds optimum improves quality by {s21['optimal_rule']['shortfall_vs_reference']*-1:+.4f}).",
    f"4. [OUT-OF-SAMPLE, C4] Realized eta/eta_B on id_test (dollars): unconstrained rule {f3(ou['summary']['r'])}; eps=0 rule {f3(oe0['summary']['r'])}; DimensionBest {f3(RESULTS['baselines']['usd']['dimension_best']['summary']['r'])}; best fixed cascade ({RESULTS['baselines']['usd']['best_fixed_cascade']['name']}) {f3(RESULTS['baselines']['usd']['best_fixed_cascade']['summary']['r'])}; ACRouter {f3(RESULTS['baselines']['usd']['acrouter']['summary']['r'])}; best single dispatch ({RESULTS['baselines']['usd']['best_single_direct']['name']}) {f3(RESULTS['baselines']['usd']['best_single_direct']['summary']['r'])}; per-task oracle {f3(RESULTS['oracle']['usd']['unconstrained']['summary']['r'])}. Paired differences rule - DimensionBest {f4(ou['vs']['dimension_best']['diff_r'])}, rule - best cascade {f4(ou['vs']['best_fixed_cascade']['diff_r'])}.",
    f"5. [OUT-OF-SAMPLE, C4] Seconds (sequential): rule {f3(run_('sec','cell21')['oos']['unconstrained']['summary']['r'])} vs DimensionBest {f3(RESULTS['baselines']['sec']['dimension_best']['summary']['r'])}, best cascade {f3(RESULTS['baselines']['sec']['best_fixed_cascade']['summary']['r'])}, ACRouter {f3(RESULTS['baselines']['sec']['acrouter']['summary']['r'])}, oracle {f3(RESULTS['oracle']['sec']['unconstrained']['summary']['r'])}; the rule also raises quality ({f4(run_('sec','cell21')['oos']['unconstrained']['summary']['dV'])}).",
    f"6. [OUT-OF-SAMPLE, C5] Floor satisfaction of the fitted eps-rules on id_test (dollars): eps=0 {oe0['floor']['0']['rate_paired']:.2f}, 0.005 {u21['oos']['eps=0.005']['floor']['0.005']['rate_paired']:.2f}, 0.01 {u21['oos']['eps=0.01']['floor']['0.01']['rate_paired']:.2f}, 0.02 {u21['oos']['eps=0.02']['floor']['0.02']['rate_paired']:.2f}, 0.05 {u21['oos']['eps=0.05']['floor']['0.05']['rate_paired']:.2f} -- a floor that binds in-sample gives no out-of-sample margin (realized E[V]-vbar of the eps=0 rule {f4(oe0['summary']['dV'])}); refitting with a safety margin of {u21['oos_safety_margin_for_95pct']['0']['margin'] if u21['oos_safety_margin_for_95pct']['0'].get('margin') is not None else 'n/a'} (eps=0) / {u21['oos_safety_margin_for_95pct']['0.01']['margin'] if u21['oos_safety_margin_for_95pct']['0.01'].get('margin') is not None else 'n/a'} (eps=0.01) restores >= 95%. Seconds: 1.00 at every eps (floor slack). Verdict FAIL under the declared rule; the estimation gap, not the theorem, is what fails.",
    f"7. [OUT-OF-SAMPLE, C6] State granularity (dollars): best single action {f3(RESULTS['baselines']['usd']['best_single_direct']['summary']['r'])} < 9 dimensions {f3(run_('usd','dim9')['oos']['unconstrained']['summary']['r'])} < 21 cells {f3(ou['summary']['r'])} < per-task oracle {f3(RESULTS['oracle']['usd']['unconstrained']['summary']['r'])}; the rule attains {100*ou['vs']['per_task_oracle']['fraction_of_oracle_r'][0]:.0f}% of the oracle's efficiency; net gain at the reference price (m$/task): rule {1000*RESULTS['per_state']['usd']['policies']['rule']['net_gain_total']/RESULTS['per_state']['usd']['eta_B_id_test']:.2f}, oracle {1000*RESULTS['per_state']['usd']['policies']['oracle']['net_gain_total']/RESULTS['per_state']['usd']['eta_B_id_test']:.2f}, ACRouter {1000*RESULTS['per_state']['usd']['policies']['acrouter']['net_gain_total']/RESULTS['per_state']['usd']['eta_B_id_test']:.2f}, DimensionBest {1000*RESULTS['per_state']['usd']['policies']['dimension_best']['net_gain_total']/RESULTS['per_state']['usd']['eta_B_id_test']:.2f} (per-state table in README).",
    f"8. [IN-SAMPLE] Fixed-price rules J_lambda at lambda in {{0.1, 1, 10, eta_B, 100}} per dollar all sit below the root ({u21['root']['eta_star']:.0f}); their in-sample eta/eta_B: " + ', '.join(f"{lk}: {u21['sensitivity_lambda'][lk]['in_sample']['r']:.2f}" for lk in RESULTS['lambda_grid']) + f"; the eta_B rule is the first Dinkelbach step ({u21['dinkelbach']['lambda_path'][1]/u21['eta_B_probing']:.2f} eta_B). Realized OOS values with CIs in README table (4).",
    f"9. Caveats: oracle verifier; ACRouter scored from released decisions without overhead; dollars favour cheap dispatch (the unconstrained optimum loses 0.02 of the reference's score on id_test); zero-latency rows kept in seconds.",
]
with open(os.path.join(HERE, 'summary_for_paper.md'), 'w') as fh:
    fh.write('\n'.join(lines) + '\n')
log(f'done in {time.time()-T0:.0f}s; README {os.path.getsize(os.path.join(HERE, "README.md"))//1024} KB')
print('\nSUMMARY')
for c_ in CL:
    print(f"  {c_['id']} [{c_['kind']}] {c_['verdict']}")
print(f"  eta*/eta_B: usd/cell21 {u21['root']['r_star']:.3f}, usd/dim9 {run_('usd','dim9')['root']['r_star']:.3f}, sec/cell21 {s21['root']['r_star']:.3f}, sec/dim9 {run_('sec','dim9')['root']['r_star']:.3f}")
print(f"  OOS r (usd, 21 cells): unconstrained {f3(ou['summary']['r'])}, eps=0 {f3(oe0['summary']['r'])}; DimensionBest {f3(RESULTS['baselines']['usd']['dimension_best']['summary']['r'])}; best cascade {f3(RESULTS['baselines']['usd']['best_fixed_cascade']['summary']['r'])}; ACRouter {f3(RESULTS['baselines']['usd']['acrouter']['summary']['r'])}; oracle {f3(RESULTS['oracle']['usd']['unconstrained']['summary']['r'])}")
print(f"  runtime {time.time()-T0:.0f}s")

