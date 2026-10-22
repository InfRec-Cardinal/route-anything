#!/usr/bin/env python3
"""
X6_allocation -- prop:allocation (coverage allocation across independent positions), app:allocation
active-set procedure eq:power-active-set, app:exponential-allocation eq:exponential-allocation,
app:budget-scope weights w_k = nu_k (v_{h,k} - v_{m,k} + lambda b_k) and the penalized problem, and the
remark that thm:tail motivates r = 1 - 1/z only asymptotically.  Real data: CodeRouterBench
(9,999 tasks x 8 models; probing 7,080 / id_test 2,919).

Positions k = 9 dimensions (main) and 21 dimension x source-benchmark cells.  F_k = number of prepared
cheap models at position k (pool = the 7 non-Opus models; claude-opus-4-6 is the reference B and the
fallback).  Unit order: cheapest-first by mean probing cost_usd (task specification), and coverage-greedy
(E3 audit variant; makes the discrete coverage curve concave on probing by submodularity).
p_k(F) = P(at least one of the F prepared models has score >= theta), measured on PROBING; every
allocation is evaluated on the held-out id_test split.

numpy / pandas / scipy / matplotlib only.  Fixed seeds.  Runtime ~1-2 min.
"""
import json, os, time, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np, pandas as pd
from scipy import optimize, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T0 = time.time()
SEED = 20260911
rng = np.random.default_rng(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
DATA = os.environ.get('ACROUTER_DATA', 'agent-as-a-router/data') + '/coderouterbench/id_results_long.csv'
B_MODEL = "claude-opus-4-6"
N_BOOT_TEST = 1000      # id_test task resamples for CIs on held-out objectives / gaps (paired)
N_BOOT_FIT = 500        # probing task resamples for CIs on fitted exponents
THETAS = [1.0, 0.5]
R_LO, R_HI = 1e-3, 30.0  # search box for r (log scale); hitting a bound means r -> 0 or r -> inf

# paper figure style (scripts/reproduce.py); usetex works on this machine
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times'], 'text.usetex': True,
    'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 7.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': .65, 'lines.linewidth': 1.4,
    'text.latex.preamble': r'\usepackage{amsmath,amssymb}'})
BLUE, RED, GREEN, GREY, ORANGE, PURPLE = '#23639b', '#ba5934', '#3a7c5e', '.5', '#d08a2e', '#6b4c9a'

def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)
def tl(x): return x.tolist() if isinstance(x, np.ndarray) else x

# =============================================================================
# 1. Load real data
# =============================================================================
df = pd.read_csv(DATA)
df["src"] = df.task_id.str.split("_").str[1]
df["cell"] = df.dimension + "/" + df.src
MODELS = sorted(df.model.unique()); assert len(MODELS) == 8 and df.shape[0] == 79992
POOL = [m for m in MODELS if m != B_MODEL]; FMAX = len(POOL)   # 7 cheap models; Opus reserved as B
tasks = df.drop_duplicates("task_id").set_index("task_id")[["split", "dimension", "cell"]]
S_all = df.pivot(index="task_id", columns="model", values="score")
C_all = df.pivot(index="task_id", columns="model", values="cost_usd")
L_all = df.pivot(index="task_id", columns="model", values="latency_ms")
tasks = tasks.loc[S_all.index]
te = tasks.split.values == "id_test"; pr = tasks.split.values == "probing"
NTE, NPR = int(te.sum()), int(pr.sum())
opus_s, opus_c, opus_l = S_all[B_MODEL].values, C_all[B_MODEL].values, L_all[B_MODEL].values
vbar, cbar = opus_s[pr].mean(), opus_c[pr].mean()
ETA_B = vbar / cbar                       # reference efficiency on probing (Sec 2), lambda of Sec 4
xcheck = dict(n_probing=NPR, n_id_test=NTE, opus_probing_mean_score=float(vbar), opus_probing_mean_cost=float(cbar),
              eta_B_probing=float(ETA_B), opus_id_test_total_usd=float(opus_c[te].sum()),
              opus_id_test_mean_score=float(opus_s[te].mean()))
log("cross-checks:", xcheck)

# bootstrap weights for id_test (paired across methods)
boot_idx = rng.integers(0, NTE, size=(N_BOOT_TEST, NTE))
BOOT_W = np.zeros((N_BOOT_TEST, NTE))
for b in range(N_BOOT_TEST): BOOT_W[b] = np.bincount(boot_idx[b], minlength=NTE)

# =============================================================================
# 2. Positions, unit order, hit matrices, curves
# =============================================================================
def build_positions(pos_col, order_mode, theta):
    names = sorted(tasks[pos_col].unique()); K = len(names)
    pidx = pd.Categorical(tasks[pos_col], categories=names).codes
    order = {}
    for k, nm in enumerate(names):
        m = (pidx == k) & pr
        if order_mode == "price":
            order[nm] = list(C_all.loc[m, POOL].mean().sort_values().index)
        else:  # coverage-greedy on probing at theta (discrete marginals non-increasing by submodularity)
            Pm = S_all.loc[m, POOL].values >= theta; rem = list(range(FMAX)); cov = np.zeros(Pm.shape[0], bool); ch = []
            for _ in range(FMAX):
                gains = [(cov | Pm[:, j]).mean() - cov.mean() for j in rem]
                j = rem[int(np.argmax(gains))]; rem.remove(j); ch.append(j); cov |= Pm[:, j]
            order[nm] = [POOL[j] for j in ch]
    N = len(tasks); S = np.zeros((N, FMAX)); Cc = np.zeros((N, FMAX)); Lc = np.zeros((N, FMAX))
    for k, nm in enumerate(names):
        m = pidx == k
        S[m] = S_all.loc[m, order[nm]].values; Cc[m] = C_all.loc[m, order[nm]].values; Lc[m] = L_all.loc[m, order[nm]].values
    return dict(names=names, K=K, pidx=pidx, order=order, S=S, C=Cc, L=Lc)

def hit_matrix(P, theta):
    """H[t, F] = 1 if at least one of the first F prepared models passes (F = 0..FMAX)."""
    H = np.zeros((P["S"].shape[0], FMAX + 1)); H[:, 1:] = np.maximum.accumulate(P["S"] >= theta, axis=1); return H

def curves(P, H, mask):
    K = P["K"]; p = np.zeros((K, FMAX + 1)); n = np.zeros(K, int)
    for k in range(K):
        m = mask & (P["pidx"] == k); n[k] = m.sum(); p[k] = H[m].mean(0)
    return p, n

def cum_prep_cost(P, mask):
    """Cc[k, F] = mean $ per task in k of the F prepared models (preparation bill)."""
    K = P["K"]; Cc = np.zeros((K, FMAX + 1))
    for k in range(K):
        m = mask & (P["pidx"] == k); Cc[k, 1:] = np.cumsum(P["C"][m].mean(0))
    return Cc

def cascade_outcomes(P, theta, F_t):
    """Sequential verified cascade (Alg. 1) per task with F_t prepared models and Opus fallback.
    Returns hit, V (committed score), D (cascade $ + Opus $ on miss), cheap $ only, latency (sequential sum)."""
    S, C, L = P["S"], P["C"], P["L"]; n = S.shape[0]
    passm = S >= theta; idx = np.arange(FMAX)[None, :]; within = idx < F_t[:, None]
    first = np.where(passm & within, idx, FMAX + 5).min(1); hit = first < FMAX + 5
    ncalls = np.where(hit, first + 1, F_t)
    called = idx < ncalls[:, None]
    cheap_d = (C * called).sum(1); cheap_l = (L * called).sum(1)
    V = np.where(hit, S[np.arange(n), np.minimum(first, FMAX - 1)], opus_s)
    D = cheap_d + np.where(hit, 0.0, opus_c); Lat = cheap_l + np.where(hit, 0.0, opus_l)
    return hit, V, D, cheap_d, Lat

# =============================================================================
# 3. Curve fits: shifted power law (prop:allocation), exponential (app:exponential-allocation)
# =============================================================================
def _ls_d(y, x):
    d = (y * x).sum(-1) / (x * x).sum(-1); return np.clip(d, 1e-6, 1.0)

def fit_power(p, common_r, pin_d, Fmin=0):
    """1 - p(F) = d_k (1+F)^-r on F = Fmin..FMAX (LS in p).  pin_d: d_k = 1 - p_k(0) = 1 (literal hypothesis)."""
    F = np.arange(Fmin, p.shape[1]); y = 1 - p[:, Fmin:]
    def sse(logr, rows):
        r = np.exp(logr); x = (1 + F) ** (-r)
        d = np.ones(len(rows)) if pin_d else _ls_d(y[rows], x[None, :])
        return ((y[rows] - d[:, None] * x[None, :]) ** 2).sum(), d
    lo, hi = np.log(R_LO), np.log(R_HI)
    if common_r:
        rows = np.arange(p.shape[0])
        res = optimize.minimize_scalar(lambda lr: sse(lr, rows)[0], bounds=(lo, hi), method="bounded")
        r = np.full(p.shape[0], np.exp(res.x)); d = sse(res.x, rows)[1]
    else:
        r = np.zeros(p.shape[0]); d = np.zeros(p.shape[0])
        for k in range(p.shape[0]):
            res = optimize.minimize_scalar(lambda lr: sse(lr, [k])[0], bounds=(lo, hi), method="bounded")
            r[k] = np.exp(res.x); d[k] = sse(res.x, [k])[1][0]
    fit = 1 - d[:, None] * (1 + F[None, :]) ** (-r[:, None])
    return dict(r=r, d=d, rmse_k=np.sqrt(((p[:, Fmin:] - fit) ** 2).mean(1)), rmse=float(np.sqrt(((p[:, Fmin:] - fit) ** 2).mean())))

def fit_exp(p, Fmin=0):
    F = np.arange(Fmin, p.shape[1]); y = 1 - p[:, Fmin:]; th = np.zeros(p.shape[0]); d = np.zeros(p.shape[0])
    for k in range(p.shape[0]):
        def sse(lt):
            x = np.exp(-np.exp(lt) * F); return ((y[k] - _ls_d(y[k], x) * x) ** 2).sum()
        res = optimize.minimize_scalar(sse, bounds=(np.log(1e-4), np.log(10.0)), method="bounded")
        th[k] = np.exp(res.x); d[k] = _ls_d(y[k], np.exp(-th[k] * F))
    fit = 1 - d[:, None] * np.exp(-th[:, None] * F[None, :])
    return dict(vartheta=th, d=d, rmse_k=np.sqrt(((p[:, Fmin:] - fit) ** 2).mean(1)), rmse=float(np.sqrt(((p[:, Fmin:] - fit) ** 2).mean())))

def pcurve(F, d, r): return 1 - d * (1 + F) ** (-r)

# =============================================================================
# 4. Allocators
# =============================================================================
def closed_form_power(w, d, gamma, r, B):
    """eq:power-allocation with the active-set procedure eq:power-active-set (drop smallest-g while mu g_k < 1)."""
    g = (w * d / gamma) ** (1.0 / (1.0 + r)); active = w > 0
    while active.any():
        mu = (B + gamma[active].sum()) / (gamma[active] * g[active]).sum()
        if not (active & (mu * g < 1)).any(): break
        cand = np.where(active)[0]; active[cand[np.argmin(g[cand])]] = False
    mu = (B + gamma[active].sum()) / (gamma[active] * g[active]).sum() if active.any() else np.nan
    return np.where(active, mu * g - 1, 0.0), mu, active, g

def closed_form_power_sorted(w, d, gamma, r, B):
    """Same solution found by the 'upper level set of g' route (sort, check each candidate set) -- code check."""
    g = (w * d / gamma) ** (1.0 / (1.0 + r)); ordr = np.argsort(-g)
    for m in range(1, len(g) + 1):
        act = np.zeros(len(g), bool); act[ordr[:m]] = True
        mu = (B + gamma[act].sum()) / (gamma[act] * g[act]).sum()
        ok_in = (mu * g[act] > 1).all(); ok_out = (mu * g[~act] <= 1).all() if (~act).any() else True
        if ok_in and ok_out: return np.where(act, mu * g - 1, 0.0)
    return None

def kkt_power(w, d, r_k, gamma, B, Fmax, lam_rho=None):
    """KKT for power curves with per-position r_k and box [0, Fmax]: w r d (1+F)^-(r+1) = lam_rho + Lambda gamma.
    lam_rho = lambda * rho_k penalty (app:budget-scope); None -> hard budget.  Bisection on Lambda; Lambda = 0 if slack."""
    lr = np.zeros(len(w)) if lam_rho is None else lam_rho
    def F_of(L):
        den = lr + L * gamma
        with np.errstate(divide="ignore", invalid="ignore"):
            F = (w * r_k * d / den) ** (1.0 / (1.0 + r_k)) - 1
        F = np.where(den > 0, F, Fmax); F = np.where(w > 0, F, 0.0)
        return np.clip(F, 0, Fmax)
    if (gamma * F_of(0.0)).sum() <= B: return F_of(0.0), 0.0
    lo, hi = 1e-14, 1e8
    for _ in range(300):
        mid = np.sqrt(lo * hi)
        if (gamma * F_of(mid)).sum() > B: lo = mid
        else: hi = mid
    return F_of(hi), hi

def kkt_exp(w, d, th, gamma, B, Fmax):
    """eq:exponential-allocation with box: F_k = [ln(w d th / (Lambda gamma)) / th]_+ capped at Fmax."""
    def F_of(L):
        with np.errstate(divide="ignore"):
            F = np.log(np.maximum(w * d * th / (L * gamma), 1e-300)) / th
        return np.clip(np.where(w > 0, F, 0.0), 0, Fmax)
    lo, hi = 1e-14, 1e8
    if (gamma * F_of(lo)).sum() <= B: return F_of(lo), lo
    for _ in range(300):
        mid = np.sqrt(lo * hi)
        if (gamma * F_of(mid)).sum() > B: lo = mid
        else: hi = mid
    return F_of(hi), hi

def exp_all_active_lnLambda(w, d, th, gamma, B):
    a = gamma / th; return ((a * np.log(w * d * th / gamma)).sum() - B) / a.sum()

def project_int(Fc, Cc, B, Fmax, prio):
    """Continuous -> integer within the true budget: floor, add by fractional part while affordable, then by prio."""
    K = len(Fc); Fc = np.clip(Fc, 0, Fmax); F = np.floor(Fc + 1e-9).astype(int)
    spent = Cc[np.arange(K), F].sum(); frac = Fc - F
    # non-ascending per-unit prices (coverage order) can make the floored point unaffordable under the TRUE cumulative
    # cost although it is affordable under the linearized gamma_k F; repair by dropping lowest-priority units first
    while spent > B + 1e-12:
        funded = np.where(F > 0)[0]; k = funded[np.argmin(prio[funded])]
        spent -= Cc[k, F[k]] - Cc[k, F[k] - 1]; F[k] -= 1; frac[k] = 0.0
    for k in np.argsort(-frac, kind="stable"):
        if F[k] < Fmax and frac[k] > 1e-9:
            dc = Cc[k, F[k] + 1] - Cc[k, F[k]]
            if spent + dc <= B + 1e-12: F[k] += 1; spent += dc
    changed = True
    while changed:
        changed = False
        for k in np.argsort(-prio, kind="stable"):
            if F[k] < Fmax and prio[k] > 0:
                dc = Cc[k, F[k] + 1] - Cc[k, F[k]]
                if spent + dc <= B + 1e-12: F[k] += 1; spent += dc; changed = True
    return F

def dp_all_budgets(vals, cost_int):
    """Exact integer optimum of sum_k vals[k, F_k] s.t. sum_k cost_int[k, F_k] <= B for all integer B
    (multiple-choice knapsack).  Ties broken toward lower cost.  vals may be negative (penalized objective)."""
    K, F1 = vals.shape; Bmax = int(cost_int.max(1).sum()); NEG = -1e18
    best = np.full(Bmax + 1, NEG); best[0] = 0.0; choice = np.zeros((K, Bmax + 1), np.int8)
    for k in range(K):
        new = np.full(Bmax + 1, NEG); ch = np.zeros(Bmax + 1, np.int8)
        for F in range(F1):
            c = int(cost_int[k, F]); cand = np.full(Bmax + 1, NEG); cand[c:] = best[:Bmax + 1 - c] + vals[k, F]
            upd = cand > new + 1e-15; new[upd] = cand[upd]; ch[upd] = F
        best, choice[k] = new, ch
    def alloc(B):
        B = int(min(B, Bmax)); b = int(np.argmax(best[:B + 1])); F = np.zeros(K, int)
        for k in range(K - 1, -1, -1):
            F[k] = choice[k, b]; b -= int(cost_int[k, F[k]])
        return F
    return alloc, Bmax

def greedy(vals, Cc, B, Fmax):
    """Discrete equal-marginal rule on the empirical curves: fund the largest marginal gain per price."""
    K = vals.shape[0]; F = np.zeros(K, int); spent = 0.0
    while True:
        best, bk = -np.inf, -1
        for k in range(K):
            if F[k] < Fmax:
                dc = Cc[k, F[k] + 1] - Cc[k, F[k]]
                if spent + dc <= B + 1e-12:
                    ratio = (vals[k, F[k] + 1] - vals[k, F[k]]) / dc
                    if ratio > best: best, bk = ratio, k
        if bk < 0 or best <= 0: break
        spent += Cc[bk, F[bk] + 1] - Cc[bk, F[bk]]; F[bk] += 1
    return F

def uniform_rotations(B, K, Fmax):
    """AUDIT FIX: the unit-price uniform baseline has B mod K remainder slots whose positions are arbitrary.  The
    original gave them to the alphabetically first positions (algorithm, bug_fixing, code_completion), which are
    high-value ones, inflating the baseline by +0.002-0.004.  The baseline is now the average over all K rotations."""
    q, rem = divmod(int(B), K)
    if rem == 0: return [np.minimum(np.full(K, q, int), Fmax)]
    return [np.minimum(np.where(np.isin(np.arange(K), [(s + i) % K for i in range(rem)]), q + 1, q), Fmax) for s in range(K)]

def uniform_alloc(Cc, B, K, Fmax, unit_price):
    if unit_price:
        q, rem = divmod(int(B), K); F = np.full(K, q, int); F[:rem] += 1; return np.minimum(F, Fmax)
    share = B / K
    F = np.array([max([f for f in range(Fmax + 1) if Cc[k, f] <= share + 1e-12]) for k in range(K)])
    spent = Cc[np.arange(K), F].sum(); changed = True
    while changed:  # exhaust the remainder round-robin (budget-exhausting uniform; E3 audit)
        changed = False
        for k in np.argsort(F, kind="stable"):
            if F[k] < Fmax and spent + Cc[k, F[k] + 1] - Cc[k, F[k]] <= B + 1e-12:
                spent += Cc[k, F[k] + 1] - Cc[k, F[k]]; F[k] += 1; changed = True
    return F

# =============================================================================
# 5. Hard-budget configuration (tests a, b, c)
# =============================================================================
def run_hard_budget(P, theta, price_mode, tag, results, boot_fit=True):
    K, names = P["K"], P["names"]
    H = hit_matrix(P, theta); p_pr, n_pr = curves(P, H, pr); p_te, n_te = curves(P, H, te)
    nu = n_pr / n_pr.sum(); Cc_pr = cum_prep_cost(P, pr)
    # weights: unit stake -> w_k = nu_k, the objective is the stream hit rate (margin weights are used in run_budget_scope)
    stake = np.ones(K); w = nu * stake
    # prices
    if price_mode == "unit":
        gamma = np.ones(K); Cc = np.tile(np.arange(FMAX + 1), (K, 1)).astype(float)
        Bgrid = np.arange(1, K * FMAX + 1); cost_int = Cc.astype(int); scale = 1
    else:
        Cc = nu[:, None] * Cc_pr                    # expected preparation $ per stream task at k
        gamma = Cc[:, FMAX] / FMAX                  # linearized unit price (closed forms need a scalar gamma_k)
        Bmax_d = Cc[:, FMAX].sum(); Bgrid = np.linspace(Bmax_d / 48, Bmax_d, 48)
        scale = 1e6; cost_int = np.ceil(Cc * scale).astype(int)
    toB = (lambda B: int(np.floor(B * scale + 1e-9))) if price_mode == "dollar" else (lambda B: int(B))
    # ---- (a) hypothesis checks on probing curves
    dpp = np.diff(p_pr, axis=1)                                   # marginal gains F=1..7
    nonconcave = (np.diff(dpp, axis=1) > 1e-9).any(1)             # a later unit gains strictly more than an earlier one
    zero_steps = (dpp <= 1e-12).sum(1)                            # non-strictly-increasing steps
    never_hit = p_pr[:, FMAX] <= 0
    dpt = np.diff(p_te, axis=1); nonconcave_te = (np.diff(dpt, axis=1) > 1e-9).any(1)
    fits = dict(power_common_literal=fit_power(p_pr, True, True), power_common_free=fit_power(p_pr, True, False),
                power_own_free=fit_power(p_pr, False, False), power_own_literal=fit_power(p_pr, False, True),
                exp_own_free=fit_exp(p_pr),
                power_common_free_F1=fit_power(p_pr, True, False, Fmin=1), power_own_free_F1=fit_power(p_pr, False, False, Fmin=1))
    # bootstrap CIs for exponents (probing task resamples within position)
    boot = None
    if boot_fit:
        r_own = np.zeros((N_BOOT_FIT, K)); r_com = np.zeros(N_BOOT_FIT); r_coml = np.zeros(N_BOOT_FIT)
        rmse_c = np.zeros(N_BOOT_FIT); rmse_o = np.zeros(N_BOOT_FIT)
        Hpr = H[pr]; pidx_pr = P["pidx"][pr]; rows = [np.where(pidx_pr == k)[0] for k in range(K)]
        for b in range(N_BOOT_FIT):
            pb = np.zeros((K, FMAX + 1))
            for k in range(K):
                samp = rng.choice(rows[k], size=len(rows[k]), replace=True); pb[k] = Hpr[samp].mean(0)
            fo = fit_power(pb, False, False); fc = fit_power(pb, True, False); fl = fit_power(pb, True, True)
            r_own[b] = fo["r"]; r_com[b] = fc["r"][0]; r_coml[b] = fl["r"][0]; rmse_o[b] = fo["rmse"]; rmse_c[b] = fc["rmse"]
        ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
        boot = dict(r_own_ci=[ci(r_own[:, k]) for k in range(K)], r_common_free_ci=ci(r_com), r_common_literal_ci=ci(r_coml),
                    r_own_range_ci=ci(r_own.max(1) - r_own.min(1)), r_own_maxmin_ratio_ci=ci(r_own.max(1) / np.maximum(r_own.min(1), 1e-12)),
                    rmse_common_minus_own_ci=ci(rmse_c - rmse_o))
    # ---- allocations over the budget sweep
    vals_pr = w[:, None] * p_pr; vals_te = ((n_te / n_te.sum()) * stake)[:, None] * p_te
    alloc_pr, _ = dp_all_budgets(vals_pr, cost_int); alloc_te, _ = dp_all_budgets(vals_te, cost_int)
    Fint = np.arange(FMAX + 1)[None, :]
    fcf, fcl, fok, fex = fits["power_common_free"], fits["power_common_literal"], fits["power_own_free"], fits["exp_own_free"]
    fit_c = pcurve(Fint, fcf["d"][:, None], fcf["r"][:, None]); fit_o = pcurve(Fint, fok["d"][:, None], fok["r"][:, None])
    alloc_fitc, _ = dp_all_budgets(w[:, None] * fit_c, cost_int); alloc_fito, _ = dp_all_budgets(w[:, None] * fit_o, cost_int)
    # held-out evaluation helpers
    Hte = H[te]; pidx_te = P["pidx"][te]; stake_t = stake[pidx_te]
    Hk = np.zeros((NTE, K * (FMAX + 1)))
    for k in range(K):
        m = pidx_te == k; Hk[m, k * (FMAX + 1):(k + 1) * (FMAX + 1)] = Hte[m]
    Hcount = (BOOT_W @ Hk).reshape(N_BOOT_TEST, K, FMAX + 1)
    def obj_test(F): return float((stake_t * Hte[np.arange(NTE), F[pidx_te]]).mean())
    def obj_test_boot(F): return (stake[None, :] * Hcount[:, np.arange(K), F]).sum(1) / NTE
    methods = ["dp_probe", "closed_common_free", "closed_common_literal", "kkt_power_own", "kkt_exp_own", "greedy_probe",
               "uniform", "dp_test", "dp_fit_common", "dp_fit_own"]
    sweep = {m: [] for m in methods}; boots = {m: [] for m in methods}; allocs = {m: [] for m in methods}
    cont_pred_closed = []; cap_hits = 0; active_set_agree = 0; kkt_marg_spread = []; discrete_kkt = []
    for B in Bgrid:
        A = {"dp_probe": alloc_pr(toB(B)), "dp_test": alloc_te(toB(B)), "dp_fit_common": alloc_fitc(toB(B)), "dp_fit_own": alloc_fito(toB(B))}
        Fc, mu, act, g = closed_form_power(w, fcf["d"], gamma, fcf["r"][0], B)
        Fc_sorted = closed_form_power_sorted(w, fcf["d"], gamma, fcf["r"][0], B)
        active_set_agree += int(Fc_sorted is not None and np.allclose(Fc, Fc_sorted, atol=1e-9))
        cap_hits += int((Fc > FMAX).any())
        cont_pred_closed.append(float((w * pcurve(np.clip(Fc, 0, None), fcf["d"], fcf["r"][0])).sum()))
        A["closed_common_free"] = project_int(Fc, Cc, B, FMAX, g)
        Fl, _, _, gl = closed_form_power(w, fcl["d"], gamma, fcl["r"][0], B)
        A["closed_common_literal"] = project_int(Fl, Cc, B, FMAX, gl)
        Fk, _ = kkt_power(w, fok["d"], fok["r"], gamma, B, FMAX)
        A["kkt_power_own"] = project_int(Fk, Cc, B, FMAX, w * fok["d"] * fok["r"] / gamma)
        Fe, _ = kkt_exp(w, fex["d"], fex["vartheta"], gamma, B, FMAX)
        A["kkt_exp_own"] = project_int(Fe, Cc, B, FMAX, w * fex["d"] * fex["vartheta"] / gamma)
        A["greedy_probe"] = greedy(vals_pr, Cc, B, FMAX)
        A["uniform"] = uniform_alloc(Cc, B, K, FMAX, price_mode == "unit")
        for m in methods:
            F = A[m]; assert Cc[np.arange(K), F].sum() <= B * (1 + 1e-9) + 1e-12, (m, B)
            if m == "uniform" and price_mode == "unit":   # audit fix: rotation-averaged remainder slots
                rots = uniform_rotations(B, K, FMAX)
                sweep[m].append(float(np.mean([obj_test(Fr) for Fr in rots]))); boots[m].append(np.mean([obj_test_boot(Fr) for Fr in rots], axis=0)); allocs[m].append(F.tolist()); continue
            sweep[m].append(obj_test(F)); boots[m].append(obj_test_boot(F)); allocs[m].append(F.tolist())
        # continuous KKT on the fitted concave problem: equal price-normalized marginals across funded positions (code check)
        if act.sum() > 1:
            marg = w * fcf["r"][0] * fcf["d"] * (1 + Fc) ** (-fcf["r"][0] - 1) / gamma
            kkt_marg_spread.append(float((marg[act].max() - marg[act].min()) / marg[act].mean()))
        # discrete equal-marginal diagnostics at the integer optimum (unit prices only: transfers are budget-neutral)
        if price_mode == "unit":
            F = A["dp_probe"]
            back = np.array([w[k] * (p_pr[k, F[k]] - p_pr[k, F[k] - 1]) if F[k] > 0 else np.nan for k in range(K)])
            fwd = np.array([w[k] * (p_pr[k, F[k] + 1] - p_pr[k, F[k]]) if F[k] < FMAX else np.nan for k in range(K)])
            viol = max([fwd[kk] - back[j] for j in range(K) for kk in range(K) if j != kk and np.isfinite(back[j]) and np.isfinite(fwd[kk])] or [0.0])
            discrete_kkt.append(dict(B=float(B), no_improving_transfer=bool(viol <= 1e-12),
                                     cv_back=float(np.nanstd(back) / np.nanmean(back)) if np.isfinite(back).sum() > 1 and np.nanmean(back) > 0 else None))
    # SLSQP agreement at a mid budget (code check of eq:power-allocation)
    Bmid = Bgrid[len(Bgrid) // 2]; Fc, mu, act, g = closed_form_power(w, fcf["d"], gamma, fcf["r"][0], Bmid)
    res = optimize.minimize(lambda F: -(w * pcurve(F, fcf["d"], fcf["r"][0])).sum(), np.full(K, Bmid / gamma.sum()), method="SLSQP",
                            bounds=[(0, None)] * K, constraints=[{"type": "ineq", "fun": lambda F: Bmid - (gamma * F).sum()}],
                            options={"ftol": 1e-14, "maxiter": 2000})
    slsqp_gap = float(np.abs(res.x - Fc).max())
    exp_formula_gap = None
    for Bx in [Bmid, Bgrid[-1], 3 * Bgrid[-1], 10 * Bgrid[-1], 30 * Bgrid[-1]]:
        lnL = exp_all_active_lnLambda(w, fex["d"], fex["vartheta"], gamma, Bx)
        Fe_c = np.log(w * fex["d"] * fex["vartheta"] / gamma) / fex["vartheta"] - lnL / fex["vartheta"]
        if (Fe_c > 0).all():
            Fe_b, _ = kkt_exp(w, fex["d"], fex["vartheta"], gamma, Bx, 1e9); exp_formula_gap = float(np.abs(Fe_b - Fe_c).max()); break
    # proportionality 1+F_k ∝ (w_k d_k/gamma_k)^{1/(1+r)} at the held-out-evaluated integer optimum (mid budgets)
    prop = []
    for B in Bgrid[len(Bgrid) // 4: 3 * len(Bgrid) // 4]:
        F = alloc_pr(toB(B)); m = F > 0
        if m.sum() >= 3:
            x = np.log(w[m] * fcf["d"][m] / gamma[m]); y = np.log(1 + F[m])
            if np.std(x) > 0 and np.std(y) > 0:
                sl = stats.linregress(x, y); prop.append(dict(B=float(B), slope=float(sl.slope), spearman=float(stats.spearmanr(x, y).correlation), n=int(m.sum())))
    def gap(mA, mB):
        d = np.array(sweep[mA]) - np.array(sweep[mB]); md = (np.array(boots[mA]) - np.array(boots[mB])).mean(0)
        return dict(mean_gap=float(d.mean()), ci=[float(np.percentile(md, 2.5)), float(np.percentile(md, 97.5))],
                    rel_mean_gap=float(d.mean() / np.array(sweep[mB]).mean()), frac_A_ge_B=float((d >= -1e-12).mean()))
    gaps = {f"{a}_vs_{b}": gap(a, b) for a, b in [
        ("closed_common_free", "dp_probe"), ("closed_common_free", "uniform"), ("closed_common_literal", "dp_probe"), ("closed_common_literal", "uniform"),
        ("kkt_power_own", "dp_probe"), ("kkt_power_own", "uniform"), ("kkt_exp_own", "dp_probe"), ("kkt_exp_own", "uniform"),
        ("greedy_probe", "dp_probe"), ("greedy_probe", "uniform"), ("dp_probe", "uniform"), ("dp_probe", "dp_test"),
        ("dp_fit_common", "closed_common_free"), ("dp_fit_common", "dp_probe"), ("dp_fit_own", "kkt_power_own"), ("dp_fit_own", "dp_probe")]}
    out = dict(theta=theta, price_mode=price_mode, order=P["order"], K=K, names=names, n_probe=tl(n_pr), n_test=tl(n_te), nu=tl(nu),
               w=tl(w), gamma=tl(gamma), p_probe=tl(p_pr), p_test=tl(p_te), prep_cost_probe_usd=tl(Cc_pr),
               nonconcave_probe=[names[k] for k in np.where(nonconcave)[0]], n_nonconcave_probe=int(nonconcave.sum()),
               nonconcave_test=[names[k] for k in np.where(nonconcave_te)[0]], n_nonconcave_test=int(nonconcave_te.sum()),
               zero_marginal_steps_probe=tl(zero_steps), never_hit_positions=[names[k] for k in np.where(never_hit)[0]],
               fits={k: {kk: tl(vv) for kk, vv in v.items()} for k, v in fits.items()}, fit_boot=boot,
               Bgrid=[float(b) for b in Bgrid], sweep_test=sweep, allocs=allocs,
               sweep_test_ci={m: [[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] for b in boots[m]] for m in methods},
               closed_form_continuous_prediction=cont_pred_closed, closed_form_cap_binds_frac=cap_hits / len(Bgrid),
               active_set_vs_sorted_agree_frac=active_set_agree / len(Bgrid),
               cont_kkt_marginal_spread_max=float(max(kkt_marg_spread)) if kkt_marg_spread else None,
               slsqp_vs_closed_form_maxabs=slsqp_gap, exp_all_active_formula_vs_bisect_maxabs=exp_formula_gap,
               discrete_kkt=discrete_kkt,
               discrete_kkt_no_transfer_frac=float(np.mean([c["no_improving_transfer"] for c in discrete_kkt])) if discrete_kkt else None,
               discrete_kkt_cv_back_median=float(np.nanmedian([c["cv_back"] if c["cv_back"] is not None else np.nan for c in discrete_kkt])) if discrete_kkt else None,
               proportionality=prop,
               proportionality_mean_slope=float(np.mean([q["slope"] for q in prop])) if prop else None,
               proportionality_mean_spearman=float(np.mean([q["spearman"] for q in prop])) if prop else None,
               proportionality_predicted_slope=float(1 / (1 + fcf["r"][0])), gaps=gaps)
    results[tag] = out; return out, H

# =============================================================================
# 6. app:budget-scope: weights, fixed-value hypothesis, penalized problem (test d)
# =============================================================================
def run_budget_scope(P, theta, tag, results, lam=ETA_B):
    K, names = P["K"], P["names"]; H = hit_matrix(P, theta); p_pr, n_pr = curves(P, H, pr); nu = n_pr / n_pr.sum()
    pidx = P["pidx"]
    # exact per-position profiles on probing: hit/miss values, fallback cost, cheap resource, J_k(F) (eq:accounting)
    vh = np.full((K, FMAX + 1), np.nan); vm = np.full((K, FMAX + 1), np.nan); bk = np.full((K, FMAX + 1), np.nan)
    Rk = np.zeros((K, FMAX + 1)); Jk = np.zeros((K, FMAX + 1)); Vk = np.zeros((K, FMAX + 1)); Dk = np.zeros((K, FMAX + 1))
    for F in range(FMAX + 1):
        hit, V, D, cheap, _ = cascade_outcomes(P, theta, np.full(len(pidx), F))
        for k in range(K):
            m = pr & (pidx == k); h = hit[m]
            Vk[k, F] = V[m].mean(); Dk[k, F] = D[m].mean(); Rk[k, F] = cheap[m].mean()
            Jk[k, F] = nu[k] * (V[m] - lam * D[m]).mean()
            if h.any(): vh[k, F] = V[m][h].mean()
            if (~h).any(): vm[k, F] = opus_s[m][~h].mean(); bk[k, F] = opus_c[m][~h].mean()
    # F-independent primitives for the weights: hit/miss values and fallback cost pooled over F = 1..FMAX (task-weighted)
    def pooled(vals, sel_fn):
        num = np.zeros(K); den = np.zeros(K)
        for F in range(1, FMAX + 1):
            hit, V, D, cheap, _ = cascade_outcomes(P, theta, np.full(len(pidx), F))
            for k in range(K):
                m = pr & (pidx == k); s = sel_fn(hit[m]); num[k] += vals(V, m)[s].sum(); den[k] += s.sum()
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)
    vh_bar = pooled(lambda V, m: V[m], lambda h: h); vm_bar = pooled(lambda V, m: opus_s[m], lambda h: ~h); b_bar = pooled(lambda V, m: opus_c[m], lambda h: ~h)
    vb = np.array([opus_s[pr & (pidx == k)].mean() for k in range(K)])       # value of going directly to B in k
    vh_bar = np.where(np.isnan(vh_bar), 0.0, vh_bar); vm_bar = np.where(np.isnan(vm_bar), vb, vm_bar); b_bar = np.where(np.isnan(b_bar), cbar, b_bar)
    W_lam = vh_bar - vm_bar + lam * b_bar                                     # margin with the miss value (app:budget-scope)
    W_lam_sec4 = vh_bar - vb + lam * b_bar                                   # eq:margin with v_b (direct-fallback value)
    w = nu * np.maximum(W_lam, 0.0)                                          # nonpositive-margin positions excluded
    # linear resource hypothesis rho_k F: OLS through origin on nu_k R_k(F), F = 0..FMAX
    Fv = np.arange(FMAX + 1, dtype=float)
    rho = np.array([(nu[k] * Rk[k] * Fv).sum() / (Fv * Fv).sum() for k in range(K)])
    R_lin_r2 = np.array([1 - ((nu[k] * Rk[k] - rho[k] * Fv) ** 2).sum() / max(((nu[k] * Rk[k] - (nu[k] * Rk[k]).mean()) ** 2).sum(), 1e-30) for k in range(K)])
    # decomposition check (in-sample): J_k(F) - J_k(0) vs w_k [p_k(F) - p_k(0)] - lambda nu_k [R_k(F) - R_k(0)]  (exact rho) and - lambda rho_k F (linear rho)
    dJ = Jk - Jk[:, [0]]
    pred_exact_rho = w[:, None] * (p_pr - p_pr[:, [0]]) - lam * nu[:, None] * (Rk - Rk[:, [0]])
    pred_lin_rho = w[:, None] * (p_pr - p_pr[:, [0]]) - lam * rho[:, None] * Fv[None, :]
    def r2(pred): return float(1 - ((dJ[:, 1:] - pred[:, 1:]) ** 2).sum() / ((dJ[:, 1:] - dJ[:, 1:].mean()) ** 2).sum())
    decomp = dict(r2_exact_rho=r2(pred_exact_rho), r2_linear_rho=r2(pred_lin_rho),
                  rmse_exact_rho=float(np.sqrt(((dJ[:, 1:] - pred_exact_rho[:, 1:]) ** 2).mean())),
                  rmse_linear_rho=float(np.sqrt(((dJ[:, 1:] - pred_lin_rho[:, 1:]) ** 2).mean())),
                  dJ_scale_rms=float(np.sqrt((dJ[:, 1:] ** 2).mean())))
    # ---- allocations (unit prices, slots): penalized KKT on own-r_k fitted curves; DP on exact probing J profile; hard-budget coverage DP; uniform; F=0
    fok = fit_power(p_pr, False, False); gamma = np.ones(K); Cc = np.tile(np.arange(FMAX + 1), (K, 1)).astype(float)
    Bgrid = np.arange(1, K * FMAX + 1)
    alloc_J, _ = dp_all_budgets(Jk, Cc.astype(int))                     # exact integer optimum of the penalized objective (probing primitives)
    alloc_cov, _ = dp_all_budgets(w[:, None] * p_pr, Cc.astype(int))    # hard-budget weighted coverage (ignores the lambda rho penalty)
    alloc_hit, _ = dp_all_budgets(nu[:, None] * p_pr, Cc.astype(int))   # hard-budget unweighted coverage
    # held-out exact J_lambda per task for every F (id_test)
    Jte = np.zeros((NTE, FMAX + 1)); Vte = np.zeros((NTE, FMAX + 1)); Dte = np.zeros((NTE, FMAX + 1)); Lte = np.zeros((NTE, FMAX + 1)); Hte = np.zeros((NTE, FMAX + 1))
    for F in range(FMAX + 1):
        hit, V, D, cheap, Lat = cascade_outcomes(P, theta, np.full(len(pidx), F))
        Jte[:, F] = (V - lam * D)[te]; Vte[:, F] = V[te]; Dte[:, F] = D[te]; Lte[:, F] = Lat[te]; Hte[:, F] = hit[te]
    pidx_te = pidx[te]
    def per_task(M, F): return M[np.arange(NTE), F[pidx_te]]
    def boot_mean(x): return (BOOT_W @ x) / NTE
    methods = ["penalized_kkt_own", "dp_J_probe", "dp_coverage_w", "dp_coverage_hit", "kkt_power_own_hard", "uniform", "always_B"]
    sweep = {m: [] for m in methods}; boots = {m: [] for m in methods}; allocs = {m: [] for m in methods}
    realized = {m: [] for m in methods}; spent = {m: [] for m in methods}; Lambda_list = []
    for B in Bgrid:
        A = {"dp_J_probe": alloc_J(int(B)), "dp_coverage_w": alloc_cov(int(B)), "dp_coverage_hit": alloc_hit(int(B)),
             "uniform": uniform_alloc(Cc, B, K, FMAX, True), "always_B": np.zeros(K, int)}
        Fp, Lam = kkt_power(w, fok["d"], fok["r"], gamma, B, FMAX, lam_rho=lam * rho); Lambda_list.append(float(Lam))
        # integer projection of the penalized solution: floor, then add units only if they help the fitted penalized objective
        Fi = np.floor(Fp + 1e-9).astype(int); frac = Fp - Fi
        for k in np.argsort(-frac, kind="stable"):
            if Fi[k] < FMAX and frac[k] > 1e-9 and Fi.sum() + 1 <= B:
                gain = w[k] * (pcurve(Fi[k] + 1, fok["d"][k], fok["r"][k]) - pcurve(Fi[k], fok["d"][k], fok["r"][k])) - lam * rho[k]
                if gain > 0: Fi[k] += 1
        A["penalized_kkt_own"] = Fi
        Fh, _ = kkt_power(w, fok["d"], fok["r"], gamma, B, FMAX); A["kkt_power_own_hard"] = project_int(Fh, Cc, B, FMAX, w * fok["d"] * fok["r"] / gamma)
        for m in methods:
            F = A[m]; assert F.sum() <= B + 1e-9
            if m == "uniform":   # audit fix: rotation-averaged remainder slots (realized[] below still uses the base rotation)
                js = [per_task(Jte, Fr) for Fr in uniform_rotations(B, K, FMAX)]
                sweep[m].append(float(np.mean([j.mean() for j in js]))); boots[m].append(np.mean([boot_mean(j) for j in js], axis=0))
            else:
                j = per_task(Jte, F); sweep[m].append(float(j.mean())); boots[m].append(boot_mean(j))
            allocs[m].append(F.tolist()); spent[m].append(int(F.sum()))
            realized[m].append(dict(hit_rate=float(per_task(Hte, F).mean()), mean_score=float(per_task(Vte, F).mean()), mean_usd=float(per_task(Dte, F).mean()),
                                    total_usd=float(per_task(Dte, F).sum()), mean_latency_s=float(per_task(Lte, F).mean() / 1000),
                                    eta=float(per_task(Vte, F).mean() / per_task(Dte, F).mean())))
    def gap(mA, mB):
        d = np.array(sweep[mA]) - np.array(sweep[mB]); md = (np.array(boots[mA]) - np.array(boots[mB])).mean(0)
        return dict(mean_gap=float(d.mean()), ci=[float(np.percentile(md, 2.5)), float(np.percentile(md, 97.5))], frac_A_ge_B=float((d >= -1e-12).mean()))
    # slack: does the penalized optimum exhaust the budget?  (paper: "the budget need not bind")
    Bs = np.array(Bgrid); sp_pen = np.array(spent["penalized_kkt_own"]); sp_J = np.array(spent["dp_J_probe"])
    Bsat_pen = int(Bs[np.argmax(sp_pen < Bs)]) if (sp_pen < Bs).any() else None
    Bsat_J = int(Bs[np.argmax(sp_J < Bs)]) if (sp_J < Bs).any() else None
    # also: J at the unconstrained penalized optimum = the Sec-4 threshold rule per position (prop:threshold) when F in {0,1}
    out = dict(theta=theta, lambda_=float(lam), names=names, nu=tl(nu), vh_by_F=tl(vh), vm_by_F=tl(vm), b_by_F=tl(bk), R_by_F=tl(Rk), J_by_F=tl(Jk),
               V_by_F=tl(Vk), D_by_F=tl(Dk), vh_bar=tl(vh_bar), vm_bar=tl(vm_bar), b_bar=tl(b_bar), v_b=tl(vb), W_lambda=tl(W_lam), W_lambda_sec4=tl(W_lam_sec4),
               w=tl(w), rho=tl(rho), rho_linear_r2=tl(R_lin_r2), fixed_value_ranges=dict(
                   vh_range=tl(np.nanmax(vh[:, 1:], 1) - np.nanmin(vh[:, 1:], 1)), vm_range=tl(np.nanmax(vm[:, 1:], 1) - np.nanmin(vm[:, 1:], 1)),
                   b_range_rel=tl((np.nanmax(bk[:, 1:], 1) - np.nanmin(bk[:, 1:], 1)) / np.nanmean(bk[:, 1:], 1))),
               decomposition=decomp, Bgrid=[int(b) for b in Bgrid], sweep_J_test=sweep, allocs=allocs, spent=spent, Lambda=Lambda_list,
               sweep_J_test_ci={m: [[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] for b in boots[m]] for m in methods},
               realized=realized, first_budget_with_slack_penalized=Bsat_pen, first_budget_with_slack_dpJ=Bsat_J,
               gaps={f"{a}_vs_{b}": gap(a, b) for a, b in [("penalized_kkt_own", "dp_J_probe"), ("penalized_kkt_own", "uniform"), ("penalized_kkt_own", "dp_coverage_w"),
                                                           ("penalized_kkt_own", "kkt_power_own_hard"), ("dp_J_probe", "dp_coverage_w"), ("dp_J_probe", "dp_coverage_hit"),
                                                           ("dp_J_probe", "uniform"), ("dp_J_probe", "always_B"), ("dp_coverage_w", "always_B"), ("uniform", "always_B"),
                                                           ("penalized_kkt_own", "always_B")]},
               J_test_always_B=float(sweep["always_B"][0]))
    results[tag] = out; return out

# =============================================================================
# 7. thm:tail remark: Zipf families and the exponent r = 1 - 1/z (semi-synthetic, test e)
# =============================================================================
def run_zipf(results):
    """Families of cells with nu_j ∝ j^-z built from real tasks (fine: the 2,919 id_test tasks as singleton cells with a
    seeded random rank; coarse: 21 real cells ranked by probing size).  Clean writes with probability omega, TTL E.
    Coverage curve p(E) = sum_j nu_j [1 - (1 - omega nu_j)^E] (eq:app-ttl, exact) and a simulation with real task arrivals.
    Fit 1 - p(E) = d (1+E)^-r over windows of E and compare r-hat with 1 - 1/z (thm:tail)."""
    out = {}
    Egrid_full = np.unique(np.concatenate([np.arange(1, 64), np.geomspace(64, 2e5, 60).astype(int)]))
    windows = {"F_range_0_7": (0, 7), "E_1_63": (1, 63), "E_64_1023": (64, 1023), "E_1024_16383": (1024, 16383), "E_16384_2e5": (16384, 200000)}
    # real-task Delta_j (fine cells): informed = route to a model recorded as passing this task among the 7 cheap models;
    # cold = cheapest model (Qwen3-Max on price).  Delta_j = 1[any cheap model passes] - 1[cheapest passes] at theta = 1.0.
    te_ids = np.where(te)[0]; cheap_price = C_all.loc[pr, POOL].mean().sort_values().index[0]
    any_pass = (S_all.loc[te, POOL].values >= 1.0).any(1).astype(float); cheapest_pass = (S_all.loc[te, cheap_price].values >= 1.0).astype(float)
    Delta_real = any_pass - cheapest_pass
    for fam in ["fine", "coarse"]:
        if fam == "fine":
            J = NTE; rank = rng.permutation(J) + 1
        else:
            sizes = tasks.loc[pr].cell.value_counts(); J = len(sizes); rank = np.arange(1, J + 1)
        for z in [1.5, 2.0, 3.0]:
            nu = rank ** (-z); nu = nu / nu.sum()
            for omega in [1.0, 0.5]:
                E = Egrid_full.astype(float)
                unc = (nu[:, None] * (1 - omega * nu[:, None]) ** E[None, :]).sum(0)     # 1 - p(E), common Delta
                p = 1 - unc
                rec = dict(z=z, omega=omega, J=int(J), r_target=float(1 - 1 / z), fits={})
                for wname, (lo, hi) in windows.items():
                    m = (E >= lo) & (E <= hi)
                    if m.sum() < 3: continue
                    # (i) LS-in-p shifted power law (what an allocator fits to a coverage curve on this range)
                    y = unc[m]; Fw = E[m]
                    if (y <= 0).any():      # finite family fully covered: no power law to fit
                        rec["fits"][wname] = dict(r_ls_in_p=None, r_loglog=None, n_points=int(m.sum()), note="uncovered mass reaches 0 in window"); continue
                    def sse(lr):
                        r = np.exp(lr); x = (1 + Fw) ** (-r); d = np.clip((y * x).sum() / (x * x).sum(), 1e-9, 1.0); return ((y - d * x) ** 2).sum()
                    res = optimize.minimize_scalar(sse, bounds=(np.log(1e-3), np.log(30)), method="bounded"); r_ls = float(np.exp(res.x))
                    # (ii) log-log slope of 1-p(E) vs 1+E (asymptotic exponent estimate)
                    sl = stats.linregress(np.log(1 + Fw), np.log(y)); r_ll = float(-sl.slope)
                    rec["fits"][wname] = dict(r_ls_in_p=r_ls, r_loglog=r_ll, n_points=int(m.sum()))
                if fam == "fine":
                    # heterogeneous real Delta_j (thm:tail assumes a common Delta): deficit sum_j nu_j Delta_j (1-omega nu_j)^E
                    unc_D = (nu[:, None] * Delta_real[:, None] * (1 - omega * nu[:, None]) ** E[None, :]).sum(0)
                    m = (E >= 1024) & (E <= 200000) & (unc_D > 0); sl = stats.linregress(np.log(1 + E[m]), np.log(unc_D[m]))
                    rec["real_Delta_loglog_r_E1024_2e5"] = float(-sl.slope); rec["real_Delta_mean"] = float(Delta_real.mean())
                rec["E"] = tl(E); rec["uncovered"] = tl(unc)
                out[f"{fam}|z={z}|omega={omega}"] = rec
    # simulation with real task arrivals (fine, z = 2, omega = 1): i.i.d. cells from nu; TTL coverage measured on the stream
    J = NTE; rank = rng.permutation(J) + 1; nu = rank ** (-2.0); nu /= nu.sum(); Nsim = 300000; Eg = np.array([1, 3, 7, 15, 63, 255, 1023, 4095, 16383])
    arrivals = rng.choice(J, size=Nsim, p=nu); last = np.full(J, -10 ** 9); covered = np.zeros((len(Eg),)); cnt = 0
    for t in range(Nsim):
        j = arrivals[t]
        if t >= Nsim // 3:
            age = t - last[j]; covered += (age <= Eg); cnt += 1
        last[j] = t
    p_sim = covered / cnt; p_exact = 1 - (nu[:, None] * (1 - nu[:, None]) ** Eg[None, :].astype(float)).sum(0)
    out["simulation_fine_z2_omega1"] = dict(E=tl(Eg), p_sim=tl(p_sim), p_exact=tl(p_exact), max_abs_diff=float(np.abs(p_sim - p_exact).max()), N=Nsim,
                                            note="real id_test tasks arrive i.i.d. with Zipf(2) cell frequencies; records = task identity; the arriving task's real outcome is unused here because coverage is an occupancy quantity")
    results["zipf"] = out; return out

# =============================================================================
# 8. Noisy verifier robustness (no prop:allocation hypothesis concerns verifier error; reported for the protocol)
# =============================================================================
def run_noisy(P, theta, beta_fa, beta_fr, tag, results):
    K = P["K"]; H = hit_matrix(P, theta); pidx = P["pidx"]
    r = np.random.default_rng(SEED + 7)
    passm = P["S"] >= theta; flip = r.random(passm.shape)
    declared = np.where(passm, flip >= beta_fr, flip < beta_fa)
    Hd = np.zeros_like(H); Hd[:, 1:] = np.maximum.accumulate(declared, axis=1)
    pd_pr, n_pr = curves(P, Hd, pr); nu = n_pr / n_pr.sum(); p_pr, _ = curves(P, H, pr)
    Cc = np.tile(np.arange(FMAX + 1), (K, 1)); alloc_d, _ = dp_all_budgets(nu[:, None] * pd_pr, Cc); alloc_e, _ = dp_all_budgets(nu[:, None] * p_pr, Cc)
    rows = []
    for B in [9, 18, 27, 36, 45]:
        Fd, Fe = alloc_d(B), alloc_e(B); pidx_te = pidx[te]
        # committed output under the noisy verifier: first *declared* pass; true value = its real score >= theta
        idx = np.arange(FMAX)[None, :]
        def true_pass(F):
            Ft = F[pidx_te]; within = idx < Ft[:, None]; first = np.where(declared[te] & within, idx, FMAX + 5).min(1); hit = first < FMAX + 5
            return float((hit & passm[te][np.arange(NTE), np.minimum(first, FMAX - 1)]).mean()), float(hit.mean())
        tp_d, dh_d = true_pass(Fd); tp_e, dh_e = true_pass(Fe)
        rows.append(dict(B=B, alloc_declared=Fd.tolist(), declared_hit_rate=dh_d, true_pass_declared_alloc=tp_d, alloc_exact=Fe.tolist(),
                         declared_hit_rate_exact_alloc=dh_e, true_pass_exact_alloc=tp_e))
    results[tag] = dict(theta=theta, beta_fa=beta_fa, beta_fr=beta_fr, rows=rows)

# =============================================================================
# 9. Run everything
# =============================================================================
results = {"cross_checks": xcheck, "config": dict(seed=SEED, n_boot_test=N_BOOT_TEST, n_boot_fit=N_BOOT_FIT, pool=POOL, reference=B_MODEL, Fmax=FMAX)}
POSC = {}
for pos_col in ["dimension", "cell"]:
    for order in ["price", "coverage"]:
        for theta in THETAS:
            POSC[(pos_col, order, theta)] = build_positions(pos_col, order, theta)
HB = {}
for pos_col in ["dimension", "cell"]:
    for order in ["price", "coverage"]:
        for theta in THETAS:
            P = POSC[(pos_col, order, theta)]
            for price_mode in ["unit", "dollar"]:
                tag = f"{pos_col}|{order}|theta={theta}|{price_mode}"
                out, H = run_hard_budget(P, theta, price_mode, tag, results, boot_fit=(price_mode == "unit"))
                HB[tag] = out
                g = out["gaps"]
                log(tag, f"r_common(lit)={out['fits']['power_common_literal']['r'][0]:.3f} r_common(free)={out['fits']['power_common_free']['r'][0]:.3f}",
                    f"rmse common/own/exp={out['fits']['power_common_free']['rmse']:.3f}/{out['fits']['power_own_free']['rmse']:.3f}/{out['fits']['exp_own_free']['rmse']:.3f}",
                    f"nonconcave={out['n_nonconcave_probe']}/{out['K']}",
                    f"closed-DP={g['closed_common_free_vs_dp_probe']['mean_gap']:+.4f} closed-uni={g['closed_common_free_vs_uniform']['mean_gap']:+.4f}",
                    f"kkt_own-DP={g['kkt_power_own_vs_dp_probe']['mean_gap']:+.4f} kkt_own-uni={g['kkt_power_own_vs_uniform']['mean_gap']:+.4f}",
                    f"greedy-DP={g['greedy_probe_vs_dp_probe']['mean_gap']:+.4f} DP-uni={g['dp_probe_vs_uniform']['mean_gap']:+.4f} DPpr-DPte={g['dp_probe_vs_dp_test']['mean_gap']:+.4f}")
BS = {}
for order in ["price", "coverage"]:
    for theta in THETAS:
        tag = f"budget_scope|dimension|{order}|theta={theta}"
        out = run_budget_scope(POSC[("dimension", order, theta)], theta, tag, results); BS[tag] = out
        g = out["gaps"]
        log(tag, "W_lambda>0:", int((np.array(out["W_lambda"]) > 0).sum()), "/9", f"decomp R2 exact-rho={out['decomposition']['r2_exact_rho']:.3f} lin-rho={out['decomposition']['r2_linear_rho']:.3f}",
            f"pen-DPJ={g['penalized_kkt_own_vs_dp_J_probe']['mean_gap']:+.5f} pen-uni={g['penalized_kkt_own_vs_uniform']['mean_gap']:+.5f} pen-covDP={g['penalized_kkt_own_vs_dp_coverage_w']['mean_gap']:+.5f}",
            f"DPJ-covDP={g['dp_J_probe_vs_dp_coverage_w']['mean_gap']:+.5f} slack from B={out['first_budget_with_slack_penalized']}/{out['first_budget_with_slack_dpJ']}")
log("zipf ..."); Z = run_zipf(results)
for key in ["fine|z=1.5|omega=1.0", "fine|z=2.0|omega=1.0", "fine|z=3.0|omega=1.0", "coarse|z=2.0|omega=1.0"]:
    rec = Z[key]; log(key, "target", f"{rec['r_target']:.3f}", {w: (round(f['r_ls_in_p'], 3), round(f['r_loglog'], 3)) for w, f in rec["fits"].items()})
log("sim vs exact max diff", Z["simulation_fine_z2_omega1"]["max_abs_diff"])
for theta in THETAS:
    run_noisy(POSC[("dimension", "price", theta)], theta, 0.1, 0.1, f"noisy|dimension|price|theta={theta}|beta=0.1", results)

# =============================================================================
# 10. Figures (vector PDF, paper style)
# =============================================================================
short = {"algorithm": "algorithm", "bug_fixing": "bug fixing", "code_completion": "completion", "code_generation": "generation",
         "code_refactoring": "refactoring", "code_understanding": "understanding", "data_science": "data science",
         "multi_language": "multi-language", "test_generation": "test generation"}
for theta in THETAS:
    o1 = HB[f"dimension|price|theta={theta}|unit"]; o2 = HB[f"dimension|coverage|theta={theta}|unit"]
    fig, axes = plt.subplots(3, 3, figsize=(6.75, 5.2), layout='constrained', sharex=True, sharey=True)
    F = np.arange(FMAX + 1); Ff = np.linspace(0, FMAX, 200)
    for k, ax in enumerate(axes.flat):
        ax.plot(F, o1["p_probe"][k], 'o-', color=BLUE, ms=2.5, label='price order (probing)')
        ax.plot(F, o2["p_probe"][k], 's-', color=GREEN, ms=2.5, label='coverage order (probing)')
        ax.plot(F, o1["p_test"][k], 'o', mfc='none', color=BLUE, ms=3.5, label='price order (id\\_test)')
        fc = o1["fits"]["power_common_free"]; fo = o1["fits"]["power_own_free"]
        ax.plot(Ff, pcurve(Ff, fc["d"][k], fc["r"][k]), ':', color=RED, label=f'common $r$ fit')
        ax.plot(Ff, pcurve(Ff, fo["d"][k], fo["r"][k]), '--', color=ORANGE, label=f'own $r_k$ fit')
        ax.set_title(short[o1["names"][k]] + f"  ($r_k={fo['r'][k]:.2f}$)", fontsize=8)
        ax.set_ylim(-0.02, 1.02)
    for ax in axes[-1]: ax.set_xlabel('$F_k$ (prepared models)')
    for ax in axes[:, 0]: ax.set_ylabel('$p_k(F)$')
    axes[0, 0].legend(loc='upper left', frameon=False, fontsize=6.5)
    fig.savefig(os.path.join(FIG, f"curves_theta{theta}.pdf")); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.3), layout='constrained')
for ax, theta in zip(axes, THETAS):
    o = HB[f"dimension|price|theta={theta}|unit"]; B = np.array(o["Bgrid"])
    lo, hi = np.array(o["sweep_test_ci"]["dp_probe"]).T
    ax.fill_between(B, lo, hi, color=GREY, alpha=.18, lw=0)
    ax.plot(B, o["sweep_test"]["dp_probe"], color='.25', label='exact integer optimum (probing curves)')
    ax.plot(B, o["sweep_test"]["greedy_probe"], color=GREEN, ls='-.', label='discrete equal-marginal (empirical)')
    ax.plot(B, o["sweep_test"]["kkt_power_own"], color=ORANGE, ls='--', label='KKT, own $r_k$ fits')
    ax.plot(B, o["sweep_test"]["closed_common_free"], color=RED, label='closed form, common $r$ (free $d_k$)')
    ax.plot(B, o["sweep_test"]["closed_common_literal"], color=PURPLE, ls=':', label='closed form, common $r$ ($d_k=1$)')
    ax.plot(B, o["sweep_test"]["uniform"], color=BLUE, ls=(0, (1, 1)), label='uniform')
    ax.set(xlabel='Budget $B$ (prepared slots, 9 positions)', ylabel='Held-out hit rate (id\\_test)', title=fr'$\theta={theta}$, price order', xlim=(1, B[-1]))
axes[0].legend(loc='lower right', frameon=False, fontsize=6.5)
fig.savefig(os.path.join(FIG, "sweep_unit.pdf")); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.3), layout='constrained')
for ax, theta in zip(axes, THETAS):
    o = BS[f"budget_scope|dimension|price|theta={theta}"]; B = np.array(o["Bgrid"])
    lo, hi = np.array(o["sweep_J_test_ci"]["dp_J_probe"]).T
    ax.fill_between(B, lo, hi, color=GREY, alpha=.18, lw=0)
    ax.plot(B, o["sweep_J_test"]["dp_J_probe"], color='.25', label='integer optimum of exact $J_\\lambda$ profile')
    ax.plot(B, o["sweep_J_test"]["penalized_kkt_own"], color=ORANGE, ls='--', label='penalized KKT (weights $w_k$, own $r_k$)')
    ax.plot(B, o["sweep_J_test"]["dp_coverage_w"], color=RED, label='hard-budget weighted coverage')
    ax.plot(B, o["sweep_J_test"]["uniform"], color=BLUE, ls=(0, (1, 1)), label='uniform')
    ax.axhline(o["J_test_always_B"], color=GREY, lw=.65, ls=':', label='always $B$ (Opus)')
    ax.set(xlabel='Budget $B$ (prepared slots)', ylabel=r'Held-out $J_\lambda=\mathbb{E}[V-\lambda D]$', title=fr'$\theta={theta}$, $\lambda=\eta_B={ETA_B:.1f}$/\$', xlim=(1, B[-1]))
axes[0].legend(loc='lower right', frameon=False, fontsize=6.5)
fig.savefig(os.path.join(FIG, "penalized.pdf")); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.3), layout='constrained')
ax = axes[0]
for z, col in zip([1.5, 2.0, 3.0], [BLUE, GREEN, RED]):
    rec = Z[f"fine|z={z}|omega=1.0"]; E = np.array(rec["E"]); u = np.array(rec["uncovered"])
    ax.loglog(1 + E, u, color=col, label=f'$z={z}$')
    i0 = int(np.argmin(np.abs(E - 1024))); c = u[i0] * (1 + E[i0]) ** (1 - 1 / z); ax.loglog(1 + E, c * (1 + E) ** (-(1 - 1 / z)), color=col, ls='--', lw=.8)
ax.axvspan(1, 8, color=GREY, alpha=.15, lw=0); ax.text(2.2, 2e-3, 'allocation\nrange $F\\le7$', fontsize=7)
ax.set(xlabel='$1+E$ (TTL epochs)', ylabel='Uncovered mass $1-p(E)$'); ax.legend(loc='lower left', frameon=False)
ax = axes[1]
wnames = list(windows := {"F_range_0_7": "$[0,7]$", "E_1_63": "$[1,63]$", "E_64_1023": "$[64,10^3]$", "E_1024_16383": "$[10^3,1.6\\cdot10^4]$", "E_16384_2e5": "$[1.6\\cdot10^4,2\\cdot10^5]$"})
xs = np.arange(len(wnames))
for i, (z, col) in enumerate(zip([1.5, 2.0, 3.0], [BLUE, GREEN, RED])):
    rec = Z[f"fine|z={z}|omega=1.0"]
    ax.plot(xs - .1 + .1 * i, [rec["fits"][w]["r_loglog"] for w in wnames], 'o', color=col, ms=3.5, label=f'log-log slope, $z={z}$')
    ax.plot(xs - .1 + .1 * i, [rec["fits"][w]["r_ls_in_p"] for w in wnames], 'x', color=col, ms=4, label=f'LS-in-$p$ fit, $z={z}$')
    ax.axhline(1 - 1 / z, color=col, lw=.65, ls=':')
ax.set_xticks(xs); ax.set_xticklabels([windows[w] for w in wnames], fontsize=6.5)
ax.set(xlabel='Fitting window for $E$', ylabel='Fitted $r$ (dotted: $1-1/z$)'); ax.legend(loc='upper left', frameon=False, fontsize=6, ncol=2)
fig.savefig(os.path.join(FIG, "zipf_exponent.pdf")); plt.close(fig)

# =============================================================================
# 11. Summary table and results.json
# =============================================================================
summary = []
for tag, o in HB.items():
    g = o["gaps"]
    summary.append(dict(config=tag, K=o["K"], r_common_literal=o["fits"]["power_common_literal"]["r"][0], r_common_free=o["fits"]["power_common_free"]["r"][0],
                        rmse_common_free=o["fits"]["power_common_free"]["rmse"], rmse_common_literal=o["fits"]["power_common_literal"]["rmse"],
                        rmse_own=o["fits"]["power_own_free"]["rmse"], rmse_exp=o["fits"]["exp_own_free"]["rmse"],
                        r_own_min=float(min(o["fits"]["power_own_free"]["r"])), r_own_max=float(max(o["fits"]["power_own_free"]["r"])),
                        n_nonconcave_probe=o["n_nonconcave_probe"], n_never_hit=len(o["never_hit_positions"]),
                        **{f"{k}_mean": v["mean_gap"] for k, v in g.items()}, **{f"{k}_ci_lo": v["ci"][0] for k, v in g.items()}, **{f"{k}_ci_hi": v["ci"][1] for k, v in g.items()},
                        prop_slope=o["proportionality_mean_slope"], prop_spearman=o["proportionality_mean_spearman"], prop_pred_slope=o["proportionality_predicted_slope"],
                        slsqp_gap=o["slsqp_vs_closed_form_maxabs"], active_set_agree=o["active_set_vs_sorted_agree_frac"], exp_formula_gap=o["exp_all_active_formula_vs_bisect_maxabs"]))
pd.DataFrame(summary).to_csv(os.path.join(HERE, "summary_table.csv"), index=False)
results["summary_table"] = summary
results["runtime_s"] = time.time() - T0
with open(os.path.join(HERE, "results.json"), "w") as f: json.dump(results, f, indent=1, default=lambda x: tl(x) if isinstance(x, np.ndarray) else float(x) if isinstance(x, (np.floating,)) else int(x) if isinstance(x, np.integer) else str(x))
log("done; results.json written; runtime", round(results["runtime_s"], 1), "s")

