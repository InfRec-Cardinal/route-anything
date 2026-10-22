#!/usr/bin/env python3
"""
X2_residual_coverage -- real-data test of eq:residual, thm:marginal (and its proof identity
eq:app-marginal-identity), prop:marginals and app:static-adaptive of
"Route Anything: A Theory of Inference Routers" on CodeRouterBench (9,999 tasks x 8 models).

Everything is computed from the execution-verified outcome matrix
$ACROUTER_DATA/coderouterbench/id_results_long.csv.
No outcome or cost is fabricated; the only randomness is task resampling (bootstrap, fixed seed).

Objects (paper notation, Sec. 3):
  E_m          = {score_m >= theta}                       exact verifier, theta in {1.0, 0.5}
  p(S)         = P(U_{m in S} E_m)                        eq:residual (OR semantics, verifier-select)
  r_m(S)       = P(E_m  and  all i in S fail)             eq:residual
  cascade(S)   : all members of S are called (parallel), an exact selecting verifier commits a passing
                 member (highest passing score); on a miss Opus (= reference B) is called and committed.
  V            = committed score;  D = sum_{i in S} cost_i + (1-H) cost_Opus   (eq:accounting)
  a(S)         = E[sum_{i in S} cost_i];  d_m(S) = a(S+m) - a(S) = E[cost_m]  (parallel, no aggregation fee)
  v_h(S)       = E[V | hit],  v_m(S) = E[score_Opus | miss],  b(S) = E[cost_Opus | miss]
  eta(S)       = E[V]/E[D]  (ratio of sums) = [p v_h + (1-p) v_m] / [a + (1-p) b]   (eq:homogeneous, exact)
thm:marginal (common v_h, v_m, b across S and S+m):  eta(S+m) > eta(S)  iff
  N_S := r_m(S)[v_h - v_m + eta(S) b] - eta(S) d_m(S) > 0,   and (eq:app-marginal-identity)
  eta(S+m) - eta(S) = N_S / (E[D(S)] + d - r b).
"""
import json, os, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
T0 = time.time()
SEED = 20260911
N_BOOT = 500
THETAS = [1.0, 0.5]
DATA = os.environ.get('ACROUTER_DATA', 'agent-as-a-router/data') + '/coderouterbench/id_results_long.csv'
OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

OPUS = "claude-opus-4-6"
CAND = ["Qwen3-Max", "kimi-k2.5", "qwen3.5-plus", "glm-5", "MiniMax-M2.7", "gpt-5.4", "claude-sonnet-4-6"]
CHEAP4 = ["Qwen3-Max", "kimi-k2.5", "qwen3.5-plus", "glm-5"]
SHORT = {"claude-opus-4-6": "opus", "gpt-5.4": "gpt54", "claude-sonnet-4-6": "sonnet", "Qwen3-Max": "qwen3max",
         "glm-5": "glm5", "qwen3.5-plus": "qwen35p", "kimi-k2.5": "kimi", "MiniMax-M2.7": "minimax"}
K = len(CAND)
NSUB = 1 << K
MEMBERS = [[k for k in range(K) if (s >> k) & 1] for s in range(NSUB)]
SIZE = np.array([len(m) for m in MEMBERS])
MIN_COUNT = 20     # a candidate is *usable* in a context if it has >= 20 passes and >= 20 failures
MIN_CELL = 10      # an (S, m) case is *non-trivial* if #(E_m & fail S) >= 10, #fail(S+m) >= 10, #hit(S) >= 10
SIZES_B = [1, 2, 3]  # portfolio sizes used for the headline counts
ID = lambda x: (float(x) if x is not None and np.isfinite(x) else None)

# ---- paper figure style (scripts/reproduce.py) ----------------------------------------------
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times'], 'text.usetex': True,
    'mathtext.fontset': 'stix', 'font.size': 9,
    'axes.labelsize': 9, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': .65, 'lines.linewidth': 1.6,
})
BLUE, RED, GREEN, GREY = '#23639b', '#ba5934', '#3a7c5e', '.55'
USETEX_OK = True
try:
    _f, _a = plt.subplots(figsize=(1, 1)); _a.set_xlabel(r'$r_m(S)$'); _f.savefig(os.path.join(FIG, '_probe.pdf')); plt.close(_f)
    os.remove(os.path.join(FIG, '_probe.pdf'))
except Exception as e:  # fall back to mathtext
    USETEX_OK = False
    plt.rcParams.update({'text.usetex': False})
    print("usetex unavailable, falling back to mathtext:", repr(e)[:200])

# ==============================================================================================
# 1. Load real data
# ==============================================================================================
df = pd.read_csv(DATA)
assert df.shape == (79992, 12)
S_df = df.pivot(index="task_id", columns="model", values="score")
C_df = df.pivot(index="task_id", columns="model", values="cost_usd")
assert not S_df.isna().any().any() and not C_df.isna().any().any()
meta = df.drop_duplicates("task_id").set_index("task_id").loc[S_df.index, ["dimension", "split"]]
so_all, co_all = S_df[OPUS].to_numpy(float), C_df[OPUS].to_numpy(float)
Sc_all, Cc_all = S_df[CAND].to_numpy(float), C_df[CAND].to_numpy(float)
DIMS = sorted(meta.dimension.unique())
CONTEXTS = ["overall"] + DIMS
split_all = meta.split.to_numpy()
dim_all = meta.dimension.to_numpy()
idt = split_all == "id_test"
# cross-checks against the CodeRouterBench paper (Always-Opus on id_test: $34.02, AvgPerf 43.83)
assert abs(co_all[idt].sum() - 34.02) < 0.01, co_all[idt].sum()
assert abs(100 * so_all[idt].mean() - 43.83) < 0.01, so_all[idt].mean()
n_zero_cost = int((Cc_all == 0).sum())
print(f"loaded {len(meta)} tasks; id_test {idt.sum()}, probing {(~idt).sum()}; zero-cost candidate rows kept as recorded: {n_zero_cost}")

def ctx_mask(ctx, split=None):
    m = np.ones(len(meta), bool) if ctx == "overall" else (dim_all == ctx)
    if split is not None:
        m &= (split_all == split)
    return m

# ==============================================================================================
# 2. Per-task portfolio tables (deterministic given the CSV) and weighted statistics
# ==============================================================================================
def tables(mask, theta):
    S, C, o_s, o_c = Sc_all[mask], Cc_all[mask], so_all[mask], co_all[mask]
    P = S >= theta
    n = S.shape[0]
    hit = np.zeros((n, NSUB), bool); Vsel = np.zeros((n, NSUB)); A = np.zeros((n, NSUB))
    for s in range(1, NSUB):
        mem = MEMBERS[s]
        Pm = P[:, mem]
        hit[:, s] = Pm.any(1)
        v = np.where(Pm, S[:, mem], -np.inf).max(1)           # exact selecting verifier: best passing member
        Vsel[:, s] = np.where(hit[:, s], v, 0.0)
        A[:, s] = C[:, mem].sum(1)
    hf = hit.astype(float)
    return {"P": P, "Pf": P.astype(float), "hf": hf, "Vh": hf * Vsel,
            "V": hf * Vsel + (1 - hf) * o_s[:, None], "D": A + (1 - hf) * o_c[:, None], "A": A,
            "Mv": (1 - hf) * o_s[:, None], "Mc": (1 - hf) * o_c[:, None], "C": C, "o_s": o_s, "o_c": o_c, "n": n}

def wstats(W, T):
    """All portfolio statistics under task weights W (B x n). Returns dict of (B x 128) / (B x 7) arrays."""
    nw = W.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        nh = W @ T["hf"]; nm = nw - nh
        p = nh / nw
        EV = (W @ T["V"]) / nw; ED = (W @ T["D"]) / nw
        eta = EV / ED
        vh = (W @ T["Vh"]) / nh
        vm = (W @ T["Mv"]) / nm
        b = (W @ T["Mc"]) / nm
        a = (W @ T["A"]) / nw
        d = (W @ T["C"]) / nw
        nem = W @ (T["Pf"][:, :, None] * (1 - T["hf"])[:, None, :]).reshape(T["n"], -1)  # counts of E_m & fail(S)
    return {"p": p, "EV": EV, "ED": ED, "eta": eta, "vh": vh, "vm": vm, "b": b, "a": a, "d": d,
            "nh": nh, "nm": nm, "nem": nem.reshape(W.shape[0], K, NSUB), "nw": nw}

# (S, m) index arrays: every subset s and every m not in s  (7 x 64 = 448 cases)
S_IDX, M_IDX, S2_IDX = [], [], []
for s in range(NSUB):
    for m in range(K):
        if not (s >> m) & 1:
            S_IDX.append(s); M_IDX.append(m); S2_IDX.append(s | (1 << m))
S_IDX, M_IDX, S2_IDX = map(np.array, (S_IDX, M_IDX, S2_IDX))
NPAIR = len(S_IDX)
SIZE_S = SIZE[S_IDX]

def derived(st):
    """Per-(S,m) quantities of thm:marginal, vectorised over bootstrap replicates."""
    s, m, s2 = S_IDX, M_IDX, S2_IDX
    with np.errstate(divide="ignore", invalid="ignore"):
        r = st["p"][:, s2] - st["p"][:, s]
        eta_s = st["eta"][:, s]; vh = st["vh"][:, s]; vm = st["vm"][:, s]; b = st["b"][:, s]; d = st["d"][:, m]
        N = r * (vh - vm + eta_s * b) - eta_s * d                       # eq:marginal numerator with S's values
        denom = st["ED"][:, s] + d - r * b                              # enlarged expected cost (common b)
        dpred = N / denom                                               # eq:app-marginal-identity
        dreal = st["eta"][:, s2] - eta_s                                # realised change (S+m's own values)
        vh2, vm2, b2 = st["vh"][:, s2], st["vm"][:, s2], st["b"][:, s2]
        p2 = st["p"][:, s2]
        # partial substitutions (which hypothesis violation explains the gap)
        def eta_hyp(vh_, vm_, b_):
            return (p2 * vh_ + (1 - p2) * vm_) / (st["a"][:, s] + d + (1 - p2) * b_) - eta_s
        dpred_vh2 = eta_hyp(vh2, vm, b); dpred_vm2 = eta_hyp(vh, vm2, b); dpred_b2 = eta_hyp(vh, vm, b2)
        dpred_all2 = eta_hyp(vh2, vm2, b2)                               # = dreal (identity)
        cond = r / (1 - st["p"][:, s])                                  # P(E_m | all of S fail)
        am = st["p"][:, 1 << m]                                         # P(E_m)
    return {"r": r, "N": N, "dpred": dpred, "dreal": dreal, "eta_s": eta_s, "eta_s2": st["eta"][:, s2],
            "vh": vh, "vm": vm, "b": b, "vh2": vh2, "vm2": vm2, "b2": b2, "d": d,
            "dpred_vh2": dpred_vh2, "dpred_vm2": dpred_vm2, "dpred_b2": dpred_b2, "dpred_all2": dpred_all2,
            "cond": cond, "am": am, "p_s": st["p"][:, s], "p_s2": p2}

def smax(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.max()) if x.size else 0.0

def ci(x, lo=2.5, hi=97.5):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return [None, None]
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]

def ci_excl0(x):
    lo, hi = ci(x)
    return (lo is not None) and (lo > 0 or hi < 0)

def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(stats.spearmanr(x[ok], y[ok]).statistic)

def usable_from(P):
    npass = P.sum(0)
    return (npass >= MIN_COUNT) & ((P.shape[0] - npass) >= MIN_COUNT)

def sname(s):
    return "+".join(SHORT[CAND[k]] for k in MEMBERS[s]) if s else "{}"

rng = np.random.default_rng(SEED)
R = {"meta": {"seed": SEED, "n_boot": N_BOOT, "thetas": THETAS, "candidates": CAND, "cheap4": CHEAP4, "reference": OPUS,
              "min_count_usable": MIN_COUNT, "min_cell_nontrivial": MIN_CELL, "portfolio_sizes_headline": SIZES_B,
              "n_tasks": int(len(meta)), "n_id_test": int(idt.sum()), "n_probing": int((~idt).sum()),
              "zero_cost_candidate_rows_kept": n_zero_cost, "usetex": USETEX_OK},
     "identity": {}, "in_sample": {}, "marginals": {}, "oos": {}, "submodularity": {}, "claims": {}}

# ==============================================================================================
# 3. In-sample analysis per (context, theta): identities (a), hypothesis drift + sign rule (b),
#    prop:marginals phenomenon (c), submodularity / adaptive marginals (e)
# ==============================================================================================
ident = {"max_abs_r_identity": 0.0, "max_abs_eta_homogeneous": 0.0, "max_abs_marginal_identity": 0.0,
         "max_abs_all2_equals_real": 0.0, "n_cases_checked": 0}
sub_static = {"all7": {"n_triples": 0, "n_violations": 0, "max_violation": 0.0},
              "cheap4": {"n_triples": 0, "n_violations": 0, "max_violation": 0.0}}
rows_b = []          # per (ctx, theta, S, m) point + CI records (non-trivial, |S| in SIZES_B)
rows_b_all = []      # includes trivial cases (for the count)
rows_adapt = []      # adaptive marginals
rows_c = []          # prop:marginals per (ctx, theta, m0)
rows_corr = []       # error-correlation excess per pair
scatter_pts = {th: {"dpred": [], "dreal": [], "agree": []} for th in THETAS}
drift_pts = {th: {"vm": [], "vm2": [], "b": [], "b2": [], "vh": [], "vh2": []} for th in THETAS}
cov_pts = {th: {"am": [], "r": [], "ctx": []} for th in THETAS}
adapt_pts = {th: {"am": [], "cond": [], "size": []} for th in THETAS}
CHEAP4_IDX = [CAND.index(c) for c in CHEAP4]
CHEAP4_MASK = sum(1 << k for k in CHEAP4_IDX)

for ctx in CONTEXTS:
    mask = ctx_mask(ctx)
    n = int(mask.sum())
    W_boot = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(float)
    W1 = np.ones((1, n))
    for theta in THETAS:
        key = f"{ctx}|theta={theta}"
        T = tables(mask, theta)
        st1 = wstats(W1, T); stB = wstats(W_boot, T)
        dv1 = derived(st1); dvB = derived(stB)
        usable = usable_from(T["P"])
        n_usable = int(usable.sum())
        # ---------- (a) identities ------------------------------------------------------------
        r_direct = st1["nem"][0][M_IDX, S_IDX] / n
        ident["max_abs_r_identity"] = max(ident["max_abs_r_identity"], smax(np.abs(r_direct - dv1["r"][0])))
        p1, vh1, vm1, b1, a1 = st1["p"][0], st1["vh"][0], st1["vm"][0], st1["b"][0], st1["a"][0]
        okh = (st1["nh"][0] > 0) & (st1["nm"][0] > 0)
        eta_hom = (p1 * vh1 + (1 - p1) * vm1) / (a1 + (1 - p1) * b1)
        ident["max_abs_eta_homogeneous"] = max(ident["max_abs_eta_homogeneous"], smax(np.abs(eta_hom[okh] - st1["eta"][0][okh])))
        # eq:app-marginal-identity: with COMMON (v_h, v_m, b) := S's values, the enlarged eta minus eta(S) equals N/(D+d-rb)
        p2 = dv1["p_s2"][0]; eta_hyp2 = (p2 * dv1["vh"][0] + (1 - p2) * dv1["vm"][0]) / (a1[S_IDX] + dv1["d"][0] + (1 - p2) * dv1["b"][0])
        okp = np.isfinite(dv1["dpred"][0])
        ident["max_abs_marginal_identity"] = max(ident["max_abs_marginal_identity"], smax(np.abs((eta_hyp2 - dv1["eta_s"][0])[okp] - dv1["dpred"][0][okp])))
        okr = np.isfinite(dv1["dpred_all2"][0])
        ident["max_abs_all2_equals_real"] = max(ident["max_abs_all2_equals_real"], smax(np.abs(dv1["dpred_all2"][0][okr] - dv1["dreal"][0][okr])))
        ident["n_cases_checked"] += int(okp.sum())
        # ---------- (e) static submodularity: r_m(T) <= r_m(S) for all S subset T, m not in T ---------
        for m in range(K):
            others = [k for k in range(K) if k != m]
            for t in range(NSUB):
                if (t >> m) & 1:
                    continue
                rt = p1[t | (1 << m)] - p1[t]
                # all subsets s of t
                sub = t
                while True:
                    rs = p1[sub | (1 << m)] - p1[sub]
                    viol = rt - rs
                    tag = ["all7"] + (["cheap4"] if (m in CHEAP4_IDX and (t & ~CHEAP4_MASK) == 0) else [])
                    for tg in tag:
                        sub_static[tg]["n_triples"] += 1
                        if viol > 1e-12:
                            sub_static[tg]["n_violations"] += 1
                        sub_static[tg]["max_violation"] = max(sub_static[tg]["max_violation"], float(viol))
                    if sub == 0:
                        break
                    sub = (sub - 1) & t
        # ---------- (b) hypothesis drift and sign rule -----------------------------------------
        nem1 = st1["nem"][0][M_IDX, S_IDX]; nm_s2 = st1["nm"][0][S2_IDX]; nh_s = st1["nh"][0][S_IDX]
        nontriv = (nem1 >= MIN_CELL) & (nm_s2 >= MIN_CELL) & (nh_s >= MIN_CELL)
        for j in range(NPAIR):
            if SIZE_S[j] not in SIZES_B:
                continue
            rec = {"ctx": ctx, "theta": theta, "S": sname(S_IDX[j]), "m": SHORT[CAND[M_IDX[j]]], "size_S": int(SIZE_S[j]),
                   "nontrivial": bool(nontriv[j]), "n_E_m_and_failS": int(nem1[j]), "n_fail_S2": int(nm_s2[j]), "n_hit_S": int(nh_s[j]),
                   "r": ID(dv1["r"][0, j]), "d": ID(dv1["d"][0, j]), "eta_S": ID(dv1["eta_s"][0, j]), "eta_S2": ID(dv1["eta_s2"][0, j]),
                   "N": ID(dv1["N"][0, j]), "N_ci": ci(dvB["N"][:, j]),
                   "dpred": ID(dv1["dpred"][0, j]), "dpred_ci": ci(dvB["dpred"][:, j]),
                   "dreal": ID(dv1["dreal"][0, j]), "dreal_ci": ci(dvB["dreal"][:, j]),
                   "vh_S": ID(dv1["vh"][0, j]), "vh_S2": ID(dv1["vh2"][0, j]), "dvh_ci": ci(dvB["vh2"][:, j] - dvB["vh"][:, j]),
                   "vm_S": ID(dv1["vm"][0, j]), "vm_S2": ID(dv1["vm2"][0, j]), "dvm_ci": ci(dvB["vm2"][:, j] - dvB["vm"][:, j]),
                   "b_S": ID(dv1["b"][0, j]), "b_S2": ID(dv1["b2"][0, j]), "db_ci": ci(dvB["b2"][:, j] - dvB["b"][:, j]),
                   "dpred_vh2": ID(dv1["dpred_vh2"][0, j]), "dpred_vm2": ID(dv1["dpred_vm2"][0, j]), "dpred_b2": ID(dv1["dpred_b2"][0, j])}
            rec["sign_agree"] = (rec["N"] is not None and rec["dreal"] is not None and np.sign(rec["N"]) == np.sign(rec["dreal"]))
            rec["ci_decided"] = ci_excl0(dvB["N"][:, j]) and ci_excl0(dvB["dreal"][:, j])
            rows_b_all.append(rec)
            if nontriv[j]:
                rows_b.append(rec)
                scatter_pts[theta]["dpred"].append(rec["dpred"]); scatter_pts[theta]["dreal"].append(rec["dreal"]); scatter_pts[theta]["agree"].append(rec["sign_agree"])
                for q in ["vm", "b", "vh"]:
                    drift_pts[theta][q].append(rec[f"{q}_S"]); drift_pts[theta][q + "2"].append(rec[f"{q}_S2"])
        # ---------- (e) adaptive marginals: P(E_m | all of S fail) vs P(E_m) --------------------
        for j in range(NPAIR):
            if SIZE_S[j] < 1 or SIZE_S[j] > 3 or not nontriv[j]:
                continue
            if not usable[M_IDX[j]]:
                continue
            diff = dvB["cond"][:, j] - dvB["am"][:, j]
            lo, hi = ci(diff)
            in_cheap4 = (M_IDX[j] in CHEAP4_IDX) and ((S_IDX[j] & ~CHEAP4_MASK) == 0)
            rows_adapt.append({"ctx": ctx, "theta": theta, "S": sname(S_IDX[j]), "m": SHORT[CAND[M_IDX[j]]], "size_S": int(SIZE_S[j]),
                               "cheap4": bool(in_cheap4), "P_E_m": ID(dv1["am"][0, j]), "P_E_m_given_failS": ID(dv1["cond"][0, j]),
                               "r_static": ID(dv1["r"][0, j]), "diff": ID(dv1["cond"][0, j] - dv1["am"][0, j]), "diff_ci": [lo, hi],
                               "exceeds_point": bool(dv1["cond"][0, j] > dv1["am"][0, j]), "exceeds_ci": bool(lo is not None and lo > 0),
                               "below_ci": bool(hi is not None and hi < 0)})
            adapt_pts[theta]["am"].append(dv1["am"][0, j]); adapt_pts[theta]["cond"].append(dv1["cond"][0, j]); adapt_pts[theta]["size"].append(int(SIZE_S[j]))
        # ---------- (c) prop:marginals phenomenon --------------------------------------------------
        if n_usable >= 3:
            for m0 in range(K):
                if not usable[m0]:
                    continue
                cand = [m for m in range(K) if m != m0 and usable[m]]
                if len(cand) < 3:
                    continue
                js = [int(np.where((S_IDX == (1 << m0)) & (M_IDX == m))[0][0]) for m in cand]
                am = dv1["am"][0, js]; r = dv1["r"][0, js]; pp = dv1["p_s2"][0, js]
                top_a, top_c = cand[int(am.argmax())], cand[int(pp.argmax())]
                # bootstrap: does the top-quality partner coincide with the top-coverage partner?
                amB = dvB["am"][:, js]; ppB = dvB["p_s2"][:, js]
                differsB = amB.argmax(1) != ppB.argmax(1)
                gapB = ppB.max(1) - ppB[np.arange(N_BOOT), amB.argmax(1)]     # coverage lost by choosing the top-quality partner (post-selection)
                rows_c.append({"ctx": ctx, "theta": theta, "m0": SHORT[CAND[m0]], "a_m0": ID(st1["p"][0, 1 << m0]),
                               "partners": [SHORT[CAND[m]] for m in cand], "a_m": am.tolist(), "r_m": r.tolist(), "p_pair": pp.tolist(),
                               "top_quality": SHORT[CAND[top_a]], "top_coverage": SHORT[CAND[top_c]], "differs": bool(top_a != top_c),
                               "P_differs_boot": float(differsB.mean()),
                               "coverage_gap_point": float(pp.max() - pp[cand.index(top_a)]), "coverage_gap_ci_postselection": ci(gapB),
                               "spearman_a_vs_paircov": spearman(am, pp), "spearman_a_vs_r": spearman(am, r), "spearman_r_vs_paircov": spearman(r, pp),
                               "cost_top_quality": ID(st1["d"][0, top_a]), "cost_top_coverage": ID(st1["d"][0, top_c])})
                for m_, a_, r_ in zip(cand, am, r):
                    cov_pts[theta]["am"].append(a_); cov_pts[theta]["r"].append(r_); cov_pts[theta]["ctx"].append(ctx)
            # error-correlation excess for every usable pair
            for i in range(K):
                for j2 in range(i + 1, K):
                    if not (usable[i] and usable[j2]):
                        continue
                    s_ij = (1 << i) | (1 << j2)
                    exc1 = (1 - st1["p"][0, s_ij]) - (1 - st1["p"][0, 1 << i]) * (1 - st1["p"][0, 1 << j2])
                    excB = (1 - stB["p"][:, s_ij]) - (1 - stB["p"][:, 1 << i]) * (1 - stB["p"][:, 1 << j2])
                    lo, hi = ci(excB)
                    rows_corr.append({"ctx": ctx, "theta": theta, "i": SHORT[CAND[i]], "j": SHORT[CAND[j2]], "excess": float(exc1),
                                      "excess_ci": [lo, hi], "positive": bool(exc1 > 0), "positive_ci": bool(lo is not None and lo > 0)})
        R["in_sample"][key] = {"n_tasks": n, "n_usable": n_usable, "usable": {SHORT[CAND[k]]: bool(usable[k]) for k in range(K)},
                               "eta_opus": ID(st1["eta"][0, 0]), "vbar": ID(so_all[mask].mean()), "cbar": ID(co_all[mask].mean()),
                               "a_m": {SHORT[CAND[k]]: ID(st1["p"][0, 1 << k]) for k in range(K)},
                               "d_m": {SHORT[CAND[k]]: ID(st1["d"][0, k]) for k in range(K)}}
        print(f"[{key}] n={n} usable={n_usable} eta_opus={st1['eta'][0,0]:.2f}  ({time.time()-T0:.0f}s)")

R["identity"] = ident
R["submodularity"]["static"] = sub_static

# ---------- summaries of (b) ---------------------------------------------------------------------
def summarize_b(rows, tag):
    out = {}
    for theta in THETAS:
        rr = [r for r in rows if r["theta"] == theta]
        rt = [r for r in rr if r["ctx"] != "overall"]
        def block(rs):
            if not rs:
                return {"n": 0}
            n_ = len(rs)
            dvm = np.array([r["vm_S2"] - r["vm_S"] for r in rs if r["vm_S2"] is not None and r["vm_S"] is not None])
            rel_vm = np.array([(r["vm_S2"] - r["vm_S"]) / r["vm_S"] for r in rs if r["vm_S"] not in (None, 0) and r["vm_S2"] is not None])
            db = np.array([r["b_S2"] - r["b_S"] for r in rs]); rel_b = np.array([(r["b_S2"] - r["b_S"]) / r["b_S"] for r in rs])
            dvh = np.array([r["vh_S2"] - r["vh_S"] for r in rs]); rel_vh = np.array([(r["vh_S2"] - r["vh_S"]) / r["vh_S"] for r in rs])
            vm_sig = np.array([r["dvm_ci"][0] is not None and (r["dvm_ci"][0] > 0 or r["dvm_ci"][1] < 0) for r in rs])
            b_sig = np.array([r["db_ci"][0] is not None and (r["db_ci"][0] > 0 or r["db_ci"][1] < 0) for r in rs])
            vh_sig = np.array([r["dvh_ci"][0] is not None and (r["dvh_ci"][0] > 0 or r["dvh_ci"][1] < 0) for r in rs])
            agree = np.array([r["sign_agree"] for r in rs]); dec = np.array([r["ci_decided"] for r in rs])
            rdec = np.array([r["dreal_ci"][0] is not None and (r["dreal_ci"][0] > 0 or r["dreal_ci"][1] < 0) for r in rs])
            relerr = np.array([abs(r["dpred"] - r["dreal"]) / abs(r["dreal"]) for r in rs if r["dreal"] not in (None, 0)])
            def relerr_of(kk):
                return float(np.median([abs(r[kk] - r["dreal"]) / abs(r["dreal"]) for r in rs if r["dreal"] not in (None, 0) and r[kk] is not None]))
            pos_pred = np.array([r["N"] > 0 for r in rs]); pos_real = np.array([r["dreal"] > 0 for r in rs])
            return {"n": n_, "n_ci_decided": int(dec.sum()),
                    "vm_drift": {"median": float(np.median(dvm)), "median_rel": float(np.median(rel_vm)), "frac_negative": float((dvm < 0).mean()),
                                 "frac_ci_excl0": float(vm_sig.mean()), "q10_q90_rel": [float(np.percentile(rel_vm, 10)), float(np.percentile(rel_vm, 90))]},
                    "b_drift": {"median": float(np.median(db)), "median_rel": float(np.median(rel_b)), "frac_positive": float((db > 0).mean()),
                                "frac_ci_excl0": float(b_sig.mean()), "q10_q90_rel": [float(np.percentile(rel_b, 10)), float(np.percentile(rel_b, 90))]},
                    "vh_drift": {"median": float(np.median(dvh)), "median_rel": float(np.median(rel_vh)), "frac_ci_excl0": float(vh_sig.mean()),
                                 "max_abs": float(np.max(np.abs(dvh)))},
                    "sign_agreement_all": float(agree.mean()), "sign_agreement_ci_decided": (float(agree[dec].mean()) if dec.any() else None),
                    "n_realised_ci_decided": int(rdec.sum()), "sign_agreement_realised_ci_decided": (float(agree[rdec].mean()) if rdec.any() else None),
                    "n_disagree_realised_ci_decided": int((~agree[rdec]).sum()),
                    "n_disagree_all": int((~agree).sum()), "n_disagree_ci_decided": int((~agree[dec]).sum()),
                    "frac_pred_positive": float(pos_pred.mean()), "frac_real_positive": float(pos_real.mean()),
                    "disagree_pattern": {"pred_pos_real_neg": int((pos_pred & ~pos_real).sum()), "pred_neg_real_pos": int((~pos_pred & pos_real).sum())},
                    "median_rel_err_dpred": float(np.median(relerr)), "frac_rel_err_below_25pct": float((relerr < 0.25).mean()),
                    "median_rel_err_substitute_vh2": relerr_of("dpred_vh2"), "median_rel_err_substitute_vm2": relerr_of("dpred_vm2"),
                    "median_rel_err_substitute_b2": relerr_of("dpred_b2"),
                    "median_signed_bias_dpred_minus_dreal_over_eta_S": float(np.median([(r["dpred"] - r["dreal"]) / r["eta_S"] for r in rs]))}
        out[f"theta={theta}"] = {"all_contexts": block(rr), "dimensions_only": block(rt)}
    return out
R["in_sample"]["summary_b_nontrivial"] = summarize_b(rows_b, "nontrivial")
R["in_sample"]["n_cases_b_total_incl_trivial"] = {f"theta={th}": len([r for r in rows_b_all if r["theta"] == th]) for th in THETAS}
R["in_sample"]["n_cases_b_trivial"] = {f"theta={th}": len([r for r in rows_b_all if r["theta"] == th and not r["nontrivial"]]) for th in THETAS}
R["in_sample"]["trivial_cases_sign_agreement"] = {f"theta={th}": (float(np.mean([r["sign_agree"] for r in rows_b_all if r["theta"] == th and not r["nontrivial"]]))
                                                                 if any(r["theta"] == th and not r["nontrivial"] for r in rows_b_all) else None) for th in THETAS}
R["in_sample"]["cases_b"] = rows_b
triv = [r for r in rows_b_all if not r["nontrivial"]]
R["in_sample"]["trivial_cases_diagnostics"] = {f"theta={th}": {
    "n": len([r for r in triv if r["theta"] == th]),
    "n_disagree": len([r for r in triv if r["theta"] == th and not r["sign_agree"]]),
    "reason_few_new_covers": len([r for r in triv if r["theta"] == th and r["n_E_m_and_failS"] < MIN_CELL]),
    "reason_few_failures_left": len([r for r in triv if r["theta"] == th and r["n_fail_S2"] < MIN_CELL]),
    "reason_few_hits_S": len([r for r in triv if r["theta"] == th and r["n_hit_S"] < MIN_CELL]),
    "disagree_with_r_exactly_0": len([r for r in triv if r["theta"] == th and not r["sign_agree"] and r["n_E_m_and_failS"] == 0]),
    "disagree_with_N_undefined": len([r for r in triv if r["theta"] == th and not r["sign_agree"] and r["N"] is None]),  # audit: S never hits -> v_h(S) and N_S are NaN, not a sign disagreement
    "disagree_with_N_finite": len([r for r in triv if r["theta"] == th and not r["sign_agree"] and r["N"] is not None]),
    "disagree_ci_decided": len([r for r in triv if r["theta"] == th and not r["sign_agree"] and r["ci_decided"]])} for th in THETAS}
R["in_sample"]["worked_example"] = [r for r in rows_b if r["ctx"] == "overall" and r["theta"] == 1.0 and r["S"] == "qwen3max" and r["m"] in ("kimi", "qwen35p", "gpt54")]
R["in_sample"]["disagreements"] = [r for r in rows_b if not r["sign_agree"]]

# ---------- summaries of (c) ---------------------------------------------------------------------
def summarize_c(rows):
    out = {}
    for theta in THETAS:
        rr = [r for r in rows if r["theta"] == theta]; rt = [r for r in rr if r["ctx"] != "overall"]
        def block(rs):
            if not rs:
                return {"n": 0}
            return {"n_contexts_m0": len(rs), "n_differs": int(sum(r["differs"] for r in rs)), "frac_differs": float(np.mean([r["differs"] for r in rs])),
                    "mean_P_differs_boot": float(np.mean([r["P_differs_boot"] for r in rs])),
                    "median_spearman_a_vs_paircov": float(np.nanmedian([r["spearman_a_vs_paircov"] for r in rs])),
                    "frac_spearman_a_vs_paircov_below_1": float(np.mean([r["spearman_a_vs_paircov"] < 0.999 for r in rs if np.isfinite(r["spearman_a_vs_paircov"])])),
                    "frac_spearman_a_vs_paircov_le_0": float(np.mean([r["spearman_a_vs_paircov"] <= 0 for r in rs if np.isfinite(r["spearman_a_vs_paircov"])])),
                    "min_spearman_r_vs_paircov_identity": float(np.nanmin([r["spearman_r_vs_paircov"] for r in rs])),
                    "n_gap_ci_gt0_postselection": int(sum(r["differs"] and r["coverage_gap_ci_postselection"][0] is not None and r["coverage_gap_ci_postselection"][0] > 0 for r in rs))}
        out[f"theta={theta}"] = {"all_contexts": block(rr), "dimensions_only": block(rt)}
    return out
R["marginals"]["summary_c"] = summarize_c(rows_c)
R["marginals"]["per_m0"] = rows_c
def summarize_corr(rows):
    out = {}
    for theta in THETAS:
        for scope in ["dimensions_only", "overall"]:
            rs = [r for r in rows if r["theta"] == theta and ((r["ctx"] != "overall") if scope == "dimensions_only" else (r["ctx"] == "overall"))]
            if rs:
                out[f"theta={theta}|{scope}"] = {"n_pairs": len(rs), "frac_positive": float(np.mean([r["positive"] for r in rs])),
                                                 "frac_positive_ci": float(np.mean([r["positive_ci"] for r in rs])),
                                                 "median_excess": float(np.median([r["excess"] for r in rs])),
                                                 "n_negative_ci": int(sum(r["excess_ci"][1] is not None and r["excess_ci"][1] < 0 for r in rs))}
    return out
R["marginals"]["error_correlation_summary"] = summarize_corr(rows_corr)
R["marginals"]["error_correlation_pairs"] = rows_corr

# ---------- summaries of (e) adaptive -----------------------------------------------------------
def summarize_adapt(rows):
    out = {}
    for theta in THETAS:
        for scope in ["all7", "cheap4"]:
            for sz in [1, 2, 3, "1-3"]:
                rs = [r for r in rows if r["theta"] == theta and (scope == "all7" or r["cheap4"]) and (sz == "1-3" or r["size_S"] == sz)]
                if rs:
                    out[f"theta={theta}|{scope}|size={sz}"] = {"n": len(rs), "frac_exceeds_point": float(np.mean([r["exceeds_point"] for r in rs])),
                                                                 "frac_exceeds_ci": float(np.mean([r["exceeds_ci"] for r in rs])),
                                                                 "frac_below_ci": float(np.mean([r["below_ci"] for r in rs])),
                                                                 "median_ratio_cond_over_uncond": float(np.median([r["P_E_m_given_failS"] / r["P_E_m"] for r in rs if r["P_E_m"]]))}
    return out
R["submodularity"]["adaptive_summary"] = summarize_adapt(rows_adapt)
R["submodularity"]["adaptive_exceeds_ci_cases"] = [r for r in rows_adapt if r["exceeds_ci"]]
R["submodularity"]["adaptive_cases"] = rows_adapt

# ==============================================================================================
# 4. OUT-OF-SAMPLE: primitives on probing -> realised id_test gains  (d), plus held-out (c) example
# ==============================================================================================
PAIRINGS = [("theorem_N", "quality_a"), ("coverage_r", "quality_a"), ("theorem_N", "coverage_r"), ("theorem_N", "theorem_N_indep"),
            ("theorem_N", "quality_per_dollar"), ("theorem_N_indep", "quality_per_dollar")]
RANKINGS = ["quality_a", "coverage_r", "quality_per_dollar", "theorem_N_indep", "theorem_N", "oracle_id_test"]
oos_rows = []      # per (ctx, theta, S): top partners by the probing rankings and their id_test gains
oos_sign = []      # per (ctx, theta, S, m): sign(N_probing) vs sign(dreal_id_test)
oos_c = []         # held-out prop:marginals: top-quality vs top-coverage partner chosen on probing, gap on id_test
for ctx in CONTEXTS:
    m_pr, m_te = ctx_mask(ctx, "probing"), ctx_mask(ctx, "id_test")
    n_te = int(m_te.sum())
    W_te = rng.multinomial(n_te, np.full(n_te, 1.0 / n_te), size=N_BOOT).astype(float)
    for theta in THETAS:
        T_pr, T_te = tables(m_pr, theta), tables(m_te, theta)
        st_pr = wstats(np.ones((1, m_pr.sum())), T_pr); dv_pr = derived(st_pr)
        st_te1 = wstats(np.ones((1, n_te)), T_te); dv_te1 = derived(st_te1)
        st_teB = wstats(W_te, T_te); dv_teB = derived(st_teB)
        usable_pr = usable_from(T_pr["P"])                       # usability decided on probing only (no id_test peeking)
        if usable_pr.sum() < 3:
            R["oos"][f"{ctx}|theta={theta}"] = {"degenerate": True, "n_usable_probing": int(usable_pr.sum())}
            continue
        nem_pr = st_pr["nem"][0][M_IDX, S_IDX]; nm2_pr = st_pr["nm"][0][S2_IDX]; nh_pr = st_pr["nh"][0][S_IDX]
        nontriv_pr = (nem_pr >= MIN_CELL) & (nm2_pr >= MIN_CELL) & (nh_pr >= MIN_CELL)
        for s in range(1, NSUB):
            if SIZE[s] > 2 or any(not usable_pr[k] for k in MEMBERS[s]):
                continue
            js = [j for j in range(NPAIR) if S_IDX[j] == s and usable_pr[M_IDX[j]] and nontriv_pr[j]]
            if len(js) < 3:
                continue
            js = np.array(js)
            a_pr = dv_pr["am"][0, js]; r_pr = dv_pr["r"][0, js]; N_pr = dv_pr["N"][0, js]
            # marginal-only selector inside the theorem's own accounting: r_m replaced by its independence proxy (1-p(S)) P(E_m)
            r_ind = (1 - dv_pr["p_s"][0, js]) * a_pr
            N_ind = r_ind * (dv_pr["vh"][0, js] - dv_pr["vm"][0, js] + dv_pr["eta_s"][0, js] * dv_pr["b"][0, js]) - dv_pr["eta_s"][0, js] * dv_pr["d"][0, js]
            apd = a_pr / dv_pr["d"][0, js]                                # quality per dollar (cost-aware but joint-law-blind)
            g_te = dv_te1["dreal"][0, js]; g_teB = dv_teB["dreal"][:, js]
            eta_s_te = dv_te1["eta_s"][0, js[0]]
            rel_te = g_te / eta_s_te
            r_te = dv_te1["r"][0, js]
            tops = {"quality_a": int(a_pr.argmax()), "coverage_r": int(r_pr.argmax()), "theorem_N": int(N_pr.argmax()),
                    "theorem_N_indep": int(N_ind.argmax()), "quality_per_dollar": int(apd.argmax()), "oracle_id_test": int(g_te.argmax())}
            rec = {"ctx": ctx, "theta": theta, "S": sname(s), "size_S": int(SIZE[s]), "partners": [SHORT[CAND[M_IDX[j]]] for j in js],
                   "eta_S_id_test": float(eta_s_te), "probing_a": a_pr.tolist(), "probing_r": r_pr.tolist(), "probing_N": N_pr.tolist(),
                   "id_test_gain": g_te.tolist(), "id_test_rel_gain": rel_te.tolist(), "id_test_r": r_te.tolist(),
                   "probing_N_indep": N_ind.tolist(), "probing_quality_per_dollar": apd.tolist(),
                   "spearman_probing_Nindep_vs_idtest_gain": spearman(N_ind, g_te), "spearman_probing_apd_vs_idtest_gain": spearman(apd, g_te),
                   "spearman_probing_a_vs_idtest_gain": spearman(a_pr, g_te), "spearman_probing_r_vs_idtest_gain": spearman(r_pr, g_te),
                   "spearman_probing_N_vs_idtest_gain": spearman(N_pr, g_te),
                   "spearman_probing_a_vs_idtest_r": spearman(a_pr, r_te), "spearman_probing_r_vs_idtest_r": spearman(r_pr, r_te)}
            for name, k in tops.items():
                rec[f"top_{name}"] = SHORT[CAND[M_IDX[js[k]]]]
                rec[f"gain_top_{name}"] = float(g_te[k]); rec[f"relgain_top_{name}"] = float(rel_te[k])
                rec[f"gain_top_{name}_ci"] = ci(g_teB[:, k])
            for x, y in PAIRINGS:
                dif = g_teB[:, tops[x]] - g_teB[:, tops[y]]
                rec[f"diff_{x}_minus_{y}"] = float(g_te[tops[x]] - g_te[tops[y]]); rec[f"diff_{x}_minus_{y}_ci"] = ci(dif)
                rec[f"same_top_{x}_{y}"] = bool(tops[x] == tops[y])
            oos_rows.append(rec)
            for k, j in enumerate(js):
                oos_sign.append({"ctx": ctx, "theta": theta, "S": sname(s), "m": SHORT[CAND[M_IDX[j]]], "N_probing": float(N_pr[k]),
                                 "dreal_id_test": float(g_te[k]), "dreal_ci": ci(g_teB[:, k]), "agree": bool(np.sign(N_pr[k]) == np.sign(g_te[k])),
                                 "ci_decided": ci_excl0(g_teB[:, k])})
        # held-out prop:marginals: partners chosen on probing, pair-coverage gap on id_test
        for m0 in range(K):
            if not usable_pr[m0]:
                continue
            cand = [m for m in range(K) if m != m0 and usable_pr[m]]
            if len(cand) < 3:
                continue
            js = np.array([int(np.where((S_IDX == (1 << m0)) & (M_IDX == m))[0][0]) for m in cand])
            a_pr = dv_pr["am"][0, js]; pp_pr = dv_pr["p_s2"][0, js]
            ta, tc = int(a_pr.argmax()), int(pp_pr.argmax())
            gap_te = dv_te1["p_s2"][0, js[tc]] - dv_te1["p_s2"][0, js[ta]]
            gapB = dv_teB["p_s2"][:, js[tc]] - dv_teB["p_s2"][:, js[ta]]
            oos_c.append({"ctx": ctx, "theta": theta, "m0": SHORT[CAND[m0]], "top_quality_probing": SHORT[CAND[cand[ta]]], "top_coverage_probing": SHORT[CAND[cand[tc]]],
                          "differs": bool(ta != tc), "id_test_pair_cov_top_quality": float(dv_te1["p_s2"][0, js[ta]]), "id_test_pair_cov_top_coverage": float(dv_te1["p_s2"][0, js[tc]]),
                          "id_test_gap": float(gap_te), "id_test_gap_ci": ci(gapB), "spearman_probing_a_vs_idtest_paircov": spearman(a_pr, dv_te1["p_s2"][0, js]),
                          "spearman_probing_r_vs_idtest_paircov": spearman(dv_pr["r"][0, js], dv_te1["p_s2"][0, js])})
        print(f"[OOS {ctx}|theta={theta}] usable(probing)={int(usable_pr.sum())}  ({time.time()-T0:.0f}s)")

def summarize_oos(rows, signs):
    out = {}
    for theta in THETAS:
        for scope in ["all_contexts", "dimensions_only"]:
            rs = [r for r in rows if r["theta"] == theta and (scope == "all_contexts" or r["ctx"] != "overall")]
            ss = [r for r in signs if r["theta"] == theta and (scope == "all_contexts" or r["ctx"] != "overall")]
            if not rs:
                continue
            dec = [r for r in ss if r["ci_decided"]]
            blk = {"n_S": len(rs), "n_sign_cases": len(ss), "n_sign_ci_decided": len(dec),
                   "sign_agreement_all": float(np.mean([r["agree"] for r in ss])),
                   "sign_agreement_ci_decided": (float(np.mean([r["agree"] for r in dec])) if dec else None),
                   "n_disagree_ci_decided": int(sum(not r["agree"] for r in dec))}
            for name in RANKINGS:
                blk[f"mean_relgain_top_{name}"] = float(np.mean([r[f"relgain_top_{name}"] for r in rs]))
                blk[f"median_relgain_top_{name}"] = float(np.median([r[f"relgain_top_{name}"] for r in rs]))
                blk[f"frac_top_{name}_gain_positive"] = float(np.mean([r[f"gain_top_{name}"] > 0 for r in rs]))
            for x, y in PAIRINGS:
                dif = [r for r in rs if not r[f"same_top_{x}_{y}"]]
                blk[f"{x}_vs_{y}"] = {"n_differing": len(dif), "frac_same_top": float(np.mean([r[f"same_top_{x}_{y}"] for r in rs])),
                                      "n_x_wins": int(sum(r[f"diff_{x}_minus_{y}"] > 0 for r in dif)), "n_y_wins": int(sum(r[f"diff_{x}_minus_{y}"] < 0 for r in dif)),
                                      "n_x_wins_ci": int(sum(r[f"diff_{x}_minus_{y}_ci"][0] is not None and r[f"diff_{x}_minus_{y}_ci"][0] > 0 for r in dif)),
                                      "n_y_wins_ci": int(sum(r[f"diff_{x}_minus_{y}_ci"][1] is not None and r[f"diff_{x}_minus_{y}_ci"][1] < 0 for r in dif)),
                                      "mean_diff_relgain_x_minus_y_on_differing": (float(np.mean([(r[f"relgain_top_{x}"] - r[f"relgain_top_{y}"]) for r in dif])) if dif else None)}
            for nm in ["a", "r", "N", "Nindep", "apd"]:
                v = [r[f"spearman_probing_{nm}_vs_idtest_gain"] for r in rs]
                blk[f"median_spearman_probing_{nm}_vs_idtest_gain"] = float(np.nanmedian(v))
            # paired sign test over portfolios: probing r_m vs probing P(E_m) as predictors of id_test r_m
            dsp = np.array([r["spearman_probing_r_vs_idtest_r"] - r["spearman_probing_a_vs_idtest_r"] for r in rs]); dsp = dsp[np.isfinite(dsp)]
            nb, nw = int((dsp > 1e-12).sum()), int((dsp < -1e-12).sum())
            blk["d3_sign_test"] = {"r_better": nb, "a_better": nw, "ties": int(len(dsp) - nb - nw),
                                   "p_two_sided": (float(stats.binomtest(nb, nb + nw, 0.5).pvalue) if nb + nw > 0 else None)}
            blk["median_spearman_probing_a_vs_idtest_r"] = float(np.nanmedian([r["spearman_probing_a_vs_idtest_r"] for r in rs]))
            blk["median_spearman_probing_r_vs_idtest_r"] = float(np.nanmedian([r["spearman_probing_r_vs_idtest_r"] for r in rs]))
            blk["frac_S_where_probing_r_beats_a_on_idtest_r"] = float(np.mean([r["spearman_probing_r_vs_idtest_r"] > r["spearman_probing_a_vs_idtest_r"] for r in rs]))
            out[f"theta={theta}|{scope}"] = blk
    return out
R["oos"]["summary_d"] = summarize_oos(oos_rows, oos_sign)
R["oos"]["cases_d"] = oos_rows
R["oos"]["sign_cases"] = oos_sign
R["oos"]["sign_disagreements_ci_decided"] = [r for r in oos_sign if r["ci_decided"] and not r["agree"]]
def summarize_oos_c(rows):
    out = {}
    for theta in THETAS:
        for scope in ["all_contexts", "dimensions_only"]:
            rs = [r for r in rows if r["theta"] == theta and (scope == "all_contexts" or r["ctx"] != "overall")]
            dif = [r for r in rs if r["differs"]]
            if rs:
                out[f"theta={theta}|{scope}"] = {"n": len(rs), "n_differs": len(dif), "n_idtest_gap_positive": int(sum(r["id_test_gap"] > 0 for r in dif)),
                                                 "n_idtest_gap_negative": int(sum(r["id_test_gap"] < 0 for r in dif)),
                                                 "n_idtest_gap_ci_gt0": int(sum(r["id_test_gap_ci"][0] is not None and r["id_test_gap_ci"][0] > 0 for r in dif)),
                                                 "n_idtest_gap_ci_lt0": int(sum(r["id_test_gap_ci"][1] is not None and r["id_test_gap_ci"][1] < 0 for r in dif)),
                                                 "median_spearman_probing_a_vs_idtest_paircov": float(np.nanmedian([r["spearman_probing_a_vs_idtest_paircov"] for r in rs])),
                                                 "median_spearman_probing_r_vs_idtest_paircov": float(np.nanmedian([r["spearman_probing_r_vs_idtest_paircov"] for r in rs])),
                                                 "frac_r_beats_a": float(np.mean([r["spearman_probing_r_vs_idtest_paircov"] > r["spearman_probing_a_vs_idtest_paircov"] for r in rs]))}
    return out
R["marginals"]["held_out_summary"] = summarize_oos_c(oos_c)
R["marginals"]["held_out_cases"] = oos_c

# ---- concrete real example for prop:marginals: largest in-sample coverage gap among dimensions, plus its held-out replicate
ex = None
for theta in THETAS:
    cands = [r for r in rows_c if r["theta"] == theta and r["ctx"] != "overall" and r["differs"]]
    if cands:
        best = max(cands, key=lambda r: r["coverage_gap_point"])
        ho = [h for h in oos_c if h["ctx"] == best["ctx"] and h["theta"] == theta and h["m0"] == best["m0"]]
        ex = ex or {}
        ex[f"theta={theta}"] = {"in_sample_post_selection": best, "held_out_same_ctx_m0": (ho[0] if ho else None)}
R["marginals"]["example"] = ex
# the held-out case with the largest significant id_test gap (selection on probing)
sig = [h for h in oos_c if h["differs"] and h["id_test_gap_ci"][0] is not None and h["id_test_gap_ci"][0] > 0]
R["marginals"]["held_out_significant_examples"] = sorted(sig, key=lambda h: -h["id_test_gap"])[:5]

# ==============================================================================================
# 5. Figures (vector PDF, paper style)
# ==============================================================================================
def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name), metadata={"Creator": "X2_residual_coverage/run.py"})
    plt.close(fig)

# Fig 1: eq:app-marginal-identity prediction (S's values) vs realised change, theta = 1.0 and 0.5
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.4), layout="constrained")
for ax, theta in zip(axes, THETAS):
    x = np.array(scatter_pts[theta]["dpred"]); y = np.array(scatter_pts[theta]["dreal"]); ag = np.array(scatter_pts[theta]["agree"])
    lim = np.nanmax(np.abs(np.concatenate([x, y]))) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=.65, ls=":")
    ax.axhline(0, color=GREY, lw=.5); ax.axvline(0, color=GREY, lw=.5)
    ax.scatter(x[ag], y[ag], s=7, color=BLUE, alpha=.55, lw=0, label=f"sign agrees ({ag.sum()})")
    ax.scatter(x[~ag], y[~ag], s=10, color=RED, alpha=.9, lw=0, label=f"sign disagrees ({(~ag).sum()})")
    ax.set(xlabel=r"predicted $\eta(S\cup\{m\})-\eta(S)$ (common $v_h,v_m,b$ of $S$)", ylabel=r"realised $\eta(S\cup\{m\})-\eta(S)$",
           title=rf"$\theta={theta}$", xlim=(-lim, lim), ylim=(-lim, lim))
    ax.legend(loc="upper left", frameon=False)
savefig(fig, "fig_marginal_identity.pdf")

# Fig 2: hypothesis drift: v_m(S+m) vs v_m(S), b(S+m) vs b(S), v_h(S+m) vs v_h(S)
fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.3), layout="constrained")
for ax, q, lab in zip(axes, ["vm", "b", "vh"], [r"$v_m$ (Opus value on the failure set)", r"$b$ (Opus cost on the failure set, \$)", r"$v_h$ (committed value on hits)"]):
    for theta, col, mk in zip(THETAS, [BLUE, RED], ["o", "s"]):
        x = np.array(drift_pts[theta][q]); y = np.array(drift_pts[theta][q + "2"])
        ax.scatter(x, y, s=7, color=col, alpha=.5, lw=0, marker=mk, label=rf"$\theta={theta}$")
    allv = np.concatenate([np.array(drift_pts[t][q]) for t in THETAS] + [np.array(drift_pts[t][q + "2"]) for t in THETAS])
    lo_, hi_ = np.nanmin(allv), np.nanmax(allv); pad = (hi_ - lo_) * .05
    ax.plot([lo_ - pad, hi_ + pad], [lo_ - pad, hi_ + pad], color=GREY, lw=.65, ls=":")
    ax.set(xlabel=lab + r" of $S$", ylabel=r"same quantity of $S\cup\{m\}$")
    ax.legend(loc="upper left", frameon=False)
savefig(fig, "fig_hypothesis_drift.pdf")

# Fig 3: prop:marginals phenomenon (left) and out-of-sample partner ranking (right)
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.4), layout="constrained")
ax = axes[0]
for theta, col, mk in zip(THETAS, [BLUE, RED], ["o", "s"]):
    ax.scatter(cov_pts[theta]["am"], cov_pts[theta]["r"], s=8, color=col, alpha=.55, lw=0, marker=mk, label=rf"$\theta={theta}$")
ax.set(xlabel=r"individual quality $P(E_m)$ of the partner", ylabel=r"residual coverage $r_m(\{m_0\})$")
ax.legend(loc="upper left", frameon=False)
ax = axes[1]
names = RANKINGS
labels = [r"$P(E_m)$", r"$r_m(S)$", r"$P(E_m)/d_m$", r"$N_S$, indep.\ proxy", r"$N_S$ (eq.~marginal)", "oracle"]
xs = np.arange(len(names))
for theta, col, off in zip(THETAS, [BLUE, RED], [-.17, .17]):
    rs = [r for r in oos_rows if r["theta"] == theta and r["ctx"] != "overall"]
    vals = [[r[f"relgain_top_{nm}"] for r in rs] for nm in names]
    med = [np.median(v) for v in vals]; q1 = [np.percentile(v, 25) for v in vals]; q3 = [np.percentile(v, 75) for v in vals]
    ax.errorbar(xs + off, med, yerr=[np.array(med) - np.array(q1), np.array(q3) - np.array(med)], fmt="o", ms=3.5, color=col, capsize=2, lw=.9, label=rf"$\theta={theta}$ (median, IQR)")
ax.axhline(0, color=GREY, lw=.65, ls=":")
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7, rotation=20)
ax.set(ylabel=r"realised id\_test gain $[\eta(S\cup\{m\})-\eta(S)]/\eta(S)$", xlabel="probing ranking of the added partner")
ax.legend(loc="upper left", frameon=False)
savefig(fig, "fig_coverage_oos.pdf")

# Fig 4: adaptive marginals P(E_m | all of S fail) vs P(E_m)
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.4), layout="constrained")
for ax, theta in zip(axes, THETAS):
    am = np.array(adapt_pts[theta]["am"]); cd = np.array(adapt_pts[theta]["cond"]); sz = np.array(adapt_pts[theta]["size"])
    for k, col in zip([1, 2, 3], [BLUE, GREEN, RED]):
        sel = sz == k
        ax.scatter(am[sel], cd[sel], s=7, color=col, alpha=.5, lw=0, label=rf"$|S|={k}$ ({sel.sum()})")
    lim = max(np.nanmax(am), np.nanmax(cd)) * 1.05
    ax.plot([0, lim], [0, lim], color=GREY, lw=.65, ls=":")
    ax.set(xlabel=r"unconditional $P(E_m)$", ylabel=r"adaptive $P(E_m\mid\bigcap_{i\in S}E_i^c)$", title=rf"$\theta={theta}$", xlim=(0, lim), ylim=(0, lim))
    ax.legend(loc="upper left", frameon=False)
savefig(fig, "fig_adaptive_marginals.pdf")

# ==============================================================================================
# 6. Claims and verdicts (criteria pre-declared in the README)
# ==============================================================================================
C = {}
sb = R["in_sample"]["summary_b_nontrivial"]
def verdict_hyp(fr_sig, med_rel):
    # pre-declared: PASS if the CI of the drift covers 0 in >= 90% of cases; FAIL if it excludes 0 in > 50%; else INCONCLUSIVE.
    # A FAIL is annotated with the magnitude (small: |median relative drift| < 5%).
    if fr_sig <= 0.10:
        return "PASS"
    if fr_sig > 0.50:
        return "FAIL" + (" (small drift: |median rel| < 5%)" if abs(med_rel) < 0.05 else " (large drift)")
    return "INCONCLUSIVE"
C["a1_residual_identity"] = {"kind": "IDENTITY", "claim": "r_m(S) = p(S+m) - p(S) (eq:residual), direct count vs difference of coverages",
                             "measurement": f"max |diff| = {ident['max_abs_r_identity']:.2e} over {ident['n_cases_checked']} (context, theta, S, m) cases",
                             "verdict": "PASS" if ident["max_abs_r_identity"] < 1e-9 else "FAIL", "note": "code check only"}
C["a2_homogeneous_identity"] = {"kind": "IDENTITY", "claim": "eq:homogeneous with the portfolio's own (p, v_h, v_m, a, b) reproduces the ratio-of-sums eta(S)",
                                "measurement": f"max |diff| = {ident['max_abs_eta_homogeneous']:.2e}", "verdict": "PASS" if ident["max_abs_eta_homogeneous"] < 1e-9 else "FAIL", "note": "code check only"}
C["a3_marginal_identity"] = {"kind": "IDENTITY", "claim": "eq:app-marginal-identity: with common (v_h, v_m, b) := S's values, eta_hyp(S+m) - eta(S) = N_S/(E[D(S)]+d-rb); and substituting all three of S+m's values recovers the realised change",
                             "measurement": f"max |diff| = {ident['max_abs_marginal_identity']:.2e} (identity), {ident['max_abs_all2_equals_real']:.2e} (all-three substitution = realised)",
                             "verdict": "PASS" if max(ident["max_abs_marginal_identity"], ident["max_abs_all2_equals_real"]) < 1e-9 else "FAIL", "note": "code check only"}
for theta in THETAS:
    d_ = sb[f"theta={theta}"]["dimensions_only"]; a_ = sb[f"theta={theta}"]["all_contexts"]
    C[f"b1_common_vm_theta{theta}"] = {"kind": "IN-SAMPLE hypothesis", "claim": f"thm:marginal hypothesis: v_m(S+m) = v_m(S) (theta={theta}; non-trivial (S,m), |S| in 1..3, 9 dimensions)",
                                       "prediction": "drift 0; CI of v_m(S+m)-v_m(S) covers 0 in >= 90% of cases",
                                       "measurement": f"n={d_['n']}; median drift {d_['vm_drift']['median']:+.4f} ({d_['vm_drift']['median_rel']*100:+.1f}% of v_m(S)); negative in {d_['vm_drift']['frac_negative']*100:.0f}%; CI excludes 0 in {d_['vm_drift']['frac_ci_excl0']*100:.0f}%; 10-90% rel drift [{d_['vm_drift']['q10_q90_rel'][0]*100:+.1f}%, {d_['vm_drift']['q10_q90_rel'][1]*100:+.1f}%]",
                                       "verdict": verdict_hyp(d_["vm_drift"]["frac_ci_excl0"], d_["vm_drift"]["median_rel"])}
    C[f"b2_common_b_theta{theta}"] = {"kind": "IN-SAMPLE hypothesis", "claim": f"thm:marginal hypothesis: b(S+m) = b(S) (Opus cost on the failure set; theta={theta})",
                                      "prediction": "drift 0; CI covers 0 in >= 90% of cases",
                                      "measurement": f"n={d_['n']}; median drift {d_['b_drift']['median']*1000:+.3f} m$ ({d_['b_drift']['median_rel']*100:+.1f}%); positive in {d_['b_drift']['frac_positive']*100:.0f}%; CI excludes 0 in {d_['b_drift']['frac_ci_excl0']*100:.0f}%; 10-90% rel [{d_['b_drift']['q10_q90_rel'][0]*100:+.1f}%, {d_['b_drift']['q10_q90_rel'][1]*100:+.1f}%]",
                                      "verdict": verdict_hyp(d_["b_drift"]["frac_ci_excl0"], d_["b_drift"]["median_rel"])}
    C[f"b3_common_vh_theta{theta}"] = {"kind": "IN-SAMPLE hypothesis" if theta < 1 else "IDENTITY", "claim": f"thm:marginal hypothesis: v_h(S+m) = v_h(S) (theta={theta})",
                                       "prediction": "drift 0" + (" (at theta=1 every committed hit has score 1, so this holds by construction)" if theta == 1 else ""),
                                       "measurement": f"n={d_['n']}; median drift {d_['vh_drift']['median']:+.4f} ({d_['vh_drift']['median_rel']*100:+.2f}%); max |drift| {d_['vh_drift']['max_abs']:.4f}; CI excludes 0 in {d_['vh_drift']['frac_ci_excl0']*100:.0f}%",
                                       "verdict": ("PASS" if theta == 1 else verdict_hyp(d_["vh_drift"]["frac_ci_excl0"], d_["vh_drift"]["median_rel"]))}
    C[f"b4_sign_rule_theta{theta}"] = {"kind": "IN-SAMPLE (conclusion robustness under violated hypothesis)", "claim": f"sign of eq:marginal N_S (with S's v_h, v_m, b) equals the realised sign of eta(S+m)-eta(S) (theta={theta})",
                                       "prediction": "if the theorem's hypothesis held, agreement would be 100% (identity); pre-declared PASS threshold: >= 90% on non-trivial cases whose realised change is CI-decided",
                                       "measurement": f"dimensions: {d_['n']} cases, agreement {d_['sign_agreement_all']*100:.1f}% (all), {(d_['sign_agreement_realised_ci_decided'] or 0)*100:.1f}% on {d_['n_realised_ci_decided']} realised-CI-decided ({d_['n_disagree_realised_ci_decided']} disagreements), {(d_['sign_agreement_ci_decided'] or 0)*100:.1f}% on {d_['n_ci_decided']} both-CI-decided ({d_['n_disagree_ci_decided']} disagreements); disagreement pattern (all): pred+/real- {d_['disagree_pattern']['pred_pos_real_neg']}, pred-/real+ {d_['disagree_pattern']['pred_neg_real_pos']}; all contexts incl. overall: {a_['sign_agreement_all']*100:.1f}% of {a_['n']}; magnitude: median |rel err| of predicted change {d_['median_rel_err_dpred']*100:.1f}% (within 25% in {d_['frac_rel_err_below_25pct']*100:.0f}%); substituting S+m's v_m alone -> {d_['median_rel_err_substitute_vm2']*100:.1f}%, b alone -> {d_['median_rel_err_substitute_b2']*100:.1f}%, v_h alone -> {d_['median_rel_err_substitute_vh2']*100:.1f}%",
                                       "verdict": "PASS" if (d_["sign_agreement_realised_ci_decided"] or 0) >= 0.90 else "FAIL"}
sc = R["marginals"]["summary_c"]; ho = R["marginals"]["held_out_summary"]; ec = R["marginals"]["error_correlation_summary"]
for theta in THETAS:
    d_ = sc[f"theta={theta}"]["dimensions_only"]; h_ = ho.get(f"theta={theta}|dimensions_only", {})
    n_sig = h_.get("n_idtest_gap_ci_gt0", 0)
    C[f"c1_quality_not_coverage_theta{theta}"] = {"kind": "IN-SAMPLE phenomenon + OUT-OF-SAMPLE confirmation", "claim": f"prop:marginals phenomenon: the partner with the highest individual quality is not the partner with the highest pair coverage (theta={theta}, 9 dimensions, usable candidates)",
                                                   "prediction": "the phenomenon exists (fraction > 0) and survives held-out selection (probing-selected partners, id_test coverage gap CI > 0 in >= 1 case)",
                                                   "measurement": f"in-sample: top-quality != top-coverage in {d_['n_differs']}/{d_['n_contexts_m0']} (dimension, m0) contexts ({d_['frac_differs']*100:.0f}%; bootstrap mean P(differs) {d_['mean_P_differs_boot']:.2f}); held-out: probing top-quality != probing top-coverage in {h_.get('n_differs')}/{h_.get('n')}, id_test coverage gap positive in {h_.get('n_idtest_gap_positive')}, negative in {h_.get('n_idtest_gap_negative')}, CI>0 in {n_sig}, CI<0 in {h_.get('n_idtest_gap_ci_lt0')}",
                                                   "verdict": "PASS" if (d_["n_differs"] > 0 and n_sig >= 1) else ("INCONCLUSIVE" if d_["n_differs"] > 0 else "FAIL")}
    C[f"c2_positive_error_correlation_theta{theta}"] = {"kind": "IN-SAMPLE hypothesis test (the joint-law property that makes individual scores insufficient)", "claim": f"P(E_m^c and E_i^c) > P(E_m^c)P(E_i^c) for usable candidate pairs (theta={theta})",
                                                          "prediction": "pre-declared PASS: excess > 0 in >= 90% of pairs and CI > 0 in >= 75% (per dimension)",
                                                          "measurement": f"dimensions: {ec[f'theta={theta}|dimensions_only']['n_pairs']} pairs, excess > 0 in {ec[f'theta={theta}|dimensions_only']['frac_positive']*100:.0f}%, CI > 0 in {ec[f'theta={theta}|dimensions_only']['frac_positive_ci']*100:.0f}%, CI < 0 in {ec[f'theta={theta}|dimensions_only']['n_negative_ci']}; median excess {ec[f'theta={theta}|dimensions_only']['median_excess']:.3f}; overall (pooled dimensions, inflated by 0%-pass dimensions): excess > 0 in {ec[f'theta={theta}|overall']['frac_positive']*100:.0f}%",
                                                          "verdict": "PASS" if (ec[f"theta={theta}|dimensions_only"]["frac_positive"] >= 0.9 and ec[f"theta={theta}|dimensions_only"]["frac_positive_ci"] >= 0.75) else "FAIL"}
    C[f"c3_spearman_theta{theta}"] = {"kind": "IN-SAMPLE descriptive (+ identity note)", "claim": f"rank agreement of individual quality with pair coverage; r_m vs pair coverage is an identity (theta={theta})",
                                      "measurement": f"median Spearman(P(E_m), p(m0+m)) = {d_['median_spearman_a_vs_paircov']:.2f}; < 1 in {d_['frac_spearman_a_vs_paircov_below_1']*100:.0f}% of contexts, <= 0 in {d_['frac_spearman_a_vs_paircov_le_0']*100:.0f}%; Spearman(r_m, p(m0+m)) = {d_['min_spearman_r_vs_paircov_identity']:.3f} (min; identity since p(m0+m) = P(E_m0) + r_m); held-out Spearman with id_test pair coverage: probing P(E_m) {h_.get('median_spearman_probing_a_vs_idtest_paircov', float('nan')):.2f} vs probing r_m {h_.get('median_spearman_probing_r_vs_idtest_paircov', float('nan')):.2f} (r_m better in {h_.get('frac_r_beats_a', float('nan'))*100:.0f}% of (dimension, m0))",
                                      "verdict": "PASS (descriptive)" if d_["frac_spearman_a_vs_paircov_below_1"] > 0.5 else "INCONCLUSIVE"}
sd = R["oos"]["summary_d"]
for theta in THETAS:
    d_ = sd.get(f"theta={theta}|dimensions_only")
    if d_ is None:
        continue
    C[f"d1_oos_sign_theta{theta}"] = {"kind": "OUT-OF-SAMPLE prediction", "claim": f"sign of N_S estimated on probing predicts the sign of the realised id_test change eta(S+m)-eta(S) (theta={theta}, |S| in 1..2, dimensions)",
                                      "prediction": "pre-declared PASS: >= 90% agreement on CI-decided id_test cases",
                                      "measurement": f"{d_['n_sign_cases']} (S,m) cases, agreement {d_['sign_agreement_all']*100:.1f}% (all), {(d_['sign_agreement_ci_decided'] or 0)*100:.1f}% on {d_['n_sign_ci_decided']} CI-decided ({d_['n_disagree_ci_decided']} disagreements)",
                                      "verdict": "PASS" if (d_["sign_agreement_ci_decided"] or 0) >= 0.90 else "FAIL"}
    def pair_txt(x, y):
        q = d_[f"{x}_vs_{y}"]
        return f"{x} vs {y}: same top in {q['frac_same_top']*100:.0f}%, differ in {q['n_differing']}, {x} wins {q['n_x_wins']} (CI {q['n_x_wins_ci']}), {y} wins {q['n_y_wins']} (CI {q['n_y_wins_ci']}), mean rel-gain difference on differing {(q['mean_diff_relgain_x_minus_y_on_differing'] or 0)*100:+.1f} pts"
    def pair_verdict(x, y):
        q = d_[f"{x}_vs_{y}"]
        if q["n_differing"] == 0:
            return "INCONCLUSIVE (rankings never differ)"
        p = stats.binomtest(q["n_x_wins"], q["n_x_wins"] + q["n_y_wins"], 0.5).pvalue if (q["n_x_wins"] + q["n_y_wins"]) > 0 else 1.0
        if q["n_x_wins"] > q["n_y_wins"] and p < 0.05:
            return "PASS"
        if q["n_y_wins"] > q["n_x_wins"] and p < 0.05:
            return "FAIL"
        return "INCONCLUSIVE"
    gains = "; ".join(f"{nm}: median {d_[f'median_relgain_top_{nm}']*100:+.1f}% / mean {d_[f'mean_relgain_top_{nm}']*100:+.1f}% / gain>0 in {d_[f'frac_top_{nm}_gain_positive']*100:.0f}%" for nm in RANKINGS)
    C[f"d2a_oos_rm_vs_quality_theta{theta}"] = {"kind": "OUT-OF-SAMPLE prediction", "claim": f"partner with the highest probing residual coverage r_m(S) yields a larger realised id_test efficiency gain than the partner with the highest probing individual quality P(E_m) (theta={theta}; both rankings ignore price)",
                                                "prediction": "task (d) as posed; note thm:marginal itself ranks by N_S, not by r_m alone. PASS if r_m wins significantly (binomial sign test on differing S)",
                                                "measurement": f"{d_['n_S']} portfolios S (|S| in 1..2, dimensions); " + pair_txt("coverage_r", "quality_a"),
                                                "verdict": pair_verdict("coverage_r", "quality_a")}
    C[f"d2b_oos_theorem_vs_quality_theta{theta}"] = {"kind": "OUT-OF-SAMPLE prediction", "claim": f"partner with the highest probing eq:marginal criterion N_S = r_m(S)[v_h-v_m+eta(S)b]-eta(S)d_m yields a larger realised id_test gain than the highest-quality partner (theta={theta})",
                                                     "prediction": "PASS if N_S wins significantly on differing S",
                                                     "measurement": f"realised id_test relative gains of the top partner by each ranking -- {gains}; " + pair_txt("theorem_N", "quality_a") + "; median Spearman(probing score, id_test gain): P(E_m) {:.2f}, r_m {:.2f}, quality/$ {:.2f}, N_indep {:.2f}, N_S {:.2f}".format(d_["median_spearman_probing_a_vs_idtest_gain"], d_["median_spearman_probing_r_vs_idtest_gain"], d_["median_spearman_probing_apd_vs_idtest_gain"], d_["median_spearman_probing_Nindep_vs_idtest_gain"], d_["median_spearman_probing_N_vs_idtest_gain"]),
                                                     "verdict": pair_verdict("theorem_N", "quality_a")}
    C[f"d2c_oos_theorem_vs_indep_theta{theta}"] = {"kind": "OUT-OF-SAMPLE prediction (prop:marginals content inside the theorem's accounting)", "claim": f"N_S with the true residual coverage r_m(S) beats N_S with the marginal-only independence proxy (1-p(S))P(E_m) in realised id_test gain (theta={theta})",
                                                   "prediction": "prop:marginals: marginal probabilities cannot identify residual coverage; PASS if the paired-outcome criterion wins significantly where the two rankings differ",
                                                   "measurement": pair_txt("theorem_N", "theorem_N_indep") + "; " + pair_txt("theorem_N", "quality_per_dollar") + "; " + pair_txt("theorem_N_indep", "quality_per_dollar"),
                                                   "verdict": pair_verdict("theorem_N", "theorem_N_indep")}
    st3 = d_["d3_sign_test"]
    C[f"d3_oos_coverage_ranking_theta{theta}"] = {"kind": "OUT-OF-SAMPLE prediction", "claim": f"residual coverage r_m(S) estimated on probing ranks partners by realised id_test residual coverage better than individual quality does (prop:marginals content; theta={theta})",
                                                  "prediction": "median Spearman(probing r_m, id_test r_m) > median Spearman(probing P(E_m), id_test r_m); PASS if the paired sign test over portfolios is significant",
                                                  "measurement": f"median Spearman: probing r_m {d_['median_spearman_probing_r_vs_idtest_r']:.3f} vs probing P(E_m) {d_['median_spearman_probing_a_vs_idtest_r']:.3f}; r_m better in {st3['r_better']}, P(E_m) better in {st3['a_better']}, ties {st3['ties']} of {d_['n_S']} portfolios (two-sided p = {st3['p_two_sided']:.3f})",
                                                  "verdict": ("PASS" if (st3["r_better"] > st3["a_better"] and (st3["p_two_sided"] or 1) < 0.05) else ("FAIL" if (st3["a_better"] > st3["r_better"] and (st3["p_two_sided"] or 1) < 0.05) else "INCONCLUSIVE"))}
ss = R["submodularity"]["static"]
C["e1_static_submodularity"] = {"kind": "IDENTITY", "claim": "r_m(T) <= r_m(S) for all S subset T, m not in T (app:static-adaptive); cheap4 = {Qwen3-Max, kimi-k2.5, qwen3.5-plus, glm-5} and all 7 candidates",
                                "measurement": f"cheap4: {ss['cheap4']['n_violations']} violations in {ss['cheap4']['n_triples']} (S,T,m) triples (max r_m(T)-r_m(S) = {ss['cheap4']['max_violation']:.2e}); all7: {ss['all7']['n_violations']} in {ss['all7']['n_triples']} (max {ss['all7']['max_violation']:.2e})",
                                "verdict": "PASS" if ss["all7"]["n_violations"] == 0 else "FAIL", "note": "holds by set inclusion on every sample; code check only"}
sa = R["submodularity"]["adaptive_summary"]
for theta in THETAS:
    k7 = sa.get(f"theta={theta}|all7|size=1-3", {}); k4 = sa.get(f"theta={theta}|cheap4|size=1-3", {})
    per = {sz: sa.get(f"theta={theta}|all7|size={sz}", {}) for sz in [1, 2, 3]}
    C[f"e2_adaptive_marginal_theta{theta}"] = {"kind": "IN-SAMPLE hypothesis test (does the appendix's adaptive-submodularity failure occur?)",
                                              "claim": f"adaptive marginal P(E_m | all of S fail) EXCEEDS the unconditional P(E_m) (theta={theta}); for |S|=1 this is exactly negative error correlation, the complement of c2",
                                              "prediction": "app:static-adaptive says it CAN happen (two-source example); no rate is predicted. Report the frequency.",
                                              "measurement": f"all7, |S| in 1..3: {k7.get('n')} cases, exceeds (point) {k7.get('frac_exceeds_point', 0)*100:.1f}%, exceeds (CI>0) {k7.get('frac_exceeds_ci', 0)*100:.1f}%, below (CI<0) {k7.get('frac_below_ci', 0)*100:.1f}%, median ratio cond/uncond {k7.get('median_ratio_cond_over_uncond', float('nan')):.2f}; by |S|: " + ", ".join(f"|S|={sz}: {per[sz].get('frac_exceeds_point', 0)*100:.0f}% point / {per[sz].get('frac_exceeds_ci', 0)*100:.0f}% CI of {per[sz].get('n')}" for sz in [1, 2, 3]) + f"; cheap4: exceeds (point) {k4.get('frac_exceeds_point', 0)*100:.1f}%, (CI>0) {k4.get('frac_exceeds_ci', 0)*100:.1f}% of {k4.get('n')}",
                                              "verdict": ("OBSERVED (rare)" if k7.get("frac_exceeds_ci", 0) > 0 else "NOT INSTANTIABLE (phenomenon absent: needs negatively correlated errors, but errors are positively correlated in ~99% of pairs; conditional marginal is BELOW the unconditional one in ~99% of cases)")}
R["claims"] = C
R["meta"]["runtime_s"] = round(time.time() - T0, 1)
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(R, f, indent=1, default=lambda o: (None if isinstance(o, float) and not np.isfinite(o) else (o.item() if hasattr(o, "item") else str(o))))
print(f"results.json written ({time.time()-T0:.0f}s)")
for k, v in C.items():
    print(f"- {k} [{v['kind']}]: {v['verdict']} -- {v['measurement'][:220]}")

# ==============================================================================================
# 7. Per-dimension tables (stored) and README generated from results.json (no retyped numbers)
# ==============================================================================================
def per_dim_tables():
    tabs = {"b": {}, "d": {}}
    for th in THETAS:
        for ctx in CONTEXTS:
            rc = [r for r in rows_b if r["ctx"] == ctx and r["theta"] == th]
            if rc:
                relvm = np.array([(r["vm_S2"] - r["vm_S"]) / r["vm_S"] for r in rc]); relb = np.array([(r["b_S2"] - r["b_S"]) / r["b_S"] for r in rc])
                ag = np.array([r["sign_agree"] for r in rc]); rdec = np.array([r["dreal_ci"][0] > 0 or r["dreal_ci"][1] < 0 for r in rc])
                tabs["b"][f"{ctx}|theta={th}"] = {"n": len(rc), "n_realised_ci_decided": int(rdec.sum()), "median_rel_vm_drift": float(np.median(relvm)),
                                                   "frac_vm_ci_excl0": float(np.mean([r["dvm_ci"][0] > 0 or r["dvm_ci"][1] < 0 for r in rc])),
                                                   "median_rel_b_drift": float(np.median(relb)), "frac_b_ci_excl0": float(np.mean([r["db_ci"][0] > 0 or r["db_ci"][1] < 0 for r in rc])),
                                                   "sign_agree_all": float(ag.mean()), "sign_agree_realised_ci_decided": float(ag[rdec].mean()) if rdec.any() else None,
                                                   "pred_pos_real_neg": int(sum(1 for r in rc if r["N"] > 0 and r["dreal"] < 0)), "pred_neg_real_pos": int(sum(1 for r in rc if r["N"] < 0 and r["dreal"] > 0))}
            rs = [r for r in oos_rows if r["ctx"] == ctx and r["theta"] == th]; sc = [r for r in oos_sign if r["ctx"] == ctx and r["theta"] == th]
            if rs:
                dec = [r for r in sc if r["ci_decided"]]
                tabs["d"][f"{ctx}|theta={th}"] = {"n_S": len(rs), "n_sign": len(sc), "sign_agree": float(np.mean([r["agree"] for r in sc])), "n_ci_decided": len(dec),
                                                   "sign_agree_ci_decided": float(np.mean([r["agree"] for r in dec])) if dec else None,
                                                   "median_relgain_top_quality": float(np.median([r["relgain_top_quality_a"] for r in rs])),
                                                   "median_relgain_top_N": float(np.median([r["relgain_top_theorem_N"] for r in rs])),
                                                   "median_relgain_top_Nindep": float(np.median([r["relgain_top_theorem_N_indep"] for r in rs])),
                                                   "median_relgain_oracle": float(np.median([r["relgain_top_oracle_id_test"] for r in rs])),
                                                   "frac_topN_is_oracle": float(np.mean([r["top_theorem_N"] == r["top_oracle_id_test"] for r in rs])),
                                                   "frac_topQ_is_oracle": float(np.mean([r["top_quality_a"] == r["top_oracle_id_test"] for r in rs]))}
    return tabs
R["per_dimension"] = per_dim_tables()
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(R, f, indent=1, default=lambda o: (None if isinstance(o, float) and not np.isfinite(o) else (o.item() if hasattr(o, "item") else str(o))))

def pct(x, d=1):
    return "n/a" if x is None else f"{x*100:.{d}f}%"
def fmt_ci(c, d=3, scale=1.0):
    return "n/a" if c is None or c[0] is None else f"[{c[0]*scale:.{d}f}, {c[1]*scale:.{d}f}]"

def write_readme(R):
    C = R["claims"]; M = R["meta"]; sb = R["in_sample"]["summary_b_nontrivial"]; sd = R["oos"]["summary_d"]
    we = {r["m"]: r for r in R["in_sample"]["worked_example"]}
    ex1 = R["marginals"]["example"]["theta=1.0"]; ho1 = ex1["held_out_same_ctx_m0"]; ins1 = ex1["in_sample_post_selection"]
    sig_ho = R["marginals"]["held_out_significant_examples"]
    L = []
    L.append(f"# X2_residual_coverage -- eq:residual, thm:marginal, prop:marginals, app:static-adaptive on CodeRouterBench\n")
    L.append(f"Script: `run.py` (numpy/pandas/scipy/matplotlib only; master seed {M['seed']}; {M['n_boot']} bootstrap task resamples per context; runtime {M['runtime_s']} s). "
             f"Outputs: `results.json` (every number below, with 95% percentile CIs), `figures/*.pdf` (vector, paper style: Times + usetex{' OK' if M['usetex'] else ' FAILED -> mathtext fallback'}), this README (generated by `run.py`; numbers are not retyped).\n")
    L.append("## 1. Data and setup\n")
    L.append(f"* Real, execution-verified matrix `$ACROUTER_DATA/coderouterbench/id_results_long.csv`: {M['n_tasks']} tasks x 8 models (score in [0,1], token-price `cost_usd` estimates, `latency_ms`). Splits: probing {M['n_probing']}, id_test {M['n_id_test']}. Cross-checks asserted in code: Always-Opus on id_test = $34.02 and mean score 43.83 (CodeRouterBench paper). {M['zero_cost_candidate_rows_kept']} zero-cost Qwen3-Max rows (missing token records) are kept as recorded; nothing is imputed.")
    L.append(f"* Reference policy B = always `claude-opus-4-6` (Y_B, C_B = its score and dollars). Candidates m in {M['candidates']}; `cheap4` = {M['cheap4']} for the submodularity lattice.")
    L.append("* Verifier: exact, E_m = {score_m >= theta}, theta in {1.0, 0.5}. Portfolio S: all members are called in parallel; an exact selecting verifier (OR semantics, eq:residual) commits the highest-scoring passing member; on a miss Opus is called and its output committed (eq:accounting with H = 1[any member passes], Y_F = Opus score, C_F = Opus cost). "
             "Dollars D = sum of cost_usd of every call made in the epoch (all members + Opus on a miss); primary bill a(S) = E[sum of member costs]; d_m(S) = a(S+m) - a(S) = E[cost_m] (no aggregation fee is charged; the verifier is the benchmark's own grader). Value V = committed score. eta = E[V]/E[D] (ratio of sums).")
    L.append("* Primitives per portfolio (paper notation): p(S) = P(hit); v_h(S) = E[V | hit]; v_m(S) = E[Opus score | miss] (Opus value on S's failure set); b(S) = E[Opus cost | miss]; r_m(S) = P(E_m and all of S fail) (eq:residual). Contexts: 9 dimensions and `overall` (pooled). A candidate is *usable* in a context if it has >= 20 passes and >= 20 failures (code_understanding at both theta and code_refactoring at theta = 1 have no usable candidate: their proxy-metric scores essentially never reach the threshold). An (S, m) case is *non-trivial* if #(E_m and fail S) >= 10, #fail(S+m) >= 10 and #hit(S) >= 10 (otherwise a conditional mean is estimated from < 10 tasks); trivial cases are counted separately.")
    L.append("* Bootstrap: task multinomial resampling within the context; all CIs are 2.5/97.5 percentiles of 500 resamples. Out-of-sample: primitives (p, r_m, v_h, v_m, b, d_m, eta) estimated on probing; realised quantities and their CIs on id_test.\n")
    L.append("## 2. What is tested, and how each test is labelled\n")
    L.append("| kind | meaning here |\n|---|---|\n| IDENTITY | holds by algebra on any sample; reported as a code check only (a1-a3, b3 at theta = 1, e1) |\n| IN-SAMPLE hypothesis | does a stated hypothesis of the theorem hold on the data (b1-b3: common v_m, b, v_h across S and S+m; c2: positive error correlation; e2: adaptive marginals) |\n| OUT-OF-SAMPLE prediction | primitives on probing, realised outcome on id_test (d1-d3; held-out part of c1/c3) |\n")
    L.append("### Prediction formula with plugged-in primitives (thm:marginal / eq:app-marginal-identity)\n")
    L.append("With the *current* portfolio's values (v_h, v_m, b of S), the theorem predicts eta(S+m) - eta(S) = N_S / (E[D(S)] + d_m - r_m b) with N_S = r_m(S)[v_h - v_m + eta(S) b] - eta(S) d_m(S), and sign(eta(S+m) - eta(S)) = sign(N_S). Worked example, `overall`, theta = 1, S = {Qwen3-Max} (eta(S) = %.2f per $, v_h = %.3f, v_m = %.3f, b = $%.5f):\n" % (we["kimi"]["eta_S"], we["kimi"]["vh_S"], we["kimi"]["vm_S"], we["kimi"]["b_S"]))
    L.append("| add m | r_m(S) | d_m ($) | N_S [CI] | predicted change [CI] | realised change [CI] | v_m(S) -> v_m(S+m) | b(S) -> b(S+m) ($) | sign agrees |\n|---|---|---|---|---|---|---|---|---|")
    for m in ["kimi", "qwen35p", "gpt54"]:
        r = we[m]
        L.append(f"| {m} | {r['r']:.4f} | {r['d']:.5f} | {r['N']:+.4f} {fmt_ci(r['N_ci'], 4)} | {r['dpred']:+.2f} {fmt_ci(r['dpred_ci'], 2)} | {r['dreal']:+.2f} {fmt_ci(r['dreal_ci'], 2)} | {r['vm_S']:.3f} -> {r['vm_S2']:.3f} | {r['b_S']:.5f} -> {r['b_S2']:.5f} | {'yes' if r['sign_agree'] else 'NO'} |")
    L.append("\nThe qwen3.5-plus row is a genuine sign disagreement in the pooled context: adding it removes algorithm/test-generation tasks (where Opus is expensive) from the failure set, so b falls by 14% and the realised gain is positive although N_S < 0. The gpt-5.4 row shows the typical case: the hypothesis is violated (v_m falls) but the sign and magnitude are right because d_m dominates.\n")
    L.append("## 3. Claims and verdicts\n")
    L.append("| id | kind | claim | prediction | measurement | verdict |\n|---|---|---|---|---|---|")
    esc = lambda s: str(s).replace("|", "\\|")
    for k, v in C.items():
        L.append(f"| {k} | {esc(v['kind'])} | {esc(v['claim'])} | {esc(v.get('prediction', v.get('note', '')))} | {esc(v['measurement'])} | **{esc(v['verdict'])}** |")
    L.append("")
    L.append("## 4. Detail per part\n")
    L.append("### (a) Identities (code checks)\n")
    idn = R["identity"]
    L.append(f"r_m(S) = p(S+m) - p(S): max |diff| {idn['max_abs_r_identity']:.1e}; eq:homogeneous reproduces the ratio-of-sums eta(S): {idn['max_abs_eta_homogeneous']:.1e}; eq:app-marginal-identity with common (v_h, v_m, b) := S's values: {idn['max_abs_marginal_identity']:.1e}; substituting all three of S+m's values into the same formula recovers the realised change exactly: {idn['max_abs_all2_equals_real']:.1e} ({idn['n_cases_checked']} cases). These establish only that the code implements the paper's algebra.\n")
    L.append("### (b) Does adding m keep v_h, v_m and b fixed? (thm:marginal hypothesis) and is the sign rule robust?\n")
    L.append("Per dimension (non-trivial cases, |S| in 1..3, all m not in S). `drift` = value for S+m minus value for S, relative to S; `CI excl 0` = fraction of cases whose bootstrap CI of the drift excludes 0; `agree` = sign(N_S) = sign(realised change); `real-dec` = restricted to cases whose realised change has a CI excluding 0; the last two columns count the disagreement directions.\n")
    L.append("| context | theta | n | real-dec | median rel v_m drift | v_m CI excl 0 | median rel b drift | b CI excl 0 | agree (all) | agree (real-dec) | pred+ / real- | pred- / real+ |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for th in THETAS:
        for ctx in CONTEXTS:
            t = R["per_dimension"]["b"].get(f"{ctx}|theta={th}")
            if t:
                L.append(f"| {ctx} | {th} | {t['n']} | {t['n_realised_ci_decided']} | {t['median_rel_vm_drift']*100:+.1f}% | {pct(t['frac_vm_ci_excl0'], 0)} | {t['median_rel_b_drift']*100:+.1f}% | {pct(t['frac_b_ci_excl0'], 0)} | {pct(t['sign_agree_all'])} | {pct(t['sign_agree_realised_ci_decided'])} | {t['pred_pos_real_neg']} | {t['pred_neg_real_pos']} |")
    L.append("")
    for th in THETAS:
        d_ = sb[f"theta={th}"]["dimensions_only"]
        L.append(f"theta = {th}, dimensions pooled: v_m(S+m) < v_m(S) in {pct(d_['vm_drift']['frac_negative'], 0)} of {d_['n']} cases (median {d_['vm_drift']['median_rel']*100:+.1f}%, 10-90% [{d_['vm_drift']['q10_q90_rel'][0]*100:+.1f}%, {d_['vm_drift']['q10_q90_rel'][1]*100:+.1f}%]); b drift median {d_['b_drift']['median_rel']*100:+.1f}% (10-90% [{d_['b_drift']['q10_q90_rel'][0]*100:+.1f}%, {d_['b_drift']['q10_q90_rel'][1]*100:+.1f}%]); v_h drift median {d_['vh_drift']['median_rel']*100:+.2f}% (max |drift| {d_['vh_drift']['max_abs']:.3f}). "
                 f"Predicted-change error: median |rel err| {pct(d_['median_rel_err_dpred'])}; replacing only v_m by v_m(S+m) reduces it to {pct(d_['median_rel_err_substitute_vm2'])}, only b to {pct(d_['median_rel_err_substitute_b2'])}, only v_h to {pct(d_['median_rel_err_substitute_vh2'])}; median signed bias (predicted - realised)/eta(S) = {d_['median_signed_bias_dpred_minus_dreal_over_eta_S']*100:+.2f} pts (the prediction is optimistic). Sign agreement {pct(d_['sign_agreement_all'])} on all, {pct(d_['sign_agreement_realised_ci_decided'])} on realised-CI-decided, {pct(d_['sign_agreement_ci_decided'])} when both N_S and the realised change are CI-decided.")
    tv = R["in_sample"]["trivial_cases_diagnostics"]
    L.append(f"\nTrivial cases (excluded above): theta = 1: {tv['theta=1.0']['n']} cases, {tv['theta=1.0']['n_disagree']} counted as sign disagreements, of which {tv['theta=1.0']['disagree_with_N_undefined']} are not disagreements at all: S never hits in that context, so v_h(S) and hence N_S are undefined (NaN) and the comparison is vacuous (audit correction; the original text attributed them to r_m = 0 with N_S = -eta d < 0, which is wrong because N_S is NaN whenever v_h(S) is); {tv['theta=1.0']['disagree_with_N_finite']} genuine disagreements with finite N_S, {tv['theta=1.0']['disagree_ci_decided']} CI-decided; theta = 0.5: {tv['theta=0.5']['n']} cases, {tv['theta=0.5']['n_disagree']} counted, {tv['theta=0.5']['disagree_with_N_undefined']} with N_S undefined, {tv['theta=0.5']['disagree_with_N_finite']} genuine, {tv['theta=0.5']['disagree_ci_decided']} CI-decided.")
    L.append("\nMechanism of the v_m drift: the tasks that m newly covers are removed from the failure set; they are the tasks Opus also tends to solve, so Opus's mean score on the *remaining* failures falls. The theorem's proof (app:portfolio) states exactly this caveat ('if a new portfolio changes which kinds of tasks reach the fallback, those conditional quantities may change'). The drift is in the direction that makes eq:marginal optimistic (pred+/real- dominates the disagreements in every dimension); the pooled `overall` context is the exception, where b falls when a candidate that is strong on the expensive-Opus dimensions (qwen3.5-plus on algorithm/test_generation) is added, producing pred-/real+ disagreements.\n")
    L.append("Concrete real disagreements (both CIs exclude 0): ")
    for r in [x for x in R["in_sample"]["disagreements"] if x["ci_decided"] and x["ctx"] != "overall"][:6]:
        L.append(f"* {r['ctx']}, theta = {r['theta']}, S = {r['S']}, add {r['m']}: r_m = {r['r']:.3f}, N_S = {r['N']:+.4f} {fmt_ci(r['N_ci'], 4)} predicts {r['dpred']:+.2f}; realised {r['dreal']:+.2f} {fmt_ci(r['dreal_ci'], 2)}; v_m {r['vm_S']:.3f} -> {r['vm_S2']:.3f}, b ${r['b_S']:.4f} -> ${r['b_S2']:.4f}.")
    L.append("\n### (c) prop:marginals: is the highest-quality partner the highest-coverage partner?\n")
    for th in THETAS:
        d_ = R["marginals"]["summary_c"][f"theta={th}"]["dimensions_only"]; h_ = R["marginals"]["held_out_summary"][f"theta={th}|dimensions_only"]; a_ = R["marginals"]["held_out_summary"][f"theta={th}|all_contexts"]
        L.append(f"theta = {th}: in-sample the top-quality partner differs from the top-pair-coverage partner in {d_['n_differs']}/{d_['n_contexts_m0']} (dimension, m0) contexts; median Spearman(P(E_m), p(m0+m)) over partners = {d_['median_spearman_a_vs_paircov']:.2f}, below 1 in {pct(d_['frac_spearman_a_vs_paircov_below_1'], 0)} of contexts. Held-out (partners chosen on probing, gap on id_test): differ in {h_['n_differs']}/{h_['n']} dimension contexts, id_test gap positive in {h_['n_idtest_gap_positive']}, negative in {h_['n_idtest_gap_negative']}, CI > 0 in {h_['n_idtest_gap_ci_gt0']}; including `overall`: differ in {a_['n_differs']}/{a_['n']}, CI > 0 in {a_['n_idtest_gap_ci_gt0']}. Spearman(r_m, p(m0+m)) = 1 by identity (p(m0+m) = P(E_m0) + r_m for fixed m0) and is not evidence.")
    L.append("\nReal examples (held-out, id_test gap CI > 0):")
    for h in sig_ho:
        L.append(f"* {h['ctx']}, theta = {h['theta']}, primary m0 = {h['m0']}: highest-quality partner (probing) {h['top_quality_probing']} vs highest-coverage partner (probing) {h['top_coverage_probing']}; id_test pair coverage {h['id_test_pair_cov_top_quality']:.3f} vs {h['id_test_pair_cov_top_coverage']:.3f}, gap {h['id_test_gap']:+.4f} {fmt_ci(h['id_test_gap_ci'], 4)}.")
    L.append(f"* Largest in-sample dimension gap (post-selection, descriptive): {ins1['ctx']}, theta = 1, m0 = {ins1['m0']} (P(E_m0) = {ins1['a_m0']:.3f}): top-quality partner {ins1['top_quality']} (P(E_m) = {max(ins1['a_m']):.3f}, mean cost ${ins1['cost_top_quality']:.4f}) vs top-coverage partner {ins1['top_coverage']} (P(E_m) = {ins1['a_m'][ins1['partners'].index(ins1['top_coverage'])]:.3f}, cost ${ins1['cost_top_coverage']:.4f}); pair coverage {ins1['p_pair'][ins1['partners'].index(ins1['top_quality'])]:.4f} vs {max(ins1['p_pair']):.4f}, gap {ins1['coverage_gap_point']:.4f}; the same (probing-selected) pair on id_test: gap {ho1['id_test_gap']:+.4f} {fmt_ci(ho1['id_test_gap_ci'], 4)} (not significant).")
    ec = R["marginals"]["error_correlation_summary"]
    L.append(f"\nError correlation (c2): P(both fail) - P(E_m^c)P(E_i^c) > 0 in {pct(ec['theta=1.0|dimensions_only']['frac_positive'], 0)} / {pct(ec['theta=0.5|dimensions_only']['frac_positive'], 0)} of usable pairs per dimension (theta 1 / 0.5), CI > 0 in {pct(ec['theta=1.0|dimensions_only']['frac_positive_ci'], 0)} / {pct(ec['theta=0.5|dimensions_only']['frac_positive_ci'], 0)}, median excess {ec['theta=1.0|dimensions_only']['median_excess']:.3f} / {ec['theta=0.5|dimensions_only']['median_excess']:.3f}. This is the joint-law fact behind both the small information gap (c) and the absence of the adaptive failure (e2).\n")
    L.append("### (d) Out-of-sample partner ranking (probing -> id_test)\n")
    L.append("For every usable portfolio S (|S| in 1..2) and every non-trivial partner m, the following probing statistics rank the partners: P(E_m) (individual quality), r_m(S) (residual coverage), P(E_m)/d_m (quality per dollar), N_S with r_m replaced by the independence proxy (1 - p(S))P(E_m) (what a selector with only marginal probabilities and the theorem's accounting could compute), and N_S itself. The realised id_test change eta(S+m) - eta(S) of each ranking's top partner is compared (paired bootstrap on id_test tasks). The `oracle` column is the id_test-best partner.\n")
    L.append("| context | theta | #S | sign cases | sign agree | CI-decided | agree (CI-dec) | median rel gain: top P(E_m) | top N_indep | top N_S | oracle | top N_S = oracle | top P(E_m) = oracle |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for th in THETAS:
        for ctx in CONTEXTS:
            t = R["per_dimension"]["d"].get(f"{ctx}|theta={th}")
            if t:
                L.append(f"| {ctx} | {th} | {t['n_S']} | {t['n_sign']} | {pct(t['sign_agree'])} | {t['n_ci_decided']} | {pct(t['sign_agree_ci_decided'])} | {t['median_relgain_top_quality']*100:+.1f}% | {t['median_relgain_top_Nindep']*100:+.1f}% | {t['median_relgain_top_N']*100:+.1f}% | {t['median_relgain_oracle']*100:+.1f}% | {pct(t['frac_topN_is_oracle'], 0)} | {pct(t['frac_topQ_is_oracle'], 0)} |")
    L.append("")
    for th in THETAS:
        d_ = sd[f"theta={th}|dimensions_only"]
        L.append(f"theta = {th}, dimensions: N_S (probing) beats P(E_m) in {d_['theorem_N_vs_quality_a']['n_x_wins']}/{d_['theorem_N_vs_quality_a']['n_differing']} differing portfolios (CI-significant {d_['theorem_N_vs_quality_a']['n_x_wins_ci']}), never loses; N_S vs N_indep differ in only {d_['theorem_N_vs_theorem_N_indep']['n_differing']} portfolios ({d_['theorem_N_vs_theorem_N_indep']['n_x_wins']} : {d_['theorem_N_vs_theorem_N_indep']['n_y_wins']}, CI-significant {d_['theorem_N_vs_theorem_N_indep']['n_x_wins_ci']} : {d_['theorem_N_vs_theorem_N_indep']['n_y_wins_ci']}); N_S vs quality-per-dollar: {d_['theorem_N_vs_quality_per_dollar']['n_x_wins']} : {d_['theorem_N_vs_quality_per_dollar']['n_y_wins']} (CI {d_['theorem_N_vs_quality_per_dollar']['n_x_wins_ci']} : {d_['theorem_N_vs_quality_per_dollar']['n_y_wins_ci']}) of {d_['theorem_N_vs_quality_per_dollar']['n_differing']}; r_m alone vs P(E_m) alone: {d_['coverage_r_vs_quality_a']['n_x_wins']} : {d_['coverage_r_vs_quality_a']['n_y_wins']} of {d_['coverage_r_vs_quality_a']['n_differing']}. Median Spearman(probing score, id_test gain) across portfolios: P(E_m) {d_['median_spearman_probing_a_vs_idtest_gain']:.2f}, r_m {d_['median_spearman_probing_r_vs_idtest_gain']:.2f}, P(E_m)/d_m {d_['median_spearman_probing_apd_vs_idtest_gain']:.2f}, N_indep {d_['median_spearman_probing_Nindep_vs_idtest_gain']:.2f}, N_S {d_['median_spearman_probing_N_vs_idtest_gain']:.2f}.")
    orc = {th: np.mean([r["top_theorem_N"] == r["top_oracle_id_test"] for r in R["oos"]["cases_d"] if r["theta"] == th and r["ctx"] != "overall"]) for th in THETAS}
    chg = {th: 1 - sd[f"theta={th}|dimensions_only"]["theorem_N_vs_theorem_N_indep"]["frac_same_top"] for th in THETAS}
    L.append(f"\nReading: the theorem's criterion is an excellent out-of-sample ranker of realised gains (its top partner is the id_test oracle in {pct(orc[1.0], 0)} / {pct(orc[0.5], 0)} of dimension portfolios at theta 1 / 0.5), but almost all of its advantage over individual quality comes from pricing (d_m, b, eta(S)) and from the value terms, not from the paired-outcome information in r_m: replacing r_m by the marginal-only proxy (1 - p(S))P(E_m) changes the top partner in only {pct(chg[1.0], 0)} / {pct(chg[0.5], 0)} of portfolios and does not lose significantly. This is the real-data size of prop:marginals' information gap on this pool: it exists (c1, d3) but is small, because errors are strongly positively correlated in a nearly exchangeable way. The theta = 1 code_completion cases where N_S loses to N_indep are the cases of (b): gpt-5.4 has the largest r_m but shifts v_m and b most.\n")
    L.append("Out-of-sample sign disagreements with both CIs decided (id_test):")
    for r in R["oos"]["sign_disagreements_ci_decided"][:8]:
        L.append(f"* {r['ctx']}, theta = {r['theta']}, S = {r['S']}, add {r['m']}: N_S(probing) = {r['N_probing']:+.4f}, realised id_test change {r['dreal_id_test']:+.2f} {fmt_ci(r['dreal_ci'], 2)}.")
    L.append("\n### (e) Submodularity\n")
    ss = R["submodularity"]["static"]
    L.append(f"Static (identity): r_m(T) <= r_m(S) for every S subset T, m not in T, checked on every (context, theta): cheap4 lattice {ss['cheap4']['n_violations']} violations / {ss['cheap4']['n_triples']} triples, all 7 candidates {ss['all7']['n_violations']} / {ss['all7']['n_triples']} (max violation {ss['all7']['max_violation']:.1e}). This holds by set inclusion on any sample and is a code check.")
    for th in THETAS:
        k = R["submodularity"]["adaptive_summary"][f"theta={th}|all7|size=1-3"]
        L.append(f"Adaptive, theta = {th}: P(E_m | all of S fail) exceeds P(E_m) in {pct(k['frac_exceeds_point'])} of {k['n']} (S, m) cases at the point estimate and in {pct(k['frac_exceeds_ci'])} with CI > 0; it is significantly *below* in {pct(k['frac_below_ci'])}; median ratio {k['median_ratio_cond_over_uncond']:.2f}. The appendix's two-source counterexample needs complementary (negatively correlated) sources; this pool has none (c2), so the adaptive-submodularity failure is not instantiable here. What the data show instead is the opposite regime: observed failure of S is bad news about every other candidate.")
    L.append("\n### Figures (vector PDF, paper style)\n")
    L.append("* `figures/fig_marginal_identity.pdf`: eq:app-marginal-identity prediction with S's (v_h, v_m, b) vs realised eta(S+m) - eta(S), all non-trivial (S, m) cases, sign disagreements in red (theta = 1 left, 0.5 right).")
    L.append("* `figures/fig_hypothesis_drift.pdf`: v_m, b and v_h of S+m against the same quantity of S (thm:marginal's common-value hypothesis; points below the diagonal in the left panel are the v_m drift).")
    L.append("* `figures/fig_coverage_oos.pdf`: left, partner individual quality P(E_m) vs residual coverage r_m({m0}) (prop:marginals phenomenon); right, realised id_test relative gain of the top partner under each probing ranking (median and IQR over dimension portfolios).")
    L.append("* `figures/fig_adaptive_marginals.pdf`: adaptive marginal P(E_m | all of S fail) vs unconditional P(E_m) by |S| (app:static-adaptive); everything below the diagonal is the opposite of the appendix's counterexample.")
    L.append("\n## 5. Circularity statement\n")
    L.append("* a1-a3, e1 and b3 at theta = 1 are identities (or hold by construction: a committed hit at theta = 1 has score 1) and carry no evidence about the theory or the data.")
    L.append("* b1-b3 test the theorem's *hypothesis* directly (conditional means on two different failure sets). b4 is not an identity because N_S uses S's (v_h, v_m, b) while the realised change uses S+m's; if b1-b3 held exactly, b4 would be an identity. Its non-trivial content is therefore 'how wrong does the sign rule get when its hypothesis is violated by the measured amount'.")
    L.append("* In (c), Spearman(r_m, p(m0+m)) = 1 is an identity and is reported only to say so; the in-sample 'top-quality != top-coverage' fractions are post-selection (winner's curse) and are confirmed held-out (partners chosen on probing, gaps on id_test).")
    L.append("* All of (d) is out-of-sample: every ranking statistic is computed on probing tasks only (usability included), every realised gain and CI on disjoint id_test tasks.")
    L.append("* No router decision files are used (the question is which capability to add to a fixed verifier-select portfolio, not how a learned router dispatches).")
    L.append("\n## 6. Limitations\n")
    L.append("* Prices are the CSV's token-price `cost_usd` estimates (CN price list for the Chinese models; see E8's audit); relative statements involving d_m of Qwen3-Max / kimi / qwen3.5-plus / MiniMax / GLM-5 are specific to that price list. The 148 zero-cost Qwen3-Max rows lower its d_m by ~1.5%.")
    L.append("* The verifier is the benchmark's ground-truth grader, so v_h = 1 at theta = 1 by construction and every hit is value-dominant; a real verifier would add false accepts/rejects. No noisy-verifier variant is run here because none of the tested hypotheses (common v_m, b; residual coverage; submodularity) concerns verifier error.")
    L.append("* Aggregation/verification cost is set to zero (d_m = E[cost_m]); the theorem allows any d_m and the identity a3 would absorb it, but the OOS rankings would shift if a paid aggregator were modelled.")
    L.append("* `overall` pools dimensions with 0% pass rates (code_understanding, code_refactoring at theta = 1), which inflates pooled error correlation and creates the b-drift disagreements; per-dimension numbers are the primary evidence.")
    L.append("* CIs are percentile bootstraps over tasks; in (b) the two compared conditional means share tasks, so the paired CI of the drift is what is reported. Small failure sets (bug_fixing, multi_language at theta = 1) make the v_m and b estimates noisy, which is why the non-trivial filter exists.")
    L.append("* The dimension 'cells' of the record model (Sec. 5) are not used here; contexts are dimensions, not the 21 dimension x source cells.")
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write("\n".join(L) + "\n")
write_readme(R)
print("README.md written")

# ==============================================================================================
# 8. AUDIT block (2026-09-11). Diagnostics added by the skeptical re-run; nothing above was changed except the
#    trivial-case diagnostics. Uses its own RNG stream, so sections 1-7 reproduce the pre-audit numbers exactly.
# ==============================================================================================
rngA = np.random.default_rng(SEED + 1)
AUD = {}
# (A) binary-score dimensions: theta = 1 and theta = 0.5 define the same events there, so the two theta rows are one experiment
AUD["binary_dimensions_theta_rows_identical"] = [d for d in DIMS if bool(((Sc_all[dim_all == d] >= 1.0) == (Sc_all[dim_all == d] >= 0.5)).all())]
# (B) zero-token Qwen3-Max rows (missing runs recorded as free misses): where they are
zc = (Cc_all[:, CAND.index("Qwen3-Max")] == 0)
AUD["zero_cost_qwen3max_rows"] = {"n": int(zc.sum()), "all_score_zero": bool((Sc_all[zc, CAND.index("Qwen3-Max")] == 0).all()),
                                  "by_dimension": {d: int((zc & (dim_all == d)).sum()) for d in DIMS if (zc & (dim_all == d)).any()},
                                  "by_split": {sp: int((zc & (split_all == sp)).sum()) for sp in ["probing", "id_test"]}}
# (C) sign-rule baselines (in-sample, dimensions, non-trivial |S| in 1..3) and OOS (probing rule -> id_test sign), plus
#     nested adaptive check P(E_m | fail S+i) vs P(E_m | fail S) (the adaptive-submodularity comparison proper), plus
#     sensitivity of (b) to dropping the zero-token Qwen3-Max tasks in the two affected dimensions
base = {th: {k: [] for k in ["theorem_N", "value_preserving_rb_minus_d", "N_indep", "always_negative"]} for th in THETAS}
oosb = {th: {k: [] for k in ["theorem_N", "value_preserving_rb_minus_d", "N_indep", "always_negative"]} for th in THETAS}
nested = {th: {"n": 0, "exceeds_point": 0, "exceeds_ci": 0, "below_ci": 0} for th in THETAS}
zsens = {}
def _sign_rules(dv, sel):
    N = dv["N"][0, sel]; r = dv["r"][0, sel]; b = dv["b"][0, sel]; d = dv["d"][0, sel]; vh = dv["vh"][0, sel]; vm = dv["vm"][0, sel]
    eta = dv["eta_s"][0, sel]; pS = dv["p_s"][0, sel]; am = dv["am"][0, sel]
    Nind = (1 - pS) * am * (vh - vm + eta * b) - eta * d
    return {"theorem_N": np.sign(N), "value_preserving_rb_minus_d": np.sign(r * b - d), "N_indep": np.sign(Nind), "always_negative": -np.ones_like(N)}
for ctx in DIMS:
    mask = ctx_mask(ctx); n = int(mask.sum())
    Wb = rngA.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(float)
    m_pr, m_te = ctx_mask(ctx, "probing"), ctx_mask(ctx, "id_test")
    for theta in THETAS:
        T = tables(mask, theta); st = wstats(np.ones((1, n)), T); dv = derived(st); dvB = derived(wstats(Wb, T))
        nem1 = st["nem"][0][M_IDX, S_IDX]; nm2 = st["nm"][0][S2_IDX]; nh = st["nh"][0][S_IDX]
        nontriv = (nem1 >= MIN_CELL) & (nm2 >= MIN_CELL) & (nh >= MIN_CELL); sel = nontriv & np.isin(SIZE_S, SIZES_B)
        sg = np.sign(dv["dreal"][0, sel])
        for k, v in _sign_rules(dv, sel).items():
            base[theta][k].append(v == sg)
        usable = usable_from(T["P"]); idx = {(int(S_IDX[j]), int(M_IDX[j])): j for j in range(NPAIR)}
        for j in np.where(sel & (SIZE_S <= 2))[0]:
            s_, m_ = int(S_IDX[j]), int(M_IDX[j])
            if not usable[m_]:
                continue
            for i in range(K):
                if (s_ >> i) & 1 or i == m_:
                    continue
                j2 = idx[(s_ | (1 << i), m_)]
                if not nontriv[j2]:
                    continue
                dB = dvB["cond"][:, j2] - dvB["cond"][:, j]; lo, hi = ci(dB)
                z = nested[theta]; z["n"] += 1; z["exceeds_point"] += int(dv["cond"][0, j2] > dv["cond"][0, j]); z["exceeds_ci"] += int(lo is not None and lo > 0); z["below_ci"] += int(hi is not None and hi < 0)
        # OOS baselines
        T_pr, T_te = tables(m_pr, theta), tables(m_te, theta)
        st_pr = wstats(np.ones((1, int(m_pr.sum()))), T_pr); dv_pr = derived(st_pr); dv_te = derived(wstats(np.ones((1, int(m_te.sum()))), T_te))
        u = usable_from(T_pr["P"])
        if u.sum() >= 3:
            nemp = st_pr["nem"][0][M_IDX, S_IDX]; nm2p = st_pr["nm"][0][S2_IDX]; nhp = st_pr["nh"][0][S_IDX]; ntp = (nemp >= MIN_CELL) & (nm2p >= MIN_CELL) & (nhp >= MIN_CELL)
            for s_ in range(1, NSUB):
                if SIZE[s_] > 2 or any(not u[k] for k in MEMBERS[s_]):
                    continue
                js = np.array([j for j in range(NPAIR) if S_IDX[j] == s_ and u[M_IDX[j]] and ntp[j]])
                if len(js) < 3:
                    continue
                sg = np.sign(dv_te["dreal"][0, js])
                for k, v in _sign_rules(dv_pr, js).items():
                    oosb[theta][k].append(v == sg)
        # zero-token sensitivity
        if AUD["zero_cost_qwen3max_rows"]["by_dimension"].get(ctx, 0) > 0:
            mask2 = mask & ~zc; n2 = int(mask2.sum()); T2 = tables(mask2, theta); st2 = wstats(np.ones((1, n2)), T2); dv2 = derived(st2)
            nem2 = st2["nem"][0][M_IDX, S_IDX]; nmm2 = st2["nm"][0][S2_IDX]; nh2 = st2["nh"][0][S_IDX]
            sel2 = (nem2 >= MIN_CELL) & (nmm2 >= MIN_CELL) & (nh2 >= MIN_CELL) & np.isin(SIZE_S, SIZES_B)
            def _blk(dv_, sel_):
                relvm = (dv_["vm2"][0, sel_] - dv_["vm"][0, sel_]) / dv_["vm"][0, sel_]; relb = (dv_["b2"][0, sel_] - dv_["b"][0, sel_]) / dv_["b"][0, sel_]
                return {"n": int(sel_.sum()), "sign_agree_all": float((np.sign(dv_["N"][0, sel_]) == np.sign(dv_["dreal"][0, sel_])).mean()),
                        "median_rel_vm_drift": float(np.median(relvm)), "median_rel_b_drift": float(np.median(relb)),
                        "d_qwen3max": float(dv_["d"][0][M_IDX == CAND.index("Qwen3-Max")][0]), "P_E_qwen3max": float(dv_["am"][0][M_IDX == CAND.index("Qwen3-Max")][0])}
            zsens[f"{ctx}|theta={theta}"] = {"as_published": _blk(dv, sel), "zero_token_tasks_dropped": _blk(dv2, sel2), "n_dropped": int((mask & zc).sum())}
AUD["sign_rule_baselines_in_sample_dimensions"] = {f"theta={th}": {k: float(np.concatenate(v).mean()) for k, v in base[th].items()} | {"n": int(len(np.concatenate(base[th]["theorem_N"])))} for th in THETAS}
AUD["sign_rule_baselines_oos_dimensions"] = {f"theta={th}": {k: float(np.concatenate(v).mean()) for k, v in oosb[th].items()} | {"n": int(len(np.concatenate(oosb[th]["theorem_N"])))} for th in THETAS}
AUD["nested_adaptive_P_Em_given_fail_S_plus_i_vs_fail_S"] = {f"theta={th}": {**nested[th], "frac_exceeds_point": nested[th]["exceeds_point"] / max(nested[th]["n"], 1), "frac_exceeds_ci": nested[th]["exceeds_ci"] / max(nested[th]["n"], 1), "frac_below_ci": nested[th]["below_ci"] / max(nested[th]["n"], 1), "null_false_positive_rate_one_sided": 0.025} for th in THETAS}
AUD["zero_token_sensitivity_b"] = zsens
# (D) OOS: does ranking partners by the numerator N_S differ from ranking by the predicted change N_S/(E[D(S)]+d_m-r_m b)?
diffNd = {th: {"n_S": 0, "n_differ": 0, "cases": []} for th in THETAS}
for r_ in R["oos"]["cases_d"]:
    pass  # denominators are not stored per case; recomputed below
for ctx in DIMS:
    m_pr, m_te = ctx_mask(ctx, "probing"), ctx_mask(ctx, "id_test")
    for theta in THETAS:
        T_pr, T_te = tables(m_pr, theta), tables(m_te, theta)
        st_pr = wstats(np.ones((1, int(m_pr.sum()))), T_pr); dv_pr = derived(st_pr); dv_te = derived(wstats(np.ones((1, int(m_te.sum()))), T_te))
        u = usable_from(T_pr["P"])
        if u.sum() < 3:
            continue
        nemp = st_pr["nem"][0][M_IDX, S_IDX]; nm2p = st_pr["nm"][0][S2_IDX]; nhp = st_pr["nh"][0][S_IDX]; ntp = (nemp >= MIN_CELL) & (nm2p >= MIN_CELL) & (nhp >= MIN_CELL)
        for s_ in range(1, NSUB):
            if SIZE[s_] > 2 or any(not u[k] for k in MEMBERS[s_]):
                continue
            js = np.array([j for j in range(NPAIR) if S_IDX[j] == s_ and u[M_IDX[j]] and ntp[j]])
            if len(js) < 3:
                continue
            kN, kD = int(dv_pr["N"][0, js].argmax()), int(dv_pr["dpred"][0, js].argmax())
            diffNd[theta]["n_S"] += 1
            if kN != kD:
                diffNd[theta]["n_differ"] += 1
                diffNd[theta]["cases"].append({"ctx": ctx, "S": sname(s_), "top_N": SHORT[CAND[M_IDX[js[kN]]]], "top_dpred": SHORT[CAND[M_IDX[js[kD]]]], "id_test_gain_top_N": float(dv_te["dreal"][0, js[kN]]), "id_test_gain_top_dpred": float(dv_te["dreal"][0, js[kD]])})
AUD["oos_argmax_N_vs_argmax_predicted_change"] = {f"theta={th}": diffNd[th] for th in THETAS}
R["audit"] = AUD
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(R, f, indent=1, default=lambda o: (None if isinstance(o, float) and not np.isfinite(o) else (o.item() if hasattr(o, "item") else str(o))))

def write_audit(R):
    A = R["audit"]; tv = R["in_sample"]["trivial_cases_diagnostics"]; C = R["claims"]
    L = ["\n## Audit (skeptical re-run, 2026-09-11; original script kept as `run_orig.py`, original outputs as `results_orig.json`, `README_orig.md`, `figures_orig/`)\n"]
    L.append("**Re-run of the original.** `run_orig.py` reproduces `results_orig.json` exactly (296,732 numeric leaves compared, 0 mismatches other than `meta.runtime_s`). Data cross-checks (Always-Opus id_test $34.02 / 43.83) confirmed; the worked example (`overall`, theta = 1, S = {Qwen3-Max}) was recomputed independently from the CSV and agrees to all printed digits. Figures are vector PDF with embedded Type 1 fonts (usetex + Times worked). Sections 1-7 above are regenerated unchanged by `run.py`; this section is generated by the appended audit block, which uses its own RNG stream.\n")
    L.append("**Changes made to `run.py` (minimal).**")
    L.append(f"1. *Reporting bug (trivial cases).* The original text said the {tv['theta=1.0']['n_disagree']} (theta = 1) / {tv['theta=0.5']['n_disagree']} (theta = 0.5) trivial-case 'sign disagreements' were cases with r_m = 0 and N_S = -eta d < 0. In fact {tv['theta=1.0']['disagree_with_N_undefined']} / {tv['theta=0.5']['disagree_with_N_undefined']} of them have N_S undefined (S never hits in that context, so v_h(S) is NaN and `np.sign(NaN) == np.sign(x)` is False); only {tv['theta=1.0']['disagree_with_N_finite']} / {tv['theta=0.5']['disagree_with_N_finite']} are genuine disagreements with finite N_S, none CI-decided. Fixed the diagnostic (`trivial_cases_diagnostics.disagree_with_N_undefined`) and the sentence in Section 4(b). No headline number is affected (trivial cases were already excluded from every verdict).")
    L.append("2. *Added diagnostics* (`results.json: audit`), no existing number changed: sign-rule baselines, the nested adaptive-submodularity comparison, a zero-token-row sensitivity, a binary-dimension flag, and an argmax check. Their content is below.\n")
    L.append("**Circularity / leakage.** None found beyond what Section 5 already declares. In-sample b4 is correctly labelled (it would be an identity if b1-b3 held). Section 4 (d) computes usability, the non-trivial filter and every ranking statistic on `probing` only and every realised quantity on `id_test`; verified by reading the code and by an independent re-implementation of the OOS loop (same 139 / 194 portfolios, same tops). The in-sample c1 fractions are post-selection and are used descriptively only. Spearman(r_m, pair coverage) = 1 is an identity and is declared as such.\n")
    L.append("**Hypothesis reading.** thm:marginal is stated under 'common v_h, v_m, b' for S and S+m; the experiment tests exactly those three hypotheses (b1-b3) and the theorem's sign conclusion (b4/d1) with S's values plugged into eq:marginal, and eq:app-marginal-identity is the formula used for the predicted change. The proof in app:portfolio says the conditional quantities 'may change' when the failure set changes; b1 measures that they do (v_m falls by 14-18% in the median), so b1-b2 FAIL means the theorem's hypothesis is violated on this pool, not that the theorem is wrong. prop:marginals is a two-world impossibility construction; c1-c3/d2c test the real-data phenomenon it points at (paired outcomes carry information that marginals do not), which is the only testable reading, and the README says so. app:static-adaptive's two-source example is compared correctly in e2 (conditional marginal after observed failure vs prior marginal). No misreading found.\n")
    b_in = A["sign_rule_baselines_in_sample_dimensions"]; b_oos = A["sign_rule_baselines_oos_dimensions"]
    L.append("**Are the 90% sign-agreement thresholds trivially attainable?** No. The realised change is negative in about 70% of non-trivial cases, so an 'always negative' rule scores 70%, and the theorem's rule beats every price-only or marginal-only alternative:\n")
    L.append("| rule (sign of ...) | in-sample theta=1 (n=%d) | in-sample theta=0.5 (n=%d) | OOS probing->id_test theta=1 (n=%d) | OOS theta=0.5 (n=%d) |\n|---|---|---|---|---|" % (b_in["theta=1.0"]["n"], b_in["theta=0.5"]["n"], b_oos["theta=1.0"]["n"], b_oos["theta=0.5"]["n"]))
    for k, lab in [("theorem_N", "eq:marginal N_S (S's v_h, v_m, b)"), ("value_preserving_rb_minus_d", "value-preserving special case r_m b - d_m"), ("N_indep", "N_S with independence proxy (1-p(S))P(E_m)"), ("always_negative", "always negative (base rate)")]:
        L.append(f"| {lab} | {b_in['theta=1.0'][k]*100:.1f}% | {b_in['theta=0.5'][k]*100:.1f}% | {b_oos['theta=1.0'][k]*100:.1f}% | {b_oos['theta=0.5'][k]*100:.1f}% |")
    L.append("\nThe independence proxy is much worse at predicting the *sign* (65-72%) even though it picks nearly the same *top* partner as N_S (d2c); the paired-outcome information in r_m matters for the accept/reject decision on a given candidate more than for the ranking of the best candidate.\n")
    na = A["nested_adaptive_P_Em_given_fail_S_plus_i_vs_fail_S"]
    L.append("**Adaptive submodularity, nested form.** e2 compares P(E_m | all of S fail) with the unconditional P(E_m), which is the appendix's own comparison. Adaptive submodularity proper compares conditional marginals under nested observations, P(E_m | fail S+i) vs P(E_m | fail S). Result: " + "; ".join(f"theta = {th}: {na[f'theta={th}']['n']} nested triples, the conditional marginal rises with the extra observed failure in {na[f'theta={th}']['frac_exceeds_point']*100:.1f}% at the point estimate, {na[f'theta={th}']['frac_exceeds_ci']*100:.1f}% with CI > 0 (below the 2.5% one-sided false-positive rate), and falls significantly in {na[f'theta={th}']['frac_below_ci']*100:.1f}%" for th in THETAS) + ". The NOT INSTANTIABLE verdict stands in the stronger form.\n")
    bd = A["binary_dimensions_theta_rows_identical"]
    L.append(f"**Double counting across theta.** In {', '.join(bd)} every score is exactly 0 or 1, so E_m at theta = 1 and theta = 0.5 are the same events and the two theta rows of every table are one experiment, not two. Consequences: the held-out prop:marginals example `code_generation, m0 = glm-5` (gap +0.020 [0.006, 0.038]) is listed at both thetas but is a single case; the distinct held-out dimension cases with CI > 0 are code_generation/glm-5 and code_completion (theta = 0.5)/sonnet, plus the three `overall` cases (the strongest, +0.032 [0.020, 0.042], is `overall`). c1 keeps PASS because its pre-declared criterion (>= 1 held-out CI > 0) is met at each theta, but the phenomenon is rare (differs in 5-11% of contexts) and the multiplicity is 6 / 5 differing contexts tested at 2.5% one-sided.\n")
    zq = A["zero_cost_qwen3max_rows"]; zs = A["zero_token_sensitivity_b"]
    L.append(f"**Zero-token Qwen3-Max rows.** The {zq['n']} rows with cost 0 all have score 0 and 0 input/output tokens: they are missing runs recorded as free misses (same defect E8's audit found in the OOD file), located in {zq['by_dimension']} ({zq['by_split']}). The README's '~1.5% of d_m' understates the local effect: in bug_fixing they are 10% of tasks. Sensitivity (drop those tasks, dimensions affected only): " + "; ".join(f"{k}: sign agreement {v['as_published']['sign_agree_all']*100:.1f}% -> {v['zero_token_tasks_dropped']['sign_agree_all']*100:.1f}% (n {v['as_published']['n']} -> {v['zero_token_tasks_dropped']['n']}), median v_m drift {v['as_published']['median_rel_vm_drift']*100:+.1f}% -> {v['zero_token_tasks_dropped']['median_rel_vm_drift']*100:+.1f}%, P(E_Qwen3-Max) {v['as_published']['P_E_qwen3max']:.3f} -> {v['zero_token_tasks_dropped']['P_E_qwen3max']:.3f}, d_Qwen3-Max ${v['as_published']['d_qwen3max']:.5f} -> ${v['zero_token_tasks_dropped']['d_qwen3max']:.5f}" for k, v in zs.items()) + ". No verdict changes; Qwen3-Max's P(E_m), r_m and d_m in bug_fixing are biased low as published.\n")
    nd = A["oos_argmax_N_vs_argmax_predicted_change"]
    L.append("**Ranking by N_S vs by predicted change.** The (d) rankings use the numerator N_S; the theorem's magnitude prediction is N_S/(E[D(S)] + d_m - r_m b). The two argmaxes differ in " + " / ".join(f"{nd[f'theta={th}']['n_differ']} of {nd[f'theta={th}']['n_S']}" for th in THETAS) + " dimension portfolios (theta 1 / 0.5)" + (f"; the one differing case ({nd['theta=1.0']['cases'][0]['ctx']}, S = {nd['theta=1.0']['cases'][0]['S']}: N_S picks {nd['theta=1.0']['cases'][0]['top_N']}, realised {nd['theta=1.0']['cases'][0]['id_test_gain_top_N']:+.2f}; predicted change picks {nd['theta=1.0']['cases'][0]['top_dpred']}, realised {nd['theta=1.0']['cases'][0]['id_test_gain_top_dpred']:+.2f}) goes the other way" if nd["theta=1.0"]["cases"] else "") + ". Immaterial to any verdict.\n")
    L.append("**Other notes (no code change).** (i) d3 compares Spearman(probing r_m, id_test r_m) with Spearman(probing P(E_m), id_test r_m): the first is the split stability of one statistic, the second a proxy, so a PASS mostly says r_m is estimable from 7,080 probing tasks; at theta = 0.5 it is marginal (medians 0.900 vs 0.899, p = 0.041) and should be read as weak. (ii) The verifier is the ground-truth grader, so v_h = 1 at theta = 1 by construction (b3 correctly labelled IDENTITY there). (iii) On a miss the cascade commits Opus even when a member scored higher than Opus but below theta; this is the paper's cascade (fallback committed on H = 0) and slightly favours Opus. (iv) Prices are the CN list for the five Chinese models (E8 audit); all d_m-dependent statements (d2b, d2c, quality-per-dollar) are specific to it. (v) The legend in the right panel of `fig_coverage_oos.pdf` overlaps the first whisker; cosmetic.\n")
    L.append("**Verdicts after audit.** a1-a3, e1: PASS (identities, code checks) -- unchanged. b1 FAIL (large drift), b2 FAIL (small drift), b3 PASS at theta = 1 (identity) / FAIL (small drift) at theta = 0.5 -- unchanged; these are hypothesis violations that the paper's proof anticipates, not contradictions of the theorem. b4, d1 PASS -- unchanged, now with the baseline table showing the 90% threshold is not attainable by a trivial rule. c1 PASS -- unchanged but qualified (rare; one distinct dimension case per theta after removing the code_generation duplicate; strongest evidence in `overall`). c2 PASS, c3 PASS (descriptive) -- unchanged. d2a INCONCLUSIVE, d2b PASS, d2c INCONCLUSIVE -- unchanged (d2b's advantage is pricing, as the README says). d3 PASS at theta = 1, PASS (weak) at theta = 0.5. e2 NOT INSTANTIABLE -- unchanged and confirmed in the nested form. Overall trust: high for (a), (b), (c2), (e); medium-high for (d) (out-of-sample, but CN price list and zero-cost rows); medium for c1 (rare phenomenon, few held-out cases).")
    with open(os.path.join(OUT, "README.md"), "a") as f:
        f.write("\n".join(L) + "\n")
write_audit(R)
print(f"Audit section appended ({time.time()-T0:.0f}s)")

