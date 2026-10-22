#!/usr/bin/env python3
"""
X11_residual_coverage_routerbench -- identifiability of residual coverage on RouterBench (0-shot):
eq:residual, thm:marginal / eq:marginal, prop:marginals and rem:frechet (Frechet range of r_m and the excess
cost of a marginal-only selector) of "Route Anything: A Theory of Inference Routers".

Data: RouterBench 0-shot pickle ($ROUTERBENCH_PKL; 36,497 prompts x 11 models; per-model score column `<model>`
and cost column `<model>|total_cost`; 86 `eval_name`). Reference B = gpt-4-1106-preview (fallback on a miss).
Everything is computed from the recorded scores and costs; the only randomness is the stratified 70/30 split
(seed 20260917) and task resampling (bootstrap, same seed stream). Fully offline.

Objects (paper notation, Sec. 3), copied from experiments/X2_residual_coverage/run.py:
  E_m    = {score_m >= theta}, theta in {0.5, 1.0}                         exact verifier
  p(S)   = P(U_{m in S} E_m);  r_m(S) = P(E_m & all i in S fail)           (eq:residual; OR semantics)
  cascade(S): all members of S are called (parallel); an exact selecting verifier commits the highest-scoring
              passing member; on a miss gpt-4 is called and committed.
  V = committed score;  D = sum_{i in S} cost_i + (1-H) cost_gpt4        (eq:accounting)
  a(S) = E[sum member costs];  d_m(S) = a(S+m) - a(S) = E[cost_m]  (parallel, no aggregation fee)
  v_h(S) = E[V | hit];  v_m(S) = E[Y_gpt4 | miss];  b(S) = E[C_gpt4 | miss];  eta(S) = E[V]/E[D] (ratio of sums)
  eq:marginal with S's (v_h, v_m, b):  N_S = r_m(S)[v_h - v_m + eta(S) b] - eta(S) d_m(S);
      sign(N_S) predicts sign(eta(S+m) - eta(S));  eq:app-marginal-identity: predicted change = N_S/(E[D(S)] + d - r b).
  rem:frechet: for S = {m0}:  max{0, p_m - p_0} <= r_m(S) <= min{p_m, 1 - p_0};  for equal marginals p_1 = p_2 = p the
      minimax excess of a marginal-only selector is (b/2)[min{p, 1-p_0} - max{0, p - p_0}]; for unequal marginals it is
      b xy/(x+y), x = (u_1 - l_2)^+, y = (u_2 - l_1)^+, attained by the selector q = x/(x+y).
Estimator names follow X2: b4 (in-sample sign rule), d1 (out-of-sample sign rule, fit -> held-out), d2 (out-of-sample
top-partner rankings), d3 (Spearman of fit-split r_m vs fit-split P(E_m) against held-out quantities).
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
SEED = 20260917
N_BOOT = 500                     # 500 (not 1,000): three bootstraps per context (full, fit, held-out) x 66 contexts
THETA_MAIN = 0.5
THETAS = [0.5, 1.0]              # theta = 1.0 only changes E_m where scores are not binary (gsm8k, rag-other, mt-bench)
FIT_FRAC = 0.7
PKL = os.environ.get("ROUTERBENCH_PKL", "./data/routerbench/routerbench_0shot.pkl")
OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

REF = "gpt-4-1106-preview"
POOL_MAIN = ["mistralai/mixtral-8x7b-chat", "zero-one-ai/Yi-34B-Chat", "claude-instant-v1", "gpt-3.5-turbo-1106"]
POOL_EXTRA = ["WizardLM/WizardLM-13B-V1.2", "meta/llama-2-70b-chat"]
POOL = POOL_MAIN + POOL_EXTRA
SHORT = {"WizardLM/WizardLM-13B-V1.2": "wizard13b", "claude-instant-v1": "claude-inst", "claude-v1": "claude-v1",
         "claude-v2": "claude-v2", "gpt-3.5-turbo-1106": "gpt35", "gpt-4-1106-preview": "gpt4",
         "meta/code-llama-instruct-34b-chat": "codellama34b", "meta/llama-2-70b-chat": "llama2-70b",
         "mistralai/mistral-7b-chat": "mistral7b", "mistralai/mixtral-8x7b-chat": "mixtral", "zero-one-ai/Yi-34B-Chat": "yi34b"}
MIN_COUNT = 20     # a model is *usable* in a sample if it has >= 20 passes and >= 20 failures (X2 convention)
MIN_CELL = 10      # an (S, m) case is *non-trivial* if #(E_m & fail S) >= 10, #fail(S+m) >= 10, #hit(S) >= 10 (X2)
EQ_TOL = 0.02      # equal-marginal pairs: |P(E_m1) - P(E_m2)| < 0.02
MIN_MMLU = 100
ID = lambda x: (float(x) if x is not None and np.isfinite(x) else None)

# ---- figure style (CONVENTIONS: matplotlib, usetex False, 9-pt serif, no titles inside plots) -------------------
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'], 'text.usetex': False,
    'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 7.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': .65, 'lines.linewidth': 1.2,
})
BLUE, RED, GREEN, ORANGE, GREY = '#23639b', '#ba5934', '#3a7c5e', '#c98a2b', '.55'

# ==============================================================================================
# 1. Load real data
# ==============================================================================================
df = pd.read_pickle(PKL)
MODELS = [c for c in df.columns if f"{c}|total_cost" in df.columns]
assert len(MODELS) == 11 and REF in MODELS and df.shape[0] == 36497, (len(MODELS), df.shape)
NONREF = [m for m in MODELS if m != REF]                       # 10 candidate / primary models, fixed order
K = len(NONREF)
POOL_IDX = [NONREF.index(m) for m in POOL]
POOL_MAIN_IDX = [NONREF.index(m) for m in POOL_MAIN]
S_all = df[MODELS].astype(float).to_numpy()
C_all = df[[f"{m}|total_cost" for m in MODELS]].astype(float).to_numpy()
assert not np.isnan(S_all).any() and not np.isnan(C_all).any() and S_all.min() >= 0 and S_all.max() <= 1 and C_all.min() > 0
so_all, co_all = S_all[:, MODELS.index(REF)], C_all[:, MODELS.index(REF)]
Sc_all, Cc_all = S_all[:, [MODELS.index(m) for m in NONREF]], C_all[:, [MODELS.index(m) for m in NONREF]]
ev_all = df["eval_name"].astype(str).to_numpy()
N_ALL = len(df)
# cross-checks against the RouterBench paper / CONVENTIONS: gpt-4 mean score 0.781, $0.0033 per prompt
assert abs(so_all.mean() - 0.7814) < 0.001 and abs(co_all.mean() - 0.003293) < 1e-5, (so_all.mean(), co_all.mean())

# ---- contexts: 8 source families (explicit grouping of eval_name) + MMLU subsets with >= 100 prompts + pooled 'overall'
FAMILY_ORDER = ["mmlu", "hellaswag", "grade-school-math", "arc-challenge", "winogrande", "mbpp", "mt-bench", "rag-other"]
def family_of(e):
    if e.startswith("mmlu-"):
        return "mmlu"
    if e in ("hellaswag", "winogrande", "arc-challenge", "grade-school-math", "mbpp"):
        return e
    if e.startswith("mtbench"):
        return "mt-bench"
    return "rag-other"          # every remaining eval_name (RouterBench's RAG / Chinese / summarisation datasets)
fam_all = np.array([family_of(e) for e in ev_all])
FAMILY_MEMBERS = {f: sorted(set(ev_all[fam_all == f])) for f in FAMILY_ORDER}
assert set(fam_all) == set(FAMILY_ORDER)
ev_counts = pd.Series(ev_all).value_counts()
MMLU_SUBSETS = sorted([e for e, n in ev_counts.items() if e.startswith("mmlu-") and n >= MIN_MMLU])
CONTEXTS = [("overall", "overall")] + [(f, "family") for f in FAMILY_ORDER] + [(e, "mmlu_subset") for e in MMLU_SUBSETS]
def ctx_rows(name, kind):
    if kind == "overall":
        return np.ones(N_ALL, bool)
    if kind == "family":
        return fam_all == name
    return ev_all == name
def is_binary_ctx(mask):
    return bool((((Sc_all[mask] == 0) | (Sc_all[mask] == 1)).all()) and (((so_all[mask] == 0) | (so_all[mask] == 1)).all()))
BINARY = {name: is_binary_ctx(ctx_rows(name, kind)) for name, kind in CONTEXTS}
def thetas_for(name):
    return [THETA_MAIN] if BINARY[name] else THETAS

# ---- stratified 70/30 split (strata = eval_name x 1{gpt-4 score >= 0.5}); every context inherits the same assignment
rng = np.random.default_rng(SEED)
is_fit = np.zeros(N_ALL, bool)
strata = pd.Series([f"{e}|{int(s >= THETA_MAIN)}" for e, s in zip(ev_all, so_all)])
for key in sorted(strata.unique()):
    idx = np.where(strata.to_numpy() == key)[0]
    perm = rng.permutation(idx)
    is_fit[perm[:int(round(FIT_FRAC * len(idx)))]] = True
print(f"loaded {N_ALL} prompts x {len(MODELS)} models; fit {is_fit.sum()} / held-out {(~is_fit).sum()}; "
      f"contexts: 8 families + {len(MMLU_SUBSETS)} MMLU subsets + overall; non-binary contexts: {[n for n, k in CONTEXTS if not BINARY[n]]}  ({time.time()-T0:.0f}s)")

# ==============================================================================================
# 2. Portfolios: every subset of the 10 non-reference models with |S| <= 3 (bitmask), per-task tables, weighted stats
# ==============================================================================================
def popcount(s):
    return bin(s).count("1")
MASKS = [s for s in range(1 << K) if popcount(s) <= 3]                 # 176 subsets incl. the empty portfolio (= always gpt-4)
COL = {s: i for i, s in enumerate(MASKS)}
MEMBERS = [[k for k in range(K) if (s >> k) & 1] for s in MASKS]
SIZE = np.array([len(m) for m in MEMBERS])
NSUB = len(MASKS)
SINGLE_COL = np.array([COL[1 << k] for k in range(K)])
# (S, m) cases: S = portfolio of size 1-2 from the 6-model pool, m = any other non-reference model
POOL_PORTFOLIOS = sorted({(1 << i) for i in POOL_IDX} | {(1 << i) | (1 << j) for i in POOL_IDX for j in POOL_IDX if i < j})
S_IDX, M_IDX, S2_IDX, S_MASK = [], [], [], []
for s in POOL_PORTFOLIOS:
    for m in range(K):
        if not (s >> m) & 1:
            S_IDX.append(COL[s]); M_IDX.append(m); S2_IDX.append(COL[s | (1 << m)]); S_MASK.append(s)
S_IDX, M_IDX, S2_IDX, S_MASK = map(np.array, (S_IDX, M_IDX, S2_IDX, S_MASK))
NPAIR = len(S_IDX)
SIZE_S = SIZE[S_IDX]
MEMB_S = np.array([[1.0 if (s >> k) & 1 else 0.0 for k in range(K)] for s in S_MASK])   # NPAIR x K membership of S
M0_OF = np.array([MEMBERS[S_IDX[j]][0] if SIZE_S[j] == 1 else -1 for j in range(NPAIR)])
MAIN4_S = np.array([all(k in POOL_MAIN_IDX for k in MEMBERS[S_IDX[j]]) for j in range(NPAIR)])
print(f"{NSUB} subsets, {len(POOL_PORTFOLIOS)} pool portfolios, {NPAIR} (S, m) cases per context")

def tables(rows, theta):
    S, C, o_s, o_c = Sc_all[rows], Cc_all[rows], so_all[rows], co_all[rows]
    P = S >= theta
    n = S.shape[0]
    hit = np.zeros((n, NSUB), bool); Vsel = np.zeros((n, NSUB)); A = np.zeros((n, NSUB))
    for i, mem in enumerate(MEMBERS):
        if not mem:
            continue                                        # empty portfolio: always miss -> always gpt-4 (reference policy)
        Pm = P[:, mem]
        hit[:, i] = Pm.any(1)
        v = np.where(Pm, S[:, mem], -np.inf).max(1)         # exact selecting verifier: best passing member
        Vsel[:, i] = np.where(hit[:, i], v, 0.0)
        A[:, i] = C[:, mem].sum(1)
    hf = hit.astype(float)
    return {"P": P, "Pf": P.astype(float), "hf": hf, "Vh": hf * Vsel, "V": hf * Vsel + (1 - hf) * o_s[:, None],
            "D": A + (1 - hf) * o_c[:, None], "A": A, "Mv": (1 - hf) * o_s[:, None], "Mc": (1 - hf) * o_c[:, None],
            "C": C, "o_s": o_s, "o_c": o_c, "n": n}

def wstats(W, T):
    """Portfolio statistics under task weights W (B x n): arrays (B x NSUB) / (B x K) / (B x K x NSUB)."""
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
        fail = 1 - T["hf"]
        nem = np.stack([(W * T["Pf"][:, m][None, :]) @ fail for m in range(K)], axis=1)   # counts of E_m & fail(S)
    return {"p": p, "EV": EV, "ED": ED, "eta": eta, "vh": vh, "vm": vm, "b": b, "a": a, "d": d,
            "nh": nh, "nm": nm, "nem": nem, "nw": nw}

def derived(st):
    """Per-(S, m) quantities of thm:marginal and rem:frechet, vectorised over bootstrap replicates."""
    s, m, s2 = S_IDX, M_IDX, S2_IDX
    with np.errstate(divide="ignore", invalid="ignore"):
        p_s = st["p"][:, s]; p2 = st["p"][:, s2]
        r = p2 - p_s
        eta_s = st["eta"][:, s]; vh = st["vh"][:, s]; vm = st["vm"][:, s]; b = st["b"][:, s]; d = st["d"][:, m]
        W_ = vh - vm + eta_s * b                                          # value-plus-avoided-fallback weight
        N = r * W_ - eta_s * d                                            # eq:marginal numerator with S's values
        denom = st["ED"][:, s] + d - r * b
        dpred = N / denom                                                 # eq:app-marginal-identity
        dreal = st["eta"][:, s2] - eta_s                                  # realised change (S+m's own values)
        vh2, vm2, b2 = st["vh"][:, s2], st["vm"][:, s2], st["b"][:, s2]
        am = st["p"][:, SINGLE_COL[m]]                                    # P(E_m)
        psing = st["p"][:, SINGLE_COL]                                    # B x K individual success probabilities
        sum_pS = psing @ MEMB_S.T                                         # sum_{i in S} p_i
        max_pS = np.max(np.where(MEMB_S[None, :, :] > 0, psing[:, None, :], -np.inf), axis=2)
        ell = np.maximum(0.0, am - sum_pS)                                # Frechet lower bound (rem:frechet, general form)
        u = np.minimum(am, 1 - max_pS)                                    # Frechet upper bound
        r_ind = (1 - p_s) * am                                            # independence proxy for r_m(S)
        N_ind = r_ind * W_ - eta_s * d
        N_fu = u * W_ - eta_s * d                                         # optimistic marginal-only rule (upper bound)
        N_fl = ell * W_ - eta_s * d                                       # pessimistic marginal-only rule (lower bound)
        cond = r / (1 - p_s)
        def eta_hyp(vh_, vm_, b_):
            return (p2 * vh_ + (1 - p2) * vm_) / (st["a"][:, s] + d + (1 - p2) * b_) - eta_s
        dpred_vm2 = eta_hyp(vh, vm2, b); dpred_b2 = eta_hyp(vh, vm, b2); dpred_vh2 = eta_hyp(vh2, vm, b); dpred_all2 = eta_hyp(vh2, vm2, b2)
    return {"r": r, "N": N, "dpred": dpred, "dreal": dreal, "eta_s": eta_s, "eta_s2": st["eta"][:, s2], "W": W_,
            "vh": vh, "vm": vm, "b": b, "vh2": vh2, "vm2": vm2, "b2": b2, "d": d, "am": am, "p_s": p_s, "p_s2": p2,
            "ell": ell, "u": u, "r_ind": r_ind, "N_ind": N_ind, "N_fu": N_fu, "N_fl": N_fl, "cond": cond,
            "dpred_vm2": dpred_vm2, "dpred_b2": dpred_b2, "dpred_vh2": dpred_vh2, "dpred_all2": dpred_all2}

def sign_rules(dv, sel, k=0):
    """Sign predictions of the compared rules at replicate k on the selected cases."""
    g = lambda key: dv[key][k, sel]
    return {"theorem_N": np.sign(g("N")), "N_indep": np.sign(g("N_ind")), "frechet_upper": np.sign(g("N_fu")),
            "frechet_lower": np.sign(g("N_fl")), "value_preserving_rb_minus_d": np.sign(g("r") * g("b") - g("d")),
            "marginal_quality_gt_pS": np.sign(g("am") - g("p_s")), "always_negative": -np.ones(g("N").shape[0]),
            "always_positive": np.ones(g("N").shape[0])}
RULES = ["theorem_N", "N_indep", "frechet_upper", "frechet_lower", "value_preserving_rb_minus_d",
         "marginal_quality_gt_pS", "marginal_quality_fitted_tau", "always_negative", "always_positive"]

def fit_threshold(am, sgn):
    """Best P(E_m)-only threshold rule (+ iff P(E_m) >= tau) on the fit split's own realised signs; ties -> larger tau."""
    best_tau, best_acc = np.inf, -1.0
    for tau in np.concatenate([np.unique(am), [np.inf]]):
        acc = float((np.where(am >= tau, 1, -1) == sgn).mean())
        if acc >= best_acc - 1e-12:
            best_tau, best_acc = float(tau), acc
    return best_tau

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
    return bool((lo is not None) and (lo > 0 or hi < 0))

def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(stats.spearmanr(x[ok], y[ok]).statistic)

def usable_from(P):
    npass = P.sum(0)
    return (npass >= MIN_COUNT) & ((P.shape[0] - npass) >= MIN_COUNT)

def sname(col):
    return "+".join(SHORT[NONREF[k]] for k in MEMBERS[col]) if MEMBERS[col] else "{}"

def nontrivial_from(st):
    nem1 = st["nem"][0][M_IDX, S_IDX]; nm2 = st["nm"][0][S2_IDX]; nh = st["nh"][0][S_IDX]
    return (nem1 >= MIN_CELL) & (nm2 >= MIN_CELL) & (nh >= MIN_CELL), nem1, nm2, nh

# ==============================================================================================
# 3. Main loop over contexts: in-sample identities / b4 / drift (C4) / c2; Frechet (C3); OOS d1 / d2 / d3; OOS selector
# ==============================================================================================
R = {"meta": {"seed": SEED, "n_boot": N_BOOT, "thetas": THETAS, "theta_main": THETA_MAIN, "fit_frac": FIT_FRAC, "reference": REF,
              "pool_primaries": POOL, "pool_main4": POOL_MAIN, "pool_extras": POOL_EXTRA, "candidates": NONREF,
              "min_count_usable": MIN_COUNT, "min_cell_nontrivial": MIN_CELL, "equal_marginal_tol": EQ_TOL,
              "n_prompts": int(N_ALL), "n_fit": int(is_fit.sum()), "n_heldout": int((~is_fit).sum()),
              "stratification": "eval_name x 1{gpt-4 score >= 0.5}", "n_pool_portfolios": len(POOL_PORTFOLIOS), "n_cases_per_context": int(NPAIR),
              "family_grouping": FAMILY_MEMBERS, "mmlu_subsets": MMLU_SUBSETS, "binary_contexts": BINARY,
              "gpt4_mean_score": float(so_all.mean()), "gpt4_mean_cost": float(co_all.mean())},
     "contexts": {}, "identity": {}, "in_sample": {}, "frechet": {}, "oos": {}, "claims": {}}
ident = {"max_abs_r_identity": 0.0, "max_abs_eta_homogeneous": 0.0, "max_abs_marginal_identity": 0.0, "max_abs_all2_equals_real": 0.0, "n_cases_checked": 0}
rows_b, rows_corr, rows_fre, rows_exc, rows_fre_oos, oos_rows, oos_sign, oos_exc = [], [], [], [], [], [], [], []
excB_store = []          # per context: dict with per-replicate arrays for the pooled Spearman(excess, width) CI
oosexcB_store = []
PAIRINGS = [("theorem_N", "quality_a"), ("coverage_r", "quality_a"), ("theorem_N", "theorem_N_indep"), ("theorem_N", "quality_per_dollar")]
RANKINGS = ["quality_a", "coverage_r", "quality_per_dollar", "theorem_N_indep", "theorem_N", "oracle_heldout"]
J_OF = {(int(S_IDX[j]), int(M_IDX[j])): j for j in range(NPAIR)}

for ci_idx, (name, kind) in enumerate(CONTEXTS):
    rows = ctx_rows(name, kind); rows_fit = rows & is_fit; rows_te = rows & ~is_fit
    n, n_fit, n_te = int(rows.sum()), int(rows_fit.sum()), int(rows_te.sum())
    rng_c = np.random.default_rng([SEED, ci_idx])
    W_full = rng_c.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(float)
    W_te = rng_c.multinomial(n_te, np.full(n_te, 1.0 / n_te), size=N_BOOT).astype(float)
    R["contexts"][name] = {"kind": kind, "n": n, "n_fit": n_fit, "n_heldout": n_te, "binary_scores": BINARY[name],
                           "members": (FAMILY_MEMBERS[name] if kind == "family" else [name] if kind == "mmlu_subset" else "all")}
    for theta in thetas_for(name):
        key = f"{name}|theta={theta}"
        T_full, T_fit, T_te = tables(rows, theta), tables(rows_fit, theta), tables(rows_te, theta)
        st1 = wstats(np.ones((1, n)), T_full); stB = wstats(W_full, T_full)
        st_fit = wstats(np.ones((1, n_fit)), T_fit)
        st_te = wstats(np.ones((1, n_te)), T_te); st_teB = wstats(W_te, T_te)
        dv1, dvB, dv_fit, dv_te, dv_teB = derived(st1), derived(stB), derived(st_fit), derived(st_te), derived(st_teB)
        usable = usable_from(T_full["P"]); usable_fit = usable_from(T_fit["P"])
        nontriv, nem1, nm2, nh = nontrivial_from(st1)
        nontriv_fit, _, _, _ = nontrivial_from(st_fit)
        R["contexts"][name][f"theta={theta}"] = {"eta_gpt4": ID(st1["eta"][0, COL[0]]), "vbar_gpt4": ID(so_all[rows].mean()), "cbar_gpt4": ID(co_all[rows].mean()),
                                                 "p_gpt4": ID(np.mean(so_all[rows] >= theta)),
                                                 "P_E_m": {SHORT[NONREF[k]]: ID(st1["p"][0, SINGLE_COL[k]]) for k in range(K)},
                                                 "d_m": {SHORT[NONREF[k]]: ID(st1["d"][0, k]) for k in range(K)},
                                                 "usable_full": [SHORT[NONREF[k]] for k in range(K) if usable[k]],
                                                 "usable_fit": [SHORT[NONREF[k]] for k in range(K) if usable_fit[k]]}
        # ---------- (a) identities (code checks) ----------------------------------------------------------------
        r_direct = st1["nem"][0][M_IDX, S_IDX] / n
        ident["max_abs_r_identity"] = max(ident["max_abs_r_identity"], smax(np.abs(r_direct - dv1["r"][0])))
        p1, vh1, vm1, b1, a1 = st1["p"][0], st1["vh"][0], st1["vm"][0], st1["b"][0], st1["a"][0]
        okh = (st1["nh"][0] > 0) & (st1["nm"][0] > 0)
        eta_hom = (p1 * vh1 + (1 - p1) * vm1) / (a1 + (1 - p1) * b1)
        ident["max_abs_eta_homogeneous"] = max(ident["max_abs_eta_homogeneous"], smax(np.abs(eta_hom[okh] - st1["eta"][0][okh])))
        p2 = dv1["p_s2"][0]; eta_hyp2 = (p2 * dv1["vh"][0] + (1 - p2) * dv1["vm"][0]) / (a1[S_IDX] + dv1["d"][0] + (1 - p2) * dv1["b"][0])
        okp = np.isfinite(dv1["dpred"][0])
        ident["max_abs_marginal_identity"] = max(ident["max_abs_marginal_identity"], smax(np.abs((eta_hyp2 - dv1["eta_s"][0])[okp] - dv1["dpred"][0][okp])))
        okr = np.isfinite(dv1["dpred_all2"][0])
        ident["max_abs_all2_equals_real"] = max(ident["max_abs_all2_equals_real"], smax(np.abs(dv1["dpred_all2"][0][okr] - dv1["dreal"][0][okr])))
        ident["n_cases_checked"] += int(okp.sum())
        # ---------- (b) in-sample b4 sign rule and hypothesis drift (C4) ------------------------------------------
        for j in np.where(nontriv)[0]:
            rec = {"ctx": name, "kind": kind, "theta": theta, "S": sname(S_IDX[j]), "m": SHORT[NONREF[M_IDX[j]]], "size_S": int(SIZE_S[j]), "main4": bool(MAIN4_S[j]),
                   "n_E_m_and_failS": int(nem1[j]), "n_fail_S2": int(nm2[j]), "n_hit_S": int(nh[j]),
                   "r": ID(dv1["r"][0, j]), "r_ind": ID(dv1["r_ind"][0, j]), "P_E_m": ID(dv1["am"][0, j]), "p_S": ID(dv1["p_s"][0, j]), "d": ID(dv1["d"][0, j]),
                   "eta_S": ID(dv1["eta_s"][0, j]), "eta_S2": ID(dv1["eta_s2"][0, j]), "N": ID(dv1["N"][0, j]), "N_ci": ci(dvB["N"][:, j]),
                   "N_ind": ID(dv1["N_ind"][0, j]), "dpred": ID(dv1["dpred"][0, j]), "dreal": ID(dv1["dreal"][0, j]), "dreal_ci": ci(dvB["dreal"][:, j]),
                   "vh_S": ID(dv1["vh"][0, j]), "vh_S2": ID(dv1["vh2"][0, j]), "dvh_ci": ci(dvB["vh2"][:, j] - dvB["vh"][:, j]),
                   "vm_S": ID(dv1["vm"][0, j]), "vm_S2": ID(dv1["vm2"][0, j]), "dvm_ci": ci(dvB["vm2"][:, j] - dvB["vm"][:, j]),
                   "b_S": ID(dv1["b"][0, j]), "b_S2": ID(dv1["b2"][0, j]), "db_ci": ci(dvB["b2"][:, j] - dvB["b"][:, j]),
                   "dpred_vm2": ID(dv1["dpred_vm2"][0, j]), "dpred_b2": ID(dv1["dpred_b2"][0, j]), "dpred_vh2": ID(dv1["dpred_vh2"][0, j])}
            rec["sign_agree"] = bool(rec["N"] is not None and rec["dreal"] is not None and np.sign(rec["N"]) == np.sign(rec["dreal"]))
            rec["sign_agree_ind"] = bool(rec["N_ind"] is not None and rec["dreal"] is not None and np.sign(rec["N_ind"]) == np.sign(rec["dreal"]))
            rec["real_ci_decided"] = ci_excl0(dvB["dreal"][:, j]); rec["both_ci_decided"] = rec["real_ci_decided"] and ci_excl0(dvB["N"][:, j])
            rows_b.append(rec)
        # ---------- (c2) error-correlation excess for usable pairs (full sample) ---------------------------------
        for i in range(K):
            for j2 in range(i + 1, K):
                if not (usable[i] and usable[j2]):
                    continue
                c_ij = COL[(1 << i) | (1 << j2)]
                exc1 = (1 - st1["p"][0, c_ij]) - (1 - st1["p"][0, SINGLE_COL[i]]) * (1 - st1["p"][0, SINGLE_COL[j2]])
                excB_ = (1 - stB["p"][:, c_ij]) - (1 - stB["p"][:, SINGLE_COL[i]]) * (1 - stB["p"][:, SINGLE_COL[j2]])
                lo, hi = ci(excB_)
                rows_corr.append({"ctx": name, "kind": kind, "theta": theta, "i": SHORT[NONREF[i]], "j": SHORT[NONREF[j2]], "excess": float(exc1), "excess_ci": [lo, hi],
                                  "positive": bool(exc1 > 0), "positive_ci": bool(lo is not None and lo > 0), "negative_ci": bool(hi is not None and hi < 0)})
        # ---------- (C3) Frechet interval, |S| = 1, m0 in pool and m usable (full sample = identity; fit bounds vs held-out r = OOS) --
        for j in range(NPAIR):
            if SIZE_S[j] != 1:
                continue
            m0, m = int(M0_OF[j]), int(M_IDX[j])
            if not (usable[m0] and usable[m]):
                continue
            ell, u, r = float(dv1["ell"][0, j]), float(dv1["u"][0, j]), float(dv1["r"][0, j])
            w = u - ell
            rows_fre.append({"ctx": name, "kind": kind, "theta": theta, "main4": bool(m0 in POOL_MAIN_IDX), "m0": SHORT[NONREF[m0]], "m": SHORT[NONREF[m]], "p0": ID(dv1["p_s"][0, j]), "p_m": ID(dv1["am"][0, j]),
                             "ell": ell, "u": u, "width": w, "r": r, "r_ci": ci(dvB["r"][:, j]), "r_ind": ID(dv1["r_ind"][0, j]),
                             "inside": bool(r >= ell - 1e-12 and r <= u + 1e-12), "position": (float((r - ell) / w) if w > 1e-12 else None),
                             "position_ind": (float((dv1["r_ind"][0, j] - ell) / w) if w > 1e-12 else None),
                             "inside_boot_frac": float(np.mean((dvB["r"][:, j] >= dvB["ell"][:, j] - 1e-12) & (dvB["r"][:, j] <= dvB["u"][:, j] + 1e-12)))})
            if usable_fit[m0] and usable_fit[m]:
                ell_f, u_f, r_f = float(dv_fit["ell"][0, j]), float(dv_fit["u"][0, j]), float(dv_fit["r"][0, j]); r_t = float(dv_te["r"][0, j])
                rows_fre_oos.append({"ctx": name, "kind": kind, "theta": theta, "m0": SHORT[NONREF[m0]], "m": SHORT[NONREF[m]], "ell_fit": ell_f, "u_fit": u_f, "r_fit": r_f,
                                     "r_heldout": r_t, "r_heldout_ci": ci(dv_teB["r"][:, j]), "inside_fit_bounds": bool(r_t >= ell_f - 1e-12 and r_t <= u_f + 1e-12),
                                     "ci_overlaps_fit_interval": bool((ci(dv_teB["r"][:, j])[1] or 0) >= ell_f - 1e-12 and (ci(dv_teB["r"][:, j])[0] or 1) <= u_f + 1e-12),
                                     "width_fit": u_f - ell_f})
        # ---------- (C3) equal-marginal triples: marginal-only selector's excess vs rem:frechet bound (full sample) ---
        def triples(dv, use, k=0):
            out = []
            for m0 in POOL_IDX:
                if not use[m0]:
                    continue
                cands = [m for m in range(K) if m != m0 and use[m]]
                for ia in range(len(cands)):
                    for ib in range(ia + 1, len(cands)):
                        ja, jb = J_OF[(COL[1 << m0], cands[ia])], J_OF[(COL[1 << m0], cands[ib])]
                        if abs(dv["am"][k, ja] - dv["am"][k, jb]) < EQ_TOL:
                            out.append((m0, cands[ia], cands[ib], ja, jb))
            return out
        def excess_arrays(dv, ja, jb):
            """Per-replicate excess of the minimax marginal-only selector and the rem:frechet bounds (vectorised over replicates)."""
            r1, r2 = dv["r"][:, ja], dv["r"][:, jb]; l1, u1, l2, u2 = dv["ell"][:, ja], dv["u"][:, ja], dv["ell"][:, jb], dv["u"][:, jb]
            b = dv["b"][:, ja]; p0 = dv["p_s"][:, ja]; pbar = 0.5 * (dv["am"][:, ja] + dv["am"][:, jb])
            x = np.maximum(0.0, u1 - l2); y = np.maximum(0.0, u2 - l1)
            with np.errstate(divide="ignore", invalid="ignore"):
                q = np.where(x + y > 0, x / np.where(x + y > 0, x + y, 1.0), 0.5)
                bound_exact_prob = np.where(x + y > 0, x * y / np.where(x + y > 0, x + y, 1.0), 0.0)     # xy/(x+y)
            informed = np.maximum(r1, r2)
            exc_prob = informed - (q * r1 + (1 - q) * r2)                                                  # excess / b (probability units)
            width_eq = np.minimum(pbar, 1 - p0) - np.maximum(0.0, pbar - p0)                              # equal-marginal range width at p = pbar
            return {"exc_prob": exc_prob, "exc_usd": b * exc_prob, "bound_exact_usd": b * bound_exact_prob, "bound_exact_prob": bound_exact_prob,
                    "bound_eq_usd": 0.5 * b * width_eq, "width_eq": width_eq, "width_max": np.maximum(u1 - l1, u2 - l2), "q": q, "b": b,
                    "r1": r1, "r2": r2, "gap_r": np.abs(r1 - r2)}
        tri = triples(dv1, usable)
        if tri:
            EB = {k_: [] for k_ in ["exc_prob", "width_eq", "exc_usd", "bound_exact_usd"]}
            for (m0, ma, mb, ja, jb) in tri:
                e1 = excess_arrays(dv1, ja, jb); eB = excess_arrays(dvB, ja, jb)
                for k_ in EB:
                    EB[k_].append(eB[k_])
                rows_exc.append({"ctx": name, "kind": kind, "theta": theta, "main4": bool(m0 in POOL_MAIN_IDX), "m0": SHORT[NONREF[m0]], "m1": SHORT[NONREF[ma]], "m2": SHORT[NONREF[mb]],
                                 "p0": ID(dv1["p_s"][0, ja]), "p1": ID(dv1["am"][0, ja]), "p2": ID(dv1["am"][0, jb]), "r1": ID(e1["r1"][0]), "r2": ID(e1["r2"][0]),
                                 "d1": ID(dv1["d"][0, ja]), "d2": ID(dv1["d"][0, jb]), "b": ID(e1["b"][0]), "q_marginal_selector": ID(e1["q"][0]),
                                 "excess_prob": ID(e1["exc_prob"][0]), "excess_prob_ci": ci(eB["exc_prob"]), "excess_usd": ID(e1["exc_usd"][0]), "excess_usd_ci": ci(eB["exc_usd"]),
                                 "bound_exact_usd": ID(e1["bound_exact_usd"][0]), "bound_exact_prob": ID(e1["bound_exact_prob"][0]),
                                 "bound_eq_usd": ID(e1["bound_eq_usd"][0]), "width_eq": ID(e1["width_eq"][0]), "width_max": ID(e1["width_max"][0]),
                                 "within_exact_bound": bool(e1["exc_usd"][0] <= e1["bound_exact_usd"][0] + 1e-12),
                                 "within_eq_bound": bool(e1["exc_usd"][0] <= e1["bound_eq_usd"][0] + 1e-12),
                                 "ratio_excess_over_exact_bound": (float(e1["exc_usd"][0] / e1["bound_exact_usd"][0]) if e1["bound_exact_usd"][0] > 1e-12 else None)})
            excB_store.append({"ctx": name, "kind": kind, "theta": theta} | {k_: np.stack(v, axis=1) for k_, v in EB.items()})   # B x n_triples
        # ---------- (d) OUT-OF-SAMPLE: fit-split primitives -> held-out realised change --------------------------
        am_fit_all = dv_fit["am"][0]; sgn_fit_all = np.sign(dv_fit["dreal"][0])
        for s in POOL_PORTFOLIOS:
            if any(not usable_fit[k] for k in MEMBERS[COL[s]]):
                continue
            js = np.array([j for j in range(NPAIR) if S_IDX[j] == COL[s] and usable_fit[M_IDX[j]] and nontriv_fit[j]])
            if len(js) == 0:
                continue
            # marginal-only threshold rule fitted on the fit split's own realised signs (all nontrivial cases of this context)
            selfit = nontriv_fit & np.isfinite(dv_fit["dreal"][0]) & np.isfinite(am_fit_all)
            tau = fit_threshold(am_fit_all[selfit], sgn_fit_all[selfit]) if selfit.any() else np.inf
            rules = sign_rules(dv_fit, js); rules["marginal_quality_fitted_tau"] = np.where(am_fit_all[js] >= tau, 1.0, -1.0)
            g_te = dv_te["dreal"][0, js]; g_teB = dv_teB["dreal"][:, js]; r_te = dv_te["r"][0, js]
            for k_, j in enumerate(js):
                rec = {"ctx": name, "kind": kind, "theta": theta, "S": sname(COL[s]), "size_S": int(SIZE[COL[s]]), "main4": bool(MAIN4_S[j]), "m": SHORT[NONREF[M_IDX[j]]],
                       "N_fit": ID(dv_fit["N"][0, j]), "N_ind_fit": ID(dv_fit["N_ind"][0, j]), "r_fit": ID(dv_fit["r"][0, j]), "P_E_m_fit": ID(dv_fit["am"][0, j]), "p_S_fit": ID(dv_fit["p_s"][0, j]),
                       "d_fit": ID(dv_fit["d"][0, j]), "dpred_fit": ID(dv_fit["dpred"][0, j]), "eta_S_heldout": ID(dv_te["eta_s"][0, j]),
                       "dreal_heldout": ID(g_te[k_]), "dreal_heldout_ci": ci(g_teB[:, k_]), "r_heldout": ID(r_te[k_]), "ci_decided": ci_excl0(g_teB[:, k_]), "tau_fit": ID(tau)}
                sg = np.sign(g_te[k_])
                for rname in RULES:
                    v = rules[rname][k_]
                    rec[f"agree_{rname}"] = bool(np.isfinite(v) and np.isfinite(sg) and v == sg)
                    rec[f"pred_{rname}"] = ID(v)
                oos_sign.append(rec)
            if len(js) < 3:
                continue
            a_pr = dv_fit["am"][0, js]; r_pr = dv_fit["r"][0, js]; N_pr = dv_fit["N"][0, js]; N_ind = dv_fit["N_ind"][0, js]; apd = a_pr / dv_fit["d"][0, js]
            eta_s_te = float(dv_te["eta_s"][0, js[0]]); rel_te = g_te / eta_s_te
            tops = {"quality_a": int(a_pr.argmax()), "coverage_r": int(r_pr.argmax()), "theorem_N": int(np.nanargmax(N_pr)), "theorem_N_indep": int(np.nanargmax(N_ind)),
                    "quality_per_dollar": int(apd.argmax()), "oracle_heldout": int(np.nanargmax(g_te))}
            rec = {"ctx": name, "kind": kind, "theta": theta, "S": sname(COL[s]), "size_S": int(SIZE[COL[s]]), "main4": bool(all(k in POOL_MAIN_IDX for k in MEMBERS[COL[s]])), "partners": [SHORT[NONREF[M_IDX[j]]] for j in js], "n_partners": int(len(js)),
                   "eta_S_heldout": eta_s_te, "fit_P_E_m": a_pr.tolist(), "fit_r": r_pr.tolist(), "fit_N": N_pr.tolist(), "fit_N_indep": N_ind.tolist(),
                   "heldout_gain": g_te.tolist(), "heldout_rel_gain": rel_te.tolist(), "heldout_r": r_te.tolist(),
                   "spearman_fit_r_vs_heldout_gain": spearman(r_pr, g_te), "spearman_fit_a_vs_heldout_gain": spearman(a_pr, g_te),
                   "spearman_fit_N_vs_heldout_gain": spearman(N_pr, g_te), "spearman_fit_Nindep_vs_heldout_gain": spearman(N_ind, g_te), "spearman_fit_apd_vs_heldout_gain": spearman(apd, g_te),
                   "spearman_fit_r_vs_heldout_r": spearman(r_pr, r_te), "spearman_fit_a_vs_heldout_r": spearman(a_pr, r_te),
                   # bootstrap (held-out tasks) of the two Spearman correlations against the realised gain, fit rankings fixed
                   "spearman_fit_r_vs_heldout_gain_boot": [spearman(r_pr, g_teB[kk]) for kk in range(N_BOOT)],
                   "spearman_fit_a_vs_heldout_gain_boot": [spearman(a_pr, g_teB[kk]) for kk in range(N_BOOT)]}
            for nm, k_ in tops.items():
                rec[f"top_{nm}"] = SHORT[NONREF[M_IDX[js[k_]]]]; rec[f"gain_top_{nm}"] = float(g_te[k_]); rec[f"relgain_top_{nm}"] = float(rel_te[k_]); rec[f"gain_top_{nm}_ci"] = ci(g_teB[:, k_])
            for x_, y_ in PAIRINGS:
                dif = g_teB[:, tops[x_]] - g_teB[:, tops[y_]]
                rec[f"diff_{x_}_minus_{y_}"] = float(g_te[tops[x_]] - g_te[tops[y_]]); rec[f"diff_{x_}_minus_{y_}_ci"] = ci(dif); rec[f"same_top_{x_}_{y_}"] = bool(tops[x_] == tops[y_])
            oos_rows.append(rec)
        # ---------- (C5) OOS selector excess: equal-marginal triples defined on the fit split, realised on held-out ------
        tri_fit = triples(dv_fit, usable_fit)
        if tri_fit:
            dB = []
            for (m0, ma, mb, ja, jb) in tri_fit:
                e_f = excess_arrays(dv_fit, ja, jb)
                q = float(e_f["q"][0]); inf_pick = 0 if dv_fit["r"][0, ja] >= dv_fit["r"][0, jb] else 1
                def realised(dv_, k_all):
                    r1, r2, b = dv_["r"][:, ja], dv_["r"][:, jb], dv_["b"][:, ja]
                    best = np.maximum(r1, r2)
                    x_marg = b * (best - (q * r1 + (1 - q) * r2)); x_inf = b * (best - (r1 if inf_pick == 0 else r2)); x_rand = b * (best - 0.5 * (r1 + r2))
                    return x_marg, x_inf, x_rand
                xm1, xi1, xr1 = realised(dv_te, 0); xmB, xiB, xrB = realised(dv_teB, None)
                dB.append(xmB - xiB)
                oos_exc.append({"ctx": name, "kind": kind, "theta": theta, "main4": bool(m0 in POOL_MAIN_IDX), "m0": SHORT[NONREF[m0]], "m1": SHORT[NONREF[ma]], "m2": SHORT[NONREF[mb]], "q_fit": q,
                                "informed_pick_fit": SHORT[NONREF[ma if inf_pick == 0 else mb]], "r1_fit": ID(dv_fit["r"][0, ja]), "r2_fit": ID(dv_fit["r"][0, jb]),
                                "r1_heldout": ID(dv_te["r"][0, ja]), "r2_heldout": ID(dv_te["r"][0, jb]), "b_heldout": ID(dv_te["b"][0, ja]),
                                "excess_marginal_usd": ID(xm1[0]), "excess_informed_usd": ID(xi1[0]), "excess_random_usd": ID(xr1[0]),
                                "diff_marginal_minus_informed_usd": ID(xm1[0] - xi1[0]), "diff_ci": ci(xmB - xiB),
                                "bound_exact_usd_fit": ID(e_f["bound_exact_usd"][0]), "informed_pick_is_heldout_best": bool((dv_te["r"][0, ja] >= dv_te["r"][0, jb]) == (inf_pick == 0))})
            oosexcB_store.append({"ctx": name, "kind": kind, "theta": theta, "diff": np.stack(dB, axis=1)})
        print(f"[{key}] n={n} (fit {n_fit}/held-out {n_te}) usable(full)={int(usable.sum())} usable(fit)={int(usable_fit.sum())} "
              f"nontrivial(full)={int(nontriv.sum())} triples={len(tri)} eta_gpt4={st1['eta'][0, COL[0]]:.1f}  ({time.time()-T0:.0f}s)")
R["identity"] = ident

# ==============================================================================================
# 4. Summaries
# ==============================================================================================
SCOPES_STORE = {"all_non_overall": lambda r: r["kind"] != "overall", "families": lambda r: r["kind"] == "family",
                "mmlu_subsets": lambda r: r["kind"] == "mmlu_subset", "overall": lambda r: r["kind"] == "overall"}
SCOPES = SCOPES_STORE | {"main4_non_overall": lambda r: r["kind"] != "overall" and r.get("main4", False)}
def med(x):
    x = np.asarray([v for v in x if v is not None], float); x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else None
def frac(x):
    x = np.asarray(list(x), float)
    return float(x.mean()) if x.size else None
def verdict_hyp(fr_sig, med_rel):
    # pre-declared (X2): PASS if the CI of the drift covers 0 in >= 90% of cases; FAIL if it excludes 0 in > 50%; else INCONCLUSIVE
    if fr_sig is None:
        return "NOT-INSTANTIABLE"
    if fr_sig <= 0.10:
        return "PASS"
    if fr_sig > 0.50:
        return "FAIL" + (" (small drift: |median rel| < 5%)" if abs(med_rel or 0) < 0.05 else " (large drift)")
    return "INCONCLUSIVE"

def summarize_b(rows):
    out = {}
    for theta in THETAS:
        for scope, f in SCOPES.items():
            rs = [r for r in rows if r["theta"] == theta and f(r)]
            if not rs:
                continue
            dvm = np.array([r["vm_S2"] - r["vm_S"] for r in rs if r["vm_S2"] is not None and r["vm_S"] is not None])
            rel_vm = np.array([(r["vm_S2"] - r["vm_S"]) / r["vm_S"] for r in rs if r["vm_S"] not in (None, 0) and r["vm_S2"] is not None])
            db = np.array([r["b_S2"] - r["b_S"] for r in rs if r["b_S2"] is not None and r["b_S"] is not None]); rel_b = np.array([(r["b_S2"] - r["b_S"]) / r["b_S"] for r in rs if r["b_S"] not in (None, 0) and r["b_S2"] is not None])
            dvh = np.array([r["vh_S2"] - r["vh_S"] for r in rs if r["vh_S2"] is not None and r["vh_S"] is not None])
            sig = lambda kk: np.array([r[kk][0] is not None and (r[kk][0] > 0 or r[kk][1] < 0) for r in rs])
            agree = np.array([r["sign_agree"] for r in rs]); agree_ind = np.array([r["sign_agree_ind"] for r in rs])
            rdec = np.array([r["real_ci_decided"] for r in rs]); bdec = np.array([r["both_ci_decided"] for r in rs])
            neg = np.array([r["dreal"] < 0 for r in rs])
            relerr = np.array([abs(r["dpred"] - r["dreal"]) / abs(r["dreal"]) for r in rs if r["dreal"] not in (None, 0) and r["dpred"] is not None])
            def relerr_of(kk):
                v = [abs(r[kk] - r["dreal"]) / abs(r["dreal"]) for r in rs if r["dreal"] not in (None, 0) and r[kk] is not None]
                return float(np.median(v)) if v else None
            out[f"theta={theta}|{scope}"] = {
                "n": len(rs), "n_contexts": len({r["ctx"] for r in rs}), "n_realised_ci_decided": int(rdec.sum()), "n_both_ci_decided": int(bdec.sum()),
                "sign_agreement_all": float(agree.mean()), "sign_agreement_realised_ci_decided": (float(agree[rdec].mean()) if rdec.any() else None),
                "sign_agreement_both_ci_decided": (float(agree[bdec].mean()) if bdec.any() else None),
                "baseline_N_indep_all": float(agree_ind.mean()), "baseline_N_indep_realised_ci_decided": (float(agree_ind[rdec].mean()) if rdec.any() else None),
                "baseline_always_negative_all": float(neg.mean()), "baseline_always_negative_realised_ci_decided": (float(neg[rdec].mean()) if rdec.any() else None),
                "disagree_pattern": {"pred_pos_real_neg": int(sum(1 for r in rs if r["N"] is not None and r["N"] > 0 and r["dreal"] < 0)), "pred_neg_real_pos": int(sum(1 for r in rs if r["N"] is not None and r["N"] < 0 and r["dreal"] > 0))},
                "vm_drift": {"n": int(dvm.size), "median": med(dvm), "median_rel": med(rel_vm), "frac_negative": frac(dvm < 0), "frac_ci_excl0": frac(sig("dvm_ci")),
                             "q10_q90_rel": ([float(np.percentile(rel_vm, 10)), float(np.percentile(rel_vm, 90))] if rel_vm.size else None)},
                "b_drift": {"median": med(db), "median_rel": med(rel_b), "frac_positive": frac(db > 0), "frac_ci_excl0": frac(sig("db_ci")),
                            "q10_q90_rel": ([float(np.percentile(rel_b, 10)), float(np.percentile(rel_b, 90))] if rel_b.size else None)},
                "vh_drift": {"median": med(dvh), "max_abs": (float(np.max(np.abs(dvh))) if dvh.size else None), "frac_ci_excl0": frac(sig("dvh_ci"))},
                "median_rel_err_dpred": med(relerr), "frac_rel_err_below_25pct": frac(relerr < 0.25),
                "median_rel_err_substitute_vm2": relerr_of("dpred_vm2"), "median_rel_err_substitute_b2": relerr_of("dpred_b2"), "median_rel_err_substitute_vh2": relerr_of("dpred_vh2")}
    return out
R["in_sample"]["summary_b4_and_drift"] = summarize_b(rows_b)
R["in_sample"]["cases_b"] = rows_b

def summarize_corr(rows):
    out = {}
    for theta in THETAS:
        for scope, f in SCOPES.items():
            rs = [r for r in rows if r["theta"] == theta and f(r)]
            if rs:
                out[f"theta={theta}|{scope}"] = {"n_pairs": len(rs), "frac_positive": frac(r["positive"] for r in rs), "frac_positive_ci": frac(r["positive_ci"] for r in rs),
                                                 "frac_negative_ci": frac(r["negative_ci"] for r in rs), "median_excess": med(r["excess"] for r in rs),
                                                 "negative_ci_cases": [f"{r['ctx']}: {r['i']} x {r['j']} ({r['excess']:+.3f})" for r in rs if r["negative_ci"]][:20]}
    return out
R["in_sample"]["error_correlation_summary"] = summarize_corr(rows_corr)
R["in_sample"]["error_correlation_pairs"] = rows_corr

def summarize_frechet(rows, rows_oos):
    out = {}
    for theta in THETAS:
        for scope, f in SCOPES.items():
            rs = [r for r in rows if r["theta"] == theta and f(r)]
            if not rs:
                continue
            pos = np.array([r["position"] for r in rs if r["position"] is not None]); posi = np.array([r["position_ind"] for r in rs if r["position_ind"] is not None])
            ro = [r for r in rows_oos if r["theta"] == theta and f(r)]
            out[f"theta={theta}|{scope}"] = {
                "n_pairs": len(rs), "n_contexts": len({r["ctx"] for r in rs}), "frac_inside_same_sample": frac(r["inside"] for r in rs), "n_outside_same_sample": int(sum(not r["inside"] for r in rs)),
                "mean_boot_inside_frac": frac(r["inside_boot_frac"] for r in rs), "n_degenerate_width0": int(sum(r["position"] is None for r in rs)),
                "position_median": med(pos), "position_q10_q90": ([float(np.percentile(pos, 10)), float(np.percentile(pos, 90))] if pos.size else None),
                "position_frac_below_0.5": frac(pos < 0.5), "position_frac_at_lower_bound_1pct": frac(pos < 0.01), "position_frac_at_upper_bound_1pct": frac(pos > 0.99),
                "position_indep_proxy_median": med(posi), "frac_r_below_indep_proxy": frac(r["r"] < r["r_ind"] for r in rs),
                "width_median": med(r["width"] for r in rs), "width_q10_q90": [float(np.percentile([r["width"] for r in rs], 10)), float(np.percentile([r["width"] for r in rs], 90))],
                "oos_n_pairs": len(ro), "oos_frac_heldout_r_inside_fit_bounds": frac(r["inside_fit_bounds"] for r in ro), "oos_frac_heldout_ci_overlaps_fit_interval": frac(r["ci_overlaps_fit_interval"] for r in ro)}
    return out
R["frechet"]["interval_summary"] = summarize_frechet(rows_fre, rows_fre_oos)
R["frechet"]["interval_pairs"] = rows_fre
R["frechet"]["interval_pairs_oos"] = rows_fre_oos

def summarize_excess(rows, store):
    out = {}
    for theta in THETAS:
        for scope, f in SCOPES_STORE.items():
            rs = [r for r in rows if r["theta"] == theta and f(r)]
            if not rs:
                continue
            st_ = [s for s in store if s["theta"] == theta and f(s)]
            EP = np.concatenate([s["exc_prob"] for s in st_], axis=1); WD = np.concatenate([s["width_eq"] for s in st_], axis=1)
            EU = np.concatenate([s["exc_usd"] for s in st_], axis=1); BU = np.concatenate([s["bound_exact_usd"] for s in st_], axis=1)
            sp_boot = np.array([spearman(EP[k], WD[k]) for k in range(N_BOOT)]); sp_boot_usd = np.array([spearman(EU[k], BU[k]) for k in range(N_BOOT)])
            ex = np.array([r["excess_prob"] for r in rs]); wd = np.array([r["width_eq"] for r in rs])
            # within-context Spearman (contexts with >= 5 triples)
            per_ctx = {}
            for c_ in sorted({r["ctx"] for r in rs}):
                rc = [r for r in rs if r["ctx"] == c_]
                if len(rc) >= 5:
                    per_ctx[c_] = spearman([r["excess_prob"] for r in rc], [r["width_eq"] for r in rc])
            pcv = np.array([v for v in per_ctx.values() if np.isfinite(v)])
            ratio = np.array([r["ratio_excess_over_exact_bound"] for r in rs if r["ratio_excess_over_exact_bound"] is not None])
            out[f"theta={theta}|{scope}"] = {
                "n_triples": len(rs), "n_contexts": len({r["ctx"] for r in rs}), "frac_within_exact_bound": frac(r["within_exact_bound"] for r in rs),
                "n_violations_exact_bound": int(sum(not r["within_exact_bound"] for r in rs)), "frac_within_equal_marginal_bound_pbar": frac(r["within_eq_bound"] for r in rs),
                "n_violations_equal_marginal_bound_pbar": int(sum(not r["within_eq_bound"] for r in rs)),
                "max_violation_equal_marginal_bound_usd": float(max([r["excess_usd"] - r["bound_eq_usd"] for r in rs] + [0.0])),
                "median_ratio_excess_over_exact_bound": med(ratio), "q10_q90_ratio": ([float(np.percentile(ratio, 10)), float(np.percentile(ratio, 90))] if ratio.size else None),
                "frac_excess_zero": frac(ex <= 1e-12), "median_excess_prob": med(ex), "median_width_eq": med(wd), "median_excess_usd": med(r["excess_usd"] for r in rs), "median_bound_usd": med(r["bound_exact_usd"] for r in rs),
                "spearman_excess_prob_vs_width_pooled": spearman(ex, wd), "spearman_excess_prob_vs_width_pooled_ci": ci(sp_boot),
                "spearman_excess_usd_vs_bound_usd_pooled": spearman([r["excess_usd"] for r in rs], [r["bound_exact_usd"] for r in rs]), "spearman_excess_usd_vs_bound_usd_pooled_ci": ci(sp_boot_usd),
                "pearson_excess_prob_vs_width_pooled": (float(np.corrcoef(ex, wd)[0, 1]) if len(rs) > 2 and np.std(ex) > 0 and np.std(wd) > 0 else None),
                "within_context_spearman": {"n_contexts_ge5": int(pcv.size), "median": (float(np.median(pcv)) if pcv.size else None), "frac_positive": (float((pcv > 0).mean()) if pcv.size else None), "values": per_ctx}}
    return out
R["frechet"]["excess_summary"] = summarize_excess(rows_exc, excB_store)
R["frechet"]["excess_triples"] = rows_exc

def summarize_oos_excess(rows, store):
    out = {}
    for theta in THETAS:
        for scope, f in SCOPES_STORE.items():
            rs = [r for r in rows if r["theta"] == theta and f(r)]
            if not rs:
                continue
            st_ = [s for s in store if s["theta"] == theta and f(s)]
            D = np.concatenate([s["diff"] for s in st_], axis=1)                 # B x n_triples: marginal minus informed (usd)
            xm = np.array([r["excess_marginal_usd"] for r in rs]); xi = np.array([r["excess_informed_usd"] for r in rs]); xr = np.array([r["excess_random_usd"] for r in rs])
            nb, nw = int((xm - xi > 1e-12).sum()), int((xm - xi < -1e-12).sum())
            out[f"theta={theta}|{scope}"] = {
                "n_triples": len(rs), "n_contexts": len({r["ctx"] for r in rs}), "mean_excess_marginal_usd": float(xm.mean()), "mean_excess_informed_usd": float(xi.mean()), "mean_excess_random_usd": float(xr.mean()),
                "mean_diff_marginal_minus_informed_usd": float((xm - xi).mean()), "mean_diff_ci": ci(D.mean(1)), "median_diff_marginal_minus_informed_usd": float(np.median(xm - xi)),
                "frac_informed_pick_is_heldout_best": frac(r["informed_pick_is_heldout_best"] for r in rs),
                "sign_test": {"informed_better": nb, "marginal_better": nw, "ties": int(len(rs) - nb - nw), "p_two_sided": (float(stats.binomtest(nb, nb + nw, 0.5).pvalue) if nb + nw > 0 else None)},
                "ratio_informed_over_marginal_mean_excess": (float(xi.mean() / xm.mean()) if xm.mean() > 0 else None)}
    return out
R["oos"]["selector_excess_summary"] = summarize_oos_excess(oos_exc, oosexcB_store)
R["oos"]["selector_excess_triples"] = oos_exc

def summarize_oos(rows, signs):
    out = {}
    for theta in THETAS:
        for scope, f in SCOPES.items():
            rs = [r for r in rows if r["theta"] == theta and f(r)]; ss = [r for r in signs if r["theta"] == theta and f(r)]
            if not ss:
                continue
            dec = [r for r in ss if r["ci_decided"]]
            blk = {"n_S": len(rs), "n_sign_cases": len(ss), "n_sign_ci_decided": len(dec), "n_contexts": len({r["ctx"] for r in ss}),
                   "frac_realised_positive_all": frac(r["dreal_heldout"] > 0 for r in ss), "frac_realised_positive_ci_decided": frac(r["dreal_heldout"] > 0 for r in dec),
                   "sign_agreement_all": {rn: frac(r[f"agree_{rn}"] for r in ss) for rn in RULES},
                   "sign_agreement_ci_decided": {rn: frac(r[f"agree_{rn}"] for r in dec) for rn in RULES},
                   "n_disagree_ci_decided": {rn: int(sum(not r[f"agree_{rn}"] for r in dec)) for rn in RULES},
                   "disagree_pattern_theorem_ci_decided": {"pred_pos_real_neg": int(sum(1 for r in dec if r["pred_theorem_N"] == 1 and r["dreal_heldout"] < 0)), "pred_neg_real_pos": int(sum(1 for r in dec if r["pred_theorem_N"] == -1 and r["dreal_heldout"] > 0))}}
            # bootstrap CI of the CI-decided agreement rate? not meaningful (selection); report a binomial CI instead
            if dec:
                k_ = int(sum(r["agree_theorem_N"] for r in dec)); n_ = len(dec)
                lo_, hi_ = stats.binomtest(k_, n_).proportion_ci(0.95, method="wilson")
                blk["theorem_N_ci_decided_wilson95"] = [float(lo_), float(hi_)]
            if rs:
                for nm in RANKINGS:
                    blk[f"mean_relgain_top_{nm}"] = frac(r[f"relgain_top_{nm}"] for r in rs); blk[f"median_relgain_top_{nm}"] = med(r[f"relgain_top_{nm}"] for r in rs)
                    blk[f"frac_top_{nm}_gain_positive"] = frac(r[f"gain_top_{nm}"] > 0 for r in rs); blk[f"frac_top_{nm}_is_oracle"] = frac(r[f"top_{nm}"] == r["top_oracle_heldout"] for r in rs)
                for x_, y_ in PAIRINGS:
                    dif = [r for r in rs if not r[f"same_top_{x_}_{y_}"]]
                    blk[f"{x_}_vs_{y_}"] = {"n_differing": len(dif), "frac_same_top": frac(r[f"same_top_{x_}_{y_}"] for r in rs),
                                            "n_x_wins": int(sum(r[f"diff_{x_}_minus_{y_}"] > 0 for r in dif)), "n_y_wins": int(sum(r[f"diff_{x_}_minus_{y_}"] < 0 for r in dif)),
                                            "n_x_wins_ci": int(sum(r[f"diff_{x_}_minus_{y_}_ci"][0] is not None and r[f"diff_{x_}_minus_{y_}_ci"][0] > 0 for r in dif)),
                                            "n_y_wins_ci": int(sum(r[f"diff_{x_}_minus_{y_}_ci"][1] is not None and r[f"diff_{x_}_minus_{y_}_ci"][1] < 0 for r in dif)),
                                            "mean_diff_relgain_x_minus_y_on_differing": (frac(r[f"relgain_top_{x_}"] - r[f"relgain_top_{y_}"] for r in dif) if dif else None)}
                # d3 (C2): Spearman of fit-split rankings against the held-out realised gain, per portfolio
                sr = np.array([r["spearman_fit_r_vs_heldout_gain"] for r in rs]); sa = np.array([r["spearman_fit_a_vs_heldout_gain"] for r in rs])
                ok = np.isfinite(sr) & np.isfinite(sa); dsp = sr[ok] - sa[ok]
                nb, nw = int((dsp > 1e-12).sum()), int((dsp < -1e-12).sum())
                SRB = np.array([r["spearman_fit_r_vs_heldout_gain_boot"] for r in rs], float); SAB = np.array([r["spearman_fit_a_vs_heldout_gain_boot"] for r in rs], float)
                med_diff_boot = np.nanmedian(SRB - SAB, axis=0)
                blk["d3_gain_target"] = {"median_spearman_fit_r": med(sr), "median_spearman_fit_a": med(sa), "mean_spearman_fit_r": (float(np.nanmean(sr)) if ok.any() else None), "mean_spearman_fit_a": (float(np.nanmean(sa)) if ok.any() else None),
                                         "median_spearman_fit_N": med(r["spearman_fit_N_vs_heldout_gain"] for r in rs), "median_spearman_fit_Nindep": med(r["spearman_fit_Nindep_vs_heldout_gain"] for r in rs),
                                         "median_spearman_fit_apd": med(r["spearman_fit_apd_vs_heldout_gain"] for r in rs),
                                         "median_diff_r_minus_a": (float(np.median(dsp)) if dsp.size else None), "median_diff_r_minus_a_boot_ci": ci(med_diff_boot),
                                         "mean_diff_r_minus_a_boot_ci": ci(np.nanmean(SRB - SAB, axis=0)),
                                         "sign_test": {"r_better": nb, "a_better": nw, "ties": int(dsp.size - nb - nw), "p_two_sided": (float(stats.binomtest(nb, nb + nw, 0.5).pvalue) if nb + nw > 0 else None)}}
                sr2 = np.array([r["spearman_fit_r_vs_heldout_r"] for r in rs]); sa2 = np.array([r["spearman_fit_a_vs_heldout_r"] for r in rs]); ok2 = np.isfinite(sr2) & np.isfinite(sa2); d2 = sr2[ok2] - sa2[ok2]
                nb2, nw2 = int((d2 > 1e-12).sum()), int((d2 < -1e-12).sum())
                blk["d3_r_target"] = {"median_spearman_fit_r": med(sr2), "median_spearman_fit_a": med(sa2),
                                      "sign_test": {"r_better": nb2, "a_better": nw2, "ties": int(d2.size - nb2 - nw2), "p_two_sided": (float(stats.binomtest(nb2, nb2 + nw2, 0.5).pvalue) if nb2 + nw2 > 0 else None)}}
            out[f"theta={theta}|{scope}"] = blk
    return out
R["oos"]["summary_d"] = summarize_oos(oos_rows, oos_sign)
for r in oos_rows:   # drop the per-replicate arrays from the stored cases (they live in the summary CIs)
    r.pop("spearman_fit_r_vs_heldout_gain_boot", None); r.pop("spearman_fit_a_vs_heldout_gain_boot", None)
R["oos"]["cases_d"] = oos_rows
R["oos"]["sign_cases"] = oos_sign
R["oos"]["sign_disagreements_ci_decided_theorem"] = [r for r in oos_sign if r["ci_decided"] and not r["agree_theorem_N"]]

# per-context OOS table (families, MMLU subsets pooled, overall) for the README and the figure
def per_context_oos(theta):
    tab = {}
    groups = [(n_, lambda r, n_=n_: r["ctx"] == n_) for n_, k_ in CONTEXTS if k_ in ("overall", "family")] + [("MMLU subsets (pooled)", lambda r: r["kind"] == "mmlu_subset")]
    for gname, f in groups:
        ss = [r for r in oos_sign if r["theta"] == theta and f(r)]
        if not ss:
            continue
        dec = [r for r in ss if r["ci_decided"]]; rs = [r for r in oos_rows if r["theta"] == theta and f(r)]
        tab[gname] = {"n_sign": len(ss), "n_ci_decided": len(dec), "agree_all": {rn: frac(r[f"agree_{rn}"] for r in ss) for rn in RULES},
                      "agree_ci_decided": {rn: frac(r[f"agree_{rn}"] for r in dec) for rn in RULES}, "n_S": len(rs),
                      "median_spearman_fit_r_vs_gain": med(r["spearman_fit_r_vs_heldout_gain"] for r in rs), "median_spearman_fit_a_vs_gain": med(r["spearman_fit_a_vs_heldout_gain"] for r in rs),
                      "median_relgain_top_theorem_N": med(r["relgain_top_theorem_N"] for r in rs), "median_relgain_top_quality_a": med(r["relgain_top_quality_a"] for r in rs),
                      "median_relgain_top_oracle": med(r["relgain_top_oracle_heldout"] for r in rs), "frac_top_theorem_is_oracle": frac(r["top_theorem_N"] == r["top_oracle_heldout"] for r in rs)}
    return tab
R["oos"]["per_context"] = {f"theta={th}": per_context_oos(th) for th in THETAS}

# ==============================================================================================
# 5. Claims and verdicts (criteria pre-declared in the task / README)
# ==============================================================================================
C = {}
C["A1_residual_identity"] = {"id": "A1", "kind": "IDENTITY", "statement": "r_m(S) = p(S+m) - p(S) (eq:residual): direct count vs difference of coverages", "result": f"max |diff| = {ident['max_abs_r_identity']:.2e} over {ident['n_cases_checked']} (context, theta, S, m) cases", "verdict": "PASS" if ident["max_abs_r_identity"] < 1e-9 else "FAIL"}
C["A2_homogeneous_identity"] = {"id": "A2", "kind": "IDENTITY", "statement": "eq:homogeneous with the portfolio's own (p, v_h, v_m, a, b) reproduces the ratio-of-sums eta(S)", "result": f"max |diff| = {ident['max_abs_eta_homogeneous']:.2e}", "verdict": "PASS" if ident["max_abs_eta_homogeneous"] < 1e-9 else "FAIL"}
C["A3_marginal_identity"] = {"id": "A3", "kind": "IDENTITY", "statement": "eq:app-marginal-identity with common (v_h, v_m, b) := S's values equals N_S/(E[D(S)]+d-rb); substituting S+m's three values recovers the realised change", "result": f"max |diff| = {ident['max_abs_marginal_identity']:.2e} / {ident['max_abs_all2_equals_real']:.2e}", "verdict": "PASS" if max(ident["max_abs_marginal_identity"], ident["max_abs_all2_equals_real"]) < 1e-9 else "FAIL"}

sd = R["oos"]["summary_d"]
for theta in THETAS:
    d_ = sd.get(f"theta={theta}|all_non_overall")
    if d_ is None:
        continue
    ag = d_["sign_agreement_ci_decided"]; agA = d_["sign_agreement_all"]
    t_ = ag["theorem_N"] or 0.0
    base_ind = ag["N_indep"] or 0.0; base_tau = ag["marginal_quality_fitted_tau"] or 0.0; base_gt = ag["marginal_quality_gt_pS"] or 0.0
    base_neg = max(ag["always_negative"] or 0.0, ag["always_positive"] or 0.0)
    ok85 = t_ >= 0.85; above = t_ > base_ind and t_ > base_tau and t_ > base_gt
    verdict = "PASS" if (ok85 and above) else ("FAIL" if (not ok85 or t_ < min(base_ind, base_tau, base_gt)) else "INCONCLUSIVE")
    if d_["n_sign_ci_decided"] < 30:
        verdict = "INCONCLUSIVE (too few CI-decided cases)"
    scope_note = "" if theta == THETA_MAIN else " (theta = 1 differs from theta = 0.5 only in the non-binary contexts grade-school-math, mt-bench, rag-other, overall)"
    C[f"C1_oos_sign_theta{theta}"] = {"id": "C1", "kind": "OUT-OF-SAMPLE", "statement": f"eq:marginal sign rule fitted on the 70% split predicts the sign of the realised held-out change eta(S+m)-eta(S) in >= 85% of CI-decided (S,m) cases and beats the independence proxy (1-p(S))P(E_m) and the marginal-P(E_m)-only rules (theta={theta}; 8 families + {len(MMLU_SUBSETS)} MMLU subsets){scope_note}",
        "result": f"{d_['n_sign_cases']} (S,m) cases in {d_['n_contexts']} contexts, {d_['n_sign_ci_decided']} CI-decided: eq:marginal N_S {t_*100:.1f}% (Wilson 95% {d_.get('theorem_N_ci_decided_wilson95', [0,0])[0]*100:.1f}-{d_.get('theorem_N_ci_decided_wilson95', [0,0])[1]*100:.1f}%; {d_['n_disagree_ci_decided']['theorem_N']} disagreements: pred+/real- {d_['disagree_pattern_theorem_ci_decided']['pred_pos_real_neg']}, pred-/real+ {d_['disagree_pattern_theorem_ci_decided']['pred_neg_real_pos']}); independence proxy {base_ind*100:.1f}%; P(E_m) >= tau (tau fitted on the fit split) {base_tau*100:.1f}%; P(E_m) > p(S) {base_gt*100:.1f}%; Frechet-upper-bound rule {(ag['frechet_upper'] or 0)*100:.1f}%; Frechet-lower-bound rule {(ag['frechet_lower'] or 0)*100:.1f}%; value-preserving r b - d {(ag['value_preserving_rb_minus_d'] or 0)*100:.1f}%; constant sign (best of always +/-) {base_neg*100:.1f}%; on all cases: N_S {(agA['theorem_N'] or 0)*100:.1f}%, indep {(agA['N_indep'] or 0)*100:.1f}%, tau {(agA['marginal_quality_fitted_tau'] or 0)*100:.1f}%",
        "verdict": verdict}
    d3 = d_.get("d3_gain_target")
    if d3:
        st3 = d3["sign_test"]; lo_, hi_ = d3["median_diff_r_minus_a_boot_ci"]
        better = (d3["median_spearman_fit_r"] or -9) > (d3["median_spearman_fit_a"] or -9)
        sig_ = (st3["p_two_sided"] is not None and st3["p_two_sided"] < 0.05)
        if better and sig_ and st3["r_better"] > st3["a_better"]:
            v3 = "PASS"
        elif (not better) and sig_ and st3["a_better"] > st3["r_better"]:
            v3 = "FAIL"
        else:
            v3 = "INCONCLUSIVE"
        C[f"C2_oos_spearman_theta{theta}"] = {"id": "C2", "kind": "OUT-OF-SAMPLE", "statement": f"ranking candidates by fit-split residual coverage r_m(S) correlates better (Spearman) with the realised held-out change eta(S+m)-eta(S) than ranking by fit-split marginal P(E_m) (theta={theta}; portfolios with >= 3 non-trivial partners)",
            "result": f"{d_['n_S']} portfolios: median Spearman(fit r_m, held-out gain) {d3['median_spearman_fit_r']:.3f} vs Spearman(fit P(E_m), held-out gain) {d3['median_spearman_fit_a']:.3f} (means {d3['mean_spearman_fit_r']:.3f} vs {d3['mean_spearman_fit_a']:.3f}); r_m better in {st3['r_better']}, P(E_m) better in {st3['a_better']}, ties {st3['ties']} (two-sided sign test p = {st3['p_two_sided']:.2g}); median paired difference {d3['median_diff_r_minus_a']:+.3f} (held-out bootstrap CI [{lo_:+.3f}, {hi_:+.3f}]); for reference the cost-aware rankings: N_S {d3['median_spearman_fit_N']:.3f}, N_indep {d3['median_spearman_fit_Nindep']:.3f}, P(E_m)/d_m {d3['median_spearman_fit_apd']:.3f}; X2-style target held-out r_m: fit r_m {d_['d3_r_target']['median_spearman_fit_r']:.3f} vs fit P(E_m) {d_['d3_r_target']['median_spearman_fit_a']:.3f} (r better {d_['d3_r_target']['sign_test']['r_better']}, a better {d_['d3_r_target']['sign_test']['a_better']}, p = {(d_['d3_r_target']['sign_test']['p_two_sided'] if d_['d3_r_target']['sign_test']['p_two_sided'] is not None else float('nan')):.2g})",
            "verdict": v3}
        q = d_["theorem_N_vs_quality_a"]; q2 = d_["theorem_N_vs_theorem_N_indep"]; q3 = d_["coverage_r_vs_quality_a"]
        def pv(qq):
            if qq["n_differing"] == 0:
                return "INCONCLUSIVE (rankings never differ)"
            p = stats.binomtest(qq["n_x_wins"], qq["n_x_wins"] + qq["n_y_wins"], 0.5).pvalue if (qq["n_x_wins"] + qq["n_y_wins"]) > 0 else 1.0
            return "PASS" if (qq["n_x_wins"] > qq["n_y_wins"] and p < 0.05) else ("FAIL" if (qq["n_y_wins"] > qq["n_x_wins"] and p < 0.05) else "INCONCLUSIVE")
        C[f"C2b_oos_top_partner_theta{theta}"] = {"id": "C2b", "kind": "OUT-OF-SAMPLE", "statement": f"X2's d2: the partner with the largest fit-split N_S (eq:marginal) yields a larger realised held-out gain than the partner with the largest fit-split P(E_m); and than N_S with the independence proxy (theta={theta})",
            "result": f"theorem_N vs quality_a: same top in {q['frac_same_top']*100:.0f}%, differ in {q['n_differing']}, N_S wins {q['n_x_wins']} (CI {q['n_x_wins_ci']}), P(E_m) wins {q['n_y_wins']} (CI {q['n_y_wins_ci']}), mean rel-gain difference on differing {(q['mean_diff_relgain_x_minus_y_on_differing'] or 0)*100:+.1f} pts; theorem_N vs N_indep: differ in {q2['n_differing']}, N_S wins {q2['n_x_wins']} (CI {q2['n_x_wins_ci']}), N_indep wins {q2['n_y_wins']} (CI {q2['n_y_wins_ci']}); r_m vs P(E_m) (both price-blind): differ in {q3['n_differing']}, r_m wins {q3['n_x_wins']} (CI {q3['n_x_wins_ci']}), P(E_m) wins {q3['n_y_wins']} (CI {q3['n_y_wins_ci']}); top N_S = held-out oracle in {d_['frac_top_theorem_N_is_oracle']*100:.0f}% of portfolios, top P(E_m) in {d_['frac_top_quality_a_is_oracle']*100:.0f}%; median realised rel. gain of the top partner: P(E_m) {d_['median_relgain_top_quality_a']*100:+.1f}%, r_m {d_['median_relgain_top_coverage_r']*100:+.1f}%, N_indep {d_['median_relgain_top_theorem_N_indep']*100:+.1f}%, N_S {d_['median_relgain_top_theorem_N']*100:+.1f}%, oracle {d_['median_relgain_top_oracle_heldout']*100:+.1f}%",
            "verdict": f"{pv(q)} (N_S vs P(E_m)); {pv(q2)} (N_S vs N_indep); {pv(q3)} (r_m vs P(E_m))"}

fs = R["frechet"]["interval_summary"]; es = R["frechet"]["excess_summary"]
for theta in THETAS:
    f_ = fs.get(f"theta={theta}|all_non_overall"); e_ = es.get(f"theta={theta}|all_non_overall")
    if f_ is None:
        continue
    C[f"C3a_frechet_inclusion_theta{theta}"] = {"id": "C3a", "kind": "IDENTITY", "statement": f"realised r_m({{m0}}) lies in the Frechet interval [max{{0,p_m-p_0}}, min{{p_m,1-p_0}}] (rem:frechet) in 100% of (m0, m) pairs, same sample (theta={theta})",
        "result": f"{f_['n_pairs']} pairs in {f_['n_contexts']} contexts: inside {f_['frac_inside_same_sample']*100:.1f}% ({f_['n_outside_same_sample']} outside); bootstrap replicates inside {f_['mean_boot_inside_frac']*100:.1f}%; position (r-l)/(u-l): median {f_['position_median']:.2f}, 10-90% [{f_['position_q10_q90'][0]:.2f}, {f_['position_q10_q90'][1]:.2f}], within 1% of the lower bound in {f_['position_frac_at_lower_bound_1pct']*100:.0f}%, of the upper bound in {f_['position_frac_at_upper_bound_1pct']*100:.0f}%; independence proxy position median {f_['position_indep_proxy_median']:.2f}; realised r_m below the independence proxy in {f_['frac_r_below_indep_proxy']*100:.0f}% (positively correlated errors); median width {f_['width_median']:.3f}; OUT-OF-SAMPLE diagnostic: held-out r_m inside the FIT-split interval in {f_['oos_frac_heldout_r_inside_fit_bounds']*100:.1f}% of {f_['oos_n_pairs']} pairs (held-out CI overlaps it in {f_['oos_frac_heldout_ci_overlaps_fit_interval']*100:.1f}%)",
        "verdict": "PASS" if f_["frac_inside_same_sample"] >= 1.0 else "FAIL"}
    if e_ is not None:
        lo_, hi_ = e_["spearman_excess_prob_vs_width_pooled_ci"]
        within = e_["frac_within_exact_bound"] >= 1.0
        sp_pos = (lo_ is not None and lo_ > 0)
        v = "PASS" if (within and sp_pos) else ("FAIL" if (not within or (hi_ is not None and hi_ < 0)) else "INCONCLUSIVE")
        C[f"C3b_excess_bound_theta{theta}"] = {"id": "C3b", "kind": "IN-SAMPLE (bound part is implied by C3a; width correlation is empirical)", "statement": f"for equal-marginal pairs (|p_1-p_2| < {EQ_TOL}) the marginal-only minimax selector's realised excess fallback cost is <= the rem:frechet bound b xy/(x+y) (= (b/2)[min{{p,1-p_0}}-max{{0,p-p_0}}] at p_1=p_2) and grows with the interval width (Spearman > 0, CI) (theta={theta})",
            "result": f"{e_['n_triples']} (m0; m1, m2) triples in {e_['n_contexts']} contexts: excess <= exact bound in {e_['frac_within_exact_bound']*100:.1f}% ({e_['n_violations_exact_bound']} violations); <= equal-marginal formula at p = (p_1+p_2)/2 in {e_['frac_within_equal_marginal_bound_pbar']*100:.1f}% ({e_['n_violations_equal_marginal_bound_pbar']} violations, max ${e_['max_violation_equal_marginal_bound_usd']:.2e}); median excess/bound {e_['median_ratio_excess_over_exact_bound']:.2f} (10-90% [{e_['q10_q90_ratio'][0]:.2f}, {e_['q10_q90_ratio'][1]:.2f}]); excess exactly 0 in {e_['frac_excess_zero']*100:.0f}%; median excess ${e_['median_excess_usd']:.2e} vs median bound ${e_['median_bound_usd']:.2e}; Spearman(excess/b, width) pooled {e_['spearman_excess_prob_vs_width_pooled']:+.3f} [CI {lo_:+.3f}, {hi_:+.3f}]; Spearman(excess $, bound $) {e_['spearman_excess_usd_vs_bound_usd_pooled']:+.3f} [CI {e_['spearman_excess_usd_vs_bound_usd_pooled_ci'][0]:+.3f}, {e_['spearman_excess_usd_vs_bound_usd_pooled_ci'][1]:+.3f}]; within-context Spearman median {(e_['within_context_spearman']['median'] if e_['within_context_spearman']['median'] is not None else float('nan')):.3f} over {e_['within_context_spearman']['n_contexts_ge5']} contexts with >= 5 triples (positive in {(e_['within_context_spearman']['frac_positive'] or 0)*100:.0f}%)",
            "verdict": v}

sb = R["in_sample"]["summary_b4_and_drift"]; ec = R["in_sample"]["error_correlation_summary"]
for theta in THETAS:
    d_ = sb.get(f"theta={theta}|all_non_overall")
    if d_ is None:
        continue
    vm = d_["vm_drift"]
    C[f"C4a_common_vm_theta{theta}"] = {"id": "C4a", "kind": "IN-SAMPLE hypothesis (diagnostic)", "statement": f"thm:marginal hypothesis of common v_m: v_m(S+m) = v_m(S) (gpt-4's mean score on the failure set does not change when m is added) (theta={theta}; non-trivial (S,m), |S| in 1..2)",
        "result": f"n={d_['n']} cases in {d_['n_contexts']} contexts; CI of v_m(S+m)-v_m(S) excludes 0 in {vm['frac_ci_excl0']*100:.0f}% (violation frequency); median drift {vm['median']:+.4f} ({(vm['median_rel'] or 0)*100:+.1f}% of v_m(S)); negative in {vm['frac_negative']*100:.0f}%; 10-90% rel [{vm['q10_q90_rel'][0]*100:+.1f}%, {vm['q10_q90_rel'][1]*100:+.1f}%]; b drift: median {(d_['b_drift']['median_rel'] or 0)*100:+.1f}%, CI excl 0 in {d_['b_drift']['frac_ci_excl0']*100:.0f}%; v_h drift: CI excl 0 in {d_['vh_drift']['frac_ci_excl0']*100:.0f}% (max |drift| {d_['vh_drift']['max_abs']:.3f}); in-sample b4 sign agreement {d_['sign_agreement_all']*100:.1f}% (all), {(d_['sign_agreement_realised_ci_decided'] or 0)*100:.1f}% on {d_['n_realised_ci_decided']} realised-CI-decided, {(d_['sign_agreement_both_ci_decided'] or 0)*100:.1f}% on {d_['n_both_ci_decided']} both-CI-decided; baselines (realised-CI-decided): indep proxy {(d_['baseline_N_indep_realised_ci_decided'] or 0)*100:.1f}%, always negative {(d_['baseline_always_negative_realised_ci_decided'] or 0)*100:.1f}%; median |rel err| of the predicted change {(d_['median_rel_err_dpred'] or 0)*100:.1f}% -> {(d_['median_rel_err_substitute_vm2'] or 0)*100:.1f}% with S+m's v_m, {(d_['median_rel_err_substitute_b2'] or 0)*100:.1f}% with S+m's b",
        "verdict": verdict_hyp(vm["frac_ci_excl0"], vm["median_rel"])}
    e_ = ec.get(f"theta={theta}|all_non_overall")
    if e_:
        C[f"C4b_error_correlation_theta{theta}"] = {"id": "C4b", "kind": "IN-SAMPLE hypothesis (diagnostic, X2 c2)", "statement": f"errors are positively correlated: P(E_i^c & E_j^c) > P(E_i^c) P(E_j^c) for usable model pairs (theta={theta})",
            "result": f"{e_['n_pairs']} pairs: excess > 0 in {e_['frac_positive']*100:.0f}%, CI > 0 in {e_['frac_positive_ci']*100:.0f}%, CI < 0 in {e_['frac_negative_ci']*100:.1f}%; median excess {e_['median_excess']:.3f}; families only: > 0 in {ec[f'theta={theta}|families']['frac_positive']*100:.0f}% of {ec[f'theta={theta}|families']['n_pairs']}",
            "verdict": "PASS" if (e_["frac_positive"] >= 0.9 and e_["frac_positive_ci"] >= 0.75) else "FAIL"}

so_ = R["oos"]["selector_excess_summary"]
for theta in THETAS:
    s_ = so_.get(f"theta={theta}|all_non_overall")
    if s_ is None:
        continue
    lo_, hi_ = s_["mean_diff_ci"]
    v = "PASS" if (lo_ is not None and lo_ > 0) else ("FAIL" if (hi_ is not None and hi_ < 0) else "INCONCLUSIVE")
    C[f"C5_oos_selector_excess_theta{theta}"] = {"id": "C5", "kind": "OUT-OF-SAMPLE", "statement": f"prop:marginals content: for equal-marginal candidate pairs (fit split), the r_m-informed selector (argmax fit-split r_m) has a smaller realised held-out excess fallback cost than the minimax marginal-only selector q = x/(x+y) (theta={theta})",
        "result": f"{s_['n_triples']} triples in {s_['n_contexts']} contexts: mean held-out excess marginal-only ${s_['mean_excess_marginal_usd']:.2e}, informed ${s_['mean_excess_informed_usd']:.2e} (ratio {s_['ratio_informed_over_marginal_mean_excess']:.2f}), uniform-random ${s_['mean_excess_random_usd']:.2e}; mean difference marginal - informed ${s_['mean_diff_marginal_minus_informed_usd']:.2e} [CI ${lo_:.2e}, ${hi_:.2e}]; informed better in {s_['sign_test']['informed_better']}, marginal better in {s_['sign_test']['marginal_better']}, ties {s_['sign_test']['ties']} (p = {(s_['sign_test']['p_two_sided'] if s_['sign_test']['p_two_sided'] is not None else float('nan')):.2g}); the fit-split informed pick is the held-out best partner in {s_['frac_informed_pick_is_heldout_best']*100:.0f}%",
        "verdict": v}
R["claims"] = list(C.values())
R["claims_by_key"] = C

# ==============================================================================================
# 6. Figures (vector PDF; matplotlib, usetex False, 9-pt serif, no titles inside plots)
# ==============================================================================================
def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name), metadata={"Creator": "X11_residual_coverage_routerbench/run.py"})
    fig.savefig(os.path.join(FIG, name.replace(".pdf", ".png")), dpi=160)
    plt.close(fig)
FAM_LABEL = {"mmlu": "MMLU", "hellaswag": "HellaSwag", "grade-school-math": "GSM8K", "arc-challenge": "ARC-C", "winogrande": "WinoGrande", "mbpp": "MBPP", "mt-bench": "MT-Bench", "rag-other": "RAG/other"}

# Fig 1 (C3a): Frechet intervals with realised r_m, theta = 0.5, one panel per main primary, 8 families
fig, axes = plt.subplots(2, 2, figsize=(6.75, 4.6), layout="constrained", sharey=True)
for ax, m0 in zip(axes.ravel(), POOL_MAIN):
    rs = [r for r in rows_fre if r["theta"] == THETA_MAIN and r["kind"] == "family" and r["m0"] == SHORT[m0]]
    xt, xl = [], []
    for g, fam in enumerate(FAMILY_ORDER):
        rf = sorted([r for r in rs if r["ctx"] == fam], key=lambda r: r["p_m"])
        if not rf:
            continue
        xs = g + np.linspace(-0.36, 0.36, len(rf))
        for x, r in zip(xs, rf):
            ax.plot([x, x], [r["ell"], r["u"]], color=GREY, lw=1.1, solid_capstyle="butt")
        ax.scatter(xs, [r["r_ind"] for r in rf], s=9, facecolors="none", edgecolors=ORANGE, lw=.6, zorder=3)
        ax.scatter(xs, [r["r"] for r in rf], s=9, color=BLUE, lw=0, zorder=4)
        xt.append(g); xl.append(FAM_LABEL[fam])
    ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=35, ha="right", fontsize=7)
    ax.set_ylim(-0.02, 0.62); ax.grid(axis="y", lw=.3, color=".85")
    ax.text(0.02, 0.96, r"$S=\{$" + SHORT[m0] + r"$\}$", transform=ax.transAxes, va="top", fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel(r"residual coverage $r_m(\{m_0\})$")
h = [plt.Line2D([], [], color=GREY, lw=1.1, label=r"Fr\'echet range $[\max\{0,p_m-p_0\},\ \min\{p_m,1-p_0\}]$".replace(r"\'e", "é")),
     plt.Line2D([], [], marker="o", color=BLUE, lw=0, ms=3.5, label=r"realised $r_m$"),
     plt.Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor=ORANGE, lw=0, ms=3.5, label=r"independence proxy $(1-p_0)\,p_m$")]
fig.legend(handles=h, loc="outside upper center", ncol=3, frameon=False, fontsize=7.5)
savefig(fig, "fig_frechet_intervals.pdf")

# Fig 2 (C3b): excess of the marginal-only selector vs interval width, and vs the rem:frechet bound (dollars)
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.6), layout="constrained")
ax = axes[0]
rs = [r for r in rows_exc if r["theta"] == THETA_MAIN and r["kind"] != "overall"]
for kind_, col, lab, mk in [("family", BLUE, "8 families", "o"), ("mmlu_subset", GREEN, "57 MMLU subsets", "s")]:
    rk = [r for r in rs if r["kind"] == kind_]
    ax.scatter([r["width_eq"] for r in rk], [r["excess_prob"] for r in rk], s=7 if kind_ == "mmlu_subset" else 12, color=col, alpha=.5, lw=0, marker=mk, label=lab)
wmax = max(r["width_eq"] for r in rs) * 1.05
ax.plot([0, wmax], [0, wmax / 2], color=RED, lw=1, ls="--", label=r"bound $\frac{1}{2}[\min\{p,1-p_0\}-\max\{0,p-p_0\}]$")
ax.set(xlabel=r"Fréchet interval width $\min\{p,1-p_0\}-\max\{0,p-p_0\}$", ylabel="excess / b of the marginal-only selector", xlim=(0, wmax))
ax.legend(loc="upper left", frameon=False)
ax = axes[1]
for kind_, col, lab, mk in [("family", BLUE, "8 families", "o"), ("mmlu_subset", GREEN, "57 MMLU subsets", "s")]:
    rk = [r for r in rs if r["kind"] == kind_ and r["excess_usd"] > 0 and r["bound_exact_usd"] > 0]
    ax.scatter([r["bound_exact_usd"] for r in rk], [r["excess_usd"] for r in rk], s=7 if kind_ == "mmlu_subset" else 12, color=col, alpha=.5, lw=0, marker=mk, label=lab)
allb = [r["bound_exact_usd"] for r in rs if r["bound_exact_usd"] > 0]
lo_, hi_ = min(allb) * 0.5, max(allb) * 2
ax.plot([lo_, hi_], [lo_, hi_], color=RED, lw=1, ls="--", label=r"excess = bound $b\,xy/(x+y)$")
ax.set(xscale="log", yscale="log", xlabel="rem:frechet bound (USD per prompt)", ylabel="realised excess (USD per prompt)")
ax.legend(loc="upper left", frameon=False)
savefig(fig, "fig_excess_vs_width.pdf")

# Fig 3 (C1): out-of-sample sign agreement by context (CI-decided held-out cases), theta = 0.5
pc = R["oos"]["per_context"][f"theta={THETA_MAIN}"]
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.8), layout="constrained", width_ratios=[1.35, 1])
ax = axes[0]
names_ = [n_ for n_ in ["overall"] + FAMILY_ORDER + ["MMLU subsets (pooled)"] if n_ in pc]
ys = np.arange(len(names_))[::-1]
for rn, col, mk, lab in [("theorem_N", BLUE, "o", r"eq:marginal $N_S$ (fit split)"), ("N_indep", ORANGE, "s", r"$N_S$ with independence proxy"),
                         ("marginal_quality_fitted_tau", GREEN, "^", r"$P(E_m)\geq\tau$ (fitted)"), ("always_negative", GREY, "x", "always negative")]:
    ax.scatter([100 * (pc[n_]["agree_ci_decided"][rn] or 0) for n_ in names_], ys, color=col, marker=mk, s=18, lw=.8 if mk == "x" else 0, label=lab, zorder=3)
ax.axvline(85, color=RED, lw=.8, ls=":")
ax.set_yticks(ys); ax.set_yticklabels([FAM_LABEL.get(n_, n_) + f" ({pc[n_]['n_ci_decided']})" for n_ in names_], fontsize=7.5)
ax.set_xlabel("held-out sign agreement on CI-decided cases (%)"); ax.set_xlim(0, 102); ax.grid(axis="x", lw=.3, color=".85")
ax.legend(loc="lower left", frameon=False, fontsize=6.8)
ax = axes[1]
xs_, ys_, ns_ = [], [], []
for e in MMLU_SUBSETS:
    ss = [r for r in oos_sign if r["theta"] == THETA_MAIN and r["ctx"] == e and r["ci_decided"]]
    if len(ss) >= 5:
        xs_.append(100 * np.mean([r["agree_N_indep"] for r in ss])); ys_.append(100 * np.mean([r["agree_theorem_N"] for r in ss])); ns_.append(len(ss))
ax.scatter(xs_, ys_, s=np.array(ns_) * 0.5, color=GREEN, alpha=.55, lw=0)
ax.plot([40, 101], [40, 101], color=GREY, lw=.6, ls=":"); ax.axhline(85, color=RED, lw=.8, ls=":")
ax.set(xlabel=r"agreement of $N_S$ with independence proxy (%)", ylabel=r"agreement of eq:marginal $N_S$ (%)", xlim=(40, 101), ylim=(40, 101))
ax.text(0.03, 0.04, f"{len(xs_)} MMLU subsets (marker area = # CI-decided cases)", transform=ax.transAxes, fontsize=6.8)
savefig(fig, "fig_sign_agreement_by_context.pdf")

# Fig 4 (C2): per-portfolio Spearman of fit-split rankings against the realised held-out change, theta = 0.5
fig, ax = plt.subplots(figsize=(3.4, 2.6), layout="constrained")
rs = [r for r in oos_rows if r["theta"] == THETA_MAIN and r["kind"] != "overall"]
keys = ["spearman_fit_a_vs_heldout_gain", "spearman_fit_r_vs_heldout_gain", "spearman_fit_apd_vs_heldout_gain", "spearman_fit_Nindep_vs_heldout_gain", "spearman_fit_N_vs_heldout_gain"]
labels = [r"$P(E_m)$", r"$r_m(S)$", r"$P(E_m)/d_m$", r"$N_S$, indep.", r"$N_S$"]
data = [[r[k] for r in rs if r[k] is not None and np.isfinite(r[k])] for k in keys]
bp = ax.boxplot(data, widths=.55, showfliers=False, medianprops={"color": RED, "lw": 1.2}, boxprops={"color": ".3", "lw": .7}, whiskerprops={"color": ".3", "lw": .7}, capprops={"color": ".3", "lw": .7})
for i, d_ in enumerate(data):
    ax.scatter(np.full(len(d_), i + 1) + np.random.default_rng(1).uniform(-.18, .18, len(d_)), d_, s=3, color=BLUE, alpha=.18, lw=0, zorder=0)
ax.axhline(0, color=GREY, lw=.6, ls=":")
ax.set_xticks(range(1, len(keys) + 1)); ax.set_xticklabels(labels, fontsize=7.5)
ax.set(ylabel=r"Spearman(fit-split ranking, held-out gain)", ylim=(-1.05, 1.05))
ax.text(0.02, 0.03, f"{len(rs)} portfolios", transform=ax.transAxes, fontsize=7)
savefig(fig, "fig_spearman_rankings.pdf")
print(f"figures written ({time.time()-T0:.0f}s)")

# ==============================================================================================
# 7. AUDIT: independent recomputation of headline numbers with pandas (no tables/wstats/derived machinery)
# ==============================================================================================
AUD = {}
pdf = pd.DataFrame({m: S_all[:, MODELS.index(m)] for m in MODELS}); cdf = pd.DataFrame({m: C_all[:, MODELS.index(m)] for m in MODELS})
pdf["ctx_family"] = fam_all; pdf["eval"] = ev_all; pdf["fit"] = is_fit
def ctx_frame(name, kind, split=None):
    sel = np.ones(N_ALL, bool) if kind == "overall" else (pdf["ctx_family"].to_numpy() == name) if kind == "family" else (pdf["eval"].to_numpy() == name)
    if split == "fit":
        sel &= is_fit
    elif split == "heldout":
        sel &= ~is_fit
    return pdf[sel], cdf[sel]
def eta_of(P_, Cc_, members, theta):
    """Cascade efficiency of portfolio `members` (list of model names) with gpt-4 fallback: ratio of sums, plain pandas."""
    E = (P_[members] >= theta)
    hit = E.any(axis=1)
    Vsel = P_[members].where(E, -np.inf).max(axis=1).where(hit, 0.0)
    V = Vsel.where(hit, P_[REF]); D = Cc_[members].sum(axis=1) + (~hit) * Cc_[REF]
    return float(V.sum() / D.sum()), hit, V, D
# (1) C1: held-out sign agreement of the fit-split N_S, recomputed case by case; CI-decided flags reused from the main run
agree, agree_dec, n_dec, n_all, maxdiffN = 0, 0, 0, 0, 0.0
for rec in [r for r in oos_sign if r["theta"] == THETA_MAIN and r["kind"] != "overall"]:
    name = rec["ctx"]; kind = rec["kind"]
    Sm = [NONREF[k] for k in range(K) if SHORT[NONREF[k]] in rec["S"].split("+")]; m = [NONREF[k] for k in range(K) if SHORT[NONREF[k]] == rec["m"]][0]
    Pf, Cf = ctx_frame(name, kind, "fit"); Pt, Ct = ctx_frame(name, kind, "heldout")
    etaS, hit, V, D = eta_of(Pf, Cf, Sm, THETA_MAIN)
    p_S = hit.mean(); vh = V[hit].mean(); vm = Pf[REF][~hit].mean(); b = Cf[REF][~hit].mean(); d = Cf[m].mean()
    r = float(((Pf[m] >= THETA_MAIN) & ~hit).mean())
    N = r * (vh - vm + etaS * b) - etaS * d
    maxdiffN = max(maxdiffN, abs(N - rec["N_fit"]))
    dreal = eta_of(Pt, Ct, Sm + [m], THETA_MAIN)[0] - eta_of(Pt, Ct, Sm, THETA_MAIN)[0]
    ok = np.sign(N) == np.sign(dreal); n_all += 1; agree += int(ok)
    if rec["ci_decided"]:
        n_dec += 1; agree_dec += int(ok)
AUD["C1_recomputed"] = {"n_cases": n_all, "agreement_all": agree / n_all, "n_ci_decided_reused_flags": n_dec, "agreement_ci_decided": agree_dec / n_dec, "max_abs_diff_N_fit_vs_main": maxdiffN,
                        "main_run": {"agreement_all": sd[f"theta={THETA_MAIN}|all_non_overall"]["sign_agreement_all"]["theorem_N"], "agreement_ci_decided": sd[f"theta={THETA_MAIN}|all_non_overall"]["sign_agreement_ci_decided"]["theorem_N"]}}
# (1b) independent bootstrap of the CI-decided flags for one context (hellaswag), explicit resampling loop
name_chk = "hellaswag"; Pt, Ct = ctx_frame(name_chk, "family", "heldout"); n_te = len(Pt)
rng_chk = np.random.default_rng([SEED, [i for i, (n_, k_) in enumerate(CONTEXTS) if n_ == name_chk][0]])
_ = rng_chk.multinomial(int((fam_all == name_chk).sum()), np.full(int((fam_all == name_chk).sum()), 1.0 / (fam_all == name_chk).sum()), size=N_BOOT)   # skip W_full (same stream order as the main run)
W_chk = rng_chk.multinomial(n_te, np.full(n_te, 1.0 / n_te), size=N_BOOT).astype(float)
flags_match, n_chk = 0, 0
for rec in [r for r in oos_sign if r["theta"] == THETA_MAIN and r["ctx"] == name_chk]:
    Sm = [NONREF[k] for k in range(K) if SHORT[NONREF[k]] in rec["S"].split("+")]; m = [NONREF[k] for k in range(K) if SHORT[NONREF[k]] == rec["m"]][0]
    _, _, V1, D1 = eta_of(Pt, Ct, Sm, THETA_MAIN); _, _, V2, D2 = eta_of(Pt, Ct, Sm + [m], THETA_MAIN)
    dB = (W_chk @ V2.to_numpy()) / (W_chk @ D2.to_numpy()) - (W_chk @ V1.to_numpy()) / (W_chk @ D1.to_numpy())
    lo_, hi_ = np.percentile(dB, 2.5), np.percentile(dB, 97.5)
    flags_match += int(((lo_ > 0) or (hi_ < 0)) == rec["ci_decided"]); n_chk += 1
AUD["C1_ci_flags_recomputed_hellaswag"] = {"n_cases": n_chk, "n_flags_matching": flags_match}
# (2) C3a: Frechet inclusion and position, recomputed from the raw frame for every (context, m0 in pool, m) pair with both usable
inside, n_pairs, positions = 0, 0, []
for name, kind in CONTEXTS:
    if kind == "overall":
        continue
    P_, _ = ctx_frame(name, kind)
    E = P_[NONREF] >= THETA_MAIN; npass = E.sum(); use = (npass >= MIN_COUNT) & ((len(P_) - npass) >= MIN_COUNT)
    for m0 in POOL:
        if not use[m0]:
            continue
        for m in NONREF:
            if m == m0 or not use[m]:
                continue
            p0, pm = E[m0].mean(), E[m].mean(); r = (E[m] & ~E[m0]).mean()
            ell, u = max(0.0, pm - p0), min(pm, 1 - p0)
            n_pairs += 1; inside += int(ell - 1e-12 <= r <= u + 1e-12)
            if u - ell > 1e-12:
                positions.append((r - ell) / (u - ell))
AUD["C3a_recomputed"] = {"n_pairs": n_pairs, "frac_inside": inside / n_pairs, "position_median": float(np.median(positions)),
                         "main_run": {"n_pairs": fs[f"theta={THETA_MAIN}|all_non_overall"]["n_pairs"], "frac_inside": fs[f"theta={THETA_MAIN}|all_non_overall"]["frac_inside_same_sample"], "position_median": fs[f"theta={THETA_MAIN}|all_non_overall"]["position_median"]}}
# (3) C3b: equal-marginal triples, excess of the minimax marginal-only selector vs the bound, pooled Spearman with width
ratios, exc_p, wid, n_tri, n_within = [], [], [], 0, 0
for name, kind in CONTEXTS:
    if kind == "overall":
        continue
    P_, C_ = ctx_frame(name, kind)
    E = P_[NONREF] >= THETA_MAIN; npass = E.sum(); use = (npass >= MIN_COUNT) & ((len(P_) - npass) >= MIN_COUNT)
    for m0 in POOL:
        if not use[m0]:
            continue
        p0 = E[m0].mean(); b = C_[REF][~E[m0]].mean(); cands = [m for m in NONREF if m != m0 and use[m]]
        for ia in range(len(cands)):
            for ib in range(ia + 1, len(cands)):
                m1, m2 = cands[ia], cands[ib]; p1, p2 = E[m1].mean(), E[m2].mean()
                if abs(p1 - p2) >= EQ_TOL:
                    continue
                r1, r2 = (E[m1] & ~E[m0]).mean(), (E[m2] & ~E[m0]).mean()
                l1, u1, l2, u2 = max(0, p1 - p0), min(p1, 1 - p0), max(0, p2 - p0), min(p2, 1 - p0)
                x, y = max(0.0, u1 - l2), max(0.0, u2 - l1); q = x / (x + y) if x + y > 0 else 0.5
                exc = b * (max(r1, r2) - q * r1 - (1 - q) * r2); bound = b * (x * y / (x + y) if x + y > 0 else 0.0)
                n_tri += 1; n_within += int(exc <= bound + 1e-12)
                if bound > 1e-12:
                    ratios.append(exc / bound)
                pbar = 0.5 * (p1 + p2); exc_p.append(exc / b); wid.append(min(pbar, 1 - p0) - max(0, pbar - p0))
AUD["C3b_recomputed"] = {"n_triples": n_tri, "frac_within_exact_bound": n_within / n_tri, "median_ratio": float(np.median(ratios)), "spearman_excess_prob_vs_width": spearman(exc_p, wid),
                         "main_run": {"n_triples": es[f"theta={THETA_MAIN}|all_non_overall"]["n_triples"], "frac_within_exact_bound": es[f"theta={THETA_MAIN}|all_non_overall"]["frac_within_exact_bound"],
                                      "median_ratio": es[f"theta={THETA_MAIN}|all_non_overall"]["median_ratio_excess_over_exact_bound"], "spearman_excess_prob_vs_width": es[f"theta={THETA_MAIN}|all_non_overall"]["spearman_excess_prob_vs_width_pooled"]}}
# (4) C2: median Spearman(fit r_m, held-out gain) and (fit P(E_m), held-out gain) over the stored portfolios, recomputed with pandas
sr_l, sa_l = [], []
for rec in [r for r in oos_rows if r["theta"] == THETA_MAIN and r["kind"] != "overall"]:
    Sm = [NONREF[k] for k in range(K) if SHORT[NONREF[k]] in rec["S"].split("+")]; ms = [[NONREF[k] for k in range(K) if SHORT[NONREF[k]] == p_][0] for p_ in rec["partners"]]
    Pf, Cf = ctx_frame(rec["ctx"], rec["kind"], "fit"); Pt, Ct = ctx_frame(rec["ctx"], rec["kind"], "heldout")
    _, hit, _, _ = eta_of(Pf, Cf, Sm, THETA_MAIN)
    r_fit = [float(((Pf[m] >= THETA_MAIN) & ~hit).mean()) for m in ms]; a_fit = [float((Pf[m] >= THETA_MAIN).mean()) for m in ms]
    e0 = eta_of(Pt, Ct, Sm, THETA_MAIN)[0]; g = [eta_of(Pt, Ct, Sm + [m], THETA_MAIN)[0] - e0 for m in ms]
    sr_l.append(spearman(r_fit, g)); sa_l.append(spearman(a_fit, g))
AUD["C2_recomputed"] = {"n_portfolios": len(sr_l), "median_spearman_fit_r_vs_gain": float(np.nanmedian(sr_l)), "median_spearman_fit_a_vs_gain": float(np.nanmedian(sa_l)),
                        "r_better": int(np.sum(np.array(sr_l) > np.array(sa_l) + 1e-12)), "a_better": int(np.sum(np.array(sa_l) > np.array(sr_l) + 1e-12)),
                        "main_run": {k_: sd[f"theta={THETA_MAIN}|all_non_overall"]["d3_gain_target"][k_] for k_ in ["median_spearman_fit_r", "median_spearman_fit_a"]} | {"sign_test": sd[f"theta={THETA_MAIN}|all_non_overall"]["d3_gain_target"]["sign_test"]}}
# (5) data facts used in the README
AUD["duplicate_prompts"] = int(pd.Series(df["prompt"]).duplicated().sum())
AUD["negative_error_correlation_examples"] = R["in_sample"]["error_correlation_summary"][f"theta={THETA_MAIN}|all_non_overall"]["negative_ci_cases"][:8]
R["audit"] = AUD
R["meta"]["runtime_s"] = round(time.time() - T0, 1)
print(f"audit done ({time.time()-T0:.0f}s): C1 {AUD['C1_recomputed']['agreement_ci_decided']:.4f} vs {AUD['C1_recomputed']['main_run']['agreement_ci_decided']:.4f}; "
      f"C3a inside {AUD['C3a_recomputed']['frac_inside']:.4f}; C3b ratio {AUD['C3b_recomputed']['median_ratio']:.3f} vs {AUD['C3b_recomputed']['main_run']['median_ratio']:.3f}; "
      f"C2 {AUD['C2_recomputed']['median_spearman_fit_r_vs_gain']:.3f}/{AUD['C2_recomputed']['median_spearman_fit_a_vs_gain']:.3f}; flags {AUD['C1_ci_flags_recomputed_hellaswag']}")

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(R, f, indent=1, default=lambda o: (None if isinstance(o, float) and not np.isfinite(o) else (o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o)))))
print(f"results.json written ({time.time()-T0:.0f}s)")
for k, v in C.items():
    print(f"- {k} [{v['kind']}]: {v['verdict']} -- {v['result'][:200]}")

# ==============================================================================================
# 8. README.md and summary_for_paper.md generated from R (numbers are not retyped)
# ==============================================================================================
def pct(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"
def f3(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"
def usd(x):
    return "n/a" if x is None else f"${x:.2e}"
def write_readme(R):
    M = R["meta"]; Cl = R["claims_by_key"]; sd = R["oos"]["summary_d"]; fs = R["frechet"]["interval_summary"]; es = R["frechet"]["excess_summary"]
    sb = R["in_sample"]["summary_b4_and_drift"]; ec = R["in_sample"]["error_correlation_summary"]; so_ = R["oos"]["selector_excess_summary"]; A = R["audit"]
    th = THETA_MAIN; K0 = f"theta={th}|all_non_overall"
    L = [f"# X11_residual_coverage_routerbench -- identifiability of residual coverage on RouterBench (eq:residual, thm:marginal, prop:marginals, rem:frechet)\n",
         f"Script: `run.py` (numpy/pandas/scipy/matplotlib only; fully offline; master seed {M['seed']}; {M['n_boot']} bootstrap task resamples per context and split, 500 rather than 1,000 because three bootstraps (full, fit, held-out) are drawn for each of {len(CONTEXTS)} contexts; runtime {M['runtime_s']} s on the M4). Outputs: `results.json` (every number below with 95% percentile CIs), `figures/*.pdf` (+ `.png` previews), `summary_for_paper.md`, this README (generated by `run.py`).\n",
         "## 1. Setup\n",
         f"* **Data.** RouterBench 0-shot pickle `$ROUTERBENCH_PKL` ({M['n_prompts']} prompts x 11 models; score column `<model>`, cost column `<model>|total_cost` in dollars; 86 `eval_name`). Cross-checks asserted in code: gpt-4-1106-preview mean score {M['gpt4_mean_score']:.4f} and mean cost ${M['gpt4_mean_cost']:.6f} per prompt (RouterBench / CONVENTIONS). No score or cost is NaN or zero; {A['duplicate_prompts']} prompts have duplicated text (different `sample_id`) and are kept as recorded.",
         f"* **Reference / fallback** B = always `{REF}` (Y_B, C_B = its score and dollars); it is called and committed whenever the portfolio misses. **Primaries** for the portfolios S (|S| in 1..2): {', '.join(POOL_MAIN)} (the protocol's four) plus the cheap extras {', '.join(POOL_EXTRA)}; every statistic is also reported for the four-primary subset (`main4`). **Candidates** m: all other non-reference models ({K} models incl. claude-v1, claude-v2, code-llama-34b, mistral-7b). {M['n_pool_portfolios']} portfolios x candidates = {M['n_cases_per_context']} (S, m) cases per context.",
         f"* **Verifier and cascade** (as X2): exact selecting verifier, E_m = {{score_m >= theta}} with theta = {th} (main) and theta = 1.0. All members of S are called in parallel; the verifier commits the highest-scoring passing member; on a miss gpt-4 is called and committed. D = sum of member costs + (1-H) C_gpt4; a(S) = E[sum member costs]; d_m(S) = E[cost_m] (no aggregation fee); V = committed score; eta = E[V]/E[D] (ratio of sums). Primitives: p(S), r_m(S) = P(E_m & all of S fail), v_h(S) = E[V | hit], v_m(S) = E[Y_gpt4 | miss], b(S) = E[C_gpt4 | miss].",
         f"* **Thresholds.** {pct(np.mean([BINARY[n] for n, k in CONTEXTS if k != 'overall']), 0)} of the contexts have only 0/1 scores, where E_m at theta = 0.5 and theta = 1 are the same events; theta = 1 is therefore run only in the contexts with fractional scores ({', '.join(n for n, k in CONTEXTS if not BINARY[n])}: GSM8K scores in {{0, .25, .5, .75, 1}}, MT-Bench in tenths, `consensus_summary` and `chinese_idioms` in quarters) and its rows are identical to theta = 0.5 everywhere else.",
         f"* **Contexts** ({len(CONTEXTS) - 1} + pooled `overall`): the 8 source families below and every MMLU subset with >= {MIN_MMLU} prompts ({len(MMLU_SUBSETS)} of 57; the smallest MMLU subsets have exactly 100). The `overall` context pools everything and is reported separately (X2 showed that pooling heterogeneous costs inflates the b-drift).",
         f"* **Split.** Every prompt is assigned once to fit (70%) or held-out (30%) by stratified sampling within strata `eval_name x 1{{gpt-4 score >= 0.5}}` (seed {M['seed']}): fit {M['n_fit']}, held-out {M['n_heldout']}. Contexts inherit the assignment, so each context is itself split ~70/30. OUT-OF-SAMPLE = primitives (p, r_m, v_h, v_m, b, d_m, eta, N_S, rankings, usability, the non-trivial filter) on the fit split, realised quantities and their CIs on the held-out split.",
         f"* **Filters** (X2's): a model is *usable* in a sample if it has >= {MIN_COUNT} passes and >= {MIN_COUNT} failures; an (S, m) case is *non-trivial* if #(E_m & fail S) >= {MIN_CELL}, #fail(S+m) >= {MIN_CELL}, #hit(S) >= {MIN_CELL}. Sign agreement is reported on all non-trivial cases and on *CI-decided* cases (the held-out realised change has a bootstrap CI excluding 0), always next to baselines. Cost basis: dollars throughout.",
         "\n### eval_name grouping used for the 8 source families\n", "| family | eval_name members | prompts | fit / held-out | binary scores | P(gpt-4 >= 0.5) | gpt-4 $/prompt |", "|---|---|---|---|---|---|---|"]
    for f_ in FAMILY_ORDER:
        c = R["contexts"][f_]; t_ = c[f"theta={th}"]
        mem = FAMILY_MEMBERS[f_]
        mem_txt = (f"{len(mem)} `mmlu-*` subsets" if f_ == "mmlu" else ", ".join(f"`{e}`" for e in mem))
        L.append(f"| {f_} | {mem_txt} | {c['n']} | {c['n_fit']} / {c['n_heldout']} | {'yes' if c['binary_scores'] else 'no'} | {t_['p_gpt4']:.3f} | {t_['cbar_gpt4']:.5f} |")
    L.append(f"\n`rag-other` is RouterBench's residual: its RAG / Chinese-language / summarisation / bias datasets (`{'`, `'.join(FAMILY_MEMBERS['rag-other'])}`), several with < 40 prompts; it is analysed only as one pooled family. `mt-bench` pools `mtbench`, `mtbench-math`, `mtbench-reference` (80 prompts; after the split only {len(R['contexts']['mt-bench'][f'theta={th}']['usable_fit'])} models are usable on its fit split, so it contributes few out-of-sample cases).\n")
    # ---- claims table
    L += ["## 2. Claims\n", "| id | kind | statement | result | verdict |", "|---|---|---|---|---|"]
    for k, v in Cl.items():
        L.append(f"| {k} | {v['kind']} | {v['statement']} | {v['result']} | **{v['verdict']}** |")
    # ---- results: C1
    d_ = sd[K0]; ag = d_["sign_agreement_ci_decided"]; agA = d_["sign_agreement_all"]
    L += ["\n## 3. Results\n", f"### 3.1 C1 -- out-of-sample sign rule eq:marginal (X2's d1), theta = {th}\n",
          f"Fit-split N_S = r_m(S)[v_h - v_m + eta(S) b] - eta(S) d_m(S) (S's own v_h, v_m, b) predicts the sign of the held-out change eta(S+m) - eta(S). {d_['n_sign_cases']} non-trivial (S, m) cases in {d_['n_contexts']} contexts, {d_['n_sign_ci_decided']} CI-decided; the realised change is positive in {pct(d_['frac_realised_positive_ci_decided'])} of the CI-decided cases, so a constant rule is at chance.\n",
          "| rule (sign of ...) | inputs | CI-decided | all cases | main4 primaries, CI-decided | families, CI-decided | MMLU subsets, CI-decided |", "|---|---|---|---|---|---|---|"]
    rule_desc = {"theorem_N": ("eq:marginal N_S with the paired-outcome r_m(S)", "joint outcomes + prices"), "value_preserving_rb_minus_d": ("value-preserving special case r_m b - d_m", "joint outcomes + prices"),
                 "N_indep": ("N_S with the independence proxy (1-p(S)) P(E_m)", "marginals + prices"), "frechet_upper": ("N_S with r_m := Frechet upper bound min{p_m, 1-max p_i}", "marginals + prices"),
                 "frechet_lower": ("N_S with r_m := Frechet lower bound max{0, p_m - sum p_i}", "marginals + prices"), "marginal_quality_fitted_tau": ("P(E_m) >= tau, tau fitted on the fit split's realised signs", "marginals only"),
                 "marginal_quality_gt_pS": ("P(E_m) > p(S)", "marginals only"), "always_negative": ("always negative", "none"), "always_positive": ("always positive", "none")}
    for rn in RULES:
        L.append(f"| {rule_desc[rn][0]} | {rule_desc[rn][1]} | {pct(ag[rn])} | {pct(agA[rn])} | {pct(sd[f'theta={th}|main4_non_overall']['sign_agreement_ci_decided'][rn])} | {pct(sd[f'theta={th}|families']['sign_agreement_ci_decided'][rn])} | {pct(sd[f'theta={th}|mmlu_subsets']['sign_agreement_ci_decided'][rn])} |")
    pc = R["oos"]["per_context"][f"theta={th}"]
    L += ["\nPer context (held-out; `MMLU subsets (pooled)` pools the 57 subset contexts):\n", "| context | #S | sign cases | CI-decided | N_S | indep. proxy | P(E_m) >= tau | always neg. | median Spearman(fit r_m, gain) | median Spearman(fit P(E_m), gain) | median rel. gain of top N_S / top P(E_m) / oracle | top N_S = oracle |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n_, v in pc.items():
        L.append(f"| {n_} | {v['n_S']} | {v['n_sign']} | {v['n_ci_decided']} | {pct(v['agree_ci_decided']['theorem_N'])} | {pct(v['agree_ci_decided']['N_indep'])} | {pct(v['agree_ci_decided']['marginal_quality_fitted_tau'])} | {pct(v['agree_ci_decided']['always_negative'])} | {f3(v['median_spearman_fit_r_vs_gain'], 2)} | {f3(v['median_spearman_fit_a_vs_gain'], 2)} | {pct(v['median_relgain_top_theorem_N'])} / {pct(v['median_relgain_top_quality_a'])} / {pct(v['median_relgain_top_oracle'])} | {pct(v['frac_top_theorem_is_oracle'], 0)} |")
    dis_all = [r for r in R["oos"]["sign_disagreements_ci_decided_theorem"] if r["theta"] == th]
    dis = [r for r in dis_all if r["kind"] != "overall"]; dis_ov = [r for r in dis_all if r["kind"] == "overall"]
    import collections as _c
    L.append(f"\nThe {len(dis)} CI-decided disagreements at theta = {th} (contexts other than `overall`) are {d_['disagree_pattern_theorem_ci_decided']['pred_pos_real_neg']} pred+/real- and {d_['disagree_pattern_theorem_ci_decided']['pred_neg_real_pos']} pred-/real+; most frequent contexts {dict(_c.Counter(r['ctx'] for r in dis).most_common(4))}, most frequent added models {dict(_c.Counter(r['m'] for r in dis).most_common(4))}. They are cases with |N_S| near zero whose realised change is driven by the drift of v_m and b when the failure set changes (Section 4), e.g. " + "; ".join(f"{r['ctx']}, S = {r['S']}, add {r['m']}: N_S(fit) = {r['N_fit']:+.4f} predicts {r['dpred_fit']:+.2f}, realised {r['dreal_heldout']:+.2f} [{r['dreal_heldout_ci'][0]:+.2f}, {r['dreal_heldout_ci'][1]:+.2f}] (fit r_m {r['r_fit']:.3f}, held-out r_m {r['r_heldout']:.3f})" for r in dis[:3]) + f". The pooled `overall` context adds {len(dis_ov)} disagreements of {pc['overall']['n_ci_decided']} CI-decided cases, all driven by the b-drift of pooled prices (e.g. " + "; ".join(f"S = {r['S']}, add {r['m']}: N_S(fit) = {r['N_fit']:+.4f} predicts {r['dpred_fit']:+.2f}, realised {r['dreal_heldout']:+.2f} [{r['dreal_heldout_ci'][0]:+.2f}, {r['dreal_heldout_ci'][1]:+.2f}]" for r in dis_ov[:2]) + ").\n")
    if f"theta=1.0|all_non_overall" in sd:
        d1 = sd["theta=1.0|all_non_overall"]
        L.append(f"theta = 1.0 (fractional-score contexts only, {d1['n_contexts']} contexts): {d1['n_sign_cases']} cases, {d1['n_sign_ci_decided']} CI-decided, N_S {pct(d1['sign_agreement_ci_decided']['theorem_N'])} vs independence proxy {pct(d1['sign_agreement_ci_decided']['N_indep'])} and P(E_m) >= tau {pct(d1['sign_agreement_ci_decided']['marginal_quality_fitted_tau'])}.\n")
    # ---- C2
    d3 = d_["d3_gain_target"]
    L += [f"### 3.2 C2 -- ranking candidates by r_m(S) vs by P(E_m) (X2's d2/d3), theta = {th}\n",
          f"For each of the {d_['n_S']} (context, S) portfolios with >= 3 non-trivial partners, the fit-split ranking of the partners is correlated (Spearman) with their realised held-out change eta(S+m) - eta(S). Median over portfolios: fit r_m(S) {f3(d3['median_spearman_fit_r'])}, fit P(E_m) {f3(d3['median_spearman_fit_a'])} (means {f3(d3['mean_spearman_fit_r'])} vs {f3(d3['mean_spearman_fit_a'])}); r_m better in {d3['sign_test']['r_better']}, P(E_m) better in {d3['sign_test']['a_better']}, ties {d3['sign_test']['ties']} (two-sided sign test p = {d3['sign_test']['p_two_sided']:.2g}); median paired difference {d3['median_diff_r_minus_a']:+.3f} with held-out bootstrap CI [{d3['median_diff_r_minus_a_boot_ci'][0]:+.3f}, {d3['median_diff_r_minus_a_boot_ci'][1]:+.3f}]. Both price-blind rankings are weak predictors of the realised *efficiency* change because it is dominated by the candidate's price d_m and by eta(S) b: the cost-aware rankings reach medians N_S {f3(d3['median_spearman_fit_N'])}, N_S with the independence proxy {f3(d3['median_spearman_fit_Nindep'])}, P(E_m)/d_m {f3(d3['median_spearman_fit_apd'])} (`fig_spearman_rankings.pdf`). P(E_m) is *negatively* correlated with the realised gain because the highest-quality candidates (claude-v1, claude-v2) cost nearly as much as gpt-4.\n",
          f"X2's d3 target (held-out r_m instead of the gain): fit r_m {f3(d_['d3_r_target']['median_spearman_fit_r'])} vs fit P(E_m) {f3(d_['d3_r_target']['median_spearman_fit_a'])}; r_m better in {d_['d3_r_target']['sign_test']['r_better']}, P(E_m) better in {d_['d3_r_target']['sign_test']['a_better']} (p = {d_['d3_r_target']['sign_test']['p_two_sided']:.2g}). This says r_m is estimable from the fit split and that marginal quality is a poor proxy for it.\n",
          "Top-partner comparison (X2's d2; realised held-out relative gain of the partner ranked first by each fit-split statistic):\n", "| fit-split ranking | median rel. gain of its top partner | gain > 0 | top = held-out oracle |", "|---|---|---|---|"]
    for nm, lab in zip(RANKINGS, ["P(E_m) (individual quality)", "r_m(S) (residual coverage)", "P(E_m)/d_m (quality per dollar)", "N_S with independence proxy", "N_S (eq:marginal)", "held-out oracle"]):
        L.append(f"| {lab} | {pct(d_[f'median_relgain_top_{nm}'])} | {pct(d_[f'frac_top_{nm}_gain_positive'], 0)} | {pct(d_[f'frac_top_{nm}_is_oracle'], 0)} |")
    for x_, y_ in PAIRINGS:
        q = d_[f"{x_}_vs_{y_}"]
        L.append(f"\n* {x_} vs {y_}: same top in {pct(q['frac_same_top'], 0)}; where they differ ({q['n_differing']}), {x_} wins {q['n_x_wins']} (paired CI {q['n_x_wins_ci']}), {y_} wins {q['n_y_wins']} (CI {q['n_y_wins_ci']}); mean rel-gain difference {(q['mean_diff_relgain_x_minus_y_on_differing'] or 0)*100:+.1f} pts.")
    # ---- C3
    f_ = fs[K0]; e_ = es[K0]
    L += [f"\n### 3.3 C3 -- Frechet range of r_m and the excess cost of a marginal-only selector (rem:frechet), theta = {th}\n",
          f"**Interval (IDENTITY on the same sample).** For S = {{m0}}, m0 in the six primaries and every usable m: {f_['n_pairs']} pairs in {f_['n_contexts']} contexts, realised r_m inside [max{{0, p_m - p_0}}, min{{p_m, 1 - p_0}}] in {pct(f_['frac_inside_same_sample'])} (and in {pct(f_['mean_boot_inside_frac'])} of bootstrap replicates). Position (r - l)/(u - l): median {f3(f_['position_median'], 2)}, 10-90% [{f_['position_q10_q90'][0]:.2f}, {f_['position_q10_q90'][1]:.2f}]; within 1% of the lower bound in {pct(f_['position_frac_at_lower_bound_1pct'], 1)}, of the upper bound in {pct(f_['position_frac_at_upper_bound_1pct'], 1)}. The independence proxy sits at median position {f3(f_['position_indep_proxy_median'], 2)}; the realised r_m is below it in {pct(f_['frac_r_below_indep_proxy'], 0)} of pairs (positively correlated errors). Median interval width {f3(f_['width_median'])} (10-90% [{f_['width_q10_q90'][0]:.3f}, {f_['width_q10_q90'][1]:.3f}]) -- the marginals leave most of r_m unidentified. `fig_frechet_intervals.pdf` shows the 8 families for the four protocol primaries.",
          f"**Out-of-sample diagnostic.** The held-out r_m lies inside the interval computed from the *fit-split* marginals in {pct(f_['oos_frac_heldout_r_inside_fit_bounds'])} of {f_['oos_n_pairs']} pairs (its held-out CI overlaps the fit interval in {pct(f_['oos_frac_heldout_ci_overlaps_fit_interval'])}); the misses are sampling noise in the small MMLU subsets (families: {pct(fs[f'theta={th}|families']['oos_frac_heldout_r_inside_fit_bounds'])} of {fs[f'theta={th}|families']['oos_n_pairs']}).",
          f"**Excess cost of the marginal-only selector (IN-SAMPLE).** For every primary m0 and every pair of usable candidates with |P(E_m1) - P(E_m2)| < {EQ_TOL} ({e_['n_triples']} triples in {e_['n_contexts']} contexts), the excess is evaluated in the proposition's accounting (perfect fallback of cost b = b({{m0}}) = gpt-4's mean cost on m0's failure set, equal candidate costs, unit values): the minimax marginal-only selector chooses m1 with probability q = x/(x+y), x = (u_1 - l_2)^+, y = (u_2 - l_1)^+, and its realised excess over the informed choice is b[max{{r_1, r_2}} - q r_1 - (1-q) r_2]. Excess <= exact bound b xy/(x+y) in {pct(e_['frac_within_exact_bound'])} and <= the equal-marginal formula (b/2)[min{{p, 1-p_0}} - max{{0, p-p_0}}] at p = (p_1+p_2)/2 in {pct(e_['frac_within_equal_marginal_bound_pbar'])} -- implied by the interval identity, since the bound is the worst case over the intervals. Median excess/bound {f3(e_['median_ratio_excess_over_exact_bound'], 2)} (10-90% [{e_['q10_q90_ratio'][0]:.2f}, {e_['q10_q90_ratio'][1]:.2f}]); excess exactly 0 in {pct(e_['frac_excess_zero'], 0)} of triples; median excess {usd(e_['median_excess_usd'])} vs median bound {usd(e_['median_bound_usd'])} per prompt. **Width dependence:** Spearman(excess/b, width) pooled over triples {e_['spearman_excess_prob_vs_width_pooled']:+.3f} [CI {e_['spearman_excess_prob_vs_width_pooled_ci'][0]:+.3f}, {e_['spearman_excess_prob_vs_width_pooled_ci'][1]:+.3f}]; Spearman(excess $, bound $) {e_['spearman_excess_usd_vs_bound_usd_pooled']:+.3f} [CI {e_['spearman_excess_usd_vs_bound_usd_pooled_ci'][0]:+.3f}, {e_['spearman_excess_usd_vs_bound_usd_pooled_ci'][1]:+.3f}]; within-context Spearman median {f3(e_['within_context_spearman']['median'], 2)} over {e_['within_context_spearman']['n_contexts_ge5']} contexts with >= 5 triples, positive in {pct(e_['within_context_spearman']['frac_positive'], 0)} (families only: pooled {es[f'theta={th}|families']['spearman_excess_prob_vs_width_pooled']:+.3f}, within-context median {f3(es[f'theta={th}|families']['within_context_spearman']['median'], 2)}). The realised excess grows with the width but stays far below the worst case: the observed joint laws are nearly exchangeable (both candidates' successes concentrate on the same easy prompts), so r_1 and r_2 are close even when their Frechet ranges are wide (`fig_excess_vs_width.pdf`).\n"]
    s_ = so_[K0]
    L.append(f"**Out-of-sample selector test (C5, prop:marginals content).** Same triples defined on the fit split ({s_['n_triples']} in {s_['n_contexts']} contexts), realised on the held-out split: mean excess fallback cost of the marginal-only minimax selector {usd(s_['mean_excess_marginal_usd'])}, of the r_m-informed selector (argmax fit-split r_m) {usd(s_['mean_excess_informed_usd'])} (ratio {f3(s_['ratio_informed_over_marginal_mean_excess'], 2)}), of a uniform coin {usd(s_['mean_excess_random_usd'])}; mean difference marginal - informed {usd(s_['mean_diff_marginal_minus_informed_usd'])} [CI {usd(s_['mean_diff_ci'][0])}, {usd(s_['mean_diff_ci'][1])}]; informed better in {s_['sign_test']['informed_better']}, marginal better in {s_['sign_test']['marginal_better']}, ties {s_['sign_test']['ties']} (p = {s_['sign_test']['p_two_sided']:.2g}); the fit-split informed pick is the held-out best partner in {pct(s_['frac_informed_pick_is_heldout_best'], 0)}. Paired outcomes carry real but modest information about which equal-marginal partner to add.\n")
    # ---- C4 hypothesis violations
    b_ = sb[K0]
    L += ["## 4. Hypothesis violations measured (C4)\n",
          f"* **Common v_m (thm:marginal hypothesis).** Over {b_['n']} non-trivial (S, m) cases in {b_['n_contexts']} contexts (full sample, theta = {th}), the CI of v_m(S+m) - v_m(S) excludes 0 in {pct(b_['vm_drift']['frac_ci_excl0'], 0)} (violation frequency); the drift is negative in {pct(b_['vm_drift']['frac_negative'], 0)} with median {b_['vm_drift']['median']:+.4f} ({pct(b_['vm_drift']['median_rel'])} of v_m(S)), 10-90% [{pct(b_['vm_drift']['q10_q90_rel'][0])}, {pct(b_['vm_drift']['q10_q90_rel'][1])}]. Mechanism (X2 b1): the prompts a new member covers are the ones gpt-4 also tends to solve, so gpt-4's mean score on the *remaining* failures falls. Families only: {pct(sb[f'theta={th}|families']['vm_drift']['frac_ci_excl0'], 0)} of {sb[f'theta={th}|families']['n']}; MMLU subsets: {pct(sb[f'theta={th}|mmlu_subsets']['vm_drift']['frac_ci_excl0'], 0)} of {sb[f'theta={th}|mmlu_subsets']['n']} (smaller samples, wider CIs).",
          f"* **Common b.** CI excludes 0 in {pct(b_['b_drift']['frac_ci_excl0'], 0)} (median {pct(b_['b_drift']['median_rel'])}); families {pct(sb[f'theta={th}|families']['b_drift']['frac_ci_excl0'], 0)}, `overall` {pct(sb[f'theta={th}|overall']['b_drift']['frac_ci_excl0'], 0)} (pooling heterogeneous per-family gpt-4 prices makes b move with the failure set's family mix). **Common v_h.** CI excludes 0 in {pct(b_['vh_drift']['frac_ci_excl0'], 0)} (max |drift| {f3(b_['vh_drift']['max_abs'])}; v_h = 1 by construction in binary contexts).",
          f"* **Consequence for the sign rule (in-sample b4).** Sign agreement {pct(b_['sign_agreement_all'])} on all cases, {pct(b_['sign_agreement_realised_ci_decided'])} on {b_['n_realised_ci_decided']} realised-CI-decided, {pct(b_['sign_agreement_both_ci_decided'])} on {b_['n_both_ci_decided']} both-CI-decided (baselines on realised-CI-decided: independence proxy {pct(b_['baseline_N_indep_realised_ci_decided'])}, always negative {pct(b_['baseline_always_negative_realised_ci_decided'])}). Magnitude: median |relative error| of the predicted change {pct(b_['median_rel_err_dpred'])} (within 25% in {pct(b_['frac_rel_err_below_25pct'], 0)}); substituting S+m's v_m alone reduces it to {pct(b_['median_rel_err_substitute_vm2'])}, S+m's b alone to {pct(b_['median_rel_err_substitute_b2'])}. The violated hypothesis biases the magnitude, rarely the sign.",
          f"* **Error correlation (X2 c2).** P(E_i^c & E_j^c) - P(E_i^c) P(E_j^c) > 0 in {pct(ec[K0]['frac_positive'], 0)} of {ec[K0]['n_pairs']} usable pairs (CI > 0 in {pct(ec[K0]['frac_positive_ci'], 0)}, CI < 0 in {pct(ec[K0]['frac_negative_ci'])}; median excess {f3(ec[K0]['median_excess'])}); families only {pct(ec[f'theta={th}|families']['frac_positive'], 0)} of {ec[f'theta={th}|families']['n_pairs']}. RouterBench is less uniformly positively correlated than CodeRouterBench (99%): the significantly negative pairs involve the weakest models (mistral-7b, code-llama-34b, WizardLM-13B) against the Claude family or gpt-3.5 on MMLU / HellaSwag, e.g. {'; '.join(A['negative_error_correlation_examples'][:4])}. This is exactly the regime in which r_m carries information the marginals do not (C3a: median position 0.32 vs independence 0.58).\n"]
    # ---- audit
    L += ["## 5. Audit (independent recomputation with pandas, no shared numpy machinery)\n",
          f"* **C1** held-out sign agreement recomputed case by case from the raw frame (fit-split N_S via explicit conditional means, held-out change via two ratio-of-sums): all cases {pct(A['C1_recomputed']['agreement_all'], 2)} vs main run {pct(A['C1_recomputed']['main_run']['agreement_all'], 2)}; CI-decided {pct(A['C1_recomputed']['agreement_ci_decided'], 2)} vs {pct(A['C1_recomputed']['main_run']['agreement_ci_decided'], 2)} on {A['C1_recomputed']['n_ci_decided_reused_flags']} cases (the CI-decided flags are reused from the main run); max |N_S(fit) recomputed - main| = {A['C1_recomputed']['max_abs_diff_N_fit_vs_main']:.1e}. The CI-decided flags themselves were re-derived for `hellaswag` with an explicit resampling loop (same RNG stream): {A['C1_ci_flags_recomputed_hellaswag']['n_flags_matching']}/{A['C1_ci_flags_recomputed_hellaswag']['n_cases']} flags match.",
          f"* **C3a** Frechet inclusion recomputed from the raw frame: {A['C3a_recomputed']['n_pairs']} pairs (main {A['C3a_recomputed']['main_run']['n_pairs']}), inside {pct(A['C3a_recomputed']['frac_inside'], 2)} (main {pct(A['C3a_recomputed']['main_run']['frac_inside'], 2)}), median position {f3(A['C3a_recomputed']['position_median'])} (main {f3(A['C3a_recomputed']['main_run']['position_median'])}).",
          f"* **C3b** equal-marginal triples recomputed: {A['C3b_recomputed']['n_triples']} (main {A['C3b_recomputed']['main_run']['n_triples']}), within the exact bound {pct(A['C3b_recomputed']['frac_within_exact_bound'], 2)} (main {pct(A['C3b_recomputed']['main_run']['frac_within_exact_bound'], 2)}), median excess/bound {f3(A['C3b_recomputed']['median_ratio'])} (main {f3(A['C3b_recomputed']['main_run']['median_ratio'])}), pooled Spearman(excess/b, width) {f3(A['C3b_recomputed']['spearman_excess_prob_vs_width'])} (main {f3(A['C3b_recomputed']['main_run']['spearman_excess_prob_vs_width'])}).",
          f"* **C2** per-portfolio Spearman recomputed: median fit r_m {f3(A['C2_recomputed']['median_spearman_fit_r_vs_gain'])} vs fit P(E_m) {f3(A['C2_recomputed']['median_spearman_fit_a_vs_gain'])} (main {f3(A['C2_recomputed']['main_run']['median_spearman_fit_r'])} vs {f3(A['C2_recomputed']['main_run']['median_spearman_fit_a'])}); r better {A['C2_recomputed']['r_better']} / a better {A['C2_recomputed']['a_better']} (main {A['C2_recomputed']['main_run']['sign_test']['r_better']} / {A['C2_recomputed']['main_run']['sign_test']['a_better']}).",
          f"* **Identities** (code checks): r_m = p(S+m) - p(S) max |diff| {R['identity']['max_abs_r_identity']:.1e}; eq:homogeneous vs ratio of sums {R['identity']['max_abs_eta_homogeneous']:.1e}; eq:app-marginal-identity {R['identity']['max_abs_marginal_identity']:.1e}; substituting S+m's (v_h, v_m, b) recovers the realised change to {R['identity']['max_abs_all2_equals_real']:.1e} ({R['identity']['n_cases_checked']} cases).\n"]
    # ---- limitations
    L += ["## 6. Limitations and deviations\n",
          "* RouterBench's grader is the benchmark's own scorer (exact match / execution / GPT-4 judge for MT-Bench), so the exact selecting verifier is an oracle: r_m(S) here is the *coverage* content of thm:marginal, not the verifier's selection probability. Costs are the benchmark's recorded per-prompt dollars (no latency), so only the dollar basis is reported.",
          f"* C3a and the bound part of C3b are identities of the empirical joint law (labelled IDENTITY / IN-SAMPLE); the empirical content is the *position* of r_m inside the range, the width dependence of the excess and the out-of-sample tests C1, C2, C5. The excess in C3b is computed in the proposition's stylised accounting (perfect fallback at b({{m0}}), equal candidate costs, unit values); the realised dollar difference between pairing with m1 or m2 additionally involves d_1 - d_2 and the drift of b, which C1 covers.",
          f"* Pooled Spearman CIs are percentile intervals of the statistic recomputed on task-resampled data; with few triples (theta = 1 rows: 3 contexts) they can exclude the point estimate. Within-context correlations are weak (median {f3(e_['within_context_spearman']['median'], 2)}) because equal-marginal triples inside one context share p_0 and hence have similar widths.",
          f"* {M['n_boot']} bootstrap resamples (CONVENTIONS allows 500 for heavy computations; three bootstraps per context). The 70/30 split is one draw (seed {M['seed']}); no cross-fitting. MMLU subsets have 100-1,534 prompts, so their held-out splits (30-460 prompts) decide fewer cases by CI and their fit-split intervals miss the held-out r_m more often than the families'.",
          f"* theta = 1.0 duplicates theta = {th} in every binary-score context and is therefore reported only for the {sum(1 for n, k in CONTEXTS if not BINARY[n])} fractional-score contexts; the equal-marginal tolerance {EQ_TOL} and X2's usability / non-triviality filters are conventions, not results. The VLM-ExecRouterBench replication of the protocol (P2) was not attempted (data not available locally)."]
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write("\n".join(L) + "\n")

def write_summary(R):
    Cl = R["claims_by_key"]; th = THETA_MAIN; K0 = f"theta={th}|all_non_overall"; d_ = R["oos"]["summary_d"][K0]; f_ = R["frechet"]["interval_summary"][K0]; e_ = R["frechet"]["excess_summary"][K0]
    b_ = R["in_sample"]["summary_b4_and_drift"][K0]; ec = R["in_sample"]["error_correlation_summary"][K0]; s_ = R["oos"]["selector_excess_summary"][K0]; d3 = d_["d3_gain_target"]
    L = [f"# X11 summary for the paper (RouterBench 0-shot, 36,497 prompts x 11 models, reference gpt-4-1106-preview, 8 families + {len(MMLU_SUBSETS)} MMLU subsets, 70/30 stratified split, seed {SEED}; dollars)\n",
         f"1. eq:marginal sign rule OUT-OF-SAMPLE (fit 70% -> held-out 30%, theta = {th}): {pct(d_['sign_agreement_ci_decided']['theorem_N'])} agreement on {d_['n_sign_ci_decided']} CI-decided (S, m) cases ({pct(d_['sign_agreement_all']['theorem_N'])} on all {d_['n_sign_cases']}), vs independence proxy (1-p(S))P(E_m) {pct(d_['sign_agreement_ci_decided']['N_indep'])}, marginal-only P(E_m) >= tau {pct(d_['sign_agreement_ci_decided']['marginal_quality_fitted_tau'])}, Frechet-upper-bound proxy {pct(d_['sign_agreement_ci_decided']['frechet_upper'])}, constant sign {pct(max(d_['sign_agreement_ci_decided']['always_negative'], d_['sign_agreement_ci_decided']['always_positive']))} -- supports thm:marginal (C1 {Cl[f'C1_oos_sign_theta{th}']['verdict']}; X2 on CodeRouterBench: 96.7-99.0%).",
         f"2. Ranking partners by fit-split r_m(S) vs by P(E_m), Spearman with the realised held-out gain over {d_['n_S']} portfolios: medians {f3(d3['median_spearman_fit_r'])} vs {f3(d3['median_spearman_fit_a'])} (r_m better in {d3['sign_test']['r_better']}, P(E_m) better in {d3['sign_test']['a_better']}, p = {d3['sign_test']['p_two_sided']:.1g}) -- OUT-OF-SAMPLE, C2 {Cl[f'C2_oos_spearman_theta{th}']['verdict']}; but both price-blind rankings are weak; the full criterion N_S reaches {f3(d3['median_spearman_fit_N'])} and picks the held-out-optimal partner in {pct(d_['frac_top_theorem_N_is_oracle'], 0)} of portfolios (P(E_m): {pct(d_['frac_top_quality_a_is_oracle'], 0)}; median realised gain of the top partner {pct(d_['median_relgain_top_theorem_N'])} vs {pct(d_['median_relgain_top_quality_a'])}).",
         f"3. Frechet range (rem:frechet), IDENTITY: realised r_m({{m0}}) inside [max{{0,p_m-p_0}}, min{{p_m,1-p_0}}] in {pct(f_['frac_inside_same_sample'], 0)} of {f_['n_pairs']} pairs; median position {f3(f_['position_median'], 2)} of the range (independence would give {f3(f_['position_indep_proxy_median'], 2)}); median width {f3(f_['width_median'], 2)} -- marginals leave most of r_m unidentified. Held-out r_m inside the fit-split range in {pct(f_['oos_frac_heldout_r_inside_fit_bounds'])} (OUT-OF-SAMPLE diagnostic).",
         f"4. Marginal-only selector's excess fallback cost for equal-marginal pairs ({e_['n_triples']} triples), IN-SAMPLE: <= the rem:frechet bound in {pct(e_['frac_within_exact_bound'], 0)} (implied by 3), median excess/bound {f3(e_['median_ratio_excess_over_exact_bound'], 2)}, grows with the range width: Spearman {e_['spearman_excess_prob_vs_width_pooled']:+.2f} [CI {e_['spearman_excess_prob_vs_width_pooled_ci'][0]:+.2f}, {e_['spearman_excess_prob_vs_width_pooled_ci'][1]:+.2f}] (C3b {Cl[f'C3b_excess_bound_theta{th}']['verdict']}); realised excess is far below the worst case because the joint laws are nearly exchangeable.",
         f"5. OUT-OF-SAMPLE selector (prop:marginals content): informed argmax-r_m selector's held-out excess is {pct(s_['ratio_informed_over_marginal_mean_excess'], 0)} of the minimax marginal-only selector's (mean difference {usd(s_['mean_diff_marginal_minus_informed_usd'])}/prompt, CI [{usd(s_['mean_diff_ci'][0])}, {usd(s_['mean_diff_ci'][1])}]; informed better in {s_['sign_test']['informed_better']} vs {s_['sign_test']['marginal_better']} of {s_['n_triples']} triples) -- C5 {Cl[f'C5_oos_selector_excess_theta{th}']['verdict']}.",
         f"6. Hypothesis diagnostic (C4, IN-SAMPLE): common-v_m violated (CI excludes 0) in {pct(b_['vm_drift']['frac_ci_excl0'], 0)} of {b_['n']} cases, median drift {pct(b_['vm_drift']['median_rel'])} of v_m(S); b drift {pct(b_['b_drift']['frac_ci_excl0'], 0)}; in-sample sign agreement nevertheless {pct(b_['sign_agreement_realised_ci_decided'])} on CI-decided cases; errors positively correlated in {pct(ec['frac_positive'], 0)} of {ec['n_pairs']} model pairs (CI < 0 in {pct(ec['frac_negative_ci'])}; CodeRouterBench: 99%).",
         f"7. eval_name grouping: mmlu-* (57), hellaswag, grade-school-math, arc-challenge, winogrande, mbpp, mt-bench (mtbench*), rag-other (the {len(FAMILY_MEMBERS['rag-other'])} remaining RAG/Chinese/summarisation sets). theta = 1 differs from theta = 0.5 only in the fractional-score contexts (GSM8K, MT-Bench, rag-other, overall): there C1 = {pct(R['oos']['summary_d'].get('theta=1.0|all_non_overall', {}).get('sign_agreement_ci_decided', {}).get('theorem_N'))} on {R['oos']['summary_d'].get('theta=1.0|all_non_overall', {}).get('n_sign_ci_decided')} CI-decided cases."]
    with open(os.path.join(OUT, "summary_for_paper.md"), "w") as f:
        f.write("\n".join(L) + "\n")
write_readme(R); write_summary(R)
print(f"README.md and summary_for_paper.md written; total runtime {time.time()-T0:.0f}s")

