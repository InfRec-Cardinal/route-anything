#!/usr/bin/env python3
"""
X10_escalation_chains -- Proposition prop:attempts (escalation inside a loop) on CodeRouterBench, OFFLINE part.

Chains m_1 -> ... -> m_K -> claude-opus-4-6 (terminal), K in {1,2,3}, over the 7 non-Opus primaries (7 + 42 + 210 = 259
ordered chains), oracle acceptance H_k = 1{score_{m_k} >= theta}, theta in {1, 0.5}.  States x in {global, 9 dimensions,
21 dimension x source cells}.  On the probing split (7,080 tasks) each state gives, per attempt k, the post-failure
conditional success p_k = P(H_k = 1 | H_1 = ... = H_{k-1} = 0), the attempt cost a_k, the state-matched terminal
primitives b_k, v_b,k (Opus on the same failure set), W_lambda = lambda b + v_h - v_b, and by backward induction
Pi_k = max{J^direct_k, p_k v_h - lambda a_k + (1-p_k) Pi_{k+1}} with option value Delta_{k+1} = Pi_{k+1} - J^direct_{k+1}.
Rule (prop:attempts): attempt k iff p_k W_lambda + (1-p_k) Delta_{k+1} >= lambda a_k, lambda in {0.1, 1, 10, eta_B, 100}
per dollar (seconds basis: the same multiples of eta_B, in score per second, on tasks with all 8 latencies logged).
Realization on id_test (2,919 tasks; never used for fitting): per-segment Delta J_lambda = gain of making attempt k (then
following the probing plan; also the myopic 'then Opus') versus stopping at the state and dispatching Opus; paired task
bootstrap within the state's cell (1,000 resamples, seed 20260917, 95% percentile CIs).  Naive baseline: the same
backward induction with the marginal P(E_m) and unconditional primitives (the marginal-planned chain).
Variants: 'released' matrix as is, and 'bcb_allzero_excluded' (BigCodeBench tasks with score 0 for all 8 models are
grader failures -- CONVENTIONS Addendum 2 -- treated as ungraded).  Part (4): ACRouter's released OOD176 escalation
traces replayed against the OOD outcome matrix.  Fully offline; no API calls.  Runtime ~5 min on an M4.
"""
import os, json, time, itertools, math
from fractions import Fraction
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T0 = time.time()
SEED = 20260917
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
DATA = os.environ.get("ACROUTER_DATA", "data/agent-as-a-router/data")
ACR_OUT = os.environ.get("ACROUTER_OUTPUTS", os.path.join(os.path.dirname(DATA.rstrip("/")), "outputs_hf", "outputs"))
NBOOT = int(os.environ.get("X10_NBOOT", 1000))
KMAX = int(os.environ.get("X10_KMAX", 3))
THETAS = [1.0, 0.5]
LAM_MULT = {"0.1": 0.1, "1": 1.0, "10": 10.0, "eta_B": None, "100": 100.0}   # dollars; None -> eta_B of the configuration
STRONG = "claude-opus-4-6"
PRIMS = ["claude-sonnet-4-6", "gpt-5.4", "glm-5", "kimi-k2.5", "MiniMax-M2.7", "Qwen3-Max", "qwen3.5-plus"]
RULES = {"rule_stated": ("pred_stated", "real"), "exact_dp": ("pred_exact", "real"), "naive_marginal": ("pred_naive", "real"),
         "myopic_rule_target_then_opus": ("pred_myopic", "real0"), "naive_myopic_target_then_opus": ("pred_naive_myopic", "real0")}
BLUE, RED, GREEN, PURPLE, GOLD, GREY = "#23639b", "#ba5934", "#3a7c5e", "#7a5195", "#c9a227", "0.55"
LAMCOL = {"0.1": GREY, "1": GREEN, "10": BLUE, "eta_B": PURPLE, "100": RED}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"], "text.usetex": False,
                     "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.65, "lines.linewidth": 1.4, "pdf.fonttype": 42})
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
np.seterr(invalid="ignore", divide="ignore")   # empty realized states in some bootstrap resamples give nan; ci() drops them

# ------------------------------------------------------------------ data (loader copied from X3_threshold_voi/run.py)
df = pd.read_csv(f"{DATA}/coderouterbench/id_results_long.csv")
S = df.pivot(index="task_id", columns="model", values="score")
Cc = df.pivot(index="task_id", columns="model", values="cost_usd")
Lat = df.pivot(index="task_id", columns="model", values="latency_ms").loc[S.index] / 1000.0
meta = df.drop_duplicates("task_id").set_index("task_id")[["split", "dimension"]].loc[S.index]
meta["bench"] = meta.index.str.split("_").str[1]
meta["cell21"] = meta.dimension + "|" + meta.bench
meta["all"] = "ALL"
assert S.shape == (9999, 8) and not S.isna().any().any() and not Cc.isna().any().any()
assert (meta.split == "probing").sum() == 7080 and (meta.split == "id_test").sum() == 2919
MODELS = list(S.columns); col = {m: j for j, m in enumerate(MODELS)}
Yall = S.to_numpy(); Call = {"usd": Cc.to_numpy(), "sec": Lat.to_numpy()}
allzero = (S.max(axis=1) == 0).to_numpy(); is_bcb = (meta.bench == "bcb").to_numpy()
lat_ok = (Lat > 0).all(axis=1).to_numpy()
VARIANTS = {"released": np.ones(len(S), bool), "bcb_allzero_excluded": ~(allzero & is_bcb)}
BASES = {"usd": np.ones(len(S), bool), "sec": lat_ok}
PARTS = {"all": "all", "dim9": "dimension", "cell21": "cell21"}
CHAINS = [ch for K in range(1, KMAX + 1) for ch in itertools.permutations(PRIMS, K)]
azc = pd.Series(allzero, index=S.index).groupby(meta.cell21).mean()
DATA_NOTES = dict(n_allzero_tasks_total=int(allzero.sum()), n_allzero_bcb=int((allzero & is_bcb).sum()), n_bcb=int(is_bcb.sum()),
                  n_tasks_all_latencies_logged=int(lat_ok.sum()), latency_zero_rows_by_model={m: int((Lat[m] <= 0).sum()) for m in MODELS},
                  zero_cost_rows_qwen3max=int((df.cost_usd <= 0).sum()), allzero_fraction_by_cell={c: float(v) for c, v in azc.items()},
                  cells_fully_allzero=[c for c, v in azc.items() if v >= 0.99])

def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (float("nan"), float("nan")) if len(x) == 0 else tuple(float(v) for v in np.percentile(x, [2.5, 97.5]))
def fmean(x): return float(x.mean()) if len(x) else float("nan")
def boot_weights(rng, n):
    idx = rng.integers(0, n, size=(NBOOT, n)); W = np.zeros((NBOOT, n))
    np.add.at(W, (np.repeat(np.arange(NBOOT), n), idx.ravel()), 1.0); return W

# ------------------------------------------------------------------ primitives, backward induction, realization
def chain_arrays(Y, C, ids, chain, theta):
    idx = [col[m] for m in chain]
    Ym, Cm = Y[np.ix_(ids, idx)], C[np.ix_(ids, idx)]
    return Ym, Cm, Ym >= theta, Y[ids, col[STRONG]], C[ids, col[STRONG]]

def primitives(Ym, Cm, Hm, yB, cB):
    """Per attempt k (state = attempts 1..k-1 failed): post-failure conditional and marginal (unconditional) primitives."""
    n, K = Hm.shape; F = np.ones(n, bool); pr = []
    for k in range(K):
        H = Hm[:, k]; nF = int(F.sum()); nH = int((F & H).sum()); nHm = int(H.sum())
        pr.append(dict(n=nF, p=nH / nF if nF else float("nan"), p_marg=nHm / n, a=fmean(Cm[F, k]), a_marg=float(Cm[:, k].mean()),
                       v_h=fmean(Ym[F & H, k]) if nH else 0.0, v_h_marg=fmean(Ym[H, k]) if nHm else 0.0,
                       b=fmean(cB[F]), v_b=fmean(yB[F]), b_hit=fmean(cB[F & H]), v_b_hit=fmean(yB[F & H])))
        F = F & ~H
    term = dict(n=int(F.sum()), b=fmean(cB[F]), v_b=fmean(yB[F]))
    return pr, term, dict(b=float(cB.mean()), v_b=float(yB.mean()))

def backward_induction(ps, as_, vhs, bs, vbs, bT, vbT, lam):
    """prop:attempts.  bs[k], vbs[k] = terminal-path (Opus) cost/value at state k; bT, vbT after K failures.
    Per k: W, Delta_{k+1}, stated margin p W + (1-p) Delta - lam a, exact margin (attempt branch - J^direct_k), myopic margin (Delta := 0)."""
    K = len(ps); Jd = [vb - lam * b for vb, b in zip(vbs, bs)] + [vbT - lam * bT]
    Pi_next = Jd[K] if math.isfinite(Jd[K]) else 0.0; out = [None] * K
    for k in reversed(range(K)):
        p = ps[k]
        if math.isfinite(Jd[k + 1]): Delta = Pi_next - Jd[k + 1]; cont = Pi_next
        else: Delta = 0.0; cont = 0.0                      # empty next state: (1-p) = 0, the continuation is never reached
        att = p * vhs[k] - lam * as_[k] + (1 - p) * cont
        W = lam * bs[k] + vhs[k] - vbs[k]
        out[k] = dict(W=W, Delta=Delta, stated=p * W + (1 - p) * Delta - lam * as_[k], exact=att - Jd[k], myopic=p * W - lam * as_[k], Pi=max(Jd[k], att), Jd=Jd[k])
        Pi_next = out[k]["Pi"]
    return out

def plan_length(plan):
    L = 0
    for x in plan:
        if x: L += 1
        else: break
    return L

def realized_combos(Ym, Cm, Hm, yB, cB):
    """F[k] = realized state mask (attempts 0..k-1 failed); combos[(k, m)] = per-task (dV, dD) of 'attempt k..k+m, then Opus'
    minus 'Opus now', zero outside F[k]."""
    n, K = Hm.shape; F = [np.ones(n, bool)]
    for k in range(K - 1): F.append(F[-1] & ~Hm[:, k])
    combos = {}
    for k in range(K):
        for m in range(K - k):
            alive = F[k].copy(); V = np.zeros(n); D = np.zeros(n)
            for j in range(k, k + m + 1):
                D += alive * Cm[:, j]; V += alive * Hm[:, j] * Ym[:, j]; alive = alive & ~Hm[:, j]
            D += alive * cB; V += alive * yB
            combos[(k, m)] = ((V - yB) * F[k], (D - cB) * F[k])
    return F, combos

def policy_VD(Y, C, ids, chain, theta):
    """Whole-state per-task (V, D) of 'attempt the chain in order, stop at the first acceptance, Opus if all fail'."""
    yB, cB = Y[ids, col[STRONG]], C[ids, col[STRONG]]; alive = np.ones(len(ids), bool); V = np.zeros(len(ids)); D = np.zeros(len(ids)); Hany = np.zeros(len(ids), bool); YP = np.zeros(len(ids))
    for m in chain:
        y, c = Y[ids, col[m]], C[ids, col[m]]; H = y >= theta
        D += alive * c; V += alive * H * y; YP = np.where(alive & H, y, YP); Hany |= alive & H; alive = alive & ~H
    D += alive * cB; V += alive * yB
    return V, D, Hany, YP, yB, cB

# ------------------------------------------------------------------ C1: identities (sympy + exact rational arithmetic + DP vs brute force)
def identity_checks():
    import sympy as sp
    p, vh, vb, a, b, lam, Pi = sp.symbols("p v_h v_b a b lambda Pi_next", real=True)
    Jd = vb - lam * b; W = lam * b + vh - vb; Delta = Pi - Jd
    lhs = (p * vh - lam * a + (1 - p) * Pi) - Jd; rhs = p * W - lam * a + (1 - p) * Delta
    sym_ok = sp.simplify(lhs - rhs) == 0
    F = Fraction; one = F(1); Jd0 = one - one          # paper's instance: v_h = v_b = b = lambda = 1, (p1,a1) = (1/10,1/5), (p2,a2) = (9/10,1/4)
    att2 = F(9, 10) * one - F(1, 4) + F(1, 10) * Jd0; Pi2 = max(Jd0, att2); Delta2 = Pi2 - Jd0
    lhs1 = F(1, 10) * one + F(9, 10) * Delta2; own1 = F(1, 10) * one >= F(1, 5)
    inst = dict(Delta_2=str(Delta2), Delta_2_expected="13/20", lhs_1=str(lhs1), lhs_1_expected="137/200", rhs_1="1/5",
                first_attempt_fails_own_threshold=(not own1), first_attempt_optimal_with_option_value=bool(lhs1 >= F(1, 5)),
                ok=(Delta2 == F(13, 20) and lhs1 == F(137, 200) and not own1 and lhs1 >= F(1, 5)))
    return dict(symbolic_rule_identity=bool(sym_ok), paper_instance=inst)

def dp_vs_bruteforce(Y, C, ids, theta, lam):
    """Pi_1 from backward induction (state-matched primitives on a task set) equals the best 'attempt the first j, then Opus'
    policy, j = 0..K, evaluated on the same tasks; and J^direct_k = p_k J^direct_hit + (1-p_k) J^direct_{k+1} (state decomposition)."""
    worst = 0.0
    for chain in [("Qwen3-Max", "kimi-k2.5", "gpt-5.4"), ("glm-5", "claude-sonnet-4-6", "MiniMax-M2.7"), ("qwen3.5-plus", "Qwen3-Max")]:
        Ym, Cm, Hm, yB, cB = chain_arrays(Y, C, ids, chain, theta); pr, term, st = primitives(Ym, Cm, Hm, yB, cB)
        dp = backward_induction([r["p"] for r in pr], [r["a"] for r in pr], [r["v_h"] for r in pr], [r["b"] for r in pr], [r["v_b"] for r in pr], term["b"], term["v_b"], lam)
        vals = []
        for j in range(len(chain) + 1):
            V, D, *_ = policy_VD(Y, C, ids, chain[:j], theta); vals.append((V - lam * D).mean())
        worst = max(worst, abs(dp[0]["Pi"] - max(vals)))
        for k, r in enumerate(pr):
            nxt = pr[k + 1] if k + 1 < len(pr) else term
            if r["n"] and math.isfinite(nxt["v_b"]) and math.isfinite(r["v_b_hit"]):
                worst = max(worst, abs((r["v_b"] - lam * r["b"]) - (r["p"] * (r["v_b_hit"] - lam * r["b_hit"]) + (1 - r["p"]) * (nxt["v_b"] - lam * nxt["b"]))))
    return worst

# ------------------------------------------------------------------ main computation per configuration
COLS = ["variant", "basis", "theta", "part", "state", "chain", "K", "k", "model", "lam_name", "lam", "n_prob_state", "p_k", "p_marg", "a_k", "a_marg",
        "v_h", "v_h_marg", "b", "v_b", "b_hit", "v_b_hit", "W", "Delta_next", "W_naive", "Delta_naive_next", "pred_stated", "pred_exact", "pred_myopic",
        "pred_naive", "pred_naive_myopic", "plan_attempt", "plan_naive_attempt", "m_cont", "n_test_state", "real_dJ", "real_lo", "real_hi", "real_decided",
        "real0_dJ", "real0_lo", "real0_hi", "real0_decided", "real_dV", "real_dD", "degenerate"]
BEST_COLS = ["variant", "basis", "theta", "part", "state", "lam_name", "lam", "n_prob", "n_test", "best_chain", "L_star", "Pi_star", "best_naive_chain", "L_naive",
             "Pi_naive_pred", "Pi_naive_under_cond", "J_opus_test", "J_star_test", "J_naive_test", "J_star_minus_naive", "J_star_minus_naive_lo", "J_star_minus_naive_hi",
             "J_star_minus_opus", "J_star_minus_opus_lo", "J_star_minus_opus_hi", "n_chains_tied_star"]

class AgreeAcc:
    """Task-resampling distribution of sign-agreement rates: for each decided row, agree_b = 1{sign(pred) == sign(realized_b)}."""
    def __init__(self): self.d = {}
    def add(self, key, pred, boot, decided, degenerate):
        if not decided or not np.isfinite(pred): return
        ok = np.isfinite(boot); ag = ((pred >= 0) == (boot >= 0)) & ok
        for sub in (["all"] + ([] if degenerate else ["nondeg"])):
            e = self.d.setdefault((key, sub), [np.zeros(NBOOT), np.zeros(NBOOT)]); e[0] += ag; e[1] += ok
    def rate(self, key, sub="all"):
        e = self.d.get((key, sub)); return None if e is None else e[0] / e[1]

def run_config(variant, basis, cfg_index, ACC):
    U = VARIANTS[variant] & BASES[basis]; Y = Yall[U]; C = Call[basis][U]; mU = meta[U]
    is_prob = (mU.split == "probing").to_numpy(); is_test = (mU.split == "id_test").to_numpy()
    Yp, Cp, Yt, Ct = Y[is_prob], C[is_prob], Y[is_test], C[is_test]; mp, mt = mU[is_prob], mU[is_test]
    eta_B = float(Yp[:, col[STRONG]].mean() / Cp[:, col[STRONG]].mean())
    eta_B_usd = float(Yp[:, col[STRONG]].mean() / Call["usd"][U][is_prob][:, col[STRONG]].mean())
    scale = 1.0 if basis == "usd" else eta_B / eta_B_usd
    lams = {k: (eta_B if v is None else v * scale) for k, v in LAM_MULT.items()}
    states = [(part, c) for part, colname in PARTS.items() for c in sorted(mU[colname].unique())]
    rng = np.random.default_rng([SEED, cfg_index])
    rows, best_rows, c2_rows, decomp, curves = [], [], [], {}, {}
    for si, (part, cname) in enumerate(states):
        ip = np.flatnonzero((mp[PARTS[part]] == cname).to_numpy()); it = np.flatnonzero((mt[PARTS[part]] == cname).to_numpy())
        if len(ip) == 0 or len(it) == 0: continue
        Wt = boot_weights(rng, len(it)); Wp = boot_weights(rng, len(ip))
        for theta in THETAS:
            # ---- C2: post-failure conditional vs marginal for every ordered pair (probing; paired bootstrap)
            Hp = Yp[np.ix_(ip, [col[m] for m in PRIMS])] >= theta; Wsum = Wp @ np.ones(len(ip)); WH = Wp @ Hp.astype(float)
            for i, j in itertools.permutations(range(len(PRIMS)), 2):
                Fi = ~Hp[:, i]; nF = int(Fi.sum())
                if nF == 0: continue
                pc = float((Fi & Hp[:, j]).sum() / nF); pm = float(Hp[:, j].mean())
                bc = (Wp @ (Fi & Hp[:, j]).astype(float)) / (Wp @ Fi.astype(float)); bm = WH[:, j] / Wsum; lo, hi = ci(bc - bm)
                c2_rows.append(dict(variant=variant, basis=basis, theta=theta, part=part, state=cname, fail=PRIMS[i], next=PRIMS[j], n_state=nF,
                                    p_cond=pc, p_marg=pm, diff=pc - pm, diff_lo=lo, diff_hi=hi, ratio=pc / pm if pm > 0 else float("nan")))
            # ---- chains
            dp_cache = {}
            for chain in CHAINS:
                Ym, Cm, Hm, yB, cB = chain_arrays(Yp, Cp, ip, chain, theta); pr, term, st = primitives(Ym, Cm, Hm, yB, cB); K = len(chain)
                tYm, tCm, tHm, tyB, tcB = chain_arrays(Yt, Ct, it, chain, theta); F, combos = realized_combos(tYm, tCm, tHm, tyB, tcB)
                Mcols = [f.astype(float) for f in F] + [arr for k in range(K) for m in range(K - k) for arr in combos[(k, m)]]
                G = Wt @ np.column_stack(Mcols); nF = [int(f.sum()) for f in F]
                gpos = {}; c0 = K
                for k in range(K):
                    for m in range(K - k): gpos[(k, m)] = c0; c0 += 2
                for lname, lam in lams.items():
                    dp = backward_induction([r["p"] for r in pr], [r["a"] for r in pr], [r["v_h"] for r in pr], [r["b"] for r in pr], [r["v_b"] for r in pr], term["b"], term["v_b"], lam)
                    nv = backward_induction([r["p_marg"] for r in pr], [r["a_marg"] for r in pr], [r["v_h_marg"] for r in pr], [st["b"]] * K, [st["v_b"]] * K, st["b"], st["v_b"], lam)
                    plan = [d["exact"] >= 0 for d in dp]; nplan = [d["exact"] >= 0 for d in nv]
                    dp_cache[(chain, lname)] = (dp[0]["Pi"], plan_length(plan), nv[0]["Pi"], plan_length(nplan))
                    for k in range(K):
                        r, d, nvk = pr[k], dp[k], nv[k]; m = 0
                        for j in range(k + 1, K):
                            if plan[j]: m += 1
                            else: break
                        degenerate = bool(r["n"] == 0 or not (r["p"] > 0))
                        rec = [variant, basis, theta, part, cname, "->".join(chain), K, k + 1, chain[k], lname, lam, r["n"], r["p"], r["p_marg"], r["a"], r["a_marg"],
                               r["v_h"], r["v_h_marg"], r["b"], r["v_b"], r["b_hit"], r["v_b_hit"], d["W"], d["Delta"], nvk["W"], nvk["Delta"], d["stated"], d["exact"], d["myopic"],
                               nvk["stated"], nvk["myopic"], plan[k], nplan[k], m]
                        if nF[k] > 0:
                            dV, dD = combos[(k, m)]; dV0, dD0 = combos[(k, 0)]; denom = G[:, k]; g = gpos[(k, m)]; g0 = gpos[(k, 0)]
                            bj = (G[:, g] - lam * G[:, g + 1]) / denom; bj0 = (G[:, g0] - lam * G[:, g0 + 1]) / denom
                            lo, hi = ci(bj); lo0, hi0 = ci(bj0); dec = bool(np.isfinite(lo) and not (lo <= 0 <= hi)); dec0 = bool(np.isfinite(lo0) and not (lo0 <= 0 <= hi0))
                            rj = float((dV.sum() - lam * dD.sum()) / nF[k]); rj0 = float((dV0.sum() - lam * dD0.sum()) / nF[k])
                            rec += [nF[k], rj, lo, hi, dec, rj0, lo0, hi0, dec0, float(dV.sum() / nF[k]), float(dD.sum() / nF[k])]
                            preds = dict(pred_stated=d["stated"], pred_exact=d["exact"], pred_naive=nvk["stated"], pred_myopic=d["myopic"], pred_naive_myopic=nvk["myopic"])
                            if np.isfinite(d["stated"]):          # all rules are scored on the same rows: the probing state must be non-empty
                                for rule, (pc_, which) in RULES.items():
                                    ACC.add((variant, basis, theta, part, lname, rule), preds[pc_], bj if which == "real" else bj0, dec if which == "real" else dec0, degenerate)
                                    ACC.add((variant, basis, theta, part, "all_lambda", rule), preds[pc_], bj if which == "real" else bj0, dec if which == "real" else dec0, degenerate)
                        else:
                            rec += [0, np.nan, np.nan, np.nan, False, np.nan, np.nan, np.nan, False, np.nan, np.nan]
                        rec.append(degenerate); rows.append(rec)
            # ---- C4: optimal chain (by Pi_1) versus the marginal-planned chain, per lambda; realized on id_test
            for lname, lam in lams.items():
                Pis = {ch: dp_cache[(ch, lname)][0] for ch in CHAINS}; Pin = {ch: dp_cache[(ch, lname)][2] for ch in CHAINS}
                best = max(CHAINS, key=lambda ch: (Pis[ch], -len(ch))); bestn = max(CHAINS, key=lambda ch: (Pin[ch], -len(ch)))
                Ls, Ln = dp_cache[(best, lname)][1], dp_cache[(bestn, lname)][3]; ties = int(sum(abs(Pis[ch] - Pis[best]) < 1e-12 for ch in CHAINS))
                V1, D1, *_ = policy_VD(Yt, Ct, it, best[:Ls], theta); V2, D2, *_ = policy_VD(Yt, Ct, it, bestn[:Ln], theta)
                J1, J2 = V1 - lam * D1, V2 - lam * D2; JB = Yt[it, col[STRONG]] - lam * Ct[it, col[STRONG]]
                b12 = (Wt @ (J1 - J2)) / len(it); b1B = (Wt @ (J1 - JB)) / len(it)
                best_rows.append([variant, basis, theta, part, cname, lname, lam, len(ip), len(it), "->".join(best), Ls, Pis[best], "->".join(bestn), Ln,
                                  Pin[bestn], dp_cache[(bestn, lname)][0], float(JB.mean()), float(J1.mean()), float(J2.mean()), float((J1 - J2).mean()), *ci(b12),
                                  float((J1 - JB).mean()), *ci(b1B), ties])
                if part == "all" and lname == "eta_B":
                    for tag, ch, L in [("optimal_chain", best, Ls), ("marginal_planned_chain", bestn, Ln)]:
                        V, D, Hany, YP, yB, cB = policy_VD(Yt, Ct, it, ch[:L], theta); A = D - (~Hany) * cB
                        eta = float(V.sum() / D.sum()); etaB = float(yB.sum() / cB.sum()); gain = V - eta_B * D - (yB - eta_B * cB)
                        bV = (Wt @ V) / len(it); floor = {str(e): float((bV >= yB.mean() - e).mean()) for e in [0, 0.01, 0.02, 0.05]}
                        decomp[f"theta{theta}_{tag}"] = dict(chain="->".join(ch[:L]), L=L, n_test=len(it), eta=eta, eta_B_test=etaB, eta_over_eta_B=eta / etaB,
                            net_gain_E_V_minus_etaB_D_relative_to_opus=float(gain.mean()), net_gain_CI=ci((Wt @ gain) / len(it)),
                            avoided_reference_work_etaB_E_H_CB=float(eta_B * (Hany * cB).mean()), attempt_cost_etaB_E_A=float(eta_B * A.mean()),
                            accepted_quality_change_E_H_YP_minus_YB=float((Hany * (YP - yB)).mean()), recovery_cost_extra_fallback=0.0,
                            decomposition_residual=float(abs(gain.mean() - ((Hany * (YP - yB)).mean() + eta_B * (Hany * cB).mean() - eta_B * A.mean()))),
                            E_V=float(V.mean()), vbar_opus=float(yB.mean()), quality_floor_satisfaction_rate=floor, erroneous_acceptance_e=float((Hany & (YP < yB)).mean()), P_any_hit=float(Hany.mean()))
            # ---- curves for figure 1 (global state): conditional vs marginal success by attempt index over all chains
            if part == "all":
                pk = {k: [] for k in range(1, KMAX + 1)}; pm = {k: [] for k in range(1, KMAX + 1)}
                for chain in CHAINS:
                    if len(chain) < KMAX: continue
                    Ym, Cm, Hm, yB, cB = chain_arrays(Yp, Cp, ip, chain, theta); pr, _, _ = primitives(Ym, Cm, Hm, yB, cB)
                    for k, r in enumerate(pr, 1): pk[k].append(r["p"]); pm[k].append(r["p_marg"])
                curves[f"theta{theta}"] = {"k": list(pk), "p_cond_mean": [float(np.mean(pk[k])) for k in pk], "p_cond_q10": [float(np.quantile(pk[k], .1)) for k in pk],
                                           "p_cond_q90": [float(np.quantile(pk[k], .9)) for k in pk], "p_marg_mean": [float(np.mean(pm[k])) for k in pm],
                                           "p_marg_q10": [float(np.quantile(pm[k], .1)) for k in pm], "p_marg_q90": [float(np.quantile(pm[k], .9)) for k in pm], "n_chains": len(pk[1])}
        if si % 10 == 0: log(f"{variant}/{basis}: state {si+1}/{len(states)} done, rows so far {len(rows)}")
    R = pd.DataFrame(rows, columns=COLS); Bst = pd.DataFrame(best_rows, columns=BEST_COLS); C2 = pd.DataFrame(c2_rows)
    return dict(eta_B=eta_B, eta_B_usd_same_universe=eta_B_usd, lambdas=lams, n_prob=int(is_prob.sum()), n_test=int(is_test.sum()), n_states=len(states)), R, Bst, C2, decomp, curves

# ------------------------------------------------------------------ summaries
def agreement(sub, pred_col, which, acc_key=None, ACC=None):
    real_col, dec_col = ("real_dJ", "real_decided") if which == "real" else ("real0_dJ", "real0_decided")
    ok = sub[np.isfinite(sub[pred_col]) & np.isfinite(sub[real_col])]; dec = ok[ok[dec_col]]
    out = dict(n_rows=int(len(sub)), n_decided=int(len(dec)))
    if len(dec) == 0: return out
    agree = (dec[pred_col] >= 0) == (dec[real_col] >= 0); nd = dec[~dec.degenerate]
    out.update(agreement=float(agree.mean()), n_pred_attempt=int((dec[pred_col] >= 0).sum()), n_real_attempt=int((dec[real_col] >= 0).sum()),
               baseline_always_attempt=float((dec[real_col] >= 0).mean()), baseline_never_attempt=float((dec[real_col] < 0).mean()),
               false_attempt=int(((dec[pred_col] >= 0) & (dec[real_col] < 0)).sum()), false_stop=int(((dec[pred_col] < 0) & (dec[real_col] >= 0)).sum()),
               n_decided_non_degenerate=int(len(nd)), agreement_non_degenerate=float(((nd[pred_col] >= 0) == (nd[real_col] >= 0)).mean()) if len(nd) else None,
               baseline_never_attempt_non_degenerate=float((nd[real_col] < 0).mean()) if len(nd) else None)
    if ACC is not None and acc_key is not None:
        r = ACC.rate(acc_key); rn = ACC.rate(acc_key, "nondeg")
        if r is not None: out["agreement_CI"] = ci(r)
        if rn is not None: out["agreement_non_degenerate_CI"] = ci(rn)
    return out

def paired_counts(sub, a, b):
    dec = sub[sub.real_decided & np.isfinite(sub[a]) & np.isfinite(sub[b])]
    if len(dec) == 0: return None
    ra = (dec[a] >= 0) == (dec.real_dJ >= 0); rb = (dec[b] >= 0) == (dec.real_dJ >= 0)
    return dict(n=int(len(dec)), a_right_b_wrong=int((ra & ~rb).sum()), b_right_a_wrong=int((~ra & rb).sum()), both_right=int((ra & rb).sum()), both_wrong=int((~ra & ~rb).sum()))

def summarize(R, Bst, C2, ACC):
    out = {}
    for (variant, basis, theta), sub in R.groupby(["variant", "basis", "theta"]):
        key = f"{variant}|{basis}|theta{theta}"; out[key] = {}
        for part in PARTS:
            sp_all = sub[sub.part == part]; sp_ = sp_all[np.isfinite(sp_all.pred_stated)]      # rows with an empty probing state have no prediction
            o = {"n_rows_probing_state_empty_excluded": int(len(sp_all) - len(sp_))}
            for lname in list(LAM_MULT) + ["all_lambda"]:
                s = sp_ if lname == "all_lambda" else sp_[sp_.lam_name == lname]
                o[lname] = {rule: agreement(s, pc_, which, (variant, basis, theta, part, lname, rule), ACC) for rule, (pc_, which) in RULES.items()}
                o[lname]["stated_vs_naive_counts"] = paired_counts(s, "pred_stated", "pred_naive")
                o[lname]["exact_vs_stated_counts"] = paired_counts(s, "pred_exact", "pred_stated")
                ra, rb = ACC.rate((variant, basis, theta, part, lname, "rule_stated")), ACC.rate((variant, basis, theta, part, lname, "naive_marginal"))
                if ra is not None and rb is not None: o[lname]["stated_minus_naive_agreement_CI"] = ci(ra - rb)
                ra, rb = ACC.rate((variant, basis, theta, part, lname, "myopic_rule_target_then_opus")), ACC.rate((variant, basis, theta, part, lname, "naive_myopic_target_then_opus"))
                if ra is not None and rb is not None: o[lname]["myopic_minus_naive_myopic_agreement_CI"] = ci(ra - rb)
                if lname != "all_lambda": o[lname]["by_k"] = {int(k): dict(stated=agreement(g, "pred_stated", "real"), naive=agreement(g, "pred_naive", "real"), exact=agreement(g, "pred_exact", "real")) for k, g in s.groupby("k")}
            fin = sp_[np.isfinite(sp_.real_dJ)]
            o["magnitude"] = dict(MAE_stated=float((fin.pred_stated - fin.real_dJ).abs().mean()), MAE_exact=float((fin.pred_exact - fin.real_dJ).abs().mean()), MAE_naive=float((fin.pred_naive - fin.real_dJ).abs().mean()),
                                  signed_stated_minus_real=float((fin.pred_stated - fin.real_dJ).mean()), signed_exact_minus_real=float((fin.pred_exact - fin.real_dJ).mean()), signed_naive_minus_real=float((fin.pred_naive - fin.real_dJ).mean()))
            g = fin[fin.p_k > 0]
            o["miss_independence_gap"] = dict(n=int(len(g)), mean_stated_minus_exact=float((g.pred_stated - g.pred_exact).mean()), median=float((g.pred_stated - g.pred_exact).median()),
                                              mean_v_b_hit_minus_v_b=float((g.v_b_hit - g.v_b).mean()), mean_b_hit_minus_b=float((g.b_hit - g.b).mean()), v_h_range_over_k=[float(g.v_h.min()), float(g.v_h.max())])
            if part == "cell21":
                o["per_state_all_lambda"] = {}
                for st, d in sp_[sp_.real_decided].groupby("state"):
                    o["per_state_all_lambda"][st] = dict(n_decided=int(len(d)), stated=float(((d.pred_stated >= 0) == (d.real_dJ >= 0)).mean()), exact=float(((d.pred_exact >= 0) == (d.real_dJ >= 0)).mean()),
                                                         naive=float(((d.pred_naive >= 0) == (d.real_dJ >= 0)).mean()), never=float((d.real_dJ < 0).mean()), degenerate_state=bool(d.degenerate.all()),
                                                         allzero_fraction=float(azc.get(st, float("nan"))))
            out[key][part] = o
        bs = Bst[(Bst.variant == variant) & (Bst.basis == basis) & (Bst.theta == theta)]; c4 = {}
        for part in PARTS:
            b = bs[bs.part == part]
            c4[part] = dict(n_state_lambda=int(len(b)), L_star_hist={int(L): int(v) for L, v in b.L_star.value_counts().sort_index().items()},
                            L_naive_hist={int(L): int(v) for L, v in b.L_naive.value_counts().sort_index().items()},
                            n_star_shorter=int((b.L_star < b.L_naive).sum()), n_equal=int((b.L_star == b.L_naive).sum()), n_star_longer=int((b.L_star > b.L_naive).sum()),
                            mean_L_star=float(b.L_star.mean()), mean_L_naive=float(b.L_naive.mean()),
                            realized_star_minus_naive_CI_above_0=int((b.J_star_minus_naive_lo > 0).sum()), realized_star_minus_naive_CI_below_0=int((b.J_star_minus_naive_hi < 0).sum()),
                            n_same_realized_policy=int((b.J_star_minus_naive == 0).sum()), mean_realized_star_minus_naive=float(b.J_star_minus_naive.mean()),
                            realized_star_minus_opus_CI_above_0=int((b.J_star_minus_opus_lo > 0).sum()), realized_star_minus_opus_CI_below_0=int((b.J_star_minus_opus_hi < 0).sum()),
                            first_model_of_optimal_chain={k: int(v) for k, v in b[b.L_star > 0].best_chain.str.split("->").str[0].value_counts().items()},
                            by_lambda={ln: dict(n=int(len(g)), n_star_shorter=int((g.L_star < g.L_naive).sum()), n_star_longer=int((g.L_star > g.L_naive).sum()), mean_L_star=float(g.L_star.mean()), mean_L_naive=float(g.L_naive.mean()),
                                                realized_star_minus_naive_CI_above_0=int((g.J_star_minus_naive_lo > 0).sum()), realized_star_minus_naive_CI_below_0=int((g.J_star_minus_naive_hi < 0).sum())) for ln, g in b.groupby("lam_name")})
        out[key]["C4_chain_length"] = c4
        c2 = C2[(C2.variant == variant) & (C2.basis == basis) & (C2.theta == theta)]; cc = {}
        for part in PARTS:
            g = c2[c2.part == part]; pos = g[g.p_marg > 0]
            if len(g) == 0: continue
            cc[part] = dict(n_pairs=int(len(g)), n_states=int(g.state.nunique()), n_pairs_marginal_positive=int(len(pos)), n_cond_below_marginal_point=int((pos["diff"] < 0).sum()),
                            n_CI_below_0=int((pos.diff_hi < 0).sum()), n_CI_above_0=int((pos.diff_lo > 0).sum()), median_ratio_cond_over_marginal=float(pos.ratio.median()),
                            q25_ratio=float(pos.ratio.quantile(.25)), q75_ratio=float(pos.ratio.quantile(.75)), n_pairs_marginal_zero=int((g.p_marg == 0).sum()),
                            CI_above_0_pairs=pos[pos.diff_lo > 0][["state", "fail", "next", "n_state", "p_cond", "p_marg"]].to_dict("records"))
        k3 = R[(R.variant == variant) & (R.basis == basis) & (R.theta == theta) & (R.part == "all") & (R.k == 3) & (R.lam_name == "1")]
        k2 = R[(R.variant == variant) & (R.basis == basis) & (R.theta == theta) & (R.part == "all") & (R.k == 2) & (R.lam_name == "1")]
        cc["global_k3_second_order"] = dict(n=int(len(k3)), n_cond_below_marginal=int((k3.p_k < k3.p_marg).sum()), median_ratio=float((k3.p_k / k3.p_marg).median()), mean_p_cond=float(k3.p_k.mean()), mean_p_marg=float(k3.p_marg.mean()))
        cc["global_k2_first_order"] = dict(n=int(len(k2)), n_cond_below_marginal=int((k2.p_k < k2.p_marg).sum()), median_ratio=float((k2.p_k / k2.p_marg).median()), mean_p_cond=float(k2.p_k.mean()), mean_p_marg=float(k2.p_marg.mean()))
        cc["retry_same_model"] = "NOT OBSERVABLE offline: the released matrix has one recorded output per (task, model), so p_retry|fail and the ratio p_retry|fail / p_switch|fail need the live retry experiment (pending); only the switch case is reported."
        out[key]["C2_post_failure_vs_marginal"] = cc
    return out

# ------------------------------------------------------------------ part (4): ACRouter OOD176 escalation traces
def ood_section():
    o = pd.read_csv(f"{DATA}/coderouterbench/ood176_results_long.csv"); tasks = pd.read_json(f"{DATA}/coderouterbench/ood176_tasks.jsonl", lines=True)
    R = o.pivot(index="task_id", columns="model", values="resolved"); Cst = o.pivot(index="task_id", columns="model", values="cost_usd"); Ap = o.pivot(index="task_id", columns="model", values="apply_ok")
    assert R.shape == (176, 8) and set(tasks.task_id) == set(R.index)
    tr = pd.read_json(f"{ACR_OUT}/acrouter_ood176/ood_decisions.jsonl", lines=True).set_index("task_id").loc[R.index]
    idd = pd.read_json(f"{ACR_OUT}/current/id_decisions.jsonl", lines=True)
    rel_metrics = json.load(open(f"{ACR_OUT}/acrouter_ood176/ood_metrics.json"))
    CHAIN = ["MiniMax-M2.7", "kimi-k2.5", "gpt-5.4", "glm-5"]; OPUS = "claude-opus-4-6"; FULL = CHAIN + [OPUS]
    LAM_ACR = 0.1                                   # REWARD_COST_WEIGHT of the released code: reward = resolved - 0.1 cost  =>  lambda = -eps2/eps1 = 0.1 per dollar
    eta_B = float(R[OPUS].mean() / Cst[OPUS].mean()); LAMS = {"eta_B_ood": eta_B, "1": 1.0, "10": 10.0, "0.1_acrouter_reward": LAM_ACR}
    n = len(R); Rm = R[FULL].to_numpy().astype(float); Cm = Cst[FULL].to_numpy(); Am = Ap[FULL].to_numpy().astype(bool)
    rep_ok = 0
    for i, t in enumerate(R.index):
        run = []; applied = 0; res = 0
        for j, m in enumerate(CHAIN):
            run.append(m); applied += int(Am[i, j]); res = Rm[i, j]
            if res >= 1: break
        esc = False
        if res < 1 and applied >= 2: run.append(OPUS); esc = True; res = max(res, Rm[i, 4])
        cost = sum(Cst.loc[t, m] for m in run)
        rep_ok += (run == list(tr.loc[t, "chain_run"])) and (bool(res) == bool(tr.loc[t, "resolved"])) and abs(cost - tr.loc[t, "cost_usd"]) < 1e-6 and esc == bool(tr.loc[t, "escalated"])
    replay = dict(n_traces=int(len(tr)), n_reproduced_exactly=int(rep_ok), rule="run MiniMax-M2.7 -> kimi-k2.5 -> gpt-5.4 -> glm-5 until resolved; escalate to Opus iff unresolved and >= 2 cheap patches applied (k = 2)",
                  n_steps_hist={int(k): int(v) for k, v in tr.n_steps.value_counts().sort_index().items()}, n_escalated=int(tr.escalated.sum()), n_resolved=int(tr.resolved.sum()),
                  total_cost_usd=float(tr.cost_usd.sum()), mean_steps=float(tr.n_steps.mean()), released_metrics=rel_metrics)
    rng = np.random.default_rng([SEED, 176]); idx = rng.integers(0, n, size=(NBOOT, n)); W = np.zeros((NBOOT, n)); np.add.at(W, (np.repeat(np.arange(NBOOT), n), idx.ravel()), 1.0)
    F = np.ones(n, bool); steps = []
    for k, m in enumerate(FULL):
        H = Rm[:, k] >= 1; nF = int(F.sum()); p = float(H[F].mean()); a = float(Cm[F, k].mean())
        bp = (W @ (F & H).astype(float)) / (W @ F.astype(float)); ba = (W @ (F * Cm[:, k])) / (W @ F.astype(float)); bm = (W @ H.astype(float)) / n
        steps.append(dict(k=k + 1, model=m, n_state=nF, p_cond=p, p_cond_CI=ci(bp), a_k=a, a_k_CI=ci(ba), p_marg=float(H.mean()), p_marg_CI=ci(bm), a_marg=float(Cm[:, k].mean()),
                          cond_minus_marg_CI=ci(bp - bm), opus_resolved_on_state=float(Rm[F, 4].mean()), opus_cost_on_state=float(Cm[F, 4].mean())))
        F = F & ~H
    n_fail_all = int((Rm[:, :4] < 1).all(axis=1).sum())
    def dp_rows(ps, as_, lam, bs=None, vbs=None, bT=0.0, vbT=0.0):
        K = len(ps); bs = bs or [0.0] * K; vbs = vbs or [0.0] * K
        Jd = [vb - lam * b for vb, b in zip(vbs, bs)] + [vbT - lam * bT]; Pi_next = Jd[K]; out = [None] * K
        for k in reversed(range(K)):
            Delta = Pi_next - Jd[k + 1]; W_ = lam * bs[k] + 1.0 - vbs[k]; att = ps[k] - lam * as_[k] + (1 - ps[k]) * Pi_next
            out[k] = dict(W=W_, Delta_next=Delta, lhs=ps[k] * W_ + (1 - ps[k]) * Delta, rhs=lam * as_[k], margin=ps[k] * W_ + (1 - ps[k]) * Delta - lam * as_[k], satisfied=bool(att >= Jd[k]), Pi=max(Jd[k], att))
            Pi_next = out[k]["Pi"]
        return out
    ps = [s["p_cond"] for s in steps]; as_ = [s["a_k"] for s in steps]; rule = {}
    for ln, lam in LAMS.items():
        give_up = dp_rows(ps, as_, lam); F = np.ones(n, bool); bs, vbs = [], []
        for k in range(4): bs.append(float(Cm[F, 4].mean())); vbs.append(float(Rm[F, 4].mean())); F = F & ~(Rm[:, k] >= 1)
        opus_term = dp_rows(ps[:4], as_[:4], lam, bs, vbs, float(Cm[F, 4].mean()), float(Rm[F, 4].mean()))
        rule[ln] = dict(lam=lam, terminal_give_up=[dict(k=k + 1, model=FULL[k], **give_up[k]) for k in range(5)], terminal_opus=[dict(k=k + 1, model=FULL[k], **opus_term[k]) for k in range(4)])
    esc = tr.escalated.to_numpy().astype(bool); fail4 = (Rm[:, :4] < 1).all(axis=1); gated = fail4 & esc; ungated = fail4 & ~esc; obs = {}
    for k in range(2, 6):
        taken = np.array([len(c) >= k for c in tr.chain_run]); succ = int((taken & (Rm[:, k - 1] >= 1)).sum())
        obs[f"step{k}_{FULL[k-1]}"] = dict(n_steps_taken=int(taken.sum()), n_resolved_at_step=succ, cost_paid_usd=float(Cm[taken, k - 1].sum()),
                                           rule_satisfied_on_state={ln: rule[ln]["terminal_give_up"][k - 1]["satisfied"] for ln in LAMS},
                                           rule_margin_on_state={ln: rule[ln]["terminal_give_up"][k - 1]["margin"] for ln in LAMS})
    def gain_rows(mask, lam):
        g = Rm[mask, 4] - lam * Cm[mask, 4]; Wm = W[:, mask]; b = (Wm @ g) / Wm.sum(1); return dict(mean=float(g.mean()), CI=ci(b), n=int(mask.sum()))
    escal = dict(n_failed_all_four=n_fail_all, n_escalated_to_opus=int(gated.sum()), n_not_escalated_apply_gate=int(ungated.sum()),
                 opus_resolved_escalated=int(Rm[gated, 4].sum()), opus_resolved_not_escalated=int(Rm[ungated, 4].sum()),
                 opus_cost_escalated_mean=float(Cm[gated, 4].mean()), opus_cost_not_escalated_mean=float(Cm[ungated, 4].mean()),
                 p_opus_given_escalated=float(Rm[gated, 4].mean()), p_opus_given_not_escalated=float(Rm[ungated, 4].mean()), p_opus_given_fail4=float(Rm[fail4, 4].mean()),
                 rule_last_attempt_p_ge_lambda_a={ln: dict(escalated_state=bool(Rm[gated, 4].mean() >= lam * Cm[gated, 4].mean()), not_escalated_state=bool(Rm[ungated, 4].mean() >= lam * Cm[ungated, 4].mean()),
                                                           realized_gain_escalated=gain_rows(gated, lam), realized_gain_not_escalated=gain_rows(ungated, lam)) for ln, lam in LAMS.items()},
                 applied_in_cheap_hist_fail4={int(k): int(v) for k, v in tr.applied_in_cheap[fail4].value_counts().sort_index().items()},
                 opus_resolved_by_applied_in_cheap_fail4={int(k): dict(n=int(len(g)), resolved=int(Rm[[R.index.get_loc(t) for t in g.index], 4].sum())) for k, g in tr[fail4].groupby("applied_in_cheap")})
    id_mode = dict(n_rows=int(len(idd)), n_unique_tasks=int(idd.task_id.nunique()), columns=list(idd.columns), has_chain_fields=bool(any(c in idd.columns for c in ["chain_run", "n_steps", "escalated"])),
                   chosen_model_hist={k: int(v) for k, v in idd.chosen_model.value_counts().items()},
                   statement="ACRouter's released ID mode routes ONCE per task (per-task lookup policy; no within-task chain, retry or escalation): the K = 1 case without a terminal fallback of prop:attempts; nothing to replay for X10 on id_test.")
    return dict(eta_B_ood_per_dollar=eta_B, lambdas=LAMS, replay_of_released_rule=replay, per_step_conditional=steps, rule_by_lambda=rule, observed_steps=obs, opus_escalation_gate=escal, id_mode=id_mode,
                notes=["IN-SAMPLE / descriptive: OOD176 has no probing split; primitives and realizations come from the same 176 tasks (bootstrap CIs over tasks).",
                       "H = resolved (binary sandbox verdict); v_h = 1; ACRouter's terminal path delivers the failed output (b = 0, v_b = 0), so W_lambda = 1 and J^direct = 0; the Opus-terminal alternative is also reported.",
                       "Costs are the released token-price cost_usd of each step; ACRouter's reward weight implies lambda = 0.1 per dollar."])

# ------------------------------------------------------------------ AUDIT: independent recomputation (pandas / plain Python on the long table)
def audit(R):
    dfp = df[df.split == "probing"]; dft = df[df.split == "id_test"]
    q = dfp[dfp.model == "Qwen3-Max"].set_index("task_id").score; g = dfp[dfp.model == "gpt-5.4"].set_index("task_id").score.reindex(q.index)
    p2_cond = float(((q < 1) & (g >= 1)).sum() / (q < 1).sum()); p2_marg = float((g >= 1).mean())
    row = R[(R.variant == "released") & (R.basis == "usd") & (R.theta == 1.0) & (R.part == "all") & (R.chain == "Qwen3-Max->gpt-5.4") & (R.k == 2) & (R.lam_name == "1")].iloc[0]
    t = dft.pivot(index="task_id", columns="model", values="score"); c = dft.pivot(index="task_id", columns="model", values="cost_usd")
    hit = t["Qwen3-Max"] >= 1; Vc = hit * t["Qwen3-Max"] + (~hit) * t[STRONG]; Dc = c["Qwen3-Max"] + (~hit) * c[STRONG]
    dJ1 = float(((Vc - Dc) - (t[STRONG] - c[STRONG])).mean())
    row1 = R[(R.variant == "released") & (R.basis == "usd") & (R.theta == 1.0) & (R.part == "all") & (R.chain == "Qwen3-Max") & (R.k == 1) & (R.lam_name == "1")].iloc[0]
    sub = R[(R.variant == "released") & (R.basis == "usd") & (R.theta == 1.0) & (R.part == "cell21") & R.real_decided & np.isfinite(R.pred_stated)]
    agree = float(((sub.pred_stated >= 0) == (sub.real_dJ >= 0)).mean()); n_naive = float(((sub.pred_naive >= 0) == (sub.real_dJ >= 0)).mean())
    return dict(A1_global_theta1_p2_gpt54_given_qwen_fail=dict(pandas=p2_cond, run=float(row.p_k), marginal_pandas=p2_marg, marginal_run=float(row.p_marg), match=bool(abs(p2_cond - row.p_k) < 1e-12 and abs(p2_marg - row.p_marg) < 1e-12)),
                A2_global_theta1_lambda1_realized_dJ_attempt_qwen_then_opus=dict(pandas=dJ1, run_real0=float(row1.real0_dJ), match=bool(abs(dJ1 - row1.real0_dJ) < 1e-12),
                                                                                  cross_experiment_X3_README="0.0706 [0.0618, 0.0800] (X3_threshold_voi, pooled Qwen3-Max, theta = 1, lambda = 1)"),
                A3_cell21_theta1_all_lambda_decided_agreement_recomputed=dict(stated_pandas=agree, naive_pandas=n_naive, n_decided=int(len(sub))))

# ------------------------------------------------------------------ figures
def make_figures(results, R, Bst, curves_main, ood):
    # Fig 1: conditional success vs attempt index (global ID state, theta 1 / 0.5) and the OOD176 released chain
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.1), layout="constrained")
    for ax, th in zip(axes[:2], ["theta1.0", "theta0.5"]):
        cv = curves_main[th]; k = np.array(cv["k"])
        ax.fill_between(k, cv["p_marg_q10"], cv["p_marg_q90"], color=RED, alpha=.12, lw=0); ax.fill_between(k, cv["p_cond_q10"], cv["p_cond_q90"], color=BLUE, alpha=.12, lw=0)
        ax.plot(k, cv["p_marg_mean"], "-s", color=RED, ms=4, label=r"marginal $P(H_k)$"); ax.plot(k, cv["p_cond_mean"], "-o", color=BLUE, ms=4, label=r"post-failure $p_k$")
        ax.set(xticks=k, xlabel=r"attempt index $k$", ylim=(0, None)); ax.set_ylabel(r"success probability ($\theta=%s$, %d chains)" % (th[5:], cv["n_chains"]), fontsize=8)
    axes[0].legend(frameon=False, loc="upper right")
    ax = axes[2]; st = ood["per_step_conditional"]; k = np.array([s["k"] for s in st])
    pm = np.array([s["p_marg"] for s in st]); pc = np.array([s["p_cond"] for s in st])
    ax.errorbar(k - .06, pm, yerr=[pm - np.array([s["p_marg_CI"][0] for s in st]), np.array([s["p_marg_CI"][1] for s in st]) - pm], fmt="s", color=RED, ms=4, elinewidth=.7, capsize=0)
    ax.errorbar(k + .06, pc, yerr=[pc - np.array([s["p_cond_CI"][0] for s in st]), np.array([s["p_cond_CI"][1] for s in st]) - pc], fmt="o", color=BLUE, ms=4, elinewidth=.7, capsize=0)
    ax.set_xticks(k); ax.set_xticklabels([s["model"].replace("claude-opus-4-6", "Opus").replace("MiniMax-M2.7", "MiniMax").replace("kimi-k2.5", "kimi") for s in st], rotation=30, ha="right", fontsize=7)
    ax.set(ylim=(0, 1), ylabel="resolved on OOD176 (released chain)"); ax.yaxis.label.set_size(8)
    for s in st: ax.annotate(f"n={s['n_state']}", (s["k"], 0.97), ha="center", va="top", fontsize=6, color=".35")
    fig.savefig(os.path.join(FIG, "fig1_conditional_success.pdf"), metadata={"Creator": "X10 run.py"}); plt.close(fig)
    # Fig 2: predicted vs realized per-segment gain, cell21, theta = 1, dollars: stated rule (a) and naive marginal rule (b)
    sub = R[(R.variant == "released") & (R.basis == "usd") & (R.theta == 1.0) & (R.part == "cell21") & np.isfinite(R.real_dJ)]
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.5), layout="constrained")
    lim = float(max(sub.pred_stated.abs().max(), sub.pred_naive.abs().max(), sub.real_dJ.abs().max()) * 1.15)
    for ax, pc_, lab in zip(axes, ["pred_stated", "pred_naive"], [r"predicted $p_kW_\lambda+(1-p_k)\Delta_{k+1}-\lambda a_k$ (probing)", r"naive: marginal $P(E_m)$ in the same recursion"]):
        for ln in LAM_MULT:
            ss = sub[sub.lam_name == ln]; ax.scatter(ss[pc_], ss.real_dJ, s=4, color=LAMCOL[ln], alpha=.45, linewidths=0, label=(r"$\lambda=\eta_B$" if ln == "eta_B" else fr"$\lambda={ln}$"), rasterized=True)
        ax.plot([-lim, lim], [-lim, lim], color=".6", lw=.65, ls=":"); ax.axhline(0, color=".6", lw=.65); ax.axvline(0, color=".6", lw=.65)
        ax.set_xscale("symlog", linthresh=1e-3); ax.set_yscale("symlog", linthresh=1e-3); ax.set(xlabel=lab, xlim=(-lim, lim), ylim=(-lim, lim))
    axes[0].set_ylabel(r"realized $\Delta J_\lambda$ of attempt $k$ (id_test)"); axes[0].legend(frameon=False, loc="upper left", handletextpad=.1, borderpad=.2, markerscale=2)
    fig.savefig(os.path.join(FIG, "fig2_predicted_vs_realized.pdf"), metadata={"Creator": "X10 run.py"}, dpi=300); plt.close(fig)
    # Fig 3: chain-length histograms, optimal (conditional) vs marginal-planned, cell21 states x 5 lambda
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.0), layout="constrained")
    panels = [("released", "usd", 1.0, r"dollars, $\theta=1$"), ("released", "usd", 0.5, r"dollars, $\theta=0.5$"), ("released", "sec", 1.0, r"seconds, $\theta=1$")]
    for ax, (v, b, th, lab) in zip(axes, panels):
        bb = Bst[(Bst.variant == v) & (Bst.basis == b) & (Bst.theta == th) & (Bst.part == "cell21")]; Ls = np.arange(0, KMAX + 1); w = .38
        hs = [(bb.L_star == L).sum() for L in Ls]; hn = [(bb.L_naive == L).sum() for L in Ls]
        ax.bar(Ls - w / 2, hs, w, color=BLUE, label="optimal (post-failure $p_k$)"); ax.bar(Ls + w / 2, hn, w, color=RED, label="marginal-planned")
        ax.set(xticks=Ls, xlabel=r"chain length $L$ (%s)" % lab); ax.set_ylabel("states $\\times\\ \\lambda$ (%d)" % len(bb), fontsize=8)
    axes[0].legend(frameon=False, loc="upper left")
    fig.savefig(os.path.join(FIG, "fig3_chain_length.pdf"), metadata={"Creator": "X10 run.py"}); plt.close(fig)

# ------------------------------------------------------------------ run
results = dict(experiment_id="X10_escalation_chains", seed=SEED, nboot=NBOOT, kmax=KMAX, thetas=THETAS, n_chains=len(CHAINS), primaries=PRIMS, terminal=STRONG,
               lambdas_dollars={k: (v if v is not None else "eta_B (probing) of the configuration") for k, v in LAM_MULT.items()}, data_notes=DATA_NOTES, configs={}, identities={})
results["identities"]["C1_symbolic_and_paper_instance"] = identity_checks()
worst = 0.0; probmask = (meta.split == "probing").to_numpy()
for theta in THETAS:
    for lam in [0.1, 1.0, 10.0, 100.0]: worst = max(worst, dp_vs_bruteforce(Yall[probmask], Call["usd"][probmask], np.arange(int(probmask.sum())), theta, lam))
results["identities"]["C1_dp_equals_best_stopping_policy_max_abs_resid"] = worst
log("identities done", results["identities"])
ACC = AgreeAcc(); ALL_R, ALL_B, ALL_C2 = [], [], []; curves_main = None
for ci_, (variant, basis) in enumerate([(v, b) for v in VARIANTS for b in BASES]):
    info, R, Bst, C2, decomp, curves = run_config(variant, basis, ci_, ACC)
    results["configs"][f"{variant}|{basis}"] = dict(info=info, global_decomposition_at_eta_B=decomp, global_success_curves=curves)
    if variant == "released" and basis == "usd": curves_main = curves
    ALL_R.append(R); ALL_B.append(Bst); ALL_C2.append(C2)
    if basis == "usd": R.to_csv(os.path.join(HERE, f"segments_{variant}_usd.csv.gz"), index=False, compression="gzip")
    log(f"config {variant}/{basis} done: {len(R)} segment rows, eta_B={info['eta_B']:.4f}")
R = pd.concat(ALL_R, ignore_index=True); Bst = pd.concat(ALL_B, ignore_index=True); C2 = pd.concat(ALL_C2, ignore_index=True)
Bst.to_csv(os.path.join(HERE, "best_chains.csv"), index=False); C2.to_csv(os.path.join(HERE, "c2_pairs.csv"), index=False)
results["summary"] = summarize(R, Bst, C2, ACC); log("summaries done")
results["ood176_acrouter_traces"] = ood_section(); log("OOD section done")
results["audit"] = audit(R)
results["global_worked_example"] = R[(R.variant == "released") & (R.basis == "usd") & (R.part == "all") & (R.chain == "Qwen3-Max->kimi-k2.5->gpt-5.4") & (R.lam_name == "1")].drop(columns=["variant", "basis", "part", "state"]).to_dict("records")
make_figures(results, R, Bst, curves_main, results["ood176_acrouter_traces"]); log("figures done")

# ------------------------------------------------------------------ claims
S1 = results["summary"]["released|usd|theta1.0"]; S5 = results["summary"]["released|usd|theta0.5"]; SX = results["summary"]["bcb_allzero_excluded|usd|theta1.0"]; SS = results["summary"]["released|sec|theta1.0"]
def ag(Sm, part, lname, rule): return Sm[part][lname][rule]
c3 = {p: {v: dict(stated=ag(Sm, p, "all_lambda", "rule_stated"), naive=ag(Sm, p, "all_lambda", "naive_marginal"), exact=ag(Sm, p, "all_lambda", "exact_dp"), diff_CI=Sm[p]["all_lambda"].get("stated_minus_naive_agreement_CI"),
                  myopic=ag(Sm, p, "all_lambda", "myopic_rule_target_then_opus"), naive_myopic=ag(Sm, p, "all_lambda", "naive_myopic_target_then_opus"), diff_myopic_CI=Sm[p]["all_lambda"].get("myopic_minus_naive_myopic_agreement_CI"))
          for v, Sm in [("theta1", S1), ("theta0.5", S5), ("theta1_bcb_excluded", SX), ("theta1_seconds", SS)]} for p in ["dim9", "cell21"]}
c3_pass = all(c3[p][v]["diff_CI"][0] > 0 for p in c3 for v in ["theta1", "theta0.5", "theta1_bcb_excluded"])
c2s = {v: Sm["C2_post_failure_vs_marginal"] for v, Sm in [("theta1", S1), ("theta0.5", S5), ("theta1_bcb_excluded", SX)]}
c2_pass = all(c2s[v]["dim9"]["n_CI_below_0"] / c2s[v]["dim9"]["n_pairs_marginal_positive"] >= 0.75 and c2s[v]["dim9"]["n_CI_above_0"] == 0 for v in c2s)
c4s = {v: Sm["C4_chain_length"] for v, Sm in [("theta1", S1), ("theta0.5", S5), ("theta1_bcb_excluded", SX), ("theta1_seconds", SS)]}
ood = results["ood176_acrouter_traces"]; obs = ood["observed_steps"]; gate = ood["opus_escalation_gate"]
steps_sat = {ln: [name.split("_", 1)[0] for name, o_ in obs.items() if o_["rule_satisfied_on_state"][ln]] for ln in ood["lambdas"]}
c5_verdict = ("INCONCLUSIVE (descriptive, n = 176; counts, not rates): observed steps whose realized state satisfies the rule -- "
              + "; ".join(f"lambda = {ln}: {', '.join(v) if v else 'none'}" for ln, v in steps_sat.items())
              + f". The gated Opus step resolved {gate['opus_resolved_escalated']} of {gate['n_escalated_to_opus']} escalated tasks at ${gate['opus_cost_escalated_mean']:.2f} each "
              + f"(rule violated at every lambda >= eta_B_ood = {ood['eta_B_ood_per_dollar']:.2f}/$, satisfied only at ACRouter's reward price 0.1/$); "
              + f"Opus would have resolved {gate['opus_resolved_not_escalated']} of the {gate['n_not_escalated_apply_gate']} all-fail tasks the apply-count gate withheld.")
a3 = results["audit"]["A3_cell21_theta1_all_lambda_decided_agreement_recomputed"]
a3.update(run_stated=S1["cell21"]["all_lambda"]["rule_stated"]["agreement"], run_naive=S1["cell21"]["all_lambda"]["naive_marginal"]["agreement"], run_n_decided=S1["cell21"]["all_lambda"]["rule_stated"]["n_decided"])
a3["match"] = bool(abs(a3["stated_pandas"] - a3["run_stated"]) < 1e-12 and abs(a3["naive_pandas"] - a3["run_naive"]) < 1e-12 and a3["n_decided"] == a3["run_n_decided"])
results["claims"] = [
    dict(id="C1", kind="IDENTITY", statement="Backward-induction identity of prop:attempts: [p_k v_h - lambda a_k + (1-p_k)Pi_{k+1}] - J^direct = p_k W_lambda - lambda a_k + (1-p_k)Delta_{k+1}; Pi_1 equals the best stopping policy; the appendix instance gives Delta_2 = 13/20 and 137/200 >= 1/5",
         result=dict(symbolic=results["identities"]["C1_symbolic_and_paper_instance"]["symbolic_rule_identity"], paper_instance_ok=results["identities"]["C1_symbolic_and_paper_instance"]["paper_instance"]["ok"], dp_vs_bruteforce_max_abs_resid=worst),
         verdict="PASS" if (results["identities"]["C1_symbolic_and_paper_instance"]["symbolic_rule_identity"] and results["identities"]["C1_symbolic_and_paper_instance"]["paper_instance"]["ok"] and worst < 1e-9) else "FAIL"),
    dict(id="C2", kind="IN-SAMPLE", statement="Post-failure conditional success p_k < marginal P(H_k) for most model pairs (positively correlated errors)", result=c2s,
         verdict="PASS" if c2_pass else "FAIL"),
    dict(id="C3", kind="OUT-OF-SAMPLE", statement="Sign agreement of the prop:attempts rule (probing primitives) with the realized per-segment Delta J_lambda on id_test, CI-decided cells, >= the naive rule that uses the marginal P(E_m)",
         result=c3, verdict="PASS" if c3_pass else "FAIL"),
    dict(id="C4", kind="IN-SAMPLE (plan lengths) / OUT-OF-SAMPLE (realized J of the two chains)", statement="The optimal chain (post-failure p_k) is shorter than the marginal-planned chain in a substantial fraction of states and never longer; where the two differ the optimal chain realizes a higher J_lambda more often than not",
         result=c4s, verdict="PASS" if all(c4s[v]["cell21"]["n_star_longer"] == 0 and c4s[v]["cell21"]["n_star_shorter"] > 0 and c4s[v]["cell21"]["realized_star_minus_naive_CI_above_0"] > c4s[v]["cell21"]["realized_star_minus_naive_CI_below_0"] for v in c4s) else "FAIL"),
    dict(id="C5", kind="DESCRIPTIVE (in-sample, n = 176)", statement="ACRouter's observed OOD176 escalation steps are consistent with the rule at lambda in {eta_B, 1, 10} (counts, not rates)",
         result=dict(replay=ood["replay_of_released_rule"]["n_reproduced_exactly"], observed_steps=obs, opus_gate=dict(n_fail4=gate["n_failed_all_four"], n_escalated=gate["n_escalated_to_opus"], opus_resolved_escalated=gate["opus_resolved_escalated"],
                     opus_resolved_not_escalated=gate["opus_resolved_not_escalated"], n_not_escalated=gate["n_not_escalated_apply_gate"], rule=gate["rule_last_attempt_p_ge_lambda_a"])),
         verdict=c5_verdict)]
results["pending"] = ["LIVE retry part (same-model resampling k <= 3 on a 500-task subsample with sandbox verification; p_retry|fail vs p_switch|fail) -- separate later experiment; not observable offline."]
results["runtime_seconds"] = time.time() - T0

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating, float)): return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return o
with open(os.path.join(HERE, "results.json"), "w") as f: json.dump(clean(results), f, indent=1)

# ------------------------------------------------------------------ auto-generated tables for the README
def fmt(x, d=3): return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
lines = ["## Auto-generated tables (from results.json)\n"]
for key, Sm in [("released|usd|theta1.0", S1), ("released|usd|theta0.5", S5), ("bcb_allzero_excluded|usd|theta1.0", SX), ("released|sec|theta1.0", SS)]:
    lines.append(f"### Sign agreement on CI-decided (state, chain, k) cells -- {key}\n")
    lines.append("| part | lambda | n decided | rule (stated) [CI] | exact DP | naive marginal [CI] | stated - naive [CI] | always attempt | never attempt | non-degenerate: n / stated / naive / never | myopic rule vs naive-myopic (then Opus) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for part in ["all", "dim9", "cell21"]:
        for ln in list(LAM_MULT) + ["all_lambda"]:
            o = Sm[part][ln]; a, e, nv, m0, n0 = o["rule_stated"], o["exact_dp"], o["naive_marginal"], o["myopic_rule_target_then_opus"], o["naive_myopic_target_then_opus"]
            if "agreement" not in a: continue
            lines.append(f"| {part} | {ln} | {a['n_decided']} | {fmt(a['agreement'])} [{fmt(a.get('agreement_CI', (None, None))[0])}, {fmt(a.get('agreement_CI', (None, None))[1])}] | {fmt(e['agreement'])} | {fmt(nv['agreement'])} [{fmt(nv.get('agreement_CI', (None, None))[0])}, {fmt(nv.get('agreement_CI', (None, None))[1])}] | "
                         f"[{fmt(o.get('stated_minus_naive_agreement_CI', (None, None))[0])}, {fmt(o.get('stated_minus_naive_agreement_CI', (None, None))[1])}] | {fmt(a['baseline_always_attempt'])} | {fmt(a['baseline_never_attempt'])} | "
                         f"{a['n_decided_non_degenerate']} / {fmt(a['agreement_non_degenerate'])} / {fmt(nv['agreement_non_degenerate'])} / {fmt(a['baseline_never_attempt_non_degenerate'])} | {fmt(m0['agreement'])} vs {fmt(n0['agreement'])} (n {m0['n_decided']}) |")
    lines.append("")
lines.append("### C4 chain lengths (optimal = post-failure recursion; marginal-planned = naive recursion), states x 5 lambda\n")
lines.append("| config | part | n | L* hist 0/1/2/3 | L_naive hist 0/1/2/3 | optimal shorter / equal / longer | mean L* / L_naive | realized J*-J_naive CI>0 / CI<0 / identical policy |"); lines.append("|---|---|---|---|---|---|---|---|")
for key, Sm in [("released|usd|theta1.0", S1), ("released|usd|theta0.5", S5), ("bcb_allzero_excluded|usd|theta1.0", SX), ("released|sec|theta1.0", SS)]:
    for part in ["all", "dim9", "cell21"]:
        c = Sm["C4_chain_length"][part]; h = lambda d: "/".join(str(d.get(str(L), d.get(L, 0))) for L in range(KMAX + 1))
        lines.append(f"| {key} | {part} | {c['n_state_lambda']} | {h(c['L_star_hist'])} | {h(c['L_naive_hist'])} | {c['n_star_shorter']} / {c['n_equal']} / {c['n_star_longer']} | {fmt(c['mean_L_star'], 2)} / {fmt(c['mean_L_naive'], 2)} | {c['realized_star_minus_naive_CI_above_0']} / {c['realized_star_minus_naive_CI_below_0']} / {c['n_same_realized_policy']} |")
lines.append("\n### C2 post-failure conditional vs marginal success (probing, ordered pairs with marginal > 0)\n")
lines.append("| config | part | pairs (marg>0) | cond < marg (point) | CI < 0 | CI > 0 | median ratio cond/marg [IQR] |"); lines.append("|---|---|---|---|---|---|---|")
for key, Sm in [("released|usd|theta1.0", S1), ("released|usd|theta0.5", S5), ("bcb_allzero_excluded|usd|theta1.0", SX), ("released|sec|theta1.0", SS)]:
    for part in ["all", "dim9", "cell21"]:
        c = Sm["C2_post_failure_vs_marginal"][part]
        lines.append(f"| {key} | {part} | {c['n_pairs_marginal_positive']} (of {c['n_pairs']}) | {c['n_cond_below_marginal_point']} | {c['n_CI_below_0']} | {c['n_CI_above_0']} | {fmt(c['median_ratio_cond_over_marginal'])} [{fmt(c['q25_ratio'])}, {fmt(c['q75_ratio'])}] |")
lines.append("\n### Per-state decided agreement, cell21, dollars, theta = 1, all lambda\n")
lines.append("| state | all-zero frac | n decided | stated | exact DP | naive | never attempt |"); lines.append("|---|---|---|---|---|---|---|")
for st, d in S1["cell21"]["per_state_all_lambda"].items():
    lines.append(f"| {st}{' (degenerate)' if d['degenerate_state'] else ''} | {fmt(d['allzero_fraction'], 2)} | {d['n_decided']} | {fmt(d['stated'])} | {fmt(d['exact'])} | {fmt(d['naive'])} | {fmt(d['never'])} |")
lines.append("\n### OOD176 released chain: per-step post-failure success and cost (in-sample, 95% CIs)\n")
lines.append("| k | model | n in state | p_k (cond) [CI] | marginal P(H_k) [CI] | a_k $ [CI] | Opus resolved / cost on state |"); lines.append("|---|---|---|---|---|---|---|")
for s in ood["per_step_conditional"]:
    lines.append(f"| {s['k']} | {s['model']} | {s['n_state']} | {fmt(s['p_cond'])} [{fmt(s['p_cond_CI'][0])}, {fmt(s['p_cond_CI'][1])}] | {fmt(s['p_marg'])} [{fmt(s['p_marg_CI'][0])}, {fmt(s['p_marg_CI'][1])}] | {fmt(s['a_k'])} [{fmt(s['a_k_CI'][0])}, {fmt(s['a_k_CI'][1])}] | {fmt(s['opus_resolved_on_state'])} / {fmt(s['opus_cost_on_state'])} |")
lines.append("\n### OOD176 rule check per step (terminal = ACRouter's give-up path, b = 0, v_b = 0; W = 1)\n")
lines.append("| lambda | k=1 MiniMax | k=2 kimi | k=3 gpt-5.4 | k=4 glm-5 | k=5 Opus |"); lines.append("|---|---|---|---|---|---|")
for ln, rr in ood["rule_by_lambda"].items():
    lines.append(f"| {ln} ({fmt(rr['lam'], 3)}) | " + " | ".join(f"{'attempt' if x['satisfied'] else 'stop'} ({fmt(x['margin'])})" for x in rr["terminal_give_up"]) + " |")
with open(os.path.join(HERE, "tables_autogen.md"), "w") as f: f.write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ summary_for_paper.md
a1, n1 = S1["cell21"]["all_lambda"]["rule_stated"], S1["cell21"]["all_lambda"]["naive_marginal"]; d1 = S1["dim9"]["all_lambda"]["rule_stated"]; dn = S1["dim9"]["all_lambda"]["naive_marginal"]
a5, n5 = S5["cell21"]["all_lambda"]["rule_stated"], S5["cell21"]["all_lambda"]["naive_marginal"]; e1 = S1["cell21"]["all_lambda"]["exact_dp"]
c2g = S1["C2_post_failure_vs_marginal"]; c4c = S1["C4_chain_length"]["cell21"]; c4d = S1["C4_chain_length"]["dim9"]; dec = results["configs"]["released|usd"]["global_decomposition_at_eta_B"]
sfp = [f"# X10_escalation_chains -- numbers that could enter the paper (all offline; seed {SEED}; {NBOOT} paired bootstraps)\n",
       f"1. [IDENTITY, C1] prop:attempts recursion and rule: symbolic identity holds; appendix instance reproduced exactly (Delta_2 = 13/20, 137/200 >= 1/5); Pi_1 = best stopping policy to {worst:.1e} on the real matrix.",
       f"2. [IN-SAMPLE, C2] Post-failure success is about half the marginal: global theta=1, all 42 ordered pairs have p_2|fail < P(H_2) (CI < 0 in 42/42), median ratio {c2g['all']['median_ratio_cond_over_marginal']:.2f}; after two failures (210 chains) the median ratio is {c2g['global_k3_second_order']['median_ratio']:.2f}; per dimension {c2g['dim9']['n_CI_below_0']}/{c2g['dim9']['n_pairs_marginal_positive']} pairs CI < 0, {c2g['dim9']['n_CI_above_0']} CI > 0 (supports the 'positively correlated errors' premise; the only CI > 0 exceptions are algorithm|lcb MiniMax pairs).",
       f"3. [OUT-OF-SAMPLE, C3] Sign agreement of the rule on CI-decided (state, chain, k) cells, theta=1, dollars, 5 lambdas pooled: 21 cells {a1['agreement']:.3f} [{a1['agreement_CI'][0]:.3f}, {a1['agreement_CI'][1]:.3f}] (n={a1['n_decided']}) vs naive marginal rule {n1['agreement']:.3f} [{n1['agreement_CI'][0]:.3f}, {n1['agreement_CI'][1]:.3f}]; 9 dimensions {d1['agreement']:.3f} vs {dn['agreement']:.3f}; theta=0.5, 21 cells {a5['agreement']:.3f} vs {n5['agreement']:.3f}. Baselines: always attempt {a1['baseline_always_attempt']:.3f}, never attempt {a1['baseline_never_attempt']:.3f} (cell21, theta=1).",
       f"4. [OUT-OF-SAMPLE, C3 caveat] {a1['n_decided'] - a1['n_decided_non_degenerate']} of {a1['n_decided']} decided cell21 rows are degenerate states (p_k = 0 on probing: attempt k never succeeded there); excluding them: rule {a1['agreement_non_degenerate']:.3f}, naive {n1['agreement_non_degenerate']:.3f}, never-attempt {a1['baseline_never_attempt_non_degenerate']:.3f}. Errors are almost all false attempts ({a1['false_attempt']} vs {a1['false_stop']} false stops): E[Y_B | H=1] exceeds E[Y_B] by {S1['cell21']['miss_independence_gap']['mean_v_b_hit_minus_v_b']:.2f} on average, so the stated W_lambda is optimistic (X3's miss-independence violation); the exact state-matched recursion reaches {e1['agreement']:.3f}.",
       f"5. [OUT-OF-SAMPLE, magnitude] MAE of the predicted per-segment gain (cell21, theta=1, J units): stated {S1['cell21']['magnitude']['MAE_stated']:.3f}, exact {S1['cell21']['magnitude']['MAE_exact']:.3f}, naive {S1['cell21']['magnitude']['MAE_naive']:.3f}; signed error stated-real {S1['cell21']['magnitude']['signed_stated_minus_real']:+.3f} (optimistic).",
       f"6. [IN-SAMPLE, C4] Optimal chains are shorter than marginal-planned chains in {c4c['n_star_shorter']}/{c4c['n_state_lambda']} cell21 (state, lambda) pairs (dim9: {c4d['n_star_shorter']}/{c4d['n_state_lambda']}), equal otherwise, never longer; mean length {c4c['mean_L_star']:.2f} vs {c4c['mean_L_naive']:.2f}; where they differ the optimal chain realizes a higher J_lambda on id_test with CI > 0 in {c4c['realized_star_minus_naive_CI_above_0']} cases vs CI < 0 in {c4c['realized_star_minus_naive_CI_below_0']} (theta=1).",
       f"7. [OUT-OF-SAMPLE] Global optimal chain at lambda = eta_B, theta=1: {dec['theta1.0_optimal_chain']['chain']} -> Opus: eta/eta_B = {dec['theta1.0_optimal_chain']['eta_over_eta_B']:.2f}, net gain E[V - eta_B D] = {dec['theta1.0_optimal_chain']['net_gain_E_V_minus_etaB_D_relative_to_opus']:.3f} [{dec['theta1.0_optimal_chain']['net_gain_CI'][0]:.3f}, {dec['theta1.0_optimal_chain']['net_gain_CI'][1]:.3f}] = avoided reference work {dec['theta1.0_optimal_chain']['avoided_reference_work_etaB_E_H_CB']:.3f} - attempt cost {dec['theta1.0_optimal_chain']['attempt_cost_etaB_E_A']:.3f} + accepted quality change {dec['theta1.0_optimal_chain']['accepted_quality_change_E_H_YP_minus_YB']:.3f} (recovery cost 0); e = 0 at theta = 1.",
       f"8. [Seconds basis, theta=1] Opus is the fastest path, so the optimal chain has length 0 in {c4s['theta1_seconds']['cell21']['L_star_hist'].get(0, 0)} of {c4s['theta1_seconds']['cell21']['n_state_lambda']} cell21 (state, lambda) pairs (marginal-planned: {c4s['theta1_seconds']['cell21']['L_naive_hist'].get(0, 0)}); pooled over lambda the rule scores {SS['cell21']['all_lambda']['rule_stated']['agreement']:.3f} vs naive {SS['cell21']['all_lambda']['naive_marginal']['agreement']:.3f} and never-attempt {SS['cell21']['all_lambda']['rule_stated']['baseline_never_attempt']:.3f}; at lambda = eta_B the rule ({SS['cell21']['eta_B']['rule_stated']['agreement']:.3f}) is indistinguishable from never-attempt ({SS['cell21']['eta_B']['rule_stated']['baseline_never_attempt']:.3f}).",
       f"9. [DESCRIPTIVE, C5] ACRouter OOD176 traces reproduced exactly ({ood['replay_of_released_rule']['n_reproduced_exactly']}/176; k = 2 apply-gate rule). Per-step post-failure success on the released chain: {', '.join(f'{s['model']} {s['p_cond']:.2f} (marginal {s['p_marg']:.2f})' for s in ood['per_step_conditional'])}. Steps 2-3 satisfy the rule at eta_B and lambda = 1; step 4 (glm-5: 3 of 54 resolved at $0.29) and the Opus step ({gate['opus_resolved_escalated']} of {gate['n_escalated_to_opus']} escalated tasks resolved at ${gate['opus_cost_escalated_mean']:.2f} each) do not at any lambda >= eta_B = {ood['eta_B_ood_per_dollar']:.2f}/$; at ACRouter's own reward price lambda = 0.1 every step but glm-5 passes. The apply-count gate escalated {gate['n_escalated_to_opus']} of {gate['n_failed_all_four']} all-fail tasks; Opus solved {gate['opus_resolved_escalated']} of those vs {gate['opus_resolved_not_escalated']} of the {gate['n_not_escalated_apply_gate']} it withheld.",
       "10. ACRouter's released ID mode routes once per task (2,919 single dispatches, no chain fields): K = 1 without a terminal fallback; nothing to replay for X10 on id_test.",
       "11. Same-model retry (p_retry|fail) is NOT observable offline (one recorded output per task and model); the retry-vs-switch comparison is pending the live experiment.",
       f"12. bcb caveat: excluding the {DATA_NOTES['n_allzero_bcb']} all-zero BigCodeBench tasks raises cell21 rule agreement (theta=1) from {a1['agreement']:.3f} to {SX['cell21']['all_lambda']['rule_stated']['agreement']:.3f} and the naive rule from {n1['agreement']:.3f} to {SX['cell21']['all_lambda']['naive_marginal']['agreement']:.3f}; C2-C4 verdicts unchanged."]
with open(os.path.join(HERE, "summary_for_paper.md"), "w") as f: f.write("\n".join(sfp) + "\n")
print(json.dumps(clean([{k: c[k] for k in ["id", "kind", "verdict"]} for c in results["claims"]]), indent=1))
print("audit", json.dumps(clean(results["audit"]), indent=1))
log("done; runtime %.1f s" % results["runtime_seconds"])

