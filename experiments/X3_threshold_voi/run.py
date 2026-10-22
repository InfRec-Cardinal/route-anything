#!/usr/bin/env python3
"""
X3_threshold_voi -- "Route Anything" Sec. 4 on CodeRouterBench real outcomes.

Results tested (labels from main_body.tex / appendix_core.tex):
  eq:margin       W_lambda = lambda*b + v_h - v_b
  prop:threshold  cheap-first >= direct  iff  p*W_lambda >= lambda*a   (refinement: Delta p*W_lambda >= lambda*k)
  eq:gate-costs   C_cascade = a+(1-s)b, C_direct = b, C_info = k + s a + (1-s) b
  prop:voi        exact information strictly cheapest iff k/(b-a) < s < 1-k/a; nonempty iff k < a(b-a)/b, contains a/b
  app:imperfect-voi  f(s)-E f(s_Z)-k <= f(s)-g(s) = min{s(b-a), a(1-s)}  (Blackwell)

Everything in results.json / figures / README numbers is produced by this file.
numpy / pandas / scipy / matplotlib only.  Fixed seed.  Runtime well under 5 minutes.
"""
import os, json, time, glob, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T0 = time.time()
SEED = 20260911
rng = np.random.default_rng(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
DATA = os.environ.get('ACROUTER_DATA', 'agent-as-a-router/data')
NBOOT = 1000
STRONG = "claude-opus-4-6"                      # reference policy B
CHEAPS = ["Qwen3-Max", "kimi-k2.5"]
THETAS = [1.0, 0.5]
LAMBDAS = [0.1, 1.0, 10.0, 100.0]               # per dollar; eta_B (probing) is appended below
KS = [3e-5, 1e-4, 5e-4, 1e-3]                   # dollars per task; 3e-5 = ACRouter-scale router bill (extra), the other three as specified
NOISES = [(0.0, 0.0), (0.1, 0.1), (0.2, 0.2)]   # (beta_fa, beta_fr)
ROUTERS = ["finetuned_router_qwen35_08b", "finetuned_router_qwen35_2b", "finetuned_router_qwen35_9b_v3",
           "finetuned_router_qwen35_27b_v3", "logreg", "tfidf_mlp", "routellm_mf", "routellm_sw", "dimension_best"]

# ---------------------------------------------------------------- paper figure style (scripts/reproduce.py)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times'], 'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath,amssymb}',
    'mathtext.fontset': 'stix', 'font.size': 9,
    'axes.labelsize': 9, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': .65, 'lines.linewidth': 1.6,
})
BLUE, RED, GREEN, GREY = '#23639b', '#ba5934', '#3a7c5e', '.5'
USETEX_OK = True
try:
    _f, _a = plt.subplots(); _a.set_xlabel(r'$s$'); _f.savefig(os.path.join(FIG, "_usetex_probe.pdf")); plt.close(_f)
    os.remove(os.path.join(FIG, "_usetex_probe.pdf"))
except Exception as e:  # fall back to mathtext and say so
    USETEX_OK = False
    plt.rcParams.update({'text.usetex': False})
    print("usetex unavailable, falling back to mathtext:", type(e).__name__)

# ---------------------------------------------------------------- 0. real data
df = pd.read_csv(f"{DATA}/coderouterbench/id_results_long.csv")
S = df.pivot(index="task_id", columns="model", values="score")
Cc = df.pivot(index="task_id", columns="model", values="cost_usd")
meta = df.drop_duplicates("task_id").set_index("task_id")[["split", "dimension"]].loc[S.index]
meta["bench"] = meta.index.str.split("_").str[1]
meta["cell21"] = meta.dimension + "|" + meta.bench
meta["all"] = "ALL"
assert S.shape == (9999, 8) and not S.isna().any().any() and not Cc.isna().any().any()
PROB = meta.index[meta.split == "probing"]; TEST = meta.index[meta.split == "id_test"]
assert len(PROB) == 7080 and len(TEST) == 2919
n_zero_cost = int((df.cost_source != "token_log_pricing").sum())
# router decisions restricted to id_test
DEC = {}
for r in ROUTERS:
    d = pd.read_json(f"{DATA}/id/voter_decisions/{r}.jsonl", lines=True)
    d = d[d.task_id.isin(TEST)].drop_duplicates("task_id").set_index("task_id").chosen_model
    assert len(d) == 2919, (r, len(d))
    DEC[r] = d.reindex(TEST)
PARTS = {"dim9": "dimension", "cell21": "cell21", "all": "all"}
CELLS = {part: sorted(meta[col].unique()) for part, col in PARTS.items()}
prob_ids = {(part, c): PROB[meta.loc[PROB, col] == c] for part, col in PARTS.items() for c in CELLS[part]}
test_ids = {(part, c): TEST[meta.loc[TEST, col] == c] for part, col in PARTS.items() for c in CELLS[part]}
idx_prob = {k: rng.integers(0, len(v), size=(NBOOT, len(v))) for k, v in prob_ids.items()}
idx_test = {k: rng.integers(0, len(v), size=(NBOOT, len(v))) for k, v in test_ids.items()}
# noisy-verifier coin flips on id_test: one uniform per task, reused across settings (fixed seed)
U_TEST = pd.Series(rng.random(len(TEST)), index=TEST)

# reference efficiency on probing (eta_B = vbar/cbar) appended as a fifth lambda
eta_B_prob = float(S.loc[PROB, STRONG].mean() / Cc.loc[PROB, STRONG].mean())
LAMS = LAMBDAS + [round(eta_B_prob, 3)]

def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return (float("nan"), float("nan"))
    lo, hi = np.percentile(x, [2.5, 97.5]); return float(lo), float(hi)

def arrays(ids, cheap):
    return (S.loc[ids, cheap].to_numpy(), S.loc[ids, STRONG].to_numpy(),
            Cc.loc[ids, cheap].to_numpy(), Cc.loc[ids, STRONG].to_numpy())

def safe_div(num, den):
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

def prims(yP, yB, cP, cB, hit, idx=None):
    """Sec. 4 primitives from an outcome sample; idx=(B,n) resamples -> length-B arrays.
    p: hit prob; v_h: primary value on a hit; v_b: reference (Opus) value, unconditional; a: primary bill; b: Opus bill.
    Diagnostics: hit/miss-conditional Opus value and cost, cheap cost, cheap value on a miss."""
    if idx is None: idx = np.arange(len(yP))[None, :]
    H = hit[idx].astype(float); YP = yP[idx]; YB = yB[idx]; CP = cP[idx]; CB = cB[idx]
    n = H.shape[1]; nh = H.sum(1); nm = n - nh
    out = dict(n=np.full(len(nh), n), p=nh / n, v_b=YB.mean(1), a=CP.mean(1), b=CB.mean(1),
               v_h=safe_div((YP * H).sum(1), nh), v_fail=safe_div((YP * (1 - H)).sum(1), nm),
               v_b_hit=safe_div((YB * H).sum(1), nh), v_b_miss=safe_div((YB * (1 - H)).sum(1), nm),
               b_hit=safe_div((CB * H).sum(1), nh), b_miss=safe_div((CB * (1 - H)).sum(1), nm),
               a_hit=safe_div((CP * H).sum(1), nh), a_miss=safe_div((CP * (1 - H)).sum(1), nm))
    return out

def noisy_expected(pr, bfa, bfr):
    """Expected acceptance primitives under flip rates: accepted = hit&~reject | miss&false-accept."""
    p, vh, vf = pr["p"], pr["v_h"], pr["v_fail"]
    pe = p * (1 - bfr) + (1 - p) * bfa
    num = p * (1 - bfr) * np.nan_to_num(vh) + (1 - p) * bfa * np.nan_to_num(vf)
    vh_e = safe_div(num, pe)
    q = dict(pr); q["p"] = pe; q["v_h"] = vh_e; return q

def dJ_pred(pr, lam):
    """prop:threshold prediction: J_cascade - J_direct = p*W_lambda - lambda*a,  W_lambda = lambda*b + v_h - v_b (eq:margin)."""
    p, vh, vb, a, b = pr["p"], pr["v_h"], pr["v_b"], pr["a"], pr["b"]
    W = lam * b + vh - vb
    return np.where(p > 0, p * np.nan_to_num(W) - lam * a, -lam * a), W

results = {"experiment_id": "X3_threshold_voi", "seed": SEED, "nboot": NBOOT, "usetex": USETEX_OK,
           "data": {"n_tasks": 9999, "n_probing": 7080, "n_id_test": 2919, "n_models": 8,
                    "zero_cost_rows_missing_token_record": n_zero_cost, "eta_B_probing_per_dollar": eta_B_prob,
                    "lambdas": LAMS, "ks": KS, "cheaps": CHEAPS, "thetas": THETAS, "noises": NOISES,
                    "cells": {k: len(v) for k, v in CELLS.items()}}}

# ================================================================ (a) prop:threshold, OUT-OF-SAMPLE
rows = []
for part in PARTS:
    for c in CELLS[part]:
        pid, tid = prob_ids[(part, c)], test_ids[(part, c)]
        for cheap in CHEAPS:
            yP, yB, cP, cB = arrays(pid, cheap); tyP, tyB, tcP, tcB = arrays(tid, cheap)
            u = U_TEST.loc[tid].to_numpy()
            for theta in THETAS:
                hit = yP >= theta; thit = tyP >= theta
                pr0 = prims(yP, yB, cP, cB, hit); prB = prims(yP, yB, cP, cB, hit, idx_prob[(part, c)])
                for (bfa, bfr) in NOISES:
                    if bfa or bfr:
                        acc = np.where(thit, u >= bfr, u < bfa)
                        q0, qB = noisy_expected(pr0, bfa, bfr), noisy_expected(prB, bfa, bfr)
                    else:
                        acc = thit; q0, qB = pr0, prB
                    # measured (id_test) primitives, for the in-sample decomposition
                    pt0 = prims(tyP, tyB, tcP, tcB, acc); ptB = prims(tyP, tyB, tcP, tcB, acc, idx_test[(part, c)])
                    for lam in LAMS:
                        pred0, W0 = dJ_pred(q0, lam); predB, WB = dJ_pred(qB, lam)
                        # measured objective difference per task: cascade minus direct (eq:accounting)
                        Vcf = np.where(acc, tyP, tyB); Dcf = tcP + np.where(acc, 0.0, tcB)
                        dj = (Vcf - lam * Dcf) - (tyB - lam * tcB)
                        meas = float(dj.mean()); mlo, mhi = ci(dj[idx_test[(part, c)]].mean(1))
                        plo, phi = ci(predB)
                        # in-sample as-stated formula and exact identity on id_test
                        ins0, _ = dJ_pred(pt0, lam)
                        exact_id = np.where(pt0["p"] > 0, pt0["p"] * (lam * np.nan_to_num(pt0["b_hit"]) + np.nan_to_num(pt0["v_h"]) - np.nan_to_num(pt0["v_b_hit"])) - lam * pt0["a"], -lam * pt0["a"])
                        pred_cf = bool(pred0[0] >= 0); meas_cf = meas >= 0
                        meas_dec = not (mlo <= 0 <= mhi); pred_dec = not (plo <= 0 <= phi)
                        verdict = "INCONCLUSIVE" if not meas_dec else ("PASS" if pred_cf == meas_cf else "FAIL")
                        Wv = float(W0[0]); pstar = float(lam * pr0["a"][0] / Wv) if (Wv > 0) else float("inf")
                        rows.append(dict(part=part, cell=c, cheap=cheap, theta=theta, beta_fa=bfa, beta_fr=bfr, lam=lam,
                            n_prob=len(pid), n_test=len(tid),
                            p=float(q0["p"][0]), v_h=float(q0["v_h"][0]), v_b=float(pr0["v_b"][0]), a=float(pr0["a"][0]), b=float(pr0["b"][0]),
                            W=Wv, p_star=pstar, price_rule_cf=bool(pr0["p"][0] >= pr0["a"][0] / pr0["b"][0]),
                            dJ_pred=float(pred0[0]), dJ_pred_lo=plo, dJ_pred_hi=phi, pred_decided=pred_dec,
                            dJ_meas=meas, dJ_meas_lo=mlo, dJ_meas_hi=mhi, meas_decided=meas_dec,
                            dJ_insample_formula=float(ins0[0]), dJ_exact_identity=float(exact_id[0]),
                            p_test=float(pt0["p"][0]), v_h_test=float(pt0["v_h"][0]), v_b_test=float(pt0["v_b"][0]),
                            v_b_hit_test=float(pt0["v_b_hit"][0]), v_b_miss_test=float(pt0["v_b_miss"][0]),
                            b_test=float(pt0["b"][0]), b_hit_test=float(pt0["b_hit"][0]), b_miss_test=float(pt0["b_miss"][0]),
                            a_test=float(pt0["a"][0]),
                            pred_cheap_first=pred_cf, meas_cheap_first=meas_cf, verdict=verdict,
                            meas_in_pred_CI=bool(plo <= meas <= phi), abs_err=abs(float(pred0[0]) - meas)))
A = pd.DataFrame(rows)
A.to_csv(os.path.join(HERE, "a_threshold_cells.csv"), index=False)
assert np.allclose(A.dJ_exact_identity, A.dJ_meas, atol=1e-12), "accounting identity broken"

def agreement_ci(sub):
    """task-resampling CI of the sign-agreement rate across cells (uses the stored per-cell bootstraps)."""
    aggs = []
    for _, r in sub.iterrows():
        tid = test_ids[(r.part, r.cell)]; yP, yB, cP, cB = arrays(tid, r.cheap)
        if r.beta_fa or r.beta_fr:
            hit = yP >= r.theta; u = U_TEST.loc[tid].to_numpy(); acc = np.where(hit, u >= r.beta_fr, u < r.beta_fa)
        else:
            acc = yP >= r.theta
        Vcf = np.where(acc, yP, yB); Dcf = cP + np.where(acc, 0.0, cB)
        dj = (Vcf - r.lam * Dcf) - (yB - r.lam * cB)
        aggs.append((dj[idx_test[(r.part, r.cell)]].mean(1) >= 0) == r.pred_cheap_first)
    return ci(np.mean(np.array(aggs), axis=0))

def summarize_a(sub, with_ci=False):
    conf = {"pred_CF/meas_CF": int((sub.pred_cheap_first & sub.meas_cheap_first).sum()),
            "pred_D/meas_D": int((~sub.pred_cheap_first & ~sub.meas_cheap_first).sum()),
            "pred_CF/meas_D": int((sub.pred_cheap_first & ~sub.meas_cheap_first).sum()),
            "pred_D/meas_CF": int((~sub.pred_cheap_first & sub.meas_cheap_first).sum())}
    ver = sub.verdict.value_counts().reindex(["PASS", "FAIL", "INCONCLUSIVE"]).fillna(0).astype(int).to_dict()
    dec = sub[sub.meas_decided]
    out = dict(n_cells=int(len(sub)), confusion=conf, verdicts=ver,
               sign_agreement_all=float((sub.pred_cheap_first == sub.meas_cheap_first).mean()),
               sign_agreement_decided=float((dec.pred_cheap_first == dec.meas_cheap_first).mean()) if len(dec) else None,
               baseline_always_cheap_first_decided=float(dec.meas_cheap_first.mean()) if len(dec) else None,
               baseline_price_rule_decided=float((dec.price_rule_cf == dec.meas_cheap_first).mean()) if len(dec) else None,
               frac_meas_inside_pred_CI=float(sub.meas_in_pred_CI.mean()),
               MAE_dJ=float(sub.abs_err.mean()), median_abs_err_dJ=float(sub.abs_err.median()),
               mean_signed_err_pred_minus_meas=float((sub.dJ_pred - sub.dJ_meas).mean()),
               MAE_insample_formula=float((sub.dJ_insample_formula - sub.dJ_meas).abs().mean()),
               n_pred_never_cheap_first_W_le_0=int((sub.W <= 0).sum()))
    if with_ci: out["sign_agreement_all_CI"] = agreement_ci(sub)
    return out

resA = {}
exact = A[(A.beta_fa == 0) & (A.beta_fr == 0)]
for part in ["dim9", "cell21"]:
    sub = exact[exact.part == part]
    resA[f"{part}_exact_all_lambda_theta_cheap"] = summarize_a(sub, with_ci=True)
    for theta in THETAS:
        for cheap in CHEAPS:
            resA[f"{part}_exact_theta{theta}_{cheap}"] = summarize_a(sub[(sub.theta == theta) & (sub.cheap == cheap)])
    for lam in LAMS:
        resA[f"{part}_exact_lambda{lam}"] = summarize_a(sub[sub.lam == lam], with_ci=True)
    for (bfa, bfr) in NOISES[1:]:
        resA[f"{part}_noisy_fa{bfa}_fr{bfr}"] = summarize_a(A[(A.part == part) & (A.beta_fa == bfa) & (A.beta_fr == bfr)])
resA["all_pooled_exact"] = exact[exact.part == "all"][["cheap", "theta", "lam", "p", "v_h", "v_b", "a", "b", "W", "p_star", "dJ_pred", "dJ_pred_lo", "dJ_pred_hi", "dJ_meas", "dJ_meas_lo", "dJ_meas_hi", "verdict", "abs_err"]].to_dict("records")
# --- miss-independence hypothesis (IN-SAMPLE on id_test): v_b_miss vs v_b, b_miss vs b, with CIs
rows = []
for part in PARTS:
    for c in CELLS[part]:
        tid = test_ids[(part, c)]
        for cheap in CHEAPS:
            tyP, tyB, tcP, tcB = arrays(tid, cheap)
            for theta in THETAS:
                hit = tyP >= theta
                p0 = prims(tyP, tyB, tcP, tcB, hit); pB = prims(tyP, tyB, tcP, tcB, hit, idx_test[(part, c)])
                dv = pB["v_b_miss"] - pB["v_b"]; dvh = pB["v_b_hit"] - pB["v_b"]; db = pB["b_miss"] - pB["b"]; dbh = pB["b_hit"] - pB["b"]
                rows.append(dict(part=part, cell=c, cheap=cheap, theta=theta, n_test=len(tid), p_test=float(p0["p"][0]),
                                 v_b=float(p0["v_b"][0]), v_b_miss=float(p0["v_b_miss"][0]), v_b_hit=float(p0["v_b_hit"][0]),
                                 d_vb_miss=float(p0["v_b_miss"][0] - p0["v_b"][0]), d_vb_miss_lo=ci(dv)[0], d_vb_miss_hi=ci(dv)[1],
                                 d_vb_hit=float(p0["v_b_hit"][0] - p0["v_b"][0]), d_vb_hit_lo=ci(dvh)[0], d_vb_hit_hi=ci(dvh)[1],
                                 b=float(p0["b"][0]), b_miss=float(p0["b_miss"][0]), b_hit=float(p0["b_hit"][0]),
                                 d_b_miss=float(p0["b_miss"][0] - p0["b"][0]), d_b_miss_lo=ci(db)[0], d_b_miss_hi=ci(db)[1],
                                 d_b_hit=float(p0["b_hit"][0] - p0["b"][0]), d_b_hit_lo=ci(dbh)[0], d_b_hit_hi=ci(dbh)[1]))
MI = pd.DataFrame(rows); MI.to_csv(os.path.join(HERE, "a_miss_independence_cells.csv"), index=False)
def mi_summary(sub):
    ok = sub[np.isfinite(sub.d_vb_miss)]
    return dict(n_cells=int(len(ok)),
                n_vb_miss_below_vb_CI=int((ok.d_vb_miss_hi < 0).sum()), n_vb_miss_above_vb_CI=int((ok.d_vb_miss_lo > 0).sum()),
                mean_d_vb_miss=float(ok.d_vb_miss.mean()), median_d_vb_miss=float(ok.d_vb_miss.median()),
                mean_d_vb_hit=float(ok.d_vb_hit.mean()),
                n_b_miss_above_b_CI=int((ok.d_b_miss_lo > 0).sum()), n_b_miss_below_b_CI=int((ok.d_b_miss_hi < 0).sum()),
                mean_d_b_miss=float(ok.d_b_miss.mean()), mean_d_b_hit=float(ok.d_b_hit.mean()))
resA["miss_independence"] = {f"{part}_theta{theta}_{cheap}": mi_summary(MI[(MI.part == part) & (MI.theta == theta) & (MI.cheap == cheap)])
                             for part in ["dim9", "cell21"] for theta in THETAS for cheap in CHEAPS}
resA["miss_independence"]["all_pooled"] = MI[MI.part == "all"].drop(columns=["part"]).to_dict("records")
results["a_prop_threshold"] = resA

# ================================================================ (b) refinement: NOT INSTANTIABLE + diagnostic
rows = []
tid = TEST; base = "Qwen3-Max"
yQ, yB, cQ, cB = arrays(tid, base); yK, _, cK, _ = arrays(tid, "kimi-k2.5")
pyQ, pyB, pcQ, pcB = arrays(PROB, base)
for theta in THETAS:
    hitQ = yQ >= theta; hitK = yK >= theta
    pr_base_prob = prims(pyQ, pyB, pcQ, pcB, pyQ >= theta)          # W_lambda from probing base policy
    pr_base = prims(yQ, yB, cQ, cB, hitQ)
    for r in ROUTERS:
        useK = (DEC[r].to_numpy() == "kimi-k2.5")
        yP = np.where(useK, yK, yQ); cP = np.where(useK, cK, cQ); hit = np.where(useK, hitK, hitQ)
        pr_ref = prims(yP, yB, cP, cB, hit)
        for lam in LAMBDAS:
            W = float(lam * pr_base_prob["b"][0] + pr_base_prob["v_h"][0] - pr_base_prob["v_b"][0])
            dp = float(pr_ref["p"][0] - pr_base["p"][0])
            stip = dp * W                                            # stipulated change (k = 0)
            Vb = np.where(hitQ, yQ, yB); Db = cQ + np.where(hitQ, 0, cB); Vr = np.where(hit, yP, yB); Dr = cP + np.where(hit, 0, cB)
            dj = (Vr - lam * Dr) - (Vb - lam * Db); real = float(dj.mean()); lo, hi = ci(dj[idx_test[("all", "ALL")]].mean(1))
            rows.append(dict(theta=theta, router=r, lam=lam, frac_tasks_switched_to_kimi=float(useK.mean()), delta_p=dp,
                             delta_a=float(pr_ref["a"][0] - pr_base["a"][0]), delta_v_h=float(pr_ref["v_h"][0] - pr_base["v_h"][0]),
                             delta_v_b_hit=float(pr_ref["v_b_hit"][0] - pr_base["v_b_hit"][0]), W=W,
                             stipulated_dJ_k0=stip, realized_dJ_k0=real, realized_lo=lo, realized_hi=hi,
                             breakeven_k_stipulated=stip / lam, breakeven_k_realized=real / lam))
Bdf = pd.DataFrame(rows); Bdf.to_csv(os.path.join(HERE, "b_refinement_diagnostic.csv"), index=False)
results["b_refinement"] = {"verdict": "NOT INSTANTIABLE",
    "reason": ["router decision files are contingent choices between fixed actions per task; app:decisions states the refinement clause does not model a signal allowing contingent choice (that is prop:voi / app:imperfect-voi, tested in (d))",
               "the router's own inference cost k is not recorded anywhere in the data (no router token log); k would have to be assumed",
               "the nearest construction (router picks the cheap primary in {Qwen3-Max, kimi-k2.5}, else Qwen3-Max) changes a and v_h, violating 'preserving the other quantities' -- see diagnostic"],
    "diagnostic_rows": Bdf.to_dict("records")}

# ================================================================ AUDIT (b2): the refinement clause IS instantiable as a NON-contingent better proposer.
# app:decisions: "This comparison changes the success probability; it does not model a signal allowing contingent choice between fixed
# actions"; main text: "An information decision differs from a better proposer". A refinement that swaps the primary for every task in
# the state (Qwen3-Max -> another non-Opus model) raises p by Delta p and adds k = Delta a (recorded token-price cost estimates), with v_b, b the state's
# constants and v_h preserved exactly at theta = 1 (both = 1). Prediction (probing): Delta p * W_lambda - lambda * k, W_lambda of the base
# policy (eq:margin). Measurement (id_test): mean_task[J_casc(alt) - J_casc(base)]. OUT-OF-SAMPLE. The only hidden shift is E[Y_B | H=1].
SWAP_ALTS = ["kimi-k2.5", "qwen3.5-plus", "glm-5", "MiniMax-M2.7", "gpt-5.4", "claude-sonnet-4-6"]
rows = []
for part in PARTS:
    for c in CELLS[part]:
        pid, tid = prob_ids[(part, c)], test_ids[(part, c)]; IP, IT = idx_prob[(part, c)], idx_test[(part, c)]
        yQ, yB, cQ, cB = arrays(pid, "Qwen3-Max"); tyQ, tyB, tcQ, tcB = arrays(tid, "Qwen3-Max")
        for theta in THETAS:
            hQ = yQ >= theta; thQ = tyQ >= theta
            pr0 = prims(yQ, yB, cQ, cB, hQ); prB0 = prims(yQ, yB, cQ, cB, hQ, IP)
            pt0 = prims(tyQ, tyB, tcQ, tcB, thQ)
            for m in SWAP_ALTS:
                yM, _, cM, _ = arrays(pid, m); tyM, _, tcM, _ = arrays(tid, m); hM = yM >= theta; thM = tyM >= theta
                pr1 = prims(yM, yB, cM, cB, hM); prB1 = prims(yM, yB, cM, cB, hM, IP); pt1 = prims(tyM, tyB, tcM, tcB, thM)
                dp0 = float(pr1["p"][0] - pr0["p"][0]); k0 = float(pr1["a"][0] - pr0["a"][0])
                dpB = prB1["p"] - prB0["p"]; kB = prB1["a"] - prB0["a"]
                for lam in LAMS:
                    W0 = float(lam * pr0["b"][0] + pr0["v_h"][0] - pr0["v_b"][0]); WB = lam * prB0["b"] + prB0["v_h"] - prB0["v_b"]
                    pred0 = dp0 * W0 - lam * k0; plo, phi = ci(dpB * WB - lam * kB)
                    J0 = np.where(thQ, tyQ, tyB) - lam * (tcQ + np.where(thQ, 0.0, tcB)); J1 = np.where(thM, tyM, tyB) - lam * (tcM + np.where(thM, 0.0, tcB))
                    d = J1 - J0; meas = float(d.mean()); mlo, mhi = ci(d[IT].mean(1))
                    defined = bool(np.isfinite(W0)); meas_dec = not (mlo <= 0 <= mhi)
                    verdict = "UNDEFINED" if not defined else ("INCONCLUSIVE" if not meas_dec else ("PASS" if (pred0 >= 0) == (meas >= 0) else "FAIL"))
                    rows.append(dict(part=part, cell=c, theta=theta, alt=m, lam=lam, n_prob=len(pid), n_test=len(tid),
                                     p_base=float(pr0["p"][0]), delta_p=dp0, k=k0, W=W0, delta_v_h_prob=float(pr1["v_h"][0] - pr0["v_h"][0]),
                                     delta_v_b_hit_test=float(pt1["v_b_hit"][0] - pt0["v_b_hit"][0]), delta_b_hit_test=float(pt1["b_hit"][0] - pt0["b_hit"][0]),
                                     dJ_pred=pred0, dJ_pred_lo=plo, dJ_pred_hi=phi, pred_decided=not (plo <= 0 <= phi),
                                     dJ_meas=meas, dJ_meas_lo=mlo, dJ_meas_hi=mhi, meas_decided=meas_dec,
                                     pred_worthwhile=bool(pred0 >= 0), meas_worthwhile=bool(meas >= 0), verdict=verdict,
                                     meas_in_pred_CI=bool(plo <= meas <= phi), abs_err=abs(pred0 - meas)))
B2 = pd.DataFrame(rows); B2.to_csv(os.path.join(HERE, "b2_refinement_swap_cells.csv"), index=False)
def summarize_b2(sub):
    ok = sub[sub.verdict != "UNDEFINED"]; dec = ok[ok.meas_decided]
    return dict(n_rows=int(len(sub)), n_undefined_base_p0=int((sub.verdict == "UNDEFINED").sum()),
                verdicts=ok.verdict.value_counts().reindex(["PASS", "FAIL", "INCONCLUSIVE"]).fillna(0).astype(int).to_dict(),
                sign_agreement_decided=float((dec.pred_worthwhile == dec.meas_worthwhile).mean()) if len(dec) else None,
                baseline_never_swap_decided=float((~dec.meas_worthwhile).mean()) if len(dec) else None,
                n_pred_worthwhile_decided=int(dec.pred_worthwhile.sum()), n_meas_worthwhile_decided=int(dec.meas_worthwhile.sum()),
                n_fail_with_pred_CI_covering_0=int(((ok.verdict == "FAIL") & ~ok.pred_decided).sum()),
                frac_meas_inside_pred_CI=float(ok.meas_in_pred_CI.mean()) if len(ok) else None,
                MAE_dJ=float(ok.abs_err.mean()) if len(ok) else None, mean_signed_err_pred_minus_meas=float((ok.dJ_pred - ok.dJ_meas).mean()) if len(ok) else None,
                max_abs_delta_v_h_prob=float(ok.delta_v_h_prob.abs().max()) if len(ok) else None,
                mean_abs_delta_v_b_hit_test=float(ok.delta_v_b_hit_test.abs().mean()) if len(ok) else None)
swap = {}
for part in ["dim9", "cell21"]:
    for theta in THETAS:
        sub = B2[(B2.part == part) & (B2.theta == theta)]
        swap[f"{part}_theta{theta}"] = summarize_b2(sub)
        for lam in LAMS: swap[f"{part}_theta{theta}_lambda{lam}"] = summarize_b2(sub[sub.lam == lam])
swap["all_pooled"] = B2[B2.part == "all"][["theta", "alt", "lam", "p_base", "delta_p", "k", "W", "delta_v_h_prob", "delta_v_b_hit_test", "dJ_pred", "dJ_pred_lo", "dJ_pred_hi", "dJ_meas", "dJ_meas_lo", "dJ_meas_hi", "verdict"]].to_dict("records")
_d1 = B2[(B2.part != "all") & (B2.theta == 1.0) & B2.meas_decided & (B2.verdict != "UNDEFINED")]
swap["verdict"] = ("PASS (qualified) as an out-of-sample sign rule at theta=1: decided agreement %.3f over %d dim9+cell21 rows; the 'never swap' baseline scores %.3f "
                   "(Qwen3-Max is already the cheapest primary, so most swaps are not worthwhile); skill over the baseline appears only at lambda >= 10; "
                   "theta=0.5 violates the 'preserving v_h' clause (|Delta v_h| up to %.3f)"
                   % ((_d1.pred_worthwhile == _d1.meas_worthwhile).mean(), len(_d1), (~_d1.meas_worthwhile).mean(),
                      B2[(B2.part != "all") & (B2.theta == 0.5)].delta_v_h_prob.abs().max()))
results["b_refinement"]["contingent_router_construction"] = {"verdict": "NOT INSTANTIABLE", "reason": results["b_refinement"]["reason"]}
results["b_refinement"]["swap_test"] = swap
results["b_refinement"]["verdict"] = "see swap_test.verdict (AUDIT): the clause is instantiable as a non-contingent primary swap with k = Delta a"

# ================================================================ (c) prop:voi / eq:gate-costs
# (c1) IDENTITY check with real per-cell a, b and hypothetical k
grid = np.linspace(0, 1, 2001); n_checks = 0; n_fail = 0
for part in ["dim9", "cell21"]:
    for c in CELLS[part]:
        for cheap in CHEAPS:
            yP, yB, cP, cB = arrays(prob_ids[(part, c)], cheap); a, b = cP.mean(), cB.mean()
            for k in KS:
                lo, hi = k / (b - a), 1 - k / a
                nonempty = lo < hi
                n_checks += 1
                if nonempty != (k < a * (b - a) / b): n_fail += 1
                if nonempty and not (lo < a / b < hi): n_fail += 1
                Cc_ = a + (1 - grid) * b; Cd_ = np.full_like(grid, b); Ci_ = k + grid * a + (1 - grid) * b
                inside = (grid > lo + 1e-12) & (grid < hi - 1e-12); cheapest = (Ci_ < Cc_ - 1e-15) & (Ci_ < Cd_ - 1e-15)
                on_edge = (np.abs(grid - lo) < 1e-9) | (np.abs(grid - hi) < 1e-9)
                if not np.array_equal(inside[~on_edge], cheapest[~on_edge]): n_fail += 1
results["c_prop_voi"] = {"identity_check": {"n_cell_k_combinations": n_checks, "n_failures": n_fail, "verdict": "PASS (IDENTITY; code check only)" if n_fail == 0 else "FAIL"}}

# (c2) substantive: realized costs with exact information (the data contain the sufficiency label)
rows = []
for part in PARTS:
    for c in CELLS[part]:
        pid, tid = prob_ids[(part, c)], test_ids[(part, c)]
        for cheap in CHEAPS:
            yP, yB, cP, cB = arrays(pid, cheap); tyP, tyB, tcP, tcB = arrays(tid, cheap)
            for theta in THETAS:
                hit = yP >= theta; thit = tyP >= theta
                s_pr, a_pr, b_pr = hit.mean(), cP.mean(), cB.mean()
                s_te, a_te, b_te = thit.mean(), tcP.mean(), tcB.mean()
                casc = tcP + np.where(thit, 0.0, tcB); dire = tcB; info0 = np.where(thit, tcP, tcB)
                I = idx_test[(part, c)]
                cascB, direB, info0B = casc[I].mean(1), dire[I].mean(1), info0[I].mean(1)
                v_casc = float(np.where(thit, tyP, tyB).mean()); v_dir = float(tyB.mean())
                for k in KS:
                    for src, (s_, a_, b_) in {"oos": (s_pr, a_pr, b_pr), "ins": (s_te, a_te, b_te)}.items():
                        lo, hi = k / (b_ - a_), 1 - k / a_
                        pred_in = bool(lo < s_ < hi)
                        th = {"cascade": a_ + (1 - s_) * b_, "direct": b_, "info": k + s_ * a_ + (1 - s_) * b_}
                        pred_arg = min(th, key=th.get)
                        real = {"cascade": float(casc.mean()), "direct": float(dire.mean()), "info": float(info0.mean() + k)}
                        real_arg = min(real, key=real.get)
                        m = (info0B + k) - np.minimum(cascB, direB); mlo, mhi = ci(m)
                        real_in = real_arg == "info"; dec = not (mlo <= 0 <= mhi)
                        verdict = "INCONCLUSIVE" if not dec else ("PASS" if pred_in == real_in else "FAIL")
                        rows.append(dict(part=part, cell=c, cheap=cheap, theta=theta, k=k, src=src, n_test=len(tid), s=s_, a=a_, b=b_,
                                         band_lo=lo, band_hi=hi, band_nonempty=bool(lo < hi), k_max=a_ * (b_ - a_) / b_, u=(s_ - lo) / (hi - lo) if hi > lo else float("nan"),
                                         pred_info_in_band=pred_in, pred_argmin=pred_arg,
                                         C_cascade_real=real["cascade"], C_direct_real=real["direct"], C_info_real=real["info"], real_argmin=real_arg,
                                         info_minus_best_uninformed=float(m.mean()) if False else float(real["info"] - min(real["cascade"], real["direct"])),
                                         info_minus_best_lo=mlo, info_minus_best_hi=mhi, real_info_cheapest=real_in, decided=dec, verdict=verdict,
                                         V_cascade=v_casc, V_direct=v_dir))
Cdf = pd.DataFrame(rows); Cdf.to_csv(os.path.join(HERE, "c_voi_band_cells.csv"), index=False)
def summarize_c(sub):
    ver = sub.verdict.value_counts().reindex(["PASS", "FAIL", "INCONCLUSIVE"]).fillna(0).astype(int).to_dict()
    dec = sub[sub.decided]
    return dict(n_cells=int(len(sub)), verdicts=ver, n_pred_in_band=int(sub.pred_info_in_band.sum()), n_real_info_cheapest=int(sub.real_info_cheapest.sum()),
                confusion={"in/in": int((sub.pred_info_in_band & sub.real_info_cheapest).sum()), "out/out": int((~sub.pred_info_in_band & ~sub.real_info_cheapest).sum()),
                           "in/out": int((sub.pred_info_in_band & ~sub.real_info_cheapest).sum()), "out/in": int((~sub.pred_info_in_band & sub.real_info_cheapest).sum())},
                agreement_all=float((sub.pred_info_in_band == sub.real_info_cheapest).mean()),
                agreement_decided=float((dec.pred_info_in_band == dec.real_info_cheapest).mean()) if len(dec) else None,
                argmin_agreement_all=float((sub.pred_argmin == sub.real_argmin).mean()),
                argmin_agreement_decided=float((dec.pred_argmin == dec.real_argmin).mean()) if len(dec) else None,
                n_band_nonempty=int(sub.band_nonempty.sum()))
resC = results["c_prop_voi"]
for src in ["oos", "ins"]:
    for theta in THETAS:
        sub = Cdf[(Cdf.src == src) & (Cdf.theta == theta) & (Cdf.part != "all")]
        resC[f"{src}_theta{theta}_dim9+cell21"] = summarize_c(sub)
        for k in KS: resC[f"{src}_theta{theta}_k{k}"] = summarize_c(sub[sub.k == k])
        for cheap in CHEAPS: resC[f"{src}_theta{theta}_{cheap}"] = summarize_c(sub[sub.cheap == cheap])
resC["all_pooled"] = Cdf[Cdf.part == "all"][["cheap", "theta", "k", "src", "s", "a", "b", "band_lo", "band_hi", "k_max", "pred_argmin", "C_cascade_real", "C_direct_real", "C_info_real", "real_argmin", "info_minus_best_lo", "info_minus_best_hi", "verdict", "V_cascade", "V_direct"]].to_dict("records")
fails = Cdf[(Cdf.verdict == "FAIL") & (Cdf.part != "all")]
resC["failures"] = fails[["part", "cell", "cheap", "theta", "k", "src", "n_test", "s", "band_lo", "band_hi", "u", "pred_argmin", "real_argmin", "info_minus_best_uninformed", "info_minus_best_lo", "info_minus_best_hi"]].to_dict("records")

# ================================================================ (d) Blackwell bound with real routers as experiments Z
rows = []
inv = {cheap: int((Cc.loc[TEST, STRONG] < Cc.loc[TEST, cheap]).sum()) for cheap in CHEAPS}
for part in PARTS:
    for c in CELLS[part]:
        tid = test_ids[(part, c)]; I = idx_test[(part, c)]
        for cheap in CHEAPS:
            tyP, tyB, tcP, tcB = arrays(tid, cheap)
            for theta in THETAS:
                h = tyP >= theta
                casc = tcP + np.where(h, 0.0, tcB); dire = tcB; perf = np.where(h, tcP, tcB)
                cascB, direB, perfB = casc[I].mean(1), dire[I].mean(1), perf[I].mean(1)
                fB = np.minimum(cascB, direB); boundB = fB - perfB
                f0 = min(casc.mean(), dire.mean()); bound0 = f0 - perf.mean()
                s = h.mean(); a, b = tcP.mean(), tcB.mean(); bound_theory = min(s * (b - a), a * (1 - s))
                for r in ROUTERS:
                    ch = DEC[r].loc[tid].to_numpy()
                    for zdef, Z in {"chose_this_cheap": ch == cheap, "chose_any_non_opus": ch != STRONG}.items():
                        cz = np.where(Z, casc, dire); czB = cz[I].mean(1)
                        save0 = f0 - cz.mean(); saveB = fB - czB
                        frac0 = save0 / bound0 if bound0 > 1e-6 else float("nan")
                        fracB = np.where(boundB > 1e-6, saveB / np.where(boundB > 1e-6, boundB, 1), np.nan)
                        # Blackwell optimal use of Z (theory costs with cell means): E f(s_Z)
                        z = Z.mean(); s1 = h[Z].mean() if Z.any() else float("nan"); s0 = h[~Z].mean() if (~Z).any() else float("nan")
                        fth = lambda t: min(a + (1 - t) * b, b)
                        Ef_sZ = (z * fth(s1) if Z.any() else 0) + ((1 - z) * fth(s0) if (~Z).any() else 0)
                        val_opt = fth(s) - Ef_sZ
                        vz = float(np.where(Z, np.where(h, tyP, tyB), tyB).mean())
                        rows.append(dict(part=part, cell=c, cheap=cheap, theta=theta, router=r, zdef=zdef, n_test=len(tid), s=s, a=a, b=b,
                                         P_Z1=float(z), s_given_Z1=float(s1), s_given_Z0=float(s0),
                                         C_cascade=float(casc.mean()), C_direct=float(dire.mean()), C_perfect=float(perf.mean()), C_Z=float(cz.mean()),
                                         f_real=float(f0), bound_real=float(bound0), bound_real_lo=ci(boundB)[0], bound_real_hi=ci(boundB)[1],
                                         bound_theory_cellmeans=float(bound_theory),
                                         saving_Z=float(save0), saving_lo=ci(saveB)[0], saving_hi=ci(saveB)[1],
                                         bound_violated_point=bool(save0 > bound0 + 1e-12), excess_over_bound=float(save0 - bound0),
                                         bound_violated_CI=bool(ci(saveB - boundB)[0] > 0),
                                         frac_captured=float(frac0), frac_lo=ci(fracB)[0], frac_hi=ci(fracB)[1],
                                         frac_captured_optimal_use_of_Z=float(val_opt / bound_theory) if bound_theory > 1e-6 else float("nan"),
                                         V_Z=vz, V_direct=float(tyB.mean()), V_perfect_or_cascade=float(np.where(h, tyP, tyB).mean())))
Ddf = pd.DataFrame(rows); Ddf.to_csv(os.path.join(HERE, "d_blackwell_cells.csv"), index=False)
def summarize_d(sub):
    inf = sub[np.isfinite(sub.frac_captured)]
    return dict(n_rows=int(len(sub)), n_bound_violated_point=int(sub.bound_violated_point.sum()), n_bound_violated_CI=int(sub.bound_violated_CI.sum()),
                max_excess_over_bound_dollars_per_task=float(sub.excess_over_bound.max()),
                n_informative_cells=int(len(inf)), frac_captured_mean=float(inf.frac_captured.mean()) if len(inf) else None,
                frac_captured_median=float(inf.frac_captured.median()) if len(inf) else None,
                frac_captured_task_weighted=float((inf.frac_captured * inf.n_test).sum() / inf.n_test.sum()) if len(inf) else None,
                n_cells_frac_negative=int((inf.frac_captured < 0).sum()), n_cells_frac_CI_above_0=int((inf.frac_lo > 0).sum()))
resD = {"n_tasks_opus_cheaper_than_cheap_model_id_test": inv,
        "verdict_bound": None, "pooled_all": Ddf[Ddf.part == "all"][["cheap", "theta", "router", "zdef", "P_Z1", "s", "s_given_Z1", "s_given_Z0", "C_cascade", "C_direct", "C_perfect", "C_Z", "f_real", "bound_real", "saving_Z", "saving_lo", "saving_hi", "bound_violated_point", "frac_captured", "frac_lo", "frac_hi", "frac_captured_optimal_use_of_Z", "V_Z", "V_direct", "V_perfect_or_cascade"]].to_dict("records")}
for part in ["dim9", "cell21"]:
    for theta in THETAS:
        for zdef in ["chose_this_cheap", "chose_any_non_opus"]:
            sub = Ddf[(Ddf.part == part) & (Ddf.theta == theta) & (Ddf.zdef == zdef)]
            resD[f"{part}_theta{theta}_{zdef}"] = summarize_d(sub)
            for r in ROUTERS:
                for cheap in CHEAPS:
                    resD[f"{part}_theta{theta}_{zdef}_{r}_{cheap}"] = summarize_d(sub[(sub.router == r) & (sub.cheap == cheap)])
viol = Ddf[Ddf.bound_violated_point & (Ddf.part != "all")]
resD["bound_violations_point"] = viol[["part", "cell", "cheap", "theta", "router", "zdef", "n_test", "saving_Z", "bound_real", "excess_over_bound", "bound_violated_CI"]].to_dict("records")
resD["verdict_bound"] = ("PASS" if int(Ddf.bound_violated_CI.sum()) == 0 else "FAIL") + f" ({int(Ddf.bound_violated_point.sum())} point violations, {int(Ddf.bound_violated_CI.sum())} CI-supported, of {len(Ddf)} router x cell rows)"
results["d_blackwell"] = resD

# ================================================================ AUDIT bookkeeping (no verdict rule above was changed)
aud = {}
_ex = A[(A.beta_fa == 0) & (A.beta_fr == 0) & A.part.isin(["dim9", "cell21"])]
for part in ["dim9", "cell21"]:
    s = _ex[_ex.part == part]; f = s[s.verdict == "FAIL"]; dec = s[s.meas_decided]
    deg = s[s.p_test == 0]; nd = dec[dec.p_test > 0]; diff = nd[nd.pred_cheap_first != nd.price_rule_cf]
    aud[f"a_{part}"] = dict(
        n_FAIL=int(len(f)), n_FAIL_zero_id_test_hits=int((f.p_test == 0).sum()), n_FAIL_pred_CI_covers_0=int((~f.pred_decided).sum()),
        n_FAIL_clean_two_sided=int(((f.p_test > 0) & f.pred_decided).sum()),
        n_degenerate_rows_zero_id_test_hits=int(len(deg)), degenerate_verdicts=deg.verdict.value_counts().to_dict(),
        decided_agreement_excluding_degenerate=float((nd.pred_cheap_first == nd.meas_cheap_first).mean()), n_decided_excluding_degenerate=int(len(nd)),
        baseline_any_probing_hit_decided=float(((dec.p > 0) == dec.meas_cheap_first).mean()),
        baseline_any_probing_hit_by_lambda={str(l): float(((g.p > 0) == g.meas_cheap_first).mean()) for l, g in dec.groupby("lam")},
        theorem_by_lambda={str(l): float((g.pred_cheap_first == g.meas_cheap_first).mean()) for l, g in dec.groupby("lam")},
        n_rows_theorem_disagrees_with_price_rule=int(len(diff)), n_theorem_right_when_disagree=int((diff.pred_cheap_first == diff.meas_cheap_first).sum()),
        max_p_star_W_positive=float(s[(s.W > 0)].p_star.max()), n_rows_p_star_above_p=int(((s.W > 0) & (s.p_star > s.p)).sum()))
_c = Cdf[(Cdf.src == "oos") & Cdf.part.isin(["dim9", "cell21"])]
for theta in THETAS:
    d = _c[(_c.theta == theta) & _c.decided]
    aud[f"c_theta{theta}"] = dict(n_decided=int(len(d)), theorem_decided=float((d.pred_info_in_band == d.real_info_cheapest).mean()),
        baseline_info_iff_band_nonempty=float((d.band_nonempty == d.real_info_cheapest).mean()),
        baseline_info_iff_k_le_1e4=float(((d.k <= 1e-4) == d.real_info_cheapest).mean()),
        n_rows_theorem_differs_from_nonempty_baseline=int((d.pred_info_in_band != d.band_nonempty).sum()),
        n_theorem_right_there=int(((d.pred_info_in_band != d.band_nonempty) & (d.pred_info_in_band == d.real_info_cheapest)).sum()))
_dd = Ddf[Ddf.part.isin(["dim9", "cell21"]) & (Ddf.bound_theory_cellmeans > 1e-6)]
_pp = Ddf[(Ddf.part == "all") & (Ddf.bound_theory_cellmeans > 1e-6)]
aud["d_closed_form_bound"] = dict(
    note="saving_Z <= bound_real is an arithmetic identity whenever cost_cheap <= cost_Opus task by task; the paper's closed form min{s(b-a), a(1-s)} with cell means is the non-tautological comparison",
    n_rows_dim9_cell21=int(len(_dd)), n_violations_point=int((_dd.saving_Z > _dd.bound_theory_cellmeans + 1e-12).sum()),
    n_violations_CI=int((_dd.saving_lo > _dd.bound_theory_cellmeans).sum()),
    max_ratio_saving_over_closed_form=float((_dd.saving_Z / _dd.bound_theory_cellmeans).max()),
    n_rows_pooled=int(len(_pp)), n_violations_pooled=int((_pp.saving_Z > _pp.bound_theory_cellmeans + 1e-12).sum()),
    ratio_bound_real_over_closed_form_per_cell=dict(zip(["n", "median", "q25", "q75", "min", "max"],
        [float(x) for x in (lambda r: [len(r), r.median(), r.quantile(.25), r.quantile(.75), r.min(), r.max()])(
            (lambda u: u.bound_real / u.bound_theory_cellmeans)(_dd.drop_duplicates(["part", "cell", "cheap", "theta"])))])))
results["audit"] = aud

# ================================================================ figures (vector PDF)
# Fig 1: prop:threshold predicted vs measured dJ (dim9+cell21, exact verifier), and the miss-independence violation
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.35), layout='constrained')
ax = axes[0]
sub = exact[exact.part.isin(["dim9", "cell21"])]
cols = {0.1: '.7', 1.0: GREEN, 10.0: BLUE, LAMS[-1]: '.3', 100.0: RED}
for lam in LAMS:
    ss = sub[sub.lam == lam]
    ax.scatter(ss.dJ_pred, ss.dJ_meas, s=7, color=cols[lam], alpha=.75, linewidths=0, label=fr'$\lambda={lam:g}$')
lim = max(sub.dJ_pred.abs().max(), sub.dJ_meas.abs().max()) * 1.1
ax.plot([-lim, lim], [-lim, lim], color='.6', lw=.65, ls=':')
ax.axhline(0, color='.6', lw=.65); ax.axvline(0, color='.6', lw=.65)
ax.set_xscale('symlog', linthresh=1e-3); ax.set_yscale('symlog', linthresh=1e-3)
ax.set(xlabel=r'Predicted $pW_\lambda-\lambda a$ (probing)', ylabel=r'Measured $J_\lambda^{\rm casc}-J_\lambda^{\rm direct}$ (id\_test)')
ax.legend(loc='upper left', frameon=False, fontsize=7, handletextpad=.1, borderpad=.2)
ax = axes[1]
mi = MI[MI.part.isin(["dim9", "cell21"]) & (MI.theta == 1.0) & np.isfinite(MI.v_b_miss)]
for cheap, col, mk in zip(CHEAPS, [BLUE, RED], ['o', 's']):
    mm = mi[mi.cheap == cheap]
    ax.errorbar(mm.v_b, mm.v_b_miss, yerr=[mm.v_b_miss - mm.d_vb_miss_lo - mm.v_b, mm.d_vb_miss_hi + mm.v_b - mm.v_b_miss], fmt=mk, ms=3, color=col, elinewidth=.5, capsize=0, label=cheap, alpha=.8)
ax.plot([0, 1], [0, 1], color='.6', lw=.65, ls=':')
ax.set(xlabel=r'$v_b=\mathbb{E}[Y_B]$ in cell (id\_test)', ylabel=r'$\mathbb{E}[Y_B\mid H=0]$ ($\theta=1$)', xlim=(0, 1), ylim=(0, 1))
ax.legend(loc='upper left', frameon=False, fontsize=7)
fig.savefig(os.path.join(FIG, "fig1_threshold.pdf"), metadata={'Creator': 'X3 run.py'}); plt.close(fig)

# Fig 2: VOI band -- normalized position u of s inside the band vs realized info-minus-best-uninformed; Blackwell fraction pooled
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.35), layout='constrained')
ax = axes[0]
sub = Cdf[(Cdf.src == "oos") & (Cdf.theta == 1.0) & Cdf.part.isin(["dim9", "cell21"]) & Cdf.band_nonempty]
kcols = {3e-5: GREEN, 1e-4: BLUE, 5e-4: RED, 1e-3: '.3'}
for k in KS:
    ss = sub[sub.k == k]
    if len(ss) == 0: continue
    ax.scatter(np.clip(ss.u, -0.6, 1.6), 1e3 * ss.info_minus_best_uninformed, s=8, color=kcols[k], alpha=.8, linewidths=0, label=fr'$k=\${k:g}$')
ax.axvspan(0, 1, color=GREEN, alpha=.10); ax.axhline(0, color='.6', lw=.65)
ax.set(xlabel=r'Position of $s$ in band: $(s-\frac{k}{b-a})/(1-\frac{k}{a}-\frac{k}{b-a})$', ylabel=r'$C_{\rm info}-\min\{C_{\rm casc},C_{\rm direct}\}$ (m\$/task)')
ax.set_yscale('symlog', linthresh=0.01); ax.set_yticks([-1, -0.1, -0.01, 0, 0.01, 0.1, 1]); ax.set_yticklabels([r'$-1$', r'$-0.1$', r'$-0.01$', r'$0$', r'$0.01$', r'$0.1$', r'$1$'])
ax.legend(loc='lower left', frameon=False, fontsize=7)
ax = axes[1]
pool = Ddf[(Ddf.part == "all") & (Ddf.theta == 1.0) & (Ddf.zdef == "chose_this_cheap")]
xs = np.arange(len(ROUTERS)); w = .38
for i, (cheap, col) in enumerate(zip(CHEAPS, [BLUE, RED])):
    pp = pool[pool.cheap == cheap].set_index("router").loc[ROUTERS]
    ax.bar(xs + (i - .5) * w, pp.frac_captured, w, color=col, label=cheap, yerr=[pp.frac_captured - pp.frac_lo, pp.frac_hi - pp.frac_captured], error_kw=dict(lw=.6), alpha=.9)
ax.axhline(1, color='.4', lw=.65, ls='--'); ax.axhline(0, color='.6', lw=.65)
ax.set_xticks(xs); ax.set_xticklabels([r.replace("finetuned_router_qwen35_", "ft-").replace("_v3", "").replace("_", r"\_") for r in ROUTERS], rotation=45, ha='right', fontsize=6.5)
ax.set(ylabel=r'Fraction of $f(s)-g(s)$ captured', ylim=(min(-0.1, pool.frac_lo.min() - .1), 1.6))
ax.legend(loc='upper center', ncol=2, frameon=False, fontsize=7)
fig.savefig(os.path.join(FIG, "fig2_voi_blackwell.pdf"), metadata={'Creator': 'X3 run.py'}); plt.close(fig)

# Fig 3: heatmap of fraction captured per router x cell21 (theta=1, Qwen3-Max, Z = chose Qwen3-Max)
fig, ax = plt.subplots(figsize=(6.75, 2.6), layout='constrained')
hm = Ddf[(Ddf.part == "cell21") & (Ddf.theta == 1.0) & (Ddf.cheap == "Qwen3-Max") & (Ddf.zdef == "chose_this_cheap")].pivot(index="router", columns="cell", values="frac_captured").loc[ROUTERS]
from matplotlib.colors import TwoSlopeNorm
# AUDIT: imshow embedded 3 raster XObjects (paper figures must be vector); pcolormesh + non-rasterised colorbar is fully vector
M = np.ma.masked_invalid(np.clip(hm.to_numpy(), -3, 1))
im = ax.pcolormesh(np.arange(hm.shape[1] + 1), np.arange(hm.shape[0] + 1), M, cmap='RdBu', norm=TwoSlopeNorm(vcenter=0, vmin=-3, vmax=1), edgecolors='none', rasterized=False)
ax.invert_yaxis()
ax.set_xticks(np.arange(hm.shape[1]) + .5); ax.set_xticklabels([c.replace("_", r"\_") for c in hm.columns], rotation=60, ha='right', fontsize=6)
ax.set_yticks(np.arange(hm.shape[0]) + .5); ax.set_yticklabels([r.replace("finetuned_router_qwen35_", "ft-").replace("_v3", "").replace("_", r"\_") for r in hm.index], fontsize=7)
ax.grid(False); ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.02); cb.solids.set_rasterized(False)
cb.set_label(r'Captured fraction (clipped at $-3$)', fontsize=7)  # AUDIT: blank cells have f-g < 1e-6 (README)
fig.savefig(os.path.join(FIG, "fig3_blackwell_heatmap.pdf"), metadata={'Creator': 'X3 run.py'}); plt.close(fig)

# ================================================================ write results.json
def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (isinstance(o, float) or True) and not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return o
results["runtime_seconds"] = time.time() - T0
with open(os.path.join(HERE, "results.json"), "w") as f: json.dump(clean(results), f, indent=1)
print(json.dumps(clean({k: results["a_prop_threshold"][k] for k in ["dim9_exact_all_lambda_theta_cheap", "cell21_exact_all_lambda_theta_cheap"]}), indent=1))
print("runtime", round(results["runtime_seconds"], 1), "s")

