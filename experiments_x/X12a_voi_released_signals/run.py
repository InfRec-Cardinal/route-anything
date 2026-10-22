#!/usr/bin/env python3
"""
X12a_voi_released_signals -- "Route Anything" experiment X12, OFFLINE part (X12-a):
the value of the routing information carried by (a) the nine RELEASED ex-ante router decision files of
CodeRouterBench (+ the ACRouter release lookup policy) and (b) a self-trained logistic-regression router on
RouterBench.  The live-judge signal part (X12-b) is NOT run here (pending).

Theory objects (labels as in main_body.tex, appendix_core.tex):
  eq:gate-costs          C_cascade(s) = a + (1-s) b,  C_direct = b,  C_info = k + s a + (1-s) b
  prop:voi (eq:band)     exact information strictly cheapest iff k/(b-a) < s < 1-k/a; nonempty iff k < a(b-a)/b
  eq:imperfect-voi       net value of an imperfect experiment Z:  f(s) - E f(s_Z) - k,   f(t) = min{a+(1-t)b, b}
  app:imperfect-voi      f(s) - E f(s_Z) <= f(s) - g(s) = min{s(b-a), a(1-s)}  (Blackwell / Jensen);  pays only if k < a;
                         a learned score used through "s_hat >= a/b" can lose against no signal
  cor:info-vs-proposer   C_info - C_dec = b Delta p - a(1-s) at equal cost k

Definitions copied from experiments/X3_threshold_voi/run.py (READ ONLY there):
  f(s) = min{a+(1-s)b, b}; g(s) = s a + (1-s) b; sufficiency H = 1{score_primary >= theta};
  realized cascade cost = cost_P + (1-H) cost_B; direct = cost_B; perfect information = H cost_P + (1-H) cost_B;
  capture ratio = saving / (f(s)-g(s)); decided cell = realized 95% CI excludes 0.

Claims: C1 bound never exceeded (IDENTITY-like inequality on expectations, realized OUT-OF-SAMPLE);
        C2 Spearman(capture, AUROC) > Spearman(capture, ECE) (OUT-OF-SAMPLE);
        C3 permuted-Z net VOI ~ -k (SEMI-SYNTHETIC); C4 cor:info-vs-proposer sign agreement (OUT-OF-SAMPLE);
        C5 negative native capture of the classical routers reproduced and explained (IN-SAMPLE).
numpy / pandas / scipy / scikit-learn / matplotlib only.  Seed 20260917.  Deterministic.  Dollars throughout.
"""
import os, sys, json, time, math, ast, csv, collections
import numpy as np, pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T0 = time.time()
SEED = 20260917
NBOOT = 1000                     # paired task resamples, 95% percentile CIs
NSPLIT = 10                      # random stratified half-splits of id_test (cross-fitted calibration of binary released signals)
NPERM = 20                       # permutations of Z within state (SEMI-SYNTHETIC uninformative signal)
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
ACR = os.environ.get("ACROUTER_DATA", "data/agent-as-a-router/data")
RB_PKL = os.environ.get("ROUTERBENCH_PKL", "./data/routerbench/routerbench_0shot.pkl")
ACR_RELEASE = os.environ.get("ACROUTER_RELEASE_DECISIONS", "data/agent-as-a-router/outputs/current/id_decisions.jsonl")
ROUTER_TOKEN_PRICE = 0.054e-6    # $/token: self-hosted router price used by the protocol (X7 R1); k_tok = prompt tokens x price

STRONG = "claude-opus-4-6"       # reference B
CRB_PRIMARIES = ["Qwen3-Max", "kimi-k2.5", "qwen3.5-plus", "MiniMax-M2.7", "glm-5", "gpt-5.4", "claude-sonnet-4-6"]
THETAS = [1.0, 0.5]; THETA_MAIN = 1.0
ROUTERS = ["finetuned_router_qwen35_08b", "finetuned_router_qwen35_2b", "finetuned_router_qwen35_9b_v3",
           "finetuned_router_qwen35_27b_v3", "logreg", "tfidf_mlp", "routellm_mf", "routellm_sw", "dimension_best"]
FINETUNED = set(ROUTERS[:4]); CLASSICAL = set(ROUTERS[4:])
SIGNALS_CRB = ROUTERS + ["acrouter_release"]
ZDEFS = ["any_non_opus", "this_primary"]; ZDEF_MAIN = "any_non_opus"
K_REL = [0.001, 0.01]            # router cost sensitivity: k = K_REL x b

RB_REF = "gpt-4-1106-preview"
RB_PRIMARIES = ["mistralai/mixtral-8x7b-chat", "zero-one-ai/Yi-34B-Chat", "claude-instant-v1", "gpt-3.5-turbo-1106"]
RB_THETA = 0.5
RB_ZVARIANTS = ["lr_tfidf", "lr_tfidf_uni", "lr_tfidf_shrunk"]   # shrunk = lr_tfidf/10: the paper's miscalibration example (SEMI-SYNTHETIC)
RB_REAL_VARIANTS = ["lr_tfidf", "lr_tfidf_uni"]

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'text.usetex': False,
                     'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 7.5,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': .65, 'lines.linewidth': 1.4})
BLUE, ORANGE, PURPLE, GREY = '#23639b', '#ba5934', '#6a4c93', '#808080'   # categorical triple validated (dataviz validator, light mode)

rng = np.random.default_rng(SEED)
NAN = float("nan")

# ------------------------------------------------------------------------------------------------ helpers
def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return (NAN, NAN)
    lo, hi = np.percentile(x, [2.5, 97.5]); return float(lo), float(hi)

def decided(lo, hi):
    return bool(np.isfinite(lo) and np.isfinite(hi) and not (lo <= 0 <= hi))

def bm(v, idx):
    """bootstrap means of a per-task vector under a (B, n) index matrix"""
    return v[idx].mean(1)

def fdiv(n, d):
    n = np.asarray(n, float); d = np.asarray(d, float)
    return np.where(d != 0, n / np.where(d != 0, d, 1.0), NAN)

def auroc_binary(Z, H):
    n1 = int(H.sum()); n0 = len(H) - n1
    if n1 == 0 or n0 == 0: return NAN
    return float((1 + (Z & H).sum() / n1 - (Z & ~H).sum() / n0) / 2)

def ece_binary(Z, H):
    """ECE of the binary forecast Z (Z=1 read as 'sufficient'): P(Z=1)(1-P(H|Z=1)) + P(Z=0)P(H|Z=0) = error rate of Z as a classifier of H"""
    return float((Z != H).mean())

def auroc_cont(z, H):
    n1 = int(H.sum()); n0 = len(H) - n1
    if n1 == 0 or n0 == 0: return NAN
    r = stats.rankdata(z); return float((r[H].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def auroc_cont_boot(z, Hf, idx):
    zb = z[idx]; hb = Hf[idx]; r = stats.rankdata(zb, axis=1)
    n1 = hb.sum(1); n0 = idx.shape[1] - n1
    out = fdiv((r * hb).sum(1) - n1 * (n1 + 1) / 2, n1 * n0); out[(n1 == 0) | (n0 == 0)] = NAN; return out

def ece_bins(z, nb=10):
    edges = np.unique(np.quantile(z, np.linspace(0, 1, nb + 1)[1:-1])); return np.searchsorted(edges, z, side="right"), len(edges) + 1

def ece_cont(z, H, nb=10):
    b, nbin = ece_bins(z, nb); tot = 0.0
    for j in range(nbin):
        m = b == j
        if m.any(): tot += abs(H[m].sum() - z[m].sum())
    return float(tot / len(z))

def ece_cont_boot(z, Hf, idx, nb=10):
    b, nbin = ece_bins(z, nb); bb = b[idx]; zb = z[idx]; hb = Hf[idx]; tot = np.zeros(idx.shape[0])
    for j in range(nbin):
        m = bb == j; tot += np.abs((hb * m).sum(1) - (zb * m).sum(1))
    return tot / idx.shape[1]

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4: return NAN, int(m.sum())
    return float(stats.spearmanr(x[m], y[m]).statistic), int(m.sum())

def half_splits(labels, n_splits, rng):
    labels = np.asarray(labels); out = []
    for _ in range(n_splits):
        A = np.zeros(len(labels), bool)
        for lab in np.unique(labels):
            pos = rng.permutation(np.flatnonzero(labels == lab)); A[pos[:len(pos) // 2]] = True
        out.append(A)
    return out

def crossfit_binary(H, Z, splits, ab, s_prior, mono=False):
    """Blackwell-optimal use of a BINARY signal with cross-fitted posteriors, anchored to the probing prior:
    on calibration half A, lift_z = P(H|Z=z, A) - P(H|A); posterior on the other half s_Z = s_prior + lift_Z (so E[s_Z] = s_prior,
    eq:imperfect-voi's premise); attempt iff a + (1-s_Z) b < b  <=>  s_Z > a/b.  mono=True pools an anti-monotone pair (isotonic).
    Returns the per-task attempt frequency over splits and the number of anti-monotone calibrations."""
    attfrac = np.zeros(len(H)); n_anti = 0
    for A in splits:
        for cal, tgt in ((A, ~A), (~A, A)):
            c1 = cal & Z; c0 = cal & ~Z; base = H[cal].mean() if cal.any() else s_prior
            l1 = (H[c1].mean() - base) if c1.any() else 0.0
            l0 = (H[c0].mean() - base) if c0.any() else 0.0
            if l1 < l0:
                n_anti += 1
                if mono: l1 = l0 = 0.0
            post = np.clip(s_prior + np.where(Z[tgt], l1, l0), 0.0, 1.0)
            attfrac[tgt] += post > ab
    return attfrac / len(splits), n_anti

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating, float)): return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    return o

results = {"experiment_id": "X12a_voi_released_signals", "seed": SEED, "nboot": NBOOT, "n_half_splits": NSPLIT, "n_permutations": NPERM,
           "cost_basis": "dollars (CodeRouterBench csv cost_usd; RouterBench <model>|total_cost); router cost k in dollars per task",
           "pending": "X12-b (live-judge signals) is a later experiment; not run here"}

# ================================================================================================ A. CodeRouterBench
_CACHE = os.path.join(HERE, "_rows_cache.pkl"); DEV = os.environ.get("X12A_RESUME") == "1"; RESUME = DEV and os.path.exists(_CACHE)   # development shortcut only; the deliverable run sets no env var
print("[CRB] loading", flush=True)
df = pd.read_csv(f"{ACR}/coderouterbench/id_results_long.csv")
S = df.pivot(index="task_id", columns="model", values="score"); Cc = df.pivot(index="task_id", columns="model", values="cost_usd")
TOK = df.pivot(index="task_id", columns="model", values="input_tokens").mean(axis=1)   # prompt-token proxy: mean over the 8 models' tokenizers
meta = df.drop_duplicates("task_id").set_index("task_id")[["split", "dimension"]].loc[S.index]
meta["bench"] = meta.index.str.split("_").str[1]; meta["cell21"] = meta.dimension + "|" + meta.bench; meta["all"] = "ALL"
assert S.shape == (9999, 8) and not S.isna().any().any() and not Cc.isna().any().any()
PROB = meta.index[meta.split == "probing"]; TEST = meta.index[meta.split == "id_test"]
assert len(PROB) == 7080 and len(TEST) == 2919
eta_B_prob = float(S.loc[PROB, STRONG].mean() / Cc.loc[PROB, STRONG].mean())

# --- signal files: document exactly which fields exist
sigdoc = {}; DEC = {}
for r in SIGNALS_CRB:
    path = f"{ACR}/id/voter_decisions/{r}.jsonl" if r != "acrouter_release" else ACR_RELEASE
    rows = [json.loads(l) for l in open(path) if l.strip()]
    keys = collections.Counter(k for d in rows for k in d)
    types = {k: sorted({type(d[k]).__name__ for d in rows if k in d}) for k in keys}
    d = pd.DataFrame(rows); in_test = d.task_id.isin(TEST); in_prob = d.task_id.isin(PROB); in_mat = d.task_id.isin(S.index)
    sigdoc[r] = dict(path=path, n_rows=len(d), fields={k: types[k] for k in keys},
                     numeric_or_score_fields=[k for k in keys if any(t in ("float", "int") for t in types[k])],
                     n_in_id_test=int(in_test.sum()), n_in_probing=int(in_prob.sum()), n_not_in_outcome_matrix=int((~in_mat).sum()),
                     unmatched_task_id_prefixes=sorted({t.split("_")[0] for t in d.task_id[~in_mat]}),
                     n_duplicate_task_ids=int(d.task_id.duplicated().sum()),
                     chosen_model_counts_id_test=d[in_test].chosen_model.value_counts().to_dict())
    dd = d[in_test].drop_duplicates("task_id").set_index("task_id").chosen_model
    assert len(dd) == 2919, (r, len(dd)); DEC[r] = dd.reindex(TEST)
results["signal_fields_crb"] = sigdoc
results["signal_definition_crb"] = ("Every released decision file carries only {task_id, chosen_model}: no score, probability or logit. "
                                    "Z is therefore binary: any_non_opus = 1{chosen_model != claude-opus-4-6} (protocol definition), "
                                    "this_primary = 1{chosen_model == primary m} (X3 definition, used for C5).")

PARTS = {"dim9": "dimension", "cell21": "cell21", "all": "all"}
STATES = [(p, c) for p, col in PARTS.items() for c in sorted(meta[col].unique())]
prob_ids = {(p, c): PROB[meta.loc[PROB, PARTS[p]] == c] for p, c in STATES}
test_ids = {(p, c): TEST[meta.loc[TEST, PARTS[p]] == c] for p, c in STATES}
IP = {k: rng.integers(0, len(v), size=(NBOOT, len(v)), dtype=np.int32) for k, v in prob_ids.items()}
IT = {k: rng.integers(0, len(v), size=(NBOOT, len(v)), dtype=np.int32) for k, v in test_ids.items()}
SPL = {k: half_splits(meta.loc[v, "cell21"].to_numpy(), NSPLIT, rng) for k, v in test_ids.items()}
PERM = {k: [rng.permutation(len(v)) for _ in range(NPERM)] for k, v in test_ids.items()}

crb_rows, band_rows, ivp_rows, kcond_rows = [], [], [], []
BV = {}   # B-vectors (float32) of the main-configuration rows, for the task-bootstrap of the Spearman test
t_crb = time.time()
for (part, cell) in (STATES if not RESUME else []):
    pid, tid = prob_ids[(part, cell)], test_ids[(part, cell)]; ip, it = IP[(part, cell)], IT[(part, cell)]
    nP, nT = len(pid), len(tid)
    cB_p = Cc.loc[pid, STRONG].to_numpy(); cB_t = Cc.loc[tid, STRONG].to_numpy(); yB_t = S.loc[tid, STRONG].to_numpy()
    b_p = float(cB_p.mean()); bB = bm(cB_p, ip); dirB = bm(cB_t, it); Cdir = cB_t
    ktok_task = (TOK.loc[tid] * ROUTER_TOKEN_PRICE).to_numpy(); ktok = float(ktok_task.mean())
    for theta in THETAS:
        Hp_all = {m: S.loc[pid, m].to_numpy() >= theta for m in CRB_PRIMARIES}
        Ht_all = {m: S.loc[tid, m].to_numpy() >= theta for m in CRB_PRIMARIES}
        for m in CRB_PRIMARIES:
            Hp = Hp_all[m]; cP_p = Cc.loc[pid, m].to_numpy(); s_p = float(Hp.mean()); a_p = float(cP_p.mean()); ab = a_p / b_p
            sB = bm(Hp.astype(float), ip); aB = bm(cP_p, ip)
            bound_c = min(s_p * (b_p - a_p), a_p * (1 - s_p)); boundB = np.minimum(sB * (bB - aB), aB * (1 - sB))
            Ht = Ht_all[m]; Hf = Ht.astype(float); cP_t = Cc.loc[tid, m].to_numpy(); yP_t = S.loc[tid, m].to_numpy()
            casc = cP_t + (1 - Hf) * cB_t; perf = Hf * cP_t + (1 - Hf) * cB_t; dlt = casc - Cdir; Vcasc = Hf * yP_t + (1 - Hf) * yB_t
            u0_att = bool(s_p > ab)                     # uninformed choice from probing: cascade iff a + (1-s) b < b
            Cu0 = casc if u0_att else Cdir; Vu0 = Vcasc if u0_att else yB_t; u0B = bm(Cu0, it); cascB = bm(casc, it); perfB = bm(perf, it)
            f_ins = min(casc.mean(), Cdir.mean()); bound_real = f_ins - perf.mean()
            s_t = float(Ht.mean()); a_t = float(cP_t.mean()); b_t = float(cB_t.mean()); Hsum = bm(Hf, it) * nT
            kcond_rows.append(dict(dataset="CRB", part=part, state=cell, theta=theta, primary=m, n_prob=nP, a=a_p, b=b_p, s=s_p, a_over_b=ab,
                                   k_lt_a_bound=a_p, k_lt_a_1ms=a_p * (1 - s_p), k_lt_band_nonempty=a_p * (b_p - a_p) / b_p,
                                   k_tok=ktok, k_001b=K_REL[0] * b_p, k_01b=K_REL[1] * b_p))
            # --- band position of s and the realized cheapest option with EXACT information at cost k (prop:voi, OOS)
            for kname, kval in [("0", 0.0), ("0.001b", K_REL[0] * b_p), ("0.01b", K_REL[1] * b_p), ("tok", ktok)]:
                lo_b, hi_b = kval / (b_p - a_p), 1 - kval / a_p
                th = {"cascade": a_p + (1 - s_p) * b_p, "direct": b_p, "info": kval + s_p * a_p + (1 - s_p) * b_p}
                real = {"cascade": float(casc.mean()), "direct": float(Cdir.mean()), "info": float(perf.mean() + kval)}
                mB = (perfB + kval) - np.minimum(cascB, dirB); mlo, mhi = ci(mB)
                band_rows.append(dict(dataset="CRB", part=part, state=cell, theta=theta, primary=m, k_name=kname, k=kval, n_prob=nP, n_test=nT,
                                      s=s_p, a=a_p, b=b_p, a_over_b=ab, band_lo=lo_b, band_hi=hi_b, band_nonempty=bool(lo_b < hi_b),
                                      u=(s_p - lo_b) / (hi_b - lo_b) if hi_b > lo_b else NAN, pred_info_in_band=bool(lo_b < s_p < hi_b),
                                      pred_argmin=min(th, key=th.get), C_cascade_real=real["cascade"], C_direct_real=real["direct"],
                                      C_info_real=real["info"], real_argmin=min(real, key=real.get),
                                      info_minus_best_uninformed=real["info"] - min(real["cascade"], real["direct"]), lo=mlo, hi=mhi, decided=decided(mlo, mhi)))
            # --- cor:info-vs-proposer: exact information at cost k' vs upgrading the primary to a costlier model m2 (k' = a2 - a)
            for m2 in CRB_PRIMARIES:
                if m2 == m: continue
                cP2_p = Cc.loc[pid, m2].to_numpy(); a2_p = float(cP2_p.mean())
                if a2_p <= a_p: continue
                s2_p = float(Hp_all[m2].mean()); dp = s2_p - s_p; kp = a2_p - a_p
                pred = b_p * dp - a_p * (1 - s_p); predB = bB * (bm(Hp_all[m2].astype(float), ip) - sB) - aB * (1 - sB)
                H2 = Ht_all[m2].astype(float); casc2 = Cc.loc[tid, m2].to_numpy() + (1 - H2) * cB_t
                d_eq = (perf + kp) - casc2; dB = bm(d_eq, it); lo, hi = ci(dB)
                d0 = perf - casc2; d0B = bm(d0, it); lo0, hi0 = ci(d0B)
                ivp_rows.append(dict(dataset="CRB", part=part, state=cell, theta=theta, primary=m, upgrade=m2, n_prob=nP, n_test=nT, s=s_p, a=a_p, b=b_p,
                                     s_upgrade=s2_p, delta_p=dp, k_prime=kp, pred_info_minus_dec=pred, pred_lo=ci(predB)[0], pred_hi=ci(predB)[1],
                                     pred_info_better_eq=bool(pred < 0), real_info_minus_dec_eq=float(d_eq.mean()), real_lo=lo, real_hi=hi,
                                     real_info_better_eq=bool(d_eq.mean() < 0), decided_eq=decided(lo, hi),
                                     pred_info_better_k0=bool(pred - kp < 0), real_info_minus_dec_k0=float(d0.mean()), real_lo_k0=lo0, real_hi_k0=hi0,
                                     real_info_better_k0=bool(d0.mean() < 0), decided_k0=decided(lo0, hi0)))
            # --- the released signals
            for r in SIGNALS_CRB:
                ch = DEC[r].loc[tid].to_numpy()
                for zdef in ZDEFS:
                    Z = (ch != STRONG) if zdef == "any_non_opus" else (ch == m)
                    Zf = Z.astype(float); ZHf = (Z & Ht).astype(float)
                    z = float(Z.mean()); s1 = float(Ht[Z].mean()) if Z.any() else NAN; s0 = float(Ht[~Z].mean()) if (~Z).any() else NAN
                    au = auroc_binary(Z, Ht); ec = ece_binary(Z, Ht)
                    zs = bm(Zf, it) * nT; zh = bm(ZHf, it) * nT
                    auB = (1 + fdiv(zh, Hsum) - fdiv(zs - zh, nT - Hsum)) / 2; auB[(Hsum == 0) | (Hsum == nT)] = NAN
                    ecB = (zs + Hsum - 2 * zh) / nT
                    # native usage: attempt iff Z = 1
                    Cnat = Cdir + Zf * dlt; natB = dirB + bm(Zf * dlt, it); g_nat = float(Cu0.mean() - Cnat.mean()); g_natB = u0B - natB
                    # Blackwell-optimal usage (cross-fitted, anchored posterior; a/b from probing)
                    af, n_anti = crossfit_binary(Ht, Z, SPL[(part, cell)], ab, s_p)
                    Cbw = Cdir + af * dlt; bwB = dirB + bm(af * dlt, it); g_bw = float(Cu0.mean() - Cbw.mean()); g_bwB = u0B - bwB
                    af_m, _ = crossfit_binary(Ht, Z, SPL[(part, cell)], ab, s_p, mono=True); g_bwm = float(Cu0.mean() - (Cdir + af_m * dlt).mean())
                    # in-sample Blackwell usage (posterior from the whole id_test state, prior/threshold from id_test) vs realized min
                    p1 = s1 if np.isfinite(s1) else s_t; p0 = s0 if np.isfinite(s0) else s_t
                    Cbw_ins = np.where(np.where(Z, p1, p0) > a_t / b_t, casc, Cdir); g_bwins = float(f_ins - Cbw_ins.mean())
                    g_natins = float(f_ins - Cnat.mean())               # X3's saving definition (IN-SAMPLE)
                    # permuted Z (SEMI-SYNTHETIC): calibrated but uninformative
                    if theta == THETA_MAIN:
                        afp = np.zeros(nT)
                        for pm in PERM[(part, cell)]:
                            a_, _ = crossfit_binary(Ht, Z[pm], SPL[(part, cell)], ab, s_p); afp += a_
                        afp /= NPERM; g_perm = float(Cu0.mean() - (Cdir + afp * dlt).mean()); g_permB = u0B - (dirB + bm(afp * dlt, it))
                        aup = float(np.mean([auroc_binary(Z[pm], Ht) for pm in PERM[(part, cell)]]))
                    else:
                        g_perm = NAN; g_permB = np.full(NBOOT, NAN); aup = NAN
                    Vnat = np.where(Z, Vcasc, yB_t); Vbw = yB_t + af * (Vcasc - yB_t)
                    bc = bound_c if bound_c > 1e-9 else NAN
                    row = dict(dataset="CRB", part=part, state=cell, theta=theta, primary=m, signal=r, zdef=zdef,
                               signal_family=("finetuned" if r in FINETUNED else "classical" if r in CLASSICAL else "lookup"),
                               n_prob=nP, n_test=nT, s=s_p, a=a_p, b=b_p, a_over_b=ab, s_test=s_t, a_test=a_t, b_test=b_t, k_tok=ktok,
                               P_Z1=z, s_given_Z1=s1, s_given_Z0=s0, auroc=au, auroc_lo=ci(auB)[0], auroc_hi=ci(auB)[1], ece=ec, ece_lo=ci(ecB)[0], ece_hi=ci(ecB)[1],
                               auroc_permuted=aup, u0_attempt=u0_att, C_u0=float(Cu0.mean()), C_cascade=float(casc.mean()), C_direct=float(Cdir.mean()),
                               C_perfect=float(perf.mean()), f_ins=f_ins, bound_closed=bound_c, bound_closed_lo=ci(boundB)[0], bound_closed_hi=ci(boundB)[1],
                               bound_real=bound_real, C_native=float(Cnat.mean()), C_bw=float(Cbw.mean()), frac_attempt_native=z, frac_attempt_bw=float(af.mean()),
                               n_anti_monotone_calibrations=n_anti,
                               gross_native=g_nat, gross_native_lo=ci(g_natB)[0], gross_native_hi=ci(g_natB)[1],
                               gross_bw=g_bw, gross_bw_lo=ci(g_bwB)[0], gross_bw_hi=ci(g_bwB)[1], gross_bw_mono=g_bwm, gross_bw_ins=g_bwins, gross_native_ins=g_natins,
                               gross_perm=g_perm, gross_perm_lo=ci(g_permB)[0], gross_perm_hi=ci(g_permB)[1],
                               capture_native=g_nat / bc, capture_native_lo=ci(g_natB)[0] / bc, capture_native_hi=ci(g_natB)[1] / bc,
                               capture_bw=g_bw / bc, capture_bw_lo=ci(g_bwB)[0] / bc, capture_bw_hi=ci(g_bwB)[1] / bc,
                               capture_bw_ins=g_bwins / (bound_real if bound_real > 1e-9 else NAN), capture_native_ins=g_natins / (bound_real if bound_real > 1e-9 else NAN),
                               capture_perm=g_perm / bc, capture_perm_lo=ci(g_permB)[0] / bc, capture_perm_hi=ci(g_permB)[1] / bc,
                               net_bw_k001b=g_bw - K_REL[0] * b_p, net_bw_k01b=g_bw - K_REL[1] * b_p, net_bw_ktok=g_bw - ktok,
                               net_native_k001b=g_nat - K_REL[0] * b_p, net_native_k01b=g_nat - K_REL[1] * b_p, net_native_ktok=g_nat - ktok,
                               viol_closed_bw_pt=bool(bound_c > 1e-9 and g_bw > bound_c + 1e-12), viol_closed_bw_ci=bool(bound_c > 1e-9 and ci(g_bwB - boundB)[0] > 0),
                               viol_closed_nat_pt=bool(bound_c > 1e-9 and g_nat > bound_c + 1e-12), viol_closed_nat_ci=bool(bound_c > 1e-9 and ci(g_natB - boundB)[0] > 0),
                               viol_real_bw_pt=bool(g_bwins > bound_real + 1e-12), viol_real_nat_pt=bool(g_natins > bound_real + 1e-12),
                               degenerate_prior=bool(bound_c <= 1e-9), ratio_bw=g_bw / bc, ratio_native=g_nat / bc,
                               V_u0=float(Vu0.mean()), V_native=float(Vnat.mean()), V_bw=float(Vbw.mean()), V_direct=float(yB_t.mean()), V_cascade=float(Vcasc.mean()),
                               e_native=float((Z & Ht & (yP_t < yB_t)).mean()), e_bw=float((af * Hf * (yP_t < yB_t)).mean()),
                               eta_u0=float(Vu0.sum() / Cu0.sum()), eta_native=float(Vnat.sum() / Cnat.sum()), eta_bw=float(Vbw.sum() / Cbw.sum()), eta_B=float(yB_t.sum() / Cdir.sum()))
                    if theta == THETA_MAIN and part != "all" and bound_c > 1e-9:
                        BV[len(crb_rows)] = dict(cap_bw=(g_bwB / bound_c).astype(np.float32), cap_nat=(g_natB / bound_c).astype(np.float32),
                                                 auroc=auB.astype(np.float32), ece=ecB.astype(np.float32))
                    crb_rows.append(row)
    print(f"[CRB] {part}/{cell} done ({time.time() - t_crb:.0f}s)", flush=True)
crb = pd.DataFrame(crb_rows)
print(f"[CRB] {len(crb)} rows in {time.time() - t_crb:.0f}s", flush=True)

# ================================================================================================ B. RouterBench
print("[RB] loading", flush=True); t_rb = time.time()
rb = pd.read_pickle(RB_PKL)
def parse_prompt(s):
    try:
        v = ast.literal_eval(s)
        return "\n".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)
    except Exception:
        return str(s)
txt = rb["prompt"].map(parse_prompt)
def family(e):
    if e.startswith("mmlu"): return "mmlu"
    if e.startswith("mtbench"): return "mtbench"
    if e in ("hellaswag", "grade-school-math", "arc-challenge", "winogrande", "mbpp"): return e
    return "rag_chinese_other"
rb["family"] = rb.eval_name.map(family)
RB_MODELS = [c for c in rb.columns if (c + "|total_cost") in rb.columns]
N = len(rb)
fit_idx, held_idx = train_test_split(np.arange(N), test_size=0.3, random_state=SEED, stratify=rb.eval_name.values)
is_fit = np.zeros(N, bool); is_fit[fit_idx] = True
SUFF = {mm: rb[mm].to_numpy() >= RB_THETA for mm in RB_MODELS}; COST = {mm: rb[mm + "|total_cost"].to_numpy() for mm in RB_MODELS}
SCORE = {mm: rb[mm].to_numpy() for mm in RB_MODELS}
rb_ktok_task = (txt.str.len().to_numpy() / 4.0) * ROUTER_TOKEN_PRICE       # RouterBench has no token counts: chars/4 proxy
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True, dtype=np.float32)
Xf = vec.fit_transform(txt.iloc[fit_idx]); Xh = vec.transform(txt.iloc[held_idx])
vec_u = TfidfVectorizer(ngram_range=(1, 1), min_df=5, max_features=20000, sublinear_tf=True, dtype=np.float32)
Xf_u = vec_u.fit_transform(txt.iloc[fit_idx]); Xh_u = vec_u.transform(txt.iloc[held_idx])
ZALL = {v: {} for v in RB_ZVARIANTS}; rb_router_doc = {}
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
for m in RB_PRIMARIES:
    y = SUFF[m].astype(int); yf = y[fit_idx]
    for v, (XF, XH, C) in {"lr_tfidf": (Xf, Xh, 1.0), "lr_tfidf_uni": (Xf_u, Xh_u, 0.3)}.items():
        oof = np.zeros(len(fit_idx))
        for tr, va in skf.split(XF, yf):
            oof[va] = LogisticRegression(C=C, solver="liblinear", max_iter=2000).fit(XF[tr], yf[tr]).predict_proba(XF[va])[:, 1]
        full = LogisticRegression(C=C, solver="liblinear", max_iter=2000).fit(XF, yf)
        zz = np.zeros(N); zz[fit_idx] = oof; zz[held_idx] = full.predict_proba(XH)[:, 1]; ZALL[v][m] = zz
        rb_router_doc[f"{v}|{m}"] = dict(n_features=int(XF.shape[1]), C=C, ngram=("1-2" if v == "lr_tfidf" else "1"), fit_prevalence=float(yf.mean()),
                                         auroc_oof_fit=auroc_cont(oof, yf.astype(bool)), auroc_held=auroc_cont(zz[held_idx], y[held_idx].astype(bool)),
                                         ece_held=ece_cont(zz[held_idx], y[held_idx].astype(bool)), frac_held_below_0p1=float((zz[held_idx] < 0.1).mean()))
    ZALL["lr_tfidf_shrunk"][m] = ZALL["lr_tfidf"][m] / 10.0
results["rb_routers"] = rb_router_doc
results["rb_split"] = dict(n=N, n_fit=len(fit_idx), n_held=len(held_idx), stratified_by="eval_name", seed=SEED, theta=RB_THETA,
                           families=rb.family.value_counts().to_dict(), n_eval_ge100=int((rb.eval_name.value_counts() >= 100).sum()))
print(f"[RB] routers trained ({time.time() - t_rb:.0f}s): " + "; ".join(f"{k}: AUROC {v['auroc_held']:.3f}" for k, v in rb_router_doc.items()), flush=True)

vc = rb.eval_name.value_counts(); EVAL_GE100 = sorted(vc[vc >= 100].index)
RB_STATES = [("fam8", f) for f in sorted(rb.family.unique())] + [("eval", e) for e in EVAL_GE100] + [("all", "ALL")]
def rb_members(part, st):
    if part == "fam8": return np.flatnonzero(rb.family.values == st)
    if part == "eval": return np.flatnonzero(rb.eval_name.values == st)
    return np.arange(N)
rb_rows = []; BV_RB = {}
for (part, st) in (RB_STATES if not RESUME else []):
    mem = rb_members(part, st); fm = mem[is_fit[mem]]; hm = mem[~is_fit[mem]]; nF, nH = len(fm), len(hm)
    ipf = rng.integers(0, nF, (NBOOT, nF), dtype=np.int32); ith = rng.integers(0, nH, (NBOOT, nH), dtype=np.int32)
    perms_f = [rng.permutation(nF) for _ in range(NPERM)]; perms_h = [rng.permutation(nH) for _ in range(NPERM)]
    cB_f = COST[RB_REF][fm]; cB_h = COST[RB_REF][hm]; yB_h = SCORE[RB_REF][hm]; b_p = float(cB_f.mean()); bB = bm(cB_f, ipf); dirB = bm(cB_h, ith); Cdir = cB_h
    ktok = float(rb_ktok_task[hm].mean())
    for m in RB_PRIMARIES:
        Hp = SUFF[m][fm]; cP_f = COST[m][fm]; s_p = float(Hp.mean()); a_p = float(cP_f.mean()); ab = a_p / b_p
        sB = bm(Hp.astype(float), ipf); aB = bm(cP_f, ipf); bound_c = min(s_p * (b_p - a_p), a_p * (1 - s_p)); boundB = np.minimum(sB * (bB - aB), aB * (1 - sB))
        Ht = SUFF[m][hm]; Hf = Ht.astype(float); cP_h = COST[m][hm]; yP_h = SCORE[m][hm]
        casc = cP_h + (1 - Hf) * cB_h; perf = Hf * cP_h + (1 - Hf) * cB_h; dlt = casc - Cdir; Vcasc = Hf * yP_h + (1 - Hf) * yB_h
        u0_att = bool(s_p > ab); Cu0 = casc if u0_att else Cdir; Vu0 = Vcasc if u0_att else yB_h; u0B = bm(Cu0, ith); cascB = bm(casc, ith); perfB = bm(perf, ith)
        f_ins = min(casc.mean(), Cdir.mean()); bound_real = f_ins - perf.mean(); s_t = float(Ht.mean()); a_t = float(cP_h.mean()); b_t = float(cB_h.mean())
        kcond_rows.append(dict(dataset="RB", part=part, state=st, theta=RB_THETA, primary=m, n_prob=nF, a=a_p, b=b_p, s=s_p, a_over_b=ab,
                               k_lt_a_bound=a_p, k_lt_a_1ms=a_p * (1 - s_p), k_lt_band_nonempty=a_p * (b_p - a_p) / b_p, k_tok=ktok, k_001b=K_REL[0] * b_p, k_01b=K_REL[1] * b_p))
        for kname, kval in [("0", 0.0), ("0.001b", K_REL[0] * b_p), ("0.01b", K_REL[1] * b_p), ("tok", ktok)]:
            lo_b, hi_b = kval / (b_p - a_p), 1 - kval / a_p
            th = {"cascade": a_p + (1 - s_p) * b_p, "direct": b_p, "info": kval + s_p * a_p + (1 - s_p) * b_p}
            real = {"cascade": float(casc.mean()), "direct": float(Cdir.mean()), "info": float(perf.mean() + kval)}
            mB = (perfB + kval) - np.minimum(cascB, dirB); mlo, mhi = ci(mB)
            band_rows.append(dict(dataset="RB", part=part, state=st, theta=RB_THETA, primary=m, k_name=kname, k=kval, n_prob=nF, n_test=nH, s=s_p, a=a_p, b=b_p, a_over_b=ab,
                                  band_lo=lo_b, band_hi=hi_b, band_nonempty=bool(lo_b < hi_b), u=(s_p - lo_b) / (hi_b - lo_b) if hi_b > lo_b else NAN,
                                  pred_info_in_band=bool(lo_b < s_p < hi_b), pred_argmin=min(th, key=th.get), C_cascade_real=real["cascade"], C_direct_real=real["direct"],
                                  C_info_real=real["info"], real_argmin=min(real, key=real.get), info_minus_best_uninformed=real["info"] - min(real["cascade"], real["direct"]),
                                  lo=mlo, hi=mhi, decided=decided(mlo, mhi)))
        for m2 in RB_MODELS:
            if m2 in (m, RB_REF): continue
            a2_p = float(COST[m2][fm].mean())
            if a2_p <= a_p: continue
            H2p = SUFF[m2][fm]; s2_p = float(H2p.mean()); dp = s2_p - s_p; kp = a2_p - a_p
            pred = b_p * dp - a_p * (1 - s_p); predB = bB * (bm(H2p.astype(float), ipf) - sB) - aB * (1 - sB)
            H2 = SUFF[m2][hm].astype(float); casc2 = COST[m2][hm] + (1 - H2) * cB_h
            d_eq = (perf + kp) - casc2; dB = bm(d_eq, ith); lo, hi = ci(dB); d0 = perf - casc2; d0B = bm(d0, ith); lo0, hi0 = ci(d0B)
            ivp_rows.append(dict(dataset="RB", part=part, state=st, theta=RB_THETA, primary=m, upgrade=m2, n_prob=nF, n_test=nH, s=s_p, a=a_p, b=b_p, s_upgrade=s2_p,
                                 delta_p=dp, k_prime=kp, pred_info_minus_dec=pred, pred_lo=ci(predB)[0], pred_hi=ci(predB)[1], pred_info_better_eq=bool(pred < 0),
                                 real_info_minus_dec_eq=float(d_eq.mean()), real_lo=lo, real_hi=hi, real_info_better_eq=bool(d_eq.mean() < 0), decided_eq=decided(lo, hi),
                                 pred_info_better_k0=bool(pred - kp < 0), real_info_minus_dec_k0=float(d0.mean()), real_lo_k0=lo0, real_hi_k0=hi0,
                                 real_info_better_k0=bool(d0.mean() < 0), decided_k0=decided(lo0, hi0)))
        Hsum = bm(Hf, ith) * nH
        for v in RB_ZVARIANTS:
            z_f = ZALL[v][m][fm]; z_h = ZALL[v][m][hm]
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(z_f, Hp.astype(float))
            post = iso.predict(z_h); att_bw = post > ab; afb = att_bw.astype(float)
            Cbw = Cdir + afb * dlt; bwB = dirB + bm(afb * dlt, ith); g_bw = float(Cu0.mean() - Cbw.mean()); g_bwB = u0B - bwB
            Znat = (z_h >= 0.5).astype(float); Cnat = Cdir + Znat * dlt; natB = dirB + bm(Znat * dlt, ith); g_nat = float(Cu0.mean() - Cnat.mean()); g_natB = u0B - natB
            Zab = (z_h >= ab).astype(float); Cnab = Cdir + Zab * dlt; nabB = dirB + bm(Zab * dlt, ith); g_nab = float(Cu0.mean() - Cnab.mean()); g_nabB = u0B - nabB
            au = auroc_cont(z_h, Ht); ec = ece_cont(z_h, Ht); ec_post = ece_cont(post, Ht)
            auB = auroc_cont_boot(z_h, Hf, ith); ecB = ece_cont_boot(z_h, Hf, ith)
            afp = np.zeros(nH); aup = []
            for pf, ph in zip(perms_f, perms_h):
                iso_p = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(z_f[pf], Hp.astype(float))
                afp += iso_p.predict(z_h[ph]) > ab; aup.append(auroc_cont(z_h[ph], Ht))
            afp /= NPERM; g_perm = float(Cu0.mean() - (Cdir + afp * dlt).mean()); g_permB = u0B - (dirB + bm(afp * dlt, ith))
            g_bwins = float(f_ins - Cbw.mean()); g_natins = float(f_ins - Cnat.mean())
            Vnat = yB_h + Znat * (Vcasc - yB_h); Vbw = yB_h + afb * (Vcasc - yB_h); bc = bound_c if bound_c > 1e-9 else NAN
            row = dict(dataset="RB", part=part, state=st, theta=RB_THETA, primary=m, signal=v, zdef="prob", signal_family=("rb_lr" if v in RB_REAL_VARIANTS else "rb_lr_semi"),
                       n_prob=nF, n_test=nH, s=s_p, a=a_p, b=b_p, a_over_b=ab, s_test=s_t, a_test=a_t, b_test=b_t, k_tok=ktok,
                       P_Z1=float(Znat.mean()), s_given_Z1=float(Ht[Znat > 0].mean()) if (Znat > 0).any() else NAN, s_given_Z0=float(Ht[Znat == 0].mean()) if (Znat == 0).any() else NAN,
                       auroc=au, auroc_lo=ci(auB)[0], auroc_hi=ci(auB)[1], ece=ec, ece_lo=ci(ecB)[0], ece_hi=ci(ecB)[1], ece_posterior_held=ec_post, auroc_permuted=(float(np.nanmean(aup)) if np.isfinite(aup).any() else NAN),
                       u0_attempt=u0_att, C_u0=float(Cu0.mean()), C_cascade=float(casc.mean()), C_direct=float(Cdir.mean()), C_perfect=float(perf.mean()), f_ins=f_ins,
                       bound_closed=bound_c, bound_closed_lo=ci(boundB)[0], bound_closed_hi=ci(boundB)[1], bound_real=bound_real,
                       C_native=float(Cnat.mean()), C_native_ab=float(Cnab.mean()), C_bw=float(Cbw.mean()), frac_attempt_native=float(Znat.mean()), frac_attempt_native_ab=float(Zab.mean()),
                       frac_attempt_bw=float(afb.mean()), n_anti_monotone_calibrations=0, mean_posterior_held=float(post.mean()),
                       gross_native=g_nat, gross_native_lo=ci(g_natB)[0], gross_native_hi=ci(g_natB)[1],
                       gross_native_ab=g_nab, gross_native_ab_lo=ci(g_nabB)[0], gross_native_ab_hi=ci(g_nabB)[1],
                       gross_bw=g_bw, gross_bw_lo=ci(g_bwB)[0], gross_bw_hi=ci(g_bwB)[1], gross_bw_mono=g_bw, gross_bw_ins=g_bwins, gross_native_ins=g_natins,
                       gross_perm=g_perm, gross_perm_lo=ci(g_permB)[0], gross_perm_hi=ci(g_permB)[1],
                       capture_native=g_nat / bc, capture_native_lo=ci(g_natB)[0] / bc, capture_native_hi=ci(g_natB)[1] / bc,
                       capture_native_ab=g_nab / bc, capture_native_ab_lo=ci(g_nabB)[0] / bc, capture_native_ab_hi=ci(g_nabB)[1] / bc,
                       capture_bw=g_bw / bc, capture_bw_lo=ci(g_bwB)[0] / bc, capture_bw_hi=ci(g_bwB)[1] / bc,
                       capture_bw_ins=g_bwins / (bound_real if bound_real > 1e-9 else NAN), capture_native_ins=g_natins / (bound_real if bound_real > 1e-9 else NAN),
                       capture_perm=g_perm / bc, capture_perm_lo=ci(g_permB)[0] / bc, capture_perm_hi=ci(g_permB)[1] / bc,
                       net_bw_k001b=g_bw - K_REL[0] * b_p, net_bw_k01b=g_bw - K_REL[1] * b_p, net_bw_ktok=g_bw - ktok,
                       net_native_k001b=g_nat - K_REL[0] * b_p, net_native_k01b=g_nat - K_REL[1] * b_p, net_native_ktok=g_nat - ktok,
                       viol_closed_bw_pt=bool(bound_c > 1e-9 and g_bw > bound_c + 1e-12), viol_closed_bw_ci=bool(bound_c > 1e-9 and ci(g_bwB - boundB)[0] > 0),
                       viol_closed_nat_pt=bool(bound_c > 1e-9 and g_nat > bound_c + 1e-12), viol_closed_nat_ci=bool(bound_c > 1e-9 and ci(g_natB - boundB)[0] > 0),
                       viol_closed_nab_pt=bool(bound_c > 1e-9 and g_nab > bound_c + 1e-12), viol_closed_nab_ci=bool(bound_c > 1e-9 and ci(g_nabB - boundB)[0] > 0),
                       viol_real_bw_pt=bool(g_bwins > bound_real + 1e-12), viol_real_nat_pt=bool(g_natins > bound_real + 1e-12),
                       degenerate_prior=bool(bound_c <= 1e-9), ratio_bw=g_bw / bc, ratio_native=g_nat / bc,
                       V_u0=float(Vu0.mean()), V_native=float(Vnat.mean()), V_bw=float(Vbw.mean()), V_direct=float(yB_h.mean()), V_cascade=float(Vcasc.mean()),
                       e_native=float(((Znat > 0) & Ht & (yP_h < yB_h)).mean()), e_bw=float((afb * Hf * (yP_h < yB_h)).mean()),
                       eta_u0=float(Vu0.sum() / Cu0.sum()), eta_native=float(Vnat.sum() / Cnat.sum()), eta_bw=float(Vbw.sum() / Cbw.sum()), eta_B=float(yB_h.sum() / Cdir.sum()))
            if part != "all" and bound_c > 1e-9 and v in RB_REAL_VARIANTS:
                BV_RB[len(rb_rows)] = dict(cap_bw=(g_bwB / bound_c).astype(np.float32), cap_nat=(g_natB / bound_c).astype(np.float32),
                                           auroc=auB.astype(np.float32), ece=ecB.astype(np.float32))
            rb_rows.append(row)
    del ipf, ith
rbr = pd.DataFrame(rb_rows)
if RESUME:
    import pickle
    with open(_CACHE, "rb") as fh: _c = pickle.load(fh)
    crb, rbr, band_rows, ivp_rows, kcond_rows, BV, BV_RB = _c["crb"], _c["rbr"], _c["band_rows"], _c["ivp_rows"], _c["kcond_rows"], _c["BV"], _c["BV_RB"]
    print("[resume] rows loaded from cache (development only)", flush=True)
elif DEV:
    import pickle
    with open(_CACHE, "wb") as fh: pickle.dump(dict(crb=crb, rbr=rbr, band_rows=band_rows, ivp_rows=ivp_rows, kcond_rows=kcond_rows, BV=BV, BV_RB=BV_RB), fh)
print(f"[RB] {len(rbr)} rows in {time.time() - t_rb:.0f}s", flush=True)

# ================================================================================================ write row tables
ALL = pd.concat([crb, rbr], ignore_index=True, sort=False)
crb.to_csv(os.path.join(HERE, "crb_signal_rows.csv"), index=False); rbr.to_csv(os.path.join(HERE, "rb_signal_rows.csv"), index=False)
band = pd.DataFrame(band_rows); band.to_csv(os.path.join(HERE, "band_rows.csv"), index=False)
ivp = pd.DataFrame(ivp_rows)
# next-stronger upgrade per (dataset, state, theta, primary): the cheapest costlier model with Delta p > 0 (else the cheapest costlier model)
ivp["next_stronger"] = False
_rank = np.where(ivp.delta_p > 0, 0.0, 1e9) + ivp.k_prime.to_numpy()
pick = pd.Series(_rank, index=ivp.index).groupby([ivp.dataset, ivp.part, ivp.state, ivp.theta, ivp.primary]).idxmin()
ivp.loc[pick.to_numpy(), "next_stronger"] = True
ivp.to_csv(os.path.join(HERE, "info_vs_proposer_rows.csv"), index=False)
kcond = pd.DataFrame(kcond_rows); kcond.to_csv(os.path.join(HERE, "k_condition_rows.csv"), index=False)

# ================================================================================================ C. tests
# ---- (i) C1: realized gross VOI never exceeds the perfect-information bound
def c1_summary(sub, tag):
    ok = sub[~sub.degenerate_prior]
    out = dict(n_rows=int(len(sub)), n_degenerate_prior_excluded=int(sub.degenerate_prior.sum()), n_rows_tested=int(len(ok)),
               bw=dict(n_point_violations=int(ok.viol_closed_bw_pt.sum()), n_ci_violations=int(ok.viol_closed_bw_ci.sum()), max_ratio=float(ok.ratio_bw.max()),
                       n_ratio_above_0p9=int((ok.ratio_bw > 0.9).sum())),
               native=dict(n_point_violations=int(ok.viol_closed_nat_pt.sum()), n_ci_violations=int(ok.viol_closed_nat_ci.sum()), max_ratio=float(ok.ratio_native.max())),
               realized_bound_in_sample=dict(n_bw_violations=int(sub.viol_real_bw_pt.sum()), n_native_violations=int(sub.viol_real_nat_pt.sum()),
                                             note="saving_in_sample <= min(C_casc,C_direct) - C_perfect is arithmetic whenever cost_P <= cost_B task by task"))
    if "viol_closed_nab_pt" in sub: out["native_raw_ab"] = dict(n_point_violations=int(ok.viol_closed_nab_pt.fillna(False).sum()), n_ci_violations=int(ok.viol_closed_nab_ci.fillna(False).sum()))
    out["realized_bound_in_sample"]["bw_violations_by_primary"] = {str(k): int(v) for k, v in sub[sub.viol_real_bw_pt].groupby("primary").size().items()}
    vi = ok[ok.viol_closed_bw_pt | ok.viol_closed_nat_pt]
    out["point_violations"] = vi[["dataset", "part", "state", "theta", "primary", "signal", "zdef", "n_test", "s", "s_test", "a_over_b", "bound_closed", "gross_bw", "gross_bw_lo", "gross_bw_hi",
                                  "gross_native", "gross_native_lo", "gross_native_hi", "viol_closed_bw_ci", "viol_closed_nat_ci"]].to_dict("records")
    return out
C1 = {"all_rows": c1_summary(ALL, "all"), "crb": c1_summary(crb, "crb"), "crb_theta1": c1_summary(crb[crb.theta == 1.0], "crb1"), "crb_theta0.5": c1_summary(crb[crb.theta == 0.5], "crb05"),
      "rb": c1_summary(rbr, "rb")}
# cost heterogeneity that breaks the pointwise dominance C_Z >= C_perfect
het = {}
for m in CRB_PRIMARIES: het[f"CRB|{m}"] = float((Cc.loc[TEST, m] > Cc.loc[TEST, STRONG]).mean())
for m in RB_PRIMARIES: het[f"RB|{m}"] = float((COST[m][held_idx] > COST[RB_REF][held_idx]).mean())
C1["frac_tasks_primary_costlier_than_reference"] = het
_vi = pd.DataFrame(C1["all_rows"]["point_violations"])
C1["point_violation_breakdown"] = (_vi.groupby(["dataset", "state", "theta", "primary"]).agg(n_rows=("signal", "size"), n_test=("n_test", "first"), s=("s", "first"), s_test=("s_test", "first"),
                                   a_over_b=("a_over_b", "first"), bound_closed=("bound_closed", "first"), max_gross_native=("gross_native", "max"), n_ci_supported=("viol_closed_nat_ci", "sum"))
                                   .reset_index().to_dict("records")) if len(_vi) else []
results["C1_bound"] = C1

# ---- (ii) C2: capture ratio vs AUROC and ECE (Spearman, OOS)
def corr_block(sub, bv, cap_col, label):
    ok = sub[np.isfinite(sub[cap_col]) & np.isfinite(sub.auroc) & np.isfinite(sub.ece)]
    r_au, n = spearman(ok[cap_col], ok.auroc); r_ec, _ = spearman(ok[cap_col], ok.ece)
    # row bootstrap
    rr = np.random.default_rng(SEED + 1); diffs = []; r1s = []; r2s = []
    x = ok[cap_col].to_numpy(); y1 = ok.auroc.to_numpy(); y2 = ok.ece.to_numpy()
    absd = []
    for _ in range(NBOOT):
        i = rr.integers(0, len(ok), len(ok)); a1, _ = spearman(x[i], y1[i]); a2, _ = spearman(x[i], y2[i]); r1s.append(a1); r2s.append(a2); diffs.append(a1 - a2); absd.append(abs(a1) - abs(a2))
    out = dict(label=label, n_rows=n, spearman_auroc=r_au, spearman_ece=r_ec, diff=r_au - r_ec, abs_diff=abs(r_au) - abs(r_ec),
               row_boot=dict(auroc_ci=ci(r1s), ece_ci=ci(r2s), diff_ci=ci(diffs), abs_diff_ci=ci(absd)))
    # task bootstrap (paired resamples per state; each replicate recomputes capture, AUROC, ECE of every row)
    keys = [k for k in ok.index if k in bv]
    if len(keys) >= 4:
        capk = "cap_bw" if cap_col.startswith("capture_bw") else "cap_nat"
        Xc = np.stack([bv[k][capk] for k in keys]); Xa = np.stack([bv[k]["auroc"] for k in keys]); Xe = np.stack([bv[k]["ece"] for k in keys])
        d2 = []; a_ = []; e_ = []; ad2 = []
        for bi in range(NBOOT):
            s1, _ = spearman(Xc[:, bi], Xa[:, bi]); s2, _ = spearman(Xc[:, bi], Xe[:, bi]); a_.append(s1); e_.append(s2); d2.append(s1 - s2); ad2.append(abs(s1) - abs(s2))
        out["task_boot"] = dict(n_rows=len(keys), auroc_ci=ci(a_), ece_ci=ci(e_), diff_ci=ci(d2), abs_diff_ci=ci(ad2))
    return out
main_crb = crb[(crb.theta == THETA_MAIN) & (crb.part != "all")]
main_rb = rbr[(rbr.part != "all") & rbr.signal.isin(RB_REAL_VARIANTS)]
BVALL = {}; off = 0
main_all = pd.concat([main_crb, main_rb], ignore_index=False)
# index alignment for the pooled task bootstrap: crb rows keep their index; rb rows are offset
for k, v in BV.items(): BVALL[k] = v
main_rb_re = main_rb.copy(); main_rb_re.index = main_rb_re.index + len(crb)
for k, v in BV_RB.items(): BVALL[k + len(crb)] = v
main_pool = pd.concat([main_crb, main_rb_re])
C2 = {}
for cap_col in ["capture_bw", "capture_native"]:
    C2[cap_col] = dict(
        crb_dim9_cell21=corr_block(main_crb, BV, cap_col, "CRB theta=1, dim9+cell21, 10 signals x 2 Z-definitions x 7 primaries"),
        crb_dim9=corr_block(main_crb[main_crb.part == "dim9"], BV, cap_col, "CRB dim9"), crb_cell21=corr_block(main_crb[main_crb.part == "cell21"], BV, cap_col, "CRB cell21"),
        crb_any_non_opus=corr_block(main_crb[main_crb.zdef == "any_non_opus"], BV, cap_col, "CRB Z=any_non_opus"),
        crb_this_primary=corr_block(main_crb[main_crb.zdef == "this_primary"], BV, cap_col, "CRB Z=this_primary"),
        rb=corr_block(main_rb_re, BVALL, cap_col, "RB fam8+eval(>=100), lr_tfidf + lr_tfidf_uni x 4 primaries"),
        pooled=corr_block(main_pool, BVALL, cap_col, "CRB + RB pooled"))
# the semi-synthetic monotone distortion: same AUROC and Blackwell capture, different ECE and native (raw a/b rule) capture
sh = rbr[(rbr.signal == "lr_tfidf_shrunk") & (rbr.part != "all")].set_index(["part", "state", "primary"])
base = rbr[(rbr.signal == "lr_tfidf") & (rbr.part != "all")].set_index(["part", "state", "primary"]).loc[sh.index]
C2["shrunk_distortion_check"] = dict(n_states=int(len(sh)), max_abs_diff_auroc=float((sh.auroc - base.auroc).abs().max()), max_abs_diff_capture_bw=float((sh.capture_bw - base.capture_bw).abs().max()),
                                     mean_ece_base=float(base.ece.mean()), mean_ece_shrunk=float(sh.ece.mean()),
                                     mean_capture_native_ab_base=float(base.capture_native_ab.mean()), mean_capture_native_ab_shrunk=float(sh.capture_native_ab.mean()),
                                     n_states_shrunk_native_ab_ci_below_0=int((sh.capture_native_ab_hi < 0).sum()), frac_attempt_native_ab_shrunk=float(sh.frac_attempt_native_ab.mean()))
# decision-relevant separation: the Blackwell usage has value only if the posteriors straddle a/b (in-sample posteriors, probing threshold)
def straddle_block(sub, cap_col):
    ok = sub[np.isfinite(sub[cap_col])].copy()
    st_crb = (np.minimum(ok.s_given_Z1, ok.s_given_Z0) < ok.a_over_b) & (ok.a_over_b < np.maximum(ok.s_given_Z1, ok.s_given_Z0))
    st_rb = (ok.frac_attempt_bw > 0) & (ok.frac_attempt_bw < 1)
    ok["straddle"] = np.where(ok.dataset == "CRB", st_crb.fillna(False), st_rb)
    lo_col = cap_col + "_lo"; g1 = ok[ok.straddle]; g0 = ok[~ok.straddle]
    out = dict(n_rows=int(len(ok)), n_straddle=int(len(g1)), n_not=int(len(g0)),
               mean_capture_straddle=float(g1[cap_col].mean()) if len(g1) else NAN, mean_capture_not=float(g0[cap_col].mean()) if len(g0) else NAN,
               frac_ci_above_0_straddle=float((g1[lo_col] > 0).mean()) if len(g1) else NAN, frac_ci_above_0_not=float((g0[lo_col] > 0).mean()) if len(g0) else NAN,
               frac_exactly_0_not=float((g0[cap_col] == 0).mean()) if len(g0) else NAN, frac_exactly_0_straddle=float((g1[cap_col] == 0).mean()) if len(g1) else NAN,
               spearman_capture_vs_straddle_indicator=spearman(ok[cap_col], ok.straddle.astype(float))[0])
    c = ok[ok.dataset == "CRB"]
    if len(c):
        margin = np.minimum((c.s_given_Z1 - c.a_over_b).abs(), (c.s_given_Z0 - c.a_over_b).abs()) * np.where(c.straddle, 1.0, -1.0)
        out["crb_spearman_capture_vs_signed_margin"] = spearman(c[cap_col], margin)[0]; out["crb_spearman_capture_vs_auroc_same_rows"] = spearman(c[cap_col], c.auroc)[0]
    return out
def within_block(sub, cap_col):
    rs = []
    for _, g in sub.groupby(["dataset", "part", "state", "primary"]):
        g = g[np.isfinite(g[cap_col]) & np.isfinite(g.auroc) & np.isfinite(g.ece)]
        if len(g) >= 5 and g[cap_col].nunique() > 1 and g.auroc.nunique() > 1 and g.ece.nunique() > 1:
            rs.append((spearman(g[cap_col], g.auroc)[0], spearman(g[cap_col], g.ece)[0]))
    if not rs: return dict(n_groups=0)
    rs = np.array(rs); d = rs[:, 0] - rs[:, 1]; ad = np.abs(rs[:, 0]) - np.abs(rs[:, 1]); rr = np.random.default_rng(SEED + 2)
    boots = []; aboots = []
    for _ in range(NBOOT):
        i = rr.integers(0, len(d), len(d)); boots.append(float(np.nanmean(d[i]))); aboots.append(float(np.nanmean(ad[i])))
    return dict(n_groups=int(len(rs)), mean_spearman_auroc=float(np.nanmean(rs[:, 0])), mean_spearman_ece=float(np.nanmean(rs[:, 1])), mean_diff=float(np.nanmean(d)),
                diff_ci_group_boot=ci(boots), mean_abs_diff=float(np.nanmean(ad)), abs_diff_ci_group_boot=ci(aboots), frac_groups_auroc_gt_ece=float(np.nanmean(rs[:, 0] > rs[:, 1])),
                frac_groups_abs_auroc_gt_abs_ece=float(np.nanmean(np.abs(rs[:, 0]) > np.abs(rs[:, 1]))),
                note="groups = (state, primary) with >= 5 signal variants and non-constant capture; only CRB has enough variants (10 signals x 2 Z definitions)")
C2["zero_capture"] = dict(crb_main_rows=int(len(main_crb)), n_capture_bw_exactly_0=int((main_crb.capture_bw == 0).sum()), n_capture_bw_nan_bound_0=int(main_crb.capture_bw.isna().sum()),
                          n_capture_bw_positive=int((main_crb.capture_bw > 0).sum()), n_capture_bw_negative=int((main_crb.capture_bw < 0).sum()),
                          n_ci_above_0=int((main_crb.gross_bw_lo > 0).sum()), n_ci_below_0=int((main_crb.gross_bw_hi < 0).sum()),
                          rb_main_rows=int(len(main_rb)), rb_n_capture_bw_exactly_0=int((main_rb.capture_bw == 0).sum()), rb_n_ci_above_0=int((main_rb.gross_bw_lo > 0).sum()))
C2["straddle"] = {cap: dict(pooled=straddle_block(main_pool, cap), crb=straddle_block(main_crb, cap), rb=straddle_block(main_rb_re, cap)) for cap in ["capture_bw", "capture_native"]}
C2["within_state_primary"] = {cap: within_block(main_crb, cap) for cap in ["capture_bw", "capture_native"]}
results["C2_spearman"] = C2

# ---- (iii) C3: permuted Z (calibrated but uninformative) has net VOI ~ -k, i.e. gross ~ 0
def c3_summary(sub):
    ok = sub[np.isfinite(sub.capture_perm)]
    return dict(n_rows=int(len(ok)), mean_capture_perm=float(ok.capture_perm.mean()), median_capture_perm=float(ok.capture_perm.median()),
                median_abs_capture_perm=float(ok.capture_perm.abs().median()), q95_abs_capture_perm=float(ok.capture_perm.abs().quantile(.95)), min_capture_perm=float(ok.capture_perm.min()),
                max_capture_perm=float(ok.capture_perm.max()), n_ci_above_0=int((ok.gross_perm_lo > 0).sum()), n_ci_below_0=int((ok.gross_perm_hi < 0).sum()),
                n_ci_covers_0=int(((ok.gross_perm_lo <= 0) & (ok.gross_perm_hi >= 0)).sum()), n_exactly_0=int((ok.gross_perm == 0).sum()),
                mean_gross_perm_dollars=float(ok.gross_perm.mean()), mean_auroc_permuted=float(ok.auroc_permuted.mean()),
                mean_capture_bw_real_same_rows=float(ok.capture_bw.mean()), n_real_ci_above_0=int((ok.gross_bw_lo > 0).sum()),
                net_at_k=dict(k001b=float((ok.gross_perm - K_REL[0] * ok.b).mean() / (K_REL[0] * ok.b).mean()), k01b=float((ok.gross_perm - K_REL[1] * ok.b).mean() / (K_REL[1] * ok.b).mean()),
                              note="mean(net VOI)/mean(k): -1 means net = -k exactly"))
C3 = {"crb_theta1": c3_summary(crb[(crb.theta == 1.0) & (crb.part != "all")]), "crb_theta1_pooled_state": c3_summary(crb[(crb.theta == 1.0) & (crb.part == "all")]),
      "rb": c3_summary(rbr[(rbr.part != "all") & rbr.signal.isin(RB_REAL_VARIANTS)]), "rb_pooled_state": c3_summary(rbr[(rbr.part == "all") & rbr.signal.isin(RB_REAL_VARIANTS)])}
_c3 = pd.concat([crb[(crb.theta == 1.0) & (crb.part != "all")], rbr[(rbr.part != "all") & rbr.signal.isin(RB_REAL_VARIANTS)]])
C3["pooled"] = c3_summary(_c3)
C3["near_threshold_rows"] = dict(note="rows whose prior s lies within +-0.05 of a/b: noise in the cross-fitted lift flips decisions",
                                 n=int((np.abs(_c3.s - _c3.a_over_b) < 0.05).sum()), mean_capture_perm=float(_c3[np.abs(_c3.s - _c3.a_over_b) < 0.05].capture_perm.mean()),
                                 mean_capture_perm_far=float(_c3[np.abs(_c3.s - _c3.a_over_b) >= 0.05].capture_perm.mean()))
_ok3 = _c3[np.isfinite(_c3.capture_perm)]; _pos = _ok3[_ok3.gross_perm_lo > 0]
C3["ci_above_0_rows"] = dict(n=int(len(_pos)), n_tested=int(len(_ok3)), share=float(len(_pos) / max(len(_ok3), 1)), nominal_one_sided_rate=0.025,
                             n_near_threshold_abs_s_minus_ab_lt_0p05=int((np.abs(_pos.s - _pos.a_over_b) < 0.05).sum()),
                             n_u0_wrong_in_sample=int(((_pos.s > _pos.a_over_b) != (_pos.s_test > _pos.a_test / _pos.b_test)).sum()),
                             n_distinct_state_primary=int(_pos.groupby(["dataset", "part", "state", "primary"]).ngroups),
                             state_primary=sorted({f"{r.dataset}|{r.part}|{r.state}|{r.primary} (s={r.s:.3f}, a/b={r.a_over_b:.3f}, s_test={r.s_test:.3f}, a/b_test={r.a_test / r.b_test:.3f})" for _, r in _pos.iterrows()}),
                             mean_capture_perm=float(_pos.capture_perm.mean()), mean_capture_bw_same_rows=float(_pos.capture_bw.mean()))
C3["share_ci_above_0_by_dataset"] = {ds: float((g.gross_perm_lo > 0).mean()) for ds, g in _ok3.groupby("dataset")}
results["C3_permutation"] = C3

# ---- (iv) C4: cor:info-vs-proposer
def c4_summary(sub, eq=True):
    pc, rc, dc = ("pred_info_better_eq", "real_info_better_eq", "decided_eq") if eq else ("pred_info_better_k0", "real_info_better_k0", "decided_k0")
    dec = sub[sub[dc]]
    out = dict(n_rows=int(len(sub)), n_decided=int(len(dec)), n_pred_info=int(sub[pc].sum()), n_real_info_decided=int(dec[rc].sum()),
               agreement_all=float((sub[pc] == sub[rc]).mean()) if len(sub) else NAN,
               agreement_decided=float((dec[pc] == dec[rc]).mean()) if len(dec) else NAN,
               baseline_always_info_decided=float(dec[rc].mean()) if len(dec) else NAN, baseline_always_upgrade_decided=float((~dec[rc]).mean()) if len(dec) else NAN,
               baseline_info_iff_dp_le_0_decided=float(((dec.delta_p <= 0) == dec[rc]).mean()) if len(dec) else NAN,
               confusion_decided={"info/info": int((dec[pc] & dec[rc]).sum()), "upg/upg": int((~dec[pc] & ~dec[rc]).sum()), "info/upg": int((dec[pc] & ~dec[rc]).sum()), "upg/info": int((~dec[pc] & dec[rc]).sum())})
    dp = dec[dec.delta_p > 0]
    out["decided_delta_p_positive"] = dict(n=int(len(dp)), agreement=float((dp[pc] == dp[rc]).mean()) if len(dp) else NAN, baseline_always_info=float(dp[rc].mean()) if len(dp) else NAN)
    ns = dec[dec.next_stronger]
    out["decided_next_stronger"] = dict(n=int(len(ns)), agreement=float((ns[pc] == ns[rc]).mean()) if len(ns) else NAN, baseline_always_info=float(ns[rc].mean()) if len(ns) else NAN)
    return out
C4 = {}
for tag, sub in {"crb_theta1": ivp[(ivp.dataset == "CRB") & (ivp.theta == 1.0) & (ivp.part != "all")], "crb_theta0.5": ivp[(ivp.dataset == "CRB") & (ivp.theta == 0.5) & (ivp.part != "all")],
                 "rb": ivp[(ivp.dataset == "RB") & (ivp.part != "all")], "pooled_main": pd.concat([ivp[(ivp.dataset == "CRB") & (ivp.theta == 1.0) & (ivp.part != "all")], ivp[(ivp.dataset == "RB") & (ivp.part != "all")]])}.items():
    C4[tag] = dict(equal_cost=c4_summary(sub, True), info_free_k0=c4_summary(sub, False))
C4["pooled_state_rows"] = ivp[ivp.part == "all"][["dataset", "theta", "primary", "upgrade", "s", "a", "b", "s_upgrade", "delta_p", "k_prime", "pred_info_minus_dec", "pred_lo", "pred_hi",
                                                  "real_info_minus_dec_eq", "real_lo", "real_hi", "decided_eq", "next_stronger"]].to_dict("records")
results["C4_info_vs_proposer"] = C4

# ---- (v) k < a necessary condition
def kc(sub):
    out = {}
    for kn in ["k_tok", "k_001b", "k_01b"]:
        out[kn] = dict(n_states=int(len(sub)), n_k_ge_a=int((sub[kn] >= sub.k_lt_a_bound).sum()), n_k_ge_a_1ms=int((sub[kn] >= sub.k_lt_a_1ms).sum()),
                       n_band_empty=int((sub[kn] >= sub.k_lt_band_nonempty).sum()), mean_k=float(sub[kn].mean()), mean_a=float(sub.a.mean()))
    return out
results["C5_k_condition"] = {"crb_theta1": kc(kcond[(kcond.dataset == "CRB") & (kcond.theta == 1.0)]), "rb": kc(kcond[kcond.dataset == "RB"]),
                             "note": "k_tok = prompt tokens x $0.054/M (CRB: mean input_tokens over the 8 models; RB: chars/4 proxy); released files carry no router tokens, so k=0 is the main case"}
_kk = kcond[(kcond.dataset == "CRB") & (kcond.theta == 1.0)]
results["C5_k_condition"]["crb_theta1_states_k_tok_ge_a_1ms"] = _kk[_kk.k_tok >= _kk.k_lt_a_1ms][["part", "state", "primary", "s", "a", "k_tok", "k_lt_a_1ms"]].to_dict("records")
results["C5_k_condition"]["crb_theta1_states_band_empty_at_k_tok"] = _kk[_kk.k_tok >= _kk.k_lt_band_nonempty][["part", "state", "primary", "s", "a", "b", "k_tok", "k_lt_band_nonempty"]].to_dict("records")

# ---- C5: classical routers' negative native capture reproduced (IN-SAMPLE, X3 definitions) and explained
pool = crb[(crb.part == "all") & (crb.theta == 1.0) & (crb.primary == "Qwen3-Max")]
x3_ref = {"finetuned_router_qwen35_08b": 0.452, "finetuned_router_qwen35_2b": 0.499, "finetuned_router_qwen35_9b_v3": 0.584, "finetuned_router_qwen35_27b_v3": 0.594,
          "logreg": -1.304, "tfidf_mlp": -1.859, "routellm_mf": -1.106, "routellm_sw": -1.375, "dimension_best": -1.571}
c5rows = []
for _, r in pool[pool.zdef == "this_primary"].iterrows():
    ident = (1 - r.P_Z1) * (r.s_given_Z0 * r.b_test - r.a_test)   # cell-mean identity: C_native - C_cascade = P(Z=0)(s_0 b - a)
    c5rows.append(dict(signal=r.signal, family=r.signal_family, P_Z1=r.P_Z1, s_given_Z1=r.s_given_Z1, s_given_Z0=r.s_given_Z0, a_over_b_test=r.a_test / r.b_test, auroc=r.auroc, ece=r.ece,
                       capture_native_ins_x3=r.capture_native_ins, x3_reference=x3_ref.get(r.signal), capture_native_oos=r.capture_native, capture_native_lo=r.capture_native_lo, capture_native_hi=r.capture_native_hi,
                       capture_bw_crossfit=r.capture_bw, capture_bw_lo=r.capture_bw_lo, capture_bw_hi=r.capture_bw_hi, capture_bw_ins=r.capture_bw_ins,
                       native_minus_cascade_realized=r.C_native - r.C_cascade, native_minus_cascade_identity_cellmeans=ident, s0_exceeds_a_over_b=bool(r.s_given_Z0 > r.a_test / r.b_test)))
c5 = pd.DataFrame(c5rows).set_index("signal")
# cost heterogeneity on the Z=0 branch: realized saving of sending Z=0 tasks direct = mean_{Z=0}[cost_P - H cost_B]; cell-mean version a_0 - s_0 b
_yQ = S.loc[TEST, "Qwen3-Max"].to_numpy(); _cQ = Cc.loc[TEST, "Qwen3-Max"].to_numpy(); _cB = Cc.loc[TEST, STRONG].to_numpy(); _HQ = _yQ >= 1.0
for r in SIGNALS_CRB:
    z0 = DEC[r].loc[TEST].to_numpy() != "Qwen3-Max"
    c5.loc[r, "a_given_Z0"] = float(_cQ[z0].mean()); c5.loc[r, "b_given_Z0"] = float(_cB[z0].mean())
    c5.loc[r, "b_given_Z0_H1"] = float(_cB[z0 & _HQ].mean()) if (z0 & _HQ).any() else NAN
    c5.loc[r, "direct_on_Z0_saving_per_Z0_task_realized"] = float((_cQ[z0] - _HQ[z0] * _cB[z0]).mean())
    c5.loc[r, "direct_on_Z0_saving_per_Z0_task_cellmeans"] = float(_cQ[z0].mean() - _HQ[z0].mean() * _cB.mean())
cl = c5.loc[sorted(CLASSICAL)]
results["C5_classical_routers"] = dict(rows=c5.reset_index().to_dict("records"),
                                       n_classical=int(len(cl)), n_classical_native_ci_below_0=int((cl.capture_native_hi < 0).sum()), n_classical_s0_above_ab=int(cl.s0_exceeds_a_over_b.sum()),
                                       n_classical_bw_capture_ge_0=int((cl.capture_bw_crossfit >= -1e-9).sum()), max_abs_diff_to_x3=float((cl.capture_native_ins_x3 - cl.x3_reference).abs().max()),
                                       max_abs_diff_to_x3_all=float((c5.dropna(subset=["x3_reference"]).capture_native_ins_x3 - c5.dropna(subset=["x3_reference"]).x3_reference).abs().max()),
                                       any_non_opus_rows=pool[pool.zdef == "any_non_opus"][["signal", "P_Z1", "s_given_Z1", "s_given_Z0", "auroc", "ece", "capture_native", "capture_native_lo", "capture_native_hi",
                                                                                          "capture_bw", "capture_bw_lo", "capture_bw_hi"]].to_dict("records"))
# pooled capture vs dimension-conditional capture (is the routers' value dimension information?)
dc = []
for r in SIGNALS_CRB:
    for zdef in ZDEFS:
        pr = crb[(crb.part == "all") & (crb.theta == 1.0) & (crb.primary == "Qwen3-Max") & (crb.signal == r) & (crb.zdef == zdef)].iloc[0]
        d9 = crb[(crb.part == "dim9") & (crb.theta == 1.0) & (crb.primary == "Qwen3-Max") & (crb.signal == r) & (crb.zdef == zdef)]
        ok = d9[~d9.degenerate_prior]
        dc.append(dict(signal=r, zdef=zdef, capture_bw_pooled=float(pr.capture_bw), capture_bw_pooled_lo=float(pr.capture_bw_lo), capture_bw_pooled_hi=float(pr.capture_bw_hi),
                       gross_bw_pooled_dollars=float(pr.gross_bw), sum_gross_bw_dim9_taskweighted_dollars=float((ok.gross_bw * ok.n_test).sum() / d9.n_test.sum()),
                       sum_bound_dim9_taskweighted_dollars=float((ok.bound_closed * ok.n_test).sum() / d9.n_test.sum()),
                       capture_bw_dim9_taskweighted=float((ok.gross_bw * ok.n_test).sum() / (ok.bound_closed * ok.n_test).sum()) if (ok.bound_closed * ok.n_test).sum() > 0 else NAN,
                       n_dim9_states_ci_above_0=int((ok.gross_bw_lo > 0).sum()), n_dim9_states_ci_below_0=int((ok.gross_bw_hi < 0).sum()), n_dim9_states_capture_0=int((ok.capture_bw == 0).sum()), n_dim9_nondegenerate=int(len(ok)),
                       dim9_states_ci_above_0=[f"{x.state}:{x.capture_bw:.2f}" for _, x in ok[ok.gross_bw_lo > 0].iterrows()]))
results["dimension_conditional_capture_qwen_theta1"] = dc
# quality side for the pooled state (all primaries, theta 1 and 0.5): E[V], floor, erroneous acceptance, eta/eta_B
qual = crb[(crb.part == "all") & (crb.zdef == ZDEF_MAIN) & (crb.signal == "finetuned_router_qwen35_27b_v3")][["theta", "primary", "s", "a_over_b", "u0_attempt", "V_direct", "V_u0", "V_native", "V_bw", "e_native", "e_bw", "eta_B", "eta_u0", "eta_native", "eta_bw"]]
results["quality_pooled_ft27b_any_non_opus"] = qual.to_dict("records")

# ================================================================================================ pooled tables for README
results["pooled_crb_theta1"] = crb[(crb.part == "all") & (crb.theta == 1.0)][["primary", "signal", "zdef", "s", "a_over_b", "P_Z1", "s_given_Z1", "s_given_Z0", "auroc", "ece", "bound_closed", "gross_native", "gross_bw",
                                                                             "capture_native", "capture_native_lo", "capture_native_hi", "capture_bw", "capture_bw_lo", "capture_bw_hi", "capture_bw_ins", "capture_perm", "capture_perm_lo", "capture_perm_hi",
                                                                             "net_bw_k01b", "net_bw_ktok", "u0_attempt", "frac_attempt_bw"]].to_dict("records")
results["pooled_rb"] = rbr[rbr.part == "all"][["primary", "signal", "s", "a_over_b", "auroc", "ece", "ece_posterior_held", "bound_closed", "gross_native", "gross_native_ab", "gross_bw", "capture_native", "capture_native_lo", "capture_native_hi",
                                              "capture_native_ab", "capture_native_ab_lo", "capture_native_ab_hi", "capture_bw", "capture_bw_lo", "capture_bw_hi", "capture_perm", "capture_perm_lo", "capture_perm_hi", "frac_attempt_native", "frac_attempt_native_ab", "frac_attempt_bw", "mean_posterior_held", "u0_attempt"]].to_dict("records")
results["fam8_rb_lr_tfidf"] = rbr[(rbr.part == "fam8") & (rbr.signal == "lr_tfidf")][["state", "primary", "n_test", "s", "a_over_b", "auroc", "ece", "capture_native", "capture_native_lo", "capture_native_hi", "capture_bw", "capture_bw_lo", "capture_bw_hi", "capture_perm"]].to_dict("records")
results["band_pooled"] = band[band.part == "all"][["dataset", "theta", "primary", "k_name", "k", "s", "a_over_b", "band_lo", "band_hi", "band_nonempty", "u", "pred_argmin", "real_argmin", "info_minus_best_uninformed", "lo", "hi", "decided"]].to_dict("records")
bnd = band[(band.part != "all") & (((band.dataset == "CRB") & (band.theta == 1.0)) | (band.dataset == "RB"))]
def band_agree(sub):
    d = sub[sub.decided]; return dict(n=int(len(sub)), n_decided=int(len(d)), agreement_decided=float((d.pred_info_in_band == (d.real_argmin == "info")).mean()) if len(d) else NAN,
                                      argmin_agreement_decided=float((d.pred_argmin == d.real_argmin).mean()) if len(d) else NAN, baseline_nonempty_band=float((d.band_nonempty == (d.real_argmin == "info")).mean()) if len(d) else NAN)
results["band_prop_voi_oos"] = {f"{ds}|k={kn}": band_agree(bnd[(bnd.dataset == ds) & (bnd.k_name == kn)]) for ds in ["CRB", "RB"] for kn in ["0.001b", "0.01b", "tok"]}

# ---- hypothesis violations measured
hv = {}
mc = crb[(crb.theta == 1.0) & (crb.part != "all") & (crb.zdef == ZDEF_MAIN) & (crb.signal == "finetuned_router_qwen35_27b_v3")]
hv["prior_drift_probing_to_id_test"] = dict(mean_abs_s_minus_s_test=float((mc.s - mc.s_test).abs().mean()), max_abs=float((mc.s - mc.s_test).abs().max()),
                                            n_states_u0_flips_in_sample=int(((mc.s > mc.a_over_b) != (mc.s_test > mc.a_test / mc.b_test)).sum()), n_states=int(len(mc)))
hv["anti_monotone_calibrations"] = dict(n_rows_with_any=int((crb.n_anti_monotone_calibrations > 0).sum()), n_rows=int(len(crb)), mean_frac=float((crb.n_anti_monotone_calibrations / (2 * NSPLIT)).mean()),
                                        by_family={f: float((g.n_anti_monotone_calibrations / (2 * NSPLIT)).mean()) for f, g in crb[crb.theta == 1.0].groupby("signal_family")},
                                        mono_vs_exact_max_abs_diff_gross=float((crb.gross_bw - crb.gross_bw_mono).abs().max()))
hv["value_preservation"] = {f"theta{t}": dict(n_rows=int(len(g)), frac_V_bw_below_V_direct=float((g.V_bw < g.V_direct - 1e-12).mean()), mean_V_bw_minus_V_direct=float((g.V_bw - g.V_direct).mean()), mean_e_native=float(g.e_native.mean()), mean_e_bw=float(g.e_bw.mean()))
                            for t, g in crb[crb.part != "all"].groupby("theta")}
hv["value_preservation"]["rb"] = dict(frac_V_bw_below_V_direct=float((rbr.V_bw < rbr.V_direct - 1e-12).mean()), mean_V_bw_minus_V_direct=float((rbr.V_bw - rbr.V_direct).mean()), mean_e_bw=float(rbr.e_bw.mean()))
hv["cost_heterogeneity"] = het
hv["rb_calibration_oos"] = dict(mean_ece_raw=float(main_rb.ece.mean()), mean_ece_isotonic_posterior_held=float(main_rb.ece_posterior_held.mean()),
                                mean_abs_posterior_mean_minus_s=float((main_rb.mean_posterior_held - main_rb.s).abs().mean()))
results["hypothesis_violations"] = hv

# ================================================================================================ claims
claims = []
c1 = C1["all_rows"]; nci = c1["bw"]["n_ci_violations"] + c1["native"]["n_ci_violations"]; npt = c1["bw"]["n_point_violations"] + c1["native"]["n_point_violations"]
claims.append(dict(id="C1", statement="No realized gross VOI of any usage of any signal exceeds the perfect-information bound min{s(b-a), a(1-s)} (probing/fit primitives); equivalently net VOI <= bound - k",
                   kind="IDENTITY-like inequality on expectations; realized OUT-OF-SAMPLE (CRB: probing primitives, id_test outcomes; RB: 70% fit, 30% held-out)",
                   result=dict(n_rows_tested=c1["n_rows_tested"], n_point_violations_bw=c1["bw"]["n_point_violations"], n_point_violations_native=c1["native"]["n_point_violations"],
                               n_ci_violations=nci, largest_ratio_bw=c1["bw"]["max_ratio"], largest_ratio_native=c1["native"]["max_ratio"], point_violation_rows=len(c1["point_violations"])),
                   verdict=("PASS" if nci == 0 and npt == 0 else "PASS (qualified: point violations without CI support)" if nci == 0 else "FAIL")))
b = C2["capture_bw"]["pooled"]; tb = b.get("task_boot", {}); dci = tb.get("abs_diff_ci", b["row_boot"]["abs_diff_ci"]); dlo = dci[0]
claims.append(dict(id="C2", statement="Across (state, primary, signal), the Blackwell capture ratio correlates more with AUROC than with ECE (Spearman)", kind="OUT-OF-SAMPLE",
                   result=dict(spearman_auroc=b["spearman_auroc"], spearman_ece=b["spearman_ece"], abs_diff=b["abs_diff"], abs_diff_ci_task_boot=tb.get("abs_diff_ci"), abs_diff_ci_row_boot=b["row_boot"]["abs_diff_ci"], n_rows=b["n_rows"],
                               per_block={k: dict(auroc=v["spearman_auroc"], ece=v["spearman_ece"], abs_diff=v["abs_diff"], abs_diff_ci_task=v.get("task_boot", {}).get("abs_diff_ci"), abs_diff_ci_row=v["row_boot"]["abs_diff_ci"], n=v["n_rows"]) for k, v in C2["capture_bw"].items() if isinstance(v, dict) and "spearman_auroc" in v},
                               native_capture_pooled=dict(auroc=C2["capture_native"]["pooled"]["spearman_auroc"], ece=C2["capture_native"]["pooled"]["spearman_ece"], abs_diff=C2["capture_native"]["pooled"]["abs_diff"], abs_diff_ci_row_boot=C2["capture_native"]["pooled"]["row_boot"]["abs_diff_ci"], abs_diff_ci_task_boot=C2["capture_native"]["pooled"].get("task_boot", {}).get("abs_diff_ci")),
                               shrunk_distortion_check=C2["shrunk_distortion_check"],
                               within_state_primary_bw=C2["within_state_primary"]["capture_bw"], straddle_bw_pooled=C2["straddle"]["capture_bw"]["pooled"], zero_capture=C2["zero_capture"]),
                   verdict=("PASS" if b["abs_diff"] > 0 and dci[0] > 0 else "FAIL" if b["abs_diff"] < 0 and dci[1] < 0 else "INCONCLUSIVE"),
                   verdict_rule="Exploratory pooled diagnostic: task-bootstrap CI of |Spearman(capture_bw, AUROC)| - |Spearman(capture_bw, ECE)| over all (state, primary, signal) rows: PASS if > 0, FAIL if < 0, else INCONCLUSIVE; the within-(state, primary) refinement and the native-usage variant are reported alongside, not used for the verdict"))
p3 = C3["pooled"]
claims.append(dict(id="C3", statement="A calibrated but uninformative signal (Z permuted within state) has net VOI ~ -k: its gross VOI is ~0 and never significantly positive", kind="SEMI-SYNTHETIC",
                   result=dict(n_rows=p3["n_rows"], mean_capture_perm=p3["mean_capture_perm"], median_abs_capture_perm=p3["median_abs_capture_perm"], q95_abs=p3["q95_abs_capture_perm"], n_ci_above_0=p3["n_ci_above_0"], n_ci_below_0=p3["n_ci_below_0"],
                               n_exactly_0=p3["n_exactly_0"], mean_net_over_k_at_k01b=p3["net_at_k"]["k01b"], real_signal_mean_capture_same_rows=p3["mean_capture_bw_real_same_rows"],
                               share_ci_above_0=C3["ci_above_0_rows"]["share"], ci_above_0_rows=C3["ci_above_0_rows"]),
                   verdict=("PASS" if C3["ci_above_0_rows"]["share"] <= 0.025 and p3["median_abs_capture_perm"] <= 0.02 and abs(p3["net_at_k"]["k01b"] + 1) <= 0.1
                            else "PASS (qualified)" if C3["ci_above_0_rows"]["share"] <= 0.025 else "FAIL"),
                   verdict_rule=("share of rows whose permuted gross VOI has a 95% CI above 0 must not exceed the nominal one-sided rate 2.5%, median |capture_perm| <= 0.02, mean net/k at k=0.01b within 10% of -1. "
                                 "CAVEAT: the first draft of this script demanded zero such rows; with >3,000 simultaneous intervals that is unattainable by chance and was replaced by the nominal-rate rule before the README was written")))
p4 = C4["pooled_main"]["equal_cost"]
claims.append(dict(id="C4", statement="cor:info-vs-proposer: at equal cost, buying exact routing information beats upgrading the primary iff b*Delta p < a(1-s); sign agreement on decided (state, primary, upgrade) rows",
                   kind="OUT-OF-SAMPLE", result=dict(n_rows=p4["n_rows"], n_decided=p4["n_decided"], agreement_decided=p4["agreement_decided"], baseline_always_info=p4["baseline_always_info_decided"],
                                                     baseline_always_upgrade=p4["baseline_always_upgrade_decided"], baseline_info_iff_dp_le_0=p4["baseline_info_iff_dp_le_0_decided"],
                                                     delta_p_positive=p4["decided_delta_p_positive"], next_stronger=p4["decided_next_stronger"], crb=C4["crb_theta1"]["equal_cost"]["agreement_decided"], rb=C4["rb"]["equal_cost"]["agreement_decided"]),
                   verdict=("PASS" if p4["agreement_decided"] >= 0.85 and p4["agreement_decided"] > max(p4["baseline_always_info_decided"], p4["baseline_always_upgrade_decided"], p4["baseline_info_iff_dp_le_0_decided"])
                            else "PASS (qualified: >= 0.85 but not above the best naive baseline)" if p4["agreement_decided"] >= 0.85 else "FAIL")))
p5 = results["C5_classical_routers"]
claims.append(dict(id="C5", statement="The negative capture of the five classical routers (X3 Table, pooled, Qwen3-Max, theta=1, Z = chose the primary) is reproduced and explained: P(H=1|Z=0) > a/b, so their native rule sends sufficient tasks to Opus that the uninformed cascade would have attempted; the Blackwell-optimal use of the same Z is never negative",
                   kind="IN-SAMPLE", result=dict(n_classical=p5["n_classical"], n_native_ci_below_0=p5["n_classical_native_ci_below_0"], n_s0_above_ab=p5["n_classical_s0_above_ab"], n_bw_capture_ge_0=p5["n_classical_bw_capture_ge_0"], max_abs_diff_to_x3=p5["max_abs_diff_to_x3_all"]),
                   verdict=("PASS" if p5["n_classical_native_ci_below_0"] == p5["n_classical"] and p5["n_classical_s0_above_ab"] == p5["n_classical"] and p5["n_classical_bw_capture_ge_0"] == p5["n_classical"] and p5["max_abs_diff_to_x3_all"] < 0.01 else "FAIL")))
results["claims"] = claims

# ================================================================================================ figures
FAMCOL = {"finetuned": (BLUE, "o", "CRB fine-tuned Qwen routers"), "classical": (ORANGE, "s", "CRB classical routers / DimensionBest"), "lookup": (ORANGE, "D", "ACRouter release lookup"), "rb_lr": (PURPLE, "^", "RouterBench LR routers")}
def scatter_fam(ax, sub, xcol, ycol):
    for fam, (col, mk, lab) in FAMCOL.items():
        g = sub[sub.signal_family == fam]
        if len(g): ax.scatter(g[xcol], g[ycol], s=9, color=col, marker=mk, alpha=.6, linewidths=0, label=lab)
# Fig 1 (C2): capture vs AUROC and vs ECE, Blackwell (top) and native (bottom)
fig, axes = plt.subplots(2, 2, figsize=(6.75, 4.4), layout="constrained")
for i, cap in enumerate(["capture_bw", "capture_native"]):
    for j, (xc, xl) in enumerate([("auroc", "AUROC of Z against sufficiency (held-out)"), ("ece", "ECE of Z (held-out)")]):
        ax = axes[i, j]; scatter_fam(ax, main_pool, xc, cap)
        r = C2[cap]["pooled"]["spearman_auroc" if xc == "auroc" else "spearman_ece"]
        ax.text(0.02, 0.97, fr"Spearman $\rho={r:.2f}$ (n={C2[cap]['pooled']['n_rows']})", transform=ax.transAxes, va="top", fontsize=7.5)
        ax.axhline(0, color=GREY, lw=.6); ax.axhline(1, color=GREY, lw=.6, ls=":")
        ax.set_yscale("symlog", linthresh=(0.1 if cap == "capture_bw" else 0.5))
        ax.set_xlabel(xl); ax.set_ylabel("Blackwell capture ratio (symlog)" if cap == "capture_bw" else "Native-usage capture ratio (symlog)")
        if xc == "auroc": ax.axvline(0.5, color=GREY, lw=.6, ls=":")
hh, ll = axes[0, 0].get_legend_handles_labels()
fig.legend(hh, ll, loc="outside lower center", ncol=4, frameon=False, fontsize=7, handletextpad=.2, columnspacing=1.2)
fig.savefig(os.path.join(FIG, "fig1_capture_vs_auroc_ece.pdf"), metadata={"Creator": "X12a run.py"}); plt.close(fig)
# Fig 2 (prop:voi band): band positions of s per state with the realized cheapest option (exact information at k = 0.01 b)
CH = {"cascade": (BLUE, "o", "cascade cheapest"), "direct": (ORANGE, "s", "direct cheapest"), "info": (PURPLE, "D", "exact information cheapest")}
fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.6), layout="constrained")
for ax, (ds, sub) in zip(axes, [("CRB", band[(band.dataset == "CRB") & (band.part == "dim9") & (band.theta == 1.0) & (band.k_name == "0.01b") & band.primary.isin(["Qwen3-Max", "kimi-k2.5", "gpt-5.4", "claude-sonnet-4-6"])]),
                              ("RB", band[(band.dataset == "RB") & (band.part == "fam8") & (band.k_name == "0.01b")])]):
    sub = sub.sort_values(["a_over_b", "state"]).reset_index(drop=True)
    for y, r in sub.iterrows():
        ax.plot([max(r.band_lo, 0), min(r.band_hi, 1)], [y, y], color=GREY, lw=2.2, alpha=.35, solid_capstyle="butt")
        ax.plot([r.a_over_b], [y], marker="|", color=GREY, ms=6, mew=.8)
        col, mk, _ = CH[r.real_argmin]; ax.plot([r.s], [y], marker=mk, color=col, ms=4, mew=0, ls="none", alpha=.9 if r.decided else .45)
    ax.set_yticks(range(len(sub))); ax.set_yticklabels([f"{r.state.replace('_', ' ')[:12]} | {r.primary.replace('claude-','').replace('mistralai/','').replace('zero-one-ai/','').replace('-1106','')[:11]}" for _, r in sub.iterrows()], fontsize=5.5)
    ax.set_xlim(0, 1); ax.set_title("CodeRouterBench, 9 dimensions, $\\theta=1$" if ds == "CRB" else "RouterBench, 8 source families", fontsize=8, loc="left")
    ax.set_xlabel(r"$s$ (fit split); grey: band at $k=0.01b$; tick: $a/b$")
    ax.invert_yaxis(); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
h = [plt.Line2D([], [], marker=mk, color=col, ls="none", ms=4, label=lab) for col, mk, lab in CH.values()]
fig.legend(handles=h, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
fig.savefig(os.path.join(FIG, "fig2_band_positions.pdf"), metadata={"Creator": "X12a run.py"}); plt.close(fig)
# Fig 3 (C1): realized gross VOI vs the closed-form bound
fig, ax = plt.subplots(figsize=(3.3, 2.6), layout="constrained")
ok = ALL[~ALL.degenerate_prior]
ax.scatter(ok.bound_closed * 1e3, ok.gross_bw * 1e3, s=6, color=BLUE, alpha=.45, linewidths=0, label="Blackwell usage")
ax.scatter(ok.bound_closed * 1e3, ok.gross_native * 1e3, s=6, color=ORANGE, marker="s", alpha=.35, linewidths=0, label="native usage")
lim = ok.bound_closed.max() * 1e3 * 1.2; ax.plot([1e-3, lim], [1e-3, lim], color=GREY, lw=.7, ls=":")
ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e-2); ax.axhline(0, color=GREY, lw=.5)
ax.set_xlabel(r"bound $\min\{s(b-a),\,a(1-s)\}$ (m\$/task, probing)"); ax.set_ylabel(r"realized gross VOI $f(s)-\mathbb{E}f(s_Z)$ (m\$/task)")
ax.legend(loc="upper left", frameon=False, fontsize=6.5)
fig.savefig(os.path.join(FIG, "fig3_bound_check.pdf"), metadata={"Creator": "X12a run.py"}); plt.close(fig)
# Fig 4 (C3): permuted-Z capture vs real capture
fig, ax = plt.subplots(figsize=(3.3, 2.6), layout="constrained")
pp = _c3[np.isfinite(_c3.capture_perm) & np.isfinite(_c3.capture_bw)]
bins = np.linspace(-0.3, 1.0, 53)
ax.hist(np.clip(pp.capture_bw, -0.3, 1.0), bins=bins, color=BLUE, alpha=.75, label="real Z (Blackwell usage)")
ax.hist(np.clip(pp.capture_perm, -0.3, 1.0), bins=bins, histtype="step", color=PURPLE, lw=1.2, label="Z permuted within state")
ax.axvline(0, color=GREY, lw=.6); ax.set_yscale("log"); ax.set_xlabel("capture ratio (gross VOI / bound), clipped to [-0.3, 1]"); ax.set_ylabel("(state, primary, signal) rows (log)")
ax.legend(loc="upper right", frameon=False, fontsize=6.5)
fig.savefig(os.path.join(FIG, "fig4_permuted_signal.pdf"), metadata={"Creator": "X12a run.py"}); plt.close(fig)
# Fig 5 (C4): predicted b dp - a(1-s) vs realized C_info(k') - C_dec
fig, ax = plt.subplots(figsize=(3.3, 2.6), layout="constrained")
iv = ivp[(ivp.part != "all") & (((ivp.dataset == "CRB") & (ivp.theta == 1.0)) | (ivp.dataset == "RB"))]
for ds, col, mk in [("CRB", BLUE, "o"), ("RB", PURPLE, "^")]:
    g = iv[iv.dataset == ds]
    ax.scatter(g[g.decided_eq].pred_info_minus_dec * 1e3, g[g.decided_eq].real_info_minus_dec_eq * 1e3, s=7, color=col, marker=mk, alpha=.6, linewidths=0, label=f"{ds} decided")
    ax.scatter(g[~g.decided_eq].pred_info_minus_dec * 1e3, g[~g.decided_eq].real_info_minus_dec_eq * 1e3, s=7, facecolors="none", edgecolors=col, marker=mk, alpha=.5, linewidths=.5, label=f"{ds} inconclusive")
ax.set_xscale("symlog", linthresh=0.1); ax.set_yscale("symlog", linthresh=0.1); ax.axhline(0, color=GREY, lw=.6); ax.axvline(0, color=GREY, lw=.6)
lim = max(iv.pred_info_minus_dec.abs().max(), iv.real_info_minus_dec_eq.abs().max()) * 1e3 * 1.2; ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=.6, ls=":")
ax.set_xlabel(r"predicted $b\,\Delta p-a(1-s)$ (m\$/task, probing)"); ax.set_ylabel(r"realized $C_{\rm info}(k')-C_{\rm dec}$ (m\$/task)")
ax.legend(loc="upper left", frameon=False, fontsize=6)
fig.savefig(os.path.join(FIG, "fig5_info_vs_proposer.pdf"), metadata={"Creator": "X12a run.py"}); plt.close(fig)
# Fig 6 (C5): native vs Blackwell capture per released router (pooled, Qwen3-Max, theta=1, Z = chose the primary) and P(H|Z=0) vs a/b
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.5), layout="constrained")
order = ROUTERS + ["acrouter_release"]; c5o = c5.loc[order]; xs = np.arange(len(order)); w = .38
ax = axes[0]
ax.bar(xs - w / 2, c5o.capture_native_ins_x3, w, color=ORANGE, label="native usage (in-sample, X3 definition)")
ax.bar(xs + w / 2, c5o.capture_bw_crossfit, w, color=BLUE, yerr=[np.maximum(c5o.capture_bw_crossfit - c5o.capture_bw_lo, 0), np.maximum(c5o.capture_bw_hi - c5o.capture_bw_crossfit, 0)], error_kw=dict(lw=.6), label="Blackwell usage (cross-fitted)")
ax.axhline(0, color=GREY, lw=.6); ax.axhline(1, color=GREY, lw=.6, ls=":")
ax.set_xticks(xs); ax.set_xticklabels([r.replace("finetuned_router_qwen35_", "ft-").replace("_v3", "").replace("acrouter_release", "ACRouter") for r in order], rotation=45, ha="right", fontsize=6.5)
ax.set_ylabel(r"fraction of $f(s)-g(s)$ captured"); ax.legend(loc="lower left", frameon=False, fontsize=6.5)
ax = axes[1]
ax.bar(xs, c5o.s_given_Z0, .6, color=[BLUE if f == "finetuned" else ORANGE for f in c5o.family])
ax.axhline(float(c5o.a_over_b_test.iloc[0]), color=PURPLE, lw=.8, ls="--"); ax.set_xlim(-0.6, 10.4); ax.text(9.45, float(c5o.a_over_b_test.iloc[0]) + .004, r"$a/b$", color=PURPLE, ha="left", fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels([r.replace("finetuned_router_qwen35_", "ft-").replace("_v3", "").replace("acrouter_release", "ACRouter") for r in order], rotation=45, ha="right", fontsize=6.5)
ax.set_ylabel(r"$\mathbb{P}(H=1\mid Z=0)$ on id\_test")
fig.savefig(os.path.join(FIG, "fig6_classical_routers.pdf"), metadata={"Creator": "X12a run.py"}); plt.close(fig)

# ================================================================================================ audit: independent recomputation (pure Python, no numpy/pandas)
def audit():
    out = {}
    # (1) X3-definition native capture, pooled id_test, Qwen3-Max, theta=1, Z = chose Qwen3-Max: csv module + loops
    sc = collections.defaultdict(dict); co = collections.defaultdict(dict); split = {}
    with open(f"{ACR}/coderouterbench/id_results_long.csv") as fh:
        for r in csv.DictReader(fh):
            sc[r["task_id"]][r["model"]] = float(r["score"]); co[r["task_id"]][r["model"]] = float(r["cost_usd"]); split[r["task_id"]] = r["split"]
    tids = [t for t in split if split[t] == "id_test"]
    def native_capture(router):
        dec = {}
        with open(f"{ACR}/id/voter_decisions/{router}.jsonl") as fh:
            for l in fh:
                if l.strip(): d = json.loads(l); dec[d["task_id"]] = d["chosen_model"]
        casc = dire = perf = cz = 0.0
        for t in tids:
            h = sc[t]["Qwen3-Max"] >= 1.0; cp = co[t]["Qwen3-Max"]; cb = co[t][STRONG]
            c_casc = cp + (0.0 if h else cb); casc += c_casc; dire += cb; perf += (cp if h else cb); cz += (c_casc if dec[t] == "Qwen3-Max" else cb)
        n = len(tids); f = min(casc, dire) / n
        return (f - cz / n) / (f - perf / n)
    for router in ["finetuned_router_qwen35_27b_v3", "logreg"]:
        mine = float(c5.loc[router, "capture_native_ins_x3"]); ind = native_capture(router)
        out[f"native_capture_ins_{router}"] = dict(run_py=mine, independent=ind, abs_diff=abs(mine - ind))
    # (2) RB pooled mixtral: fit-split closed-form bound and gross native VOI (p >= 0.5 rule) from python lists
    m = RB_PRIMARIES[0]; hp = [bool(x) for x in SUFF[m][fit_idx]]; cpf = [float(x) for x in COST[m][fit_idx]]; cbf = [float(x) for x in COST[RB_REF][fit_idx]]
    s = sum(hp) / len(hp); a = sum(cpf) / len(cpf); bb_ = sum(cbf) / len(cbf); bound = min(s * (bb_ - a), a * (1 - s))
    hh = [bool(x) for x in SUFF[m][held_idx]]; cph = [float(x) for x in COST[m][held_idx]]; cbh = [float(x) for x in COST[RB_REF][held_idx]]; zh = [float(x) for x in ZALL["lr_tfidf"][m][held_idx]]
    u0 = s > a / bb_; cu0 = cn = 0.0
    for h_, cp, cb, z in zip(hh, cph, cbh, zh):
        c_casc = cp + (0.0 if h_ else cb); cu0 += (c_casc if u0 else cb); cn += (c_casc if z >= 0.5 else cb)
    gross = (cu0 - cn) / len(hh)
    rrow = rbr[(rbr.part == "all") & (rbr.primary == m) & (rbr.signal == "lr_tfidf")].iloc[0]
    out["rb_pooled_mixtral_bound_closed"] = dict(run_py=float(rrow.bound_closed), independent=bound, abs_diff=abs(float(rrow.bound_closed) - bound))
    out["rb_pooled_mixtral_gross_native"] = dict(run_py=float(rrow.gross_native), independent=gross, abs_diff=abs(float(rrow.gross_native) - gross))
    # (3) Spearman(capture_bw, AUROC) pooled: hand-written average ranks vs scipy
    def ranks(v):
        idx = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[idx[j + 1]] == v[idx[i]]: j += 1
            for k in range(i, j + 1): r[idx[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    okp = main_pool[np.isfinite(main_pool.capture_bw) & np.isfinite(main_pool.auroc) & np.isfinite(main_pool.ece)]
    x = [float(v) for v in okp.capture_bw]; y = [float(v) for v in okp.auroc]; rx, ry = ranks(x), ranks(y); n = len(x); mx = sum(rx) / n; my = sum(ry) / n
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry)); den = math.sqrt(sum((p - mx) ** 2 for p in rx) * sum((q - my) ** 2 for q in ry)); rho = num / den
    out["spearman_capture_bw_auroc_pooled"] = dict(run_py=C2["capture_bw"]["pooled"]["spearman_auroc"], independent=rho, abs_diff=abs(C2["capture_bw"]["pooled"]["spearman_auroc"] - rho))
    out["max_abs_diff"] = max(v["abs_diff"] for v in out.values() if isinstance(v, dict) and "abs_diff" in v)
    return out
results["audit"] = audit()

results["runtime_seconds"] = time.time() - T0
results["n_rows"] = dict(crb_signal_rows=int(len(crb)), rb_signal_rows=int(len(rbr)), band_rows=int(len(band)), info_vs_proposer_rows=int(len(ivp)), k_condition_rows=int(len(kcond)))
with open(os.path.join(HERE, "results.json"), "w") as fh: json.dump(clean(results), fh, indent=1)
print("\n=== claims ===")
for c in claims: print(f"{c['id']}: {c['verdict']}  | {json.dumps(clean(c['result']))[:300]}")
print(f"audit max abs diff: {results['audit']['max_abs_diff']:.3e}")
print(f"runtime {results['runtime_seconds']:.0f} s")

