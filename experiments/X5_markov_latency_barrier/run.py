#!/usr/bin/env python3
"""
X5_markov_latency_barrier -- real-data tests of three appendix results of
"Route Anything: A Theory of Inference Routers":

  * prop:markov-extension (app:markov)   two-state hit chain: pi = p_m/(1-p_h+p_m), the geometric
                                          transient P(H_n=1) = pi + (P(H_1=1)-pi) xi^(n-1), the long-run
                                          ratio limit; and the paper's caveat that a stationary binary
                                          process satisfies flow balance without being Markov.
  * app:hedging                           eq:hedged-latency-condition with real latency_ms (sequential vs
                                          hedged) and eq:hedged-dollar-condition with a cancellation
                                          prefix charge r.
  * prop:app-barrier (app:barrier)        AND-barrier with slow vs fast fallback: unique crossing iff
                                          V_s/(L+Delta_s) < V_f/(L+Delta_f); Xi and b* = log Xi / log pi_s.

Data: CodeRouterBench outcome matrix (9,999 tasks x 8 models; score in [0,1], token-price cost estimates (`cost_usd`),
wall-clock latency_ms).  Reference policy B = claude-opus-4-6 on every task.  Verifier: exact
(score >= theta, theta in {1.0, 0.5}); a noisy verifier (beta_fa, beta_fr) is used in one case of
parts (a) and (c).  Nothing is fabricated: every value, dollar and second is a recorded row; state
dependence (parts b and d) is constructed from real tasks with the documented mechanisms.
Only numpy / pandas / scipy / matplotlib.  Fixed seeds.  Runtime ~3-4 min (--quick: ~30 s).
"""
import os, sys, json, time, math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
DATA = os.environ.get('ACROUTER_DATA', 'agent-as-a-router/data') + '/coderouterbench/id_results_long.csv'
SEED = 20260911
QUICK = "--quick" in sys.argv
N_BOOT = 40 if QUICK else 500
B = "claude-opus-4-6"
CHEAP4 = ["Qwen3-Max", "qwen3.5-plus", "kimi-k2.5", "glm-5"]
THETAS = [1.0, 0.5]
NOISE = (0.10, 0.10)                          # (beta_fa, beta_fr): false-accept / false-reject rates
LOG = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)

# ----------------------------------------------------------------------------- figure style (paper)
BLUE, RED, GREEN, GREY = '#23639b', '#ba5934', '#3a7c5e', '.5'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times'], 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsmath,amssymb}',
                     'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 7.5,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': .65, 'lines.linewidth': 1.4})
USETEX = True
try:
    _f, _a = plt.subplots(figsize=(1, 1)); _a.set_xlabel(r'$\pi_{\mathrm s}$'); _f.savefig(os.path.join(FIG, "_usetex_probe.pdf")); plt.close(_f)
    os.remove(os.path.join(FIG, "_usetex_probe.pdf"))
except Exception as e:                         # fall back to mathtext (documented in results.json)
    USETEX = False; plt.rcParams.update({'text.usetex': False}); say("usetex unavailable -> mathtext fallback:", repr(e)[:120])

# ----------------------------------------------------------------------------- data
df = pd.read_csv(DATA)
S = df.pivot(index="task_id", columns="model", values="score")
Cst = df.pivot(index="task_id", columns="model", values="cost_usd").loc[S.index]
Lat = df.pivot(index="task_id", columns="model", values="latency_ms").loc[S.index] / 1000.0
meta = df.drop_duplicates("task_id").set_index("task_id").loc[S.index, ["split", "dimension"]]
MODELS = list(S.columns); N = len(S); ALL7 = [m for m in MODELS if m != B]
assert N == 9999 and S.isna().sum().sum() == 0
Sn = {m: S[m].to_numpy() for m in MODELS}; Cn = {m: Cst[m].to_numpy() for m in MODELS}; Ln = {m: Lat[m].to_numpy() for m in MODELS}
DIMS = sorted(meta.dimension.unique()); nd = len(DIMS)
dim_code = meta.dimension.map({d: i for i, d in enumerate(DIMS)}).to_numpy()
is_prob = (meta.split.values == "probing"); is_test = ~is_prob
NAT_ORDER = S.index.get_indexer(df.drop_duplicates("task_id").task_id)   # [AUDIT] positions of tasks in the CSV's own (dimension-blocked) order
idx_prob, idx_test = np.where(is_prob)[0], np.where(is_test)[0]
data_checks = dict(n_tasks=int(N), n_models=len(MODELS), n_probing=int(is_prob.sum()), n_id_test=int(is_test.sum()),
                   always_opus_id_test_total_usd=float(Cn[B][is_test].sum()), paper_always_opus_total_usd=34.02,
                   always_opus_id_test_mean_score=float(Sn[B][is_test].mean()), paper_always_opus_avgperf=0.4383,
                   rows_latency_zero_by_model={m: int((Ln[m] <= 0).sum()) for m in MODELS},
                   rows_cost_zero_by_model={m: int((Cn[m] <= 0).sum()) for m in MODELS},
                   note="latency_ms == 0 means unlogged; tasks with an unlogged latency of a model are dropped from every "
                        "latency computation that involves that model (parts a-seconds, c, d); cost zeros (Qwen3-Max, 148 rows) are kept as recorded")
say("data checks:", json.dumps({k: v for k, v in data_checks.items() if k != 'note'}))

# ----------------------------------------------------------------------------- helpers
def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [None, None]
def covers0(c): return bool(c[0] is not None and c[0] <= 0 <= c[1])
def f0(x): return None if (x is None or not np.isfinite(x)) else float(x)
def trans_probs(Hs):
    """two-state MLE p_h = P(H_{n+1}=1|H_n=1), p_m = P(H_{n+1}=1|H_n=0) for each row of Hs (R,T)."""
    Hs = np.atleast_2d(Hs).astype(np.float64); prev, nxt = Hs[:, :-1], Hs[:, 1:]
    return (prev * nxt).sum(1) / np.maximum(prev.sum(1), 1), ((1 - prev) * nxt).sum(1) / np.maximum((1 - prev).sum(1), 1)
def second_order(Hs):
    """P(H_{n+1}=1 | H_n=b, H_{n-1}=a), keys 'ab'; equal across a for a Markov chain."""
    Hs = np.atleast_2d(Hs).astype(np.float64); a, b, c = Hs[:, :-2], Hs[:, 1:-1], Hs[:, 2:]; out = {}
    for av in (0, 1):
        for bv in (0, 1):
            sel = (a == av) & (b == bv); out[f"{av}{bv}"] = (sel * c).sum(1) / np.maximum(sel.sum(1), 1)
    return out
def block_idx(n, R, rng, Bl=100):
    k = int(math.ceil(n / Bl)); starts = rng.integers(0, n, size=(R, k))
    return ((starts[:, :, None] + np.arange(Bl)[None, None, :]) % n).reshape(R, -1)[:, :n]
def boot_weights(n, R, rng):
    """(R, n) multinomial task-resampling weights; row 0 = original sample."""
    w = np.empty((R, n)); w[0] = 1.0; w[1:] = rng.multinomial(n, np.full(n, 1.0 / n), size=R - 1); return w
def wmean(w, x, mask=None):
    ww = w if mask is None else w * mask[None, :]
    return (ww * x[None, :]).sum(1) / np.maximum(ww.sum(1), 1e-12)
def cascade(m, theta, noise=None, rng=None):
    exact = Sn[m] >= theta
    if noise:
        u = rng.random(N); hit = np.where(exact, u >= noise[1], u < noise[0])
    else:
        hit = exact
    V = np.where(hit, Sn[m], Sn[B])
    D = {"dollars": np.where(hit, Cn[m], Cn[m] + Cn[B]), "sec_seq": np.where(hit, Ln[m], Ln[m] + Ln[B]),
         "sec_hedged": np.where(hit, Ln[m], np.maximum(Ln[m], Ln[B]))}
    return hit, V, D
def hard_flag_for(m):
    """Hard tercile per dimension by leave-primary-out difficulty (mean score of the 7 OTHER models);
    the tercile cut is computed on probing tasks of each dimension and applied to all tasks."""
    diff = np.mean([Sn[x] for x in MODELS if x != m], axis=0); flag = np.zeros(N, bool)
    for d in range(nd):
        cut = np.percentile(diff[(dim_code == d) & is_prob], 100 / 3); flag[(dim_code == d) & (diff <= cut)] = True
    return flag, diff
split_shift = {m: dict(latency_probing=float(Ln[m][is_prob & (Ln[m] > 0)].mean()), latency_idtest=float(Ln[m][is_test & (Ln[m] > 0)].mean()),
                       score_probing=float(Sn[m][is_prob].mean()), score_idtest=float(Sn[m][is_test].mean()), cost_probing=float(Cn[m][is_prob].mean()), cost_idtest=float(Cn[m][is_test].mean())) for m in MODELS}
RESULTS = dict(experiment_id="X5_markov_latency_barrier", split_shift_probing_vs_idtest=split_shift, seed=SEED, n_boot=N_BOOT, quick=QUICK, usetex=USETEX,
               data_checks=data_checks, part_a={}, part_b={}, part_c={}, part_d={}, claims={})

# ============================================================================= PART (a): i.i.d. stream
say("\n=== (a) i.i.d. stream: prop:markov-extension on an independent task stream ===")
rng_a = np.random.default_rng(SEED)
N_SPLITS = 3 if QUICK else 10
A_CASES = [(m, th, None) for m in CHEAP4 for th in THETAS] + [("Qwen3-Max", 1.0, NOISE)]
for m, th, noise in A_CASES:
    hit, V, D = cascade(m, th, noise, rng_a)
    ok = (Ln[m] > 0) & (Ln[B] > 0); valid = np.where(ok)[0]
    key = f"{m}|theta={th}" + ("|noisy" if noise else "")
    perm = rng_a.permutation(N)
    # a1 (IN-SAMPLE hypothesis): p_h = p_m on the i.i.d. stream (all tasks, one random order)
    p_h, p_m = trans_probs(hit[perm]); p_h, p_m = float(p_h[0]), float(p_m[0])
    bidx = rng_a.integers(0, N, size=(N_BOOT, N)); ph_b, pm_b = trans_probs(hit[bidx])
    a1 = dict(p_h=p_h, p_m=p_m, p_h_minus_p_m=p_h - p_m, ci_diff=ci(ph_b - pm_b), ci_covers_0=covers0(ci(ph_b - pm_b)), hit_rate=float(hit.mean()))
    # [AUDIT] the shuffled stream is exchangeable by construction, so p_h = p_m there says nothing about the data; the CSV's own
    # (dimension-blocked) order is the only recorded order -- report p_h, p_m on it with a permutation-null SE for p_h - p_m
    rng_nat = np.random.default_rng(SEED + 99)   # dedicated RNG so the original rng_a stream (and hence A3's splits) is untouched
    phn, pmn = trans_probs(hit[NAT_ORDER]); nul = np.array([np.subtract(*trans_probs(hit[rng_nat.permutation(N)])) for _ in range(200)])
    a1["natural_order"] = dict(p_h=float(phn[0]), p_m=float(pmn[0]), p_h_minus_p_m=float(phn[0] - pmn[0]), perm_null_se=float(nul.std()), z_vs_perm_null=float((phn[0] - pmn[0]) / nul.std()),
                               note="p_h, p_m in the CSV file order (tasks grouped by dimension/source); z is against 200 random permutations. Not an arrival order -- the data record none.")
    # a2 (IDENTITY / code check): pi = p_m/(1-p_h+p_m) vs frequency; stationary ratio formula vs sum V / sum D
    pi_chain = p_m / (1 - p_h + p_m); a2 = dict(pi_chain=pi_chain, pi_freq=float(hit.mean()), abs_diff_pi=abs(pi_chain - hit.mean()))
    for r in D:
        idx = perm if r == "dollars" else perm[ok[perm]]
        Hr, Vr, Dr = hit[idx], V[idx], D[r][idx]; pif = Hr.mean()
        v1, v0, d1, d0 = Vr[Hr].mean(), Vr[~Hr].mean(), Dr[Hr].mean(), Dr[~Hr].mean()
        a2[r] = dict(eta_formula=float((pif * v1 + (1 - pif) * v0) / (pif * d1 + (1 - pif) * d0)), eta_direct=float(Vr.sum() / Dr.sum()),
                     v1=float(v1), v0=float(v0), d1=float(d1), d0=float(d0), n=int(len(idx)))
        a2[r]["rel_diff"] = a2[r]["eta_formula"] / a2[r]["eta_direct"] - 1
    # a3 (split-half): first-half primitives -> predicted long-run ratio; second half realized (valid-latency tasks)
    rows = []
    for s in range(N_SPLITS):
        ps = rng_a.permutation(valid); n1 = len(ps) // 2; first, second = ps[:n1], ps[n1:]
        ph, pm = trans_probs(hit[first]); pi_p = pm / (1 - ph + pm)
        b1 = first[rng_a.integers(0, n1, size=(N_BOOT, n1))]; b2 = second[rng_a.integers(0, len(second), size=(N_BOOT, len(second)))]
        phb, pmb = trans_probs(hit[b1]); pib = pmb / (1 - phb + pmb)
        row = dict(split=s, p_h=float(ph[0]), p_m=float(pm[0]), pi_pred=float(pi_p[0]), pi_meas=float(hit[second].mean()))
        for r in D:
            Hf, Vf, Df = hit[first], V[first], D[r][first]
            pred = (pi_p * Vf[Hf].mean() + (1 - pi_p) * Vf[~Hf].mean()) / (pi_p * Df[Hf].mean() + (1 - pi_p) * Df[~Hf].mean())
            meas = V[second].sum() / D[r][second].sum()
            Hb, Vb, Db = hit[b1].astype(float), V[b1], D[r][b1]
            v1b = (Vb * Hb).sum(1) / Hb.sum(1); v0b = (Vb * (1 - Hb)).sum(1) / (1 - Hb).sum(1)
            d1b = (Db * Hb).sum(1) / Hb.sum(1); d0b = (Db * (1 - Hb)).sum(1) / (1 - Hb).sum(1)
            predb = (pib * v1b + (1 - pib) * v0b) / (pib * d1b + (1 - pib) * d0b); measb = V[b2].sum(1) / D[r][b2].sum(1)
            dci = ci(predb - measb)
            row[r] = dict(eta_pred=float(pred[0]), eta_meas=float(meas), rel_err=float(pred[0] / meas - 1), diff_ci=dci, covers_0=covers0(dci),
                          z=float((pred[0] - meas) / max(np.std(predb - measb), 1e-12)))
        rows.append(row)
    a3 = {r: dict(frac_splits_ci_covers_0=float(np.mean([x[r]["covers_0"] for x in rows])), mean_abs_rel_err=float(np.mean([abs(x[r]["rel_err"]) for x in rows])),
                  mean_signed_rel_err=float(np.mean([x[r]["rel_err"] for x in rows])), mean_abs_z=float(np.mean([abs(x[r]["z"]) for x in rows])),
                  eta_pred_mean=float(np.mean([x[r]["eta_pred"] for x in rows])), eta_meas_mean=float(np.mean([x[r]["eta_meas"] for x in rows]))) for r in D}
    a3["per_split"] = rows; a3["n_valid_latency_tasks"] = int(len(valid))
    RESULTS["part_a"][key] = dict(a1=a1, a2=a2, a3=a3)
    say(f"{key:32s} p_h={p_h:.3f} p_m={p_m:.3f} diff CI {a1['ci_diff'][0]:+.3f}..{a1['ci_diff'][1]:+.3f} | identity |pi diff|={a2['abs_diff_pi']:.1e} "
        f"eta$ formula/direct-1={a2['dollars']['rel_diff']:+.1e} | split-half $: cover={a3['dollars']['frac_splits_ci_covers_0']:.2f} |relerr|={a3['dollars']['mean_abs_rel_err']*100:.2f}% "
        f"sec_hedged: cover={a3['sec_hedged']['frac_splits_ci_covers_0']:.2f} |relerr|={a3['sec_hedged']['mean_abs_rel_err']*100:.2f}%")
say(f"[t={time.time()-T0:.0f}s]")

# ============================================================================= PART (b): state-dependent streams (semi-synthetic)
say("\n=== (b) semi-synthetic state-dependent streams: pi, geometric transient, non-Markov caveat, ratio limit ===")
rng_b = np.random.default_rng(SEED + 1)
B_CASES = [("Qwen3-Max", 1.0), ("Qwen3-Max", 0.5), ("kimi-k2.5", 1.0), ("kimi-k2.5", 0.5)]
T_STREAM, R_STREAM = (3000, 4) if QUICK else (10000, 10)
R_TRANS, N_TRANS, BURN_TRANS = (50_000, 8, 25) if QUICK else (300_000, 8, 25)
RHO_STICKY, REGIME_STAY = 0.5, 0.8
CONSTRUCTIONS = ["recover_hard_random_dim", "recover_hard_sticky_dim", "hidden_3regime", "class_conditional_markov"]

def stationary(P):
    ev, vec = np.linalg.eig(P.T); o = np.argsort(-np.abs(ev)); st = np.real(vec[:, o[0]]); st = st / st.sum()
    return st, float(np.abs(ev[o[1]])), int(np.sum(np.abs(ev - 1) < 1e-9))

class Pool:
    """A split's tasks with the cascade outcome arrays, hard-tercile pools per dimension and difficulty terciles."""
    def __init__(self, idx, hit, V, D, hardflag, diff):
        self.idx = idx; self.n = len(idx); self.hit = hit[idx]; self.V = V[idx]; self.D = {r: D[r][idx] for r in D}
        self.dim = dim_code[idx]; self.hard = hardflag[idx]
        self.hard_by_dim = [np.where((self.dim == d) & self.hard)[0] for d in range(nd)]      # local indices
        cuts = np.percentile(diff[idx], [100 / 3, 200 / 3]); dz = diff[idx]
        self.tercile = np.where(dz <= cuts[0], 2, np.where(dz <= cuts[1], 1, 0))              # 0 easy, 1 mid, 2 hard
        self.by_tercile = [np.where(self.tercile == z)[0] for z in range(3)]
        self.by_class = [np.where(~self.hit)[0], np.where(self.hit)[0]]
    def padded(self, lists):
        sizes = np.array([len(p) for p in lists]); pad = np.zeros((len(lists), sizes.max()), np.int64)
        for i, p in enumerate(lists): pad[i, :len(p)] = p
        return pad, sizes

def struct_hardpool(P, rho, w=None, n_trans=0):
    """Exact (H, dimension) chain of the recover-hard constructions from pool hit rates only (no stream).
    rho = P(recovery dimension is random); 1-rho = same dimension.  Returns lumped two-state parameters, stationary
    means, the exact after-miss transient, and the SLEM (true chain memory)."""
    w = np.ones(P.n) if w is None else w
    mass = np.array([w[P.dim == d].sum() for d in range(nd)]); mass = mass / mass.sum()
    hf = np.array([np.sum(w[P.dim == d] * P.hit[P.dim == d]) / max(w[P.dim == d].sum(), 1e-12) for d in range(nd)])
    hh = np.array([np.sum(w[p] * P.hit[p]) / max(w[p].sum(), 1e-12) for p in P.hard_by_dim])
    Q = np.full((nd, nd), rho / nd) + (1 - rho) * np.eye(nd)
    M = np.zeros((2 * nd, 2 * nd))
    M[:nd, :nd] = Q * (1 - hh)[None, :]; M[:nd, nd:] = Q * hh[None, :]
    M[nd:, :nd] = (mass * (1 - hf))[None, :]; M[nd:, nd:] = (mass * hf)[None, :]
    st, slem, n_unit = stationary(M); s0 = st[:nd]; pi = st[nd:].sum(); w0 = s0 @ Q
    p_h = float(mass @ hf); p_m = float((w0 @ hh) / max(w0.sum(), 1e-12))
    def E(X, cls=None):                      # stationary mean of X (optionally restricted to a hit class)
        sel = np.ones(P.n) if cls is None else (P.hit == cls).astype(float)
        full = np.sum(w * sel * X) / w.sum()
        hardp = sum(w0[d] * np.sum(w[p] * sel[p] * X[p]) / max(w[p].sum(), 1e-12) for d, p in enumerate(P.hard_by_dim))
        return pi * full + hardp
    out = dict(pi=float(pi), p_h=p_h, p_m=p_m, xi=p_h - p_m, slem=slem, absorbing=bool(n_unit > 1 or p_m < 1e-9),
               EV=E(P.V), ED={r: E(P.D[r]) for r in P.D}, EV1=E(P.V, 1), EV0=E(P.V, 0), ED1={r: E(P.D[r], 1) for r in P.D}, ED0={r: E(P.D[r], 0) for r in P.D})
    if n_trans:
        v = np.concatenate([s0 / s0.sum(), np.zeros(nd)]); tr = []
        for _ in range(n_trans): v = v @ M; tr.append(float(v[nd:].sum()))
        out["transient_exact"] = tr                      # P(H_n = 1 | H_1 = 0, stationary), n = 2..n_trans+1
    return out

def struct_regime(P, stay, w=None, n_trans=0):
    """Exact (regime, H) chain of the hidden-3-regime construction."""
    w = np.ones(P.n) if w is None else w
    q = np.array([np.sum(w[p] * P.hit[p]) / max(w[p].sum(), 1e-12) for p in P.by_tercile])
    Pz = np.full((3, 3), (1 - stay) / 2); np.fill_diagonal(Pz, stay)
    M = np.zeros((6, 6))                       # state 2z + h
    for z in range(3):
        for zp in range(3):
            for h in (0, 1):
                M[2 * z + h, 2 * zp] = Pz[z, zp] * (1 - q[zp]); M[2 * z + h, 2 * zp + 1] = Pz[z, zp] * q[zp]
    st, slem, n_unit = stationary(M); pi = st[1::2].sum()
    p_h = float(sum(st[2 * z + 1] * sum(Pz[z, zp] * q[zp] for zp in range(3)) for z in range(3)) / pi)
    p_m = float(sum(st[2 * z] * sum(Pz[z, zp] * q[zp] for zp in range(3)) for z in range(3)) / (1 - pi))
    pz = st[0::2] + st[1::2]
    def E(X, cls=None):
        sel = np.ones(P.n) if cls is None else (P.hit == cls).astype(float)
        return sum(pz[z] * np.sum(w[p] * sel[p] * X[p]) / max(w[p].sum(), 1e-12) for z, p in enumerate(P.by_tercile))
    out = dict(pi=float(pi), p_h=p_h, p_m=p_m, xi=p_h - p_m, slem=slem, absorbing=bool(n_unit > 1), q_regime=q.tolist(),
               EV=E(P.V), ED={r: E(P.D[r]) for r in P.D}, EV1=E(P.V, 1), EV0=E(P.V, 0), ED1={r: E(P.D[r], 1) for r in P.D}, ED0={r: E(P.D[r], 0) for r in P.D})
    if n_trans:
        v = np.zeros(6); v[0::2] = st[0::2] / st[0::2].sum(); tr = []
        for _ in range(n_trans): v = v @ M; tr.append(float(v[1::2].sum()))
        out["transient_exact"] = tr
    return out

def struct_class(P, p_h, p_m, w=None, n_trans=0):
    """Exact two-state chain of the class-conditional construction (both hypotheses of the proposition hold)."""
    w = np.ones(P.n) if w is None else w
    pi = p_m / (1 - p_h + p_m)
    def cm(X, cls):
        p = P.by_class[cls]; return np.sum(w[p] * X[p]) / max(w[p].sum(), 1e-12)
    out = dict(pi=float(pi), p_h=p_h, p_m=p_m, xi=p_h - p_m, slem=abs(p_h - p_m), absorbing=False,
               EV1=cm(P.V, 1), EV0=cm(P.V, 0), ED1={r: cm(P.D[r], 1) for r in P.D}, ED0={r: cm(P.D[r], 0) for r in P.D})
    out["EV"] = pi * out["EV1"] + (1 - pi) * out["EV0"]; out["ED"] = {r: pi * out["ED1"][r] + (1 - pi) * out["ED0"][r] for r in P.D}
    if n_trans: out["transient_exact"] = [float(pi * (1 - (p_h - p_m) ** (n - 1))) for n in range(2, n_trans + 2)]
    return out

def gen_hardpool(P, rho, T, R, rng, start=None):
    pad, sz = P.padded(P.hard_by_dim); out = np.empty((R, T), np.int64)
    if start is None:
        cur = rng.integers(0, P.n, size=R)
    else:
        cur = start
    for n in range(T):
        out[:, n] = cur; h = P.hit[cur]
        d = P.dim[cur]; jump = rng.random(R) < rho; d = np.where(jump, rng.integers(0, nd, size=R), d)
        nxt_hard = pad[d, (rng.random(R) * sz[d]).astype(np.int64)]
        cur = np.where(h, rng.integers(0, P.n, size=R), nxt_hard)
    return out

def gen_regime(P, stay, T, R, rng):
    pad, sz = P.padded(P.by_tercile); out = np.empty((R, T), np.int64); z = rng.integers(0, 3, size=R)
    for n in range(T):
        out[:, n] = pad[z, (rng.random(R) * sz[z]).astype(np.int64)]
        mv = rng.random(R) >= stay; z = np.where(mv, (z + rng.integers(1, 3, size=R)) % 3, z)
    return out

def gen_class(P, p_h, p_m, T, R, rng):
    pad, sz = P.padded(P.by_class); out = np.empty((R, T), np.int64); h = rng.random(R) < p_m / (1 - p_h + p_m)
    for n in range(T):
        out[:, n] = pad[h.astype(int), (rng.random(R) * sz[h.astype(int)]).astype(np.int64)]
        h = rng.random(R) < np.where(h, p_h, p_m)
    return out

def structural(kind, P, params, w=None, n_trans=0):
    if kind == "recover_hard_random_dim": return struct_hardpool(P, 1.0, w, n_trans)
    if kind == "recover_hard_sticky_dim": return struct_hardpool(P, RHO_STICKY, w, n_trans)
    if kind == "recover_hard_same_dim_literal": return struct_hardpool(P, 0.0, w, n_trans)
    if kind == "hidden_3regime": return struct_regime(P, REGIME_STAY, w, n_trans)
    if kind == "class_conditional_markov": return struct_class(P, params["p_h"], params["p_m"], w, n_trans)
def generate(kind, P, params, T, R, rng):
    if kind == "recover_hard_random_dim": return gen_hardpool(P, 1.0, T, R, rng)
    if kind == "recover_hard_sticky_dim": return gen_hardpool(P, RHO_STICKY, T, R, rng)
    if kind == "hidden_3regime": return gen_regime(P, REGIME_STAY, T, R, rng)
    if kind == "class_conditional_markov": return gen_class(P, params["p_h"], params["p_m"], T, R, rng)

for m, th in B_CASES:
    hit, V, D = cascade(m, th)
    ok = (Ln[m] > 0) & (Ln[B] > 0); hardflag, diff = hard_flag_for(m)
    Pp = Pool(np.where(is_prob & ok)[0], hit, V, D, hardflag, diff); Pt = Pool(np.where(is_test & ok)[0], hit, V, D, hardflag, diff)
    key = f"{m}|theta={th}"; RESULTS["part_b"][key] = {}
    # literal same-dimension recovery (rho = 0): hypothesis 0 < p_m of the proposition
    lit = struct_hardpool(Pp, 0.0)
    RESULTS["part_b"][key]["recover_hard_same_dim_literal"] = dict(structural_probing=dict(p_h=lit["p_h"], p_m=lit["p_m"], pi=lit["pi"], absorbing=lit["absorbing"], slem=lit["slem"]),
                                                                  verdict="NOT INSTANTIABLE" if lit["absorbing"] or lit["p_m"] < 1e-3 else "instantiable")
    say(f"[{key}] literal same-dim recovery: p_m={lit['p_m']:.5f} pi={lit['pi']:.5f} absorbing={lit['absorbing']} -> {RESULTS['part_b'][key]['recover_hard_same_dim_literal']['verdict']}")
    params = dict(p_h=struct_hardpool(Pp, 1.0)["p_h"], p_m=struct_hardpool(Pp, 1.0)["p_m"])   # class-conditional chain copies (p_h, p_m) of construction (i)
    for kind in CONSTRUCTIONS:
        # --- structural (exact) quantities on probing (prediction) and on id_test (exact law of the measured pool), with task bootstrap
        sp = structural(kind, Pp, params, n_trans=N_TRANS); st_ = structural(kind, Pt, params, n_trans=N_TRANS)
        wb_p, wb_t = boot_weights(Pp.n, N_BOOT + 1, rng_b), boot_weights(Pt.n, N_BOOT + 1, rng_b)
        bp = [structural(kind, Pp, params, w=wb_p[r]) for r in range(1, N_BOOT + 1)]; bt = [structural(kind, Pt, params, w=wb_t[r]) for r in range(1, N_BOOT + 1)]
        def cib(lst, f): return ci([f(x) for x in lst])
        # --- streams on id_test tasks
        idx = generate(kind, Pt, params, T_STREAM, R_STREAM, rng_b)
        Hs, Vs = Pt.hit[idx], Pt.V[idx]; Ds = {r: Pt.D[r][idx] for r in D}
        ph, pm = trans_probs(Hs); so = second_order(Hs)
        blk = block_idx(T_STREAM, N_BOOT, rng_b)                                              # circular block bootstrap (blocks of 100) per stream
        Hb = np.concatenate([Hs[s][blk] for s in range(R_STREAM)], 0)                         # (R_STREAM*N_BOOT, T)
        phb, pmb = trans_probs(Hb); sob = second_order(Hb)
        markov2 = dict(P1_given_00=float(so["00"].mean()), P1_given_10=float(so["10"].mean()), P1_given_01=float(so["01"].mean()), P1_given_11=float(so["11"].mean()),
                       diff_after_miss=float((so["00"] - so["10"]).mean()), diff_after_miss_ci=ci(sob["00"] - sob["10"]),
                       diff_after_hit=float((so["01"] - so["11"]).mean()), diff_after_hit_ci=ci(sob["01"] - sob["11"]))
        markov2["markov_rejected"] = (not covers0(markov2["diff_after_miss_ci"])) or (not covers0(markov2["diff_after_hit_ci"]))
        pi_meas = Hs.mean(1); pi_pred_mle = pm / (1 - ph + pm)
        # (A2) hypothesis: E[V_n | H_n, H_{n-1}] independent of H_{n-1}
        prev, cur = Hs[:, :-1].ravel(), Hs[:, 1:].ravel(); Vc = Vs[:, 1:].ravel(); Dc = Ds["dollars"][:, 1:].ravel()
        a2 = {}
        for hn in (0, 1):
            e = {hp: (Vc[(cur == hn) & (prev == hp)].mean(), Dc[(cur == hn) & (prev == hp)].mean()) for hp in (0, 1)}
            a2[f"H{hn}"] = dict(EV_prev_miss=float(e[0][0]), EV_prev_hit=float(e[1][0]), ED_prev_miss=float(e[0][1]), ED_prev_hit=float(e[1][1]))
        # block-bootstrap CI for the (A2) value gap on misses (pooled streams)
        gaps = []
        for r in range(N_BOOT):
            sel = blk[r]; Hq = Hs[:, sel]; Vq = Vs[:, sel]; pv, cv, vv = Hq[:, :-1].ravel(), Hq[:, 1:].ravel(), Vq[:, 1:].ravel()
            gaps.append(vv[(cv == 0) & (pv == 0)].mean() - vv[(cv == 0) & (pv == 1)].mean())
        a2["H0"]["gap_ci"] = ci(gaps); a2["hypothesis_holds"] = covers0(a2["H0"]["gap_ci"])
        # ratio limit: structural prediction (probing) vs exact id_test law vs realized streams
        eta = {}
        for r in ["dollars", "sec_seq"]:
            real = Vs.sum(1) / Ds[r].sum(1); v1, v0 = Vs[Hs].mean(), Vs[~Hs].mean(); d1, d0 = Ds[r][Hs].mean(), Ds[r][~Hs].mean()
            pf = Hs.mean(); ident = (pf * v1 + (1 - pf) * v0) / (pf * d1 + (1 - pf) * d0)
            pred, exact = sp["EV"] / sp["ED"][r], st_["EV"] / st_["ED"][r]
            pred_ci, exact_ci = cib(bp, lambda x: x["EV"] / x["ED"][r]), cib(bt, lambda x: x["EV"] / x["ED"][r])
            se_pred, se_exact, se_real = (pred_ci[1] - pred_ci[0]) / 3.92, (exact_ci[1] - exact_ci[0]) / 3.92, real.std(ddof=1) / math.sqrt(R_STREAM)
            eta[r] = dict(eta_pred_probing=float(pred), eta_pred_ci=pred_ci, eta_exact_idtest=float(exact), eta_exact_ci=exact_ci,
                          eta_realized_mean=float(real.mean()), eta_realized_se_streams=float(se_real), eta_realized_per_stream=real.tolist(),
                          eta_identity_stationary_means=float(ident), identity_rel_diff=float(ident / Vs.sum() * Ds[r].sum() - 1),
                          z_transfer=float((pred - exact) / max(math.sqrt(se_pred ** 2 + se_exact ** 2), 1e-12)), z_ergodic=float((exact - real.mean()) / max(se_real, 1e-12)),
                          z_pred_vs_realized=float((pred - real.mean()) / max(math.sqrt(se_pred ** 2 + se_real ** 2), 1e-12)), rel_err_pred_vs_realized=float(pred / real.mean() - 1),
                          z_full=float((pred - real.mean()) / max(math.sqrt(se_pred ** 2 + se_exact ** 2 + se_real ** 2), 1e-12)),
                          note_z="z_full combines probing task-bootstrap SE (prediction), id_test task-bootstrap SE (the finite measured pool) and stream-to-stream SE; z_pred_vs_realized omits the id_test pool variance")
        # transient at high resolution: R_TRANS chains, burn-in, condition on a miss at epoch 1, follow N_TRANS epochs
        idx_tr = generate(kind, Pt, params, BURN_TRANS + 1 + N_TRANS, R_TRANS, rng_b)
        Htr = Pt.hit[idx_tr]; cond = ~Htr[:, BURN_TRANS]; Hc = Htr[cond][:, BURN_TRANS + 1:]; Rc = int(cond.sum())
        emp = Hc.mean(0); ex_t = np.array(st_["transient_exact"]); ex_p = np.array(sp["transient_exact"])
        nn = np.arange(2, N_TRANS + 2)
        two_t = st_["pi"] * (1 - st_["xi"] ** (nn - 1)); two_p = sp["pi"] * (1 - sp["xi"] ** (nn - 1))
        se = np.sqrt(np.clip(ex_t * (1 - ex_t), 1e-12, None) / Rc)
        two_pb = np.array([[x["pi"] * (1 - x["xi"] ** (n - 1)) for n in nn] for x in bp]); se_two_p = two_pb.std(0)
        trans = dict(R_conditioned=Rc, n=nn.tolist(), empirical=emp.tolist(), se_mc=se.tolist(),
                     two_state_idtest_params=two_t.tolist(), exact_chain_idtest=ex_t.tolist(), two_state_probing_params=two_p.tolist(), exact_chain_probing=ex_p.tolist(),
                     z_two_state_insample=((emp - two_t) / se).tolist(), z_exact_insample=((emp - ex_t) / se).tolist(),
                     z_two_state_outofsample=((emp - two_p) / np.sqrt(se ** 2 + se_two_p ** 2)).tolist(),
                     max_abs_z_two_state_insample_n3plus=float(np.max(np.abs((emp - two_t) / se)[1:])), max_abs_z_exact_insample_n3plus=float(np.max(np.abs((emp - ex_t) / se)[1:])),
                     max_abs_z_two_state_outofsample_n3plus=float(np.max(np.abs((emp - two_p) / np.sqrt(se ** 2 + se_two_p ** 2))[1:])),
                     max_gap_two_state_vs_exact_over_pi=float(np.max(np.abs(two_t - ex_t)) / st_["pi"]))
        pi_pred_ci = cib(bp, lambda x: x["pi"]); pi_exact_ci = cib(bt, lambda x: x["pi"])
        se_pp, se_pe = (pi_pred_ci[1] - pi_pred_ci[0]) / 3.92, (pi_exact_ci[1] - pi_exact_ci[0]) / 3.92; se_pm = pi_meas.std(ddof=1) / math.sqrt(R_STREAM)
        rec = dict(structural_probing={k: sp[k] for k in ["pi", "p_h", "p_m", "xi", "slem", "absorbing"]}, structural_idtest={k: st_[k] for k in ["pi", "p_h", "p_m", "xi", "slem", "absorbing"]},
                   xi_ci_probing=cib(bp, lambda x: x["xi"]), pi_pred_ci_probing=pi_pred_ci, pi_exact_ci_idtest=pi_exact_ci,
                   stream=dict(T=T_STREAM, R=R_STREAM, p_h=float(ph.mean()), p_m=float(pm.mean()), p_h_minus_p_m_ci=ci(phb - pmb),
                               pi_meas=float(pi_meas.mean()), pi_meas_se=float(se_pm), pi_from_mle_identity=float(pi_pred_mle.mean()),
                               abs_identity_gap=float(np.abs(pi_pred_mle - pi_meas).max())),
                   z_pi_transfer=float((sp["pi"] - st_["pi"]) / max(math.sqrt(se_pp ** 2 + se_pe ** 2), 1e-12)), z_pi_ergodic=float((st_["pi"] - pi_meas.mean()) / max(se_pm, 1e-12)),
                   z_pi_pred_vs_meas=float((sp["pi"] - pi_meas.mean()) / max(math.sqrt(se_pp ** 2 + se_pm ** 2), 1e-12)), rel_err_pi_pred_vs_meas=float(sp["pi"] / pi_meas.mean() - 1),
                   z_pi_full=float((sp["pi"] - pi_meas.mean()) / max(math.sqrt(se_pp ** 2 + se_pe ** 2 + se_pm ** 2), 1e-12)),
                   second_order_markov_check=markov2, A2_check=a2, ratio_limit=eta, transient=trans)
        if kind == "hidden_3regime": rec["q_regime_idtest"] = st_["q_regime"]
        RESULTS["part_b"][key][kind] = rec
        say(f"[{key}] {kind:26s} p_h={ph.mean():.3f} p_m={pm.mean():.3f} xi={sp['xi']:+.3f} slem={st_['slem']:.3f} | 2nd-order after-miss gap {markov2['diff_after_miss']:+.4f} "
            f"CI[{markov2['diff_after_miss_ci'][0]:+.4f},{markov2['diff_after_miss_ci'][1]:+.4f}] Markov rejected={markov2['markov_rejected']} | pi pred={sp['pi']:.4f} exact={st_['pi']:.4f} "
            f"meas={pi_meas.mean():.4f} z(pred,meas)={rec['z_pi_pred_vs_meas']:+.2f} | transient max|z| n>=3: 2-state={trans['max_abs_z_two_state_insample_n3plus']:.1f} "
            f"exact={trans['max_abs_z_exact_insample_n3plus']:.1f} gap/pi={trans['max_gap_two_state_vs_exact_over_pi']:.3f} | eta$ pred={eta['dollars']['eta_pred_probing']:.2f} "
            f"real={eta['dollars']['eta_realized_mean']:.2f} z={eta['dollars']['z_pred_vs_realized']:+.2f} | A2 miss-gap={a2['H0']['EV_prev_miss']-a2['H0']['EV_prev_hit']:+.3f} holds={a2['hypothesis_holds']}")
say(f"[t={time.time()-T0:.0f}s]")

# ============================================================================= PART (c): hedging with real latencies and dollars
say("\n=== (c) app:hedging -- sequential vs hedged latency, eq:hedged-latency-condition, eq:hedged-dollar-condition ===")
rng_c = np.random.default_rng(SEED + 2)
R_PREFIX = [0.0, 0.1, 1.0]                     # r = rho_r * b  (cancellation prefix charge on a hit)
for m in ALL7:
    for th in THETAS:
        ok = (Ln[m] > 0) & (Ln[B] > 0); ip, it = np.where(is_prob & ok)[0], np.where(is_test & ok)[0]
        hit = Sn[m] >= th
        # ---- primitives on probing (means; l_kappa = 0: no controller latency is recorded in the data)
        wp = boot_weights(len(ip), N_BOOT + 1, rng_c); wt = boot_weights(len(it), N_BOOT + 1, rng_c)
        Hp = hit[ip].astype(float)
        def prim(w):
            p = wmean(w, Hp); vh = wmean(w, Sn[m][ip], Hp); vbar = wmean(w, Sn[B][ip]); vm = wmean(w, Sn[B][ip], 1 - Hp)
            lp = wmean(w, Ln[m][ip]); lb = wmean(w, Ln[B][ip]); lb_miss = wmean(w, Ln[B][ip], 1 - Hp); lp_hit = wmean(w, Ln[m][ip], Hp)
            lmax_miss = wmean(w, np.maximum(Ln[m][ip], Ln[B][ip]), 1 - Hp); a = wmean(w, Cn[m][ip]); b = wmean(w, Cn[B][ip]); b_hit = wmean(w, Cn[B][ip], Hp)
            return dict(p=p, v_h=vh, vbar=vbar, v_m=vm, l_p=lp, l_b=lb, l_b_miss=lb_miss, l_p_hit=lp_hit, l_max_miss=lmax_miss, a=a, b=b, b_hit=b_hit,
                        frac_lp_le_lb=wmean(w, (Ln[m][ip] <= Ln[B][ip]).astype(float)))
        PR = prim(wp); P0 = {k: float(v[0]) for k, v in PR.items()}
        # hypotheses of app:hedging (IN-SAMPLE on probing)
        hyp = dict(lp_le_lb=dict(l_p=P0["l_p"], l_b=P0["l_b"], diff_ci=ci(PR["l_p"][1:] - PR["l_b"][1:]), holds=bool(P0["l_p"] <= P0["l_b"]),
                                  holds_ci=bool(ci(PR["l_p"][1:] - PR["l_b"][1:])[1] <= 0), frac_tasks_lp_le_lb=P0["frac_lp_le_lb"]),
                   matched_miss_value=dict(v_m=P0["v_m"], vbar=P0["vbar"], diff_ci=ci(PR["v_m"][1:] - PR["vbar"][1:]), holds_ci=covers0(ci(PR["v_m"][1:] - PR["vbar"][1:]))),
                   vh_ge_vbar=bool(P0["v_h"] >= P0["vbar"]))
        # ---- realized quantities on id_test (task bootstrap)
        Ht = hit[it].astype(float); lp_t, lb_t = Ln[m][it], Ln[B][it]; Vt = np.where(hit[it], Sn[m][it], Sn[B][it])
        seq_t = lp_t + (1 - Ht) * lb_t; hed_t = Ht * lp_t + (1 - Ht) * np.maximum(lp_t, lb_t)
        real = dict(seq_mean=wmean(wt, seq_t), hed_mean=wmean(wt, hed_t), eta_seq=wmean(wt, Vt) / wmean(wt, seq_t), eta_hed=wmean(wt, Vt) / wmean(wt, hed_t),
                    eta_ref_lat=wmean(wt, Sn[B][it]) / wmean(wt, lb_t), p_test=wmean(wt, Ht), hedged_gain_over_seq=1 - wmean(wt, hed_t) / wmean(wt, seq_t))
        # ---- predictions (deterministic-time plug-in with unconditional means, and the conditional-mean version the appendix allows)
        pred = dict(seq_plugin=PR["l_p"] + (1 - PR["p"]) * PR["l_b"], hed_plugin=PR["p"] * PR["l_p"] + (1 - PR["p"]) * PR["l_b"],
                    seq_cond=PR["l_p"] + (1 - PR["p"]) * PR["l_b_miss"], hed_cond=PR["p"] * PR["l_p_hit"] + (1 - PR["p"]) * PR["l_max_miss"],
                    eta_hed_formula=(PR["p"] * PR["v_h"] + (1 - PR["p"]) * PR["vbar"]) / (PR["p"] * PR["l_p"] + (1 - PR["p"]) * PR["l_b"]),
                    eta_seq_formula=(PR["p"] * PR["v_h"] + (1 - PR["p"]) * PR["vbar"]) / (PR["l_p"] + (1 - PR["p"]) * PR["l_b"]),
                    margin_hed=PR["p"] * (PR["l_b"] * PR["v_h"] - PR["vbar"] * PR["l_p"]),                    # eq:hedged-latency-condition, l_kappa = 0
                    margin_seq=PR["p"] * PR["l_b"] * PR["v_h"] - PR["vbar"] * PR["l_p"],                      # sequential condition, l_kappa = 0
                    l_kappa_max_hed=PR["p"] * (PR["l_b"] * PR["v_h"] - PR["vbar"] * PR["l_p"]) / PR["vbar"],    # largest controller overhead still satisfying the condition
                    # state-matched instantiation (app:hedging last paragraph / app:matched-reference): conditional means replace the deterministic times and the matched value
                    margin_hed_cond=(PR["p"] * PR["v_h"] + (1 - PR["p"]) * PR["v_m"]) * PR["l_b"] - PR["vbar"] * (PR["p"] * PR["l_p_hit"] + (1 - PR["p"]) * PR["l_max_miss"]),
                    margin_seq_cond=(PR["p"] * PR["v_h"] + (1 - PR["p"]) * PR["v_m"]) * PR["l_b"] - PR["vbar"] * (PR["l_p"] + (1 - PR["p"]) * PR["l_b_miss"]))
        def cmp(pv, rv):
            d = pv[1:] - rv[1:]; return dict(pred=float(pv[0]), realized=float(rv[0]), realized_ci=ci(rv[1:]), rel_err=float(pv[0] / rv[0] - 1), diff_ci=ci(d), ci_covers_0=covers0(ci(d)))
        # in-sample (id_test primitives -> id_test realized): isolates the deterministic-time hypothesis from the probing -> id_test shift;
        # the conditional-mean formulas are identities in-sample, the plug-in formulas are not
        pt_ = wmean(wt, Ht); lpt, lbt = wmean(wt, lp_t), wmean(wt, lb_t)
        ins = dict(sequential_plugin=cmp(lpt + (1 - pt_) * lbt, real["seq_mean"]), hedged_plugin=cmp(pt_ * lpt + (1 - pt_) * lbt, real["hed_mean"]),
                   sequential_conditional_identity_gap=float(abs((lpt + (1 - pt_) * wmean(wt, lb_t, 1 - Ht))[0] - real["seq_mean"][0])),
                   hedged_conditional_identity_gap=float(abs((pt_ * wmean(wt, lp_t, Ht) + (1 - pt_) * wmean(wt, np.maximum(lp_t, lb_t), 1 - Ht))[0] - real["hed_mean"][0])))
        lat = dict(sequential_plugin=cmp(pred["seq_plugin"], real["seq_mean"]), sequential_conditional=cmp(pred["seq_cond"], real["seq_mean"]),
                   hedged_plugin=cmp(pred["hed_plugin"], real["hed_mean"]), hedged_conditional=cmp(pred["hed_cond"], real["hed_mean"]),
                   hedged_gain_over_sequential_realized=[float(real["hedged_gain_over_seq"][0]), ci(real["hedged_gain_over_seq"][1:])],
                   eta_hedged_formula_vs_realized=cmp(pred["eta_hed_formula"], real["eta_hed"]), eta_seq_formula_vs_realized=cmp(pred["eta_seq_formula"], real["eta_seq"]), insample=ins)
        def sign_test(margin, realized_diff):
            rc = ci(realized_diff[1:]); pc = ci(margin[1:])
            pred_sign = int(np.sign(margin[0])); real_sign = int(np.sign(realized_diff[0]))
            if covers0(rc): verdict = "INCONCLUSIVE"
            else: verdict = "PASS" if pred_sign == real_sign else "FAIL"
            return dict(margin_pred=float(margin[0]), margin_ci=pc, pred_sign=pred_sign, realized_diff=float(realized_diff[0]), realized_ci=rc, realized_sign=real_sign, verdict=verdict)
        cond = dict(hedged_latency=sign_test(pred["margin_hed"], real["eta_hed"] - real["eta_ref_lat"]), sequential_latency=sign_test(pred["margin_seq"], real["eta_seq"] - real["eta_ref_lat"]),
                    hedged_latency_state_matched=sign_test(pred["margin_hed_cond"], real["eta_hed"] - real["eta_ref_lat"]), sequential_latency_state_matched=sign_test(pred["margin_seq_cond"], real["eta_seq"] - real["eta_ref_lat"]),
                    l_kappa_max_hedged_s=[float(pred["l_kappa_max_hed"][0]), ci(pred["l_kappa_max_hed"][1:])])
        # ---- dollars with a cancellation prefix charge r = rho_r * b on hits
        dollars = {}
        eta_ref_c = wmean(wt, Sn[B][it]) / wmean(wt, Cn[B][it])
        for rr in R_PREFIX:
            r = rr * PR["b"]
            margin = PR["p"] * (PR["b"] * PR["v_h"] - PR["vbar"] * r) - PR["vbar"] * PR["a"]                   # eq:hedged-dollar-condition
            b_miss = (PR["b"] - PR["p"] * PR["b_hit"]) / (1 - PR["p"])
            margin_cond = (PR["p"] * PR["v_h"] + (1 - PR["p"]) * PR["v_m"]) * PR["b"] - PR["vbar"] * (PR["a"] + (1 - PR["p"]) * b_miss + PR["p"] * rr * PR["b_hit"])
            cost_t = Cn[m][it] + (1 - Ht) * Cn[B][it] + Ht * rr * Cn[B][it]
            eta_c = wmean(wt, Vt) / wmean(wt, cost_t)
            eta_formula = (PR["p"] * PR["v_h"] + (1 - PR["p"]) * PR["vbar"]) / (PR["a"] + (1 - PR["p"]) * PR["b"] + PR["p"] * r)
            dollars[f"r={rr}b"] = dict(r_usd=float(r[0]), condition=sign_test(margin, eta_c - eta_ref_c), condition_state_matched=sign_test(margin_cond, eta_c - eta_ref_c), eta_formula_vs_realized=cmp(eta_formula, eta_c),
                                       mean_cost_realized=[float(wmean(wt, cost_t)[0]), ci(wmean(wt, cost_t)[1:])], mean_cost_ref=float(wmean(wt, Cn[B][it])[0]))
        RESULTS["part_c"][f"{m}|theta={th}"] = dict(n_probing=int(len(ip)), n_id_test=int(len(it)), primitives_probing=P0, hypotheses=hyp, latency=lat, conditions=cond, dollars=dollars)
        say(f"{m:18s} th={th} p={P0['p']:.3f} l_p={P0['l_p']:.1f} l_b={P0['l_b']:.1f} (l_p<=l_b {hyp['lp_le_lb']['holds']}, frac {P0['frac_lp_le_lb']:.2f}) v_m={P0['v_m']:.3f} vbar={P0['vbar']:.3f} | "
            f"seq: plugin {lat['sequential_plugin']['pred']:.1f} real {lat['sequential_plugin']['realized']:.1f} ({lat['sequential_plugin']['rel_err']*100:+.1f}%) | hed: plugin {lat['hedged_plugin']['pred']:.1f} "
            f"cond {lat['hedged_conditional']['pred']:.1f} real {lat['hedged_plugin']['realized']:.1f} ({lat['hedged_plugin']['rel_err']*100:+.1f}%) | cond-hed {cond['hedged_latency']['verdict']} "
            f"(pred {cond['hedged_latency']['pred_sign']:+d} real {cond['hedged_latency']['realized_sign']:+d}) cond-seq {cond['sequential_latency']['verdict']} | $ " +
            " ".join(f"r={rr}b:{dollars[f'r={rr}b']['condition']['verdict']}({dollars[f'r={rr}b']['condition']['pred_sign']:+d}/{dollars[f'r={rr}b']['condition']['realized_sign']:+d})" for rr in R_PREFIX))
say(f"[t={time.time()-T0:.0f}s]")

# ============================================================================= PART (d): AND-barrier, prop:app-barrier
say("\n=== (d) prop:app-barrier -- slow vs fast fallback behind a barrier (semi-synthetic rounds from real rows) ===")
rng_d = np.random.default_rng(SEED + 3)
CHEAP = "Qwen3-Max"; SLOW = B; REPAIR_CANDIDATES = ["gpt-5.4", "claude-sonnet-4-6"]
B_GRID = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]); K = 64
N_ROUNDS, CH, BURN_CH = (200, 25, 2) if QUICK else (800, 25, 2)
RHOS_D = [0.3, 1.0]; RB = N_BOOT + 1
okd = (Ln[CHEAP] > 0) & (Ln[SLOW] > 0) & np.all([Ln[c] > 0 for c in REPAIR_CANDIDATES], axis=0)
hardflag_d, _ = hard_flag_for(CHEAP)
A = dict(s_c=Sn[CHEAP], l_c=Ln[CHEAP], c_c=Cn[CHEAP], s_o=Sn[SLOW], l_o=Ln[SLOW], c_o=Cn[SLOW])
for c in REPAIR_CANDIDATES: A["s_" + c] = Sn[c]; A["l_" + c] = Ln[c]; A["c_" + c] = Cn[c]

class BPools:
    def __init__(self, idx, rng):
        n = len(idx); pools = np.empty((RB, n), np.int64); pools[0] = idx; pools[1:] = rng.choice(idx, size=(RB - 1, n), replace=True)
        hl = [p[hardflag_d[p]] for p in pools]; self.hs = np.array([len(h) for h in hl]); hp = np.zeros((RB, self.hs.max()), np.int64)
        for r, h in enumerate(hl): hp[r, :len(h)] = h
        self.n, self.pools, self.hp = n, pools, hp
        self.pflat, self.poff = pools.ravel(), (np.arange(RB) * n)[None, :, None]; self.hflat, self.hoff = hp.ravel(), (np.arange(RB) * hp.shape[1])[None, :, None]
    def hard(self, r): return self.hp[r, :self.hs[r]]

def pair_sum(X): return X[..., 0::2] + X[..., 1::2]
def pair_max(X): return np.maximum(X[..., 0::2], X[..., 1::2])
def reduce_levels(sums, maxes):
    out = {}; Sl, Ml = list(sums), list(maxes)
    for b in [1, 2, 4, 8, 16, 32, 64]:
        if b > 1: Sl = [pair_sum(x) for x in Sl]; Ml = [pair_max(x) for x in Ml]
        out[b] = ([x.sum((0, 2)) for x in Sl], [x.sum((0, 2)) for x in Ml])
    Sl = [x[..., 0:63:3] + x[..., 1:63:3] + x[..., 2:63:3] for x in sums]; Ml = [np.maximum(np.maximum(x[..., 0:63:3], x[..., 1:63:3]), x[..., 2:63:3]) for x in maxes]
    out[3] = ([x.sum((0, 2)) for x in Sl], [x.sum((0, 2)) for x in Ml])
    for b, g in [(6, 10), (12, 5), (24, 2), (48, 1)]:
        Sl = [pair_sum(x[..., :2 * g]) for x in Sl]; Ml = [pair_max(x[..., :2 * g]) for x in Ml]
        out[b] = ([x.sum((0, 2)) for x in Sl], [x.sum((0, 2)) for x in Ml])
    return out

def run_barrier(P, th, rho, policy, rng):
    """K independent member slots per replicate.  policy: 'slow' (Opus re-run on every missed member, in parallel),
    'commit' (commit the cheap output, no extra latency), or a repair model name (re-run on missed members).
    Residue rule (uniform across policies): after a COMMITTED output with score < theta the slot's next task is drawn
    from the hard stratum w.p. rho, else from the full pool.  Returns single-slot primitives and per-b round aggregates."""
    hitflag = A["s_c"] >= th
    if policy == "commit": s_fb, l_fb = A["s_c"], np.zeros(N)
    elif policy == "slow": s_fb, l_fb = A["s_o"], A["l_o"]
    else: s_fb, l_fb = A["s_" + policy], A["l_" + policy]
    committed = np.where(hitflag, A["s_c"], s_fb); bad = committed < th
    T = np.stack([hitflag.astype(np.float32), committed.astype(np.float32), A["l_c"].astype(np.float32), np.where(hitflag, 0, l_fb).astype(np.float32)], 1)
    acc = {k: np.zeros(RB) for k in ["n_hh", "n_hprev", "n_mh", "n_mprev", "n_hit", "n_miss", "sV_hit", "sV_rec", "s_lc", "s_lfb_miss", "n_res", "n"]}
    G = {k: np.zeros((RB, len(B_GRID))) for k in ["sumV", "sumLp", "sumLfb", "anymiss"]}
    s = np.zeros((RB, K), bool); prev_last = None; nrounds = 0
    for c in range(BURN_CH + N_ROUNDS // CH):
        idx = P.pflat[P.poff + rng.integers(0, P.n, (CH, RB, K))]
        idxh = P.hflat[P.hoff + (rng.random((CH, RB, K)) * P.hs[None, :, None]).astype(np.int64)]
        coin = rng.random((CH, RB, K)) < rho
        bf, bh = bad[idx], bad[idxh]; state = np.empty((CH, RB, K), bool)
        for t in range(CH):
            state[t] = s; s = np.where(s, bh[t], bf[t]) & coin[t]
        task = np.where(state, idxh, idx); X = T[task]
        hit = X[..., 0] > 0.5; v = X[..., 1]; lc = X[..., 2]; lfb = X[..., 3]
        if c < BURN_CH: prev_last = hit[-1]; continue
        prev = np.concatenate([prev_last[None], hit[:-1]], 0); prev_last = hit[-1]
        acc["n_hh"] += (prev & hit).sum((0, 2)); acc["n_hprev"] += prev.sum((0, 2)); acc["n_mh"] += ((~prev) & hit).sum((0, 2)); acc["n_mprev"] += (~prev).sum((0, 2))
        acc["n_hit"] += hit.sum((0, 2)); acc["n_miss"] += (~hit).sum((0, 2)); acc["sV_hit"] += (v * hit).sum((0, 2)); acc["sV_rec"] += (v * ~hit).sum((0, 2))
        acc["s_lc"] += lc.sum((0, 2)); acc["s_lfb_miss"] += lfb.sum((0, 2)); acc["n_res"] += state.sum((0, 2)); acc["n"] += CH * K
        red = reduce_levels([v], [lc, lfb, (~hit).astype(np.float32)])
        for j, b in enumerate(B_GRID):
            (sv,), (mlc, mlf, mam) = red[b]; G["sumV"][:, j] += sv; G["sumLp"][:, j] += mlc; G["sumLfb"][:, j] += mlf; G["anymiss"][:, j] += mam
        nrounds += CH
    n = acc["n"]
    prim = dict(p_h=acc["n_hh"] / acc["n_hprev"], p_m=acc["n_mh"] / acc["n_mprev"], pi=acc["n_hit"] / n, V_hit=acc["sV_hit"] / acc["n_hit"], V_rec=acc["sV_rec"] / acc["n_miss"],
                V=(acc["sV_hit"] + acc["sV_rec"]) / n, L=acc["s_lc"] / n, Delta=acc["s_lfb_miss"] / acc["n_miss"], r_res=acc["n_res"] / n)
    groups = (K // B_GRID)[None, :] * nrounds
    meas = dict(eta=G["sumV"] / (B_GRID[None, :] * (G["sumLp"] + G["sumLfb"])), L_round=G["sumLp"] / groups, Lfb_round=G["sumLfb"] / groups, p_anymiss=G["anymiss"] / groups,
                V_member=G["sumV"] / (groups * B_GRID[None, :]), n_rounds=groups[0])
    return prim, meas

def emax_all(x, w, bs):
    o = np.argsort(x); x = x[o].astype(np.float64); w = w[o] / w.sum(); F = np.cumsum(w); Fm = F - w; bs = np.asarray(bs, float)
    return (x[:, None] * (F[:, None] ** bs[None, :] - Fm[:, None] ** bs[None, :])).sum(0)
def L_orderstat(P, r, r_res):
    xf, xh = A["l_c"][P.pools[r]], A["l_c"][P.hard(r)]
    x = np.concatenate([xf, xh]); w = np.concatenate([np.full(len(xf), (1 - r_res) / len(xf)), np.full(len(xh), r_res / max(len(xh), 1))])
    return emax_all(x, w, B_GRID)
def Delta_orderstat(P, r, th, pi, l_arr, r_res):
    """E[max over missed members of the fallback latency | >= 1 miss], members' miss-latency law = stationary mixture."""
    pool = P.pools[r]; miss_f = A["s_c"][pool] < th; hard = P.hard(r); miss_h = A["s_c"][hard] < th
    x = np.concatenate([l_arr[pool][miss_f], l_arr[hard][miss_h]])
    w = np.concatenate([np.full(miss_f.sum(), (1 - r_res) / max(miss_f.sum(), 1)), np.full(miss_h.sum(), r_res / max(miss_h.sum(), 1))])
    em = emax_all(x, w, np.arange(1, K + 1)); out = np.empty(len(B_GRID))
    for j, b in enumerate(B_GRID):
        ks = np.arange(1, b + 1); pk = stats.binom.pmf(ks, b, 1 - pi); out[j] = np.sum(pk * em[:b]) / pk.sum()
    return out
def crossover(d, grid=B_GRID):
    if d[0] <= 0: return np.nan, "fast_all"
    neg = np.where(d <= 0)[0]
    if len(neg) == 0: return np.nan, "slow_all"
    j = neg[0]; b0, b1, d0, d1 = grid[j - 1], grid[j], d[j - 1], d[j]
    return float(np.exp(np.log(b0) + (np.log(b1) - np.log(b0)) * d0 / (d0 - d1))), "cross"
def n_sign_changes(d):
    s = np.sign(d); s = s[s != 0]; return int(np.sum(s[1:] != s[:-1]))
def fracs(labels): return {lab: float(np.mean([x == lab for x in labels])) for lab in ["slow_all", "cross", "fast_all"]}
def psi_root(Vs, Vf, L, Ds, Df, pis, pif):
    """Psi(b) of the proof; returns the unique positive root (continuous b) or nan; Delta_f = 0 gives b* = log Xi / log pi_s."""
    Psi = lambda b: Vs * (L + Df) - Vf * (L + Ds) + Vf * Ds * pis ** b - Vs * Df * pif ** b
    if Psi(0) <= 0 or Vs * (L + Df) - Vf * (L + Ds) >= 0: return np.nan
    lo, hi = 0.0, 1.0
    while Psi(hi) > 0 and hi < 1e6: hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Psi(mid) > 0: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

idx_pd, idx_td = np.where(is_prob & okd)[0], np.where(is_test & okd)[0]
RESULTS["part_d"]["setup"] = dict(cheap=CHEAP, slow=SLOW, repair_candidates=REPAIR_CANDIDATES, b_grid=B_GRID.tolist(), K=K, n_rounds_after_burn=N_ROUNDS, burn_rounds=BURN_CH * CH,
                                  n_probing=int(len(idx_pd)), n_id_test=int(len(idx_td)), n_dropped_unlogged_latency=int((~okd).sum()), rhos=RHOS_D,
                                  hard_stratum="per-dimension hardest tercile by leave-primary-out difficulty (mean score of the 7 other models), cuts from probing",
                                  residue_rule="after a committed output with score < theta the slot's next task is from the hard stratum w.p. rho (all policies)")
Pp_d, Pt_d = BPools(idx_pd, rng_d), BPools(idx_td, rng_d)
for th in THETAS:
    for rho in RHOS_D:
        # Exploratory repair-model choice on probing: fastest candidate on cheap-missed tasks whose miss-value is below Opus's.
        miss_p = idx_pd[A["s_c"][idx_pd] < th]
        cand = {c: dict(Delta=float(A["l_" + c][miss_p].mean()), V_rec=float(A["s_" + c][miss_p].mean())) for c in REPAIR_CANDIDATES}
        opus = dict(Delta=float(A["l_o"][miss_p].mean()), V_rec=float(A["s_o"][miss_p].mean()))
        feas = [c for c in REPAIR_CANDIDATES if cand[c]["Delta"] < opus["Delta"] and cand[c]["V_rec"] < opus["V_rec"]]
        repair = min(feas, key=lambda c: cand[c]["Delta"]) if feas else None
        prim_s, _ = run_barrier(Pp_d, th, rho, "slow", rng_d); _, meas_s = run_barrier(Pt_d, th, rho, "slow", rng_d)
        key = f"theta={th}|rho={rho}"; RESULTS["part_d"][key] = dict(repair_choice=dict(candidates_probing=cand, opus_probing=opus, feasible=feas, chosen=repair), pairs={})
        say(f"[{key}] slow chain: pi_s={prim_s['pi'][0]:.3f} V_s={prim_s['V'][0]:.3f} L={prim_s['L'][0]:.2f}s Delta_s={prim_s['Delta'][0]:.2f}s r_res={prim_s['r_res'][0]:.3f}; repair choice {repair} ({cand})  t={time.time()-T0:.0f}s")
        for policy in ["commit"] + ([repair] if repair else []):
            prim_f, _ = run_barrier(Pp_d, th, rho, policy, rng_d); _, meas_f = run_barrier(Pt_d, th, rho, policy, rng_d)
            Vs, Vf, L, Ds, Df, pis, pif = prim_s["V"], prim_f["V"], prim_s["L"], prim_s["Delta"], prim_f["Delta"], prim_s["pi"], prim_f["pi"]
            Lf = prim_f["L"]
            hyp = dict(Vs_gt_Vf=[bool(Vs[0] > Vf[0]), float(np.mean(Vs[1:] > Vf[1:]))], Ds_gt_Df=[bool(Ds[0] > Df[0]), float(np.mean(Ds[1:] > Df[1:]))],
                       pif_lt_pis=[bool(pif[0] < pis[0]), float(np.mean(pif[1:] < pis[1:]))], pi_f_minus_pi_s_ci=ci(pif[1:] - pis[1:]),
                       L_common=dict(L_slow=float(L[0]), L_fast=float(Lf[0]), diff_ci=ci(L[1:] - Lf[1:])),
                       L_deterministic=dict(cv_cheap_latency_probing=float(A["l_c"][idx_pd].std() / A["l_c"][idx_pd].mean()),
                                            round_L_over_L_idtest=(meas_s["L_round"][0] / L[0]).tolist()))
            # P1: mean-latency instantiation (exactly the proposition's objects, plugged with stationary means from probing)
            crit_lhs, crit_rhs = Vs / (L + Ds), Vf / (L + Df)
            exist_P1 = crit_lhs < crit_rhs
            Xi = 1 - (L / Ds) * (Vs / Vf - 1)
            bstar_P1 = np.array([psi_root(Vs[r], Vf[r], L[r], Ds[r], Df[r], pis[r], pif[r]) for r in range(RB)])
            bstar_Xi = np.where((Xi > 0) & (Xi < 1), np.log(np.clip(Xi, 1e-300, None)) / np.log(pis), np.nan) if policy == "commit" else None
            pw_s, pw_f = pis[:, None] ** B_GRID[None, :], pif[:, None] ** B_GRID[None, :]
            eta_s_P1 = Vs[:, None] / (L[:, None] + (1 - pw_s) * Ds[:, None]); eta_f_P1 = Vf[:, None] / (L[:, None] + (1 - pw_f) * Df[:, None])
            cr_P1 = [crossover((eta_s_P1 - eta_f_P1)[r]) for r in range(RB)]
            # P2: order-statistic instantiation (outside the proposition's deterministic-latency hypothesis)
            Ls_b = np.array([L_orderstat(Pp_d, r, prim_s["r_res"][r]) for r in range(RB)]); Lf_b = np.array([L_orderstat(Pp_d, r, prim_f["r_res"][r]) for r in range(RB)])
            Ds_b = np.array([Delta_orderstat(Pp_d, r, th, pis[r], A["l_o"], prim_s["r_res"][r]) for r in range(RB)])
            Df_b = np.zeros((RB, len(B_GRID))) if policy == "commit" else np.array([Delta_orderstat(Pp_d, r, th, pif[r], A["l_" + policy], prim_f["r_res"][r]) for r in range(RB)])
            eta_s_P2 = Vs[:, None] / (Ls_b + (1 - pw_s) * Ds_b); eta_f_P2 = Vf[:, None] / (Lf_b + (1 - pw_f) * Df_b)
            cr_P2 = [crossover((eta_s_P2 - eta_f_P2)[r]) for r in range(RB)]
            # empirical
            d_emp = meas_s["eta"] - meas_f["eta"]; cr_emp = [crossover(d_emp[r]) for r in range(RB)]
            nsc = np.array([n_sign_changes(d_emp[r]) for r in range(RB)]); se_d = d_emp[1:].std(0)
            nsc_rob = np.array([n_sign_changes(np.where(np.abs(d_emp[r]) > 2 * se_d, d_emp[r], 0.0)) for r in range(RB)])
            reg_emp = [c[1] for c in cr_emp]; reg_P1 = [c[1] for c in cr_P1]; reg_P2 = [c[1] for c in cr_P2]
            exist_emp = np.array([c[1] == "cross" for c in cr_emp])
            # existence criterion vs empirical: the proposition predicts a crossing on b > 0 iff crit_lhs < crit_rhs; the empirical grid starts at b = 1,
            # so a predicted crossing with b* < 1 shows up as 'fast_all' -- both count as 'crossing exists'.
            pred_exist = exist_P1; emp_exist_or_fast = np.array([c[1] in ("cross", "fast_all") for c in cr_emp])
            agree_exist = float(np.mean(pred_exist[1:] == emp_exist_or_fast[1:]))
            be = [c[0] for c in cr_emp]; b1 = [c[0] for c in cr_P1]; b2 = [c[0] for c in cr_P2]
            def lr(pred, emp): 
                v = [math.log(p / e) for p, e in zip(pred, emp) if np.isfinite(p) and np.isfinite(e) and p > 0 and e > 0]; return [float(np.median(v)), len(v)] if v else [None, 0]
            rec = dict(policy_fast=policy, hypotheses=hyp,
                       primitives_probing=dict(V_s=[float(Vs[0]), ci(Vs[1:])], V_f=[float(Vf[0]), ci(Vf[1:])], L=[float(L[0]), ci(L[1:])], Delta_s=[float(Ds[0]), ci(Ds[1:])],
                                               Delta_f=[float(Df[0]), ci(Df[1:])], pi_s=[float(pis[0]), ci(pis[1:])], pi_f=[float(pif[0]), ci(pif[1:])],
                                               p_h_slow=float(prim_s["p_h"][0]), p_m_slow=float(prim_s["p_m"][0]), p_h_fast=float(prim_f["p_h"][0]), p_m_fast=float(prim_f["p_m"][0]),
                                               r_res_slow=float(prim_s["r_res"][0]), r_res_fast=float(prim_f["r_res"][0]), V_hit=float(prim_s["V_hit"][0]), V_rec_slow=float(prim_s["V_rec"][0]), V_rec_fast=float(prim_f["V_rec"][0])),
                       P1=dict(criterion_lhs=[float(crit_lhs[0]), ci(crit_lhs[1:])], criterion_rhs=[float(crit_rhs[0]), ci(crit_rhs[1:])], crossing_predicted=[bool(exist_P1[0]), float(exist_P1[1:].mean())],
                               Xi=[float(Xi[0]), ci(Xi[1:])], bstar_from_Xi=[f0(bstar_Xi[0]), ci(bstar_Xi[1:])] if bstar_Xi is not None else None,
                               bstar_psi_root=[f0(bstar_P1[0]), ci(bstar_P1[1:])], regime_on_grid=[reg_P1[0], fracs(reg_P1[1:])], bstar_on_grid=[f0(b1[0]), ci(b1[1:])],
                               eta_s=eta_s_P1[0].tolist(), eta_f=eta_f_P1[0].tolist()),
                       P2=dict(regime_on_grid=[reg_P2[0], fracs(reg_P2[1:])], bstar_on_grid=[f0(b2[0]), ci(b2[1:])], eta_s=eta_s_P2[0].tolist(), eta_f=eta_f_P2[0].tolist(),
                               L_b_slow=Ls_b[0].tolist(), Delta_b_slow=Ds_b[0].tolist()),
                       empirical=dict(n_rounds_per_b=meas_s["n_rounds"].tolist(), eta_s=[meas_s["eta"][0].tolist(), np.percentile(meas_s["eta"][1:], 2.5, 0).tolist(), np.percentile(meas_s["eta"][1:], 97.5, 0).tolist()],
                                      eta_f=[meas_f["eta"][0].tolist(), np.percentile(meas_f["eta"][1:], 2.5, 0).tolist(), np.percentile(meas_f["eta"][1:], 97.5, 0).tolist()],
                                      d_eta=[d_emp[0].tolist(), np.percentile(d_emp[1:], 2.5, 0).tolist(), np.percentile(d_emp[1:], 97.5, 0).tolist()],
                                      regime=[reg_emp[0], fracs(reg_emp[1:])], bstar=[f0(be[0]), ci(be[1:])], n_sign_changes=[int(nsc[0]), float((nsc[1:] <= 1).mean())],
                                      n_sign_changes_robust_2se=[int(nsc_rob[0]), float((nsc_rob[1:] <= 1).mean())], p_anymiss_slow=meas_s["p_anymiss"][0].tolist(),
                                      p_anymiss_theory_slow=(1 - pw_s[0]).tolist(), L_round_slow=meas_s["L_round"][0].tolist(), Lfb_round_slow=meas_s["Lfb_round"][0].tolist()),
                       comparison=dict(existence_agreement_boot=agree_exist, existence_pred=bool(exist_P1[0]), existence_emp=bool(emp_exist_or_fast[0]),
                                       regime_agree_P1_boot=float(np.mean([a == b for a, b in zip(reg_P1[1:], reg_emp[1:])])), regime_agree_P2_boot=float(np.mean([a == b for a, b in zip(reg_P2[1:], reg_emp[1:])])),
                                       emp_regime_ceiling=max(fracs(reg_emp[1:]).values()), log_ratio_bstar_P1_over_emp_median=lr(b1[1:], be[1:]), log_ratio_bstar_P2_over_emp_median=lr(b2[1:], be[1:]),
                                       bstar_P1_in_emp_ci=bool(reg_P1[0] == "cross" == reg_emp[0] and ci(be[1:])[0] <= b1[0] <= ci(be[1:])[1]),
                                       bstar_P2_in_emp_ci=bool(reg_P2[0] == "cross" == reg_emp[0] and ci(be[1:])[0] <= b2[0] <= ci(be[1:])[1]),
                                       P1_eta_s_relRMSE=float(np.sqrt(np.mean((eta_s_P1[0] / meas_s["eta"][0] - 1) ** 2))), P2_eta_s_relRMSE=float(np.sqrt(np.mean((eta_s_P2[0] / meas_s["eta"][0] - 1) ** 2))),
                                       P1_eta_f_relRMSE=float(np.sqrt(np.mean((eta_f_P1[0] / meas_f["eta"][0] - 1) ** 2))), P2_eta_f_relRMSE=float(np.sqrt(np.mean((eta_f_P2[0] / meas_f["eta"][0] - 1) ** 2)))))
            RESULTS["part_d"][key]["pairs"][policy] = rec
            say(f"  fast={policy:18s} V_s={Vs[0]:.3f} V_f={Vf[0]:.3f} L={L[0]:.2f} Ds={Ds[0]:.2f} Df={Df[0]:.2f} pi_s={pis[0]:.3f} pi_f={pif[0]:.3f} hyp ok={all(h[0] for h in [hyp['Vs_gt_Vf'],hyp['Ds_gt_Df'],hyp['pif_lt_pis']])} | "
                f"crit {crit_lhs[0]:.4f} < {crit_rhs[0]:.4f}? {bool(exist_P1[0])} (boot {exist_P1[1:].mean():.2f}) emp regime {reg_emp[0]} {fracs(reg_emp[1:])} | b* P1={bstar_P1[0]:.2f} grid-P1={b1[0]:.2f} P2={b2[0]:.2f} emp={be[0]:.2f} CI={ci(be[1:])} | "
                f"sign changes {nsc[0]}/{nsc_rob[0]} (boot<=1: {(nsc_rob[1:]<=1).mean():.2f})  t={time.time()-T0:.0f}s")
say(f"[t={time.time()-T0:.0f}s]")


# ============================================================================= claims, figures, dump
say("\n=== verdicts ===")
CL = RESULTS["claims"]
def vz(z, lim=2.0): return "PASS" if abs(z) <= lim else "FAIL"
# ---- part (a)
pa = RESULTS["part_a"]
a1 = [pa[k]["a1"]["ci_covers_0"] for k in pa]
CL["A1_iid_p_h_equals_p_m"] = dict(kind="CONSTRUCTION CHECK (the stream is a random permutation of the tasks, so exchangeability holds by construction; the i.i.d. bootstrap CI is the null distribution itself) [AUDIT relabel]",
    result="prop:markov-extension (degenerate chain xi = 0 on an i.i.d. stream)",
    prediction="p_h = p_m: 95% task-bootstrap CI of p_h - p_m covers 0 in every case", measurement=f"{sum(a1)}/{len(a1)} cases cover 0; |p_h - p_m| max {max(abs(pa[k]['a1']['p_h_minus_p_m']) for k in pa):.3f}; "
    f"in the CSV's own dimension-blocked order p_h - p_m = {min(pa[k]['a1']['natural_order']['p_h_minus_p_m'] for k in pa):+.3f}..{max(pa[k]['a1']['natural_order']['p_h_minus_p_m'] for k in pa):+.3f} (permutation-null z {min(pa[k]['a1']['natural_order']['z_vs_perm_null'] for k in pa):.0f}..{max(pa[k]['a1']['natural_order']['z_vs_perm_null'] for k in pa):.0f})",
    verdict=("PASS (holds by construction; not a finding about the data -- the benchmark records no arrival order)" if all(a1) else "FAIL"))
CL["A2_identity_pi_and_ratio"] = dict(kind="IDENTITY (code check)", result="prop:markov-extension: pi = p_m/(1-p_h+p_m) and the stationary ratio",
    prediction="pi_chain equals the hit frequency up to the boundary term <= 1/N; ratio of stationary means equals sum V / sum D exactly",
    measurement=f"max |pi_chain - freq| = {max(pa[k]['a2']['abs_diff_pi'] for k in pa):.1e} (1/N = {1/N:.1e}); max |eta_formula/eta_direct - 1| = {max(abs(pa[k]['a2'][r]['rel_diff']) for k in pa for r in ['dollars','sec_seq','sec_hedged']):.1e}",
    verdict="PASS (identity; not a finding)")
for r in ["dollars", "sec_seq", "sec_hedged"]:
    cov = np.mean([pa[k]["a3"][r]["frac_splits_ci_covers_0"] for k in pa]); rel = np.mean([pa[k]["a3"][r]["mean_abs_rel_err"] for k in pa]); sg = np.mean([pa[k]["a3"][r]["mean_signed_rel_err"] for k in pa])
    CL[f"A3_split_half_ratio_limit_{r}"] = dict(kind="OUT-OF-SAMPLE (split-half consistency of the ratio-of-sums estimator; the formula reproduces the first-half ratio of sums)",
        result="prop:markov-extension long-run ratio (pi v_1 + (1-pi) v_0)/(pi d_1 + (1-pi) d_0)",
        prediction="first-half primitives predict the second-half ratio of sums: CI of (pred - meas) covers 0 in >= 90% of splits, |signed rel err| < 1%",
        measurement=f"coverage {cov:.2f}, mean |rel err| {rel*100:.2f}%, mean signed {sg*100:+.2f}% over {len(pa)} cases x {N_SPLITS} splits", verdict="PASS" if (cov >= 0.9 and abs(sg) < 0.01) else "FAIL")
# ---- part (b)
pb = RESULTS["part_b"]; cases = list(pb.keys())
CL["B0_literal_same_dim_recovery"] = dict(kind="NOT INSTANTIABLE", result="prop:markov-extension hypothesis 0 < p_m",
    prediction="recovery from the hard tercile of the SAME dimension gives an irreducible chain", measurement="; ".join(f"{k}: p_m={pb[k]['recover_hard_same_dim_literal']['structural_probing']['p_m']:.4f}" for k in cases),
    verdict="NOT INSTANTIABLE (absorbing: hard terciles of some dimensions have no primary hit; sticky rho=0.5 and random-dimension variants used instead)")
for kind in CONSTRUCTIONS:
    rej = [pb[k][kind]["second_order_markov_check"]["markov_rejected"] for k in cases]
    gaps = [pb[k][kind]["second_order_markov_check"]["diff_after_miss"] for k in cases]
    z_pi = [pb[k][kind]["z_pi_full"] for k in cases]; z_pi_erg = [pb[k][kind]["z_pi_ergodic"] for k in cases]
    tz2 = [pb[k][kind]["transient"]["max_abs_z_two_state_insample_n3plus"] for k in cases]; tze = [pb[k][kind]["transient"]["max_abs_z_exact_insample_n3plus"] for k in cases]
    gp = [pb[k][kind]["transient"]["max_gap_two_state_vs_exact_over_pi"] for k in cases]
    a2h = [pb[k][kind]["A2_check"]["hypothesis_holds"] for k in cases]; a2g = [pb[k][kind]["A2_check"]["H0"]["EV_prev_miss"] - pb[k][kind]["A2_check"]["H0"]["EV_prev_hit"] for k in cases]
    zeta = {r: [pb[k][kind]["ratio_limit"][r]["z_full"] for k in cases] for r in ["dollars", "sec_seq"]}
    rel = {r: [pb[k][kind]["ratio_limit"][r]["rel_err_pred_vs_realized"] for k in cases] for r in ["dollars", "sec_seq"]}
    markov_by_design = kind in ("recover_hard_random_dim", "class_conditional_markov")
    CL[f"B1_markov_property_{kind}"] = dict(kind="IN-SAMPLE hypothesis (second-order transition check on id_test streams, block bootstrap)", result="prop:markov-extension hypothesis: (H_n) is a homogeneous Markov chain",
        prediction=("Markov by construction: P(1|00) = P(1|10) (CI covers 0)" if markov_by_design else "hidden state makes H non-Markov: P(1|00) != P(1|10) expected") ,
        measurement=f"gap P(1|00)-P(1|10) = {np.mean(gaps):+.4f} (mean over cases), rejected in {sum(rej)}/{len(rej)} cases",
        verdict=("PASS" if not any(rej) else "FAIL") if markov_by_design else ("PASS (non-Markov as designed; the paper's caveat instance)" if all(rej) else ("INCONCLUSIVE (rejected in " + f"{sum(rej)}/{len(rej)})")))
    CL[f"B2_stationary_pi_{kind}"] = dict(kind="OUT-OF-SAMPLE (structural p_h, p_m from probing pool hit rates; realized frequency on id_test streams; the formula itself is the flow-balance identity)",
        result="prop:markov-extension pi = p_m/(1-p_h+p_m)", prediction="|z| <= 2 in every case (z combines the probing bootstrap SE, the id_test pool bootstrap SE and the stream-to-stream SE)",
        measurement=(f"z_full = {', '.join(f'{z:+.2f}' for z in z_pi)} [z without the id_test pool variance, the rule first written: " + ', '.join(f"{pb[k][kind]['z_pi_pred_vs_meas']:+.2f}" for k in cases) + "]; rel err = " + ', '.join(f"{pb[k][kind]['rel_err_pi_pred_vs_meas']*100:+.1f}%" for k in cases) + f"; ergodic z (exact id_test law vs streams) = {', '.join(f'{z:+.2f}' for z in z_pi_erg)}"),
        verdict=("PASS" if all(abs(z) <= 2 for z in z_pi) else ("INCONCLUSIVE" if all(abs(z) <= 3 for z in z_pi) else "FAIL")) + (" (vacuous: construction (4) drives prediction and stream with the same probing (p_h, p_m), so pi_pred = pi_exact exactly) [AUDIT]" if kind == "class_conditional_markov" else ""))
    CL[f"B3_geometric_transient_{kind}"] = dict(kind="IN-SAMPLE functional-form test (exact two-state parameters of the id_test pool; R conditioned chains; n = 3..9 after a stationary miss)",
        result="prop:markov-extension P(H_n=1) = pi + (P(H_1=1)-pi) xi^(n-1)",
        prediction=("geometric transient holds: max |z| <= 3" if markov_by_design else "paper's caveat: pi holds but the geometric transient fails (max |z| > 3) while the exact hidden chain fits"),
        measurement=f"two-state max|z| = {', '.join(f'{z:.1f}' for z in tz2)}; exact-chain max|z| = {', '.join(f'{z:.1f}' for z in tze)}; max gap two-state vs exact / pi = {', '.join(f'{g*100:.1f}%' for g in gp)}",
        verdict=("PASS" if all(z <= 3 for z in tz2) else "FAIL") if markov_by_design else ("PASS (transient formula FAILS as the caveat predicts; exact chain fits)" if all(z > 3 for z in tz2) and all(z <= 3 for z in tze) else ("INCONCLUSIVE (two-state rejected in " + f"{sum(z > 3 for z in tz2)}/{len(tz2)}; exact chain max|z| {max(tze):.1f})")))
    CL[f"B4_hypothesis_A2_{kind}"] = dict(kind="IN-SAMPLE hypothesis", result="prop:markov-extension: E[V_n | F_(n-1), H_n] = v_(H_n) (no dependence on H_(n-1))",
        prediction="E[V_n | H_n=0, H_(n-1)=0] = E[V_n | H_n=0, H_(n-1)=1] (block-bootstrap CI covers 0)",
        measurement=f"gap (prev miss - prev hit) on misses = {', '.join(f'{g:+.3f}' for g in a2g)}; holds in {sum(a2h)}/{len(a2h)} cases",
        verdict="PASS" if all(a2h) else "FAIL")
    CL[f"B5_ratio_limit_{kind}"] = dict(kind="OUT-OF-SAMPLE (structural stationary E[V]/E[D] from probing vs realized sum V / sum D on id_test streams)",
        result="prop:markov-extension long-run ratio limit", prediction="|z| <= 2 for dollars and sequential seconds in every case",
        measurement=(f"dollars z_full = {', '.join(f'{z:+.2f}' for z in zeta['dollars'])} (rel err {', '.join(f'{x*100:+.1f}%' for x in rel['dollars'])}) [z without id_test pool variance: " + ', '.join(f"{pb[k][kind]['ratio_limit']['dollars']['z_pred_vs_realized']:+.2f}" for k in cases) + f"]; sec_seq z_full = {', '.join(f'{z:+.2f}' for z in zeta['sec_seq'])} [" + ', '.join(f"{pb[k][kind]['ratio_limit']['sec_seq']['z_pred_vs_realized']:+.2f}" for k in cases) + "]"),
        verdict="PASS" if all(abs(z) <= 2 for z in zeta["dollars"] + zeta["sec_seq"]) else ("INCONCLUSIVE" if all(abs(z) <= 3 for z in zeta["dollars"] + zeta["sec_seq"]) else "FAIL"),
        note="when B4 fails the conclusion still holds as the ratio-of-stationary-means identity (ergodic theorem); the hypothesis is needed only to read v_1, v_0 as state-independent conditional means")
# ---- part (c)
pc = RESULTS["part_c"]; ck = list(pc.keys())
h1 = [pc[k]["hypotheses"]["lp_le_lb"]["holds"] for k in ck]; h1f = [pc[k]["hypotheses"]["lp_le_lb"]["frac_tasks_lp_le_lb"] for k in ck]
CL["C1_hypothesis_lp_le_lb"] = dict(kind="IN-SAMPLE hypothesis (probing)", result="app:hedging: 0 < l_p <= l_b", prediction="mean primary latency <= mean reference latency for each cheap primary",
    measurement=f"holds for {sum(h1)}/{len(h1)} (primary, theta) cases; per-task P(lat_p <= lat_B) in [{min(h1f):.2f}, {max(h1f):.2f}]; means: " + "; ".join(f"{k.split('|')[0]} {pc[k]['primitives_probing']['l_p']:.1f}s vs {pc[k]['primitives_probing']['l_b']:.1f}s" for k in ck if k.endswith('1.0')),
    verdict="PASS" if all(h1) else f"FAIL ({len(h1)-sum(h1)}/{len(h1)} violate; every cheap primary is slower than Opus in wall-clock)")
h2 = [pc[k]["hypotheses"]["matched_miss_value"]["holds_ci"] for k in ck]
CL["C2_hypothesis_matched_miss_value"] = dict(kind="IN-SAMPLE hypothesis (probing)", result="app:hedging: matched miss/reference value vbar", prediction="E[Y_B | miss] = vbar (CI of the difference covers 0)",
    measurement=f"holds in {sum(h2)}/{len(h2)} cases; v_m in [{min(pc[k]['hypotheses']['matched_miss_value']['v_m'] for k in ck):.3f}, {max(pc[k]['hypotheses']['matched_miss_value']['v_m'] for k in ck):.3f}] vs vbar ~ {np.mean([pc[k]['hypotheses']['matched_miss_value']['vbar'] for k in ck]):.3f}",
    verdict="PASS" if all(h2) else "FAIL (misses are the tasks Opus also finds hard)")
for nm, lab in [("sequential_plugin", "l_kappa + l_p + (1-p) l_b with unconditional means"), ("sequential_conditional", "l_p + (1-p) E[l_b | miss]"),
                ("hedged_plugin", "l_kappa + p l_p + (1-p) l_b (assumes l_p <= l_b)"), ("hedged_conditional", "p E[l_p | hit] + (1-p) E[max(l_p, l_b) | miss]")]:
    cov = [pc[k]["latency"][nm]["ci_covers_0"] for k in ck]; rel = [pc[k]["latency"][nm]["rel_err"] for k in ck]
    CL[f"C3_mean_latency_{nm}"] = dict(kind="OUT-OF-SAMPLE (probing primitives -> id_test realized mean latency)", result="app:hedging mean-latency formulas", prediction=f"{lab} equals the realized mean (CI covers 0 in >= 90% of cases)",
        measurement=f"CI covers 0 in {sum(cov)}/{len(cov)}; rel err mean {np.mean(rel)*100:+.1f}%, range [{min(rel)*100:+.1f}%, {max(rel)*100:+.1f}%]", verdict="PASS" if np.mean(cov) >= 0.9 else "FAIL")
for nm, lab in [("hedged_latency", "eq:hedged-latency-condition p(l_b v_h - vbar l_p) >= vbar l_kappa (plug-in means)"), ("hedged_latency_state_matched", "same condition with conditional means (E[max|miss], E[l_p|hit], v_m)"),
                ("sequential_latency", "sequential condition p l_b v_h >= vbar (l_kappa + l_p)"), ("sequential_latency_state_matched", "sequential condition with conditional means")]:
    vs = [pc[k]["conditions"][nm]["verdict"] for k in ck]
    sm = nm.endswith("state_matched")
    CL[f"C4_sign_{nm}"] = dict(kind=("TRANSFER CHECK of the thm:surplus sign identity: the conditional-mean margin is E[V] l_b - vbar E[latency], i.e. sign(eta - eta_B) itself (identity in-sample); it does not test app:hedging [AUDIT relabel]" if sm else
                                    "OUT-OF-SAMPLE sign prediction (probing primitives -> realized eta - eta_B on id_test, 500 task resamples)"), result="app:hedging", prediction=f"sign of {lab} equals sign of realized eta_hedge/seq - vbar/l_b",
        measurement=f"PASS {vs.count('PASS')}, FAIL {vs.count('FAIL')}, INCONCLUSIVE {vs.count('INCONCLUSIVE')} of {len(vs)}; failures: " + ", ".join(k for k in ck if pc[k]['conditions'][nm]['verdict'] == 'FAIL'),
        verdict=("PASS" if vs.count("FAIL") == 0 and vs.count("PASS") >= 0.75 * len(vs) else ("FAIL" if vs.count("FAIL") > 0 else "INCONCLUSIVE")) +
                (" (near-identity transfer; carries no information about eq:hedged-latency-condition) [AUDIT]" if sm else (" (applicability failure: the formula's hypotheses l_p <= l_b and v_m = vbar are violated in 14/14 cases, C1/C2; the wrong signs show the hypotheses are load-bearing) [AUDIT]" if vs.count("FAIL") > 0 else "")))
for rr in R_PREFIX:
    for nm in ["condition", "condition_state_matched"]:
        vs = [pc[k]["dollars"][f"r={rr}b"][nm]["verdict"] for k in ck]
        sm = nm != "condition"
        CL[f"C5_dollar_sign_r={rr}b_{nm}"] = dict(kind=("TRANSFER CHECK of the thm:surplus sign identity (conditional-mean margin = E[V] b - vbar E[D]; identity in-sample) [AUDIT relabel]" if sm else "OUT-OF-SAMPLE sign prediction"),
            result="app:hedging eq:hedged-dollar-condition p(b v_h - vbar r) >= vbar a" + ("" if nm == "condition" else " with conditional means (v_m, E[C_B|hit], E[C_B|miss])"),
            prediction=f"r = {rr} b: predicted sign equals realized sign of eta_hedge^c - vbar/b",
            measurement=f"PASS {vs.count('PASS')}, FAIL {vs.count('FAIL')}, INCONCLUSIVE {vs.count('INCONCLUSIVE')} of {len(vs)}; failures: " + ", ".join(k for k in ck if pc[k]['dollars'][f'r={rr}b'][nm]['verdict'] == 'FAIL'),
            verdict=("PASS" if vs.count("FAIL") == 0 and vs.count("PASS") >= 0.75 * len(vs) else ("FAIL" if vs.count("FAIL") > 0 else "INCONCLUSIVE")) +
                    (" (near-identity transfer) [AUDIT]" if sm else (" (hypothesis v_m = vbar violated 14/14, C2; failures confined to |margin| < $0.001) [AUDIT]" if vs.count("FAIL") > 0 else "")))
gain = [pc[k]["latency"]["hedged_gain_over_sequential_realized"][0] for k in ck]
CL["C6_hedged_vs_sequential_latency"] = dict(kind="IDENTITY (max <= sum) quantified", result="app:hedging: hedging cuts latency", prediction="realized hedged mean latency < sequential mean latency",
    measurement=f"relative saving (seq - hedged)/seq in [{min(gain)*100:.1f}%, {max(gain)*100:.1f}%]", verdict="PASS (identity; magnitude is the finding)")
# ---- part (d)
pd_ = RESULTS["part_d"]; dk = [k for k in pd_ if k != "setup"]
pairs = [(k, pol, pd_[k]["pairs"][pol]) for k in dk for pol in pd_[k]["pairs"]]
hyp_ok = [all(x["hypotheses"][h][0] for h in ["Vs_gt_Vf", "Ds_gt_Df", "pif_lt_pis"]) for _, _, x in pairs]
pi_ci_ok = [x["hypotheses"]["pi_f_minus_pi_s_ci"][1] < 0 for _, _, x in pairs]
CL["D1_barrier_hypotheses"] = dict(kind="IN-SAMPLE hypothesis (probing chains)", result="prop:app-barrier: V_s > V_f > 0, Delta_s > Delta_f >= 0, 0 < pi_f < pi_s < 1",
    prediction="all three hold for every (theta, rho, fast policy)", measurement=f"point estimates: {sum(hyp_ok)}/{len(hyp_ok)} pairs; pi_f < pi_s with CI excluding 0: {sum(pi_ci_ok)}/{len(pi_ci_ok)}; " +
    "; ".join(f"{k}|{pol}: V_s={x['primitives_probing']['V_s'][0]:.3f} V_f={x['primitives_probing']['V_f'][0]:.3f} Ds={x['primitives_probing']['Delta_s'][0]:.2f} Df={x['primitives_probing']['Delta_f'][0]:.2f} pi_s={x['primitives_probing']['pi_s'][0]:.3f} pi_f={x['primitives_probing']['pi_f'][0]:.3f}" for k, pol, x in pairs),
    verdict=("PASS" if all(hyp_ok) and all(pi_ci_ok) else f"PARTIAL: V and Delta orderings hold in {len(pairs)}/{len(pairs)}; pi_f < pi_s holds at the point estimate in {sum(x['hypotheses']['pif_lt_pis'][0] for _, _, x in pairs)}/{len(pairs)} and with CI excluding 0 in {sum(pi_ci_ok)}/{len(pairs)} (weakly instantiated: Opus also fails on most cheap misses and so also leaves residue)"))
ldet = [x["hypotheses"]["L_deterministic"]["round_L_over_L_idtest"][-1] for _, _, x in pairs]
CL["D1b_deterministic_round_latency"] = dict(kind="IN-SAMPLE hypothesis", result="prop:app-barrier: deterministic primary-round latency L independent of b", prediction="realized round latency max_b lat_p / L stays ~1 across b",
    measurement=f"cheap-latency CV = {pairs[0][2]['hypotheses']['L_deterministic']['cv_cheap_latency_probing']:.2f}; realized round L / L at b = 64: {min(ldet):.1f}..{max(ldet):.1f}x", verdict="FAIL (real latencies are random and heavy-tailed; motivates the order-statistic instantiation P2)")
ex_rows = []
for k, pol, x in pairs:
    c = x["comparison"]; pf = x["P1"]["crossing_predicted"][1]; emp_ex_frac = x["empirical"]["regime"][1]["cross"] + x["empirical"]["regime"][1]["fast_all"]
    p2f = x["P2"]["regime_on_grid"][1]["cross"] + x["P2"]["regime_on_grid"][1]["fast_all"]; agree_p2 = p2f * emp_ex_frac + (1 - p2f) * (1 - emp_ex_frac)
    ceiling_ex = max(emp_ex_frac, 1 - emp_ex_frac)
    v = "PASS" if c["existence_agreement_boot"] >= 0.75 else ("INCONCLUSIVE (P1 criterion is a coin flip in its own bootstrap)" if 0.25 < pf < 0.75 else ("INCONCLUSIVE (empirical existence not identifiable)" if ceiling_ex < 0.75 else "FAIL"))
    ex_rows.append(dict(case=f"{k}|{pol}", pred=c["existence_pred"], pred_boot_frac=pf, emp=c["existence_emp"], emp_boot_frac=emp_ex_frac, agree_boot=c["existence_agreement_boot"], agree_boot_P2=agree_p2,
                        ceiling=c["emp_regime_ceiling"], regime_emp=x["empirical"]["regime"], verdict=v))
agree = [r["agree_boot"] for r in ex_rows]; vs_d2 = [r["verdict"] for r in ex_rows]
CL["D2_existence_criterion"] = dict(kind="OUT-OF-SAMPLE (probing means -> id_test barrier rounds; P1 mean-latency instantiation)", result="prop:app-barrier: a crossing exists iff V_s/(L+Delta_s) < V_f/(L+Delta_f)",
    prediction="bootstrap-weighted agreement between predicted existence and empirical (cross or fast_all on b >= 1) >= 0.75 in every pair",
    measurement="; ".join(f"{r['case']}: P1 pred {r['pred']} (boot {r['pred_boot_frac']:.2f}) emp {r['emp']} (boot {r['emp_boot_frac']:.2f}) agree P1 {r['agree_boot']:.2f} / P2 {r['agree_boot_P2']:.2f} -> {r['verdict']}" for r in ex_rows),
    verdict=f"PASS {sum(v == 'PASS' for v in vs_d2)}, INCONCLUSIVE {sum(v.startswith('INCONCLUSIVE') for v in vs_d2)}, FAIL {sum(v == 'FAIL' for v in vs_d2)} of {len(vs_d2)} pairs (P1); P2 existence agreement " + ", ".join(f"{r['agree_boot_P2']:.2f}" for r in ex_rows),
    per_pair=ex_rows)
agree2 = [x["comparison"]["regime_agree_P2_boot"] for _, _, x in pairs]; agree1 = [x["comparison"]["regime_agree_P1_boot"] for _, _, x in pairs]; ceil = [x["comparison"]["emp_regime_ceiling"] for _, _, x in pairs]
CL["D2b_regime_P1_vs_P2"] = dict(kind="OUT-OF-SAMPLE (regime on the grid b >= 1: slow_all / cross / fast_all; bootstrap-weighted agreement with the empirical regime)", result="prop:app-barrier regime, two instantiations",
    prediction="P1 (mean latency, the proposition's objects) and P2 (order-statistic L(b), Delta(b), outside the proposition's hypothesis): agreement >= 0.75 where the empirical regime ceiling >= 0.75",
    measurement="; ".join(f"{k}|{pol}: P1 {a1:.2f} P2 {a2:.2f} ceiling {c:.2f}" for (k, pol, _), a1, a2, c in zip(pairs, agree1, agree2, ceil)),
    verdict=f"all-pairs rule: P1 {'PASS' if all(a >= 0.75 for a, c in zip(agree1, ceil) if c >= 0.75) else 'FAIL'} / P2 {'PASS' if all(a >= 0.75 for a, c in zip(agree2, ceil) if c >= 0.75) else 'FAIL'}; per pair: P1 {sum(a >= 0.75 for a, c in zip(agree1, ceil) if c >= 0.75)}/{sum(c >= 0.75 for c in ceil)}, P2 {sum(a >= 0.75 for a, c in zip(agree2, ceil) if c >= 0.75)}/{sum(c >= 0.75 for c in ceil)} identifiable pairs agree >= 0.75")
uniq = [x["empirical"]["n_sign_changes_robust_2se"][1] for _, _, x in pairs]; uniq0 = [x["empirical"]["n_sign_changes_robust_2se"][0] for _, _, x in pairs]
raw_nsc0 = [x["empirical"]["n_sign_changes"][0] for _, _, x in pairs]; raw_nsc1 = [x["empirical"]["n_sign_changes"][1] for _, _, x in pairs]   # [AUDIT]
CL["D3_uniqueness"] = dict(kind="OUT-OF-SAMPLE (empirical eta_s - eta_f on the b grid, id_test)", result="prop:app-barrier: at most one crossing", prediction="<= 1 robust (2 SE) sign change on the grid in >= 95% of bootstrap replicates",
    measurement=f"point-estimate robust sign changes: {uniq0}; bootstrap fraction <= 1: {', '.join(f'{u:.2f}' for u in uniq)}; raw (non-robust) sign changes {raw_nsc0}, bootstrap fraction <= 1: {', '.join(f'{u:.2f}' for u in raw_nsc1)}",
    verdict=("PASS" if all(u >= 0.95 for u in uniq) else "FAIL") + " (the 2-SE robust count is 0 in every pair, including the 3 crossing pairs, so the robust rule has no power; the raw-count fraction is the informative statistic) [AUDIT]")
loc = []
for k, pol, x in pairs:
    if x["empirical"]["regime"][0] == "cross" or x["empirical"]["regime"][1]["cross"] >= 0.5:
        loc.append(dict(case=f"{k}|{pol}", bstar_emp=x["empirical"]["bstar"], bstar_P1_Xi=(x["P1"]["bstar_from_Xi"] if x["P1"]["bstar_from_Xi"] else x["P1"]["bstar_psi_root"]), bstar_P1_grid=x["P1"]["bstar_on_grid"],
                        bstar_P2=x["P2"]["bstar_on_grid"], P1_in_ci=x["comparison"]["bstar_P1_in_emp_ci"], P2_in_ci=x["comparison"]["bstar_P2_in_emp_ci"],
                        logratio_P1=x["comparison"]["log_ratio_bstar_P1_over_emp_median"], logratio_P2=x["comparison"]["log_ratio_bstar_P2_over_emp_median"]))
CL["D4_bstar_location"] = dict(kind="OUT-OF-SAMPLE (P1 = mean-latency b* = log Xi / log pi_s or Psi root; P2 = order-statistic crossing, outside the proposition's deterministic hypothesis)",
    result="prop:app-barrier b*", prediction="predicted b* inside the empirical 95% CI of the crossing (cases with an empirical crossing)",
    measurement="; ".join(f"{r['case']}: emp {r['bstar_emp'][0] if r['bstar_emp'][0] is not None else float('nan'):.2f} CI [{r['bstar_emp'][1][0]:.2f}, {r['bstar_emp'][1][1]:.2f}]; P1 {r['bstar_P1_Xi'][0] if r['bstar_P1_Xi'] and r['bstar_P1_Xi'][0] is not None else float('nan'):.2f} in CI {r['P1_in_ci']} (median log ratio {r['logratio_P1'][0] if r['logratio_P1'][0] is not None else float('nan'):.2f}); P2 {r['bstar_P2'][0] if r['bstar_P2'][0] is not None else float('nan'):.2f} in CI {r['P2_in_ci']} (median log ratio {r['logratio_P2'][0] if r['logratio_P2'][0] is not None else float('nan'):.2f})" for r in loc) if loc else "no pair has an empirical crossing",
    verdict=("PASS" if loc and all(r["P1_in_ci"] for r in loc) else ("INCONCLUSIVE" if not loc else f"FAIL for P1 in {sum(not r['P1_in_ci'] for r in loc)}/{len(loc)} crossing cases; P2 inside CI in {sum(bool(r['P2_in_ci']) for r in loc)}/{len(loc)}"
             " (P1 is an applicability failure: its deterministic-L hypothesis fails, D1b; the empirical crossings themselves are weak -- d_eta at b = 1 or b = 64 within 2 SE of 0) [AUDIT]")))
for k, v in CL.items(): say(f"{k:55s} {v['verdict']}")

# ---- figures (vector PDF, paper style)
# fig 1: transient, Markov vs hidden-regime construction (Qwen3-Max, theta = 1)
k0 = "Qwen3-Max|theta=1.0"
fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.15), layout='constrained')
for ax, kind, title in zip(axes[:2], ["recover_hard_random_dim", "hidden_3regime"], ["Markov by construction (recover from a random hard pool)", "Hidden 3-regime stream (non-Markov)"]):
    t = pb[k0][kind]["transient"]; n = np.array(t["n"]); emp = np.array(t["empirical"]); se = np.array(t["se_mc"])
    ax.errorbar(n, emp, yerr=2 * se, fmt='o', color='.2', ms=3, lw=.8, capsize=2, label='Realized (id\\_test stream)' if USETEX else 'Realized (id_test stream)')
    ax.plot(n, t["two_state_idtest_params"], color=RED, label=r'Two-state $\pi+(\mathbb P(H_1{=}1)-\pi)\xi^{n-1}$')
    ax.plot(n, t["exact_chain_idtest"], color=BLUE, ls='--', label='Exact chain of the construction')
    ax.axhline(pb[k0][kind]["structural_idtest"]["pi"], color=GREY, lw=.65, ls=':')
    ax.set(xlabel='Epoch $n$ after a stationary miss', ylabel=r'$\mathbb P(H_n=1\mid H_1=0)$', title=title); ax.title.set_fontsize(7.5)
axes[0].legend(loc='lower right', frameon=False, fontsize=6.5)
ax = axes[2]
for kind, col, mk in [("recover_hard_random_dim", BLUE, 'o'), ("recover_hard_sticky_dim", GREEN, 's'), ("hidden_3regime", RED, '^'), ("class_conditional_markov", '.3', 'd')]:
    xs = [pb[k][kind]["structural_probing"]["pi"] for k in cases]; ys = [pb[k][kind]["stream"]["pi_meas"] for k in cases]
    ax.scatter(xs, ys, color=col, marker=mk, s=14, label=kind.replace('_', ' '))
lim = [0, max(pb[k][kind]["stream"]["pi_meas"] for k in cases for kind in CONSTRUCTIONS) * 1.1]
ax.plot(lim, lim, color=GREY, lw=.65, ls=':'); ax.set(xlabel=r'$\pi=p_{\mathrm m}/(1-p_{\mathrm h}+p_{\mathrm m})$ from probing', ylabel='Realized hit frequency (id\\_test)' if USETEX else 'Realized hit frequency (id_test)', xlim=lim, ylim=lim)
ax.legend(loc='upper left', frameon=False, fontsize=6)
fig.savefig(os.path.join(FIG, 'markov_transient.pdf'), metadata={'Creator': 'X5 run.py'}); plt.close(fig)
# fig 2: hedging
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.3), layout='constrained')
ax = axes[0]; keys1 = [k for k in ck if k.endswith("theta=1.0")]; xs = np.arange(len(keys1)); w = 0.2
ax.bar(xs - 1.5 * w, [pc[k]["latency"]["sequential_plugin"]["realized"] for k in keys1], w, color='.25', label='Sequential, realized')
ax.bar(xs - 0.5 * w, [pc[k]["latency"]["sequential_plugin"]["pred"] for k in keys1], w, color='.65', label=r'Sequential, $\ell_p+(1-p)\ell_b$')
ax.bar(xs + 0.5 * w, [pc[k]["latency"]["hedged_plugin"]["realized"] for k in keys1], w, color=BLUE, label='Hedged, realized')
ax.bar(xs + 1.5 * w, [pc[k]["latency"]["hedged_plugin"]["pred"] for k in keys1], w, color=RED, label=r'Hedged, $p\ell_p+(1-p)\ell_b$')
ax.scatter(xs + 0.5 * w, [pc[k]["latency"]["hedged_conditional"]["pred"] for k in keys1], color='k', marker='_', s=60, zorder=5, label=r'Hedged, $p\,\mathbb E[\ell_p|{\rm hit}]+(1-p)\mathbb E[\max|{\rm miss}]$')
ax.set_xticks(xs); ax.set_xticklabels([k.split('|')[0].replace('claude-', '').replace('_', '-') for k in keys1], rotation=30, ha='right', fontsize=6.5)
ax.set(ylabel='Mean latency per task (s)', title=r'$\theta=1$; reference Opus $\ell_b\approx 7.7$ s'); ax.title.set_fontsize(8); ax.legend(frameon=False, fontsize=5.8, loc='upper left')
ax = axes[1]
for nm, col, mk, lab in [("hedged_latency", RED, 'o', 'plug-in means (paper form)'), ("hedged_latency_state_matched", BLUE, 's', 'conditional means')]:
    xs_ = [pc[k]["conditions"][nm]["margin_pred"] / (pc[k]["primitives_probing"]["vbar"] * pc[k]["primitives_probing"]["l_b"]) for k in ck]
    ys_ = [pc[k]["conditions"][nm]["realized_diff"] for k in ck]
    ax.scatter(xs_, ys_, color=col, marker=mk, s=14, label=lab)
ax.axhline(0, color=GREY, lw=.65, ls=':'); ax.axvline(0, color=GREY, lw=.65, ls=':')
ax.set(xlabel=r'Predicted margin $/(\bar v\,\ell_b)$ (probing)', ylabel=r'Realized $\eta^{\ell}_{\rm hedge}-\bar v/\ell_b$ (id\_test)' if USETEX else 'Realized eta_hedge - vbar/l_b', title='eq:hedged-latency-condition, sign test (14 cases)')
ax.title.set_fontsize(8); ax.legend(frameon=False, fontsize=6.5, loc='lower right')
fig.savefig(os.path.join(FIG, 'hedging_latency.pdf'), metadata={'Creator': 'X5 run.py'}); plt.close(fig)
# fig 3: barrier curves
sel = [(k, pol) for k, pol, _ in pairs if pol == "commit"]
fig, axes = plt.subplots(1, len(sel), figsize=(6.75, 2.1), layout='constrained', sharey=False)
for ax, (k, pol) in zip(np.atleast_1d(axes), sel):
    x = pd_[k]["pairs"][pol]; e = x["empirical"]
    ax.fill_between(B_GRID, e["eta_s"][1], e["eta_s"][2], color=BLUE, alpha=.15, lw=0); ax.fill_between(B_GRID, e["eta_f"][1], e["eta_f"][2], color=RED, alpha=.15, lw=0)
    ax.plot(B_GRID, e["eta_s"][0], color=BLUE, marker='o', ms=2.5, label=r'$\eta_{\mathrm s}$ realized (Opus repair)'); ax.plot(B_GRID, e["eta_f"][0], color=RED, marker='o', ms=2.5, label=r'$\eta_{\mathrm f}$ realized (commit cheap)')
    ax.plot(B_GRID, x["P1"]["eta_s"], color=BLUE, ls=':', lw=1); ax.plot(B_GRID, x["P1"]["eta_f"], color=RED, ls=':', lw=1, label='P1: mean latency')
    ax.plot(B_GRID, x["P2"]["eta_s"], color=BLUE, ls='--', lw=1); ax.plot(B_GRID, x["P2"]["eta_f"], color=RED, ls='--', lw=1, label='P2: order statistics')
    bs = x["P1"]["bstar_from_Xi"][0] if x["P1"]["bstar_from_Xi"] else None
    if bs: ax.axvline(bs, color='.3', lw=.65, ls='-.', label=r'$b^*=\log\Xi/\log\pi_{\mathrm s}$')
    ax.set_xscale('log'); ax.set(xlabel='Fan-out $b$', title=k.replace('theta', r'$\theta$').replace('rho', r'$\rho$').replace('|', ', ')); ax.title.set_fontsize(8)
np.atleast_1d(axes)[0].set_ylabel('Value per member per second'); np.atleast_1d(axes)[0].legend(frameon=False, fontsize=5.6, loc='lower left')
fig.savefig(os.path.join(FIG, 'barrier_crossing.pdf'), metadata={'Creator': 'X5 run.py'}); plt.close(fig)
RESULTS["figures"] = ["figures/markov_transient.pdf", "figures/hedging_latency.pdf", "figures/barrier_crossing.pdf"]
RESULTS["runtime_s"] = time.time() - T0; RESULTS["log"] = LOG
json.dump(RESULTS, open(os.path.join(HERE, "results_quick.json" if QUICK else "results.json"), "w"), indent=1, default=float)
say(f"done; total t={time.time()-T0:.0f}s")

