#!/usr/bin/env python3
"""
X16_sensitivity -- ablations and sensitivity of the X1 / X3 / X4 headline conclusions of
"Route Anything: A Theory of Inference Routers" on CodeRouterBench (fully offline).

Conclusions stressed (quoted from the X1 / X3 / X4 README claims tables; operationalised in README.md):
  (a) X1 C5/C8/C9: the homogeneous floor eq:floor is optimistic and predicts fewer realized signs than the
      state-matched condition eq:matched-reference-condition            [OUT-OF-SAMPLE]
  (b) X1 C8: eq:normalized over-predicts held-out cascades              [OUT-OF-SAMPLE]
  (c) X3 C1: prop:threshold predicts the realized sign of J_casc - J_direct in >= 90% of decided states
  (d) X3 C7: the VOI band prop:voi selects the realized-cheapest option in >= 95% of decided cells
  (e) X4 (c): records improve dollar efficiency against always-Opus but not in seconds and not against DimensionBest
  (f) X4 (d): the noise threshold eq:noise-threshold predicts record-policy decisions     [SEMI-SYNTHETIC + OOS]
Axes: reference policy {always-Opus, DimensionBest (voter file), per-task oracle, random uniform model x 20 draws};
      price lambda {0.1, 1, 10, eta_B, 100}; cost basis {dollars, seconds (sequential), tokens};
      verifier noise (beta_fa, beta_fr) in {0, .05, .1, .2, .4}^2 with the coin RE-DRAWN inside every bootstrap
      resample (binomial thinning of the multinomial task counts; X1/X3 held one draw per task fixed);
      value shift +c, c in {0.1, 0.5}, added to every score.
Primitives on probing (7,080 tasks); realized values on id_test (2,919).  1,000 paired task bootstraps
(seed 20260917); 500 for the coin-re-drawn record-stream bootstraps (the heaviest cells).
numpy / pandas / scipy / matplotlib only; no API calls.
"""
import os, sys, json, time, math, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

T0 = time.time()
SEED = 20260917
QUICK = '--quick' in sys.argv                       # debugging only: fewer bootstraps / settings
B = 200 if QUICK else 1000                          # paired task bootstraps
B_REDRAW = 100 if QUICK else 500                    # coin-re-drawn record-stream bootstraps (heaviest cells)
N_RANDOM = 3 if QUICK else 20                       # random-reference draws
OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, 'figures'); os.makedirs(FIG, exist_ok=True)
DATA = os.environ.get('ACROUTER_DATA', 'data/agent-as-a-router/data')
BASE = 'claude-opus-4-6'
PRIMARIES = ['Qwen3-Max', 'kimi-k2.5', 'qwen3.5-plus', 'glm-5', 'gpt-5.4', 'claude-sonnet-4-6', 'MiniMax-M2.7']
CHEAPS = ['Qwen3-Max', 'kimi-k2.5']                 # X3's primaries
THETAS = [1.0, 0.5]
BETAS = [0.0, 0.05, 0.1, 0.2, 0.4]
NOISES = [(fa, fr) for fa in BETAS for fr in BETAS]
LAMS_EXO = [0.1, 1.0, 10.0, 100.0]                  # exogenous prices per dollar; 'etaB' = the ratio form
SHIFTS = [0.1, 0.5]
KS_USD = [3e-5, 1e-4, 5e-4, 1e-3]                   # X3's information prices (dollars per task)
BASES = ['dollars', 'seconds', 'tokens']
EPS = 0.25                                          # exploration probability of the record logger (X4)
SEEDS_STREAM, N_ROUNDS, LATE = 50, 20000, 5000      # record stream (X4: 50 streams x 20,000 rounds, last 5,000 stationary)

import multiprocessing as _mp
def log(*a):
    if _mp.current_process().name == 'MainProcess':
        print(f'[{time.time()-T0:7.1f}s]', *a, flush=True)

# ------------------------------------------------------------------ figure style (CONVENTIONS: usetex False, 9-pt serif, no titles)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'STIXGeneral', 'DejaVu Serif'], 'text.usetex': False,
    'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 7.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': .6, 'lines.linewidth': 1.4, 'axes.edgecolor': '#c3c2b7', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'axes.labelcolor': '#0b0b0b', 'grid.color': '#e1e0d9', 'grid.linewidth': .5, 'legend.frameon': False,
})
C1_, C2_, C3_, C4_ = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'    # validated categorical slots 1-4 (dataviz palette)
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
SEQ_CMAP = LinearSegmentedColormap.from_list('seqblue', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV_CMAP = LinearSegmentedColormap.from_list('divbr', ['#2a78d6', '#f0efec', '#e34948'])

# =========================================================================================== 1. data (loaders copied from X1)
df = pd.read_csv(f'{DATA}/coderouterbench/id_results_long.csv')
assert df.shape[0] == 79992
S = df.pivot(index='task_id', columns='model', values='score')
C = df.pivot(index='task_id', columns='model', values='cost_usd')
L = df.pivot(index='task_id', columns='model', values='latency_ms') / 1000.0        # seconds
TK = df.pivot(index='task_id', columns='model', values='total_tokens').astype(float)
assert S.notna().all().all() and C.notna().all().all() and L.notna().all().all() and TK.notna().all().all()
meta = df.drop_duplicates('task_id').set_index('task_id').loc[S.index, ['split', 'dimension']].copy()
meta['bench'] = meta.index.str.split('_').str[1]
meta['cell21'] = meta.dimension + '|' + meta.bench
DIMS = sorted(meta.dimension.unique()); CELLS21 = sorted(meta.cell21.unique())
TASKS = S.index.to_numpy(); MODELS = list(S.columns); col = {m: j for j, m in enumerate(MODELS)}; OPUS = col[BASE]
T, M = S.shape
Sv, Cv, Lv, Tv = S.to_numpy(), C.to_numpy(), L.to_numpy(), TK.to_numpy()
RES = {'dollars': Cv, 'seconds': Lv, 'tokens': Tv}
idt = (meta.split == 'id_test').to_numpy(); prob = (meta.split == 'probing').to_numpy()
assert idt.sum() == 2919 and prob.sum() == 7080 and len(CELLS21) == 21
IDX_TEST = np.where(idt)[0]; IDX_PROB = np.where(prob)[0]
test_pos = np.full(T, -1); test_pos[IDX_TEST] = np.arange(len(IDX_TEST))
cell21_of = meta.cell21.map({c: i for i, c in enumerate(CELLS21)}).to_numpy()
dim_of = meta.dimension.map({d: i for i, d in enumerate(DIMS)}).to_numpy()
HIT = {th: (Sv >= th - 1e-12) for th in THETAS}

DATA_CHECKS = dict(n_tasks=int(T), n_models=int(M), n_probing=int(prob.sum()), n_id_test=int(idt.sum()),
                   always_opus_id_test_total_usd=float(Cv[idt, OPUS].sum()), paper_always_opus_total_usd=34.02,
                   always_opus_id_test_mean_score=float(Sv[idt, OPUS].mean()), paper_always_opus_avgperf=43.83,
                   rows_cost_zero=int((df.cost_usd <= 0).sum()), rows_latency_zero=int((df.latency_ms <= 0).sum()),
                   rows_latency_zero_by_model={k: int(v) for k, v in df[df.latency_ms <= 0].model.value_counts().items()},
                   mean_latency_s_id_test_by_model={m: float(Lv[idt, col[m]].mean()) for m in MODELS},
                   mean_cost_usd_id_test_by_model={m: float(Cv[idt, col[m]].mean()) for m in MODELS},
                   mean_tokens_id_test_by_model={m: float(Tv[idt, col[m]].mean()) for m in MODELS},
                   opus_is_fastest_on_average_id_test=bool(np.argmin([Lv[idt, col[m]].mean() for m in MODELS]) == OPUS))
assert abs(DATA_CHECKS['always_opus_id_test_total_usd'] - 34.02) < 0.01 and abs(DATA_CHECKS['always_opus_id_test_mean_score'] - 0.4383) < 5e-4
log('data checks', json.dumps(DATA_CHECKS)[:400])

# ------------------------------------------------------------------ reference policies (per-task model index, all 9,999 tasks)
REFS = {'opus': np.full(T, OPUS)}
_vd = pd.read_json(f'{DATA}/id/voter_decisions/dimension_best.jsonl', lines=True)
_vd = _vd[_vd.task_id.isin(S.index)].set_index('task_id'); _vd['dimension'] = meta.loc[_vd.index, 'dimension']
_const = _vd.groupby('dimension').chosen_model.nunique()
assert (_const == 1).all() and len(_vd) == 2919, 'released DimensionBest is not constant per dimension'
DIMBEST_MAP = _vd.groupby('dimension').chosen_model.first().to_dict()
REFS['dimbest'] = np.array([col[DIMBEST_MAP[d]] for d in meta.dimension])
# per-task oracle: max score, ties -> min dollar cost (fixed policy, same in every basis)
_ismax = Sv >= Sv.max(1)[:, None] - 1e-12
REFS['oracle'] = np.where(_ismax, Cv, np.inf).argmin(1)
_rng_ref = np.random.default_rng(SEED)
RANDOM_DRAWS = _rng_ref.integers(0, M, size=(N_RANDOM, T))
for d in range(N_RANDOM):
    REFS[f'random{d:02d}'] = RANDOM_DRAWS[d]
REF_TYPES = ['opus', 'dimbest', 'oracle'] + [f'random{d:02d}' for d in range(N_RANDOM)]
REF_INFO = {r: dict(id_test_mean_score=float(Sv[IDX_TEST, REFS[r][IDX_TEST]].mean()), id_test_mean_usd=float(Cv[IDX_TEST, REFS[r][IDX_TEST]].mean()),
                    id_test_mean_s=float(Lv[IDX_TEST, REFS[r][IDX_TEST]].mean()), id_test_mean_tokens=float(Tv[IDX_TEST, REFS[r][IDX_TEST]].mean()),
                    probing_eta_dollars=float(Sv[IDX_PROB, REFS[r][IDX_PROB]].sum() / Cv[IDX_PROB, REFS[r][IDX_PROB]].sum()),
                    model_shares_id_test={MODELS[k]: int(v) for k, v in zip(*np.unique(REFS[r][IDX_TEST], return_counts=True))})
            for r in REF_TYPES}
log('references', {r: (round(REF_INFO[r]['id_test_mean_score'], 4), round(1000 * REF_INFO[r]['id_test_mean_usd'], 3)) for r in REF_TYPES[:4]})
# Opus probing mean cost per basis -> information price k of prop:voi expressed as the same fraction of the reference's cost in every basis
OPUS_PROB_MEAN = {b: float(RES[b][IDX_PROB, OPUS].mean()) for b in BASES}
K_SCALE = {b: OPUS_PROB_MEAN[b] / OPUS_PROB_MEAN['dollars'] for b in BASES}

# =========================================================================================== 2. cascade module (X1 / X3 estimators, weighted sums)
def boot_weights(n, rng, nb=B):
    """(nb+1) x n int32; row 0 = ones (point estimate), rows 1..nb = multinomial task counts (paired across every setting)."""
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=nb).astype(np.int32)
    return np.vstack([np.ones((1, n), np.int32), W])

_rng_w = np.random.default_rng(SEED + 1)
CELLS = {'all': [('all', np.ones(T, bool))], 'dim9': [(d, dim_of == i) for i, d in enumerate(DIMS)], 'cell21': [(c, cell21_of == i) for i, c in enumerate(CELLS21)]}
IDX = {}; W = {}
for split, smask in [('probing', prob), ('id_test', idt)]:
    for part in CELLS:
        for cname, cmask in CELLS[part]:
            ix = np.where(smask & cmask)[0]; IDX[(split, part, cname)] = ix; W[(split, part, cname)] = boot_weights(len(ix), _rng_w)
log('bootstrap weights drawn for', len(W), '(split, cell) pairs')

def _div(a, b):
    b = np.asarray(b, float); return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)

def hit_weights(Wt, h, noise, rng):
    """Hit-weight matrix under a noisy verifier: acceptance prob q_i = (1-beta_fr) h_i + beta_fa (1-h_i).
    Row 0 = expected coin (deterministic point estimate); rows 1..B = Binomial(count_i, q_i): every resampled task copy
    draws its own coin, so the verifier coin is re-drawn inside every bootstrap resample (X1/X3 held one draw fixed)."""
    fa, fr = noise
    h = h.astype(float)
    if fa == 0.0 and fr == 0.0:
        return Wt * h
    q = h * (1 - fr) + (1 - h) * fa
    Wh = np.empty(Wt.shape, float); Wh[0] = q
    Wh[1:] = rng.binomial(Wt[1:], q[None, :])
    return Wh

def hit_weights_fixed(Wt, h, noise, u):
    """X1-style fixed coin: one uniform per task, held fixed across resamples (for the CI-width comparison only)."""
    fa, fr = noise
    acc = np.where(h, u >= fr, u < fa).astype(float) if (fa or fr) else h.astype(float)
    return Wt * acc

def casc_sums(Wt, Wh, sp, cp, so, co, cmiss=None):
    """Verify-then-escalate cascade (eq:accounting): V = H sp + (1-H) so, D = cp + (1-H) cmiss (cmiss = co: fallback = reference
    restores the reference's cost; hedged seconds pass cmiss = max(cp,co)-cp).  All quantities are weighted sums; row 0 is the
    point estimate, rows 1.. the paired bootstrap.  Returns the primitives of thm:surplus / eq:floor / app:matched-reference / prop:threshold."""
    if cmiss is None:
        cmiss = co
    Wm = Wt - Wh
    Xw = np.column_stack([so, co, cp])                       # reference value, reference cost, primary bill (paid on every task)
    Xh = np.column_stack([sp, so, co, cp, (sp < so).astype(float)])
    Xm = np.column_stack([so, cmiss])
    Mw = Wt @ Xw; Mh = Wh @ Xh; Mm = Wm @ Xm
    n = Wt.sum(1).astype(float); nh = Wh.sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        Sso, Sco, Scp = Mw[:, 0], Mw[:, 1], Mw[:, 2]
        SV = Mh[:, 0] + Mm[:, 0]; SD = Scp + Mm[:, 1]
        out = dict(n=n, nh=nh, p=nh / n, v_h=_div(Mh[:, 0], nh), vbar=Sso / n, cbar=Sco / n, vbar_h=_div(Mh[:, 1], nh), cbar_h=_div(Mh[:, 2], nh),
                   a=Scp / n, a_h=_div(Mh[:, 3], nh), EV=SV / n, ED=SD / n, eta=_div(SV, SD), eta_B=_div(Sso, Sco), e=Mh[:, 4] / n,
                   SV=SV, SD=SD, Sso=Sso, Sco=Sco, Shcp=Mh[:, 3], Smco=Mm[:, 1])
        out['Q'] = out['EV'] - out['vbar']; out['S'] = out['cbar'] - out['ED']
    return out

def ci(v):
    v = np.asarray(v, float); w = v[1:][np.isfinite(v[1:])]
    if len(w) == 0:
        return [float(v[0]), float('nan'), float('nan')]
    return [float(v[0]), float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))]

def excl0(t):
    return bool(np.isfinite(t[1]) and np.isfinite(t[2]) and (t[1] > 0 or t[2] < 0))

def sgn(x):
    return 1 if x >= 0 else -1

# ---- criteria (row-wise on the (B+1) primitive vectors; the value shift is already inside the score arrays) ----
def crit_floor(st, lam):
    """Homogeneous criterion: eq:floor (lam = 'etaB'): sign of p - (vbar/v_h)(a/cbar), returned multiplied by v_h cbar > 0;
    at an exogenous price: p(lam cbar + v_h - vbar) - lam a, i.e. prop:threshold with the reference's unconditional means (b, v_b) = (cbar, vbar).
    p = 0 -> certain loss of the primary bill a (X1 convention)."""
    p, vh, vb, cb, a = st['p'], np.nan_to_num(st['v_h']), st['vbar'], st['cbar'], st['a']
    if lam == 'etaB':
        return np.where(st['nh'] > 0, p * vh * cb - vb * a, -vb * a)
    return np.where(st['nh'] > 0, p * (lam * cb + vh - vb) - lam * a, -lam * a)

def crit_matched(st, lam):
    """State-matched condition eq:matched-reference-condition p(v_h - vbar_h) + eta_B(p cbar_h - a) (lam = 'etaB'); at an exogenous
    price the correlated-reference threshold p W_lam - lam a with W_lam = lam cbar_h + v_h - vbar_h (app:matched-reference)."""
    p, vh, vbh, cbh, a = st['p'], np.nan_to_num(st['v_h']), np.nan_to_num(st['vbar_h']), np.nan_to_num(st['cbar_h']), st['a']
    if lam == 'etaB':
        etaB = st['vbar'] / st['cbar']
        return np.where(st['nh'] > 0, p * (vh - vbh) + etaB * (p * cbh - a), -etaB * a)
    return np.where(st['nh'] > 0, p * (lam * cbh + vh - vbh) - lam * a, -lam * a)

def realized(st, lam):
    """Realized comparison on the same tasks: eta - eta_B (ratio form) or E[V - lam D] - E[Y_B - lam C_B] (exogenous price)."""
    if lam == 'etaB':
        return st['SV'] / st['SD'] - st['Sso'] / st['Sco']
    return ((st['SV'] - lam * st['SD']) - (st['Sso'] - lam * st['Sco'])) / st['n']

def r_hom(st):
    """eq:normalized (pG+1-p)/(1/P+1-p), G = v_h/vbar, P = cbar/a (matched hypothesis b = cbar, v_m = vbar)."""
    p = st['p']; G = np.nan_to_num(st['v_h']) / st['vbar']; P = st['cbar'] / st['a']
    return (p * G + 1 - p) / (1 / P + 1 - p)

def setting_arrays(ix, m, ref, basis, shift=0.0):
    """Primary and reference outcome arrays on tasks ix: primary m, reference policy ref (also the fallback), cost basis; scores shifted by +shift."""
    R = RES[basis]; rm = REFS[ref][ix]
    return Sv[ix, col[m]] + shift, R[ix, col[m]], Sv[ix, rm] + shift, R[ix, rm]

def eval_cascade_setting(ref, basis, lam, noise, shift, sid, lam_list_c=None, fixed_coin=False):
    """Conclusions (a)-(d) at one setting.  Predictions from probing primitives, realized values from id_test (OUT-OF-SAMPLE).
    lam: 'etaB' (ratio criteria; (c) then uses the numeric probing eta_B of the reference in this basis) or an exogenous price."""
    rng = np.random.default_rng([SEED, 7, sid])
    rng_u = np.random.default_rng([SEED, 11, sid])
    out = dict(ref=ref, basis=basis, lam=lam, noise=list(noise), shift=shift)
    # ---------------- (a), (b): 7 primaries x 2 theta on the pooled population
    ixp, ixt = IDX[('probing', 'all', 'all')], IDX[('id_test', 'all', 'all')]
    Wp, Wt = W[('probing', 'all', 'all')], W[('id_test', 'all', 'all')]
    rows_ab = []; rel_rows = []
    for m in PRIMARIES:
        for th in THETAS:
            hp, ht = HIT[th][ixp, col[m]], HIT[th][ixt, col[m]]
            sp, cp, so, co = setting_arrays(ixp, m, ref, basis, shift); stp = casc_sums(Wp, hit_weights(Wp, hp, noise, rng), sp, cp, so, co)
            sp, cp, so, co = setting_arrays(ixt, m, ref, basis, shift); Wh_t = hit_weights(Wt, ht, noise, rng); stt = casc_sums(Wt, Wh_t, sp, cp, so, co)
            pf, pm, re = crit_floor(stp, lam), crit_matched(stp, lam), realized(stt, lam)
            rci = ci(re)
            row = dict(primary=m, theta=th, p_prob=float(stp['p'][0]), p_test=float(stt['p'][0]), v_h=float(stp['v_h'][0]), vbar=float(stp['vbar'][0]),
                       vbar_h_over_vbar=float(stp['vbar_h'][0] / stp['vbar'][0]) if stp['nh'][0] > 0 else float('nan'),
                       cbar_h_over_cbar=float(stp['cbar_h'][0] / stp['cbar'][0]) if stp['nh'][0] > 0 else float('nan'),
                       a_over_cbar=float(stp['a'][0] / stp['cbar'][0]),
                       pred_floor=float(pf[0]), pred_matched=float(pm[0]), real=rci, decided=excl0(rci),
                       sign_floor=sgn(pf[0]), sign_matched=sgn(pm[0]), sign_real=sgn(re[0]),
                       eta_ratio=ci(stt['eta'] / stt['eta_B']), Q_test=float(stt['Q'][0]), S_test=float(stt['S'][0]),
                       vbar_test=float(stt['vbar'][0]), cbar_test=float(stt['cbar'][0]), etaB_test=float(stt['eta_B'][0]),
                       S_hom_test=float(stt['p'][0] * stt['cbar'][0] - stt['a'][0]),
                       S_matched_test=float(stt['p'][0] * np.nan_to_num(stt['cbar_h'][0]) - stt['a'][0]),
                       ci_width_real=float(rci[2] - rci[1]) if np.isfinite(rci[1]) else float('nan'), e_test=float(stt['e'][0]))
            if fixed_coin and (noise[0] or noise[1]):
                u = rng_u.random(len(ixt)); stf = casc_sums(Wt, hit_weights_fixed(Wt, ht, noise, u), sp, cp, so, co)
                rf = ci(realized(stf, lam)); row['ci_width_real_fixedcoin'] = float(rf[2] - rf[1]); row['real_fixedcoin'] = rf
            if lam == 'etaB':
                pr = r_hom(stp); me = stt['eta'] / stt['eta_B']; rel = pr / me - 1.0
                row.update(r_hom_pred=ci(pr), r_meas=ci(me), rel_err=ci(rel)); rel_rows.append(rel)
            rows_ab.append(row)
    dec = [r for r in rows_ab if r['decided']]
    n_dec = len(dec)
    agree_f = sum(r['sign_floor'] == r['sign_real'] for r in dec); agree_m = sum(r['sign_matched'] == r['sign_real'] for r in dec)
    fw = sum((r['sign_floor'] > 0) and (r['sign_real'] < 0) for r in dec); fl = sum((r['sign_floor'] < 0) and (r['sign_real'] > 0) for r in dec)
    fw_m = sum((r['sign_matched'] > 0) and (r['sign_real'] < 0) for r in dec); fl_m = sum((r['sign_matched'] < 0) and (r['sign_real'] > 0) for r in dec)
    out['a'] = dict(n=len(rows_ab), n_decided=n_dec, agree_floor=agree_f, agree_matched=agree_m, floor_false_wins=fw, floor_false_losses=fl,
                    matched_false_wins=fw_m, matched_false_losses=fl_m,
                    agree_floor_all=sum(r['sign_floor'] == r['sign_real'] for r in rows_ab), agree_matched_all=sum(r['sign_matched'] == r['sign_real'] for r in rows_ab),
                    margin=(agree_m - agree_f) / n_dec if n_dec else float('nan'),
                    holds=(n_dec > 0 and agree_m > agree_f and fl == 0), decided=n_dec >= 5,
                    naive_cascade_wins=float(np.mean([r['sign_real'] >= 0 for r in dec])) if n_dec else float('nan'),
                    n_realized_wins_decided=int(sum(r['sign_real'] > 0 for r in dec)), n_realized_losses_decided=int(sum(r['sign_real'] < 0 for r in dec)),
                    all_same_realized_sign=bool(n_dec > 0 and (all(r['sign_real'] > 0 for r in dec) or all(r['sign_real'] < 0 for r in dec))))
    if lam == 'etaB':
        RL = np.vstack(rel_rows)                                  # (14, B+1)
        med = np.median(RL, axis=0); mci = ci(med)
        out['b'] = dict(n=len(rows_ab), median_rel_err=mci, n_over=int((RL[:, 0] > 0).sum()), n_over_decided=int(sum(r['rel_err'][1] > 0 for r in rows_ab)),
                        n_under_decided=int(sum(r['rel_err'][2] < 0 for r in rows_ab)), margin=float(mci[0]), holds=bool(mci[0] > 0), decided=excl0(mci),
                        mean_abs_rel_err=float(np.abs(RL[:, 0]).mean()))
    else:
        out['b'] = dict(margin=float('nan'), holds=None, decided=False, na=True)
    out['rows_ab'] = rows_ab
    # ---------------- (c), (d): dim9 + cell21 cells x 2 cheap primaries x 2 theta
    lam_c = lam_list_c if lam_list_c is not None else [lam]
    rows_c = []; rows_d = []
    for part in ['dim9', 'cell21']:
        for cname, _ in CELLS[part]:
            ixp, ixt = IDX[('probing', part, cname)], IDX[('id_test', part, cname)]; Wp, Wt = W[('probing', part, cname)], W[('id_test', part, cname)]
            for m in CHEAPS:
                for th in THETAS:
                    hp, ht = HIT[th][ixp, col[m]], HIT[th][ixt, col[m]]
                    sp, cp, so, co = setting_arrays(ixp, m, ref, basis, shift); stp = casc_sums(Wp, hit_weights(Wp, hp, noise, rng), sp, cp, so, co)
                    sp, cp, so, co = setting_arrays(ixt, m, ref, basis, shift); stt = casc_sums(Wt, hit_weights(Wt, ht, noise, rng), sp, cp, so, co)
                    stf = None
                    if fixed_coin and (noise[0] or noise[1]):
                        u = rng_u.random(len(ixt)); stf = casc_sums(Wt, hit_weights_fixed(Wt, ht, noise, u), sp, cp, so, co)
                    for lv in lam_c:
                        lnum = float(stp['eta_B'][0]) if lv == 'etaB' else float(lv)      # (c) is stated at an exogenous price: eta_B(probing) is the ratio-consistent one
                        pred = crit_floor(stp, lnum); re = realized(stt, lnum); rci = ci(re)
                        row = dict(part=part, cell=cname, cheap=m, theta=th, lam=lv, lam_num=lnum, n_test=int(len(ixt)), p_prob=float(stp['p'][0]), p_test=float(stt['p'][0]),
                                   W=float(lnum * stp['cbar'][0] + np.nan_to_num(stp['v_h'][0]) - stp['vbar'][0]), pred=float(pred[0]), real=rci, decided=excl0(rci),
                                   sign_pred=sgn(pred[0]), sign_real=sgn(re[0]), price_rule=sgn(stp['p'][0] - stp['a'][0] / stp['cbar'][0]),
                                   ci_width_real=float(rci[2] - rci[1]) if np.isfinite(rci[1]) else float('nan'))
                        if stf is not None:
                            rf = ci(realized(stf, lnum)); row['ci_width_real_fixedcoin'] = float(rf[2] - rf[1])
                        rows_c.append(row)
                    # (d): prop:voi band with s, a, b from probing; realized per-task costs on id_test (values are not compared: value-preserving hypothesis)
                    s_, a_, b_ = float(stp['p'][0]), float(stp['a'][0]), float(stp['cbar'][0])
                    n_t = stt['n']
                    Ccasc = stt['SD'] / n_t; Cdir = stt['Sco'] / n_t; Cinfo0 = (stt['Shcp'] + stt['Smco']) / n_t
                    if stf is not None:
                        Ccasc_f = stf['SD'] / n_t; Cdir_f = stf['Sco'] / n_t; Cinfo0_f = (stf['Shcp'] + stf['Smco']) / n_t
                    for k in KS_USD:
                        kb = k * K_SCALE[basis]
                        nonempty = (b_ > a_ > 0) and (kb < a_ * (b_ - a_) / b_)
                        lo, hi = (kb / (b_ - a_), 1 - kb / a_) if (b_ > a_ > 0) else (float('inf'), float('-inf'))
                        pred_in = bool(nonempty and lo < s_ < hi)
                        mrg = (Cinfo0 + kb) - np.minimum(Ccasc, Cdir); mci = ci(mrg)
                        row = dict(part=part, cell=cname, cheap=m, theta=th, k_usd=k, k_basis=kb, s=s_, a=a_, b=b_, band_lo=lo, band_hi=hi, band_nonempty=bool(nonempty),
                                   pred_info=pred_in, info_minus_best=mci, decided=excl0(mci), real_info=bool(mci[0] < 0),
                                   ci_width_real=float(mci[2] - mci[1]) if np.isfinite(mci[1]) else float('nan'))
                        if stf is not None:
                            mf = ci((Cinfo0_f + kb) - np.minimum(Ccasc_f, Cdir_f)); row['ci_width_real_fixedcoin'] = float(mf[2] - mf[1])
                        rows_d.append(row)
    out['c'] = {}
    for lv in lam_c:
        rr = [r for r in rows_c if r['lam'] == lv]; dec = [r for r in rr if r['decided']]; nd = len(dec)
        ag = sum(r['sign_pred'] == r['sign_real'] for r in dec) / nd if nd else float('nan')
        out['c'][str(lv)] = dict(n=len(rr), n_decided=nd, agree_decided=ag, margin=ag - 0.90 if nd else float('nan'), holds=(nd >= 10 and ag >= 0.90), decided=nd >= 10,
                                 naive_always_attempt=float(np.mean([r['sign_real'] >= 0 for r in dec])) if nd else float('nan'),
                                 naive_price_rule=float(np.mean([r['price_rule'] == r['sign_real'] for r in dec])) if nd else float('nan'),
                                 agree_all=float(np.mean([r['sign_pred'] == r['sign_real'] for r in rr])),
                                 n_pred_attempt_real_direct=int(sum((r['sign_pred'] > 0) and (r['sign_real'] < 0) for r in dec)),
                                 n_pred_direct_real_attempt=int(sum((r['sign_pred'] < 0) and (r['sign_real'] > 0) for r in dec)))
    dec = [r for r in rows_d if r['decided']]; nd = len(dec)
    ag = sum(r['pred_info'] == r['real_info'] for r in dec) / nd if nd else float('nan')
    out['d'] = dict(n=len(rows_d), n_decided=nd, agree_decided=ag, margin=ag - 0.95 if nd else float('nan'), holds=(nd >= 10 and ag >= 0.95), decided=nd >= 10,
                    naive_band_nonempty=float(np.mean([r['band_nonempty'] == r['real_info'] for r in dec])) if nd else float('nan'),
                    n_pred_in_real_out=int(sum(r['pred_info'] and not r['real_info'] for r in dec)), n_pred_out_real_in=int(sum((not r['pred_info']) and r['real_info'] for r in dec)),
                    n_real_info=int(sum(r['real_info'] for r in dec)))
    out['rows_c'] = rows_c; out['rows_d'] = rows_d
    return out

# =========================================================================================== 3. record module (X4 estimators; uninformed policy = per-task reference)
_rng_s = np.random.default_rng([SEED, 4])
TID = _rng_s.choice(IDX_TEST, size=(SEEDS_STREAM, N_ROUNDS), replace=True).astype(np.int32)   # i.i.d. real id_test tasks
JC = cell21_of[TID].astype(np.int8)
EXPLORE = _rng_s.random((SEEDS_STREAM, N_ROUNDS)) < EPS
MEXP = _rng_s.integers(0, M, size=(SEEDS_STREAM, N_ROUNDS)).astype(np.int8)
U_VER = _rng_s.random((SEEDS_STREAM, N_ROUNDS))
LATE_SL = slice(N_ROUNDS - LATE, N_ROUNDS)
NU_PROB = np.bincount(cell21_of[IDX_PROB], minlength=21) / len(IDX_PROB)
NU_TEST = np.bincount(cell21_of[IDX_TEST], minlength=21) / len(IDX_TEST)
_rng_wb = np.random.default_rng([SEED, 5])
WB = np.vstack([np.ones((1, len(IDX_TEST))), _rng_wb.multinomial(len(IDX_TEST), np.full(len(IDX_TEST), 1.0 / len(IDX_TEST)), size=B).astype(float)])   # (B+1, 2919)

def make_lw_index(j):
    """Precomputed sort order for 'last write to the same cell strictly before n' (cells j are fixed across write configurations)."""
    Sd, N = j.shape
    s_idx = np.repeat(np.arange(Sd), N); n_idx = np.tile(np.arange(N), Sd)
    key = s_idx.astype(np.int64) * 64 + j.ravel().astype(np.int64)
    order = np.lexsort((n_idx, key)); key_s = key[order]
    newg = np.r_[True, key_s[1:] != key_s[:-1]]
    pos = np.arange(Sd * N); gstart = np.maximum.accumulate(np.where(newg, pos, 0))
    inv = np.empty(Sd * N, np.int64); inv[order] = np.arange(Sd * N)
    return dict(order=order, newg=newg, gstart=gstart, inv=inv, shape=(Sd, N))

def last_write_before(acc, lwi):
    order, newg, gstart, inv = lwi['order'], lwi['newg'], lwi['gstart'], lwi['inv']
    Lf = len(order); pos = np.arange(Lf)
    hw_s = acc.ravel()[order]
    cum = np.maximum.accumulate(np.where(hw_s, pos, -1))
    last_incl = np.where(cum >= gstart, cum, -1)
    last_excl = np.full(Lf, -1); last_excl[1:] = last_incl[:-1]; last_excl[newg] = -1
    lw_s = np.where(last_excl >= 0, order[np.maximum(last_excl, 0)], -1)
    return lw_s[inv].reshape(lwi['shape'])

LWI = make_lw_index(JC)
HITM = {th: HIT[th][TID, MEXP] for th in THETAS}                       # did the explored model truly hit (per round)

def record_states(theta, noise, u, lwi=LWI):
    """Write mechanism of X4: on an exploration round the logger fields a uniform model; the (noisy) verifier verdict on that real
    outcome decides the write (informative if the model truly hit, erroneous if falsely accepted); last write wins; no expiration.
    Returns the record state (0 = uninformed, 1 = informative, 2 = erroneous) and the recorded model at every round."""
    fa, fr = noise
    hm = HITM[theta]
    acc = EXPLORE & (np.where(hm, u >= fr, u < fa) if (fa or fr) else hm)
    wtype = np.where(hm, 1, 2).astype(np.int8)
    lw = last_write_before(acc, lwi); has = lw >= 0
    st = np.where(has, wtype.ravel()[np.maximum(lw, 0)], 0).astype(np.int8)
    recm = np.where(has, MEXP.ravel()[np.maximum(lw, 0)], -1).astype(np.int8)
    return st, recm

TID_L = TID[:, LATE_SL]; EXPL_L = EXPLORE[:, LATE_SL]; MEXP_L = MEXP[:, LATE_SL]
TP_L = test_pos[TID_L].ravel()
CNT_T = np.bincount(TP_L, minlength=len(IDX_TEST)).astype(float)       # late-window rounds per id_test task

def fielded(st_l, recm_l, cold):
    f_I = np.where(EXPL_L, MEXP_L, np.where(st_l > 0, recm_l, cold[TID_L]))
    f_U = np.where(EXPL_L, MEXP_L, cold[TID_L])
    return f_I, f_U

def task_sums(x):
    return np.bincount(TP_L, weights=np.asarray(x, float).ravel(), minlength=len(IDX_TEST))

# probing primitives of the record model (P_jm, V_jm, C_jm per basis) -- estimated on probing only
_cj_prob = cell21_of[IDX_PROB]; _cnt_prob = np.bincount(_cj_prob, minlength=21).astype(float)
def _cellmean(x):
    return np.bincount(_cj_prob, weights=x, minlength=21) / _cnt_prob
P_PROB = {th: np.column_stack([_cellmean(HIT[th][IDX_PROB, m].astype(float)) for m in range(M)]) for th in THETAS}
V_PROB = np.column_stack([_cellmean(Sv[IDX_PROB, m]) for m in range(M)])
C_PROB = {b: np.column_stack([_cellmean(RES[b][IDX_PROB, m]) for m in range(M)]) for b in BASES}
C_PROB['unit'] = np.ones((21, M))
VCOLD_PROB = {r: _cellmean(Sv[IDX_PROB, REFS[r][IDX_PROB]]) for r in REF_TYPES}
CCOLD_PROB = {b: {r: _cellmean(RES[b][IDX_PROB, REFS[r][IDX_PROB]]) for r in REF_TYPES} for b in BASES}
CCOLD_PROB['unit'] = {r: np.ones(21) for r in REF_TYPES}

def rec_primitives(theta, noise, cold, basis, lam='etaU', shift=0.0, nu=NU_PROB):
    """Sec. 5 primitives from probing (OUT-OF-SAMPLE for the id_test stream): omega_j, psi_j, r_j, alpha_j (delta = 0), v_j^s, c_j^s,
    eta_U (eq:cold), g_j^s at price lam (eq:record-surplus; lam = eta_U or an exogenous price), G_j (eq:stationary-record-gain),
    the erroneous fraction psi_j/r_j and the threshold g^I/(g^I-g^W) of eq:noise-threshold.  Values shifted by +shift."""
    fa, fr = noise
    P = P_PROB[theta]; V = V_PROB + shift; Cc = C_PROB[basis]
    omega = EPS * (1 - fr) * P.mean(1); psi = EPS * fa * (1 - P).mean(1); r = omega + psi
    sP = P.sum(1); sQ = (1 - P).sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        qI = np.where(sP[:, None] > 0, P / np.where(sP[:, None] > 0, sP[:, None], 1), np.nan)
        qW = np.where(sQ[:, None] > 0, (1 - P) / np.where(sQ[:, None] > 0, sQ[:, None], 1), np.nan)
    VI = np.nansum(qI * V, 1); CI_ = np.nansum(qI * Cc, 1); VW = np.nansum(qW * V, 1); CW = np.nansum(qW * Cc, 1)
    VI[sP == 0] = np.nan; CI_[sP == 0] = np.nan; VW[sQ == 0] = np.nan; CW[sQ == 0] = np.nan
    Vcold = VCOLD_PROB[cold] + shift; Ccold = CCOLD_PROB[basis][cold]
    vU = (1 - EPS) * Vcold + EPS * V.mean(1); cU = (1 - EPS) * Ccold + EPS * Cc.mean(1)
    vI = (1 - EPS) * VI + EPS * V.mean(1); cI = (1 - EPS) * CI_ + EPS * Cc.mean(1)
    vW = (1 - EPS) * VW + EPS * V.mean(1); cW = (1 - EPS) * CW + EPS * Cc.mean(1)
    eta_U = float((nu * vU).sum() / (nu * cU).sum())
    price = eta_U if lam == 'etaU' else float(lam)
    gI = (vI - vU) - price * (cI - cU); gW = (vW - vU) - price * (cW - cU)
    alpha = 1 - nu * r
    num = np.where(omega > 0, omega * np.nan_to_num(gI), 0.0) + np.where(psi > 0, psi * np.nan_to_num(gW), 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        G = np.where(r > 0, nu * num / (1 - alpha), 0.0)
        err = np.where(r > 0, psi / r, np.nan)
        thr = gI / (gI - gW)
        # stationary resource saving weight: d(sum nu G)/d(price) = -S_rec ... written so that sumG(c) - sumG(0) = (c / cbar_U) * S_rec
        S_rec = -float((nu * np.where(r > 0, nu * (np.where(omega > 0, omega * np.nan_to_num(cI - cU), 0.0) + np.where(psi > 0, psi * np.nan_to_num(cW - cU), 0.0)) / (1 - alpha), 0.0)).sum())
    hyp = np.isfinite(gI) & np.isfinite(gW) & (gI > 0) & (gW < 0)
    return dict(omega=omega, psi=psi, r=r, alpha=alpha, vU=vU, cU=cU, vI=vI, cI=cI, vW=vW, cW=cW, eta_U=eta_U, price=price, gI=gI, gW=gW, G=G,
                sumG=float((nu * G).sum()), err=err, thr=thr, hyp=hyp, S_rec=S_rec, cbar_U=float((nu * cU).sum()),
                dcI=cI - cU, dcW=cW - cU)

def rec_realized_columns(theta, noise, u=U_VER, colds=REF_TYPES, bases=BASES):
    """Late-window per-task sums for the informed (I) and uninformed (U) policies under every (uninformed policy, basis).
    Returns X (2919, ncol) with columns [V_I, D_I, V_U, D_U] per (cold, basis) and the column index."""
    st, recm = record_states(theta, noise, u)
    st_l, recm_l = st[:, LATE_SL], recm[:, LATE_SL]
    cols = {}; blocks = []
    for r in colds:
        f_I, f_U = fielded(st_l, recm_l, REFS[r])
        VI = task_sums(Sv[TID_L, f_I]); VU = task_sums(Sv[TID_L, f_U])
        for b in bases:
            R = RES[b]
            cols[(r, b)] = len(blocks)
            blocks += [VI, task_sums(R[TID_L, f_I]), VU, task_sums(R[TID_L, f_U])]
    return np.column_stack(blocks), cols, st_l

def rec_stats_from_sums(Y, Ycnt, Ycell, Ycell_cnt, j0, lam_mode, shift, eta_U):
    """Y: (B+1, 4) columns [V_I, D_I, V_U, D_U] (bootstrap rows), Ycnt: (B+1,) rounds; Ycell: (21, B+1, 2) [dV, dD] per cell, Ycell_cnt (21, B+1).
    Returns realized Delta (eta or J_lam) rows and per-cell paired gain-difference rows at the prediction's price."""
    VI, DI, VU, DU = Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3]
    with np.errstate(invalid='ignore', divide='ignore'):
        if lam_mode == 'etaU':
            delta = (VI + shift * Ycnt) / DI - (VU + shift * Ycnt) / DU
            price = eta_U
        else:
            price = float(lam_mode); delta = ((VI - price * DI) - (VU - price * DU)) / Ycnt
        cell = (Ycell[:, :, 0] - price * Ycell[:, :, 1]) / np.where(Ycell_cnt > 0, Ycell_cnt, np.nan)
    return delta, cell

CELL_MASK_T = [(cell21_of[IDX_TEST] == j) for j in range(21)]
def run_record_module(rec_settings):
    """Point estimates and fixed-dynamics task bootstraps (X4-style: dynamics fixed, epochs re-weighted by task) for every write
    configuration (theta x noise) and every (uninformed policy, basis, price, shift) in rec_settings.  Returns per-config rows for (e) and (f)."""
    rows_e = []; rows_f = []
    ycnt = WB @ CNT_T
    cell_cnt = np.stack([WB[:, mk] @ CNT_T[mk] for mk in CELL_MASK_T])                     # (21, B+1)
    for th in THETAS:
        for noise in NOISES:
            X, cols, _ = rec_realized_columns(th, noise)
            Yall = WB @ X                                                                     # (B+1, ncol)
            for (r, b), c0 in cols.items():
                Y = Yall[:, c0:c0 + 4]
                dV = X[:, c0] - X[:, c0 + 2]; dD = X[:, c0 + 1] - X[:, c0 + 3]
                Ycell = np.stack([np.column_stack([WB[:, mk] @ dV[mk], WB[:, mk] @ dD[mk]]) for mk in CELL_MASK_T])   # (21, B+1, 2)
                for (lam_mode, shift) in rec_settings.get((r, b), []):
                    pr = rec_primitives(th, noise, r, b, lam='etaU' if lam_mode == 'etaU' else lam_mode, shift=shift)
                    delta, cell = rec_stats_from_sums(Y, ycnt, Ycell, cell_cnt, 0, lam_mode, shift, pr['eta_U'])
                    dci = ci(delta)
                    rows_e.append(dict(theta=th, beta_fa=noise[0], beta_fr=noise[1], cold=r, basis=b, lam=str(lam_mode), shift=shift, eta_U=pr['eta_U'],
                                       sumG=pr['sumG'], pred_sign=sgn(pr['sumG']), real=dci, real_sign=sgn(dci[0]), decided=excl0(dci),
                                       ci_width=float(dci[2] - dci[1]) if np.isfinite(dci[1]) else float('nan'),
                                       eta_I=float(Y[0, 0] / Y[0, 1]), eta_U_late=float(Y[0, 2] / Y[0, 3]), S_rec=pr['S_rec'], cbar_U=pr['cbar_U'],
                                       n_cells_hyp=int(pr['hyp'].sum())))
                    for j in np.where(pr['hyp'])[0]:
                        cci = ci(cell[j]); 
                        rows_f.append(dict(theta=th, beta_fa=noise[0], beta_fr=noise[1], cold=r, basis=b, lam=str(lam_mode), shift=shift, cell=CELLS21[j],
                                           gI=float(pr['gI'][j]), gW=float(pr['gW'][j]), err=float(pr['err'][j]), thr=float(pr['thr'][j]),
                                           pred_pos=bool(pr['err'][j] < pr['thr'][j]), real=cci, real_pos=bool(cci[0] > 0), conclusive=excl0(cci),
                                           ci_width=float(cci[2] - cci[1]) if np.isfinite(cci[1]) else float('nan'), G=float(pr['G'][j])))
        log(f'record module theta={th}: {len(NOISES)} noise settings done')
    return rows_e, rows_f

# ---- coin-re-drawn bootstrap for the record stream (worker: re-simulates the record process with fresh verifier coins per resample)
def _redraw_worker(args):
    th, noise, eta_U, nres, seed = args
    rng = np.random.default_rng(seed)
    hm = HITM[th]; fa, fr = noise
    n_t = len(IDX_TEST)
    cold = REFS['opus']
    f_U = np.where(EXPL_L, MEXP_L, cold[TID_L]); VU = task_sums(Sv[TID_L, f_U]); DU = task_sums(Cv[TID_L, f_U])
    out_d = np.empty(nres); out_c = np.empty((nres, 21))
    for it in range(nres):
        u = rng.random(hm.shape)
        acc = EXPLORE & np.where(hm, u >= fr, u < fa)
        lw = last_write_before(acc, LWI); has = lw >= 0
        st = np.where(has, np.where(hm, 1, 2).ravel()[np.maximum(lw, 0)], 0)[:, LATE_SL]
        recm = np.where(has, MEXP.ravel()[np.maximum(lw, 0)], -1)[:, LATE_SL]
        f_I = np.where(EXPL_L, MEXP_L, np.where(st > 0, recm, cold[TID_L]))
        VI = task_sums(Sv[TID_L, f_I]); DI = task_sums(Cv[TID_L, f_I])
        w = rng.multinomial(n_t, np.full(n_t, 1.0 / n_t)).astype(float)
        out_d[it] = (w @ VI) / (w @ DI) - (w @ VU) / (w @ DU)
        dsur = (VI - VU) - eta_U * (DI - DU)
        for j, mk in enumerate(CELL_MASK_T):
            cw = w[mk] @ CNT_T[mk]
            out_c[it, j] = (w[mk] @ dsur[mk]) / cw if cw > 0 else np.nan
    return (th, noise), out_d, out_c

# =========================================================================================== 4. settings (one-at-a-time around the baseline + reference x basis)
def mk(axis, name, ref='opus', basis='dollars', lam='etaB', noise=(0.0, 0.0), shift=0.0):
    return dict(axis=axis, name=name, ref=ref, basis=basis, lam=lam, noise=tuple(noise), shift=shift)

SETTINGS = [mk('baseline', 'baseline')]
SETTINGS += [mk('reference', r, ref=r) for r in REF_TYPES[1:]]
SETTINGS += [mk('lambda', f'lam={lv:g}', lam=lv) for lv in LAMS_EXO]
SETTINGS += [mk('basis', b, basis=b) for b in BASES[1:]]
SETTINGS += [mk('noise', f'fa={nz[0]:g}_fr={nz[1]:g}', noise=nz) for nz in NOISES if nz != (0.0, 0.0)]
SETTINGS += [mk('shift', f'c={c:g}', shift=c) for c in SHIFTS]
SETTINGS += [mk('ref_x_basis', f'{r}|{b}', ref=r, basis=b) for r in REF_TYPES[1:] for b in BASES[1:]]
if QUICK:
    SETTINGS = [s for s in SETTINGS if s['axis'] != 'noise' or s['noise'] in [(0.1, 0.1), (0.4, 0.4), (0.0, 0.2)]]
AXES = ['reference', 'lambda', 'basis', 'noise', 'shift']
CONCS = ['a', 'b', 'c', 'd', 'e', 'e_pred', 'f']
CONC_LABEL = {'a': '(a) floor optimistic, fewer signs than matched', 'b': '(b) eq:normalized over-predicts', 'c': '(c) prop:threshold sign >= 90%',
              'd': '(d) VOI band >= 95%', 'e': '(e) records improve efficiency vs reference', 'e_pred': "(e') sum nu G predicts the record sign",
              'f': '(f) noise threshold predicts record decisions'}
NA = {('b', 'lambda'), ('d', 'lambda')}                 # ratio-only / cost-only criteria: the price axis does not enter them

def rec_lam(lam):
    return 'etaU' if lam == 'etaB' else lam

def conc_e_from_rows(rows_e, ref, basis, lam, shift, noise):
    rr = [r for r in rows_e if r['cold'] == ref and r['basis'] == basis and r['lam'] == str(rec_lam(lam)) and r['shift'] == shift and (r['beta_fa'], r['beta_fr']) == tuple(noise)]
    dec = [r for r in rr if r['decided']]
    out = dict(n_theta=len(rr), n_decided=len(dec), per_theta={str(r['theta']): dict(real=r['real'], sumG=r['sumG'], decided=r['decided'], eta_I=r['eta_I'], eta_U_late=r['eta_U_late']) for r in rr})
    out['margin'] = float(min(r['real'][0] for r in dec)) if dec else float('nan')
    out['holds'] = bool(dec and all(r['real_sign'] > 0 for r in dec)); out['decided'] = len(dec) > 0
    out['pred_holds'] = bool(rr and all(r['pred_sign'] > 0 for r in rr))
    out['pred_agree'] = float(np.mean([r['pred_sign'] == r['real_sign'] for r in dec])) if dec else float('nan')
    out['naive_records_help'] = float(np.mean([r['real_sign'] > 0 for r in dec])) if dec else float('nan')
    return out

def conc_f_from_rows(rows_f, ref, basis, lam, shift, noise=None, beta_fa=None):
    rr = [r for r in rows_f if r['cold'] == ref and r['basis'] == basis and r['lam'] == str(rec_lam(lam)) and r['shift'] == shift and (noise is None or (r['beta_fa'], r['beta_fr']) == tuple(noise))
          and (beta_fa is None or r['beta_fa'] == beta_fa)]
    con = [r for r in rr if r['conclusive']]
    acc = float(np.mean([r['pred_pos'] == r['real_pos'] for r in con])) if con else float('nan')
    tp = sum(r['pred_pos'] and r['real_pos'] for r in con); tn = sum((not r['pred_pos']) and (not r['real_pos']) for r in con)
    fp = sum(r['pred_pos'] and not r['real_pos'] for r in con); fn = sum((not r['pred_pos']) and r['real_pos'] for r in con)
    return dict(n_hyp=len(rr), n_conclusive=len(con), accuracy=acc, TP=tp, TN=tn, FP=fp, FN=fn, margin=acc - 0.90 if con else float('nan'),
                holds=bool(con and acc >= 0.90), decided=len(con) >= 5, one_sided=bool(con and (tn + fp == 0)),
                naive_always_positive=float(np.mean([r['real_pos'] for r in con])) if con else float('nan'),
                n_pred_negative=int(sum(not r['pred_pos'] for r in rr)), cells=sorted({r['cell'] for r in rr}))

def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_, bool)): return bool(o)
    if isinstance(o, np.ndarray): return jsonable(o.tolist())
    return o

def fmt(t, d=3):
    return f'{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]' if (isinstance(t, (list, tuple)) and np.isfinite(t[1])) else (f'{t[0]:.{d}f} [–]' if isinstance(t, (list, tuple)) else str(t))

def table(hdr, rows):
    return '| ' + ' | '.join(hdr) + ' |\n|' + '---|' * len(hdr) + '\n' + '\n'.join('| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows) + '\n'

# =========================================================================================== 5. main
def main():
    RESULTS = dict(experiment='X16_sensitivity', seed=SEED, n_bootstrap=B, n_bootstrap_redraw_records=B_REDRAW, n_random_reference_draws=N_RANDOM,
                   quick=QUICK, data_checks=DATA_CHECKS, references=REF_INFO, dimension_best_map=DIMBEST_MAP, k_scale=K_SCALE, opus_probing_mean_cost=OPUS_PROB_MEAN)
    # ------------------------------------------------------------------ 5a. cascade module over all settings
    CAS = {}
    for sid, s in enumerate(SETTINGS):
        CAS[s['name']] = eval_cascade_setting(s['ref'], s['basis'], s['lam'], s['noise'], s['shift'], sid, fixed_coin=(s['axis'] == 'noise'))
        if sid % 10 == 0:
            log(f"cascade setting {sid}/{len(SETTINGS)} {s['name']}")
    log('cascade module done')
    # hedged-seconds diagnostic for C1 (Opus reference): realized eta/eta_B of the 14 cascades per basis
    hedge = []
    ixt, Wt = IDX[('id_test', 'all', 'all')], W[('id_test', 'all', 'all')]
    for m in PRIMARIES:
        for th in THETAS:
            ht = HIT[th][ixt, col[m]]
            for b in BASES + ['seconds_hedged']:
                sp, cp, so, co = setting_arrays(ixt, m, 'opus', 'seconds' if b == 'seconds_hedged' else b)
                cmiss = (np.maximum(cp, co) - cp) if b == 'seconds_hedged' else None
                st = casc_sums(Wt, hit_weights(Wt, ht, (0.0, 0.0), None), sp, cp, so, co, cmiss=cmiss)
                r = ci(st['eta'] / st['eta_B'])
                hedge.append(dict(primary=m, theta=th, basis=b, eta_ratio=r, decided=excl0([r[0] - 1, r[1] - 1, r[2] - 1]), win=bool(r[1] > 1), loss=bool(r[2] < 1),
                                  a_over_cbar=float(st['a'][0] / st['cbar'][0])))
    RESULTS['c1_cascade_by_basis'] = {b: dict(n=14, wins_decided=int(sum(h['win'] for h in hedge if h['basis'] == b)), losses_decided=int(sum(h['loss'] for h in hedge if h['basis'] == b)),
                                            undecided=int(sum((not h['win']) and (not h['loss']) for h in hedge if h['basis'] == b)),
                                            median_eta_ratio=float(np.median([h['eta_ratio'][0] for h in hedge if h['basis'] == b])),
                                            min_a_over_cbar=float(min(h['a_over_cbar'] for h in hedge if h['basis'] == b))) for b in BASES + ['seconds_hedged']}
    RESULTS['c1_cascade_rows'] = hedge
    # ------------------------------------------------------------------ 5b. record module
    rec_settings = {}
    def add_rs(r, b, lam_mode, shift):
        rec_settings.setdefault((r, b), []).append((lam_mode, shift))
    for lam_mode in ['etaU'] + LAMS_EXO:
        for c in [0.0] + SHIFTS:
            add_rs('opus', 'dollars', lam_mode, c)
    for r in REF_TYPES[1:]:
        for b in BASES:
            add_rs(r, b, 'etaU', 0.0)
    for b in BASES[1:]:
        add_rs('opus', b, 'etaU', 0.0)
    for r in ['dimbest']:                                     # value-shift table needs the DimensionBest record criterion too
        for c in SHIFTS:
            add_rs(r, 'dollars', 'etaU', c)
    for k in rec_settings:
        rec_settings[k] = sorted(set(rec_settings[k]), key=str)
    rows_e, rows_f = run_record_module(rec_settings)
    RE = pd.DataFrame(rows_e); RF = pd.DataFrame(rows_f)
    log('record rows', len(RE), 'hypothesis-met cell rows', len(RF))
    # coin-re-drawn bootstrap (Opus, dollars, price eta_U, no shift) for every noisy write configuration
    jobs = []
    for th in THETAS:
        for k, nz in enumerate(NOISES):
            if nz == (0.0, 0.0) or (QUICK and nz not in [(0.1, 0.1), (0.4, 0.4), (0.0, 0.2)]):
                continue
            eta_U = float(RE[(RE.theta == th) & (RE.beta_fa == nz[0]) & (RE.beta_fr == nz[1]) & (RE.cold == 'opus') & (RE.basis == 'dollars') & (RE.lam == 'etaU') & (RE['shift'] == 0.0)].eta_U.iloc[0])
            jobs.append((th, nz, eta_U, B_REDRAW, int(np.random.default_rng([SEED, 9, int(th * 10), k]).integers(0, 2**31 - 1))))
    import multiprocessing as mp
    nproc = min(8, max(1, os.cpu_count() - 2))
    log(f'coin-re-drawn record bootstraps: {len(jobs)} configs x {B_REDRAW} resamples on {nproc} processes')
    with mp.get_context('spawn').Pool(nproc) as pool:
        redraw = pool.map(_redraw_worker, jobs)
    REDRAW = {}
    for (th, nz), dd, cc in redraw:
        fixed = RE[(RE.theta == th) & (RE.beta_fa == nz[0]) & (RE.beta_fr == nz[1]) & (RE.cold == 'opus') & (RE.basis == 'dollars') & (RE.lam == 'etaU') & (RE['shift'] == 0.0)].iloc[0]
        d_ci = [float(fixed['real'][0]), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
        cells = {}
        for j in range(21):
            v = cc[:, j]; v = v[np.isfinite(v)]
            frow = RF[(RF.theta == th) & (RF.beta_fa == nz[0]) & (RF.beta_fr == nz[1]) & (RF.cold == 'opus') & (RF.basis == 'dollars') & (RF.lam == 'etaU') & (RF['shift'] == 0.0) & (RF.cell == CELLS21[j])]
            if len(v) and len(frow):
                fr_ = frow.iloc[0]; rc = [float(fr_['real'][0]), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                cells[CELLS21[j]] = dict(real_redrawn=rc, real_fixed=fr_['real'], width_ratio=float((rc[2] - rc[1]) / (fr_['real'][2] - fr_['real'][1])) if fr_['real'][2] > fr_['real'][1] else float('nan'),
                                         conclusive_fixed=bool(fr_['conclusive']), conclusive_redrawn=excl0(rc), pred_pos=bool(fr_['pred_pos']))
        REDRAW[(th, nz)] = dict(theta=th, beta_fa=nz[0], beta_fr=nz[1], real_fixed=fixed['real'], real_redrawn=d_ci, decided_fixed=bool(fixed['decided']), decided_redrawn=excl0(d_ci),
                                width_ratio=float((d_ci[2] - d_ci[1]) / (fixed['real'][2] - fixed['real'][1])), sign_same=sgn(d_ci[0]) == sgn(fixed['real'][0]), cells=cells)
    log('re-drawn bootstraps done')
    # ------------------------------------------------------------------ 5c. per-setting conclusions and flips
    SUMM = []
    for s in SETTINGS:
        c = CAS[s['name']]
        rec_noise = s['noise']
        e = conc_e_from_rows(rows_e, s['ref'], s['basis'], s['lam'], s['shift'], rec_noise)
        f = conc_f_from_rows(rows_f, s['ref'], s['basis'], s['lam'], s['shift'], noise=(s['noise'] if s['axis'] == 'noise' else None))
        row = dict(axis=s['axis'], name=s['name'], ref=s['ref'], basis=s['basis'], lam=str(s['lam']), noise=list(s['noise']), shift=s['shift'])
        row['a'] = dict(margin=c['a']['margin'], holds=c['a']['holds'], decided=c['a']['decided'], detail=c['a'])
        row['b'] = dict(margin=c['b']['margin'], holds=c['b']['holds'], decided=c['b']['decided'], detail=c['b'], na=('b', s['axis']) in NA)
        cc = c['c'][str(s['lam'])]
        row['c'] = dict(margin=cc['margin'], holds=cc['holds'], decided=cc['decided'], detail=cc)
        row['d'] = dict(margin=c['d']['margin'], holds=c['d']['holds'], decided=c['d']['decided'], detail=c['d'], na=('d', s['axis']) in NA)
        row['e'] = dict(margin=e['margin'], holds=e['holds'], decided=e['decided'], detail=e)
        row['e_pred'] = dict(margin=(e['pred_agree'] - 1.0) if np.isfinite(e['pred_agree']) else float('nan'), holds=bool(np.isfinite(e['pred_agree']) and e['pred_agree'] >= 1.0), decided=e['decided'], detail={'pred_agree': e['pred_agree']})
        row['f'] = dict(margin=f['margin'], holds=f['holds'], decided=f['decided'], detail=f)
        SUMM.append(row)
    base = SUMM[0]
    BASE_HOLDS = {k: base[k]['holds'] for k in CONCS}
    for row in SUMM:
        for k in CONCS:
            na = row[k].get('na', False)
            row[k]['flip'] = (None if (na or not row[k]['decided'] or not base[k]['decided']) else bool(row[k]['holds'] != BASE_HOLDS[k]))
    # flip table (reference axis type-balanced: dimbest, oracle, random = fraction of draws)
    FLIP = {}
    for k in CONCS:
        FLIP[k] = {}
        for ax in AXES:
            rows = [r for r in SUMM if r['axis'] == ax]
            if (k, ax) in NA:
                FLIP[k][ax] = dict(frac=None, n=0, n_flip=0, n_undecided=0, na=True); continue
            if ax == 'reference':
                parts = []
                for typ in ['dimbest', 'oracle', 'random']:
                    rr = [r for r in rows if r['name'].startswith(typ)]
                    fl = [r[k]['flip'] for r in rr if r[k]['flip'] is not None]
                    parts.append(float(np.mean(fl)) if fl else float('nan'))
                FLIP[k][ax] = dict(frac=float(np.nanmean(parts)) if np.isfinite(parts).any() else None, by_type=dict(zip(['dimbest', 'oracle', 'random'], parts)),
                                   n=len(rows), n_flip=int(sum(bool(r[k]['flip']) for r in rows)), n_undecided=int(sum(r[k]['flip'] is None for r in rows)),
                                   frac_all_settings=float(np.mean([r[k]['flip'] for r in rows if r[k]['flip'] is not None])) if any(r[k]['flip'] is not None for r in rows) else None)
            else:
                fl = [r[k]['flip'] for r in rows if r[k]['flip'] is not None]
                FLIP[k][ax] = dict(frac=float(np.mean(fl)) if fl else None, n=len(rows), n_flip=int(sum(fl)), n_undecided=len(rows) - len(fl),
                                   flipped_settings=[r['name'] for r in rows if r[k]['flip']])
    # supplementary: (f) per beta_fa level pooled over beta_fr and theta (per (beta_fa, beta_fr) pair few cells are conclusive)
    F_BY_FA = {}
    for fa in BETAS:
        f_ = conc_f_from_rows(rows_f, 'opus', 'dollars', 'etaB', 0.0, beta_fa=fa)
        f_['flip'] = None if not f_['decided'] else bool(f_['holds'] != BASE_HOLDS['f'])
        F_BY_FA[str(fa)] = f_
    fl_fa = [v['flip'] for v in F_BY_FA.values() if v['flip'] is not None]
    FLIP['f']['noise']['by_beta_fa_level'] = dict(frac=float(np.mean(fl_fa)) if fl_fa else None, n=len(F_BY_FA), n_decided=len(fl_fa), flipped=[k for k, v in F_BY_FA.items() if v['flip']])
    RESULTS['f_by_beta_fa'] = F_BY_FA
    ROBUST = [k for k in CONCS if all((FLIP[k][ax]['frac'] in (None, 0.0)) for ax in AXES)]
    DEPENDS = {k: [ax for ax in AXES if FLIP[k][ax]['frac'] not in (None, 0.0)] for k in CONCS if k not in ROBUST}
    RESULTS.update(baseline_holds=BASE_HOLDS, settings=SUMM, flip_table=FLIP, robust=ROBUST, depends=DEPENDS)
    log('flip table', {k: {ax: (None if FLIP[k][ax]['frac'] is None else round(FLIP[k][ax]['frac'], 2)) for ax in AXES} for k in CONCS})
    # ------------------------------------------------------------------ 5d. C3: CI widths, re-drawn vs fixed coin
    C3 = dict(rows=[])
    for row in SUMM:
        if row['axis'] != 'noise':
            continue
        c = CAS[row['name']]; nz = tuple(row['noise']); low = max(nz) <= 0.1 + 1e-12
        wa = [r['ci_width_real'] / r['ci_width_real_fixedcoin'] for r in c['rows_ab'] if r.get('ci_width_real_fixedcoin', 0) > 0 and np.isfinite(r['ci_width_real'])]
        wc = [r['ci_width_real'] / r['ci_width_real_fixedcoin'] for r in c['rows_c'] if r.get('ci_width_real_fixedcoin', 0) > 0 and np.isfinite(r['ci_width_real'])]
        wd = [r['ci_width_real'] / r['ci_width_real_fixedcoin'] for r in c['rows_d'] if r.get('ci_width_real_fixedcoin', 0) > 0 and np.isfinite(r['ci_width_real'])]
        we = [REDRAW[(th, nz)]['width_ratio'] for th in THETAS if (th, nz) in REDRAW]
        for key, arr in [('a', wa), ('c', wc), ('d', wd), ('e', we)]:
            C3.setdefault('_dep', {}).setdefault(key, []).extend([x for x in arr if abs(x - 1) > 1e-9])
            C3.setdefault('_all', {}).setdefault(key, []).extend(list(arr))
        wf = [v['width_ratio'] for th in THETAS if (th, nz) in REDRAW for v in REDRAW[(th, nz)]['cells'].values() if np.isfinite(v['width_ratio'])]
        C3.setdefault('_dep', {}).setdefault('f', []).extend([x for x in wf if abs(x - 1) > 1e-9]); C3.setdefault('_all', {}).setdefault('f', []).extend(list(wf))
        C3['rows'].append(dict(name=row['name'], beta_fa=nz[0], beta_fr=nz[1], low_noise=low,
                               width_ratio_a=float(np.median(wa)) if wa else None, width_ratio_c=float(np.median(wc)) if wc else None,
                               width_ratio_d=float(np.median(wd)) if wd else None, width_ratio_e=float(np.median(we)) if we else None,
                               width_ratio_f=float(np.median(wf)) if wf else None,
                               frac_wider_a=float(np.mean(np.array(wa) > 1)) if wa else None, frac_wider_c=float(np.mean(np.array(wc) > 1)) if wc else None,
                               frac_wider_d=float(np.mean(np.array(wd) > 1)) if wd else None,
                               e_sign_same_redrawn=all(REDRAW[(th, nz)]['sign_same'] for th in THETAS if (th, nz) in REDRAW),
                               e_decided_fixed=all(REDRAW[(th, nz)]['decided_fixed'] for th in THETAS if (th, nz) in REDRAW),
                               e_decided_redrawn=all(REDRAW[(th, nz)]['decided_redrawn'] for th in THETAS if (th, nz) in REDRAW),
                               f_conclusive_changes=int(sum(v['conclusive_fixed'] != v['conclusive_redrawn'] for th in THETAS if (th, nz) in REDRAW for v in REDRAW[(th, nz)]['cells'].values())),
                               flips={k: row[k]['flip'] for k in CONCS}))
    lowrows = [r for r in C3['rows'] if r['low_noise']]
    C3['n_low_noise_settings'] = len(lowrows)
    C3['flips_at_low_noise'] = {k: [r['name'] for r in lowrows if r['flips'][k]] for k in CONCS}
    C3['n_flips_at_low_noise'] = int(sum(len(v) for v in C3['flips_at_low_noise'].values()))
    C3['flips_at_high_noise'] = {k: [r['name'] for r in C3['rows'] if (not r['low_noise']) and r['flips'][k]] for k in CONCS}
    for key in ['a', 'c', 'd', 'e', 'f']:
        vals = [r[f'width_ratio_{key}'] for r in C3['rows'] if r[f'width_ratio_{key}'] is not None]
        C3[f'median_width_ratio_{key}'] = float(np.median(vals)) if vals else None
        C3[f'frac_settings_wider_{key}'] = float(np.mean(np.array(vals) > 1)) if vals else None
        dep = np.array(C3.get('_dep', {}).get(key, [])); alll = np.array(C3.get('_all', {}).get(key, []))
        C3[f'coin_dependent_configs_{key}'] = dict(n_all=int(len(alll)), n_dependent=int(len(dep)), frac_coin_independent=float(1 - len(dep) / len(alll)) if len(alll) else None,
                                                    median_ratio_dependent=float(np.median(dep)) if len(dep) else None, frac_wider_dependent=float(np.mean(dep > 1)) if len(dep) else None)
    pooled_dep = np.concatenate([np.array(v) for v in C3.get('_dep', {}).values()]) if C3.get('_dep') else np.array([])
    C3['pooled_median_ratio_coin_dependent'] = float(np.median(pooled_dep)) if len(pooled_dep) else None
    C3['pooled_frac_wider_coin_dependent'] = float(np.mean(pooled_dep > 1)) if len(pooled_dep) else None
    C3['pooled_n_coin_dependent'] = int(len(pooled_dep))
    C3.pop('_dep', None); C3.pop('_all', None)
    C3['redraw_records'] = {f'theta{th}_fa{nz[0]:g}_fr{nz[1]:g}': {kk: vv for kk, vv in v.items() if kk != 'cells'} for (th, nz), v in REDRAW.items()}
    RESULTS['c3_noise_ci'] = C3
    # ------------------------------------------------------------------ 5e. C4: value-shift table (IDENTITY + IN-SAMPLE, id_test, Opus reference, dollars)
    VS = []
    ixt = IDX[('id_test', 'all', 'all')]; W0 = W[('id_test', 'all', 'all')][:1]
    def crit_rows(kind):
        out = []
        for m in PRIMARIES:
            for th in THETAS:
                ht = HIT[th][ixt, col[m]]
                vals = {}
                for c in [0.0] + SHIFTS:
                    sp, cp, so, co = setting_arrays(ixt, m, 'opus', 'dollars', c); st = casc_sums(W0, hit_weights(W0, ht, (0.0, 0.0), None), sp, cp, so, co)
                    if kind == 'floor': vals[c] = float(crit_floor(st, 'etaB')[0]); Sx = float(st['p'][0] * st['cbar'][0] - st['a'][0]); scale = 1.0
                    elif kind == 'matched': vals[c] = float(crit_matched(st, 'etaB')[0]); Sx = float(st['p'][0] * st['cbar_h'][0] - st['a'][0]); scale = float(st['cbar'][0])
                    elif kind == 'netgain': vals[c] = float(st['Q'][0] + st['eta_B'][0] * st['S'][0]); Sx = float(st['S'][0]); scale = float(st['cbar'][0])
                    elif kind.startswith('thr_lam'): lv = float(kind.split('=')[1]); vals[c] = float(crit_floor(st, lv)[0]); Sx = float(st['S'][0]); scale = float('inf')
                out.append(dict(config=f'{m}|theta={th}', S=Sx, v0=vals[0.0], v01=vals[0.1], v05=vals[0.5], pred_change01=(0.1 * Sx / scale if np.isfinite(scale) else 0.0),
                                pred_change05=(0.5 * Sx / scale if np.isfinite(scale) else 0.0)))
        return out
    def summarize(kind, label, form, reason, rows, invariant):
        ch01 = [abs(r['v01'] - r['v0']) > 1e-12 * max(1, abs(r['v0'])) for r in rows]; ch05 = [abs(r['v05'] - r['v0']) > 1e-12 * max(1, abs(r['v0'])) for r in rows]
        fl01 = [sgn(r['v01']) != sgn(r['v0']) for r in rows]; fl05 = [sgn(r['v05']) != sgn(r['v0']) for r in rows]
        Snz = [abs(r['S']) > 1e-12 for r in rows]
        resid = max([abs((r['v01'] - r['v0']) - r['pred_change01']) for r in rows] + [abs((r['v05'] - r['v0']) - r['pred_change05']) for r in rows]) if rows else 0.0
        VS.append(dict(id=kind, criterion=label, form=form, shift_invariant=invariant, algebraic_reason=reason, n=len(rows), n_S_nonzero=int(sum(Snz)),
                       n_changed_c01=int(sum(ch01)), n_changed_c05=int(sum(ch05)), n_sign_flips_c01=int(sum(fl01)), n_sign_flips_c05=int(sum(fl05)),
                       flips_only_where_S_nonzero=bool(all((not f) or s for f, s in zip(fl05, Snz)) and all((not f) or s for f, s in zip(fl01, Snz))),
                       changed_iff_S_nonzero=bool(all(c == s for c, s in zip(ch05, Snz))), max_identity_residual=float(resid),
                       flipped_configs_c05=[r['config'] for r, f in zip(rows, fl05) if f], flipped_configs_c01=[r['config'] for r, f in zip(rows, fl01) if f], rows=rows))
    summarize('floor', 'eq:floor (homogeneous), ratio form', 'p v_h cbar - vbar a', 'change = c (p cbar - a) = c S_hom; invariant iff S_hom = 0', crit_rows('floor'), False)
    summarize('matched', 'eq:matched-reference-condition', 'p(v_h - vbar_h) + eta_B (p cbar_h - a)', 'change = (c/cbar) S with S = p cbar_h - a; invariant iff S = 0', crit_rows('matched'), False)
    summarize('netgain', 'thm:surplus net gain Q + eta_B S (= sign of realized eta - eta_B)', 'Q + eta_B S', 'Q is a difference of values (invariant); eta_B -> eta_B + c/cbar, so change = (c/cbar) S; invariant iff S = 0', crit_rows('netgain'), False)
    for lv in LAMS_EXO:
        summarize(f'thr_lam={lv:g}', f'prop:threshold at exogenous lambda = {lv:g} (also the realized J_lambda difference)', 'p(lam b + v_h - v_b) - lam a', 'values enter only as the difference v_h - v_b; the price is exogenous: exactly invariant', crit_rows(f'thr_lam={lv:g}'), True)
    # VOI band: no values enter
    VS.append(dict(id='voi', criterion='prop:voi band k/(b-a) < s < 1-k/a', form='cost-only', shift_invariant=True, algebraic_reason='no value enters the band (value-preserving hypothesis): exactly invariant', n=len(CAS['baseline']['rows_d']),
                   n_S_nonzero=0, n_changed_c01=0, n_changed_c05=0, n_sign_flips_c01=0, n_sign_flips_c05=0, flips_only_where_S_nonzero=True, changed_iff_S_nonzero=True, max_identity_residual=0.0,
                   flipped_configs_c05=[], flipped_configs_c01=[], rows=[]))
    # record criteria: sum nu G at eta_U (changes by (c/cbar_U) S_rec), noise threshold (changes iff dc^I or dc^W != 0), equal-cost variant, exogenous lambda
    rec_rows = {'sumG': [], 'thr': [], 'thr_unit': [], 'sumG_lam1': [], 'thr_lam1': []}
    for th in THETAS:
        for nz in NOISES:
            for cold in ['opus', 'dimbest']:
                pr = {c: rec_primitives(th, nz, cold, 'dollars', shift=c) for c in [0.0] + SHIFTS}
                pu = {c: rec_primitives(th, nz, cold, 'unit', shift=c) for c in [0.0] + SHIFTS}
                pl = {c: rec_primitives(th, nz, cold, 'dollars', lam=1.0, shift=c) for c in [0.0] + SHIFTS}
                cfg = f'theta={th}|fa={nz[0]:g}|fr={nz[1]:g}|{cold}'
                rec_rows['sumG'].append(dict(config=cfg, S=pr[0.0]['S_rec'], v0=pr[0.0]['sumG'], v01=pr[0.1]['sumG'], v05=pr[0.5]['sumG'],
                                             pred_change01=0.1 / pr[0.0]['cbar_U'] * pr[0.0]['S_rec'], pred_change05=0.5 / pr[0.0]['cbar_U'] * pr[0.0]['S_rec']))
                rec_rows['sumG_lam1'].append(dict(config=cfg, S=pl[0.0]['S_rec'], v0=pl[0.0]['sumG'], v01=pl[0.1]['sumG'], v05=pl[0.5]['sumG'], pred_change01=0.0, pred_change05=0.0))
                for j in np.where(pr[0.0]['hyp'])[0]:
                    dc = float(abs(pr[0.0]['dcI'][j]) + abs(pr[0.0]['dcW'][j]))
                    # threshold change is not linear in c; predicted change computed from the exact formula with shifted eta_U
                    def thr_at(p_, c):
                        return float(p_[c]['thr'][j])
                    rec_rows['thr'].append(dict(config=cfg + '|' + CELLS21[j], S=dc, v0=thr_at(pr, 0.0), v01=thr_at(pr, 0.1), v05=thr_at(pr, 0.5),
                                                pred_change01=thr_at(pr, 0.1) - thr_at(pr, 0.0), pred_change05=thr_at(pr, 0.5) - thr_at(pr, 0.0),
                                                err=float(pr[0.0]['err'][j]), decision0=bool(pr[0.0]['err'][j] < pr[0.0]['thr'][j]), decision05=bool(pr[0.5]['err'][j] < pr[0.5]['thr'][j])))
                for j in np.where(pu[0.0]['hyp'])[0]:
                    rec_rows['thr_unit'].append(dict(config=cfg + '|' + CELLS21[j], S=0.0, v0=float(pu[0.0]['thr'][j]), v01=float(pu[0.1]['thr'][j]), v05=float(pu[0.5]['thr'][j]), pred_change01=0.0, pred_change05=0.0))
                for j in np.where(pl[0.0]['hyp'])[0]:
                    rec_rows['thr_lam1'].append(dict(config=cfg + '|' + CELLS21[j], S=float(abs(pl[0.0]['dcI'][j]) + abs(pl[0.0]['dcW'][j])), v0=float(pl[0.0]['thr'][j]), v01=float(pl[0.1]['thr'][j]), v05=float(pl[0.5]['thr'][j]), pred_change01=0.0, pred_change05=0.0))
    summarize('rec_sumG', 'thm:memory stationary criterion sum_j nu_j G_j at price eta_U', 'sum nu_j (omega g^I + psi g^W)/(1-alpha)', 'g^s = dv - eta_U dc; eta_U -> eta_U + c/cbar_U; change = (c/cbar_U) S_rec, S_rec = -sum nu_j^2 (omega dc^I + psi dc^W)/(1-alpha); invariant iff S_rec = 0 (equal costs)', rec_rows['sumG'], False)
    summarize('rec_thr', 'eq:noise-threshold g^I/(g^I - g^W) at price eta_U (dollars)', 'g^I/(g^I-g^W)', 'changes through the cost channel dc^I, dc^W (S column = |dc^I|+|dc^W|); invariant iff dc^I = dc^W = 0', rec_rows['thr'], False)
    summarize('rec_thr_unit', 'eq:noise-threshold with equal costs in every record state (unit-cost basis)', 'g^s = v^s - v^U', 'dc^I = dc^W = 0 so g^s is a value difference: exactly invariant (paper: threshold 3/8 in Fig. memory)', rec_rows['thr_unit'], True)
    summarize('rec_sumG_lam1', 'thm:memory stationary criterion at exogenous price lambda = 1', 'sum nu G(lambda)', 'g^s(lambda) = dv - lambda dc with lambda fixed: exactly invariant', rec_rows['sumG_lam1'], True)
    summarize('rec_thr_lam1', 'eq:noise-threshold at exogenous price lambda = 1', 'g^I(lambda)/(g^I(lambda)-g^W(lambda))', 'exogenous price: exactly invariant', rec_rows['thr_lam1'], True)
    thr_dec_flips = int(sum(r['decision0'] != r['decision05'] for r in rec_rows['thr']))
    RESULTS['c4_value_shift'] = dict(table=VS, noise_threshold_decision_flips_c05=thr_dec_flips,
                                     realized_sign_flips_from_settings={c: [r['primary'] + f"|theta={r['theta']}" for r in CAS[f'c={c:g}']['rows_ab']
                                                                            if sgn(r['real'][0]) != sgn(next(x for x in CAS['baseline']['rows_ab'] if x['primary'] == r['primary'] and x['theta'] == r['theta'])['real'][0])] for c in SHIFTS})
    # ------------------------------------------------------------------ 5f. claims
    CL = []
    def claim(cid, statement, kind, result, verdict):
        CL.append(dict(id=cid, statement=statement, kind=kind, result=result, verdict=verdict))
    e_sec = next(r for r in SUMM if r['name'] == 'seconds')['e']; e_tok = next(r for r in SUMM if r['name'] == 'tokens')['e']
    cb = RESULTS['c1_cascade_by_basis']
    c1_pass = (e_sec['flip'] is True) and cb['seconds']['wins_decided'] == 0 and cb['seconds_hedged']['wins_decided'] == 0
    claim('C1', 'The seconds basis flips the record conclusion (e) and the cascade-efficiency conclusions because Opus is the fastest model on average: no verified cascade beats always-Opus in seconds, sequential or hedged.',
          'OUT-OF-SAMPLE (records: SEMI-SYNTHETIC stream)',
          f"(e) seconds: realized late-window eta_I - eta_U = {fmt(e_sec['detail']['per_theta']['1.0']['real'], 4)} (theta=1), {fmt(e_sec['detail']['per_theta']['0.5']['real'], 4)} (theta=0.5) vs dollars {fmt(base['e']['detail']['per_theta']['1.0']['real'], 2)} / {fmt(base['e']['detail']['per_theta']['0.5']['real'], 2)}; "
          f"predicted sum nu G seconds {e_sec['detail']['per_theta']['1.0']['sumG']:+.3f} / {e_sec['detail']['per_theta']['0.5']['sumG']:+.3f}; tokens: holds={e_tok['holds']}. "
          f"Cascades (7 primaries x 2 theta, Opus reference) with realized eta/eta_B > 1 (CI-decided): dollars {cb['dollars']['wins_decided']}/14, seconds sequential {cb['seconds']['wins_decided']}/14 (losses {cb['seconds']['losses_decided']}), "
          f"seconds hedged {cb['seconds_hedged']['wins_decided']}/14 (losses {cb['seconds_hedged']['losses_decided']}), tokens {cb['tokens']['wins_decided']}/14; median eta/eta_B seconds {cb['seconds']['median_eta_ratio']:.2f}, hedged {cb['seconds_hedged']['median_eta_ratio']:.2f}; "
          f"min a/cbar in seconds {cb['seconds']['min_a_over_cbar']:.2f} (>1: every primary is slower than Opus on average). Opus mean latency {DATA_CHECKS['mean_latency_s_id_test_by_model'][BASE]:.2f} s is the minimum of the 8 models: {DATA_CHECKS['opus_is_fastest_on_average_id_test']}.",
          'PASS' if c1_pass else ('INCONCLUSIVE' if e_sec['flip'] is True else 'FAIL'))
    e_db = next(r for r in SUMM if r['name'] == 'dimbest'); e_or = next(r for r in SUMM if r['name'] == 'oracle')
    rnd = [r for r in SUMM if r['axis'] == 'reference' and r['name'].startswith('random')]
    rnd_flip = float(np.mean([bool(r['e']['flip']) for r in rnd if r['e']['flip'] is not None])) if rnd else float('nan')
    c2_pass = (e_db['e']['flip'] is True) and e_db['e_pred']['holds']
    claim('C2', 'The reference choice flips the record conclusion (e): against DimensionBest (voter file) records lower efficiency, and the stationary criterion sum nu_j G_j predicts the flip.',
          'OUT-OF-SAMPLE (SEMI-SYNTHETIC stream)',
          f"DimensionBest: realized eta_I - eta_U = {fmt(e_db['e']['detail']['per_theta']['1.0']['real'], 2)} (theta=1), {fmt(e_db['e']['detail']['per_theta']['0.5']['real'], 2)} (theta=0.5), predicted sum nu G {e_db['e']['detail']['per_theta']['1.0']['sumG']:+.3f} / {e_db['e']['detail']['per_theta']['0.5']['sumG']:+.3f} -> flip={e_db['e']['flip']}, prediction agrees={e_db['e_pred']['holds']}; "
          f"oracle: {fmt(e_or['e']['detail']['per_theta']['1.0']['real'], 2)} / {fmt(e_or['e']['detail']['per_theta']['0.5']['real'], 2)}, flip={e_or['e']['flip']}, predicted={e_or['e_pred']['holds']}; random uniform reference ({len(rnd)} draws): flip fraction {rnd_flip:.2f}, "
          f"prediction agrees in {float(np.mean([r['e_pred']['holds'] for r in rnd])):.2f} of draws. Reference-axis flip fraction of (e) (type-balanced) = {FLIP['e']['reference']['frac']:.2f}; (e') = {('undecided' if FLIP['e_pred']['reference']['frac'] is None else f'{FLIP['e_pred']['reference']['frac']:.2f}')}.",
          'PASS' if c2_pass else ('INCONCLUSIVE' if e_db['e']['flip'] is True else 'FAIL'))
    wid = {k: C3[f'median_width_ratio_{k}'] for k in ['a', 'c', 'd', 'e', 'f']}
    dep = {k: C3[f'coin_dependent_configs_{k}'] for k in ['a', 'c', 'd', 'e', 'f']}
    def widen_verdict(keys):
        meds = [dep[k]['median_ratio_dependent'] for k in keys if dep[k]['median_ratio_dependent'] is not None]
        fr = [dep[k]['frac_wider_dependent'] for k in keys if dep[k]['frac_wider_dependent'] is not None]
        if not meds:
            return 'INCONCLUSIVE'
        ok = sum(m > 1.02 and f > 0.5 for m, f in zip(meds, fr)); bad = sum(m <= 1.02 for m in meds)
        return 'PASS' if ok == len(meds) else ('FAIL' if bad >= max(1, len(meds) - 1) else 'INCONCLUSIVE')
    def dep_txt(keys):
        return ', '.join(f"({k}) median {dep[k]['median_ratio_dependent']:.3f}, wider in {dep[k]['frac_wider_dependent']:.2f} of {dep[k]['n_dependent']} coin-dependent configs ({dep[k]['n_all'] - dep[k]['n_dependent']} coin-independent)" for k in keys if dep[k]['median_ratio_dependent'] is not None)
    claim('C3a', 'Re-drawing the verifier coin inside every bootstrap resample widens the CIs of the per-task cascade statistics ((a) eta - eta_B, (c) J_lambda difference per cell, (d) C_info - min cost per cell) relative to X1/X3\'s fixed draw.',
          'OUT-OF-SAMPLE (bootstrap property)',
          f"CI-width ratio re-drawn / fixed over the {len(C3['rows'])} noisy settings: {dep_txt(['a', 'c', 'd'])}. Rule (set after the quick run showed ratios near 1): PASS if every median > 1.02 with > 50% wider. "
          f"Interpretation: a fixed draw puts one coin realization into each task's value, so resampling tasks already reproduces the coin variance to first order; the re-draw changes the point estimate (expected coin instead of one realization), not the width. X1's limitation 'CIs omit coin-flip variance' overstates the effect for per-task statistics.",
          widen_verdict(['a', 'c', 'd']))
    claim('C3b', 'Re-drawing the coin per resample widens the CIs of the record-stream statistics ((e) eta_I - eta_U, (f) per-cell record gain), where a write\'s coin sets a persistent record state.',
          'SEMI-SYNTHETIC (bootstrap property)',
          f"{dep_txt(['e', 'f'])}; (e) decided under both bootstraps in {sum(r['e_decided_fixed'] and r['e_decided_redrawn'] for r in C3['rows'])}/{len(C3['rows'])} settings, sign unchanged in {sum(r['e_sign_same_redrawn'] for r in C3['rows'])}/{len(C3['rows'])}; (f) cells whose conclusiveness changes: {sum(r['f_conclusive_changes'] for r in C3['rows'])}. Same rule as C3a.",
          widen_verdict(['e', 'f']))
    claim('C3c', 'Verifier noise with the coin re-drawn per resample flips no conclusion at beta_fa, beta_fr <= 0.1.',
          'OUT-OF-SAMPLE / SEMI-SYNTHETIC',
          f"Flips at max(beta) <= 0.1 ({C3['n_low_noise_settings']} settings x 7 conclusions, decided cells only): {C3['n_flips_at_low_noise']} {json.dumps({k: v for k, v in C3['flips_at_low_noise'].items() if v})}; flips at beta >= 0.2: {json.dumps({k: v for k, v in C3['flips_at_high_noise'].items() if v})}; "
          f"undecided (conclusion, setting) pairs on the noise axis: {sum(r['flips'][k] is None for r in C3['rows'] for k in CONCS)}.",
          'PASS' if C3['n_flips_at_low_noise'] == 0 else 'FAIL')
    ratio_ok = all(r['changed_iff_S_nonzero'] and r['flips_only_where_S_nonzero'] for r in VS if not r['shift_invariant'])
    inv_ok = all(r['n_changed_c05'] == 0 and r['n_changed_c01'] == 0 for r in VS if r['shift_invariant'])
    vs_ok = ratio_ok and inv_ok and all(r['max_identity_residual'] < 1e-9 for r in VS)
    RESULTS['c4_value_shift']['ratio_form_changed_iff_S_nonzero'] = ratio_ok; RESULTS['c4_value_shift']['invariant_criteria_never_change'] = inv_ok
    fl_frac = {r['id']: (r['n_sign_flips_c05'], r['n_S_nonzero'], r['n']) for r in VS}
    claim('C4', 'Adding a constant c to every value changes a criterion exactly when its resource saving S is nonzero (Q + eta_B S -> Q + eta_B S + c S/cbar); criteria with S = 0, an exogenous price, or cost-only form are exactly invariant; observed sign flips occur only at S != 0.',
          'IDENTITY + IN-SAMPLE',
          f"identity residual max {max(r['max_identity_residual'] for r in VS):.1e}; ratio-form criteria change iff S != 0 and flip only at S != 0: {ratio_ok}; exogenous-price, cost-only and equal-cost criteria never change: {inv_ok}; sign flips at c = 0.5 (flipped / S!=0 / n): "
          + ', '.join(f"{k} {v[0]}/{v[1]}/{v[2]}" for k, v in fl_frac.items()) + f"; realized eta - eta_B sign flips (settings module): c=0.1 {RESULTS['c4_value_shift']['realized_sign_flips_from_settings'][0.1]}, c=0.5 {RESULTS['c4_value_shift']['realized_sign_flips_from_settings'][0.5]}; "
          f"noise-threshold decisions (err < thr) changed at c = 0.5 in {thr_dec_flips}/{len(rec_rows['thr'])} hypothesis-met cell configs; equal-cost threshold invariant: {next(r for r in VS if r['id']=='rec_thr_unit')['n_changed_c05'] == 0}. "
          f"Not every S != 0 criterion flips: a flip needs |c S/cbar| > |Q + eta_B S| with opposite sign.",
          'PASS' if vs_ok else 'FAIL')
    for k in CONCS:
        dep = DEPENDS.get(k, [])
        claim(f'R-{k}', f"{CONC_LABEL[k]}: robust across ALL settings (no sign flip on any axis)", {'a': 'OUT-OF-SAMPLE', 'b': 'OUT-OF-SAMPLE', 'c': 'OUT-OF-SAMPLE', 'd': 'OUT-OF-SAMPLE', 'e': 'SEMI-SYNTHETIC + OUT-OF-SAMPLE', 'e_pred': 'SEMI-SYNTHETIC + OUT-OF-SAMPLE', 'f': 'SEMI-SYNTHETIC + OUT-OF-SAMPLE'}[k],
              f"baseline holds={BASE_HOLDS[k]}; flip fractions " + ', '.join(f"{ax} {('n/a' if FLIP[k][ax]['frac'] is None else f'{FLIP[k][ax]['frac']:.2f}')}" for ax in AXES)
              + (f"; flipped settings: " + '; '.join(f"{ax}: {FLIP[k][ax].get('flipped_settings', FLIP[k][ax].get('by_type'))}" for ax in dep) if dep else ''),
              ('NOT-INSTANTIABLE' if not BASE_HOLDS[k] else ('PASS' if k in ROBUST else 'FAIL')))
    RESULTS['claims'] = CL
    RESULTS['runtime_s'] = round(time.time() - T0, 1)
    # ------------------------------------------------------------------ 5g. audit: three headline numbers with independent code (pandas / python loops)
    AUD = {}
    dfp = df[df.split == 'probing'].pivot(index='task_id', columns='model', values=['score', 'cost_usd']); dft = df[df.split == 'id_test'].pivot(index='task_id', columns='model', values=['score', 'cost_usd'])
    agree_f = agree_m = 0
    for m in PRIMARIES:
        for th in THETAS:
            h = dfp['score'][m] >= th - 1e-12; p = h.mean(); vh = dfp['score'][m][h].mean(); vb = dfp['score'][BASE].mean(); cb_ = dfp['cost_usd'][BASE].mean(); a = dfp['cost_usd'][m].mean()
            vbh = dfp['score'][BASE][h].mean(); cbh = dfp['cost_usd'][BASE][h].mean(); etaB = vb / cb_
            floor_sign = np.sign(p - (vb / vh) * (a / cb_)); matched_sign = np.sign(p * (vh - vbh) + etaB * (p * cbh - a))
            ht = dft['score'][m] >= th - 1e-12
            V = np.where(ht, dft['score'][m], dft['score'][BASE]); D = dft['cost_usd'][m] + np.where(ht, 0.0, dft['cost_usd'][BASE])
            real_sign = np.sign(V.sum() / D.sum() - dft['score'][BASE].sum() / dft['cost_usd'][BASE].sum())
            agree_f += floor_sign == real_sign; agree_m += matched_sign == real_sign
    AUD['a_agree_all14_pandas'] = dict(floor=int(agree_f), matched=int(agree_m), script=dict(floor=base['a']['detail']['agree_floor_all'], matched=base['a']['detail']['agree_matched_all']))
    rc = pd.DataFrame([dict(pred=r['sign_pred'], real=r['sign_real'], dec=r['decided']) for r in CAS['baseline']['rows_c'] if r['lam'] == 'etaB'])
    AUD['c_decided_agreement_pandas'] = dict(pandas=float((rc[rc.dec].pred == rc[rc.dec].real).mean()), n_decided=int(rc.dec.sum()), script=base['c']['detail']['agree_decided'])
    # (e) baseline: seed-0 late-window efficiencies by a plain python loop over rounds
    st, recm = record_states(1.0, (0.0, 0.0), U_VER)
    rec = {}; VI = DI = VU = DU = 0.0
    for n in range(N_ROUNDS):
        cj = int(JC[0, n]); t = int(TID[0, n])
        if EXPLORE[0, n]:
            fI = fU = int(MEXP[0, n])
        else:
            fI = rec.get(cj, OPUS); fU = OPUS
        if n >= N_ROUNDS - LATE:
            VI += Sv[t, fI]; DI += Cv[t, fI]; VU += Sv[t, fU]; DU += Cv[t, fU]
        if EXPLORE[0, n] and HIT[1.0][t, int(MEXP[0, n])]:
            rec[cj] = int(MEXP[0, n])
    f_I, f_U = fielded(st[:, LATE_SL], recm[:, LATE_SL], REFS['opus'])
    vec = dict(eta_I=float(Sv[TID_L[0], f_I[0]].sum() / Cv[TID_L[0], f_I[0]].sum()), eta_U=float(Sv[TID_L[0], f_U[0]].sum() / Cv[TID_L[0], f_U[0]].sum()))
    AUD['e_seed0_loop'] = dict(loop=dict(eta_I=VI / DI, eta_U=VU / DU), vectorised=vec, max_abs_diff=float(max(abs(VI / DI - vec['eta_I']), abs(VU / DU - vec['eta_U']))))
    RESULTS['audit'] = AUD
    log('audit', json.dumps(jsonable(AUD)))
    # ------------------------------------------------------------------ 5h. outputs
    pd.DataFrame([dict(axis=r['axis'], name=r['name'], **{f'{k}_margin': r[k]['margin'] for k in CONCS}, **{f'{k}_holds': r[k]['holds'] for k in CONCS}, **{f'{k}_flip': r[k]['flip'] for k in CONCS}) for r in SUMM]).to_csv(os.path.join(OUT, 'settings_summary.csv'), index=False)
    RE.assign(real_lo=RE.real.apply(lambda t: t[1]), real_hi=RE.real.apply(lambda t: t[2]), real=RE.real.apply(lambda t: t[0])).to_csv(os.path.join(OUT, 'records_stationary.csv'), index=False)
    RF.assign(real_lo=RF.real.apply(lambda t: t[1]), real_hi=RF.real.apply(lambda t: t[2]), real=RF.real.apply(lambda t: t[0])).to_csv(os.path.join(OUT, 'records_noise_threshold_cells.csv'), index=False)
    pd.DataFrame([dict(setting=s['name'], **{k: v for k, v in r.items() if k != 'real'}, real=r['real'][0], real_lo=r['real'][1], real_hi=r['real'][2]) for s in SETTINGS for r in CAS[s['name']]['rows_c']]).to_csv(os.path.join(OUT, 'threshold_cells.csv'), index=False)
    pd.DataFrame([dict(setting=s['name'], **{k: (v if k != 'info_minus_best' else v[0]) for k, v in r.items()}) for s in SETTINGS for r in CAS[s['name']]['rows_d']]).to_csv(os.path.join(OUT, 'voi_cells.csv'), index=False)
    pd.DataFrame([dict(setting=s['name'], **{k: v for k, v in r.items() if k not in ('real', 'eta_ratio', 'r_hom_pred', 'r_meas', 'rel_err', 'real_fixedcoin')}, real=r['real'][0], real_lo=r['real'][1], real_hi=r['real'][2],
                       eta_ratio=r['eta_ratio'][0], r_hom_pred=r.get('r_hom_pred', [np.nan])[0], r_meas=r.get('r_meas', [np.nan])[0], rel_err=r.get('rel_err', [np.nan])[0]) for s in SETTINGS for r in CAS[s['name']]['rows_ab']]).to_csv(os.path.join(OUT, 'cascade_global_configs.csv'), index=False)
    make_figures(RESULTS, SUMM, FLIP, CAS, RE, REDRAW, VS)
    slim = dict(RESULTS); slim['settings'] = [{**{k: v for k, v in r.items() if k not in CONCS}, **{k: {kk: vv for kk, vv in r[k].items()} for k in CONCS}} for r in SUMM]
    for r in slim['settings']:
        for k in CONCS:
            if 'detail' in r[k] and isinstance(r[k]['detail'], dict):
                r[k]['detail'] = {kk: vv for kk, vv in r[k]['detail'].items() if kk not in ('cells',)}
    slim['c4_value_shift'] = dict(RESULTS['c4_value_shift']); slim['c4_value_shift']['table'] = [{k: v for k, v in r.items() if k != 'rows'} for r in VS]
    slim['c4_value_shift']['rows'] = {r['id']: r['rows'] for r in VS if r['id'] in ('floor', 'matched', 'netgain', 'rec_sumG', 'rec_thr', 'rec_thr_unit')}
    json.dump(jsonable(slim), open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
    write_readme(RESULTS, SUMM, FLIP, CAS, VS, C3, base, cb)
    log('done; runtime', RESULTS['runtime_s'], 's')
    for c in CL:
        print(f"{c['id']:8s} {c['verdict']:16s} {c['statement'][:110]}")

# =========================================================================================== 6. figures (vector PDF; no titles inside plots; palette validated with the dataviz checker)
def make_figures(RESULTS, SUMM, FLIP, CAS, RE, REDRAW, VS):
    def save(fig, name):
        fig.savefig(os.path.join(FIG, name), metadata={'Creator': 'X16_sensitivity/run.py'}); plt.close(fig)
    # ---- Fig 1: flip-fraction heatmap (conclusions x axes)
    fig, ax = plt.subplots(figsize=(6.75, 2.9), layout='constrained')
    Mx = np.full((len(CONCS), len(AXES)), np.nan)
    for i, k in enumerate(CONCS):
        for j, ax_ in enumerate(AXES):
            v = FLIP[k][ax_]['frac']; Mx[i, j] = np.nan if v is None else v
    Mm = np.ma.masked_invalid(Mx)
    im = ax.pcolormesh(np.arange(len(AXES) + 1), np.arange(len(CONCS) + 1), Mm, cmap=SEQ_CMAP, vmin=0, vmax=1, edgecolors='white', linewidth=1.5)
    for i, k in enumerate(CONCS):
        for j, ax_ in enumerate(AXES):
            f = FLIP[k][ax_]
            def fb(x, d=0):
                return '–' if (x is None or not np.isfinite(x)) else f'{x:.{d}f}'
            if f.get('na'):
                ax.add_patch(Rectangle((j, i), 1, 1, hatch='////', facecolor='#f0efec', edgecolor='#c3c2b7', linewidth=0))
                ax.text(j + .5, i + .5, 'n/a', ha='center', va='center', fontsize=7, color=MUTED)
            elif f['frac'] is None:
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor='#f0efec', edgecolor='white', linewidth=1.5))
                ax.text(j + .5, i + .5, f"undecided\n{f['n'] - f['n_undecided']}/{f['n']} decided", ha='center', va='center', fontsize=6, color=MUTED)
            else:
                txt = f"{f['frac']:.2f}" + (f"\n{f['n_flip']}/{f['n'] - f['n_undecided']}" if ax_ != 'reference' else f"\nDB {fb(f['by_type']['dimbest'])} · Or {fb(f['by_type']['oracle'])} · R {fb(f['by_type']['random'], 2)}")
                ax.text(j + .5, i + .5, txt, ha='center', va='center', fontsize=6.5, color='white' if f['frac'] > 0.55 else INK)
    ax.set_xticks(np.arange(len(AXES)) + .5); ax.set_xticklabels(['reference policy', r'price $\lambda$', 'cost basis', r'verifier noise $(\beta_{fa},\beta_{fr})$', 'value shift $+c$'], fontsize=8)
    ax.set_yticks(np.arange(len(CONCS)) + .5); ax.set_yticklabels([CONC_LABEL[k] for k in CONCS], fontsize=7.5)
    ax.invert_yaxis(); ax.tick_params(length=0); ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.02); cb.set_label('fraction of settings that flip the conclusion', fontsize=8); cb.outline.set_visible(False)
    cb.solids.set_rasterized(False)                      # keep the PDF fully vector (the colorbar gradient is an image XObject by default)
    save(fig, 'fig1_flip_heatmap.pdf')
    # ---- Fig 2: reference x basis (C1 / C2)
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.6), layout='constrained', gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    refs = ['opus', 'dimbest', 'oracle', 'random']; bcol = dict(zip(BASES, [C1_, C2_, C3_])); off = dict(zip(BASES, [-.25, 0, .25]))
    for i, r in enumerate(refs):
        for b in BASES:
            for th, mk_ in [(1.0, 'o'), (0.5, '^')]:
                if r == 'random':
                    sub = RE[(RE.cold.str.startswith('random')) & (RE.basis == b) & (RE.theta == th) & (RE.beta_fa == 0) & (RE.beta_fr == 0) & (RE.lam == 'etaU') & (RE['shift'] == 0)]
                    vals = np.array([100 * x[0] / u for x, u in zip(sub.real, sub.eta_U_late)])
                    y = np.median(vals); lo, hi = vals.min(), vals.max()
                else:
                    sub = RE[(RE.cold == r) & (RE.basis == b) & (RE.theta == th) & (RE.beta_fa == 0) & (RE.beta_fr == 0) & (RE.lam == 'etaU') & (RE['shift'] == 0)].iloc[0]
                    u = sub.eta_U_late; y, lo, hi = 100 * sub.real[0] / u, 100 * sub.real[1] / u, 100 * sub.real[2] / u
                x = i + off[b] + (-.06 if th == 1.0 else .06)
                ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=mk_, ms=4, color=bcol[b], mfc=bcol[b] if th == 1.0 else 'white', lw=.8, capsize=1.5)
    ax.axhline(0, color='#c3c2b7', lw=.6)
    ax.set_xticks(range(4)); ax.set_xticklabels(['always-Opus', 'DimensionBest', 'per-task oracle', f'random ({N_RANDOM} draws)'], fontsize=7.5)
    ax.set_ylabel(r'records: $(\eta_I-\eta_U)/\eta_U$ in the stationary window (%)'); ax.set_yscale('symlog', linthresh=10)
    hs = [plt.Line2D([], [], marker='s', ls='', color=bcol[b], label=b) for b in BASES] + [plt.Line2D([], [], marker='o', ls='', color=INK2, label=r'$\theta=1$'), plt.Line2D([], [], marker='^', ls='', color=INK2, mfc='white', label=r'$\theta=0.5$')]
    ax.legend(handles=hs, ncol=3, loc='upper right', fontsize=6.5, handletextpad=.2, columnspacing=.8)
    ax = axes[1]
    rows = RESULTS['c1_cascade_rows']; bases4 = ['dollars', 'seconds', 'seconds_hedged', 'tokens']; bcol4 = dict(zip(bases4, [C1_, C2_, C4_, C3_])); off4 = dict(zip(bases4, [-.3, -.1, .1, .3]))
    for j, m in enumerate(PRIMARIES):
        for b in bases4:
            for th, mk_ in [(1.0, 'o'), (0.5, '^')]:
                r = next(x for x in rows if x['primary'] == m and x['theta'] == th and x['basis'] == b)['eta_ratio']
                ax.errorbar(j + off4[b] + (-.04 if th == 1.0 else .04), r[0], yerr=[[r[0] - r[1]], [r[2] - r[0]]], fmt=mk_, ms=3.2, color=bcol4[b], mfc=bcol4[b] if th == 1.0 else 'white', lw=.7, capsize=1)
    ax.axhline(1, color='#c3c2b7', lw=.6); ax.set_yscale('log')
    ax.set_xticks(range(len(PRIMARIES))); ax.set_xticklabels([m.replace('claude-', '') for m in PRIMARIES], rotation=30, ha='right', fontsize=7)
    ax.set_ylabel(r'cascade $\eta/\eta_B$ vs always-Opus (id_test)')
    hs = [plt.Line2D([], [], marker='s', ls='', color=bcol4[b], label=b.replace('_', ' ')) for b in bases4]
    ax.legend(handles=hs, ncol=2, loc='upper right', fontsize=6.5, handletextpad=.2, columnspacing=.8)
    save(fig, 'fig2_reference_basis.pdf')
    # ---- Fig 3: verifier noise (C3): CI-width ratios and (c)/(d) agreement grids
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.4), layout='constrained', gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    ax = axes[0]; C3 = RESULTS['c3_noise_ci']
    ccol = {'a': C1_, 'c': C2_, 'd': C3_, 'e': C4_, 'f': '#e87ba4'}; lab = {'a': '(a) cascade vs reference', 'c': r'(c) $J_\lambda$ difference per cell', 'd': r'(d) $C_{info}-\min$ per cell', 'e': r'(e) $\eta_I-\eta_U$', 'f': '(f) per-cell record gain'}
    rng_j = np.random.default_rng(3)
    for k in ['a', 'c', 'd', 'e', 'f']:
        xs = [max(r['beta_fa'], r['beta_fr']) for r in C3['rows'] if r[f'width_ratio_{k}'] is not None]; ys = [r[f'width_ratio_{k}'] for r in C3['rows'] if r[f'width_ratio_{k}'] is not None]
        ax.scatter(np.array(xs) * (1 + rng_j.uniform(-.06, .06, len(xs))), ys, s=11, color=ccol[k], label=lab[k], alpha=.85, linewidths=0)
    ax.axhline(1, color='#c3c2b7', lw=.6); ax.set_xscale('log'); ax.set_xlabel(r'$\max(\beta_{fa},\beta_{fr})$'); ax.set_ylabel('CI width, re-drawn coin / fixed coin')
    ax.set_xticks([0.05, 0.1, 0.2, 0.4]); ax.set_xticklabels(['0.05', '0.1', '0.2', '0.4']); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.legend(fontsize=6, loc='upper left', handletextpad=.2)
    for ax, k, thr_ in [(axes[1], 'c', 0.90), (axes[2], 'd', 0.95)]:
        G = np.full((5, 5), np.nan)
        for r in SUMM:
            if r['axis'] in ('noise', 'baseline'):
                fa, fr = r['noise']; G[BETAS.index(fr), BETAS.index(fa)] = r[k]['margin'] + thr_
        im = ax.pcolormesh(np.arange(6), np.arange(6), np.ma.masked_invalid(G), cmap=SEQ_CMAP, vmin=0.5, vmax=1.0, edgecolors='white', linewidth=1.2)
        for i in range(5):
            for j in range(5):
                if np.isfinite(G[i, j]):
                    ax.text(j + .5, i + .5, f'{G[i, j]:.2f}', ha='center', va='center', fontsize=6.5, color='white' if G[i, j] > 0.8 else INK, fontweight='bold' if G[i, j] < thr_ else 'normal')
        ax.set_xticks(np.arange(5) + .5); ax.set_xticklabels([f'{b:g}' for b in BETAS], fontsize=7); ax.set_yticks(np.arange(5) + .5); ax.set_yticklabels([f'{b:g}' for b in BETAS], fontsize=7)
        ax.set_xlabel(r'$\beta_{fa}$'); ax.set_ylabel(r'$\beta_{fr}$'); ax.tick_params(length=0); ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False)
        ax.text(0.0, 5.25, {'c': r'(c) decided agreement, threshold 0.90 (bold = below)', 'd': r'(d) decided agreement, threshold 0.95 (bold = below)'}[k], fontsize=6.5, color=INK2)
    save(fig, 'fig3_noise.pdf')
    # ---- Fig 4: value shift (C4)
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.75), layout='constrained')
    cs = [0.0] + SHIFTS
    ax = axes[0]
    for rid, colr, lab_ in [('netgain', C1_, r'$Q+\eta_B\mathcal{S}$ (cascade vs Opus, 14 configs)'), ('floor', C2_, 'eq:floor criterion'), ('thr_lam=1', C3_, r'prop:threshold at $\lambda=1$ (invariant)')]:
        tab = next(r for r in VS if r['id'] == rid)
        for i, row in enumerate(tab['rows']):
            y = np.array([row['v0'], row['v01'], row['v05']]) / max(abs(row['v0']), 1e-300)
            ax.plot(cs, y, color=colr, lw=.9, alpha=.75, marker='o', ms=2.5, label=lab_ if i == 0 else None)
    ax.axhline(0, color='#c3c2b7', lw=.6); ax.set_xlabel('value shift $c$ added to every score'); ax.set_ylabel('criterion value / |value at $c=0$|'); ax.set_yscale('symlog', linthresh=1)
    h1, l1 = ax.get_legend_handles_labels()
    ax = axes[1]
    for rid, colr, lab_ in [('rec_sumG', C1_, r'$\sum_j\nu_jG_j$ at $\eta_U$ (dollars)'), ('rec_thr', C2_, 'noise threshold $g^I/(g^I-g^W)$, dollars'), ('rec_thr_unit', C3_, 'noise threshold, equal costs (invariant)'), ('rec_sumG_lam1', C4_, r'$\sum_j\nu_jG_j$ at $\lambda=1$ (invariant)')]:
        tab = next(r for r in VS if r['id'] == rid)
        for i, row in enumerate(tab['rows']):
            y = np.array([row['v0'], row['v01'], row['v05']]) / max(abs(row['v0']), 1e-300)
            ax.plot(cs, y, color=colr, lw=.7, alpha=.5, marker='o', ms=2, label=lab_ if i == 0 else None)
    ax.axhline(0, color='#c3c2b7', lw=.6); ax.set_xlabel('value shift $c$ added to every score'); ax.set_yscale('symlog', linthresh=1)
    h2, l2 = ax.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc='outside lower center', ncol=4, fontsize=6.3, handletextpad=.4, columnspacing=1.0)
    save(fig, 'fig4_value_shift.pdf')
    # ---- Fig 5: price lambda: (c) agreement and (a) floor vs matched
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.3), layout='constrained')
    lam_rows = [r for r in SUMM if r['axis'] in ('lambda', 'baseline')]
    etaB = float(REF_INFO['opus']['probing_eta_dollars'])
    xs = [etaB if r['lam'] == 'etaB' else float(r['lam']) for r in lam_rows]; order = np.argsort(xs)
    ax = axes[0]
    ax.plot(np.array(xs)[order], [lam_rows[i]['c']['detail']['agree_decided'] for i in order], marker='o', color=C1_, label='prop:threshold (as stated)')
    ax.plot(np.array(xs)[order], [lam_rows[i]['c']['detail']['naive_price_rule'] for i in order], marker='s', color=C2_, label=r'price rule $p\geq a/\bar c$', ls='-')
    ax.plot(np.array(xs)[order], [lam_rows[i]['c']['detail']['naive_always_attempt'] for i in order], marker='^', color=C3_, label='always attempt')
    ax.axhline(0.90, color='#c3c2b7', lw=.6); ax.axvline(etaB, color='#c3c2b7', lw=.6, ls=':')
    ax.set_xscale('log'); ax.set_xlabel(r'price $\lambda$ (per dollar); dotted: $\eta_B$'); ax.set_ylabel('(c) sign agreement on decided cells'); ax.legend(fontsize=6.5, loc='lower right')
    ax = axes[1]
    ax.plot(np.array(xs)[order], [lam_rows[i]['a']['detail']['agree_floor'] / lam_rows[i]['a']['detail']['n_decided'] for i in order], marker='o', color=C1_, label='homogeneous floor')
    ax.plot(np.array(xs)[order], [lam_rows[i]['a']['detail']['agree_matched'] / lam_rows[i]['a']['detail']['n_decided'] for i in order], marker='s', color=C2_, label='state-matched condition')
    ax.plot(np.array(xs)[order], [lam_rows[i]['a']['detail']['naive_cascade_wins'] for i in order], marker='^', color=C3_, label='cascade always wins')
    ax.axvline(etaB, color='#c3c2b7', lw=.6, ls=':'); ax.set_xscale('log'); ax.set_xlabel(r'price $\lambda$ (per dollar); dotted: $\eta_B$'); ax.set_ylabel('(a) sign agreement, 14 cascades (decided)')
    ax.set_ylim(0, 1.05); ax.legend(fontsize=6.5, loc='lower right')
    save(fig, 'fig5_lambda.pdf')
    log('figures written')

# =========================================================================================== 7. README / summary
def write_readme(RESULTS, SUMM, FLIP, CAS, VS, C3, base, cb):
    def yn(x):
        return 'yes' if x else 'no'
    def flipmark(r, k):
        f = r[k]['flip']
        if r[k].get('na'):
            return 'n/a'
        if f is None:
            return 'undecided' + (f" ({r['f']['detail']['n_conclusive']} conclusive)" if k == 'f' else '')
        s_ = '**FLIP**' if f else ('holds' if r[k]['holds'] else 'fails (as baseline)')
        if f and k == 'a' and r['a']['detail'].get('all_same_realized_sign'):
            s_ += ' (degenerate: every decided cascade has the same realized sign, so both criteria are right)'
        return s_
    def mg(r, k, d=3):
        m = r[k]['margin']; return '' if (r[k].get('na') or m is None or not np.isfinite(m)) else f'{m:+.{d}f} '
    def cellfmt(r, k):
        return 'n/a' if r[k].get('na') else f"{mg(r, k)}{flipmark(r, k)}"
    L_ = []
    L_.append(f"""# X16_sensitivity -- ablations and sensitivity of the X1 / X3 / X4 headline conclusions on CodeRouterBench

Generated by `run.py` (numpy / pandas / matplotlib only; fully offline; seed {SEED}; B = {B} paired task-resampling bootstraps, {B_REDRAW} coin-re-drawn
record-stream bootstraps for the heaviest cells; runtime {RESULTS['runtime_s']} s on the M4). Files: `run.py`, `results.json` (every number below with CIs;
`claims`), `settings_summary.csv` (one row per setting), `cascade_global_configs.csv`, `threshold_cells.csv`, `voi_cells.csv`, `records_stationary.csv`,
`records_noise_threshold_cells.csv`, `figures/fig1_flip_heatmap.pdf` .. `fig5_lambda.pdf`, `summary_for_paper.md`.

## Conclusions stressed (quoted from the predecessor READMEs) and how each is operationalised

| id | quoted conclusion | source | operational statement (holds iff) | margin whose sign encodes it | baseline value |
|---|---|---|---|---|---|
| (a) | "eq:floor as stated ... FAIL ... Where it disagrees with the realised sign it is always in the optimistic direction (predicts a win that does not occur)"; OOS "sign: 8/14 global" (floor) vs "12/14 global" (state-matched) | X1 C5 / C8 / C9 | on CI-decided (primary, theta) configs, the state-matched condition agrees with the realized sign in more configs than the homogeneous floor, and every floor error is a false win | (agree_matched - agree_floor)/n_decided > 0 and floor false losses = 0 | {base['a']['detail']['agree_matched']}/{base['a']['detail']['n_decided']} vs {base['a']['detail']['agree_floor']}/{base['a']['detail']['n_decided']} decided (all 14: {base['a']['detail']['agree_matched_all']} vs {base['a']['detail']['agree_floor_all']}); false wins {base['a']['detail']['floor_false_wins']}, false losses {base['a']['detail']['floor_false_losses']} |
| (b) | "eq:normalized ... FAIL: median rel. err +0.153 ... over-prediction in 13/14" | X1 C8 | the median over the 14 cascades of eq:normalized(probing)/realized(id_test) - 1 is positive | median rel. err > 0 (bootstrap CI) | {fmt(base['b']['detail']['median_rel_err'])}, over-prediction {base['b']['detail']['n_over']}/14 |
| (c) | "prop:threshold sign ... decided agreement 0.925 (dim9), 0.933 (cell21) ... PASS (qualified)" | X3 C1 | sign(p W_lambda - lambda a) from probing equals the realized sign of J_casc - J_direct on >= 90% of CI-decided (cell, cheap primary, theta) states, dim9 + cell21 pooled, at the setting's price | agreement - 0.90 >= 0 | {base['c']['detail']['agree_decided']:.3f} on {base['c']['detail']['n_decided']}/{base['c']['detail']['n']} decided at lambda = eta_B(probing) |
| (d) | "prop:voi substantive ... decided agreement 0.967 (theta=1), 0.990 (theta=0.5) ... PASS" | X3 C7 | band membership of s (probing) agrees with "exact information is the realized-cheapest option" on >= 95% of CI-decided (cell, cheap, theta, k) rows | agreement - 0.95 >= 0 | {base['d']['detail']['agree_decided']:.3f} on {base['d']['detail']['n_decided']}/{base['d']['detail']['n']} decided |
| (e) | "the sign is set entirely by (cold, resource): DimensionBest cold -> memory hurts in dollars and seconds; Opus cold -> memory helps in dollars ... but hurts in seconds" | X4 (c) | the realized stationary paired eta_I - eta_U against the setting's reference is positive (CI-decided) at both theta | min over decided theta of (eta_I - eta_U) > 0 | theta=1: {fmt(base['e']['detail']['per_theta']['1.0']['real'], 2)}, theta=0.5: {fmt(base['e']['detail']['per_theta']['0.5']['real'], 2)} score/$ |
| (e') | "Stationary sign criterion ... PASS / PASS" | X4 (c) | sum_j nu_j G_j (probing) has the realized sign at every decided theta (auxiliary: separates a change of the empirical sign, which the theory predicts, from a failure of the prediction) | agreement - 1 >= 0 | {base['e_pred']['detail']['pred_agree']:.2f} |
| (f) | "eq:noise-threshold ... PASS (one-sided in-sample) / PASS (OOS, 0.96)" | X4 (d) | among conclusive hypothesis-met cell configurations (g^I > 0 > g^W), psi_j/r_j < g^I/(g^I - g^W) predicts the sign of the realized paired record gain in >= 90% (pooled over the 25 noise settings and both theta; per noise setting on the noise axis) | accuracy - 0.90 >= 0 | {base['f']['detail']['accuracy']:.3f} on {base['f']['detail']['n_conclusive']} conclusive of {base['f']['detail']['n_hyp']} hypothesis-met rows (TN+FP = {base['f']['detail']['TN'] + base['f']['detail']['FP']}) |

Baseline setting: reference = always-Opus (also the fallback), basis = dollars (`cost_usd`), ratio criteria (price = eta_B, resp. eta_U), exact verifier, no shift.
A "flip" is a change of the statement's truth value relative to the baseline; it is counted only where the statement is decided at both settings
(realized CI excludes 0; for the rate-type statements (c), (d), (f) at least 10 / 10 / 5 decided cells). Every sign-agreement rate is reported next to a
naive baseline ("cascade always wins", "always attempt" and the price rule p >= a/cbar, "information iff the band is nonempty", "records always help", "always positive").
Kinds: (a)-(d) OUT-OF-SAMPLE (primitives on probing, realized values on id_test); (e), (e'), (f) SEMI-SYNTHETIC stream of real id_test outcomes with
probing primitives (X4's construction); C4 is IDENTITY + IN-SAMPLE.

## Setup

* Data: `{DATA}/coderouterbench/id_results_long.csv`, 9,999 tasks x 8 models; splits probing {DATA_CHECKS['n_probing']} / id_test {DATA_CHECKS['n_id_test']}; cross-check always-Opus id_test ${DATA_CHECKS['always_opus_id_test_total_usd']:.2f}, mean score {DATA_CHECKS['always_opus_id_test_mean_score']:.4f} (paper $34.02 / 43.83). {DATA_CHECKS['rows_cost_zero']} Qwen3-Max rows without a token record (cost, tokens, score = 0) and {DATA_CHECKS['rows_latency_zero']} zero-latency rows are kept as recorded (as in X1/X3/X4).
* Reference policies (the comparison policy AND the fallback of the verify-then-escalate cascade, so that the fallback restores the reference as eq:floor / app:matched-reference assume): always-Opus; DimensionBest = the released `voter_decisions/dimension_best.jsonl` choice, constant per dimension ({', '.join(f'{k}: {v}' for k, v in DIMBEST_MAP.items())}), extended to probing by dimension; per-task oracle = argmax score, ties -> min `cost_usd` (a fixed policy in every basis; id_test mean score {REF_INFO['oracle']['id_test_mean_score']:.3f}, {1000*REF_INFO['oracle']['id_test_mean_usd']:.2f} m$/task); random = a uniform model per task, {N_RANDOM} independent draws (seed {SEED}). Reference means on id_test: Opus {REF_INFO['opus']['id_test_mean_score']:.3f} / {1000*REF_INFO['opus']['id_test_mean_usd']:.2f} m$, DimensionBest {REF_INFO['dimbest']['id_test_mean_score']:.3f} / {1000*REF_INFO['dimbest']['id_test_mean_usd']:.2f} m$, random (draw 0) {REF_INFO['random00']['id_test_mean_score']:.3f} / {1000*REF_INFO['random00']['id_test_mean_usd']:.2f} m$.
* Cost bases: dollars (`cost_usd`), seconds (`latency_ms`/1000, sequential: primary latency + fallback latency on a miss), tokens (`total_tokens`). The information price k of prop:voi is the same fraction of the reference's probing mean cost in every basis: k_basis = k_usd x {K_SCALE['seconds']:.0f} s/$ or x {K_SCALE['tokens']:.0f} tokens/$ (Opus probing means {OPUS_PROB_MEAN['dollars']*1000:.3f} m$, {OPUS_PROB_MEAN['seconds']:.2f} s, {OPUS_PROB_MEAN['tokens']:.0f} tokens). A hedged-seconds variant (fallback launched concurrently: H lat_p + (1-H) max(lat_p, lat_B)) is reported for the realized cascade efficiencies only (C1); the conclusions use the sequential accounting the paper's formulas describe.
* Prices lambda in {{0.1, 1, 10, 100}} per dollar (exogenous, linear objective J_lambda = E[V - lambda D]) and the ratio form (price eta_B, resp. eta_U; eta_B(probing, dollars) = {REF_INFO['opus']['probing_eta_dollars']:.3f}). At an exogenous price the homogeneous criterion is p(lambda cbar + v_h - vbar) - lambda a and the state-matched one p(lambda cbar_h + v_h - vbar_h) - lambda a (app:matched-reference); the record criteria use g_j^s(lambda) = dv - lambda dc. (b) and (d) are ratio- / cost-only and do not depend on lambda (n/a).
* Verifier noise: the oracle verdict H = 1{{score >= theta}} is accepted with probability 1 - beta_fr and a miss is accepted with probability beta_fa, for all 25 pairs (beta_fa, beta_fr) in {{0, 0.05, 0.1, 0.2, 0.4}}^2. Point estimates use the expected coin; in the bootstrap every resampled task copy draws its own coin (the hit count of task i in resample b is Binomial(count_ib, q_i)). **This corrects X1's and X3's fixed draw** (one uniform per task held fixed across resamples, whose CIs omit coin variance); the fixed-draw CIs are recomputed here only to measure the difference (C3). In the record stream the coin decides every write; the re-drawn bootstrap re-simulates the whole record process with fresh coins in each of {B_REDRAW} resamples and re-weights rounds by resampled task (Opus reference, dollars, all 48 noisy write configurations, 8 processes).
* Value shift: +c, c in {{0.1, 0.5}}, added to every score (scores are in [0,1]); Q + eta_B S, the floor, the matched condition, eq:normalized, the record criteria and every sign-based conclusion are recomputed.
* Record stream (X4): {SEEDS_STREAM} independent streams x {N_ROUNDS:,} rounds of i.i.d. id_test tasks, exploration probability {EPS} (the logger fields a uniform model and writes on acceptance; informative if it truly hit, erroneous if falsely accepted; last write wins; no expiration), 21 dimension x source cells, stationary window = last {LATE:,} rounds; the uninformed policy fields the reference's model, the informed one the recorded model. Primitives omega_j, psi_j, v_j^s, c_j^s, eta_U, G_j from probing (OUT-OF-SAMPLE); realized quantities are paired late-window sums over the same rounds; task-cluster bootstrap with dynamics fixed (X4) plus the coin-re-drawn bootstrap above.
* Design: one axis at a time around the baseline ({len([s for s in SUMM if s['axis'] != 'ref_x_basis'])} settings) plus the full reference x basis grid ({len([s for s in SUMM if s['axis'] == 'ref_x_basis'])} settings). The bootstrap task weights are drawn once per (split, cell) and shared by every setting, so comparisons across settings are paired.

## Results

### Flip table (fraction of settings on each axis in which the conclusion's truth value changes; heatmap `figures/fig1_flip_heatmap.pdf`)
""")
    rows = []
    for k in CONCS:
        r = []
        for ax_ in AXES:
            f = FLIP[k][ax_]
            if f.get('na'):
                r.append('n/a')
            elif f['frac'] is None:
                r.append(f"undecided ({f['n'] - f['n_undecided']}/{f['n']} settings decided)" + (f"; by beta_fa level: {('undecided' if f['by_beta_fa_level']['frac'] is None else f'{f['by_beta_fa_level']['frac']:.2f}')} ({f['by_beta_fa_level']['n_decided']}/5 decided)" if 'by_beta_fa_level' in f else ''))
            elif ax_ == 'reference':
                def bt(x, d=0):
                    return 'undecided' if (x is None or not np.isfinite(x)) else f'{x:.{d}f}'
                r.append(f"{f['frac']:.2f} (DimBest {bt(f['by_type']['dimbest'])}, oracle {bt(f['by_type']['oracle'])}, random {bt(f['by_type']['random'], 2)} of {N_RANDOM})")
            else:
                r.append(f"{f['frac']:.2f} ({f['n_flip']}/{f['n'] - f['n_undecided']}" + (f", {f['n_undecided']} undecided" if f['n_undecided'] else '') + ')' + (f"; by beta_fa level: {('undecided' if f['by_beta_fa_level']['frac'] is None else f'{f['by_beta_fa_level']['frac']:.2f}')} ({f['by_beta_fa_level']['n_decided']}/5 decided)" if 'by_beta_fa_level' in f else ''))
        rows.append([CONC_LABEL[k], f"{yn(RESULTS['baseline_holds'][k])}"] + r + ['**robust**' if k in RESULTS['robust'] else 'depends on ' + ', '.join(RESULTS['depends'][k])])
    L_.append(table(['conclusion', 'holds at baseline', 'reference policy', 'price lambda', 'cost basis', 'verifier noise', 'value shift', 'verdict'], rows))
    L_.append(f"\nReference-axis fractions are type-balanced (DimensionBest and oracle count 1 each, random counts the fraction of its {N_RANDOM} draws). "
              f"Robust across ALL settings: {', '.join(RESULTS['robust']) or 'none'}. Depends on a specific setting: {json.dumps(RESULTS['depends'])}.\n")
    a_deg = [r['name'] for r in SUMM if r['a']['flip'] and r['a']['detail'].get('all_same_realized_sign')]
    a_nondeg = [r['name'] for r in SUMM if r['a']['flip'] and not r['a']['detail'].get('all_same_realized_sign') and r['axis'] != 'ref_x_basis']
    f_rnd = [r for r in SUMM if r['axis'] == 'reference' and r['name'].startswith('random') and r['f']['flip']]
    L_.append(f"\nReading the flips. (a) flips in two ways: degenerate settings where every decided cascade has the same realized sign, so both criteria are trivially right ({', '.join(a_deg) or 'none'}: at lambda <= 1 every cascade wins because the value gain dominates the price, in seconds every cascade loses), "
              f"and non-degenerate settings where the homogeneous floor is simply right in every decided cascade ({len(a_nondeg)} settings: {', '.join(a_nondeg[:3])}{', ...' if len(a_nondeg) > 3 else ''}); the floor's optimism bites only for cascades whose margin p - floor is a few points, which happens against Opus and DimensionBest but not against a random reference. "
              f"(c) fails against DimensionBest and the oracle because the as-stated prediction with the reference's unconditional means (b, v_b) predicts 'attempt' for cells that realize 'direct' (predicted-attempt/realized-direct counts: DimensionBest {next(r for r in SUMM if r['name']=='dimbest')['c']['detail']['n_pred_attempt_real_direct']}, oracle {next(r for r in SUMM if r['name']=='oracle')['c']['detail']['n_pred_attempt_real_direct']} of the decided cells) and at lambda <= 1 for the reason X3 recorded (v_h below E[Y_B | H=1] at theta = 0.5); in tokens it fails because the cheap primaries spend more tokens than Opus (a/cbar up to 3.8). "
              f"(d) fails against the oracle (band predicts information in {next(r for r in SUMM if r['name']=='oracle')['d']['detail']['n_pred_in_real_out']} decided rows that realize otherwise: with a reference that is itself cheap, b - a is small and the k-bands mislead) and in tokens. "
              f"(e) flips exactly where sum nu_j G_j predicts it (e' never flips): seconds and tokens (a random accepted model is slower and longer than Opus), DimensionBest and the oracle (already strong uninformed policies). "
              f"(f) flips in {len(f_rnd)}/{N_RANDOM} random draws; over the flipped draws the errors are FP {sum(r['f']['detail']['FP'] for r in f_rnd)} (predicted positive, realized negative) and FN {sum(r['f']['detail']['FN'] for r in f_rnd)} (predicted negative, realized positive) against TP {sum(r['f']['detail']['TP'] for r in f_rnd)} + TN {sum(r['f']['detail']['TN'] for r in f_rnd)} correct. A weak random uninformed policy makes the hypothesis g^I > 0 > g^W hold in many cells ({int(np.mean([r['f']['detail']['n_hyp'] for r in SUMM if r['axis'] == 'reference' and r['name'].startswith('random')]))} hypothesis-met rows per draw vs {base['f']['detail']['n_hyp']} at the baseline), but its per-cell primitives are estimated from the random assignment on probing while the stream realizes a different random assignment on id_test, so g^I, g^W and the realized per-cell gains are all small relative to that sampling noise; against DimensionBest and the oracle no cell meets the hypothesis conclusively (undecided).\n")
    # per-axis detail
    L_.append("\n### Per-setting margins (margin > 0 (resp. >= 0 for the rate criteria) = statement holds; FLIP = truth value differs from the baseline)\n")
    rows = []
    for r in SUMM:
        if r['axis'] == 'ref_x_basis':
            continue
        rows.append([r['axis'], r['name']] + [cellfmt(r, k) for k in CONCS])
    L_.append(table(['axis', 'setting'] + [k for k in CONCS], rows))
    # reference x basis grid
    L_.append("\n### Reference x basis grid (C1 / C2; `figures/fig2_reference_basis.pdf`): records against each reference in each basis, realized late-window eta_I - eta_U (paired, 95% CI) and the predicted sum_j nu_j G_j\n")
    rows = []
    for r in SUMM:
        if r['axis'] in ('baseline', 'reference', 'basis', 'ref_x_basis') and not r['name'].startswith('random') and not (r['axis'] == 'ref_x_basis' and r['ref'].startswith('random')):
            e = r['e']['detail']
            rows.append([r['ref'], r['basis'], fmt(e['per_theta']['1.0']['real'], 4), f"{e['per_theta']['1.0']['sumG']:+.4f}", fmt(e['per_theta']['0.5']['real'], 4), f"{e['per_theta']['0.5']['sumG']:+.4f}", yn(r['e']['holds']), f"{r['e_pred']['detail']['pred_agree']:.2f}", flipmark(r, 'e')])
    rnd_rows = [r for r in SUMM if (r['axis'] in ('reference', 'ref_x_basis')) and r['ref'].startswith('random')]
    for b in BASES:
        rr = [r for r in rnd_rows if r['basis'] == b]
        if rr:
            def med_range(th):
                v = np.array([r['e']['detail']['per_theta'][th]['real'][0] for r in rr]); return f"median {np.median(v):.4f} [min {v.min():.4f}, max {v.max():.4f}]"
            def sg(th):
                v = np.array([r['e']['detail']['per_theta'][th]['sumG'] for r in rr]); return f"median {np.median(v):+.4f}, >0 in {int((v > 0).sum())}/{len(v)}"
            rows.append([f'random ({len(rr)} draws)', b, med_range('1.0'), sg('1.0'), med_range('0.5'), sg('0.5'), f"{sum(r['e']['holds'] for r in rr)}/{len(rr)} draws", f"{np.mean([r['e_pred']['detail']['pred_agree'] for r in rr if np.isfinite(r['e_pred']['detail']['pred_agree'])]):.2f}", f"flip in {sum(bool(r['e']['flip']) for r in rr)}/{len(rr)}"])
    L_.append(table(['reference (uninformed policy)', 'basis', 'theta=1 realized', 'sum nu G', 'theta=0.5 realized', 'sum nu G', '(e) holds', "(e') agreement", 'vs baseline'], rows))
    L_.append(f"\nCascades against always-Opus by basis (realized eta/eta_B of the 7 primaries x 2 theta on id_test, CI-decided wins / losses): "
              + '; '.join(f"{b.replace('_', ' ')}: {cb[b]['wins_decided']} wins, {cb[b]['losses_decided']} losses, median eta/eta_B {cb[b]['median_eta_ratio']:.2f}" for b in BASES + ['seconds_hedged'])
              + f". Opus is the fastest model on average ({DATA_CHECKS['mean_latency_s_id_test_by_model'][BASE]:.2f} s; next gpt-5.4 {DATA_CHECKS['mean_latency_s_id_test_by_model']['gpt-5.4']:.2f} s, claude-sonnet-4-6 {DATA_CHECKS['mean_latency_s_id_test_by_model']['claude-sonnet-4-6']:.2f} s; the cheap primaries 12-39 s), so in seconds every primary has a/cbar > 1 and even a hedged cascade pays more time than Opus alone.\n")
    # C3
    L_.append("\n### Verifier noise with the coin re-drawn per resample (C3; `figures/fig3_noise.pdf`)\n")
    rows = [[r['name'], yn(r['low_noise'])] + [('-' if r[f'width_ratio_{k}'] is None else f"{r[f'width_ratio_{k}']:.3f}") for k in ['a', 'c', 'd', 'e', 'f']] + [yn(r['e_sign_same_redrawn']), str(r['f_conclusive_changes']), ', '.join(k for k in CONCS if r['flips'][k]) or 'none'] for r in C3['rows']]
    L_.append(table(['noise setting', 'max beta <= 0.1', 'width ratio (a)', '(c)', '(d)', '(e)', '(f)', '(e) sign unchanged under re-draw', '(f) cells whose conclusiveness changes', 'conclusions flipped'], rows))
    L_.append(f"\nWidth ratio = 95% CI width with the coin re-drawn per resampled task copy / width with X1's fixed draw (median over configs of the setting). Pooled medians: "
              + ', '.join(f"({k}) {C3[f'median_width_ratio_{k}']:.3f}" for k in ['a', 'c', 'd', 'e', 'f']) + f". Over coin-dependent configs only (ratio != 1): "
              + ', '.join(f"({k}) {C3[f'coin_dependent_configs_{k}']['median_ratio_dependent']:.3f} (wider in {C3[f'coin_dependent_configs_{k}']['frac_wider_dependent']:.2f} of {C3[f'coin_dependent_configs_{k}']['n_dependent']})" for k in ['a', 'c', 'd', 'e', 'f'] if C3[f'coin_dependent_configs_{k}']['median_ratio_dependent'] is not None)
              + f". Flips at max(beta) <= 0.1: {C3['n_flips_at_low_noise']}.\n\n"
              "Why the per-task cascade CIs do not widen: with a fixed draw each task carries one realized coin, so the empirical distribution being resampled already has the coin's variance inside it and the task bootstrap reproduces Var_task + Var_coin to first order; the re-draw changes the point estimate (expected coin instead of one realization) and removes the Monte-Carlo error of that single draw, not the CI width. "
              "The record stream is different: one write's coin fixes a cell's record for many later rounds, so coin variance is correlated across rounds and is not reproduced by re-weighting rounds by task with the dynamics fixed -- there the re-drawn bootstrap is wider. X1's limitation statement ('CIs omit coin-flip variance') therefore overstates the effect for per-task statistics and understates it for the stream.\n")
    # C4
    L_.append("\n### Value shift (C4; `figures/fig4_value_shift.pdf`): which criteria are shift-invariant, with the algebraic reason (IDENTITY on id_test primitives; IN-SAMPLE sign flips)\n")
    rows = [[r['criterion'], r['form'], 'yes' if r['shift_invariant'] else 'no', r['algebraic_reason'], r['n'], r['n_S_nonzero'], f"{r['n_changed_c01']} / {r['n_changed_c05']}", f"{r['n_sign_flips_c01']} / {r['n_sign_flips_c05']}", f"{r['max_identity_residual']:.1e}",
             ('never changes: ' + yn(r['n_changed_c01'] == 0 and r['n_changed_c05'] == 0)) if r['shift_invariant'] else ('changed iff S != 0, flips only at S != 0: ' + yn(r['changed_iff_S_nonzero'] and r['flips_only_where_S_nonzero']))] for r in VS]
    L_.append(table(['criterion', 'form', 'shift-invariant', 'algebraic reason', 'n configs', 'n with S != 0', 'changed at c = 0.1 / 0.5', 'sign flips at c = 0.1 / 0.5', 'max |observed - algebraic change|', 'check'], rows))
    fl = RESULTS['c4_value_shift']
    L_.append(f"\nRealized eta - eta_B sign flips on id_test (settings module, Opus reference, dollars): c = 0.1: {fl['realized_sign_flips_from_settings'][0.1] or 'none'}; c = 0.5: {fl['realized_sign_flips_from_settings'][0.5] or 'none'}. "
              f"Noise-threshold decisions (psi/r < threshold) that change at c = 0.5: {fl['noise_threshold_decision_flips_c05']}. The paper's scope paragraph predicts exactly this pattern: the shift enters only through the price eta_B (resp. eta_U) of the resource saving, so a criterion moves by c S/cbar and is invariant iff S = 0, iff the price is exogenous, or iff no value enters (VOI band); a sign flip additionally needs |c S/cbar| > |Q + eta_B S| with the opposite sign, so not every S != 0 criterion flips.\n")
    # claims
    L_.append("\n## Claims\n")
    L_.append(table(['id', 'statement', 'kind', 'result', 'verdict'], [[c['id'], c['statement'], c['kind'], c['result'], f"**{c['verdict']}**"] for c in RESULTS['claims']]))
    # hypothesis violations
    L_.append("\n## Hypothesis violations measured\n")
    rows = []
    for r in SUMM:
        if r['axis'] in ('baseline', 'reference', 'basis') and (not r['name'].startswith('random') or r['name'] == 'random00'):
            ab = [x for x in CAS[r['name']]['rows_ab'] if x['theta'] == 1.0]
            rows.append([r['ref'], r['basis'], f"{np.nanmin([x['vbar_h_over_vbar'] for x in ab]):.2f}-{np.nanmax([x['vbar_h_over_vbar'] for x in ab]):.2f}", f"{np.nanmin([x['cbar_h_over_cbar'] for x in ab]):.2f}-{np.nanmax([x['cbar_h_over_cbar'] for x in ab]):.2f}",
                         f"{min(x['a_over_cbar'] for x in ab):.3f}-{max(x['a_over_cbar'] for x in ab):.2f}", f"{sum(x['sign_real'] > 0 and x['decided'] for x in CAS[r['name']]['rows_ab'])}/{sum(x['decided'] for x in CAS[r['name']]['rows_ab'])}"])
    L_.append(table(['reference', 'basis', 'vbar_h/vbar (theta=1, 7 primaries, probing)', 'cbar_h/cbar', 'a/cbar', 'cascades beating the reference (decided wins/decided)'], rows))
    L_.append(f"\n* The matched hypothesis of eq:floor (vbar_h = vbar, cbar_h = cbar) fails in the same direction against every reference: accepted tasks are the ones the reference also solves (vbar_h/vbar > 1) -- the selection effect X1 identified is not an artefact of the Opus reference. Against the per-task oracle the reference is right on every task the primary solves at theta = 1 (vbar_h = 1), so the floor's optimism is largest there.\n"
              f"* Opus is the fastest model on average (id_test mean latency {DATA_CHECKS['mean_latency_s_id_test_by_model'][BASE]:.2f} s vs {min(v for k, v in DATA_CHECKS['mean_latency_s_id_test_by_model'].items() if k != BASE):.2f}-{max(DATA_CHECKS['mean_latency_s_id_test_by_model'].values()):.2f} s for the others); {DATA_CHECKS['rows_latency_zero']} rows have latency 0 (mostly glm-5, qwen3.5-plus, MiniMax-M2.7), which lowers those primaries' recorded seconds.\n"
              f"* The per-task oracle ties (all 8 models share the maximum) on many tasks (both all-solved and all-failed tasks); ties are broken by the cheapest dollar cost, so the oracle's cost is far below Opus ({1000*REF_INFO['oracle']['id_test_mean_usd']:.2f} vs {1000*REF_INFO['opus']['id_test_mean_usd']:.2f} m$/task) and its efficiency is {REF_INFO['oracle']['probing_eta_dollars']:.0f} score/$ (probing).\n")
    # audit
    A = RESULTS['audit']
    L_.append(f"""
## Audit (self-recomputation of three headline numbers with independent code)
1. (a) baseline agreement over all 14 cascades recomputed with plain pandas group means and no bootstrap machinery: floor {A['a_agree_all14_pandas']['floor']}/14, matched {A['a_agree_all14_pandas']['matched']}/14 (script: {A['a_agree_all14_pandas']['script']['floor']}/14, {A['a_agree_all14_pandas']['script']['matched']}/14; X1 README: 8/14 and 12/14).
2. (c) baseline decided agreement at lambda = eta_B recomputed from the per-cell rows with pandas: {A['c_decided_agreement_pandas']['pandas']:.4f} on {A['c_decided_agreement_pandas']['n_decided']} decided cells (script {A['c_decided_agreement_pandas']['script']:.4f}); X3's per-lambda values at eta_B were 1.00 (dim9) / 0.96 (cell21).
3. (e) baseline stream, seed 0, theta = 1, exact verifier: late-window eta_I = {A['e_seed0_loop']['loop']['eta_I']:.4f}, eta_U = {A['e_seed0_loop']['loop']['eta_U']:.4f} from a plain Python loop over the 20,000 rounds with a per-cell record dictionary, vs {A['e_seed0_loop']['vectorised']['eta_I']:.4f} / {A['e_seed0_loop']['vectorised']['eta_U']:.4f} from the vectorised last-write machinery (max |diff| {A['e_seed0_loop']['max_abs_diff']:.1e}).
Also checked: the baseline reproduces X1 (8/14 vs 12/14; eq:normalized median rel. err +0.153, 13/14 over-prediction) and X3 (pooled Qwen3-Max theta=1 lambda=1: predicted 0.1244 vs measured 0.0706; per-lambda decided agreement 0.88 / 0.86 / 0.96 / 0.97 / 0.97 for lambda = 0.1 / 1 / 10 / eta_B / 100) from the copied loaders and estimators, and X4's signs (Opus/dollars +, Opus/seconds -, DimensionBest -) with a different stream seed.

## Limitations
* The sensitivity design is one-at-a-time around the baseline (plus the reference x basis grid); interactions between lambda, noise and shift are not enumerated, and at an exogenous price the value shift is invariant by algebra, so the shift axis is informative only for the ratio criteria.
* "Reference policy" replaces both the comparison policy and the cascade's fallback (so that eq:floor's reference-restoring fallback holds by construction); a design with the fallback fixed to Opus and only the comparison changed would test a different object and is not run.
* The per-task oracle and the random reference are not feasible policies; they are stress references. The oracle's tie-break (cheapest dollars) makes it very cheap, which drives most cascades below it.
* The record stream is X4's semi-synthetic construction (exploration logger, i.i.d. arrivals, no expiration); its conclusions are about that mechanism on real outcomes, not about a deployed router. The coin-re-drawn record bootstrap re-simulates the states but keeps the arrival order and exploration coins fixed.
* The noise axis for (f) evaluates a per-setting accuracy over few hypothesis-met cells; many settings are undecided (fewer than 5 conclusive cells) and are excluded from the flip fraction, as reported.
* Probing and id_test are random splits of the same pools; "out-of-sample" measures split stability, not distribution shift.
* Seconds use recorded per-call latency with {DATA_CHECKS['rows_latency_zero']} zero-latency rows kept; tokens use `total_tokens`; both inherit the provider's logging.

## Deviations from the protocol
* The reference axis counts DimensionBest, the oracle and the random draws type-balanced in the heatmap (per-setting fractions are also in `results.json`).
* The hedged-seconds accounting is an addition for C1 (realized efficiencies only); the specified bases are dollars, seconds (sequential) and tokens.
* The λ axis does not apply to (b) (ratio-only) and (d) (cost-only); those cells are n/a rather than 0.
* (f) on the non-noise axes is pooled over the 25 noise settings (at beta_fa = 0 the threshold is trivially satisfied, so a single-noise (f) would be one-sided); the noise axis evaluates it per setting.
* {B_REDRAW} coin-re-drawn resamples (instead of 1,000) for the record stream, the heaviest cell; all other bootstraps use {B}.
* C3 was split into C3a (cascade CI widths), C3b (record CI widths) and C3c (no flips at beta <= 0.1) after a quick debugging run (B = 200, 3 noise settings) showed the cascade width ratios near 1; the 2%-widening rule was fixed at that point, before the full run.
""")
    open(os.path.join(OUT, 'README.md'), 'w').write(''.join(L_))
    # summary for paper
    fl = FLIP
    def ff(x):
        return 'undecided' if x is None else f'{x:.2f}'
    S_ = [f"# X16 summary for the paper (numbers with evidence kind; baseline = always-Opus reference, dollars, exact verifier)",
          f"1. Robust across all {len([s for s in SUMM if s['axis'] != 'ref_x_basis']) - 1} one-at-a-time settings (reference, price, basis, noise, shift): {', '.join(CONC_LABEL[k] for k in RESULTS['robust']) or 'none'} [OUT-OF-SAMPLE / SEMI-SYNTHETIC].",
          f"2. The homogeneous floor (a) is optimistic and predicts fewer signs than the state-matched condition: baseline {base['a']['detail']['agree_floor']} vs {base['a']['detail']['agree_matched']} of {base['a']['detail']['n_decided']} decided cascades; flip fractions reference {fl['a']['reference']['frac']:.2f}, lambda {fl['a']['lambda']['frac']:.2f}, basis {fl['a']['basis']['frac']:.2f}, noise {fl['a']['noise']['frac']:.2f}, shift {fl['a']['shift']['frac']:.2f} [OUT-OF-SAMPLE].",
          f"3. eq:normalized over-predicts (b): median rel. err {fmt(base['b']['detail']['median_rel_err'])}; flips: reference {fl['b']['reference']['frac']:.2f}, basis {fl['b']['basis']['frac']:.2f}, noise {fl['b']['noise']['frac']:.2f}, shift {fl['b']['shift']['frac']:.2f} [OUT-OF-SAMPLE].",
          f"4. prop:threshold sign agreement (c) >= 0.90 holds at lambda = eta_B ({base['c']['detail']['agree_decided']:.3f}) but depends on the price: " + ', '.join(f"lambda={r['name'][4:]}: {r['c']['detail']['agree_decided']:.3f}" for r in SUMM if r['axis'] == 'lambda') + f"; the price rule p >= a/cbar scores {base['c']['detail']['naive_price_rule']:.3f} at eta_B [OUT-OF-SAMPLE].",
          f"5. VOI band (d) >= 0.95: baseline {base['d']['detail']['agree_decided']:.3f}; flips: reference {fl['d']['reference']['frac']:.2f}, basis {fl['d']['basis']['frac']:.2f}, noise {fl['d']['noise']['frac']:.2f}; shift-invariant (cost-only) [OUT-OF-SAMPLE].",
          f"6. Records (e): +{base['e']['detail']['per_theta']['1.0']['real'][0]:.1f} score/$ against Opus in dollars (theta=1), {next(r for r in SUMM if r['name']=='seconds')['e']['detail']['per_theta']['1.0']['real'][0]:+.4f} score/s in seconds, {next(r for r in SUMM if r['name']=='dimbest')['e']['detail']['per_theta']['1.0']['real'][0]:+.1f} score/$ against DimensionBest: flips on the basis ({fl['e']['basis']['frac']:.2f}) and reference ({fl['e']['reference']['frac']:.2f}) axes, each predicted by sum nu_j G_j ((e') flip fraction {fl['e_pred']['reference']['frac']:.2f} / {fl['e_pred']['basis']['frac']:.2f}) [SEMI-SYNTHETIC + OOS].",
          f"7. C1 (seconds flips the record and cascade conclusions; Opus fastest): {next(c for c in RESULTS['claims'] if c['id']=='C1')['verdict']}: cascades beating Opus in seconds {cb['seconds']['wins_decided']}/14 sequential, {cb['seconds_hedged']['wins_decided']}/14 hedged (dollars {cb['dollars']['wins_decided']}/14).",
          f"8. C2 (DimensionBest reference flips the record conclusion, predicted by the theory): {next(c for c in RESULTS['claims'] if c['id']=='C2')['verdict']}.",
          f"9. C3 (coin re-drawn per resample): widths of per-task cascade statistics {next(c for c in RESULTS['claims'] if c['id']=='C3a')['verdict']} (median ratio " + ', '.join(f"({k}) {C3[f'coin_dependent_configs_{k}']['median_ratio_dependent']:.2f}" for k in ['a', 'c', 'd'] if C3[f'coin_dependent_configs_{k}']['median_ratio_dependent'] is not None) + f": the fixed draw already carries one coin per task), record stream {next(c for c in RESULTS['claims'] if c['id']=='C3b')['verdict']} (" + ', '.join(f"({k}) {C3[f'coin_dependent_configs_{k}']['median_ratio_dependent']:.2f}" for k in ['e', 'f'] if C3[f'coin_dependent_configs_{k}']['median_ratio_dependent'] is not None) + f"); no flips at beta <= 0.1: {next(c for c in RESULTS['claims'] if c['id']=='C3c')['verdict']} ({C3['n_flips_at_low_noise']} flips).",
          f"10. C4 (value shift moves a criterion by c S/cbar; invariant iff S = 0, exogenous price, or cost-only): {next(c for c in RESULTS['claims'] if c['id']=='C4')['verdict']}; sign flips at c = 0.5: " + ', '.join(f"{r['id']} {r['n_sign_flips_c05']}/{r['n']}" for r in VS if r['id'] in ('floor', 'matched', 'netgain', 'rec_sumG')) + " [IDENTITY + IN-SAMPLE].",
          f"11. Noise-threshold conclusion (f): baseline accuracy {base['f']['detail']['accuracy']:.3f} on {base['f']['detail']['n_conclusive']} conclusive cell configurations (pooled over noise); flips: reference {ff(fl['f']['reference']['frac'])}, basis {ff(fl['f']['basis']['frac'])}, noise per (beta_fa, beta_fr) {ff(fl['f']['noise']['frac'])} ({fl['f']['noise']['n'] - fl['f']['noise']['n_undecided']}/{fl['f']['noise']['n']} decided), per beta_fa level {ff(fl['f']['noise']['by_beta_fa_level']['frac'])} [SEMI-SYNTHETIC + OOS].",
          f"12. Paper text to keep: 'the value zero matters exactly through resources priced at eta_U' -- verified: equal-cost threshold invariant to 1e-15, exogenous-price criteria invariant, ratio criteria move by c S/cbar (identity residual {max(r['max_identity_residual'] for r in VS):.0e})."]
    open(os.path.join(OUT, 'summary_for_paper.md'), 'w').write('\n'.join(S_) + '\n')

if __name__ == '__main__':
    main()

