#!/usr/bin/env python3
"""
X4_memory_tail -- real-data test of Sec. 5 of "Route Anything" (record process, eq:record-rate,
eq:cold, eq:record-surplus, eq:stationary-record-gain, thm:memory, app:memory-overhead,
app:retention, app:memory-regret, thm:tail, lem:app-arm-reveal) on CodeRouterBench.

Everything that is an OUTCOME (score, token-price cost estimate in USD, recorded latency) is read from id_results_long.csv.
The constructed (semi-synthetic) elements are documented in README.md: i.i.d. arrival order of
real id_test tasks, the exploration coin and explored model, verifier-noise coins, expiration
coins, and (for thm:tail) Zipf cell frequencies over real tasks.

numpy / pandas / scipy / matplotlib only.  Fixed seeds.  Runtime ~2-3 min.
"""
import os, json, time
import numpy as np, pandas as pd
from scipy.special import zeta, gamma as Gamma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T0 = time.time()
DATA = os.environ.get('ACROUTER_DATA', 'agent-as-a-router/data')
OUT = os.path.dirname(os.path.abspath(__file__))
FIG = OUT + '/figures'
os.makedirs(FIG, exist_ok=True)

SEEDS = 50            # independent streams
N_EPOCHS = 20000      # epochs per stream
LATE = 5000           # stationary window = last LATE epochs
NBOOT = 500           # task-cluster bootstrap resamples
EPS = 0.25            # exploration probability of the logger (writes happen only on exploration epochs)
THETAS = [1.0, 0.5]
BFA = [0.0, 0.05, 0.1, 0.2, 0.4]
DELTAS = [0.0, 0.01, 0.05]
NOISE_CONFIGS = [(b, 0.0) for b in BFA] + [(0.1, 0.2)]        # (beta_fa, beta_fr)
RESOURCES = ['dollars', 'seconds']
N_GRID = [100, 300, 1000, 3000, 10000, 20000]
RET_GRID = [5, 10, 20, 50, 200, 1000]                           # TTL E and FIFO M
ZIPF_Z = [1.5, 2.0, 3.0]
ZIPF_OMEGA = 0.5
ZIPF_J = 20000
ZIPF_E = [10, 20, 50, 100, 200, 500, 1000, 2000]
ZIPF_N = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
SLOPE_TOL = 0.10

RES = {'experiment_id': 'X4_memory_tail', 'seeds': SEEDS, 'epochs': N_EPOCHS, 'late_window': LATE,
       'nboot': NBOOT, 'eps_explore': EPS, 'claims': {}, 'config': {}, 'tables': {}}

def log(*a):
    print(f'[{time.time()-T0:6.1f}s]', *a, flush=True)

# ----------------------------------------------------------------------------- figure style (paper)
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
    _f, _a = plt.subplots(figsize=(1, 1)); _a.set_xlabel(r'$\alpha_j$'); _f.savefig(FIG + '/_probe.pdf'); plt.close(_f)
    os.remove(FIG + '/_probe.pdf')
except Exception as e:                                        # fall back to mathtext
    USETEX_OK = False
    plt.rcParams.update({'text.usetex': False})
    print('usetex unavailable, falling back to mathtext:', e)
RES['config']['usetex'] = USETEX_OK

# ----------------------------------------------------------------------------- data
log('loading data')
df = pd.read_csv(f'{DATA}/coderouterbench/id_results_long.csv')
df['src'] = df.task_id.str.split('_').str[1]
tasks = df.drop_duplicates('task_id')[['task_id', 'split', 'dimension', 'src']].reset_index(drop=True)
tasks['cell21'] = tasks.dimension + '|' + tasks.src
T = len(tasks)
MODELS = sorted(df.model.unique())
M = len(MODELS)
OPUS = MODELS.index('claude-opus-4-6')
tix = {t: i for i, t in enumerate(tasks.task_id)}
S = np.full((T, M), np.nan); C = np.full((T, M), np.nan); LAT = np.full((T, M), np.nan)
ti = df.task_id.map(tix).values; mi = df.model.map({m: i for i, m in enumerate(MODELS)}).values
S[ti, mi] = df.score.values; C[ti, mi] = df.cost_usd.values; LAT[ti, mi] = df.latency_ms.values / 1000.0
assert not np.isnan(S).any() and not np.isnan(C).any() and not np.isnan(LAT).any()
RESMAT = {'dollars': C, 'seconds': LAT}
CELLS21 = sorted(tasks.cell21.unique()); J21 = len(CELLS21)
cell21 = tasks.cell21.map({c: i for i, c in enumerate(CELLS21)}).values
DIMS = sorted(tasks.dimension.unique()); J9 = len(DIMS)
cell9 = tasks.dimension.map({d: i for i, d in enumerate(DIMS)}).values
PROBE = np.where(tasks.split.values == 'probing')[0]; TEST = np.where(tasks.split.values == 'id_test')[0]
assert T == 9999 and len(PROBE) == 7080 and len(TEST) == 2919 and J21 == 21
HIT = {th: (S >= th) for th in THETAS}

# cold policies: DimensionBest fitted on probing (argmax of mean score per dimension); always-Opus
dimbest9 = np.array([S[PROBE][cell9[PROBE] == d].mean(0).argmax() for d in range(J9)])
COLD = {'dimbest': dimbest9[cell9], 'opus': np.full(T, OPUS)}
RES['config']['dimbest_probing_argmax'] = {DIMS[d]: MODELS[dimbest9[d]] for d in range(J9)}
rel = pd.DataFrame([json.loads(l) for l in open(f'{DATA}/id/voter_decisions/dimension_best.jsonl')])
rel = rel[rel.task_id.isin(tix)]
rel_t = rel.task_id.map(tix).values; rel_m = rel.chosen_model.map({m: i for i, m in enumerate(MODELS)}).values
RES['config']['crosscheck'] = {
    'always_opus_id_test_mean_score': float(S[TEST, OPUS].mean()),
    'always_opus_id_test_total_usd': float(C[TEST, OPUS].sum()),
    'dimbest_probing_fit_id_test_mean_score': float(S[TEST, COLD['dimbest'][TEST]].mean()),
    'released_dimension_best_id_test_mean_score': float(S[rel_t, rel_m].mean()),
    'released_differs_on_dims': sorted(set(tasks.dimension.values[rel_t][rel_m != COLD['dimbest'][rel_t]])),
    'zero_cost_entries': int((C <= 0).sum()), 'zero_latency_entries': int((LAT <= 0).sum()),
}
log('cross-check', RES['config']['crosscheck'])

nu_test = np.bincount(cell21[TEST], minlength=J21) / len(TEST)
nu_probe = np.bincount(cell21[PROBE], minlength=J21) / len(PROBE)
cold_cell = {c: np.array([COLD[c][np.where(cell21 == j)[0][0]] for j in range(J21)]) for c in COLD}

# ----------------------------------------------------------------------------- primitives of the record model
def primitives(est_idx, theta, bfa, bfr, delta, cold_cell_models, nu, resource, mode='explore', eps=EPS, cell_arr=None, J=None):
    """Per-cell primitives of Sec. 5 estimated on est_idx (task indices).
    mode='explore': the logger fields a uniform random model on an exploration epoch (prob eps) and
      writes the record iff the (noisy) verifier accepts; non-exploration epochs field the policy's model.
      Values/costs in state s are the conditional means INCLUDING exploration epochs (state-independent).
    mode='shadow': every epoch fields the policy's model AND runs a shadow uniform model whose verified
      outcome is written; the shadow call's cost is part of every state's cost (app:memory-overhead)."""
    H = HIT[theta].astype(float); R = RESMAT[resource]
    cell_arr = cell21 if cell_arr is None else cell_arr; J = J21 if J is None else J
    cj = cell_arr[est_idx]; cnt = np.bincount(cj, minlength=J).astype(float)
    P = np.zeros((J, M)); V = np.zeros((J, M)); Cc = np.zeros((J, M))
    for m in range(M):
        P[:, m] = np.bincount(cj, weights=H[est_idx, m], minlength=J)
        V[:, m] = np.bincount(cj, weights=S[est_idx, m], minlength=J)
        Cc[:, m] = np.bincount(cj, weights=R[est_idx, m], minlength=J)
    P /= cnt[:, None]; V /= cnt[:, None]; Cc /= cnt[:, None]
    scale = eps if mode == 'explore' else 1.0
    omega = scale * (1 - bfr) * P.mean(1); psi = scale * bfa * (1 - P).mean(1); r = omega + psi
    sP = P.sum(1); sQ = (1 - P).sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        qI = np.where(sP[:, None] > 0, P / sP[:, None], np.nan)
        qW = np.where(sQ[:, None] > 0, (1 - P) / sQ[:, None], np.nan)
    ar = np.arange(J)
    Vcold = V[ar, cold_cell_models]; Ccold = Cc[ar, cold_cell_models]
    VI = np.nansum(qI * V, 1); CI = np.nansum(qI * Cc, 1); VW = np.nansum(qW * V, 1); CW = np.nansum(qW * Cc, 1)
    VI[sP == 0] = np.nan; CI[sP == 0] = np.nan; VW[sQ == 0] = np.nan; CW[sQ == 0] = np.nan
    if mode == 'explore':
        vU = (1 - eps) * Vcold + eps * V.mean(1); cU = (1 - eps) * Ccold + eps * Cc.mean(1)
        vI = (1 - eps) * VI + eps * V.mean(1); cI = (1 - eps) * CI + eps * Cc.mean(1)
        vW = (1 - eps) * VW + eps * V.mean(1); cW = (1 - eps) * CW + eps * Cc.mean(1)
    else:
        vU = Vcold; cU = Ccold + Cc.mean(1)
        vI = VI; cI = CI + Cc.mean(1)
        vW = VW; cW = CW + Cc.mean(1)
    eta_U = float((nu * vU).sum() / (nu * cU).sum())
    gI = (vI - vU) - eta_U * (cI - cU); gW = (vW - vU) - eta_U * (cW - cU)
    alpha = (1 - delta) * (1 - nu * r)
    num = np.where(omega > 0, omega * np.nan_to_num(gI), 0.0) + np.where(psi > 0, psi * np.nan_to_num(gW), 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        G = np.where(r > 0, nu * num / (1 - alpha), 0.0)
        err = np.where(r > 0, psi / r, np.nan)
        thr = gI / (gI - gW)
    return dict(P=P, V=V, Cc=Cc, omega=omega, psi=psi, r=r, alpha=alpha, vU=vU, cU=cU, vI=vI, cI=cI, vW=vW, cW=cW,
                eta_U=eta_U, gI=gI, gW=gW, G=G, err=err, thr=thr, nu=nu, qI=qI, qW=qW, cnt=cnt,
                sI=vI - eta_U * cI, sU=vU - eta_U * cU, sW=vW - eta_U * cW,
                sO_cell=np.max(V - eta_U * Cc, 1) if mode == 'explore' else None,
                Vcold=Vcold, Ccold=Ccold, shadow_cost=Cc.mean(1))

def finite_memory_sum(pr, N):
    """RHS of eq:finite-memory: sum_j nu_j G_j [N - (1-alpha_j^N)/(1-alpha_j)]."""
    a = pr['alpha']
    with np.errstate(invalid='ignore', divide='ignore'):
        br = np.where(a < 1, N - (1 - a ** N) / (1 - a), 0.0)
    return float((pr['nu'] * pr['G'] * br).sum())

def record_probs(pr, n):
    """eq:record-probs: i_{j,n}, w_{j,n} for array n."""
    a = pr['alpha'][:, None]; n = np.asarray(n)[None, :]
    with np.errstate(invalid='ignore', divide='ignore'):
        istar = np.where(pr['r'] > 0, pr['nu'] * pr['omega'] / (1 - pr['alpha']), 0.0)[:, None]
        wstar = np.where(pr['r'] > 0, pr['nu'] * pr['psi'] / (1 - pr['alpha']), 0.0)[:, None]
    return istar * (1 - a ** n), wstar * (1 - a ** n)

# ----------------------------------------------------------------------------- record-process machinery
def last_write_before(j, has_write, J):
    """For each epoch (seed s, epoch n): flat index (s*N+n') of the last write in the same (s, cell j[s,n])
    strictly before n, or -1.  Vectorised over all seeds and cells."""
    Sd, N = j.shape
    s_idx = np.repeat(np.arange(Sd), N); n_idx = np.tile(np.arange(N), Sd)
    key = s_idx * J + j.ravel()
    order = np.lexsort((n_idx, key))
    key_s = key[order]; hw_s = has_write.ravel()[order]
    L = Sd * N; pos = np.arange(L)
    newg = np.r_[True, key_s[1:] != key_s[:-1]]
    gstart = np.maximum.accumulate(np.where(newg, pos, 0))
    cum = np.maximum.accumulate(np.where(hw_s, pos, -1))
    last_incl = np.where(cum >= gstart, cum, -1)
    last_excl = np.full(L, -1); last_excl[1:] = last_incl[:-1]; last_excl[newg] = -1
    lw_s = np.where(last_excl >= 0, order[np.maximum(last_excl, 0)], -1)
    inv = np.empty(L, np.int64); inv[order] = np.arange(L)
    return lw_s[inv].reshape(Sd, N)

def cell_last_write(j, has_write, jj):
    """Flat index of the last write to cell jj strictly before each epoch (all epochs, not only arrivals)."""
    Sd, N = j.shape
    pos = np.arange(N)[None, :] + (np.arange(Sd) * N)[:, None]
    cum = np.maximum.accumulate(np.where((j == jj) & has_write, pos, -1), axis=1)
    lw = np.full((Sd, N), -1, np.int64); lw[:, 1:] = cum[:, :-1]
    return lw

def gather(arr, lw):
    return np.where(lw >= 0, arr.ravel()[np.maximum(lw, 0)], -1)

class Writes:
    """Write events of one configuration on a shared stream (common random numbers across configs)."""
    def __init__(self, tid, j, theta, bfa, bfr, delta, explore, mexp, u, ulife, mode='explore'):
        Sd, N = tid.shape
        H = HIT[theta]
        hit_m = H[tid, mexp]
        acc_v = np.where(hit_m, u >= bfr, u < bfa)
        acc = (explore & acc_v) if mode == 'explore' else acc_v
        self.acc = acc
        self.wtype = np.where(acc, np.where(hit_m, 1, 2), 0).astype(np.int8)
        self.wmodel = mexp
        if delta > 0:                                     # coins survived before expiring: P(L>=k)=(1-delta)^k
            self.life = np.floor(np.log(ulife) / np.log(1 - delta)).astype(np.int64)
        else:
            self.life = np.full((Sd, N), 10 ** 12, np.int64)
        self.gcum = np.cumsum(acc, 1)                     # writes up to and including n
        self.gbefore = np.zeros((Sd, N), np.int64); self.gbefore[:, 1:] = self.gcum[:, :-1]
        self.n_idx = np.broadcast_to(np.arange(N), (Sd, N))
        self.tid, self.j, self.explore, self.mexp, self.theta, self.delta = tid, j, explore, mexp, theta, delta
        self.lw = last_write_before(j, acc, int(j.max()) + 1)

    def arriving_state(self, retention='geometric', E=None, Mfifo=None, lw=None):
        """State (0=U,1=I,2=W) and recorded model of the ARRIVING cell at each epoch."""
        lw = self.lw if lw is None else lw
        has = lw >= 0
        n_w = lw % self.tid.shape[1]
        n = self.n_idx
        if retention == 'geometric':
            valid = has & (n - n_w - 1 <= gather(self.life, lw))
        elif retention == 'ttl':
            valid = has & (n - n_w <= E)
        elif retention == 'fifo':
            valid = has & (self.gbefore - gather(self.gcum, lw) <= Mfifo - 1)
        elif retention == 'none':
            valid = has
        state = np.where(valid, gather(self.wtype, lw), 0)
        recm = np.where(valid, gather(self.wmodel, lw), -1)
        return state, recm

def evaluate(W, state, recm, coldmodel, resource, mode='explore'):
    """Epoch outcomes of the informed policy and the paired cold policy on the same tasks."""
    tid = W.tid; R = RESMAT[resource]; H = HIT[W.theta]
    informed = np.where(state > 0, recm, coldmodel[tid])
    if mode == 'explore':
        f_I = np.where(W.explore, W.mexp, informed); f_U = np.where(W.explore, W.mexp, coldmodel[tid])
        extra = 0.0
    else:
        f_I = informed; f_U = coldmodel[tid]; extra = R[tid, W.mexp]
    return dict(V=S[tid, f_I], D=R[tid, f_I] + extra, hit=H[tid, f_I],
                VU=S[tid, f_U], DU=R[tid, f_U] + extra, hitU=H[tid, f_U], f_I=f_I)

# ----------------------------------------------------------------------------- bootstrap helpers
rng_boot = np.random.default_rng(777)
WBOOT = rng_boot.multinomial(len(TEST), np.ones(len(TEST)) / len(TEST), size=NBOOT).astype(float)   # (NBOOT, 2919)
test_pos = np.full(T, -1); test_pos[TEST] = np.arange(len(TEST))

def task_sums(values, tid):
    """Per-id_test-task sums of epoch-level values (values, tid same shape)."""
    return np.bincount(test_pos[tid.ravel()], weights=np.asarray(values, float).ravel(), minlength=len(TEST))

def boot_ratio(num_t, den_t):
    """Point estimate and 95% CI of sum(num)/sum(den) under task-cluster resampling."""
    est = (WBOOT @ num_t) / (WBOOT @ den_t)
    return float(num_t.sum() / den_t.sum()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))

def boot_ratio_diff(numA, denA, numB, denB):
    est = (WBOOT @ numA) / (WBOOT @ denA) - (WBOOT @ numB) / (WBOOT @ denB)
    return float(numA.sum() / denA.sum() - numB.sum() / denB.sum()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))

def boot_sum(num_t):
    """Bootstrap of a total (sum over epochs), preserving the number of epochs in expectation."""
    est = WBOOT @ num_t
    return float(num_t.sum()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))

def seed_ci(per_seed, rng, nboot=NBOOT):
    per_seed = np.asarray(per_seed, float)
    # AUDIT 2026-09-11: seeds whose per-seed mean is undefined (NaN: e.g. a FIFO of size M that never filled
    # in the late window of that stream) were previously propagated as NaN -> the point was counted as NOT
    # covered.  The mark law is independent of the write count, so dropping such seeds is unbiased.
    per_seed = per_seed[np.isfinite(per_seed)]
    if len(per_seed) == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(per_seed), size=(nboot, len(per_seed)))
    est = per_seed[idx].mean(1)
    return float(per_seed.mean()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))

def loglog_slope(x, y):
    x = np.log(np.asarray(x, float)); y = np.log(np.asarray(y, float))
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 3 else np.nan

# ============================================================================= shared stream
rng = np.random.default_rng(20260911)
TID = rng.choice(TEST, size=(SEEDS, N_EPOCHS), replace=True)      # i.i.d. real id_test tasks
JC = cell21[TID]
EXPLORE = rng.random((SEEDS, N_EPOCHS)) < EPS
MEXP = rng.integers(0, M, size=(SEEDS, N_EPOCHS))
U_VER = rng.random((SEEDS, N_EPOCHS))
U_LIFE = rng.random((SEEDS, N_EPOCHS))
late = slice(N_EPOCHS - LATE, N_EPOCHS)
n_late = np.arange(N_EPOCHS - LATE, N_EPOCHS)
RES['config']['stream'] = {'type': 'iid with replacement from id_test', 'seeds': SEEDS, 'epochs': N_EPOCHS,
                           'nu_source': 'empirical id_test cell frequencies (population of the stream)'}

# ============================================================================= Block 1: main grid
log('Block 1: main grid (theta x noise x delta), 21 cells')
rec_prob_rows = []; fm_rows = []; sign_rows = []; cell_rows = []; ret_rows = []; reg_rows = []
PRIMS = {}
rng_seed = np.random.default_rng(99)
fig1_store = {}
for theta in THETAS:
    for (bfa, bfr) in NOISE_CONFIGS:
        for delta in DELTAS:
            if bfr > 0 and delta > 0:
                continue
            W = Writes(TID, JC, theta, bfa, bfr, delta, EXPLORE, MEXP, U_VER, U_LIFE)
            state, recm = W.arriving_state('geometric')
            # ---- (a) record probabilities for ALL cells at every epoch (mechanism check, in-sample primitives)
            pr_in = primitives(TEST, theta, bfa, bfr, delta, cold_cell['dimbest'], nu_test, 'dollars')
            nn = np.arange(N_EPOCHS)
            i_pred, w_pred = record_probs(pr_in, nn)
            BIN = 250
            for jj in range(J21):
                if pr_in['r'][jj] <= 0:
                    continue
                lwj = cell_last_write(JC, W.acc, jj)
                has = lwj >= 0; n_w = lwj % N_EPOCHS
                valid = has & (W.n_idx - n_w - 1 <= gather(W.life, lwj))
                st = np.where(valid, gather(W.wtype, lwj), 0)
                for sname, code, pred in [('I', 1, i_pred[jj]), ('W', 2, w_pred[jj])]:
                    if (sname == 'I' and pr_in['omega'][jj] <= 0) or (sname == 'W' and pr_in['psi'][jj] <= 0):
                        continue
                    ind = (st == code).astype(float)
                    per_seed_bins = ind.reshape(SEEDS, N_EPOCHS // BIN, BIN).mean(2)          # (Sd, nbins)
                    meas = per_seed_bins.mean(0); se_emp = per_seed_bins.std(0, ddof=1) / np.sqrt(SEEDS)
                    pbin = pred.reshape(N_EPOCHS // BIN, BIN).mean(1)
                    # model-based SE of a bin mean of a stationary indicator with lag-k autocorrelation alpha^k
                    # (the per-cell record chain has eigenvalue alpha_j); floors the empirical SE where it is degenerate
                    al = float(pr_in['alpha'][jj]); kk = np.arange(1, BIN)
                    infl = (BIN + 2 * ((BIN - kk) * al ** kk).sum()) / BIN ** 2
                    se_model = np.sqrt(np.clip(pbin * (1 - pbin), 0, None) * infl / SEEDS)
                    se = np.maximum(se_emp, se_model)
                    z = (meas - pbin) / np.maximum(se, 1e-12)
                    late_ps = ind[:, late].mean(1)
                    m_l, lo_l, hi_l = seed_ci(late_ps, rng_seed)
                    p_l = float(pred[late].mean())
                    kl = np.arange(1, LATE)
                    se_late_model = float(np.sqrt(max(p_l * (1 - p_l), 0) * (LATE + 2 * ((LATE - kl) * al ** kl).sum()) / LATE ** 2 / SEEDS))
                    rec_prob_rows.append(dict(theta=theta, bfa=bfa, bfr=bfr, delta=delta, cell=CELLS21[jj], state=sname,
                                              frac_bins_abs_z_lt2=float((np.abs(z) < 2).mean()), max_abs_z=float(np.abs(z).max()),
                                              rmse_curve=float(np.sqrt(((meas - pbin) ** 2).mean())),
                                              late_meas=m_l, late_lo=lo_l, late_hi=hi_l, late_pred=p_l,
                                              late_covered=bool((lo_l <= p_l <= hi_l) or abs(m_l - p_l) <= 0.005),
                                              late_ci_covers_strict=bool(lo_l <= p_l <= hi_l), late_z_model=float((m_l - p_l) / max(se_late_model, 1e-12)),
                                              se_late_emp=float((hi_l - lo_l) / 3.92), se_late_model=se_late_model, stationary_pred=float(pred[-1]),
                                              alpha=float(pr_in['alpha'][jj]), n_bins=int(N_EPOCHS // BIN)))
                    if theta == 1.0 and delta == 0.0 and bfa in (0.0, 0.2) and CELLS21[jj] in ('algorithm|lcb', 'code_generation|bcb', 'test_generation|he'):
                        fig1_store[(bfa, CELLS21[jj], sname)] = (meas, pbin, se, BIN)
            # ---- (b),(c),(d),(e),(f): per cold policy and resource
            for cold in COLD:
                for resource in RESOURCES:
                    ev = evaluate(W, state, recm, COLD[cold], resource)
                    for est_name, est_idx, nu_est in [('in', TEST, nu_test), ('oos', PROBE, nu_probe)]:
                        pr = primitives(est_idx, theta, bfa, bfr, delta, cold_cell[cold], nu_est, resource)
                        PRIMS[(theta, bfa, bfr, delta, cold, resource, est_name)] = pr
                        eta_U = pr['eta_U']
                        surp = ev['V'] - eta_U * ev['D']; surpU = ev['VU'] - eta_U * ev['DU']
                        # (b) finite-memory identity: cumulative expected surplus at eta_U
                        cum_unp = np.cumsum(surp.mean(0)); cum_pair = np.cumsum((surp - surpU).mean(0))
                        for Nn in N_GRID:
                            num_t = task_sums(surp[:, :Nn] / SEEDS, TID[:, :Nn]); numP_t = task_sums((surp - surpU)[:, :Nn] / SEEDS, TID[:, :Nn])
                            m_u, lo_u, hi_u = boot_sum(num_t); m_p, lo_p, hi_p = boot_sum(numP_t)
                            pred = finite_memory_sum(pr, Nn)
                            scale = Nn * float((pr['nu'] * np.abs(pr['G'])).sum())
                            fm_rows.append(dict(theta=theta, bfa=bfa, bfr=bfr, delta=delta, cold=cold, resource=resource, est=est_name, N=Nn,
                                                pred=pred, meas_unpaired=m_u, lo_u=lo_u, hi_u=hi_u, meas_paired=m_p, lo_p=lo_p, hi_p=hi_p,
                                                cover_unpaired=bool(lo_u <= pred <= hi_u), cover_paired=bool(lo_p <= pred <= hi_p),
                                                norm_err_paired=float(abs(m_p - pred) / scale) if scale > 0 else np.nan,
                                                norm_err_unpaired=float(abs(m_u - pred) / scale) if scale > 0 else np.nan,
                                                rel_err_paired=float(abs(m_p - pred) / abs(pred)) if abs(pred) > 0 else np.nan,
                                                scale=scale))
                        # (c) stationary sign criterion: eta_late(informed) - eta_late(cold), paired on the same epochs
                        nV = task_sums(ev['V'][:, late], TID[:, late]); nD = task_sums(ev['D'][:, late], TID[:, late])
                        nVU = task_sums(ev['VU'][:, late], TID[:, late]); nDU = task_sums(ev['DU'][:, late], TID[:, late])
                        d_eta, lo, hi = boot_ratio_diff(nV, nD, nVU, nDU)
                        eta_I_late = float(nV.sum() / nD.sum()); eta_U_late = float(nVU.sum() / nDU.sum())
                        sumG = float((pr['nu'] * pr['G']).sum())
                        att = 1 - pr['alpha'][:, None] ** n_late[None, :]
                        sumG_late = float((pr['nu'][:, None] * pr['G'][:, None] * att).sum(0).mean())
                        # predicted stationary cost and eq:app-memory-relative-efficiency
                        istar, wstar = record_probs(pr, np.array([10 ** 9]))
                        cinf = float((pr['nu'] * (pr['cU'] + istar[:, 0] * np.nan_to_num(pr['cI'] - pr['cU']) + wstar[:, 0] * np.nan_to_num(pr['cW'] - pr['cU']))).sum())
                        pred_deta = sumG / cinf
                        meas_sign = 'pos' if lo > 0 else ('neg' if hi < 0 else 'incl0')
                        sign_rows.append(dict(theta=theta, bfa=bfa, bfr=bfr, delta=delta, cold=cold, resource=resource, est=est_name,
                                              sumG=sumG, sumG_late=sumG_late, pred_sign='pos' if sumG >= 0 else 'neg', pred_deta=pred_deta,
                                              eta_U_prim=eta_U, eta_I_late=eta_I_late, eta_U_late=eta_U_late,
                                              d_eta=d_eta, lo=lo, hi=hi, meas_sign=meas_sign,
                                              agree=(meas_sign == ('pos' if sumG >= 0 else 'neg')) if meas_sign != 'incl0' else None,
                                              n_cells_r_pos=int((pr['r'] > 0).sum())))
                        # (d) per-cell noise threshold: paired surplus difference on late epochs of cell j
                        diff = (surp - surpU)[:, late]; jl = JC[:, late]; tl = TID[:, late]
                        num_t = task_sums(diff, tl); cnt_t = task_sums(np.ones_like(diff), tl)
                        cellpos = cell21[TEST]
                        for jj in range(J21):
                            if pr['r'][jj] <= 0:
                                continue
                            sel = cellpos == jj
                            m, lo_c, hi_c = boot_ratio(num_t * sel, cnt_t * sel) if cnt_t[sel].sum() > 0 else (np.nan, np.nan, np.nan)
                            hyp = bool(np.isfinite(pr['gI'][jj]) and np.isfinite(pr['gW'][jj]) and pr['gI'][jj] > 0 > pr['gW'][jj])
                            pred_pos = bool(pr['err'][jj] < pr['thr'][jj]) if hyp else None
                            ms = 'pos' if lo_c > 0 else ('neg' if hi_c < 0 else 'incl0')
                            cell_rows.append(dict(theta=theta, bfa=bfa, bfr=bfr, delta=delta, cold=cold, resource=resource, est=est_name,
                                                  cell=CELLS21[jj], gI=float(pr['gI'][jj]), gW=float(pr['gW'][jj]), err=float(pr['err'][jj]),
                                                  thr=float(pr['thr'][jj]) if hyp else np.nan, hyp=hyp, pred_pos=pred_pos,
                                                  G=float(pr['G'][jj]), meas=m, lo=lo_c, hi=hi_c, meas_sign=ms,
                                                  omega=float(pr['omega'][jj]), psi=float(pr['psi'][jj]), nu=float(pr['nu'][jj])))
                        # (f) oracle regret (delta = 0, empty start): s = v - eta_U c, s^O = best fixed arm of the cell
                        if delta == 0.0 and est_name in ('in', 'oos'):
                            sO = pr['sO_cell'][JC]
                            regret = sO - surp                                          # (Sd, N)
                            r_ = pr['r']; nu_ = pr['nu']
                            with np.errstate(invalid='ignore', divide='ignore'):
                                sinf = np.where(r_ > 0, (pr['omega'] * np.nan_to_num(pr['sI']) + pr['psi'] * np.nan_to_num(pr['sW'])) / r_, pr['sU'])
                            for Nn in N_GRID:
                                num_t = task_sums(regret[:, :Nn] / SEEDS, TID[:, :Nn])
                                m, lo_r, hi_r = boot_sum(num_t)
                                lin = Nn * float((nu_ * (pr['sO_cell'] - sinf)).sum())
                                with np.errstate(invalid='ignore', divide='ignore'):
                                    trans = float(np.where(r_ > 0, (1 - (1 - nu_ * r_) ** Nn) / np.where(r_ > 0, r_, 1) * (sinf - pr['sU']), 0.0).sum())
                                pred = lin + trans
                                reg_rows.append(dict(theta=theta, bfa=bfa, bfr=bfr, cold=cold, resource=resource, est=est_name, N=Nn,
                                                     pred=pred, pred_linear=lin, pred_transient=trans, meas=m, lo=lo_r, hi=hi_r,
                                                     cover=bool(lo_r <= pred <= hi_r), rel_err=float(abs(m - pred) / abs(pred)) if pred != 0 else np.nan,
                                                     n_cells_sinf_below_sU=int(((sinf < pr['sU']) & (r_ > 0)).sum()),
                                                     transient_limit=float(np.where(r_ > 0, (sinf - pr['sU']) / np.where(r_ > 0, r_, 1), 0.0).sum())))
            log(f'  theta={theta} bfa={bfa} bfr={bfr} delta={delta} done')

RP = pd.DataFrame(rec_prob_rows); FM = pd.DataFrame(fm_rows); SG = pd.DataFrame(sign_rows)
CL = pd.DataFrame(cell_rows); RG = pd.DataFrame(reg_rows)

# ---- verdicts (a)
a_frac = float(RP.frac_bins_abs_z_lt2.mean()); a_late = float(RP.late_covered.mean()); a_maxz = float(RP.max_abs_z.max())
a_strict = float(RP.late_ci_covers_strict.mean()); a_zl = RP.late_z_model.to_numpy()
RES['claims']['a_record_probs'] = dict(
    result='thm:memory eq:record-probs (and recurrence eq:app-memory-recurrence)', kind='IDENTITY / mechanism check + sampling check',
    hypotheses='iid arrivals nu_j; state-independent write types; expiration then write; last write wins; r_j>0',
    n_cell_state_configs=int(len(RP)), mean_frac_bins_abs_z_lt2=a_frac, frac_late_window_ci_covers_pred=a_late,
    max_abs_z_any_bin=a_maxz, median_rmse_curve=float(RP.rmse_curve.median()),
    frac_late_ci_covers_strict=a_strict, late_z_model_mean=float(a_zl.mean()), late_z_model_sd=float(a_zl.std()),
    frac_late_abs_z_model_lt2=float((np.abs(a_zl) < 2).mean()),
    se_note='bin and late-window SEs are max(empirical seed SE, model SE with lag-k autocorrelation alpha_j^k); late coverage = CI covers OR |meas-pred| <= 0.005 (degenerate CIs at coverage 1 or 0)',
    criterion='PASS if mean fraction of 250-epoch bins with |z|<2 >= 0.90 and late-window prediction is covered (CI or 0.005 tolerance) in >= 0.90 of (cell,state,config)',
    verdict='PASS' if (a_frac >= 0.90 and a_late >= 0.90) else 'FAIL')
log('(a)', RES['claims']['a_record_probs'])

# ---- verdicts (b)
fm_in = FM[FM.est == 'in']; fm_oos = FM[FM.est == 'oos']
b_cov_p = float(fm_in.cover_paired.mean()); b_cov_u = float(fm_in.cover_unpaired.mean())
b_norm = float(fm_in[fm_in.N == N_EPOCHS].norm_err_paired.median())
big = fm_oos[(fm_oos.N == N_EPOCHS) & (fm_oos.scale > 0)]
oos_sign = float((np.sign(big.pred) == np.sign(big.meas_unpaired)).mean())
oos_norm = float(big.norm_err_unpaired.median())
RES['claims']['b_finite_memory'] = dict(
    result='thm:memory eq:finite-memory (= eq:app-memory-finite-surplus)', kind='IDENTITY + sampling check (in-sample primitives); OUT-OF-SAMPLE prediction (probing primitives, probing nu)',
    n_points_in=int(len(fm_in)), frac_ci_covers_pred_paired=b_cov_p, frac_ci_covers_pred_unpaired=b_cov_u,
    median_norm_err_paired_at_N20000=b_norm, note_norm='normalised error = |meas-pred| / (N sum_j nu_j |G_j|)',
    oos_frac_sign_agree_N20000=oos_sign, oos_median_norm_err_N20000=oos_norm, oos_frac_ci_cover_N20000=float(big.cover_unpaired.mean()),
    oos_paired_frac_ci_cover_N20000=float(big.cover_paired.mean()), oos_paired_median_norm_err_N20000=float(big.norm_err_paired.median()),
    oos_note='unpaired OOS measurement contains the cold-term shift N*J_U(eta_U^probe) on id_test; paired removes it',
    criterion='IDENTITY PASS if paired task-bootstrap CI covers the prediction in >= 0.90 of (config,N) points and median normalised error at N=20000 < 0.05; OOS PASS if sign agrees in >= 0.80 of configs and median normalised error < 0.15',
    verdict_identity='PASS' if (b_cov_p >= 0.90 and b_norm < 0.05) else 'FAIL',
    verdict_oos='PASS' if (oos_sign >= 0.80 and oos_norm < 0.15) else 'FAIL')
log('(b)', RES['claims']['b_finite_memory'])

# ---- verdicts (c)
def confusion(df_, pred_col='pred_sign', meas_col='meas_sign'):
    c = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'inconclusive': 0}
    for _, r in df_.iterrows():
        if r[meas_col] == 'incl0':
            c['inconclusive'] += 1
        elif r[pred_col] == 'pos' and r[meas_col] == 'pos': c['TP'] += 1
        elif r[pred_col] == 'neg' and r[meas_col] == 'neg': c['TN'] += 1
        elif r[pred_col] == 'pos' and r[meas_col] == 'neg': c['FP'] += 1
        else: c['FN'] += 1
    n = c['TP'] + c['TN'] + c['FP'] + c['FN']
    c['accuracy'] = (c['TP'] + c['TN']) / n if n else np.nan
    c['n_conclusive'] = n
    return c
sg_in = SG[SG.est == 'in']; sg_oos = SG[SG.est == 'oos']
conf_in = confusion(sg_in); conf_oos = confusion(sg_oos)
conf_by = {}
for keyn, g in sg_in.groupby(['cold', 'resource']):
    conf_by['|'.join(keyn)] = confusion(g)
for keyn, g in sg_in.groupby(['bfa']):
    conf_by[f'bfa={keyn[0]}'] = confusion(g)
for keyn, g in sg_in.groupby(['delta']):
    conf_by[f'delta={keyn[0]}'] = confusion(g)
# exact relative-efficiency check (eq:app-memory-relative-efficiency) on in-sample rows
sg_in = sg_in.copy(); sg_in['deta_cover'] = (sg_in.lo <= sg_in.pred_deta) & (sg_in.pred_deta <= sg_in.hi)
RES['claims']['c_stationary_sign'] = dict(
    result='thm:memory stationary criterion eta_inf >= eta_U iff sum_j nu_j G_j >= 0; eq:app-memory-relative-efficiency',
    kind='IN-SAMPLE hypothesis test (primitives = population of the stream) and OUT-OF-SAMPLE (probing primitives)',
    n_configs=int(len(sg_in)), confusion_in=conf_in, confusion_oos=conf_oos, confusion_by_group_in=conf_by,
    frac_pos_pred_in=float((sg_in.pred_sign == 'pos').mean()), frac_deta_ci_covers_pred_in=float(sg_in.deta_cover.mean()),
    n_configs_late_transient_flips_sign=int((np.sign(sg_in.sumG) != np.sign(sg_in.sumG_late)).sum()),
    criterion='PASS if accuracy >= 0.95 among conclusive configs (in-sample) and both signs occur among predictions',
    verdict_in='PASS' if (conf_in['accuracy'] >= 0.95 and 0 < (sg_in.pred_sign == 'pos').mean() < 1) else 'FAIL',
    verdict_oos='PASS' if conf_oos['accuracy'] >= 0.90 else 'FAIL')
log('(c)', json.dumps(RES['claims']['c_stationary_sign'], default=str)[:1500])

# ---- verdicts (d)
cl_h = CL[CL.hyp & CL.pred_pos.notna()].copy()
cl_h['pred_sign'] = np.where(cl_h.pred_pos.astype(bool), 'pos', 'neg')
conf_d_in = confusion(cl_h[cl_h.est == 'in']); conf_d_oos = confusion(cl_h[cl_h.est == 'oos'])
# which cells cross the threshold as beta_fa grows (in-sample, per theta/cold/resource/delta)
cross = []
for keyn, g in cl_h[(cl_h.est == 'in') & (cl_h.bfr == 0)].groupby(['theta', 'delta', 'cold', 'resource', 'cell']):
    g = g.sort_values('bfa')
    if g.pred_pos.nunique() > 1:
        cross.append(dict(zip(['theta', 'delta', 'cold', 'resource', 'cell'], keyn), bfa_seq=list(g.bfa), pred=list(g.pred_pos.astype(bool)),
                          meas=list(g.meas_sign), thr=list(np.round(g.thr, 3)), err=list(np.round(g.err, 3))))
RES['claims']['d_noise_threshold'] = dict(
    result='thm:memory eq:noise-threshold (cells with g_j^I > 0 > g_j^W)', kind='IN-SAMPLE / OUT-OF-SAMPLE hypothesis test per cell',
    n_cell_configs_total=int(len(CL[CL.est == 'in'])), n_hypothesis_met_in=int((CL[CL.est == 'in'].hyp).sum()),
    n_hypothesis_met_oos=int((CL[CL.est == 'oos'].hyp).sum()),
    confusion_in=conf_d_in, confusion_oos=conf_d_oos, n_threshold_crossings_in=len(cross), crossings=cross[:40],
    cells_meeting_hypothesis_in=sorted(cl_h[cl_h.est == 'in'].cell.unique().tolist()),
    n_conclusive_predicted_neg_in=int(conf_d_in['TN'] + conf_d_in['FP']), n_conclusive_predicted_neg_oos=int(conf_d_oos['TN'] + conf_d_oos['FP']),
    criterion='PASS if sign accuracy >= 0.90 among conclusive hypothesis-met cell-configs; flagged one-sided if no conclusive cell-config has a predicted negative sign',
    verdict_in=('PASS' + (' (one-sided: only the err<thr side is conclusive in-sample)' if conf_d_in['TN'] + conf_d_in['FP'] == 0 else '')) if (conf_d_in['n_conclusive'] > 0 and conf_d_in['accuracy'] >= 0.90) else ('NOT INSTANTIABLE' if conf_d_in['n_conclusive'] == 0 else 'FAIL'),
    verdict_oos='PASS' if (conf_d_oos['n_conclusive'] > 0 and conf_d_oos['accuracy'] >= 0.90) else ('NOT INSTANTIABLE' if conf_d_oos['n_conclusive'] == 0 else 'FAIL'))
log('(d)', json.dumps({k: v for k, v in RES['claims']['d_noise_threshold'].items() if k != 'crossings'}, default=str))

# ---- verdicts (f)
rg_in = RG[RG.est == 'in']; rg_oos = RG[RG.est == 'oos']
f_cov = float(rg_in.cover.mean()); f_rel = float(rg_in[rg_in.N == N_EPOCHS].rel_err.median())
f_oos_rel = float(rg_oos[rg_oos.N == N_EPOCHS].rel_err.median()); f_oos_cov = float(rg_oos[rg_oos.N == N_EPOCHS].cover.mean())
RES['claims']['f_regret'] = dict(
    result='app:memory-regret eq:app-regret (delta=0, empty start, s = v - eta_U c, s^O = best fixed arm per cell)',
    kind='IDENTITY + sampling check (in-sample); OUT-OF-SAMPLE prediction (probing primitives)',
    n_points_in=int(len(rg_in)), frac_ci_covers_in=f_cov, median_rel_err_in_N20000=f_rel,
    median_rel_err_oos_N20000=f_oos_rel, frac_ci_covers_oos_N20000=f_oos_cov,
    n_configs_with_signed_transient=int((rg_in[rg_in.N == N_EPOCHS].n_cells_sinf_below_sU > 0).sum()), n_configs=int((rg_in.N == N_EPOCHS).sum()),
    criterion='IDENTITY PASS if CI covers in >= 0.90 of points and median relative error at N=20000 < 0.05; OOS PASS if median relative error < 0.15',
    verdict_identity='PASS' if (f_cov >= 0.90 and f_rel < 0.05) else 'FAIL', verdict_oos='PASS' if f_oos_rel < 0.15 else 'FAIL')
log('(f)', RES['claims']['f_regret'])


# ============================================================================= Block 1e: retention (TTL / FIFO) on a larger write-only replay
RET_SEEDS = 300
log('Block 1e: TTL / FIFO coverage (dedicated replay, real outcomes, %d seeds)' % RET_SEEDS)
rng_r = np.random.default_rng(555)
TID_R = rng_r.choice(TEST, size=(RET_SEEDS, N_EPOCHS), replace=True); JC_R = cell21[TID_R]
EXPL_R = rng_r.random((RET_SEEDS, N_EPOCHS)) < EPS; MEXP_R = rng_r.integers(0, M, size=(RET_SEEDS, N_EPOCHS))
U_R = rng_r.random((RET_SEEDS, N_EPOCHS)); UL_R = rng_r.random((RET_SEEDS, N_EPOCHS))
ret_rows = []; agg_rows = []
for theta in THETAS:
    for bfa in [0.0, 0.2]:
        W = Writes(TID_R, JC_R, theta, bfa, 0.0, 0.0, EXPL_R, MEXP_R, U_R, UL_R)
        pr = primitives(TEST, theta, bfa, 0.0, 0.0, cold_cell['dimbest'], nu_test, 'dollars')
        beta = float((pr['nu'] * pr['r']).sum()); q = pr['nu'] * pr['r'] / beta; x = pr['nu'] * pr['r']
        n_idx = W.n_idx
        lwA = W.lw; hasA = lwA >= 0; nwA = lwA % N_EPOCHS; gwA = gather(W.gcum, lwA)
        for E in RET_GRID:
            covT = (hasA & (n_idx - nwA <= E))[:, late].mean(1)
            full = (W.gbefore >= E)[:, late]
            covF_raw = (hasA & (W.gbefore - gwA <= E - 1))[:, late]
            covF = np.nanmean(np.where(full, covF_raw, np.nan), 1)
            mT, loT, hiT = seed_ci(covT, rng_r); mF, loF, hiF = seed_ci(covF, rng_r)
            n_seeds_fifo_defined = int(np.isfinite(covF).sum())                                   # AUDIT
            exT = float((pr['nu'] * (1 - (1 - x) ** E)).sum()); exF = float((pr['nu'] * (1 - (1 - q) ** E)).sum())
            naive = float((pr['nu'] * (1 - (1 - x) ** (E / beta))).sum())
            agg_rows.append(dict(theta=theta, bfa=bfa, E=E, beta=beta, ttl_meas=mT, ttl_lo=loT, ttl_hi=hiT, ttl_pred=exT, ttl_cover=bool(loT <= exT <= hiT),
                                 fifo_meas=mF, fifo_lo=loF, fifo_hi=hiF, fifo_pred=exF, fifo_cover=bool(loF <= exF <= hiF),
                                 naive_pred=naive, naive_cover=bool(loF <= naive <= hiF), naive_minus_exact=naive - exF,
                                 z_exact=float((mF - exF) / max((hiF - loF) / 3.92, 1e-12)), z_naive=float((mF - naive) / max((hiF - loF) / 3.92, 1e-12)),
                                 frac_epochs_fifo_full=float(full.mean()), n_seeds_fifo_defined=n_seeds_fifo_defined))
        for jj in range(J21):
            if pr['r'][jj] <= 0:
                continue
            lwj = cell_last_write(JC_R, W.acc, jj)
            has = lwj >= 0; n_w = lwj % N_EPOCHS; g_w = gather(W.gcum, lwj)
            for E in RET_GRID:
                cov_ttl = (has & (n_idx - n_w <= E))[:, late].mean(1)
                full = (W.gbefore >= E)[:, late]
                cov_fifo = np.nanmean(np.where(full, (has & (W.gbefore - g_w <= E - 1))[:, late], np.nan), 1)
                m1, lo1, hi1 = seed_ci(cov_ttl, rng_r); m2, lo2, hi2 = seed_ci(cov_fifo, rng_r)
                n_def = int(np.isfinite(cov_fifo).sum())                                            # AUDIT
                ttl_pred = float(1 - (1 - x[jj]) ** E); fifo_pred = float(1 - (1 - q[jj]) ** E); naive_pred = float(1 - (1 - x[jj]) ** (E / beta))
                ret_rows.append(dict(theta=theta, bfa=bfa, cell=CELLS21[jj], E=E, x=float(x[jj]), q=float(q[jj]), beta=beta,
                                     ttl_meas=m1, ttl_lo=lo1, ttl_hi=hi1, ttl_pred=ttl_pred, ttl_cover=bool((lo1 <= ttl_pred <= hi1) or abs(m1 - ttl_pred) <= 0.005),
                                     ttl_cover_strict=bool(lo1 <= ttl_pred <= hi1),
                                     fifo_meas=m2, fifo_lo=lo2, fifo_hi=hi2, fifo_pred=fifo_pred, fifo_cover=bool((lo2 <= fifo_pred <= hi2) or abs(m2 - fifo_pred) <= 0.005),
                                     fifo_cover_strict=bool(lo2 <= fifo_pred <= hi2),
                                     naive_pred=naive_pred, naive_cover=bool((lo2 <= naive_pred <= hi2) or abs(m2 - naive_pred) <= 0.005),
                                     naive_cover_strict=bool(lo2 <= naive_pred <= hi2),
                                     gap_exact=float(abs(fifo_pred - m2)), gap_naive=float(abs(naive_pred - m2)), naive_minus_exact=naive_pred - fifo_pred,
                                     se_fifo=float((hi2 - lo2) / 3.92), n_seeds_fifo_defined=n_def))
        log(f'  retention theta={theta} bfa={bfa} done')
RT = pd.DataFrame(ret_rows); AG = pd.DataFrame(agg_rows)
e_ttl = float(RT.ttl_cover.mean()); e_fifo = float(RT.fifo_cover.mean()); e_naive = float(RT.naive_cover.mean())
small = RT[RT.E <= 20]; sm_agg = AG[AG.E <= 20]
res_pts = small[(small.naive_minus_exact.abs() > 2 * small.se_fifo)]
sided_exact = float((res_pts.gap_exact < res_pts.gap_naive).mean()) if len(res_pts) else np.nan
RES['claims']['e_retention'] = dict(
    result='app:retention eq:app-ttl (TTL coverage), eq:app-fifo (FIFO coverage), and the statement that E -> M/beta in eq:app-ttl is NOT exact for the FIFO',
    kind='IDENTITY + sampling check for the two formulas; IN-SAMPLE hypothesis test of the inexactness statement',
    replay='dedicated %d-seed x %d-epoch replay with the same real-outcome write mechanism; FIFO measured only on epochs where >= M writes have occurred' % (RET_SEEDS, N_EPOCHS),
    n_cell_E_points=int(len(RT)), frac_ttl_covered=e_ttl, frac_fifo_covered=e_fifo, frac_naive_covered=e_naive,
    n_fifo_points_defined=int(RT.fifo_meas.notna().sum()), min_seeds_fifo_defined=int(RT.n_seeds_fifo_defined.min()),
    audit_note='2026-09-11: seed_ci now drops seeds with an undefined (NaN) FIFO mean; in the original run the 17 cell points and the aggregate at theta=1, beta_fa=0, M=1000 were NaN and were counted as not covered (FIFO coverage 0.907 -> see frac_fifo_covered)',
    frac_ttl_ci_strict=float(RT.ttl_cover_strict.mean()), frac_fifo_ci_strict=float(RT.fifo_cover_strict.mean()), frac_naive_ci_strict=float(RT.naive_cover_strict.mean()),
    max_abs_naive_minus_exact_cell=float(RT.naive_minus_exact.abs().max()), mean_abs_naive_minus_exact_M5=float(RT[RT.E == 5].naive_minus_exact.abs().mean()),
    sign_naive_minus_exact_frac_positive=float((RT.naive_minus_exact > 0).mean()),
    n_resolvable_points_M_le_20=int(len(res_pts)), frac_resolvable_points_measurement_closer_to_exact=sided_exact,
    frac_naive_rejected_resolvable=float((~res_pts.naive_cover_strict).mean()) if len(res_pts) else np.nan,
    frac_exact_rejected_resolvable=float((~res_pts.fifo_cover_strict).mean()) if len(res_pts) else np.nan,
    aggregate_M_le_20_frac_naive_rejected=float((~sm_agg.naive_cover).mean()), aggregate_M_le_20_frac_exact_rejected=float((~sm_agg.fifo_cover).mean()),
    aggregate_M_le_20_mean_abs_z_exact=float(sm_agg.z_exact.abs().mean()), aggregate_M_le_20_mean_abs_z_naive=float(sm_agg.z_naive.abs().mean()),
    criterion='formulas PASS if prediction covered (CI or 0.005) in >= 0.90 of (cell,E) points; inexactness PASS if among resolvable points (|naive-exact| > 2 SE, M<=20) the measurement is closer to the exact formula in >= 0.90 and the exact formula is rejected in < 0.10 of them, and the aggregate arriving-cell coverage rejects the naive formula in >= 0.50 of M<=20 configs',
    verdict_ttl='PASS' if e_ttl >= 0.90 else 'FAIL', verdict_fifo='PASS' if e_fifo >= 0.90 else 'FAIL',
    verdict_inexact=('PASS' if (len(res_pts) >= 10 and sided_exact >= 0.90 and (~res_pts.fifo_cover_strict).mean() < 0.10 and (~sm_agg.naive_cover).mean() >= 0.5)
                     else ('INCONCLUSIVE' if len(res_pts) < 10 else 'FAIL')))
log('(e)', json.dumps(RES['claims']['e_retention'], default=str))
RES['tables']['retention_aggregate'] = AG.round(8).to_dict('records')


# ============================================================================= code checks (brute-force re-implementations on one seed)
log('code checks')
CHK = {}
# (i) FIFO / TTL indicator: brute force on seed 0 of the retention replay (last config), cell code_generation|humaneval
jj = CELLS21.index('code_generation|humaneval'); s = 0; Mf = 5; Et = 50
acc0 = W.acc[s]; j0 = JC_R[s]; wpos = np.where(acc0)[0]; wmark = j0[wpos]
bf_f = []; bf_t = []
for n in range(N_EPOCHS - LATE, N_EPOCHS):
    k = np.searchsorted(wpos, n)
    bf_f.append(jj in wmark[max(0, k - Mf):k])
    pj = wpos[:k][wmark[:k] == jj]
    bf_t.append(len(pj) > 0 and n - pj[-1] <= Et)
lwj = cell_last_write(JC_R, W.acc, jj); has = lwj >= 0; g_w = gather(W.gcum, lwj); n_w = lwj % N_EPOCHS
vec_f = float((has & (W.gbefore - g_w <= Mf - 1))[s, late].mean()); vec_t = float((has & (W.n_idx - n_w <= Et))[s, late].mean())
CHK['fifo_bruteforce_seed0_M5'] = dict(brute=float(np.mean(bf_f)), vectorised=vec_f, equal=bool(abs(np.mean(bf_f) - vec_f) < 1e-12))
CHK['ttl_bruteforce_seed0_E50'] = dict(brute=float(np.mean(bf_t)), vectorised=vec_t, equal=bool(abs(np.mean(bf_t) - vec_t) < 1e-12))
marks = JC_R[W.acc]; qemp = np.bincount(marks, minlength=J21) / len(marks)
CHK['fifo_mark_law_humaneval'] = dict(q_pred=float(q[jj]), q_empirical=float(qemp[jj]), n_writes=int(len(marks)),
                                     note='all four retention configs share one replay stream, so per-cell deviations are correlated across (config, M)')
# (ii) sequential replay of the geometric-expiration record process with the same coins (seed 0, theta=1, bfa=0.2, delta=0.05)
theta_c, bfa_c, delta_c = 1.0, 0.2, 0.05
Wc = Writes(TID, JC, theta_c, bfa_c, 0.0, delta_c, EXPLORE, MEXP, U_VER, U_LIFE)
state_v, recm_v = Wc.arriving_state('geometric')
rec = {}; st_seq = np.zeros(N_EPOCHS, int); rm_seq = np.full(N_EPOCHS, -1)
for n in range(N_EPOCHS):
    c = JC[0, n]
    if c in rec:
        n_w, mdl, typ, life = rec[c]
        if n - n_w - 1 <= life:
            st_seq[n] = typ; rm_seq[n] = mdl
    if Wc.acc[0, n]:
        rec[c] = (n, int(Wc.wmodel[0, n]), int(Wc.wtype[0, n]), int(Wc.life[0, n]))
CHK['sequential_record_replay_seed0'] = dict(states_equal=bool((st_seq == state_v[0]).all()), models_equal=bool((rm_seq == recm_v[0]).all()),
                                             n_epochs=int(N_EPOCHS), frac_I=float((st_seq == 1).mean()), frac_W=float((st_seq == 2).mean()))
assert CHK['fifo_bruteforce_seed0_M5']['equal'] and CHK['ttl_bruteforce_seed0_E50']['equal']
assert CHK['sequential_record_replay_seed0']['states_equal'] and CHK['sequential_record_replay_seed0']['models_equal']
RES['code_checks'] = CHK
log('code checks', json.dumps(CHK))

# ============================================================================= Block 1f: J = 9 dimension cells (variant of (b) and (c))
log('Block 1f: J=9 variant')
nu_test9 = np.bincount(cell9[TEST], minlength=J9) / len(TEST)
cold_cell9 = {c: np.array([COLD[c][np.where(cell9 == j)[0][0]] for j in range(J9)]) for c in COLD}
JC9 = cell9[TID]
v9 = []
for theta in THETAS:
    for bfa in [0.0, 0.2]:
        W = Writes(TID, JC9, theta, bfa, 0.0, 0.0, EXPLORE, MEXP, U_VER, U_LIFE)
        state, recm = W.arriving_state('geometric')
        for cold in COLD:
            for resource in RESOURCES:
                pr = primitives(TEST, theta, bfa, 0.0, 0.0, cold_cell9[cold], nu_test9, resource, cell_arr=cell9, J=J9)
                ev = evaluate(W, state, recm, COLD[cold], resource)
                eta_U = pr['eta_U']; surp = ev['V'] - eta_U * ev['D']; surpU = ev['VU'] - eta_U * ev['DU']
                numP_t = task_sums((surp - surpU) / SEEDS, TID); m_p, lo_p, hi_p = boot_sum(numP_t)
                pred = finite_memory_sum(pr, N_EPOCHS); scale = N_EPOCHS * float((pr['nu'] * np.abs(pr['G'])).sum())
                nV = task_sums(ev['V'][:, late], TID[:, late]); nD = task_sums(ev['D'][:, late], TID[:, late])
                nVU = task_sums(ev['VU'][:, late], TID[:, late]); nDU = task_sums(ev['DU'][:, late], TID[:, late])
                d_eta, lo, hi = boot_ratio_diff(nV, nD, nVU, nDU); sumG = float((pr['nu'] * pr['G']).sum())
                ms = 'pos' if lo > 0 else ('neg' if hi < 0 else 'incl0')
                v9.append(dict(theta=theta, bfa=bfa, cold=cold, resource=resource, fm_pred=pred, fm_meas=m_p, fm_lo=lo_p, fm_hi=hi_p,
                               fm_cover=bool(lo_p <= pred <= hi_p), fm_norm_err=float(abs(m_p - pred) / scale) if scale > 0 else np.nan,
                               sumG=sumG, pred_sign='pos' if sumG >= 0 else 'neg', d_eta=d_eta, lo=lo, hi=hi, meas_sign=ms))
V9 = pd.DataFrame(v9)
RES['claims']['j9_variant'] = dict(result='eq:finite-memory identity and stationary sign criterion with J = 9 dimension cells (theta x bfa in {0,0.2} x cold x resource)',
                                   kind='IDENTITY + sampling check; IN-SAMPLE sign test', n=int(len(V9)), frac_fm_cover=float(V9.fm_cover.mean()),
                                   median_fm_norm_err=float(V9.fm_norm_err.median()), confusion=confusion(V9), rows=V9.round(6).to_dict('records'),
                                   verdict='PASS' if (V9.fm_cover.mean() >= 0.9 and confusion(V9)['accuracy'] >= 0.95) else 'FAIL')
log('(J9)', json.dumps({k: v for k, v in RES['claims']['j9_variant'].items() if k != 'rows'}, default=str))

# ============================================================================= Block 2: app:memory-overhead (shadow logger)
log('Block 2: memory-maintenance overhead (shadow logger)')
ov_rows = []
for theta in THETAS:
    for bfa in [0.0, 0.1]:
        W = Writes(TID, JC, theta, bfa, 0.0, 0.0, EXPLORE, MEXP, U_VER, U_LIFE, mode='shadow')
        state, recm = W.arriving_state('geometric')
        for cold in COLD:
            pr = primitives(TEST, theta, bfa, 0.0, 0.0, cold_cell[cold], nu_test, 'dollars', mode='shadow')
            ev = evaluate(W, state, recm, COLD[cold], 'dollars', mode='shadow')
            # no-memory reference: pure cold policy, no shadow call
            c0 = float((nu_test * pr['Ccold']).sum()); v0 = float((nu_test * pr['Vcold']).sum()); eta0 = v0 / c0
            h = float((nu_test * pr['shadow_cost']).sum())
            gI0 = (pr['vI'] - pr['vU']) - eta0 * (pr['cI'] - pr['cU']); gW0 = (pr['vW'] - pr['vU']) - eta0 * (pr['cW'] - pr['cU'])
            num = np.where(pr['omega'] > 0, pr['omega'] * np.nan_to_num(gI0), 0.0) + np.where(pr['psi'] > 0, pr['psi'] * np.nan_to_num(gW0), 0.0)
            with np.errstate(invalid='ignore', divide='ignore'):
                G0 = np.where(pr['r'] > 0, nu_test * num / (1 - pr['alpha']), 0.0)
            JU0 = float((nu_test * (pr['vU'] - eta0 * pr['cU'])).sum())
            sumG0 = float((nu_test * G0).sum())
            # identity eq:app-arbitrary-memory-surplus at N=20000 (lambda = eta0)
            a = pr['alpha']
            with np.errstate(invalid='ignore', divide='ignore'):
                br = np.where(a < 1, N_EPOCHS - (1 - a ** N_EPOCHS) / (1 - a), 0.0)
            pred_id = N_EPOCHS * JU0 + float((nu_test * G0 * br).sum())
            surp0 = ev['V'] - eta0 * ev['D']
            m_id, lo_id, hi_id = boot_sum(task_sums(surp0 / SEEDS, TID))
            # threshold eq:app-memory-maintenance-threshold vs measured eta_late(memory) - eta0 (paired on same tasks)
            nV = task_sums(ev['V'][:, late], TID[:, late]); nD = task_sums(ev['D'][:, late], TID[:, late])
            coldV = S[TID[:, late], COLD[cold][TID[:, late]]]; coldD = C[TID[:, late], COLD[cold][TID[:, late]]]
            nV0 = task_sums(coldV, TID[:, late]); nD0 = task_sums(coldD, TID[:, late])
            d_eta, lo, hi = boot_ratio_diff(nV, nD, nV0, nD0)
            ms = 'pos' if lo > 0 else ('neg' if hi < 0 else 'incl0')
            ov_rows.append(dict(theta=theta, bfa=bfa, cold=cold, eta0=eta0, c0=c0, h=h, JU0=JU0, check_JU0_eq_minus_eta0h=float(JU0 + eta0 * h),
                                sumG0=sumG0, threshold_rhs=eta0 * h, pred_memory_beats_ref=bool(sumG0 >= eta0 * h),
                                pred_sign='pos' if sumG0 >= eta0 * h else 'neg', meas_d_eta=d_eta, lo=lo, hi=hi, meas_sign=ms,
                                identity_pred=pred_id, identity_meas=m_id, identity_lo=lo_id, identity_hi=hi_id,
                                identity_cover=bool(lo_id <= pred_id <= hi_id), identity_rel_err=float(abs(m_id - pred_id) / abs(pred_id)) if pred_id != 0 else np.nan,
                                cost_share_shadow=float(h / (c0 + h))))
OV = pd.DataFrame(ov_rows)
conf_ov = confusion(OV)
RES['claims']['overhead'] = dict(
    result='app:memory-overhead eq:app-arbitrary-memory-surplus (lambda = eta_0) and eq:app-memory-maintenance-threshold',
    kind='IDENTITY + sampling check (identity) and IN-SAMPLE hypothesis test (threshold sign vs measured eta_inf - eta_0)',
    instantiation='memory-enabled cold state = cold policy + one shadow call (uniform model, recorded token-price cost estimate); no-memory reference = cold policy alone (same value, cost c_0); h = mean shadow cost',
    n_configs=int(len(OV)), frac_identity_ci_covers=float(OV.identity_cover.mean()), median_identity_rel_err=float(OV.identity_rel_err.median()),
    max_abs_JU0_plus_eta0h=float(OV.check_JU0_eq_minus_eta0h.abs().max()), confusion=conf_ov,
    rows=OV.round(6).to_dict('records'),
    criterion='identity PASS if CI covers prediction in all configs; threshold PASS if sign accuracy = 1 among conclusive configs',
    verdict_identity='PASS' if OV.identity_cover.all() else 'FAIL',
    verdict_threshold='PASS' if (conf_ov['n_conclusive'] > 0 and conf_ov['accuracy'] == 1.0) else ('INCONCLUSIVE' if conf_ov['n_conclusive'] == 0 else 'FAIL'))
log('(overhead)', json.dumps({k: v for k, v in RES['claims']['overhead'].items() if k != 'rows'}, default=str))

# ============================================================================= Block 3: thm:tail (Zipf, clean writes, common gap)
log('Block 3: heavy tails')
def zipf_infinite_sum(z, omega, E, Jmax=2 * 10 ** 6):
    j = np.arange(1, Jmax + 1, dtype=float); kz = 1 / zeta(z); nu = kz * j ** (-z)
    s = (nu * (1 - omega * nu) ** E).sum()
    tail = kz * (Jmax ** (1 - z)) / (z - 1)                # sum_{j>Jmax} nu_j <= this (all uncovered at worst)
    return float(s), float(tail)

def zipf_cum_infinite(z, omega, N, Jmax=2 * 10 ** 6):
    j = np.arange(1, Jmax + 1, dtype=float); kz = 1 / zeta(z); nu = kz * j ** (-z)
    return float(((1 - (1 - omega * nu) ** N) / omega).sum())     # cumulative deficit, Delta=1

theta_z = 1.0
Hz = HIT[theta_z]
cold_task = COLD['dimbest']
coldhit = Hz[np.arange(T), cold_task]
elig = TEST[(~coldhit[TEST]) & Hz[TEST].any(1)]                     # id_test tasks where the cold model misses and some model hits
# exact-reveal logger records the CHEAPEST hitting model (real cost); Delta = 1 - 0 = 1 on every eligible task
cheap_hit = np.array([np.where(Hz[t])[0][np.argmin(C[t, Hz[t]])] for t in elig])
RES['config']['zipf'] = dict(theta=theta_z, cold='dimbest', n_eligible_tasks=int(len(elig)), J_cells=ZIPF_J, omega=ZIPF_OMEGA,
                             cell_task_assignment='uniform with replacement from eligible tasks (a task may serve several cells)',
                             logger='on arrival, with prob omega the logger evaluates all models on the real task and records the cheapest hitting model (exact reveal, clean)')
rng_z = np.random.default_rng(4242)
zipf_rows = []; zipf_curves = {}
LATE_Z = 10000
for z in ZIPF_Z:
    jj = np.arange(1, ZIPF_J + 1, dtype=float); nuJ = jj ** (-z); nuJ /= nuJ.sum()
    cell_task = elig[rng_z.integers(0, len(elig), ZIPF_J)]; cell_rec = cheap_hit[np.searchsorted(elig, cell_task)]
    cells = rng_z.choice(ZIPF_J, size=(SEEDS, N_EPOCHS), p=nuJ)
    write = rng_z.random((SEEDS, N_EPOCHS)) < ZIPF_OMEGA
    lw = last_write_before(cells, write, ZIPF_J)
    has = lw >= 0; n_w = lw % N_EPOCHS; n_idx = np.broadcast_to(np.arange(N_EPOCHS), (SEEDS, N_EPOCHS))
    t_arr = cell_task[cells]; rec_arr = cell_rec[cells]; cold_arr = cold_task[t_arr]
    hit_inf = Hz[t_arr, rec_arr]; hit_cold = Hz[t_arr, cold_arr]
    assert hit_inf.all() and not hit_cold.any()
    # TTL deficit (stationary window): informed-everywhere hit rate minus TTL-E policy hit rate (real outcomes)
    curve_meas, curve_lo, curve_hi, curve_exact, curve_inf = [], [], [], [], []
    for E in ZIPF_E:
        cov = has & (n_idx - n_w <= E)
        hit_E = np.where(cov, hit_inf, hit_cold)
        per_seed = (hit_inf[:, -LATE_Z:].astype(float) - hit_E[:, -LATE_Z:]).mean(1)
        m, lo, hi = seed_ci(per_seed, rng_z)
        ex = float((nuJ * (1 - ZIPF_OMEGA * nuJ) ** E).sum())
        inf_s, tail = zipf_infinite_sum(z, ZIPF_OMEGA, E)
        curve_meas.append(m); curve_lo.append(lo); curve_hi.append(hi); curve_exact.append(ex); curve_inf.append(inf_s)
        zipf_rows.append(dict(z=z, kind='ttl', x=E, meas=m, lo=lo, hi=hi, exact_finiteJ=ex, infinite_sum=inf_s, infinite_tail_bound=tail,
                              asym=float((1 / zeta(z)) ** (1 / z) / z * Gamma(1 - 1 / z) * (ZIPF_OMEGA * E) ** (1 / z - 1)),
                              cover_exact=bool(lo <= ex <= hi)))
    # cumulative deficit (no expiration, empty start)
    cov_inf = has
    hit_none = np.where(cov_inf, hit_inf, hit_cold)
    per_epoch_def = (hit_inf.astype(float) - hit_none)          # (Sd, N)
    cum = np.cumsum(per_epoch_def, 1)
    cmeas, clo, chi_, cexact, cinf = [], [], [], [], []
    for Nn in ZIPF_N:
        m, lo, hi = seed_ci(cum[:, Nn - 1], rng_z)
        ex = float(((1 - (1 - ZIPF_OMEGA * nuJ) ** Nn) / ZIPF_OMEGA).sum())
        inf_s = zipf_cum_infinite(z, ZIPF_OMEGA, Nn)
        cmeas.append(m); clo.append(lo); chi_.append(hi); cexact.append(ex); cinf.append(inf_s)
        zipf_rows.append(dict(z=z, kind='cum', x=Nn, meas=m, lo=lo, hi=hi, exact_finiteJ=ex, infinite_sum=inf_s, infinite_tail_bound=np.nan,
                              asym=float((1 / zeta(z)) ** (1 / z) * Gamma(1 - 1 / z) * ZIPF_OMEGA ** (1 / z - 1) * Nn ** (1 / z)),
                              cover_exact=bool(lo <= ex <= hi)))
    top = np.array(ZIPF_E) >= 200; topN = np.array(ZIPF_N) >= 2000
    zipf_curves[z] = dict(E=ZIPF_E, ttl_meas=curve_meas, ttl_lo=curve_lo, ttl_hi=curve_hi, ttl_exact=curve_exact, ttl_inf=curve_inf,
                          N=ZIPF_N, cum_meas=cmeas, cum_lo=clo, cum_hi=chi_, cum_exact=cexact, cum_inf=cinf,
                          slope_ttl_meas_top=loglog_slope(np.array(ZIPF_E)[top], np.array(curve_meas)[top]),
                          slope_ttl_exact_top=loglog_slope(np.array(ZIPF_E)[top], np.array(curve_exact)[top]),
                          slope_ttl_inf_top=loglog_slope(np.array(ZIPF_E)[top], np.array(curve_inf)[top]),
                          slope_ttl_meas_all=loglog_slope(ZIPF_E, curve_meas), slope_ttl_target=-(z - 1) / z,
                          slope_cum_meas_top=loglog_slope(np.array(ZIPF_N)[topN], np.array(cmeas)[topN]),
                          slope_cum_exact_top=loglog_slope(np.array(ZIPF_N)[topN], np.array(cexact)[topN]),
                          slope_cum_inf_top=loglog_slope(np.array(ZIPF_N)[topN], np.array(cinf)[topN]),
                          slope_cum_meas_all=loglog_slope(ZIPF_N, cmeas), slope_cum_target=1 / z)
    log(f'  z={z}: ttl slope meas {zipf_curves[z]["slope_ttl_meas_top"]:.3f} exact {zipf_curves[z]["slope_ttl_exact_top"]:.3f} '
        f'inf {zipf_curves[z]["slope_ttl_inf_top"]:.3f} target {-(z-1)/z:.3f} | cum slope meas {zipf_curves[z]["slope_cum_meas_top"]:.3f} '
        f'exact {zipf_curves[z]["slope_cum_exact_top"]:.3f} inf {zipf_curves[z]["slope_cum_inf_top"]:.3f} target {1/z:.3f}')

# hypothesis-VIOLATED control: all 2,919 id_test tasks as singleton cells with random Zipf ranks, uniform-model logger
ctrl = {}
for z in ZIPF_Z:
    Jc = len(TEST); jj = np.arange(1, Jc + 1, dtype=float); nuJ = jj ** (-z); nuJ /= nuJ.sum()
    perm = rng_z.permutation(TEST)                                # rank -> real task
    cells = rng_z.choice(Jc, size=(SEEDS, N_EPOCHS), p=nuJ)
    t_arr = perm[cells]
    mlog = rng_z.integers(0, M, size=(SEEDS, N_EPOCHS))
    write = Hz[t_arr, mlog]                                       # uniform logger, exact verifier: omega_j = k_j / 8
    lw = last_write_before(cells, write, Jc)
    has = lw >= 0; n_w = lw % N_EPOCHS; n_idx = np.broadcast_to(np.arange(N_EPOCHS), (SEEDS, N_EPOCHS))
    rec_arr = gather(mlog, lw)
    cold_arr = cold_task[t_arr]
    anyhit = Hz[t_arr].any(-1); delta_j = anyhit.astype(float) - Hz[t_arr, cold_arr]     # informed(any record hits)-cold gap
    meas = []
    for E in ZIPF_E:
        unc = ~(has & (n_idx - n_w <= E))
        per_seed = (delta_j * unc)[:, -LATE_Z:].mean(1)
        meas.append(float(per_seed.mean()))
    top = np.array(ZIPF_E) >= 200
    ctrl[z] = dict(deficit=meas, slope_top=loglog_slope(np.array(ZIPF_E)[top], np.array(meas)[top]), target=-(z - 1) / z,
                   frac_cells_zero_gap=float((delta_j[:, -LATE_Z:] == 0).mean()))
ZP = pd.DataFrame(zipf_rows)
g_rows = []
for z in ZIPF_Z:
    c = zipf_curves[z]
    g_rows.append(dict(z=z, ttl_slope_meas=c['slope_ttl_meas_top'], ttl_slope_exact_finiteJ=c['slope_ttl_exact_top'], ttl_slope_infinite=c['slope_ttl_inf_top'],
                       ttl_target=c['slope_ttl_target'], ttl_pass=bool(abs(c['slope_ttl_meas_top'] - c['slope_ttl_target']) <= SLOPE_TOL),
                       cum_slope_meas=c['slope_cum_meas_top'], cum_slope_exact_finiteJ=c['slope_cum_exact_top'], cum_slope_infinite=c['slope_cum_inf_top'],
                       cum_target=c['slope_cum_target'], cum_pass=bool(abs(c['slope_cum_meas_top'] - c['slope_cum_target']) <= SLOPE_TOL),
                       mech_cover_ttl=float(ZP[(ZP.z == z) & (ZP.kind == 'ttl')].cover_exact.mean()), mech_cover_cum=float(ZP[(ZP.z == z) & (ZP.kind == 'cum')].cover_exact.mean()),
                       control_violated_slope=ctrl[z]['slope_top'], control_frac_zero_gap=ctrl[z]['frac_cells_zero_gap']))
GT = pd.DataFrame(g_rows)
RES['claims']['g_tail'] = dict(
    result='thm:tail: TTL deficit Theta(E^{-(z-1)/z}); cumulative deficit Theta(N^{1/z}); Zipf, clean writes, common omega, common Delta',
    kind='mechanism = IDENTITY + sampling check (exact finite-J occupancy sums); exponent = asymptotic prediction tested at finite E, N (semi-synthetic Zipf over real tasks)',
    rows=GT.round(4).to_dict('records'), slope_tol=SLOPE_TOL, slope_range='top decade: E in [200,2000], N in [2000,20000]',
    criterion='mechanism PASS if seed-CI covers the exact finite-J sum in >= 0.90 of points; exponent PASS if |fitted slope - target| <= 0.10 for all z (both TTL and cumulative); if exponent fails but the theorem\'s own infinite sum has the same finite-range slope (within 0.03 of measured) -> INCONCLUSIVE (finite range), not FAIL',
    verdict_mechanism='PASS' if (GT.mech_cover_ttl.min() >= 0.9 and GT.mech_cover_cum.min() >= 0.9) else 'FAIL')
allpass = bool(GT.ttl_pass.all() and GT.cum_pass.all())
same_as_inf = bool(((GT.ttl_slope_meas - GT.ttl_slope_infinite).abs() <= 0.03).all() and ((GT.cum_slope_meas - GT.cum_slope_infinite).abs() <= 0.03).all())
RES['claims']['g_tail']['verdict_exponent'] = 'PASS' if allpass else ('INCONCLUSIVE (finite-range: measured slope matches the theorem\'s infinite sum over the same range)' if same_as_inf else 'FAIL')
RES['claims']['g_tail']['control_hypothesis_violated'] = {str(z): ctrl[z] for z in ZIPF_Z}
log('(g)', GT.round(3).to_string())

# ============================================================================= Block 4: lem:app-arm-reveal
log('Block 4: exact-reveal arm coverage')
rng_h = np.random.default_rng(31337)
mu = np.zeros((J21, M))
for jj in range(J21):
    idx = TEST[cell21[TEST] == jj]; mu[jj] = S[idx].mean(0)
mstar = mu.argmax(1)
VIS = 100; REPS = 4000
arm_rows = []; arm_curves = {}
for e_star, e_other in [(0.02, 0.10), (0.05, 0.10), (0.10, 0.10), (0.20, 0.10), (0.05, 0.30)]:
    viol = 0; total = 0; curves_m = []; curves_b = []
    for jj in range(J21):
        e = np.full(M, e_other); e[mstar[jj]] = e_star
        # first visit at which each arm is recorded: Geometric(e_m) (inf if never within VIS)
        u = rng_h.random((REPS, M, VIS))
        logged = u < e[None, :, None]
        first = np.where(logged.any(2), logged.argmax(2) + 1, VIS + 1)          # (REPS, M) 1-based visit index
        subopt = np.zeros(VIS)
        for v in range(1, VIS + 1):
            recorded = first <= v
            sc = np.where(recorded, mu[jj][None, :], -np.inf)
            none = ~recorded.any(1)
            rec = np.where(none, cold_cell['dimbest'][jj], sc.argmax(1))
            subopt[v - 1] = (mu[jj][rec] < mu[jj][mstar[jj]] - 1e-12).mean()
        bound = (1 - e_star) ** np.arange(1, VIS + 1)
        se = np.sqrt(np.maximum(subopt * (1 - subopt), 1e-12) / REPS)
        viol += int((subopt > bound + 2.5 * se).sum()); total += VIS
        curves_m.append(subopt); curves_b.append(bound)
    curves_m = np.array(curves_m); curves_b = np.array(curves_b)
    arm_rows.append(dict(e_star=e_star, e_other=e_other, n_points=total, n_violations=viol, frac_violations=viol / total,
                         mean_ratio_meas_over_bound_v10=float(np.mean(curves_m[:, 9] / curves_b[:, 9])),
                         mean_ratio_meas_over_bound_v50=float(np.mean(curves_m[:, 49] / curves_b[:, 49])),
                         mean_meas_v50=float(curves_m[:, 49].mean()), bound_v50=float(curves_b[0, 49])))
    arm_curves[(e_star, e_other)] = (curves_m.mean(0), curves_b[0])
# best arm outside the logger's support: recommendation converges to the best within the support
conv = []
for jj in range(J21):
    e = np.full(M, 0.10); e[mstar[jj]] = 0.0
    sup = np.where(e > 0)[0]; best_sup = sup[mu[jj][sup].argmax()]
    u = rng_h.random((REPS, M, 200)); logged = u < e[None, :, None]
    first = np.where(logged.any(2), logged.argmax(2) + 1, 10 ** 6)
    recorded = first <= 200
    sc = np.where(recorded, mu[jj][None, :], -np.inf); rec = sc.argmax(1)
    conv.append(float((mu[jj][rec] == mu[jj][best_sup]).mean()))
AR = pd.DataFrame(arm_rows)
RES['claims']['h_arm_reveal'] = dict(
    result='lem:app-arm-reveal: P(suboptimal recommendation after v visits) <= (1-e_{m*})^v; convergence to best-in-support if e_{m*}=0',
    kind='probability bound (IDENTITY-type) + sampling check; arms = 8 models, mu_m = real id_test mean score per 21-cell, exact reveal',
    rows=AR.round(5).to_dict('records'), frac_converged_to_best_in_support_v200=float(np.mean(conv)), min_over_cells=float(np.min(conv)),
    criterion='PASS if the bound is violated (measured > bound + 2.5 SE) in < 1% of (cell, v) points for every e-configuration and convergence fraction >= 0.99',
    verdict='PASS' if (AR.frac_violations.max() < 0.01 and np.mean(conv) >= 0.99) else 'FAIL')
log('(h)', json.dumps({k: v for k, v in RES['claims']['h_arm_reveal'].items() if k != 'rows'}, default=str), AR.round(4).to_string())

# ============================================================================= tables
RES['tables']['record_probs'] = RP.round(6).to_dict('records')
RES['tables']['finite_memory'] = FM.round(8).to_dict('records')
RES['tables']['stationary_sign'] = SG.round(8).to_dict('records')
RES['tables']['noise_threshold_cells'] = CL.round(8).to_dict('records')
RES['tables']['retention'] = RT.round(6).to_dict('records')
RES['tables']['regret'] = RG.round(6).to_dict('records')
RES['tables']['zipf'] = ZP.round(8).to_dict('records')
RES['tables']['zipf_curves'] = {str(z): {k: (list(map(float, v)) if isinstance(v, (list, np.ndarray)) else float(v)) for k, v in c.items()} for z, c in zipf_curves.items()}
prim_dump = {}
for (theta, bfa, bfr, delta, cold, resource, est), pr in PRIMS.items():
    if delta == 0.0 and bfr == 0.0 and bfa in (0.0, 0.1, 0.2) and est == 'in':
        prim_dump[f'th{theta}_fa{bfa}_{cold}_{resource}'] = {
            'eta_U': pr['eta_U'], 'cells': [dict(cell=CELLS21[j], nu=float(pr['nu'][j]), omega=float(pr['omega'][j]), psi=float(pr['psi'][j]),
                                                 vU=float(pr['vU'][j]), cU=float(pr['cU'][j]), vI=float(pr['vI'][j]), cI=float(pr['cI'][j]),
                                                 vW=float(pr['vW'][j]), cW=float(pr['cW'][j]), gI=float(pr['gI'][j]), gW=float(pr['gW'][j]),
                                                 G=float(pr['G'][j]), alpha=float(pr['alpha'][j])) for j in range(J21)]}
RES['tables']['primitives_in_sample'] = prim_dump

# ============================================================================= figures
log('figures')
def savefig(fig, name):
    fig.savefig(f'{FIG}/{name}.pdf', metadata={'Creator': 'X4_memory_tail/run.py'}); plt.close(fig)

# Fig 1: record probabilities (a)
fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.0), layout='constrained')
for ax, cname in zip(axes, ['algorithm|lcb', 'code_generation|bcb', 'test_generation|he']):
    for (bfa, sname, col, ls) in [(0.0, 'I', BLUE, '-'), (0.2, 'I', GREEN, '-'), (0.2, 'W', RED, '-')]:
        k = (bfa, cname, sname)
        if k not in fig1_store:
            continue
        meas, pbin, se, BIN = fig1_store[k]
        x = (np.arange(len(meas)) + .5) * BIN
        ax.fill_between(x, meas - 2 * se, meas + 2 * se, color=col, alpha=.18, lw=0)
        ax.plot(x, meas, color=col, lw=1.0, label=rf'${sname}$, $\beta_{{\rm fa}}={bfa}$ (measured)')
        ax.plot(x, pbin, color='k', ls='--', lw=.8)
    ax.set(xlabel='Epoch $n$', title=cname.replace('_', r'\_'), xlim=(0, N_EPOCHS))
axes[0].set_ylabel(r'$i_{j,n},\,w_{j,n}$')
axes[0].plot([], [], color='k', ls='--', lw=.8, label='Eq.~(record-probs)')
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc='outside lower center', ncol=4, frameon=False, fontsize=7)
savefig(fig, 'fig1_record_probs')

# Fig 2: finite-memory identity and stationary sign (b, c)
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.3), layout='constrained')
ax = axes[0]
sub = FM[(FM.est == 'in') & (FM.N == N_EPOCHS)]
lim = max(sub.pred.abs().max(), sub.meas_paired.abs().max()) * 1.1
ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=.65, ls=':')
for res_, mk in [('dollars', 'o'), ('seconds', 's')]:
    s2 = sub[sub.resource == res_]
    ax.errorbar(s2.pred, s2.meas_paired, yerr=[s2.meas_paired - s2.lo_p, s2.hi_p - s2.meas_paired], fmt=mk, ms=3, color=BLUE if res_ == 'dollars' else RED,
                lw=.6, capsize=0, label=f'{res_}')
ax.set(xlabel=r'Predicted $\sum_j\nu_jG_j[N-(1-\alpha_j^N)/(1-\alpha_j)]$', ylabel=r'Measured $\sum_{n<N}\mathrm{E}[V_n-\eta_UD_n]$', title=f'Eq.~(finite-memory), $N={N_EPOCHS}$')
ax.legend(frameon=False, loc='upper left')
ax = axes[1]
sub = SG[SG.est == 'in']
for res_, col in [('dollars', BLUE), ('seconds', RED)]:
    s2 = sub[sub.resource == res_]
    ax.errorbar(s2.pred_deta, s2.d_eta, yerr=[s2.d_eta - s2.lo, s2.hi - s2.d_eta], fmt='o', ms=3, color=col, lw=.6, capsize=0, label=res_)
lim = max(sub.pred_deta.abs().max(), sub.d_eta.abs().max()) * 1.1
ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=.65, ls=':'); ax.axhline(0, color=GREY, lw=.5); ax.axvline(0, color=GREY, lw=.5)
ax.set(xlabel=r'Predicted $\eta_\infty-\eta_U=\sum_j\nu_jG_j/\bar c_\infty$', ylabel=r'Measured $\eta_{\rm late}^{I}-\eta_{\rm late}^{U}$', title='Stationary sign criterion')
ax.set_xscale('symlog', linthresh=1e-2); ax.set_yscale('symlog', linthresh=1e-2)
ax.legend(frameon=False, loc='upper left')
savefig(fig, 'fig2_finite_memory_sign')

# Fig 3: noise threshold (d)
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.4), layout='constrained')
for ax, est_, ttl in [(axes[0], 'in', 'in-sample primitives'), (axes[1], 'oos', 'probing (out-of-sample) primitives')]:
    sub = cl_h[(cl_h.est == est_)]
    for ms_, col, lab in [('pos', GREEN, 'measured $>0$'), ('neg', RED, 'measured $<0$'), ('incl0', GREY, 'CI includes 0')]:
        s2 = sub[sub.meas_sign == ms_]
        ax.scatter(s2.thr, s2.err, s=10, color=col, label=lab, alpha=.8, edgecolors='none')
    ax.plot([0, 1], [0, 1], color='k', lw=.8, ls='--')
    ax.set(xlabel=r'Threshold $g_j^I/(g_j^I-g_j^W)$', xlim=(0, 1.02), ylim=(-0.02, 1.02), title=ttl)
axes[0].set_ylabel(r'Erroneous fraction $\psi_j/r_j$')
axes[0].text(.03, .93, r'above the line: $G_j<0$ predicted', transform=axes[0].transAxes, fontsize=7)
axes[1].legend(frameon=False, loc='upper right', fontsize=7)
savefig(fig, 'fig3_noise_threshold')

# Fig 4: TTL / FIFO coverage (e)
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.3), layout='constrained')
ax = axes[0]
sub = RT[RT.theta == 1.0]
ax.plot([0, 1], [0, 1], color=GREY, lw=.65, ls=':')
ax.scatter(sub.ttl_pred, sub.ttl_meas, s=8, color=BLUE, label='TTL, Eq.~(app-ttl)', edgecolors='none')
ax.scatter(sub.fifo_pred, sub.fifo_meas, s=8, color=GREEN, marker='s', label='FIFO, Eq.~(app-fifo)', edgecolors='none')
ax.set(xlabel='Predicted coverage', ylabel='Measured coverage (late window)', title=r'$\theta=1$, $\beta_{\rm fa}\in\{0,0.2\}$, $E,M\in\{5,\dots,1000\}$')
ax.legend(frameon=False, loc='lower right')
ax = axes[1]
for E, col in zip([5, 10, 20], [RED, BLUE, GREEN]):
    s2 = RT[(RT.E == E)]
    ax.scatter(s2.q, s2.naive_pred - s2.fifo_meas, s=8, color=col, edgecolors='none', label=rf'$M={E}$: $E\to M/\beta$')
    ax.scatter(s2.q, s2.fifo_pred - s2.fifo_meas, s=8, color=col, marker='x', lw=.6, label=rf'$M={E}$: exact')
ax.axhline(0, color=GREY, lw=.5)
ax.set(xlabel=r'$q_j=\nu_jr_j/\beta$', ylabel='Prediction $-$ measured FIFO coverage', xscale='log', title=r'Inexactness of $E\to M/\beta$ (300-seed replay)')
h_, l_ = ax.get_legend_handles_labels()
fig.legend(h_, l_, loc='outside lower center', ncol=6, frameon=False, fontsize=6.5)
savefig(fig, 'fig4_retention')

# Fig 5: regret (f)
fig, ax = plt.subplots(figsize=(3.3, 2.4), layout='constrained')
for (cold, col) in [('dimbest', BLUE), ('opus', RED)]:
    for bfa, ls in [(0.0, '-'), (0.2, '--')]:
        s2 = RG[(RG.est == 'in') & (RG.theta == 1.0) & (RG.cold == cold) & (RG.bfa == bfa) & (RG.bfr == 0) & (RG.resource == 'dollars')].sort_values('N')
        ax.plot(s2.N, s2.pred, color=col, ls=ls, lw=1.0)
        ax.errorbar(s2.N, s2.meas, yerr=[s2.meas - s2.lo, s2.hi - s2.meas], fmt='o', ms=2.5, color=col, lw=.6, capsize=0)
ax.plot([], [], color=BLUE, label='cold = DimensionBest'); ax.plot([], [], color=RED, label='cold = Opus')
ax.plot([], [], color='k', ls='-', label=r'$\beta_{\rm fa}=0$'); ax.plot([], [], color='k', ls='--', label=r'$\beta_{\rm fa}=0.2$')
ax.set(xscale='log', xlabel='Horizon $N$', ylabel=r'$\mathrm{E}[\mathrm{Reg}_N]$ (lines: Eq.~(app-regret))', title=r'$\theta=1$, dollars')
ax.legend(frameon=False, fontsize=7, loc='upper left')
savefig(fig, 'fig5_regret')

# Fig 6: heavy tails (g)
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.3), layout='constrained')
for z, col in zip(ZIPF_Z, [BLUE, GREEN, RED]):
    c = zipf_curves[z]
    ax = axes[0]
    ax.errorbar(c['E'], c['ttl_meas'], yerr=[np.array(c['ttl_meas']) - np.array(c['ttl_lo']), np.array(c['ttl_hi']) - np.array(c['ttl_meas'])], fmt='o', ms=2.5, color=col, lw=.6, capsize=0)
    ax.plot(c['E'], c['ttl_exact'], color=col, lw=1.0, label=rf'$z={z}$: exact sum ($J={ZIPF_J}$)')
    E_ = np.array(c['E'], float); asym = (1 / zeta(z)) ** (1 / z) / z * Gamma(1 - 1 / z) * (ZIPF_OMEGA * E_) ** (1 / z - 1)
    ax.plot(E_, asym, color=col, ls=':', lw=.8)
    ax = axes[1]
    ax.errorbar(c['N'], c['cum_meas'], yerr=[np.array(c['cum_meas']) - np.array(c['cum_lo']), np.array(c['cum_hi']) - np.array(c['cum_meas'])], fmt='o', ms=2.5, color=col, lw=.6, capsize=0)
    ax.plot(c['N'], c['cum_exact'], color=col, lw=1.0)
    N_ = np.array(c['N'], float); asym = (1 / zeta(z)) ** (1 / z) * Gamma(1 - 1 / z) * ZIPF_OMEGA ** (1 / z - 1) * N_ ** (1 / z)
    ax.plot(N_, asym, color=col, ls=':', lw=.8)
axes[0].set(xscale='log', yscale='log', xlabel='TTL $E$', ylabel=r'Uncovered hit deficit $L_E$', title=r'Dotted: $\Theta(E^{-(z-1)/z})$ leading term')
axes[1].set(xscale='log', yscale='log', xlabel='Horizon $N$', ylabel=r'Cumulative deficit $R_N$', title=r'Dotted: $\Theta(N^{1/z})$ leading term')
axes[0].legend(frameon=False, fontsize=7, loc='lower left')
savefig(fig, 'fig6_heavy_tail')

# Fig 7: arm reveal (h)
fig, ax = plt.subplots(figsize=(3.3, 2.4), layout='constrained')
for (k, col) in zip([(0.02, 0.10), (0.05, 0.10), (0.10, 0.10), (0.20, 0.10)], [RED, BLUE, GREEN, GREY]):
    m_, b_ = arm_curves[k]
    v = np.arange(1, VIS + 1)
    ax.plot(v, m_, color=col, lw=1.0, label=rf'$e_{{m^*}}={k[0]}$ measured')
    ax.plot(v, b_, color=col, lw=.8, ls='--')
ax.set(yscale='log', xlabel='Visits $v$', ylabel='P(suboptimal recommendation)', title=r'Dashed: $(1-e_{m^*})^v$', ylim=(1e-3, 1.1))
ax.legend(frameon=False, fontsize=7)
savefig(fig, 'fig7_arm_reveal')

RES['runtime_sec'] = time.time() - T0
def _default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)
with open(f'{OUT}/results.json', 'w') as f:
    json.dump(RES, f, indent=1, default=_default)
log('done; wrote results.json; runtime %.1f s' % RES['runtime_sec'])

