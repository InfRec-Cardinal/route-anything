#!/usr/bin/env python3
"""
X7a_regimes_offline -- "Route Anything: A Theory of Inference Routers", experiment X7 (offline part X7-a):
three router regimes on the same task stream under one accounting (Lemma thm:surplus, eq:cascade-surplus, eq:floor).

Regimes (paper Table tab:regimes), reference = always-Opus on CodeRouterBench (id_test, OOD176) and always-gpt-4-1106-preview
on RouterBench:
  R1 ex-ante dispatch      : 9 released router decision files + ACRouter's released ID policy (a per-task lookup map tuned
                             on train+val, i.e. an ex-ante router with a memory-derived map); RouterBench: our own
                             logistic-regression router trained on a 70% split, decisions on the 30% held out.
  R2 verify-then-escalate  : cheap primary -> verifier (oracle score >= theta; noisy beta_fa/beta_fr variants as exact
                             expectations over the verifier's flips) -> Opus / gpt-4 fallback; verification cost v_c in A.
  R3 in-loop agentic       : OOD176: ACRouter's released escalation traces (MiniMax -> Kimi -> GPT-5.4 -> GLM-5 -> Opus,
                             sandbox verifier) replayed on the OOD176 outcome matrix; id_test: NOT observable offline
                             (ACRouter ID mode dispatches once) -> counterfactual "ACRouter map + oracle verify-then-escalate"
                             reported as what the live escalation variant would reach with a perfect, free verifier (verifier-side
                             bound; NOT an efficiency bound), plus the exact per-decision optimal-escalation bound at price eta_B.
Accounting for every policy: V = H Y_P + (1-H) Y_F, D = A + (1-H) C_F (eq:accounting); Q = E[V - Y_B], S = E[C_B - D];
eta = sum V / sum D (ratio of sums); net gain Q + eta_B S with the four terms of eq:cascade-surplus
(avoided reference work E[H C_B], attempt cost E[A], accepted quality change E[H(Y_P - Y_B)], recovery cost
E[(1-H)(C_F - C_B)]; the miss quality change E[(1-H)(Y_F - Y_B)] is 0 with a reference-restoring fallback).
Statistics: 1,000 paired task-resampling bootstrap draws (seed 20260917), 95% percentile CIs, two policies always compared
on the same resampled tasks.  Offline only: no API calls.  Runtime ~1-2 min.
Data paths from env ACROUTER_DATA, ACROUTER_REPO, ROUTERBENCH_PKL (defaults below).  Everything is written into this directory.
"""
import os, json, time, glob, csv
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T0 = time.time()
SEED = 20260917
B = 1000
EPS = [0.0, 0.01, 0.02, 0.05]
OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, 'figures'); os.makedirs(FIG, exist_ok=True)
DATA = os.environ.get('ACROUTER_DATA', 'data/agent-as-a-router/data')
REPO = os.environ.get('ACROUTER_REPO', os.path.dirname(DATA.rstrip('/')))
RB_PKL = os.environ.get('ROUTERBENCH_PKL', './data/routerbench/routerbench_0shot.pkl')
ID_DECISIONS = f'{REPO}/outputs/current/id_decisions.jsonl'
OOD_TRACE = f'{REPO}/outputs/acrouter_ood176/ood_decisions.jsonl'
OOD_BASELINES = f'{REPO}/outputs/baselines_ood176/decisions'
OOD112_SANDBOX = f'{REPO}/outputs/current/ood_sandbox_decisions.jsonl'

BASE = 'claude-opus-4-6'
PRIMARIES = ['Qwen3-Max', 'kimi-k2.5', 'qwen3.5-plus', 'glm-5', 'gpt-5.4', 'claude-sonnet-4-6', 'MiniMax-M2.7']
FOCUS = ['Qwen3-Max', 'kimi-k2.5']
THETAS = [1.0, 0.5]
ROUTERS = ['dimension_best', 'logreg', 'tfidf_mlp', 'routellm_mf', 'routellm_sw',
           'finetuned_router_qwen35_08b', 'finetuned_router_qwen35_2b', 'finetuned_router_qwen35_9b_v3', 'finetuned_router_qwen35_27b_v3']
FINETUNED = [r for r in ROUTERS if r.startswith('finetuned')]
K_ROUTER = [0.0, 0.001, 0.005]                     # $ per task, router-cost sensitivity for the fine-tuned routers (no token field released)
ROUTER_TOKEN_PRICE = 0.054e-6                       # $ per router token (ACRouter's price for the Qwen router), used for the prompt-token proxy
NOISY = [(0.05, 0.05), (0.05, 0.1), (0.1, 0.05), (0.1, 0.1)]   # (beta_fa, beta_fr)
VC_ID = [0.0, 0.001, 0.01]                          # $ per verification, CodeRouterBench id_test
VC_RB = [0.0, 1e-4, 1e-3]                           # $ per verification, RouterBench (reference costs ~3.3e-3 $/task)
VC_OOD = [0.0, 0.01, 0.05]                          # $ per sandbox verification, OOD176 (reference costs ~1.1 $/task)
OOD_CHAIN = ['MiniMax-M2.7', 'kimi-k2.5', 'gpt-5.4', 'glm-5']
ALIAS = {'通义千问Max': 'Qwen3-Max'}
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------------------------------------ figure style
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times', 'Times New Roman', 'STIX Two Text', 'DejaVu Serif'], 'text.usetex': False,
    'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 7, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'axes.linewidth': .6, 'lines.linewidth': 1.2,
    'xtick.major.width': .6, 'ytick.major.width': .6, 'legend.frameon': False,
})
COL = {'R1': '#2a78d6', 'R2': '#eb6834', 'R3': '#1baf7a', 'R3-cf': '#1baf7a', 'R1-mem': '#4a3aa7', 'const': '#8a8a86', 'ref': '#0b0b0b', 'bound': '#8a8a86'}
TERM_COL = {'avoided': '#1baf7a', 'attempt': '#eb6834', 'quality': '#2a78d6', 'recovery': '#eda100'}
INK, INK2, HAIR = '#0b0b0b', '#52514e', '#c8c7c2'
MARK = {'R1': 'o', 'R2': '^', 'R3': 'D', 'R3-cf': 'D', 'R1-mem': 'p', 'const': 'o', 'ref': 's', 'bound': 'x'}


def log(*a):
    print(f'[{time.time()-T0:6.1f}s]', *a, flush=True)


# ================================================================================================ estimators
def boot_weights(n, rng):
    """(B+1) x n; row 0 = ones (point estimate), rows 1..B = multinomial bootstrap counts (tasks resampled with replacement)."""
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(float)
    return np.vstack([np.ones((1, n)), W])


def ci(v):
    v = np.asarray(v, float); w = v[1:][np.isfinite(v[1:])]
    if len(w) == 0:
        return [float(v[0]), float('nan'), float('nan')]
    return [float(v[0]), float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))]


def excl(t, x=0.0):
    return bool(np.isfinite(t[1]) and np.isfinite(t[2]) and (t[1] > x or t[2] < x))


def nanmax(v):
    v = np.asarray(v, float); w = v[np.isfinite(v)]
    return float(w.max()) if len(w) else float('nan')


def short(name):
    s = name.replace('R1 ', '').replace('R2 ', '').replace('R3-cf ', '').replace('R3 ', '').replace('finetuned_router_qwen35_', 'ft-qwen3.5-').replace('_v3', '')
    s = s.replace(' + oracle -> Opus', ' + oracle').replace(' theta=1.0', ' θ=1').replace(' theta=0.5', ' θ=0.5').replace('ACRouter chain trace (gate k=2)', 'ACRouter chain (k=2)')
    s = s.replace('LR router (tfidf, ', 'LR (').replace(' eta_B)', 'η_B)').replace('(reference)', '').replace('always-', '')
    s = s.replace('mistralai/', '').replace('zero-one-ai/', '').replace('meta/', '').replace('WizardLM/', '').replace('-> gpt-4-1106-preview', '')
    return s.strip()


def figlab(name):
    s = short(name)
    for a, b in [('ACRouter map + oracle', 'ACRouter map+oracle'), (' -> Opus', ''), ('ACRouter chain (k=2)', 'ACRouter chain'), ('gpt-4-1106-preview', 'gpt-4'),
                 ('claude-opus-4-6', 'Opus'), ('LR (lambda=4η_B)', 'LR λ=4η_B'), ('LR (lambda=1η_B) + oracle theta=1', 'LR λ=1η_B + oracle'), ('gpt-3.5-turbo-1106', 'gpt-3.5')]:
        s = s.replace(a, b)
    return s


def pol_stats(W, c):
    """One routing policy under eq:accounting.  c: per-task arrays H (accept indicator, or its expectation in [0,1] for a
    noisy verifier), YP (value of the accepted/committed cheap output), A (attempt cost incl. routing + verification),
    YF, CF (fallback value / additional cost if invoked), YB, CB (reference outcomes); optional LS, LB (seconds).
    Everything is a weighted sum under W (row 0 point estimate, rows 1..B bootstrap), so CIs are paired across policies."""
    H = np.asarray(c['H'], float); YP, A, YF, CF, YB, CB = (np.asarray(c[k], float) for k in ['YP', 'A', 'YF', 'CF', 'YB', 'CB'])
    V = H * YP + (1 - H) * YF                      # eq:accounting
    D = A + (1 - H) * CF
    cols = [V, D, YB, CB, H * CB, A, H * (YP - YB), (1 - H) * (YF - YB), (1 - H) * (CF - CB), H * (YP < YB - 1e-12), H]
    has_sec = c.get('LS') is not None
    if has_sec:
        cols += [np.asarray(c['LS'], float), np.asarray(c['LB'], float)]
    M = W @ np.column_stack(cols); n = W.sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        EV, ED, vbar, cbar = M[:, 0] / n, M[:, 1] / n, M[:, 2] / n, M[:, 3] / n
        eta = M[:, 0] / M[:, 1]; eta_B = M[:, 2] / M[:, 3]
        avoided, attempt, Qhit, Qmiss, recov, e, p = (M[:, k] / n for k in range(4, 11))
        Q = EV - vbar; S = cbar - ED; net = Q + eta_B * S                     # eq:surplus-def, eq:unified-identity
        ident = (eta - eta_B) - net / (cbar - S)                                # eq:surplus residual
        out = dict(EV=EV, ED=ED, vbar=vbar, cbar=cbar, eta=eta, eta_B=eta_B, r=eta / eta_B, d_eta=eta - eta_B, Q=Q, S=S, net=net,
                   net_usd=net / eta_B, avoided=avoided, attempt=attempt, Qhit=Qhit, Qmiss=Qmiss, recov=recov,
                   Qhit_usd=Qhit / eta_B, Qmiss_usd=Qmiss / eta_B, e=e, p=p,
                   ident_rel=np.abs(ident) / np.abs(eta), decompQ=np.abs(Q - Qhit - Qmiss), decompS=np.abs(S - (avoided - attempt - recov)))
        if has_sec:
            ELS, ELB = M[:, 11] / n, M[:, 12] / n
            eta_s = M[:, 0] / M[:, 11]; etaB_s = M[:, 2] / M[:, 12]; S_s = ELB - ELS
            ident_s = (eta_s - etaB_s) - (Q + etaB_s * S_s) / (ELB - S_s)
            out.update(ELS=ELS, ELB=ELB, eta_sec=eta_s, eta_B_sec=etaB_s, r_sec=eta_s / etaB_s, S_sec=S_s, ident_sec_rel=np.abs(ident_s) / np.abs(eta_s))
    return out


KEEP = ['EV', 'ED', 'eta', 'eta_B', 'r', 'd_eta', 'Q', 'S', 'net', 'net_usd', 'avoided', 'attempt', 'Qhit', 'Qmiss', 'recov', 'Qhit_usd', 'e', 'p']
POL = {}      # (dataset, name) -> stats arrays (for paired comparisons)
REC = []      # records for results.json / README


def add_policy(ds, regime, name, W, comp, main=True, **tags):
    st = pol_stats(W, comp)
    POL[(ds, name)] = st
    rec = dict(dataset=ds, regime=regime, name=name, main=main, **tags)
    for k in KEEP:
        rec[k] = ci(st[k])
    if 'r_sec' in st:
        rec['r_sec'] = ci(st['r_sec']); rec['eta_sec'] = ci(st['eta_sec']); rec['eta_B_sec'] = ci(st['eta_B_sec'])
        rec['ident_sec_rel_max'] = nanmax(st['ident_sec_rel'])
    rec['ident_rel_max'] = nanmax(st['ident_rel'])
    rec['decomp_resid_max'] = float(max(nanmax(st['decompQ']), nanmax(st['decompS'])))
    Qc = rec['Q']
    for eps in EPS:
        rec[f'floor_{eps}'] = bool(Qc[0] >= -eps - 1e-12)
        rec[f'floor_{eps}_ci'] = 'yes' if Qc[1] >= -eps - 1e-12 else ('no' if Qc[2] < -eps else 'undecided')
    sat = [eps for eps in EPS if rec[f'floor_{eps}']]
    rec['floor_eps_min'] = (min(sat) if sat else None)
    rec['r_ci_excl_1'] = excl(rec['r'], 1.0)
    REC.append(rec)
    return st


def paired(ds, a, b):
    """Paired bootstrap comparison of two policies on the same resampled tasks: differences in eta/eta_B, net gain, E[V], E[D]."""
    A_, B_ = POL[(ds, a)], POL[(ds, b)]
    out = dict(a=a, b=b)
    for k in ['r', 'net_usd', 'EV', 'ED', 'Q']:
        out['d_' + k] = ci(A_[k] - B_[k])
        out['d_' + k + '_decided'] = excl(out['d_' + k])
    return out


# ================================================================================================ 1. CodeRouterBench id_test
log('loading CodeRouterBench')
df = pd.read_csv(f'{DATA}/coderouterbench/id_results_long.csv')
assert df.shape[0] == 79992
S = df.pivot(index='task_id', columns='model', values='score')
C = df.pivot(index='task_id', columns='model', values='cost_usd')
L = df.pivot(index='task_id', columns='model', values='latency_ms') / 1000.0
TI = df.pivot(index='task_id', columns='model', values='input_tokens')
assert S.notna().all().all() and C.notna().all().all() and L.notna().all().all()
meta = df.drop_duplicates('task_id').set_index('task_id').loc[S.index, ['split', 'dimension']]
TASKS = S.index.to_numpy(); MODELS = list(S.columns); col = {m: j for j, m in enumerate(MODELS)}
Sv, Cv, Lv, TIv = S.to_numpy(float), C.to_numpy(float), L.to_numpy(float), TI.to_numpy(float)
idt = np.where(meta.split.to_numpy() == 'id_test')[0]
assert len(idt) == 2919
id_ids = TASKS[idt]
so, co, lo = Sv[idt, col[BASE]], Cv[idt, col[BASE]], Lv[idt, col[BASE]]
W_id = boot_weights(len(idt), rng)
eta_B_id = so.sum() / co.sum()
# prompt-token proxy for the router bill: the task prompt is at most the smallest input_tokens any model recorded for it
prompt_tok = np.nanmin(np.where(TIv[idt] > 0, TIv[idt], np.nan), axis=1)
k_proxy = prompt_tok * ROUTER_TOKEN_PRICE
DATA_CHECKS = dict(
    id_n_tasks=int(len(TASKS)), id_n_test=int(len(idt)), id_models=MODELS,
    always_opus_id_test_total_usd=float(co.sum()), paper_always_opus_total_usd=34.02,
    always_opus_id_test_mean_score=float(so.mean()), paper_always_opus_avgperf=43.83, eta_B_id_test_per_usd=float(eta_B_id),
    always_opus_id_test_mean_latency_s=float(lo.mean()),
    qwen3max_zero_cost_rows_id_test=int((Cv[idt, col['Qwen3-Max']] <= 0).sum()),
    router_prompt_token_proxy_mean_tokens=float(prompt_tok.mean()), router_prompt_token_proxy_mean_usd=float(k_proxy.mean()),
)
assert abs(DATA_CHECKS['always_opus_id_test_total_usd'] - 34.02) < 0.01


def exante_comp(cols_, k=0.0):
    T = (cols_ != col[BASE]).astype(float)
    Vch, Dch, Lch = Sv[idt, cols_], Cv[idt, cols_], Lv[idt, cols_]
    return dict(H=T, YP=Vch, A=k + T * Dch, YF=so, CF=co, YB=so, CB=co, LS=Lch, LB=lo)


def cascade_comp(m, theta, bfa=0.0, bfr=0.0, vc=0.0):
    sp, cp, lp = Sv[idt, col[m]], Cv[idt, col[m]], Lv[idt, col[m]]
    exact = (sp >= theta - 1e-12).astype(float)
    H = exact * (1 - bfr) + (1 - exact) * bfa          # expectation over the verifier's independent flips
    return dict(H=H, YP=sp, A=cp + vc, YF=so, CF=co, YB=so, CB=co, LS=lp + (1 - H) * lo, LB=lo)


def hybrid_comp(cols_, theta, k=0.0, vc=0.0):
    """Router decisions used as the primary of a verify-then-escalate cascade: cheap choice -> oracle verifier -> Opus; Opus choices direct."""
    T = (cols_ != col[BASE]).astype(float)
    Vch, Dch, Lch = Sv[idt, cols_], Cv[idt, cols_], Lv[idt, cols_]
    hit = (Vch >= theta - 1e-12).astype(float); H = T * hit
    return dict(H=H, YP=Vch, A=k + T * (Dch + vc), YF=so, CF=co, YB=so, CB=co, LS=T * (Lch + (1 - hit) * lo) + (1 - T) * lo, LB=lo)


def hybrid_opt_comp(cols_, lam=None):
    """Exact per-decision bound for 'same primary decisions + escalate to Opus': a cheap choice is escalated iff the reference's
    outcome is worth its price at lambda, i.e. accept iff Y_P - Y_B + lam * C_B >= 0.  Needs Y_B (an oracle about the reference,
    not a verifier); with lam = eta_B it maximizes the net gain Q + eta_B S of thm:surplus over all escalation rules on these decisions."""
    lam = eta_B_id if lam is None else lam
    T = (cols_ != col[BASE]).astype(float)
    Vch, Dch, Lch = Sv[idt, cols_], Cv[idt, cols_], Lv[idt, cols_]
    acc = (Vch - so + lam * co >= -1e-12).astype(float); H = T * acc
    return dict(H=H, YP=Vch, A=T * Dch, YF=so, CF=co, YB=so, CB=co, LS=T * (Lch + (1 - acc) * lo) + (1 - T) * lo, LB=lo)


def load_decisions(path):
    r = pd.read_json(path, lines=True)
    n_file = len(r)
    r = r[r.task_id.isin(set(id_ids))].drop_duplicates('task_id').set_index('task_id').reindex(id_ids)
    assert r.chosen_model.notna().all(), path
    return np.array([col[ALIAS.get(m, m)] for m in r.chosen_model]), n_file, {k: int(v) for k, v in r.chosen_model.value_counts().items()}


DS = 'id_test'
add_policy(DS, 'ref', 'always-Opus (reference)', W_id, exante_comp(np.full(len(idt), col[BASE])))
for m in MODELS:
    if m != BASE:
        add_policy(DS, 'const', f'always-{m}', W_id, exante_comp(np.full(len(idt), col[m])), main=False)
bound_cols = np.argmax(Sv[idt] - eta_B_id * Cv[idt], axis=1)
add_policy(DS, 'bound', 'per-task oracle (max V - eta_B D)', W_id, exante_comp(bound_cols), main=False)
R1_SHARES = {}
for name in ROUTERS:
    cols_, n_file, share = load_decisions(f'{DATA}/id/voter_decisions/{name}.jsonl')
    R1_SHARES[name] = dict(n_rows_in_file=n_file, share=share)
    ks = [0.0] if name not in FINETUNED else [0.0, 'proxy'] + K_ROUTER[1:]
    for k in ks:
        kk = k_proxy if k == 'proxy' else float(k)
        add_policy(DS, 'R1', f'R1 {name}' + ('' if k == 0.0 else f' (k={k if k == "proxy" else "$%g" % k})'), W_id, exante_comp(cols_, kk),
                   main=(k == 0.0), router=name, k=(k if k == 'proxy' else float(k)))
    for th in THETAS:
        add_policy(DS, 'R3-cf', f'R3-cf {name} + oracle theta={th} -> Opus', W_id, hybrid_comp(cols_, th), main=False, router=name, theta=th)
acr_cols, acr_n, acr_share = load_decisions(ID_DECISIONS)
R1_SHARES['acrouter_id_map'] = dict(n_rows_in_file=acr_n, share=acr_share)
add_policy(DS, 'R1', 'R1 ACRouter ID map (lookup)', W_id, exante_comp(acr_cols), router='acrouter_id_map', k=0.0)
BOUND_ACR = 'bound: ACRouter map + per-task optimal escalation (price eta_B)'
add_policy(DS, 'bound', BOUND_ACR, W_id, hybrid_opt_comp(acr_cols), main=False, router='acrouter_id_map')
for th in THETAS:
    add_policy(DS, 'R3-cf', f'R3-cf ACRouter map + oracle theta={th} -> Opus', W_id, hybrid_comp(acr_cols, th), router='acrouter_id_map', theta=th)
    for vc in VC_ID[1:]:
        add_policy(DS, 'R3-cf', f'R3-cf ACRouter map + oracle theta={th} -> Opus (v_c=${vc:g})', W_id, hybrid_comp(acr_cols, th, vc=vc), main=False,
                   router='acrouter_id_map', theta=th, v_c=vc)
for m in PRIMARIES:
    for th in THETAS:
        add_policy(DS, 'R2', f'R2 {m} theta={th}', W_id, cascade_comp(m, th), primary=m, theta=th, verifier='oracle', v_c=0.0)
        for vc in VC_ID[1:]:
            add_policy(DS, 'R2', f'R2 {m} theta={th} (v_c=${vc:g})', W_id, cascade_comp(m, th, vc=vc), main=False, primary=m, theta=th, verifier='oracle', v_c=vc)
        for bfa, bfr in NOISY:
            for vc in VC_ID:
                add_policy(DS, 'R2', f'R2 {m} theta={th} noisy fa={bfa} fr={bfr} (v_c=${vc:g})', W_id, cascade_comp(m, th, bfa, bfr, vc), main=False,
                           primary=m, theta=th, verifier=f'noisy_fa{bfa}_fr{bfr}', v_c=vc)
log('id_test policies', sum(r['dataset'] == DS for r in REC))

# ================================================================================================ 2. OOD176
log('loading OOD176')
oo = pd.read_csv(f'{DATA}/coderouterbench/ood176_results_long.csv')
assert oo.shape[0] == 1408
Ro = oo.pivot(index='task_id', columns='model', values='resolved').astype(float)
Co = oo.pivot(index='task_id', columns='model', values='cost_usd').astype(float)
Ao = oo.pivot(index='task_id', columns='model', values='apply_ok').astype(float)
OTASKS = Ro.index.to_numpy(); OMODELS = list(Ro.columns); ocol = {m: j for j, m in enumerate(OMODELS)}
Rv, Cov, Av = Ro.to_numpy(), Co.to_numpy(), Ao.to_numpy()
ometa = oo.drop_duplicates('task_id').set_index('task_id').loc[OTASKS, ['source_split', 'original_task_id']]
is112 = (ometa.source_split.to_numpy() == 'old112')
n_o = len(OTASKS)
W_ood = boot_weights(n_o, rng)
# imputed-cost matrix: unrecorded cells (cost 0 = no run recorded) get the model's mean recorded old112 cost
Cimp = Cov.copy()
for j, m in enumerate(OMODELS):
    rec_ = Cov[is112, j]; mu = rec_[rec_ > 0].mean()
    Cimp[Cov[:, j] <= 0, j] = mu
ob = ocol[BASE]
yB_o, cB_o = Rv[:, ob], Cov[:, ob]
eta_B_ood = yB_o.sum() / cB_o.sum()
DATA_CHECKS.update(ood_n_tasks=int(n_o), ood_n_old112=int(is112.sum()), ood_n_new64=int((~is112).sum()),
                   always_opus_ood176_resolved=float(yB_o.mean()), always_opus_ood176_total_usd=float(cB_o.sum()), eta_B_ood176_per_usd=float(eta_B_ood),
                   ood_unrecorded_cells=int((Cov <= 0).sum()), ood_unrecorded_cells_by_model={m: int((Cov[:, ocol[m]] <= 0).sum()) for m in OMODELS},
                   ood_new64_costs_constant_per_model=bool(all(len(np.unique(Cov[~is112, j])) == 1 for j in range(len(OMODELS)))),
                   paper_table_acrouter_ood176=dict(AvgPerf=73.30, total_usd=86.72, perf_per_usd=0.85))


def chain_comp(chain, gate_k, escalate=True, Cmat=None, vc=0.0, subset=None):
    """Replay of ACRouter's OOD chain on the outcome matrix: attempt the chain in order, stop at the first resolved output;
    if none resolved and at least gate_k cheap attempts produced an applicable patch (apply_ok), escalate to Opus.
    H = 1{Opus not run}; Y_P = resolved of the last cheap attempt; A = sum of cheap attempt costs (+ v_c per attempt)."""
    Cm = Cov if Cmat is None else Cmat
    idx = np.arange(n_o) if subset is None else subset
    H = np.ones(len(idx)); YP = np.zeros(len(idx)); A = np.zeros(len(idx)); runs = []; natt = np.zeros(len(idx))
    for ii, i in enumerate(idx):
        cnt = 0; res = 0.0; a = 0.0; run = []
        for m in chain:
            j = ocol[m]; run.append(m); a += Cm[i, j] + vc; cnt += int(Av[i, j] > 0); res = Rv[i, j]
            if res >= 1:
                break
        if res < 1 and escalate and cnt >= gate_k:
            H[ii] = 0.0; run.append(BASE)
        YP[ii] = res; A[ii] = a; natt[ii] = len([m for m in run if m != BASE]); runs.append(run)
    return dict(H=H, YP=YP, A=A, YF=Rv[idx, ob], CF=Cm[idx, ob], YB=Rv[idx, ob], CB=Cm[idx, ob]), runs, natt


def ood_exante_comp(cols_, Cmat=None, subset=None):
    Cm = Cov if Cmat is None else Cmat
    idx = np.arange(n_o) if subset is None else subset
    T = (cols_[idx] != ob).astype(float)
    return dict(H=T, YP=Rv[idx, cols_[idx]], A=T * Cm[idx, cols_[idx]], YF=Rv[idx, ob], CF=Cm[idx, ob], YB=Rv[idx, ob], CB=Cm[idx, ob])


def ood_cascade_comp(m, Cmat=None, vc=0.0, subset=None):
    Cm = Cov if Cmat is None else Cmat
    idx = np.arange(n_o) if subset is None else subset
    j = ocol[m]; H = (Rv[idx, j] >= 1).astype(float)
    return dict(H=H, YP=Rv[idx, j], A=Cm[idx, j] + vc, YF=Rv[idx, ob], CF=Cm[idx, ob], YB=Rv[idx, ob], CB=Cm[idx, ob])


# the released trace, and the check that it is exactly the matrix replay with gate k = 2
trace = pd.read_json(OOD_TRACE, lines=True).set_index('task_id').reindex(OTASKS)
assert trace.chosen_model.notna().all()
comp_tr, runs_tr, natt_tr = chain_comp(OOD_CHAIN, gate_k=2)
D_replay = comp_tr['A'] + (1 - comp_tr['H']) * comp_tr['CF']
V_replay = comp_tr['H'] * comp_tr['YP'] + (1 - comp_tr['H']) * comp_tr['YF']
TRACE_CHECK = dict(
    chain_run_identical=bool(all(list(a) == list(b) for a, b in zip(trace.chain_run, runs_tr))),
    resolved_identical=bool(np.all(trace.resolved.astype(float).to_numpy() == V_replay)),
    escalated_identical=bool(np.all(trace.escalated.astype(bool).to_numpy() == (comp_tr['H'] == 0))),
    cost_max_abs_diff_usd=float(np.abs(trace.cost_usd.to_numpy(float) - D_replay).max()),
    n_steps=trace.n_steps.value_counts().sort_index().to_dict(), n_escalated=int(trace.escalated.sum()),
    n_attempts_total=int(natt_tr.sum() + (comp_tr['H'] == 0).sum()),
    n_cheap_attempts_unrecorded=int(sum(int(Cov[i, ocol[m]] <= 0) for i, run in enumerate(runs_tr) for m in run if m != BASE)),
    n_tasks_with_unrecorded_attempt=int(sum(any(Cov[i, ocol[m]] <= 0 for m in run if m != BASE) for i, run in enumerate(runs_tr))),
    metrics_reproduced=dict(AvgPerf=float(100 * V_replay.mean()), total_usd=float(D_replay.sum()), perf_per_usd=float(100 * V_replay.mean() / D_replay.sum())))
assert TRACE_CHECK['chain_run_identical'] and TRACE_CHECK['resolved_identical'] and TRACE_CHECK['cost_max_abs_diff_usd'] < 1e-4
log('OOD trace replay check', TRACE_CHECK['metrics_reproduced'])

DS = 'ood176'
add_policy(DS, 'ref', 'always-Opus (reference)', W_ood, ood_exante_comp(np.full(n_o, ob)))
for m in OMODELS:
    if m != BASE:
        add_policy(DS, 'const', f'always-{m}', W_ood, ood_exante_comp(np.full(n_o, ocol[m])), main=False)
add_policy(DS, 'bound', 'per-task oracle (max V - eta_B D)', W_ood, ood_exante_comp(np.argmax(Rv - eta_B_ood * Cov, axis=1)), main=False)
OOD_R1_FILES = {'LogReg': 'R1', 'RouteLLM_MF': 'R1', 'RouteLLM_BERT': 'R1', 'TF_IDF_MLP': 'R1', 'Qwen3_5_0_8B_Finetuned': 'R1',
                'kNN_Retrieval': 'R1-mem', 'LinTS': 'R1-mem', 'LinUCB': 'R1-mem'}
OOD_R1_SHARES = {}
for fname, regime in OOD_R1_FILES.items():
    d = pd.read_json(f'{OOD_BASELINES}/{fname}.jsonl', lines=True).drop_duplicates('task_id').set_index('task_id').reindex(OTASKS)
    assert d.chosen_model.notna().all(), fname
    cols_ = np.array([ocol[ALIAS.get(m, m)] for m in d.chosen_model])
    OOD_R1_SHARES[fname] = {k: int(v) for k, v in pd.Series([OMODELS[c] for c in cols_]).value_counts().items()}
    add_policy(DS, regime, f'{regime} {fname}', W_ood, ood_exante_comp(cols_), router=fname)
    add_policy(DS, regime, f'{regime} {fname} (imputed attempt costs)', W_ood, ood_exante_comp(cols_, Cimp), main=False, router=fname, costs='imputed')
for m in OOD_CHAIN + [x for x in OMODELS if x not in OOD_CHAIN + [BASE]]:
    add_policy(DS, 'R2', f'R2 {m} + oracle -> Opus', W_ood, ood_cascade_comp(m), primary=m, verifier='oracle', v_c=0.0)
    add_policy(DS, 'R2', f'R2 {m} + oracle -> Opus (imputed attempt costs)', W_ood, ood_cascade_comp(m, Cimp), main=False, primary=m, costs='imputed')
    for vc in VC_OOD[1:]:
        add_policy(DS, 'R2', f'R2 {m} + oracle -> Opus (v_c=${vc:g})', W_ood, ood_cascade_comp(m, vc=vc), main=False, primary=m, v_c=vc)
# R3: the released trace (D = matrix sum of attempted models' costs), the reported-cost variant, and sensitivities
add_policy(DS, 'R3', 'R3 ACRouter chain trace (gate k=2)', W_ood, comp_tr, gate_k=2, costs='matrix_sum')
comp_rep = dict(comp_tr); comp_rep['A'] = trace.cost_usd.to_numpy(float) - (1 - comp_tr['H']) * cB_o
add_policy(DS, 'R3', 'R3 ACRouter chain trace (reported cost_usd)', W_ood, comp_rep, main=False, gate_k=2, costs='reported')
add_policy(DS, 'R3', 'R3 ACRouter chain trace (imputed attempt costs)', W_ood, chain_comp(OOD_CHAIN, 2, Cmat=Cimp)[0], main=False, gate_k=2, costs='imputed')
for vc in VC_OOD[1:]:
    add_policy(DS, 'R3', f'R3 ACRouter chain trace (v_c=${vc:g} per attempt)', W_ood, chain_comp(OOD_CHAIN, 2, vc=vc)[0], main=False, gate_k=2, v_c=vc)
# chain variants (where does the verification pay?): prefixes with unconditional escalation, and the gate k
CHAIN_VARIANTS = []
for Lc in range(1, 5):
    nm = f'chain[{Lc}] -> Opus always'
    add_policy(DS, 'R3', f'R3 variant {nm}', W_ood, chain_comp(OOD_CHAIN[:Lc], 0)[0], main=False, variant=nm, chain_len=Lc, gate_k=0)
    CHAIN_VARIANTS.append(nm)
for gk in [1, 2, 3, 4]:
    nm = f'chain[4] gate k={gk}'
    if gk != 2:
        add_policy(DS, 'R3', f'R3 variant {nm}', W_ood, chain_comp(OOD_CHAIN, gk)[0], main=False, variant=nm, chain_len=4, gate_k=gk)
    CHAIN_VARIANTS.append(nm)
nm = 'chain[4] never escalate'
add_policy(DS, 'R3', f'R3 variant {nm}', W_ood, chain_comp(OOD_CHAIN, 0, escalate=False)[0], main=False, variant=nm, chain_len=4, gate_k=None)
CHAIN_VARIANTS.append(nm)
# the escalation gate as a signal about Opus: among chain-failed tasks, does "applied_in_cheap >= 2" predict that Opus resolves?
chain_failed = comp_tr['YP'] < 1
esc = comp_tr['H'] == 0
GATE = dict(n_chain_failed=int(chain_failed.sum()), n_escalated=int(esc.sum()),
            p_opus_resolves_given_escalated=float(yB_o[esc].mean()),
            p_opus_resolves_given_chain_failed_not_escalated=float(yB_o[chain_failed & ~esc].mean()), n_chain_failed_not_escalated=int((chain_failed & ~esc).sum()),
            p_opus_resolves_given_chain_succeeded=float(yB_o[~chain_failed].mean()),
            gate_TPR_for_opus_success=float(esc[chain_failed & (yB_o >= 1)].mean()), gate_FPR_for_opus_success=float(esc[chain_failed & (yB_o < 1)].mean()),
            n_chain_failed_opus_would_resolve=int((chain_failed & (yB_o >= 1)).sum()))
# old112 / new64 subsets (real per-task costs vs imputed constant costs in the released matrix)
SUBSETS = {}
for sub_name, mask in [('old112', is112), ('new64', ~is112)]:
    idx = np.where(mask)[0]; Ws = boot_weights(len(idx), rng)
    sub = {}
    sub['reference'] = pol_stats(Ws, ood_exante_comp(np.full(n_o, ob), subset=idx))
    sub['R3 trace'] = pol_stats(Ws, chain_comp(OOD_CHAIN, 2, subset=idx)[0])
    sub['R3 trace (imputed)'] = pol_stats(Ws, chain_comp(OOD_CHAIN, 2, Cmat=Cimp, subset=idx)[0])
    for m in OOD_CHAIN:
        sub[f'R2 {m}'] = pol_stats(Ws, ood_cascade_comp(m, subset=idx))
    for fname in ['LogReg', 'RouteLLM_MF', 'kNN_Retrieval']:
        d = pd.read_json(f'{OOD_BASELINES}/{fname}.jsonl', lines=True).drop_duplicates('task_id').set_index('task_id').reindex(OTASKS)
        cols_ = np.array([ocol[ALIAS.get(m, m)] for m in d.chosen_model])
        sub[f'R1 {fname}'] = pol_stats(Ws, ood_exante_comp(cols_, subset=idx))
    SUBSETS[sub_name] = {k: dict(r=ci(v['r']), EV=ci(v['EV']), ED=ci(v['ED']), Q=ci(v['Q']), e=ci(v['e']), n=int(len(idx))) for k, v in sub.items()}
# verification seconds from the OOD112 sandbox trace (per-attempt sandbox wall-clock; the verifier's verdict equals `resolved`)
VERIF = dict(available=os.path.exists(OOD112_SANDBOX))
if VERIF['available']:
    sb = pd.read_json(OOD112_SANDBOX, lines=True)
    rows = [dict(task_id=r.task_id, model=v.get('model'), elapsed_s=float(v.get('elapsed_s') or 0.0), resolved=bool(v.get('resolved')),
                 apply_ok=bool(v.get('apply_ok')), error=v.get('error'), from_cache=bool(v.get('from_cache'))) for r in sb.itertuples() for v in r.verification]
    vf = pd.DataFrame(rows)
    orig = ometa.original_task_id.to_numpy()
    m112 = {(t, m): (Rv[i, ocol[m]], Av[i, ocol[m]], Cov[i, ocol[m]]) for i, t in enumerate(orig) if is112[i] for m in OMODELS}
    vf['mat_resolved'] = [m112[(t, m)][0] for t, m in zip(vf.task_id, vf.model)]
    vf['mat_cost'] = [m112[(t, m)][2] for t, m in zip(vf.task_id, vf.model)]
    per_task = vf.groupby('task_id').elapsed_s.sum()
    chain112 = {r.task_id: list(r.chain_run) for r in sb.itertuples()}
    same_chain = all(chain112[orig[i]] == runs_tr[i] for i in range(n_o) if is112[i] and orig[i] in chain112)
    VERIF.update(n_tasks=int(len(sb)), n_attempts=int(len(vf)), verdict_equals_matrix_resolved=float((vf.resolved.astype(int) == vf.mat_resolved).mean()),
                 chain_identical_to_ood176_trace_on_old112=bool(same_chain),
                 elapsed_s_mean_per_attempt=float(vf.elapsed_s.mean()), elapsed_s_median_per_attempt=float(vf.elapsed_s.median()),
                 share_attempts_elapsed_zero=float((vf.elapsed_s <= 0).mean()), share_elapsed_zero_with_patch_error=float(vf[vf.elapsed_s <= 0].error.notna().mean()),
                 errors=vf.error.dropna().value_counts().to_dict(), from_cache_share=float(vf.from_cache.mean()),
                 elapsed_s_when_run_recorded=float(vf[vf.mat_cost > 0].elapsed_s.mean()), elapsed_s_when_run_unrecorded=float(vf[vf.mat_cost <= 0].elapsed_s.mean()),
                 per_task_total_s=dict(mean=float(per_task.mean()), median=float(per_task.median()), max=float(per_task.max())),
                 per_model_mean_s=vf.groupby('model').elapsed_s.mean().round(2).to_dict(),
                 per_attempt_mean_s_first_attempt_minimax=float(vf[vf.model == 'MiniMax-M2.7'].elapsed_s.mean()))
log('ood176 policies', sum(r['dataset'] == DS for r in REC))

# ================================================================================================ 3. RouterBench
log('loading RouterBench')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sps

rb = pd.read_pickle(RB_PKL)
REF = 'gpt-4-1106-preview'
RBM = [c for c in rb.columns if c + '|total_cost' in rb.columns]
rcol = {m: j for j, m in enumerate(RBM)}; rb_ref = rcol[REF]
Srb = rb[RBM].to_numpy(float); Crb = rb[[m + '|total_cost' for m in RBM]].to_numpy(float)
assert np.isfinite(Srb).all() and np.isfinite(Crb).all()
tr_idx, te_idx = train_test_split(np.arange(len(rb)), test_size=0.3, random_state=SEED, stratify=rb.eval_name)
tr_idx, te_idx = np.sort(tr_idx), np.sort(te_idx)
n_te = len(te_idx)
yB_rb, cB_rb = Srb[te_idx, rb_ref], Crb[te_idx, rb_ref]
eta_B_rb_train = Srb[tr_idx, rb_ref].sum() / Crb[tr_idx, rb_ref].sum()
eta_B_rb = yB_rb.sum() / cB_rb.sum()
W_rb = boot_weights(n_te, rng)
vec = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True, min_df=2, dtype=np.float32)
Xtr = vec.fit_transform(rb.prompt.iloc[tr_idx].astype(str)); Xte = vec.transform(rb.prompt.iloc[te_idx].astype(str))
ohe = OneHotEncoder(handle_unknown='ignore')
Etr = ohe.fit_transform(rb[['eval_name']].iloc[tr_idx].astype(str)); Ete = ohe.transform(rb[['eval_name']].iloc[te_idx].astype(str))
Xtr2, Xte2 = sps.hstack([Xtr, Etr]).tocsr(), sps.hstack([Xte, Ete]).tocsr()
P_hat = {}; AUROC = {}
for feat, (A_tr, A_te) in {'tfidf': (Xtr, Xte), 'tfidf+eval': (Xtr2, Xte2)}.items():
    P = np.zeros((n_te, len(RBM)))
    for j, m in enumerate(RBM):
        y = (Srb[tr_idx, j] >= 0.5).astype(int)
        lr = LogisticRegression(C=1.0, solver='liblinear', max_iter=1000)
        lr.fit(A_tr, y)
        P[:, j] = lr.predict_proba(A_te)[:, 1]
        AUROC[(feat, m)] = float(roc_auc_score((Srb[te_idx, j] >= 0.5).astype(int), P[:, j]))
    P_hat[feat] = P
    log(f'RouterBench LR routers ({feat}) fitted')
# cost prior: train mean cost per (eval_name, model); fallback model mean
ev_tr = rb.eval_name.iloc[tr_idx].to_numpy(); ev_te = rb.eval_name.iloc[te_idx].to_numpy()
cost_prior = pd.DataFrame(Crb[tr_idx], columns=RBM).assign(ev=ev_tr).groupby('ev').mean()
model_mean = Crb[tr_idx].mean(0)
C_hat = np.vstack([cost_prior.loc[e].to_numpy() if e in cost_prior.index else model_mean for e in ev_te])
DATA_CHECKS.update(rb_n=int(len(rb)), rb_n_train=int(len(tr_idx)), rb_n_test=int(n_te), rb_models=RBM, rb_reference=REF,
                   rb_ref_mean_score_test=float(yB_rb.mean()), rb_ref_mean_cost_test=float(cB_rb.mean()), eta_B_rb_test_per_usd=float(eta_B_rb),
                   eta_B_rb_train_per_usd=float(eta_B_rb_train), rb_score_binary_share=float(np.isin(Srb, [0, 1]).mean()),
                   rb_eval_names=int(rb.eval_name.nunique()), rb_duplicate_prompts=int(rb.prompt.duplicated().sum()),
                   rb_auroc_tfidf={m: AUROC[('tfidf', m)] for m in RBM}, rb_auroc_tfidf_eval={m: AUROC[('tfidf+eval', m)] for m in RBM})


def rb_exante_comp(cols_, k=0.0):
    T = (cols_ != rb_ref).astype(float)
    return dict(H=T, YP=Srb[te_idx, cols_], A=k + T * Crb[te_idx, cols_], YF=yB_rb, CF=cB_rb, YB=yB_rb, CB=cB_rb)


def rb_cascade_comp(m, theta, bfa=0.0, bfr=0.0, vc=0.0):
    j = rcol[m]; sp, cp = Srb[te_idx, j], Crb[te_idx, j]
    exact = (sp >= theta - 1e-12).astype(float); H = exact * (1 - bfr) + (1 - exact) * bfa
    return dict(H=H, YP=sp, A=cp + vc, YF=yB_rb, CF=cB_rb, YB=yB_rb, CB=cB_rb)


def rb_hybrid_comp(cols_, theta):
    T = (cols_ != rb_ref).astype(float); Vch, Dch = Srb[te_idx, cols_], Crb[te_idx, cols_]
    H = T * (Vch >= theta - 1e-12).astype(float)
    return dict(H=H, YP=Vch, A=T * Dch, YF=yB_rb, CF=cB_rb, YB=yB_rb, CB=cB_rb)


def rb_hybrid_opt_comp(cols_):
    T = (cols_ != rb_ref).astype(float); Vch, Dch = Srb[te_idx, cols_], Crb[te_idx, cols_]
    H = T * (Vch - yB_rb + eta_B_rb * cB_rb >= -1e-12).astype(float)
    return dict(H=H, YP=Vch, A=T * Dch, YF=yB_rb, CF=cB_rb, YB=yB_rb, CB=cB_rb)


DS = 'routerbench'
add_policy(DS, 'ref', f'always-{REF} (reference)', W_rb, rb_exante_comp(np.full(n_te, rb_ref)))
RB_CHEAP = [m for m in RBM if m != REF]
for m in RB_CHEAP:
    add_policy(DS, 'const', f'always-{m}', W_rb, rb_exante_comp(np.full(n_te, rcol[m])), main=False)
add_policy(DS, 'bound', 'per-task oracle (max V - eta_B D)', W_rb, rb_exante_comp(np.argmax(Srb[te_idx] - eta_B_rb * Crb[te_idx], axis=1)), main=False)
LAMBDAS = [0.0, 0.5, 1.0, 2.0, 4.0]
RB_R1 = {}
for feat in ['tfidf', 'tfidf+eval']:
    for lam_mult in LAMBDAS:
        if feat == 'tfidf+eval' and lam_mult not in (0.0, 1.0):
            continue
        lam = lam_mult * eta_B_rb_train
        cols_ = np.argmax(P_hat[feat] - lam * C_hat, axis=1)
        nm = f'R1 LR router ({feat}, lambda={lam_mult:g} eta_B)'
        RB_R1[nm] = dict(share={RBM[c]: int(v) for c, v in zip(*np.unique(cols_, return_counts=True))}, share_ref=float((cols_ == rb_ref).mean()), lam=float(lam))
        add_policy(DS, 'R1', nm, W_rb, rb_exante_comp(cols_), main=(feat == 'tfidf'), features=feat, lambda_mult=lam_mult)
        if lam_mult == 1.0:
            add_policy(DS, 'R3-cf', f'R3-cf LR router ({feat}, lambda=1 eta_B) + oracle theta=1 -> {REF}', W_rb, rb_hybrid_comp(cols_, 1.0), main=(feat == 'tfidf'), features=feat)
            if feat == 'tfidf':
                BOUND_RB = 'bound: LR router (tfidf, lambda=1 eta_B) + per-task optimal escalation (price eta_B)'
                add_policy(DS, 'bound', BOUND_RB, W_rb, rb_hybrid_opt_comp(cols_), main=False, features=feat)
RB_NOISY_PRIMARIES = ['mistralai/mixtral-8x7b-chat', 'zero-one-ai/Yi-34B-Chat', 'gpt-3.5-turbo-1106', 'claude-instant-v1']
for m in RB_CHEAP:
    for th in THETAS:
        add_policy(DS, 'R2', f'R2 {m} theta={th}', W_rb, rb_cascade_comp(m, th), primary=m, theta=th, verifier='oracle', v_c=0.0)
        if m in RB_NOISY_PRIMARIES:
            for vc in VC_RB[1:]:
                add_policy(DS, 'R2', f'R2 {m} theta={th} (v_c=${vc:g})', W_rb, rb_cascade_comp(m, th, vc=vc), main=False, primary=m, theta=th, verifier='oracle', v_c=vc)
            for bfa, bfr in NOISY:
                for vc in VC_RB:
                    add_policy(DS, 'R2', f'R2 {m} theta={th} noisy fa={bfa} fr={bfr} (v_c=${vc:g})', W_rb, rb_cascade_comp(m, th, bfa, bfr, vc), main=False,
                               primary=m, theta=th, verifier=f'noisy_fa{bfa}_fr{bfr}', v_c=vc)
log('routerbench policies', sum(r['dataset'] == DS for r in REC))

# ================================================================================================ 4. summaries, paired comparisons, claims
R = pd.DataFrame(REC)
R['r_pt'] = R.r.apply(lambda t: t[0]); R['Q_pt'] = R.Q.apply(lambda t: t[0]); R['EV_pt'] = R.EV.apply(lambda t: t[0]); R['ED_pt'] = R.ED.apply(lambda t: t[0])


def best(ds, regime, mask=None):
    d = R[(R.dataset == ds) & (R.regime == regime) & R.main]
    if mask is not None:
        d = d[mask(d)]
    return d.sort_values('r_pt', ascending=False).iloc[0]


BEST = {}
BEST['id_test'] = dict(R1=best('id_test', 'R1')['name'],
                       R2=best('id_test', 'R2')['name'],
                       R3=best('id_test', 'R3-cf', lambda d: d.router == 'acrouter_id_map')['name'])
BEST['ood176'] = dict(R1=best('ood176', 'R1')['name'], R1mem=best('ood176', 'R1-mem')['name'], R2=best('ood176', 'R2')['name'],
                      R3='R3 ACRouter chain trace (gate k=2)')
BEST['routerbench'] = dict(R1=best('routerbench', 'R1')['name'], R2=best('routerbench', 'R2')['name'], R3=best('routerbench', 'R3-cf')['name'])
# floor-feasible bests (eps <= 0.01)
BEST_FLOOR = {}
for ds in ['id_test', 'ood176', 'routerbench']:
    BEST_FLOOR[ds] = {}
    for rg in ['R1', 'R2', 'R3', 'R3-cf', 'R1-mem']:
        d = R[(R.dataset == ds) & (R.regime == rg) & R.main & (R['floor_0.01'])]
        if ds == 'id_test' and rg == 'R3-cf':
            d = d[d.router == 'acrouter_id_map']
        BEST_FLOOR[ds][rg] = None if len(d) == 0 else d.sort_values('r_pt', ascending=False).iloc[0]['name']
PAIRED = {}
for ds in ['id_test', 'ood176', 'routerbench']:
    b = BEST[ds]; PAIRED[ds] = {}
    keys = list(b.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            PAIRED[ds][f'{keys[i]}_vs_{keys[j]}'] = paired(ds, b[keys[i]], b[keys[j]])
    PAIRED[ds]['floor_feasible_bests'] = BEST_FLOOR[ds]
    fb = {k: v for k, v in BEST_FLOOR[ds].items() if v is not None}
    fk = list(fb.keys())
    for i in range(len(fk)):
        for j in range(i + 1, len(fk)):
            PAIRED[ds][f'floor_{fk[i]}_vs_{fk[j]}'] = paired(ds, fb[fk[i]], fb[fk[j]])
# R3-OOD versus the same-primary R2 and versus chain variants
PAIRED['ood176']['R3_vs_R2_MiniMax'] = paired('ood176', BEST['ood176']['R3'], 'R2 MiniMax-M2.7 + oracle -> Opus')
PAIRED['ood176']['R3_vs_chain4_always_escalate'] = paired('ood176', BEST['ood176']['R3'], 'R3 variant chain[4] -> Opus always')
PAIRED['ood176']['R3_vs_chain4_never_escalate'] = paired('ood176', BEST['ood176']['R3'], 'R3 variant chain[4] never escalate')
PAIRED['ood176']['R3_vs_R2_gpt54'] = paired('ood176', BEST['ood176']['R3'], 'R2 gpt-5.4 + oracle -> Opus')
PAIRED['id_test']['R3cf_acr_theta1_vs_R1_acr'] = paired('id_test', 'R3-cf ACRouter map + oracle theta=1.0 -> Opus', 'R1 ACRouter ID map (lookup)')
PAIRED['id_test']['R3cf_acr_theta1_vs_best_R2'] = paired('id_test', 'R3-cf ACRouter map + oracle theta=1.0 -> Opus', BEST['id_test']['R2'])


def ranking(ds, names):
    d = R[(R.dataset == ds) & R.name.isin(names)].copy()
    d = d.sort_values('r_pt', ascending=False)
    out = []
    for i, (_, x) in enumerate(d.iterrows()):
        nxt = d.iloc[i + 1]['name'] if i + 1 < len(d) else None
        out.append(dict(rank=i + 1, name=x['name'], regime=x['regime'], r=x['r'], Q=x['Q'], e=x['e'], floor_001=bool(x['floor_0.01']),
                        gap_to_next_decided=(paired(ds, x['name'], nxt)['d_r_decided'] if nxt else None)))
    return out


RANK = {ds: ranking(ds, list(BEST[ds].values())) for ds in BEST}
RANK_FLOOR = {ds: ranking(ds, [v for v in BEST_FLOOR[ds].values() if v is not None]) for ds in BEST_FLOOR}


def rank_str(ds, floor=False):
    return ' > '.join(f"{x['regime']} {x['r'][0]:.2f} [{x['r'][1]:.2f}, {x['r'][2]:.2f}] ({short(x['name'])})" for x in (RANK_FLOOR if floor else RANK)[ds])


def fbs(ds):
    return ', '.join(f"{k}: {short(v)} ({R[(R.dataset == ds) & (R.name == v)].iloc[0].r_pt:.2f})" for k, v in BEST_FLOOR[ds].items() if v is not None)


def pol(ds, name):
    return R[(R.dataset == ds) & (R.name == name)].iloc[0]


def n_viol(ds, rg, eps=0.01):
    d = R[(R.dataset == ds) & (R.regime == rg) & R.main]
    return int((~d[f'floor_{eps}']).sum()), int(len(d))
IDENT = dict(n_policies=int(len(R)), max_ident_rel_dollars=float(R.ident_rel_max.max()), max_decomp_resid=float(R.decomp_resid_max.max()),
             max_ident_rel_seconds=float(R.ident_sec_rel_max.dropna().max()) if 'ident_sec_rel_max' in R else None,
             n_with_seconds=int(R.ident_sec_rel_max.notna().sum()) if 'ident_sec_rel_max' in R else 0,
             by_dataset={ds: float(R[R.dataset == ds].ident_rel_max.max()) for ds in ['id_test', 'ood176', 'routerbench']})
# R1 lookup routers vs R2 cascades under the floor
FLOOR = {}
for ds in ['id_test', 'ood176', 'routerbench']:
    d = R[(R.dataset == ds) & R.main & R.regime.isin(['R1', 'R2', 'R3', 'R3-cf', 'R1-mem'])]
    FLOOR[ds] = {}
    for rg in ['R1', 'R1-mem', 'R2', 'R3', 'R3-cf']:
        dd = d[d.regime == rg]
        if len(dd) == 0:
            continue
        FLOOR[ds][rg] = {f'eps_{eps}': dict(n=int(len(dd)), satisfied=int(dd[f'floor_{eps}'].sum()), decided_yes=int((dd[f'floor_{eps}_ci'] == 'yes').sum()),
                                            decided_no=int((dd[f'floor_{eps}_ci'] == 'no').sum())) for eps in EPS}
        FLOOR[ds][rg]['theta1_all_Q_nonneg'] = bool((dd[dd.get('theta', pd.Series(index=dd.index)) == 1.0].Q_pt >= -1e-12).all()) if 'theta' in dd else None
# id_test noisy/v_c summaries for the two focus primaries
NOISY_SUM = []
for m in FOCUS:
    for th in THETAS:
        for vc in VC_ID:
            for (bfa, bfr) in [(0, 0)] + NOISY:
                nm = f'R2 {m} theta={th}' + ('' if vc == 0 and (bfa, bfr) == (0, 0) else (f' (v_c=${vc:g})' if (bfa, bfr) == (0, 0) else f' noisy fa={bfa} fr={bfr} (v_c=${vc:g})'))
                x = R[(R.dataset == 'id_test') & (R.name == nm)].iloc[0]
                NOISY_SUM.append(dict(primary=m, theta=th, beta_fa=bfa, beta_fr=bfr, v_c=vc, r=x.r, Q=x.Q, e=x.e, EV=x.EV, ED=x.ED, floor_001=bool(x['floor_0.01']), floor_0=bool(x['floor_0.0'])))

# ------------------------------------------------------------------------------------------------ claims
CL = []


def claim(cid, kind, statement, result, verdict, note=''):
    CL.append(dict(id=cid, statement=statement, kind=kind, result=result, verdict=verdict, note=note))


def f3(t, nd=3):
    return f'{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]'


rk = lambda ds: ' > '.join(f"{x['regime']}({x['r'][0]:.2f})" for x in RANK[ds])
rkf = lambda ds: ' > '.join(f"{x['regime']}({x['r'][0]:.2f})" for x in RANK_FLOOR[ds])
claim('C1', 'IDENTITY', 'Lemma thm:surplus / eq:unified-identity hold for every policy of every regime (R1 ex-ante, R2 cascades incl. noisy verifiers and v_c, '
      'R3 chain trace and its variants, constants, oracle bound) on id_test, OOD176 and RouterBench, in dollars and (id_test) seconds; the four terms of eq:cascade-surplus sum to Q and S',
      f"{IDENT['n_policies']} policies x {B+1} bootstrap rows: max |residual|/|eta| = {IDENT['max_ident_rel_dollars']:.1e} (dollars), "
      f"{IDENT['max_ident_rel_seconds']:.1e} (seconds, {IDENT['n_with_seconds']} id_test policies); max decomposition residual {IDENT['max_decomp_resid']:.1e}",
      'PASS' if IDENT['max_ident_rel_dollars'] < 1e-9 and IDENT['max_decomp_resid'] < 1e-9 and (IDENT['max_ident_rel_seconds'] or 0) < 1e-9 else 'FAIL',
      'Code check: the identity is algebra on paired outcomes (Y_B, C_B observed for every task); the R3 chain uses H = 1{Opus not run}, A = sum of cheap attempt costs.')
dec_all = {ds: all(x['gap_to_next_decided'] for x in RANK[ds][:-1]) for ds in RANK}
claim('C2', 'IN-SAMPLE', 'Ranking of the regimes by eta/eta_B (best policy per regime, paired 95% CIs): id_test (R3 = counterfactual ACRouter map + oracle cascade), OOD176 (R3 = released trace), RouterBench (R1 = our LR router, R3 = LR + oracle cascade)',
      f"id_test: {rk('id_test')}; OOD176: {rk('ood176')}; RouterBench: {rk('routerbench')} (adjacent gaps CI-decided: id_test {dec_all['id_test']}, OOD176 {dec_all['ood176']}, RouterBench {dec_all['routerbench']}). "
      f"Under the quality floor eps <= 0.01: id_test {rkf('id_test')}; OOD176 {rkf('ood176')}; RouterBench {rkf('routerbench')}",
      'PASS' if all(dec_all.values()) else 'INCONCLUSIVE',
      'Descriptive: the best policy of each regime is selected on the same sample it is evaluated on (in-sample selection); RouterBench R1 decisions come from a router fitted on the disjoint 70% split.')
r3 = R[(R.dataset == 'ood176') & (R.name == BEST['ood176']['R3'])].iloc[0]
v_always = R[(R.dataset == 'ood176') & (R.name == 'R3 variant chain[4] -> Opus always')].iloc[0]
v_never = R[(R.dataset == 'ood176') & (R.name == 'R3 variant chain[4] never escalate')].iloc[0]
v_mm = R[(R.dataset == 'ood176') & (R.name == 'R2 MiniMax-M2.7 + oracle -> Opus')].iloc[0]
claim('C3', 'IN-SAMPLE', 'Where the in-loop verification of the R3-OOD trace pays: four-term decomposition of its net gain against always-Opus and the same terms for its chain variants '
      '(single primary + oracle -> Opus; longer chains; unconditional / gated / no escalation)',
      f"R3 trace: eta/eta_B {f3(r3.r, 2)}, E[V] {f3(r3.EV)} vs Opus {yB_o.mean():.3f}, E[D] ${r3.ED[0]:.3f} vs ${cB_o.mean():.3f}; net gain {f3(r3.net_usd, 3)} $/task = "
      f"avoided reference work {r3.avoided[0]:.3f} - attempt cost {r3.attempt[0]:.3f} + accepted quality change/eta_B {r3.Qhit_usd[0]:+.3f} - recovery {r3.recov[0]:.3f}; e = {r3.e[0]:.3f}. "
      f"Variants: MiniMax + oracle -> Opus (R2) {f3(v_mm.r, 2)}; chain[4] always escalate {f3(v_always.r, 2)} (E[V] {v_always.EV[0]:.3f}); chain[4] never escalate {f3(v_never.r, 2)} (E[V] {v_never.EV[0]:.3f}). "
      f"Gate 'applied_in_cheap >= 2' as a signal for Opus success among chain-failed tasks: P(Opus resolves | escalated) = {GATE['p_opus_resolves_given_escalated']:.2f} vs "
      f"P(Opus resolves | not escalated) = {GATE['p_opus_resolves_given_chain_failed_not_escalated']:.2f}; TPR {GATE['gate_TPR_for_opus_success']:.2f}, FPR {GATE['gate_FPR_for_opus_success']:.2f}. "
      f"{TRACE_CHECK['n_cheap_attempts_unrecorded']}/{TRACE_CHECK['n_attempts_total']} attempts are unrecorded runs billed at $0 (imputed-cost sensitivity: eta/eta_B "
      f"{R[(R.dataset == 'ood176') & (R.name == 'R3 ACRouter chain trace (imputed attempt costs)')].iloc[0].r[0]:.2f})",
      'PASS', 'Descriptive. The trace is a deterministic replay of the outcome matrix (verified 176/176), so its verifier is the ground-truth test outcome (TPR = 1, FPR = 0 by construction) '
      'and erroneous acceptance comes only from the give-up branch (chain exhausted, no escalation).')
fl_id = FLOOR['id_test']; fl_rb = FLOOR['routerbench']
claim('C4', 'IN-SAMPLE', 'R1 lookup/ex-ante routers versus R2 oracle cascades under the quality floor E[V] >= vbar - eps: which policies satisfy eps <= 0.01, and how the efficiency ranking changes under the floor',
      f"id_test: R1 {fl_id['R1']['eps_0.01']['satisfied']}/{fl_id['R1']['eps_0.01']['n']} satisfy eps=0.01 (eps=0.05: {fl_id['R1']['eps_0.05']['satisfied']}), R2 {fl_id['R2']['eps_0.01']['satisfied']}/{fl_id['R2']['eps_0.01']['n']} "
      f"(theta=1: Q >= 0 for all: {fl_id['R2']['theta1_all_Q_nonneg']}); OOD176: R1 {FLOOR['ood176']['R1']['eps_0.01']['satisfied']}/{FLOOR['ood176']['R1']['eps_0.01']['n']}, R2 {FLOOR['ood176']['R2']['eps_0.01']['satisfied']}/{FLOOR['ood176']['R2']['eps_0.01']['n']}, "
      f"R3 trace {int(r3['floor_0.01'])}; RouterBench: R1 {fl_rb['R1']['eps_0.01']['satisfied']}/{fl_rb['R1']['eps_0.01']['n']}, R2 {fl_rb['R2']['eps_0.01']['satisfied']}/{fl_rb['R2']['eps_0.01']['n']}. "
      f"Floor-feasible bests (eta/eta_B): id_test {fbs('id_test')}; OOD176 {fbs('ood176')}; RouterBench {fbs('routerbench')}",
      'PASS', 'In-sample classification with CI-decided flags per policy (see the floor table). At theta = 1 an oracle hit has Y_P = 1 >= Y_B, so Q >= 0 is automatic for R2; R1 routers trade quality for cost and can violate the floor while winning on eta.')
claim('C5', 'NOT-INSTANTIABLE', 'X7 live parts: R2 with real verifiers (LLM judge, sandbox in-prompt tests) and R3 on id_test (ACRouter release mode + escalation variant), and the operating-point prediction of the R3 - R2 sign from measured (kappa_ver, TPR, FPR, cbar_h)',
      'Not measurable offline: ACRouter ID mode dispatches once per task (no within-task escalation in the released id_decisions.jsonl), the released routers ship no verifier scores, and the only R3 trace (OOD176) uses the ground-truth test outcome as its verifier',
      'NOT-INSTANTIABLE', 'Pending X7-b (collection A/B). The offline counterfactual "ACRouter map + oracle verify-then-escalate" is reported as what the live id_test variant would reach with a perfect, free verifier '
      '(the verifier-side bound: e = 0 at theta = 1, no verification cost); it is not an efficiency bound, see deviation 1 and the per-decision optimal-escalation bound.')

# ================================================================================================ 5. audit: independent recomputation of three headline numbers (plain python, csv/json modules)
log('audit')
raw = {}
with open(f'{DATA}/coderouterbench/id_results_long.csv') as fh:
    for row in csv.DictReader(fh):
        if row['split'] == 'id_test':
            raw[(row['task_id'], row['model'])] = (float(row['score']), float(row['cost_usd']))
sumV = sumD = sumYB = sumCB = 0.0
with open(ID_DECISIONS) as fh:
    for line in fh:
        d = json.loads(line); s, c_ = raw[(d['task_id'], d['chosen_model'])]; sb_, cb_ = raw[(d['task_id'], BASE)]
        sumV += s; sumD += c_; sumYB += sb_; sumCB += cb_
audit_acr = (sumV / sumD) / (sumYB / sumCB)
rawo = {}
with open(f'{DATA}/coderouterbench/ood176_results_long.csv') as fh:
    for row in csv.DictReader(fh):
        rawo[(row['task_id'], row['model'])] = (float(row['resolved']), float(row['cost_usd']))
sumV = sumD = sumYB = sumCB = 0.0
with open(OOD_TRACE) as fh:
    for line in fh:
        d = json.loads(line)
        sumV += float(d['resolved']); sumD += sum(rawo[(d['task_id'], m)][1] for m in d['chain_run'])
        sumYB += rawo[(d['task_id'], BASE)][0]; sumCB += rawo[(d['task_id'], BASE)][1]
audit_r3 = (sumV / sumD) / (sumYB / sumCB)
mx = 'mistralai/mixtral-8x7b-chat'
sumV = sumD = sumYB = sumCB = 0.0
for i in te_idx:
    s, c_ = float(rb[mx].iloc[i]), float(rb[mx + '|total_cost'].iloc[i]); sb_, cb_ = float(rb[REF].iloc[i]), float(rb[REF + '|total_cost'].iloc[i])
    hit = s >= 1 - 1e-12
    sumV += s if hit else sb_; sumD += c_ + (0 if hit else cb_); sumYB += sb_; sumCB += cb_
audit_rb = (sumV / sumD) / (sumYB / sumCB)
AUDIT = dict(
    acrouter_id_map_r_independent=audit_acr, acrouter_id_map_r_pipeline=float(POL[('id_test', 'R1 ACRouter ID map (lookup)')]['r'][0]),
    r3_ood_trace_r_independent=audit_r3, r3_ood_trace_r_pipeline=float(POL[('ood176', 'R3 ACRouter chain trace (gate k=2)')]['r'][0]),
    rb_mixtral_theta1_r_independent=audit_rb, rb_mixtral_theta1_r_pipeline=float(POL[('routerbench', f'R2 {mx} theta=1.0')]['r'][0]),
    acrouter_release_metrics_reproduced=dict(AvgPerf=float(100 * np.mean([raw[(json.loads(l)['task_id'], json.loads(l)['chosen_model'])][0] for l in open(ID_DECISIONS)])),
                                             paper_AvgPerf=50.14, paper_total_usd=22.31, paper_perf_per_usd=2.25),
    trace_check=TRACE_CHECK)
AUDIT['max_abs_rel_diff'] = float(max(abs(AUDIT['acrouter_id_map_r_independent'] / AUDIT['acrouter_id_map_r_pipeline'] - 1),
                                      abs(AUDIT['r3_ood_trace_r_independent'] / AUDIT['r3_ood_trace_r_pipeline'] - 1),
                                      abs(AUDIT['rb_mixtral_theta1_r_independent'] / AUDIT['rb_mixtral_theta1_r_pipeline'] - 1)))
assert AUDIT['max_abs_rel_diff'] < 1e-9

# ================================================================================================ 6. figures
log('figures')


def errs(t):
    return np.array([[t[0] - t[1]], [t[2] - t[0]]])


# Fig 1 (C1): identity residuals
fig, ax = plt.subplots(figsize=(3.25, 1.9), layout='constrained')
for i, ds in enumerate(['id_test', 'ood176', 'routerbench']):
    d = R[R.dataset == ds]
    for rg in ['R1', 'R2', 'R3', 'R3-cf', 'R1-mem', 'const', 'ref', 'bound']:
        v = d[d.regime == rg].ident_rel_max.to_numpy()
        if len(v):
            ax.plot(i + rng.uniform(-.18, .18, len(v)), np.maximum(v, 1e-18), MARK[rg], color=COL[rg], ms=3, mfc='none' if rg in ('const', 'bound') else COL[rg], mew=.6, ls='')
    vs = d.ident_sec_rel_max.dropna().to_numpy()
    if len(vs):
        ax.plot(i + 0.42 + rng.uniform(-.06, .06, len(vs)), np.maximum(vs, 1e-18), '_', color=INK2, ms=4, mew=.6, ls='')
ax.set_yscale('log'); ax.set_ylim(1e-18, 1e-9); ax.axhline(1e-9, color=HAIR, lw=.6)
ax.set_xticks([0, 0.42, 1, 2]); ax.set_xticklabels(['id_test\n($)', 'id_test\n(s)', 'OOD176\n($)', 'RouterBench\n($)'], fontsize=7)
ax.set_ylabel('max relative residual of eq:surplus', fontsize=7)
hs = [plt.Line2D([], [], marker=MARK[k], color=COL[k], ls='', ms=3.5, label=k) for k in ['R1', 'R2', 'R3']] + [plt.Line2D([], [], marker='o', color=COL['const'], mfc='none', ls='', ms=3.5, label='constants / bound')]
ax.legend(handles=hs, loc='upper left', ncol=2, fontsize=6, handletextpad=.3, columnspacing=.8)
fig.savefig(os.path.join(FIG, 'fig1_identity_residuals.pdf')); plt.close(fig)

# Fig 2 (C2): Pareto (E[D], E[V]) with the reference ray of slope eta_B
fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.35), layout='constrained')
PANEL = [('id_test', 1000, r'$\mathbb{E}[D]$ (m\$ per task), CodeRouterBench id_test', 'CodeRouterBench id_test'),
         ('ood176', 1, r'$\mathbb{E}[D]$ (\$ per task), OOD176', 'OOD176'), ('routerbench', 1000, r'$\mathbb{E}[D]$ (m\$ per task), RouterBench held-out 30%', 'RouterBench')]
for ax, (ds, scale, xl, _) in zip(axes, PANEL):
    d = R[(R.dataset == ds) & (R.main | R.regime.isin(['const', 'bound']))]
    d = d[~((d.regime == 'R2') & (d.get('v_c', 0).fillna(0) > 0))]
    ref = d[d.regime == 'ref'].iloc[0]
    etaB = ref.EV[0] / ref.ED[0]
    xmax = max(d.ED_pt.max() * scale, ref.ED[0] * scale) * 1.08
    ax.plot([0, xmax], [0, etaB * xmax / scale], color=HAIR, lw=.8, zorder=0)
    ax.axhline(ref.EV[0], color=HAIR, lw=.6, ls=(0, (1, 2)), zorder=0)
    for rg, z in [('const', 1), ('bound', 1), ('R1-mem', 2), ('R1', 3), ('R2', 3), ('R3', 4), ('R3-cf', 4), ('ref', 5)]:
        dd = d[d.regime == rg]
        if rg == 'R3-cf' and ds == 'id_test':
            dd = dd[dd.router == 'acrouter_id_map']
        if rg == 'R1' and ds == 'id_test':
            dd = dd[dd.k == 0.0]
        if len(dd) == 0:
            continue
        ax.plot(dd.ED_pt * scale, dd.EV_pt, MARK[rg], color=COL[rg], ms=4.2 if rg in ('R3', 'R3-cf') else 3.8, mfc='none' if rg in ('const', 'bound') else COL[rg], mew=.7, ls='', zorder=z)
    # sparse direct labels: reference and the best of each regime, fixed offsets per regime so labels do not collide
    OFF = {'ref': (-3, -8, 'right'), 'R1': (3, 4, 'left'), 'R2': (3, -8, 'left'), 'R3': (3, 4, 'left'), 'R3-cf': (-3, 5, 'right')}
    lab = [('ref', ref['name'], ref['ED'][0], ref['EV'][0])]
    for rg in ['R1', 'R2', 'R3', 'R3-cf']:
        dd = d[(d.regime == rg) & (d.name.isin(BEST[ds].values()))]
        for _, x in dd.iterrows():
            lab.append((rg, x['name'], x['ED'][0], x['EV'][0]))
    for rg, nm, xx, yy in lab:
        dx, dy, ha = OFF[rg]
        ax.annotate(figlab(nm), (xx * scale, yy), xytext=(dx, dy), textcoords='offset points', fontsize=5.5, color=INK2, ha=ha)
    ax.set_xlim(0, xmax); ax.set_xlabel(xl, fontsize=7.5); ax.set_ylabel(r'$\mathbb{E}[V]$ (mean score / resolved)')
    ax.tick_params(labelsize=7)
hs = [plt.Line2D([], [], marker=MARK[k], color=COL[k], ls='', ms=4, label=l) for k, l in [('ref', 'reference (always strongest)'), ('R1', 'R1 ex-ante'), ('R2', 'R2 verify-then-escalate (oracle)'), ('R3', 'R3 in-loop / counterfactual'), ('R1-mem', 'R1 online (with memory)')]]
hs += [plt.Line2D([], [], marker='o', color=COL['const'], mfc='none', ls='', ms=4, label='always-m constants'), plt.Line2D([], [], marker='x', color=COL['bound'], ls='', ms=4, label='oracle bounds (need $Y_B$)'), plt.Line2D([], [], color=HAIR, lw=.8, label=r'reference ray $V=\eta_B D$')]
fig.legend(handles=hs, loc='outside lower center', ncol=4, fontsize=6.5, handletextpad=.3, columnspacing=1.0)
fig.savefig(os.path.join(FIG, 'fig2_pareto.pdf')); plt.close(fig)

# Fig 3 (C2/C4): eta/eta_B with CIs, marker filled iff the floor eps = 0.01 holds
fig, axes = plt.subplots(1, 3, figsize=(6.75, 4.0), layout='constrained', gridspec_kw={'width_ratios': [1.05, 1, 1]})
for ax, (ds, _, _, ttl) in zip(axes, PANEL):
    d = R[(R.dataset == ds) & R.main & R.regime.isin(['R1', 'R2', 'R3', 'R3-cf', 'R1-mem'])]
    d = d[~((d.regime == 'R2') & (d.get('v_c', 0).fillna(0) > 0))]
    if ds == 'id_test':
        d = d[~((d.regime == 'R1') & (d.k != 0.0)) & ~((d.regime == 'R3-cf') & (d.router != 'acrouter_id_map'))]
    d = d.sort_values('r_pt')
    for i, (_, x) in enumerate(d.iterrows()):
        ok = bool(x['floor_0.01'])
        ax.errorbar(x['r'][0], i, xerr=errs(x['r']), fmt=MARK[x['regime']], color=COL[x['regime']], ms=3.8, mfc=COL[x['regime']] if ok else 'white', mew=.7, capsize=1.2, lw=.7)
    ax.axvline(1, color=HAIR, lw=.8)
    ax.set_yticks(range(len(d))); ax.set_yticklabels([short(n) for n in d.name], fontsize=5.8)
    ax.set_xlabel(r'$\eta/\eta_B$, ' + ttl + (' (log)' if ds != 'id_test' else ''), fontsize=7.5)
    if ds != 'id_test':
        ax.set_xscale('log')
    ax.tick_params(axis='x', labelsize=7)
hs = [plt.Line2D([], [], marker='o', color=INK2, ls='', ms=4, label=r'floor $\mathbb{E}[V]\geq\bar v-0.01$ holds'), plt.Line2D([], [], marker='o', color=INK2, mfc='white', ls='', ms=4, label='floor violated')]
hs += [plt.Line2D([], [], marker=MARK[k], color=COL[k], ls='', ms=4, label=k) for k in ['R1', 'R1-mem', 'R2', 'R3']]
fig.legend(handles=hs, loc='outside lower center', ncol=6, fontsize=6.5, handletextpad=.3, columnspacing=1.0)
fig.savefig(os.path.join(FIG, 'fig3_ranking_floor.pdf')); plt.close(fig)

# Fig 4 (C3 + decomposition): the four terms per policy, dollars per task
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.9), layout='constrained', gridspec_kw={'width_ratios': [1.25, 1]})
SEL_ID =['R1 ACRouter ID map (lookup)', 'R1 routellm_mf', 'R1 finetuned_router_qwen35_9b_v3', 'R1 dimension_best', 'R2 Qwen3-Max theta=1.0', 'R2 Qwen3-Max theta=0.5',
          'R2 kimi-k2.5 theta=1.0', 'R2 gpt-5.4 theta=1.0', 'R3-cf ACRouter map + oracle theta=1.0 -> Opus']
SEL_OOD = ['R1 LogReg', 'R1-mem kNN_Retrieval', 'R2 MiniMax-M2.7 + oracle -> Opus', 'R2 gpt-5.4 + oracle -> Opus', 'R3 variant chain[4] -> Opus always', 'R3 ACRouter chain trace (gate k=2)']
for ax, (ds, sel, scale, yl) in zip(axes, [('id_test', SEL_ID, 1000, r'm\$ per task (id_test)'), ('ood176', SEL_OOD, 1, r'\$ per task (OOD176)')]):
    d = R[(R.dataset == ds)].set_index('name').loc[sel]
    xs = np.arange(len(sel)); w = 0.2
    ax.bar(xs - 1.5 * w, d.avoided.apply(lambda t: t[0]) * scale, w, color=TERM_COL['avoided'], label=r'avoided reference work $\mathbb{E}[HC_B]$')
    ax.bar(xs - 0.5 * w, -d.attempt.apply(lambda t: t[0]) * scale, w, color=TERM_COL['attempt'], label=r'attempt cost $-\mathbb{E}[A]$')
    ax.bar(xs + 0.5 * w, d.Qhit_usd.apply(lambda t: t[0]) * scale, w, color=TERM_COL['quality'], label=r'accepted quality change $\mathbb{E}[H(Y_P-Y_B)]/\eta_B$')
    ax.bar(xs + 1.5 * w, -d.recov.apply(lambda t: t[0]) * scale, w, color=TERM_COL['recovery'], label=r'recovery cost $-\mathbb{E}[(1-H)(C_F-C_B)]$')
    net = d.net_usd.apply(lambda t: t[0]) * scale
    ax.plot(xs, net, '_', color=INK, ms=11, mew=1.3, ls='', label=r'net gain $Q/\eta_B+\mathcal{S}$')
    for x_, t in zip(xs, d.net_usd):
        ax.plot([x_, x_], [t[1] * scale, t[2] * scale], color=INK, lw=.7)
    ax.axhline(0, color=HAIR, lw=.6)
    ax.set_xticks(xs); ax.set_xticklabels([short(n) for n in sel], rotation=35, ha='right', fontsize=6)
    ax.set_ylabel(yl)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc='outside upper center', ncol=3, fontsize=6, handlelength=1.2, columnspacing=1.2)
fig.savefig(os.path.join(FIG, 'fig4_decomposition.pdf')); plt.close(fig)

# Fig 5 (C3): the chain variants -- efficiency and quality as attempts and the gate are added
order = ['R2 MiniMax-M2.7 + oracle -> Opus', 'R3 variant chain[2] -> Opus always', 'R3 variant chain[3] -> Opus always', 'R3 variant chain[4] -> Opus always',
         'R3 variant chain[4] gate k=1', 'R3 ACRouter chain trace (gate k=2)', 'R3 variant chain[4] gate k=3', 'R3 variant chain[4] gate k=4', 'R3 variant chain[4] never escalate']
labels = ['MiniMax+oracle' + r'$\to$' + 'Opus', '+Kimi', '+GPT-5.4', '+GLM-5 (always esc.)', 'gate k=1', 'gate k=2 (trace)', 'gate k=3', 'gate k=4', 'never escalate']
d = R[R.dataset == 'ood176'].set_index('name').loc[order]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), layout='constrained')
for ax, key, yl in [(axes[0], 'r', r'$\eta/\eta_B$ (OOD176)'), (axes[1], 'EV', r'$\mathbb{E}[V]$ (resolved rate)'), (axes[2], 'ED', r'$\mathbb{E}[D]$ (\$ per task)')]:
    xs = np.arange(len(order))
    for i, (nm, t) in enumerate(zip(order, d[key])):
        c = COL['R3'] if nm == 'R3 ACRouter chain trace (gate k=2)' else (COL['R2'] if i == 0 else INK2)
        ax.errorbar(i, t[0], yerr=errs(t), fmt='D' if c == COL['R3'] else ('^' if i == 0 else 'o'), color=c, ms=3.8, capsize=1.2, lw=.7)
    if key == 'r':
        ax.axhline(1, color=HAIR, lw=.8)
    if key == 'EV':
        ax.axhline(yB_o.mean(), color=HAIR, lw=.8)
        ax.annotate('always-Opus', (8.5, yB_o.mean()), xytext=(0, -7), textcoords='offset points', fontsize=6, color=INK2, ha='right')
    if key == 'ED':
        ax.axhline(cB_o.mean(), color=HAIR, lw=.8)
        ax.annotate('always-Opus', (8.5, cB_o.mean()), xytext=(0, -7), textcoords='offset points', fontsize=6, color=INK2, ha='right')
    ax.axvspan(3.5, 8.5, color='#f0efec', zorder=0)
    ax.text(3.65, 0.99, 'escalation-gate variants', transform=ax.get_xaxis_transform(), fontsize=5.5, color=INK2, va='top')
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=5.8, rotation=30, ha='right')
    ax.set_ylabel(yl, fontsize=8); ax.tick_params(axis='y', labelsize=7)
fig.savefig(os.path.join(FIG, 'fig5_ood_chain_variants.pdf')); plt.close(fig)
log('figures written')

# ================================================================================================ 7. results.json
def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_, bool)): return bool(o)
    if isinstance(o, np.ndarray): return jsonable(o.tolist())
    if isinstance(o, pd.Series): return jsonable(o.to_dict())
    return o


RUNTIME = round(time.time() - T0, 1)
RESULTS = dict(experiment='X7a_regimes_offline', seed=SEED, n_bootstrap=B, runtime_s=RUNTIME, data_checks=DATA_CHECKS, claims=CL,
               identity=IDENT, best_per_regime=BEST, best_per_regime_floor_001=BEST_FLOOR, ranking=RANK, ranking_floor_001=RANK_FLOOR, paired=PAIRED,
               floor=FLOOR, noisy_verifier_focus=NOISY_SUM, ood_trace_check=TRACE_CHECK, ood_gate=GATE, ood_subsets=SUBSETS, ood_verification_seconds=VERIF,
               ood_chain_variants=CHAIN_VARIANTS, r1_shares_id_test=R1_SHARES, r1_shares_ood176=OOD_R1_SHARES, routerbench_r1=RB_R1, audit=AUDIT,
               policies=[{k: v for k, v in r.items()} for r in REC])
json.dump(jsonable(RESULTS), open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
log('results.json written')

# ================================================================================================ 8. README.md + summary_for_paper.md
def table(hdr, rows):
    return '| ' + ' | '.join(hdr) + ' |\n|' + '---|' * len(hdr) + '\n' + '\n'.join('| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows) + '\n'


def f2(t): return f'{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]'
def fm(t, s=1000, nd=2): return f'{s*t[0]:.{nd}f} [{s*t[1]:.{nd}f}, {s*t[2]:.{nd}f}]'
def fl(x): return ' '.join(('Y' if x[f'floor_{e}'] else 'n') + ('*' if x[f'floor_{e}_ci'] != 'undecided' else '') for e in EPS)


def pol_table(ds, names, scale=1000, nd=2, sec=False):
    rows = []
    for nm in names:
        x = R[(R.dataset == ds) & (R.name == nm)].iloc[0]
        row = [nm, f'{x.EV[0]:.4f}', fm(x.ED, scale, 3), f2(x.r), fm(x.avoided, scale, nd), fm(x.attempt, scale, nd), fm(x.Qhit_usd, scale, nd), f'{x.recov[0]*scale:.0e}', fm(x.net_usd, scale, nd),
               f'{x.Q[0]:+.4f}', f'{x.e[0]:.3f}', fl(x), f'{x.ident_rel_max:.0e}']
        if sec:
            row.insert(4, f2(x.r_sec) if isinstance(x.r_sec, list) else '-')
        rows.append(row)
    hdr = ['policy', 'E[V]', f'E[D] ({"m$" if scale == 1000 else "$"}/task)', 'eta/eta_B', 'avoided E[HC_B]', 'attempt E[A]', 'Q_hit/eta_B', 'recovery', 'net gain ($)', 'Q', 'e', 'floor eps=0/.01/.02/.05', 'ident. resid.']
    if sec:
        hdr.insert(4, 'eta/eta_B (seconds)')
    return table(hdr, rows)


id_main_R1 = [n for n in R[(R.dataset == 'id_test') & (R.regime == 'R1') & R.main].name]
id_R1_k = [n for n in R[(R.dataset == 'id_test') & (R.regime == 'R1') & ~R.main].name]
id_main_R2 = [n for n in R[(R.dataset == 'id_test') & (R.regime == 'R2') & R.main].name]
id_R3cf = [n for n in R[(R.dataset == 'id_test') & (R.regime == 'R3-cf') & (R.router == 'acrouter_id_map')].name] + [BOUND_ACR]
id_R3cf_routers = [n for n in R[(R.dataset == 'id_test') & (R.regime == 'R3-cf') & (R.router != 'acrouter_id_map') & (R.theta == 1.0)].name]
id_const = [n for n in R[(R.dataset == 'id_test') & R.regime.isin(['ref', 'const', 'bound'])].name]
ood_main = [n for n in R[(R.dataset == 'ood176') & R.main & R.regime.isin(['R3', 'R2', 'R1', 'R1-mem'])].name]
ood_sens = [n for n in R[(R.dataset == 'ood176') & ~R.main & R.regime.isin(['R3'])].name]
ood_const = [n for n in R[(R.dataset == 'ood176') & R.regime.isin(['ref', 'const', 'bound'])].name]
rb_main = [n for n in R[(R.dataset == 'routerbench') & R.main & R.regime.isin(['R1', 'R3-cf', 'R2'])].name] + [n for n in R[(R.dataset == 'routerbench') & (R.regime == 'R1') & ~R.main].name] + [BOUND_RB]
rb_const = [n for n in R[(R.dataset == 'routerbench') & R.regime.isin(['ref', 'const', 'bound'])].name]
pr = PAIRED


def ptab(ds, keys):
    rows = []
    for k in keys:
        p = pr[ds].get(k)
        if p is None or 'a' not in p:
            continue
        rows.append([k, p['a'], p['b'], f3(p['d_r'], 3) + (' **' if p['d_r_decided'] else ''), f"{1000*p['d_net_usd'][0] if ds != 'ood176' else p['d_net_usd'][0]:+.3f} [{(1000 if ds != 'ood176' else 1)*p['d_net_usd'][1]:+.3f}, {(1000 if ds != 'ood176' else 1)*p['d_net_usd'][2]:+.3f}]" + (' **' if p['d_net_usd_decided'] else ''),
                     f3(p['d_EV'], 4) + (' **' if p['d_EV_decided'] else '')])
    return table(['comparison', 'A', 'B', 'Δ eta/eta_B (A−B)', f'Δ net gain ({"m$" if ds != "ood176" else "$"}/task)', 'Δ E[V]'], rows)


L_ = []
L_.append(f"""# X7a_regimes_offline -- three router regimes on one task stream under one accounting (offline part of X7)

Generated by `run.py` (numpy / pandas / scipy / scikit-learn / matplotlib; seed {SEED}; B = {B} paired task-resampling bootstrap draws, 95% percentile CIs; runtime {RUNTIME} s; no API calls).
Files: `run.py`, `results.json` (every number below, per policy, with CIs), `README.md`, `summary_for_paper.md`, `figures/fig1_identity_residuals.pdf` (C1), `figures/fig2_pareto.pdf` (C2),
`figures/fig3_ranking_floor.pdf` (C2/C4), `figures/fig4_decomposition.pdf` (C3, decomposition), `figures/fig5_ood_chain_variants.pdf` (C3).

Paper objects: Lemma thm:surplus (eq:surplus-def, eq:surplus, eq:unified-identity), eq:accounting / eq:cascade-surplus (four terms: avoided reference work E[H C_B], attempt cost E[A],
accepted quality change E[H(Y_P − Y_B)], recovery cost E[(1−H)(C_F − C_B)]), the quality floor Q ≥ −ε, the erroneous-acceptance rate e = P(H=1, Y_P<Y_B), Table tab:regimes
(ex-ante / verify-then-escalate / in-loop agentic), Appendix app:instances (ACRouter in our notation: one model per task, verifier written to memory, escalation chain = case K>1 of prop:attempts).

## Setup
**Data (READ ONLY).** CodeRouterBench `id_results_long.csv` ({DATA_CHECKS['id_n_tasks']} tasks × 8 models; id_test {DATA_CHECKS['id_n_test']} tasks; `cost_usd` = token-log pricing, `latency_ms`);
cross-check always-Opus id_test ${DATA_CHECKS['always_opus_id_test_total_usd']:.2f} (paper $34.02), mean score {DATA_CHECKS['always_opus_id_test_mean_score']:.4f} (paper 43.83); η_B = {eta_B_id:.2f} score/$ .
OOD176 `ood176_results_long.csv` ({DATA_CHECKS['ood_n_tasks']} agentic tasks × 8 models, `resolved`, `apply_ok`, `cost_usd`; old112 {DATA_CHECKS['ood_n_old112']} + new64 {DATA_CHECKS['ood_n_new64']}); always-Opus resolved {DATA_CHECKS['always_opus_ood176_resolved']:.3f}, ${DATA_CHECKS['always_opus_ood176_total_usd']:.2f} total, η_B = {eta_B_ood:.3f} resolved/$ .
RouterBench 0-shot pickle ({DATA_CHECKS['rb_n']} prompts × 11 models, {DATA_CHECKS['rb_eval_names']} eval families, {100*DATA_CHECKS['rb_score_binary_share']:.1f}% of scores binary); reference `{REF}`; stratified (by eval_name) 70/30 split, seed {SEED}: train {DATA_CHECKS['rb_n_train']}, held-out test {DATA_CHECKS['rb_n_test']}; η_B(test) = {eta_B_rb:.1f} score/$ .
Decision files: 9 released router decisions `$ACR/id/voter_decisions/*.jsonl` (5 files carry 332 extra dim10 OOD rows, dropped), ACRouter's released ID policy `outputs/current/id_decisions.jsonl` (2,919 rows; `hierarchical_group_model_map`, a per-task lookup keyed by benchmark/dimension, tuned on train+val: **treated as R1 with a memory-derived map**),
ACRouter's OOD176 escalation traces `outputs/acrouter_ood176/ood_decisions.jsonl`, the released OOD176 baseline decisions `outputs/baselines_ood176/decisions/*.jsonl` (LogReg, RouteLLM-MF/BERT, TF-IDF+MLP, Qwen3.5-0.8B fine-tuned; online kNN / LinTS / LinUCB), and the OOD112 sandbox trace `outputs/current/ood_sandbox_decisions.jsonl` (per-attempt verifier wall-clock).

**Reference policies.** always-Opus (id_test, OOD176), always-{REF} (RouterBench): Y_B, C_B observed for every task, so Q and S are paired sample means, not estimates.

**Accounting (every policy).** V = H Y_P + (1−H) Y_F, D = A + (1−H) C_F. R1 ex-ante: H = 1{{chose a non-reference model}}, Y_P = score of the chosen model, A = k + H·cost(chosen), fallback = the reference (chosen directly).
R2 cascade: H = 1{{score_primary ≥ θ}} (oracle), A = cost_primary + v_c, fallback Opus / gpt-4 on a miss (Y_F = Y_B, C_F = C_B, so recovery cost ≡ 0 and miss quality change ≡ 0); noisy verifier: H replaced by its exact expectation over independent flips,
E[H] = 1{{hit}}(1−β_fr) + 1{{miss}}β_fa (deterministic, no simulation). R3 chain (OOD176): attempts in the released order, stop at the first `resolved`; if none resolved and ≥ k cheap attempts produced `apply_ok`, escalate to Opus; H = 1{{Opus not run}}, Y_P = resolved of the last cheap attempt, A = Σ cheap attempt costs (+ v_c per attempt).
Counterfactual R3 on id_test (R3-cf): the router's cheap choice is the primary of an oracle verify-then-escalate cascade to Opus; Opus choices are dispatched directly. Router cost k: 0 for lookup / logreg / tfidf / RouteLLM / ACRouter map; the fine-tuned Qwen routers ship **no token field**, so k ∈ {{0, prompt-token proxy ({DATA_CHECKS['router_prompt_token_proxy_mean_tokens']:.0f} tokens × $0.054/M = ${DATA_CHECKS['router_prompt_token_proxy_mean_usd']:.1e}/task), $0.001, $0.005}} per task.
Verification cost v_c ∈ {{{', '.join('$%g' % v for v in VC_ID)}}} per verification (id_test), {{{', '.join('$%g' % v for v in VC_RB)}}} (RouterBench), {{{', '.join('$%g' % v for v in VC_OOD)}}} per attempt (OOD176).
Statistics: η = ΣV/ΣD (ratio of sums); η/η_B, net gain Q + η_B S (reported in $ per task as Q/η_B + S) with its four terms, quality floor E[V] ≥ v̄ − ε for ε ∈ {{0, 0.01, 0.02, 0.05}} (point; `*` = the bootstrap CI decides it), e = P(H=1, Y_P<Y_B).
Cost basis: dollars (csv `cost_usd`) everywhere; seconds (`latency_ms`, sequential steps add) for id_test only (OOD176 and RouterBench have no latency). Two policies are always compared on the same bootstrap weights.

## Claims
""")
L_.append(table(['id', 'kind', 'statement', 'verdict'], [[c['id'], c['kind'], c['statement'], f"**{c['verdict']}**"] for c in CL]))
for c in CL:
    L_.append(f"\n**{c['id']} [{c['kind']}] -- {c['verdict']}.** {c['statement']}\n- result: {c['result']}\n- note: {c['note']}\n")

L_.append("\n## Results\n### R1 ex-ante routers on id_test (dollars; m$ = 10^-3 USD per task; k = 0)\n")
L_.append(pol_table('id_test', id_main_R1, sec=True))
L_.append("\nRouter-cost sensitivity for the fine-tuned Qwen routers (no token field released; k per task added to every task's D):\n")
L_.append(table(['policy', 'eta/eta_B', 'E[D] (m$)', 'net gain (m$)', 'floor eps=0/.01/.02/.05'], [[n, f2(x.r), fm(x.ED, 1000, 3), fm(x.net_usd), fl(x)] for n in id_R1_k for x in [R[(R.dataset == 'id_test') & (R.name == n)].iloc[0]]]))
L_.append("\n### R2 verify-then-escalate cascades on id_test (oracle verifier, v_c = 0)\n")
L_.append(pol_table('id_test', id_main_R2, sec=True))
L_.append("\nNoisy verifiers and verification cost, focus primaries (exact expectation over the verifier's flips; β_fa = false-accept, β_fr = false-reject rate):\n")
L_.append(table(['primary', 'theta', 'beta_fa', 'beta_fr', 'v_c ($)', 'eta/eta_B', 'E[V]', 'E[D] (m$)', 'Q', 'e', 'floor eps=0', 'floor eps=0.01'],
                [[x['primary'], x['theta'], x['beta_fa'], x['beta_fr'], x['v_c'], f2(x['r']), f"{x['EV'][0]:.4f}", f"{1000*x['ED'][0]:.3f}", f"{x['Q'][0]:+.4f}", f"{x['e'][0]:.3f}", 'Y' if x['floor_0'] else 'n', 'Y' if x['floor_001'] else 'n'] for x in NOISY_SUM]))
L_.append("\n### R3 on id_test: not observable offline -- counterfactual with a perfect, free verifier on ACRouter's decisions (map + oracle verify-then-escalate → Opus), and the exact per-decision escalation bound\n")
L_.append(pol_table('id_test', id_R3cf, sec=True))
_acr, _a1, _a5, _ab = pol('id_test', 'R1 ACRouter ID map (lookup)'), pol('id_test', 'R3-cf ACRouter map + oracle theta=1.0 -> Opus'), pol('id_test', 'R3-cf ACRouter map + oracle theta=0.5 -> Opus'), pol('id_test', BOUND_ACR)
L_.append(f"""
What this bounds. The θ = 1 counterfactual is the live escalation variant with a verifier that never errs and costs nothing (e = 0, every accepted output fully correct); any real verifier at the same threshold adds false rejects (strictly worse in V and D), false accepts (enter e and Q) and v_c.
It is **not** an efficiency upper bound: the lookup alone reaches η/η_B {_acr.r[0]:.2f} [{_acr.r[1]:.2f}, {_acr.r[2]:.2f}] while the verified counterfactual reaches {_a1.r[0]:.2f} (θ = 1) / {_a5.r[0]:.2f} (θ = 0.5), because the map's cheap choices already beat Opus on average (E[V] {_acr.EV[0]:.3f} vs {so.mean():.3f}) and escalating their misses to Opus adds E[(1−H)C_B] = {1000*(_a1.ED[0]-_acr.ED[0]):.2f} m$/task for a quality change of {_a1.Q[0]-_acr.Q[0]:+.4f}.
The exact bound over all escalation rules on these decisions (escalate a cheap choice iff Y_P − Y_B + η_B C_B < 0; it needs Y_B, i.e. an oracle about the reference, not a verifier) reaches η/η_B {_ab.r[0]:.2f} [{_ab.r[1]:.2f}, {_ab.r[2]:.2f}] with E[V] {_ab.EV[0]:.3f}: escalation is worth it on only {100*(1-_ab.p[0]/_acr.p[0]):.1f}% of the cheap-branch tasks at price η_B.
""")
L_.append("\nThe same counterfactual for the 9 released routers (θ = 1):\n")
L_.append(pol_table('id_test', id_R3cf_routers))
L_.append("\nConstants and bound (id_test):\n")
L_.append(pol_table('id_test', id_const))
L_.append("\n### OOD176: R3 released trace, R2 cascades, R1 released baselines (dollars per task)\n")
L_.append(pol_table('ood176', ood_main, scale=1, nd=3))
L_.append("\nR3 sensitivities (reported cost, imputed attempt costs for the {0} unrecorded cheap attempts, sandbox verification priced per attempt) and chain variants:\n".format(TRACE_CHECK['n_cheap_attempts_unrecorded']))
L_.append(pol_table('ood176', ood_sens, scale=1, nd=3))
L_.append("\nConstants and bound (OOD176):\n")
L_.append(pol_table('ood176', ood_const, scale=1, nd=3))
L_.append(f"""
**Trace check.** Replaying the outcome matrix with ACRouter's rule (chain {OOD_CHAIN}, stop at first `resolved`, escalate to Opus iff ≥ 2 cheap attempts had `apply_ok`) reproduces the released trace exactly:
chain_run identical {TRACE_CHECK['chain_run_identical']}, resolved identical {TRACE_CHECK['resolved_identical']}, escalated identical {TRACE_CHECK['escalated_identical']}, max |Σ matrix cost − reported cost_usd| = {TRACE_CHECK['cost_max_abs_diff_usd']:.1e} $; release metrics reproduced: AvgPerf {TRACE_CHECK['metrics_reproduced']['AvgPerf']:.2f} (paper 73.30), $Total {TRACE_CHECK['metrics_reproduced']['total_usd']:.2f} (86.72), Perf/$ {TRACE_CHECK['metrics_reproduced']['perf_per_usd']:.2f} (0.85).
Chain lengths {TRACE_CHECK['n_steps']}; {TRACE_CHECK['n_escalated']} escalations. **The trace's verifier is the recorded test outcome (`resolved`) itself**, so within the chain TPR = 1, FPR = 0; the only "verifier" with error is the Opus-escalation gate.
**Gate as a signal.** Among the {GATE['n_chain_failed']} chain-failed tasks, Opus resolves {GATE['p_opus_resolves_given_escalated']:.2f} of the {GATE['n_escalated']} escalated ones but {GATE['p_opus_resolves_given_chain_failed_not_escalated']:.2f} of the {GATE['n_chain_failed_not_escalated']} not escalated (TPR {GATE['gate_TPR_for_opus_success']:.2f}, FPR {GATE['gate_FPR_for_opus_success']:.2f} for "Opus will resolve"): the gate escalates on the tasks Opus is *least* likely to fix; {GATE['n_chain_failed_opus_would_resolve']} chain-failed tasks Opus would have resolved.
**Unrecorded attempts.** {TRACE_CHECK['n_cheap_attempts_unrecorded']} of {TRACE_CHECK['n_attempts_total']} attempted calls hit matrix cells with no recorded run (cost 0, resolved 0, apply_ok 0; {TRACE_CHECK['n_tasks_with_unrecorded_attempt']} tasks) and are billed $0 in the release; the imputed-cost row above prices them at the model's mean recorded old112 cost.
**new64** costs are constant per model in the released matrix ({DATA_CHECKS['ood_new64_costs_constant_per_model']}: imputed by the data provider) and four models' new64 rows come from different source models; old112 has real per-task costs. Subset results:
""")
rows = []
for sub_name, sub in SUBSETS.items():
    for k, v in sub.items():
        rows.append([sub_name, v['n'], k, f2(v['r']), f"{v['EV'][0]:.3f}", f"{v['ED'][0]:.3f}", f"{v['Q'][0]:+.3f}", f"{v['e'][0]:.3f}"])
L_.append(table(['subset', 'n', 'policy', 'eta/eta_B', 'E[V]', 'E[D] ($)', 'Q', 'e'], rows))
if VERIF.get('available'):
    L_.append(f"""
**Verification seconds (OOD112 sandbox trace, {VERIF['n_tasks']} old112 tasks, {VERIF['n_attempts']} attempts; chain identical to the OOD176 trace on old112: {VERIF['chain_identical_to_ood176_trace_on_old112']}).** Verifier verdict = matrix `resolved` in {100*VERIF['verdict_equals_matrix_resolved']:.0f}% of attempts.
Sandbox wall-clock per attempt: mean {VERIF['elapsed_s_mean_per_attempt']:.1f} s, median {VERIF['elapsed_s_median_per_attempt']:.1f} s; {100*VERIF['share_attempts_elapsed_zero']:.0f}% of attempts take 0 s ({100*VERIF['share_elapsed_zero_with_patch_error']:.0f}% of those carry a patch error: {VERIF['errors']}), i.e. the verifier short-circuits on empty / non-applying patches;
attempts with a recorded run average {VERIF['elapsed_s_when_run_recorded']:.1f} s. Per task: mean {VERIF['per_task_total_s']['mean']:.1f} s, median {VERIF['per_task_total_s']['median']:.1f} s, max {VERIF['per_task_total_s']['max']:.0f} s of verification. No dollar price for the sandbox is released; the v_c rows above price it at $0.01 / $0.05 per attempt.
""")
L_.append("\n### RouterBench held-out 30% (reference gpt-4-1106-preview; m$ per task)\n")
L_.append(pol_table('routerbench', rb_main))
L_.append("\nR2 noisy verifier / v_c grid on RouterBench (four primaries) is in `results.json` (`policies`, dataset routerbench, main = false). Constants and bound:\n")
L_.append(pol_table('routerbench', rb_const))
L_.append("\nLR router: TF-IDF (1–2-grams, 30k features) of the prompt, one logistic regression per model on 1{{score >= 0.5}} (train split), cost prior = train mean cost per (eval_name, model); decision argmax_m p̂_m − λ ĉ_m with λ in multiples of η_B(train) = {:.1f}; test AUROC per model (tfidf): {}.\n".format(
    eta_B_rb_train, ', '.join(f"{short('always-'+m)} {DATA_CHECKS['rb_auroc_tfidf'][m]:.2f}" for m in RBM)))
L_.append(table(['router', 'lambda', 'share routed to reference', 'top choices'], [[k, f"{v['lam']:.0f}", f"{v['share_ref']:.3f}", ', '.join(f'{short("always-"+m)} {n}' for m, n in sorted(v['share'].items(), key=lambda kv: -kv[1])[:4])] for k, v in RB_R1.items()]))

L_.append("\n### Paired comparisons (same bootstrap weights; ** = 95% CI excludes 0)\n")
for ds, keys in [('id_test', ['R1_vs_R2', 'R1_vs_R3', 'R2_vs_R3', 'R3cf_acr_theta1_vs_R1_acr', 'R3cf_acr_theta1_vs_best_R2'] + [k for k in pr['id_test'] if k.startswith('floor_')]),
                 ('ood176', ['R1_vs_R1mem', 'R1_vs_R2', 'R1_vs_R3', 'R1mem_vs_R2', 'R1mem_vs_R3', 'R2_vs_R3', 'R3_vs_R2_MiniMax', 'R3_vs_R2_gpt54', 'R3_vs_chain4_always_escalate', 'R3_vs_chain4_never_escalate'] + [k for k in pr['ood176'] if k.startswith('floor_')]),
                 ('routerbench', ['R1_vs_R2', 'R1_vs_R3', 'R2_vs_R3'] + [k for k in pr['routerbench'] if k.startswith('floor_')])]:
    L_.append(f"\n{ds} (best per regime by point η/η_B: {BEST[ds]}; floor-feasible (ε ≤ 0.01) bests: {BEST_FLOOR[ds]}):\n")
    L_.append(ptab(ds, keys))

L_.append("\n### Quality-floor satisfaction by regime (main policies; point estimate / CI-decided yes / CI-decided no)\n")
rows = []
for ds, fl_ in FLOOR.items():
    for rg, v in fl_.items():
        rows.append([ds, rg, v['eps_0.0']['n']] + [f"{v[f'eps_{e}']['satisfied']} / {v[f'eps_{e}']['decided_yes']} / {v[f'eps_{e}']['decided_no']}" for e in EPS])
L_.append(table(['dataset', 'regime', 'n policies', 'eps=0', 'eps=0.01', 'eps=0.02', 'eps=0.05'], rows))

L_.append(f"""
## Hypothesis violations measured
* **Quality floor vs efficiency.** R1 routers reach the highest η/η_B on every dataset by routing most tasks to cheap models. On id_test this is a Pareto improvement, not a violation: all {n_viol('id_test','R1')[1]} R1 policies have Q > 0 (min Q = {R[(R.dataset=='id_test')&(R.regime=='R1')&R.main].Q_pt.min():+.3f}; the released routers pick models that beat Opus on average, and Opus's own id_test mean score is only {so.mean():.3f}). On OOD176 and RouterBench it is a violation: {n_viol('ood176','R1')[0]}/{n_viol('ood176','R1')[1]} OOD176 R1 policies ({n_viol('ood176','R1-mem')[0]}/{n_viol('ood176','R1-mem')[1]} online ones) and {n_viol('routerbench','R1')[0]}/{n_viol('routerbench','R1')[1]} RouterBench LR routers breach ε = 0.01 (RouteLLM-MF routes 174/176 OOD tasks to Qwen3-Max, Q = {pol('ood176','R1 RouteLLM_MF').Q[0]:+.3f}; LR λ = 4η_B has Q = {pol('routerbench','R1 LR router (tfidf, lambda=4 eta_B)').Q[0]:+.3f}). The ratio objective alone accepts these lossy routers (the paper's scope paragraph); only the floor rules them out.
* **Erroneous acceptance.** Oracle cascades at θ = 1 have e = 0 by construction; at θ = 0.5 e is {pol('id_test','R2 Qwen3-Max theta=0.5').e[0]:.3f} (Qwen3-Max) and Q stays positive. In the noisy grid (β_fa, β_fr ≤ 0.1) Q also stays positive (min {min(x['Q'][0] for x in NOISY_SUM):+.3f}) while e rises to {max(x['e'][0] for x in NOISY_SUM):.3f}: at these error rates false accepts cost quality but do not breach ε = 0.01 because the primaries' wrong outputs are mostly on tasks Opus also fails. R1 routers commit outputs worse than Opus on {pol('id_test','R1 ACRouter ID map (lookup)').e[0]:.3f} (ACRouter map) to {max(R[(R.dataset=='id_test')&(R.regime=='R1')&R.main].e.apply(lambda t: t[0])):.3f} of id_test tasks, and on {min(R[(R.dataset=='ood176')&(R.regime=='R1')&R.main].e.apply(lambda t: t[0])):.3f}–{max(R[(R.dataset=='ood176')&(R.regime=='R1')&R.main].e.apply(lambda t: t[0])):.3f} of OOD176 tasks.
* **Verification cost.** v_c = $0.001 per verification (≈ {100*0.001/co.mean():.0f}% of the reference's mean cost) lowers η/η_B of the Qwen3-Max θ = 1 cascade from {R[(R.dataset=='id_test')&(R.name=='R2 Qwen3-Max theta=1.0')].iloc[0].r[0]:.2f} to {R[(R.dataset=='id_test')&(R.name=='R2 Qwen3-Max theta=1.0 (v_c=$0.001)')].iloc[0].r[0]:.2f}; at v_c = $0.01 (≈ the reference's own cost) every id_test cascade falls below the reference.
* **R3-OOD's gate is anti-informative** about Opus success (TPR {GATE['gate_TPR_for_opus_success']:.2f} < FPR {GATE['gate_FPR_for_opus_success']:.2f}); unconditional escalation after the chain raises E[V] from {r3.EV[0]:.3f} to {v_always.EV[0]:.3f} at η/η_B {v_always.r[0]:.2f} (vs {r3.r[0]:.2f}), never escalating gives {v_never.r[0]:.2f} at E[V] {v_never.EV[0]:.3f}.
* **Free failed attempts.** {TRACE_CHECK['n_cheap_attempts_unrecorded']}/{TRACE_CHECK['n_attempts_total']} R3-OOD attempts are unrecorded runs billed at $0; imputing their cost moves η/η_B from {r3.r[0]:.2f} to {R[(R.dataset=='ood176')&(R.name=='R3 ACRouter chain trace (imputed attempt costs)')].iloc[0].r[0]:.2f}.

## Deviations from the X7 protocol (offline part)
1. **R3 on id_test is not observable offline**: ACRouter's released ID policy dispatches once per task (no within-task verification or escalation is recorded; the verifier score u_i is not in the release). Reported instead: the counterfactual "ACRouter map + oracle verify-then-escalate → Opus" (θ = 1 and 0.5, with v_c sensitivity), i.e. the live escalation variant with a perfect, free verifier at the same thresholds. This bounds the live variant on the verifier side only (no false accepts at θ = 1, e = 0, no verification cost); it is **not** an efficiency upper bound — the lookup alone is more efficient (η/η_B {pol('id_test','R1 ACRouter ID map (lookup)').r[0]:.2f} vs {pol('id_test','R3-cf ACRouter map + oracle theta=1.0 -> Opus').r[0]:.2f} / {pol('id_test','R3-cf ACRouter map + oracle theta=0.5 -> Opus').r[0]:.2f}) because the map's cheap choices already beat Opus on average. The exact per-decision bound (escalate iff Y_P − Y_B + η_B C_B < 0, which needs Y_B) is {pol('id_test', BOUND_ACR).r[0]:.2f} [{pol('id_test', BOUND_ACR).r[1]:.2f}, {pol('id_test', BOUND_ACR).r[2]:.2f}]. The live R3 (release mode + escalation variant, 3 orders × 3 seeds) is X7-b, pending.
2. **The only R3 evidence offline is the OOD176 trace**, and it is a deterministic replay of the released outcome matrix (verified 176/176; the ACRouter repository's `ood_repro.verify_and_escalate` is the rule). Its verifier is the recorded sandbox test outcome, i.e. an oracle within the chain; what it adds over an R2 cascade is (i) further attempts after a failed one and (ii) an `apply_ok`-count gate for the Opus escalation. The OOD112 sandbox trace supplies real verification seconds but no verifier errors.
3. **R2 with real verifiers (LLM judge, in-prompt sandbox tests) is deferred** to the live part; here the verifier is the oracle score threshold plus the (β_fa, β_fr) noise model with verification priced at v_c. Consequently the protocol's "R3 − R2 sign predicted from measured (κ_ver, TPR, FPR, c̄_h)" is NOT-INSTANTIABLE offline (C5).
4. **Router cost of the fine-tuned Qwen routers**: no token field is released, so the mandated $0.054/M charge is replaced by the k-grid {{0, prompt-token proxy, $0.001, $0.005}}.
5. **RouterBench R1** is our own TF-IDF logistic-regression router (70/30 split), not a released policy; the pickle's `oracle_model_to_route_to` is the per-task oracle and is not used as a router.
6. **OOD176 data caveats** inherited from the release: new64 costs are imputed constants per model (and four models' new64 rows come from other source models); 45% of old112 cheap-model cells have no recorded run. The released trace is scored as released (primary), with imputed-cost, old112 and new64 sensitivities.
7. Seconds basis only for id_test (no latency in OOD176 / RouterBench). Bootstrap CIs of the noisy-verifier rows cover task resampling only (the verifier's flip variance is integrated out analytically).

## Audit (independent recomputation with plain `csv`/`json` loops, no pandas pivot, no bootstrap matrix)
* ACRouter ID map η/η_B: independent {AUDIT['acrouter_id_map_r_independent']:.6f} vs pipeline {AUDIT['acrouter_id_map_r_pipeline']:.6f}; release metrics reproduced from the decision file: AvgPerf {AUDIT['acrouter_release_metrics_reproduced']['AvgPerf']:.2f} (release 50.14), $Total {R[(R.dataset=='id_test')&(R.name=='R1 ACRouter ID map (lookup)')].iloc[0].ED[0]*len(idt):.2f} (22.31).
* R3-OOD trace η/η_B: independent {AUDIT['r3_ood_trace_r_independent']:.6f} vs pipeline {AUDIT['r3_ood_trace_r_pipeline']:.6f}.
* RouterBench R2 mixtral-8x7b θ = 1 cascade η/η_B: independent {AUDIT['rb_mixtral_theta1_r_independent']:.6f} vs pipeline {AUDIT['rb_mixtral_theta1_r_pipeline']:.6f}.
* Max relative difference {AUDIT['max_abs_rel_diff']:.1e}. X1 cross-check (same data, seed 20260911): X1's ex-ante id_test η/η_B for finetuned_router_qwen35_08b 4.21, logreg 6.03, routellm_mf 6.32 and Qwen3-Max θ = 1 cascade 1.68 vs this run {R[(R.dataset=='id_test')&(R.name=='R1 finetuned_router_qwen35_08b')].iloc[0].r[0]:.2f}, {R[(R.dataset=='id_test')&(R.name=='R1 logreg')].iloc[0].r[0]:.2f}, {R[(R.dataset=='id_test')&(R.name=='R1 routellm_mf')].iloc[0].r[0]:.2f}, {R[(R.dataset=='id_test')&(R.name=='R2 Qwen3-Max theta=1.0')].iloc[0].r[0]:.2f}.

## Limitations
* All rankings are in-sample on the evaluation stream (best-per-regime selected on the same tasks); the RouterBench router is out-of-sample for its decisions but its ranking is still descriptive.
* The oracle verifier is the benchmark's own score/test outcome; real verifiers (X7-b, X8) will have TPR < 1 and FPR > 0 and a cost, which the (β, v_c) grid brackets but does not measure.
* OOD176 has 176 tasks (29 escalations); CIs are wide and the split-level caveats above (imputed new64 costs, unrecorded old112 runs, source-model substitutions) limit the R3 evidence to descriptive statements.
* 42 Qwen3-Max id_test rows have no token record (cost 0, score 0) and are kept as published (X1 sensitivity: < 2% effect).
* R1 policies pay no routing cost except where stated; the online OOD baselines (kNN, LinTS, LinUCB) depend on the task order of the released replay.
* Values are benchmark scores in [0, 1] with a meaningful zero; the dollar valuation of Q uses η_B of the same sample.
""")
open(os.path.join(OUT, 'README.md'), 'w').write(''.join(L_))

# summary for the paper
b_id = {k: R[(R.dataset == 'id_test') & (R.name == v)].iloc[0] for k, v in BEST['id_test'].items()}
b_oo = {k: R[(R.dataset == 'ood176') & (R.name == v)].iloc[0] for k, v in BEST['ood176'].items()}
b_rb = {k: R[(R.dataset == 'routerbench') & (R.name == v)].iloc[0] for k, v in BEST['routerbench'].items()}
acr = R[(R.dataset == 'id_test') & (R.name == 'R1 ACRouter ID map (lookup)')].iloc[0]
acr3 = R[(R.dataset == 'id_test') & (R.name == 'R3-cf ACRouter map + oracle theta=1.0 -> Opus')].iloc[0]
qm1 = R[(R.dataset == 'id_test') & (R.name == 'R2 Qwen3-Max theta=1.0')].iloc[0]
fb_id = {k: (R[(R.dataset == 'id_test') & (R.name == v)].iloc[0] if v else None) for k, v in BEST_FLOOR['id_test'].items()}
SUMMARY = f"""# X7-a summary for the paper (offline; seed {SEED}; B = {B}; 95% paired bootstrap CIs; dollars per task)
1. [IDENTITY, C1 PASS] eq:surplus / eq:unified-identity and the four-term decomposition hold for all {IDENT['n_policies']} policies (R1/R2/R3, three datasets, dollars and seconds): max relative residual {max(IDENT['max_ident_rel_dollars'], IDENT['max_ident_rel_seconds'] or 0):.0e}.
2. [IN-SAMPLE, C2] Best policy per regime, ranked by η/η_B (paired CIs; all adjacent gaps CI-decided on all three datasets: {all(dec_all.values())}). id_test: {rank_str('id_test')}. OOD176: {rank_str('ood176')}. RouterBench: {rank_str('routerbench')}.
3. [IN-SAMPLE, C4] Under the quality floor ε ≤ 0.01 the ranking becomes — id_test: {rank_str('id_test', True)} (all 10 R1 and 14 R2 policies satisfy it; R1 is a Pareto improvement here); OOD176: {rank_str('ood176', True)} (0/5 R1 policies satisfy it); RouterBench: {rank_str('routerbench', True)} (1/5 LR routers satisfy it).
4. [IN-SAMPLE] ACRouter's released ID policy is a per-task lookup map: η/η_B {acr.r[0]:.2f} [{acr.r[1]:.2f}, {acr.r[2]:.2f}], E[V] {acr.EV[0]:.3f} vs Opus {so.mean():.3f} (Q = {acr.Q[0]:+.3f}, floor ε=0.01 {'holds' if acr['floor_0.01'] else 'violated'}). The live escalation variant with a perfect, free verifier at θ = 1 on the same decisions (verifier-side bound) would reach η/η_B {acr3.r[0]:.2f} [{acr3.r[1]:.2f}, {acr3.r[2]:.2f}], E[V] {acr3.EV[0]:.3f} (Q = {acr3.Q[0]:+.3f}, e = 0): below the lookup, because escalating the map's cheap misses to Opus adds cost for no quality; the exact per-decision escalation bound at price η_B is {pol('id_test', BOUND_ACR).r[0]:.2f} [{pol('id_test', BOUND_ACR).r[1]:.2f}, {pol('id_test', BOUND_ACR).r[2]:.2f}].
5. [IN-SAMPLE, C3] OOD176: the released ACRouter chain trace (R3) has η/η_B {b_oo['R3'].r[0]:.2f} [{b_oo['R3'].r[1]:.2f}, {b_oo['R3'].r[2]:.2f}], E[V] {b_oo['R3'].EV[0]:.3f} vs Opus {yB_o.mean():.3f}, E[D] ${b_oo['R3'].ED[0]:.3f} vs ${cB_o.mean():.3f}; net gain ${b_oo['R3'].net_usd[0]:.3f}/task = avoided Opus work ${b_oo['R3'].avoided[0]:.3f} − attempts ${b_oo['R3'].attempt[0]:.3f} + quality ${b_oo['R3'].Qhit_usd[0]:+.3f}; e = {b_oo['R3'].e[0]:.3f}.
6. [IN-SAMPLE, C2/C3] OOD176 paired: R3 − best R2 ({short(BEST['ood176']['R2'])}) Δη/η_B = {pr['ood176']['R2_vs_R3']['d_r'][0]*-1:+.2f} [{-pr['ood176']['R2_vs_R3']['d_r'][2]:+.2f}, {-pr['ood176']['R2_vs_R3']['d_r'][1]:+.2f}]; R3 − R2(MiniMax, same first primary) Δη/η_B = {pr['ood176']['R3_vs_R2_MiniMax']['d_r'][0]:+.2f} [{pr['ood176']['R3_vs_R2_MiniMax']['d_r'][1]:+.2f}, {pr['ood176']['R3_vs_R2_MiniMax']['d_r'][2]:+.2f}]; best R1 ({short(BEST['ood176']['R1'])}) η/η_B {b_oo['R1'].r[0]:.1f} but E[V] {b_oo['R1'].EV[0]:.3f} (floor violated by {-b_oo['R1'].Q[0]:.2f}).
7. [descriptive, C3] R3-OOD's verification is the ground-truth test outcome (TPR = 1, FPR = 0 within the chain; trace = exact matrix replay); its escalation gate is anti-informative (P(Opus resolves | escalated) {GATE['p_opus_resolves_given_escalated']:.2f} vs {GATE['p_opus_resolves_given_chain_failed_not_escalated']:.2f} if not escalated): unconditional escalation gives η/η_B {v_always.r[0]:.2f}, E[V] {v_always.EV[0]:.3f}; {TRACE_CHECK['n_cheap_attempts_unrecorded']}/{TRACE_CHECK['n_attempts_total']} attempts are $0 unrecorded runs (imputed: η/η_B {R[(R.dataset=='ood176')&(R.name=='R3 ACRouter chain trace (imputed attempt costs)')].iloc[0].r[0]:.2f}).
8. [IN-SAMPLE, C2/C4] RouterBench (held-out 30%, ref gpt-4-1106-preview): LR router R1 {b_rb['R1'].r[0]:.2f} [{b_rb['R1'].r[1]:.2f}, {b_rb['R1'].r[2]:.2f}] ({short(BEST['routerbench']['R1'])}, Q = {b_rb['R1'].Q[0]:+.3f}, floor violated), best R2 {b_rb['R2'].r[0]:.2f} ({short(BEST['routerbench']['R2'])}, Q = {b_rb['R2'].Q[0]:+.3f}), LR + oracle cascade {b_rb['R3'].r[0]:.2f} (Q = {b_rb['R3'].Q[0]:+.3f}); the only floor-feasible LR router (λ = 0) has η/η_B {pol('routerbench', 'R1 LR router (tfidf, lambda=0 eta_B)').r[0]:.2f}.
9. [IN-SAMPLE] Verification priced at v_c = $0.001 (≈ {100*0.001/co.mean():.0f}% of Opus's mean cost) cuts Qwen3-Max θ=1 from η/η_B {qm1.r[0]:.2f} to {R[(R.dataset=='id_test')&(R.name=='R2 Qwen3-Max theta=1.0 (v_c=$0.001)')].iloc[0].r[0]:.2f}; β_fa = 0.1 false accepts push Q to {min(x['Q'][0] for x in NOISY_SUM if x['beta_fa']==0.1 and x['v_c']==0):+.3f} (e up to {max(x['e'][0] for x in NOISY_SUM if x['beta_fa']==0.1):.3f}).
10. [NOT-INSTANTIABLE offline, C5] R3 on id_test, R2 with real verifiers, and the operating-point sign prediction of R3 − R2 are pending X7-b (live collection); the OOD176 trace is the only R3 evidence and its verifier is oracle-quality.
"""
open(os.path.join(OUT, 'summary_for_paper.md'), 'w').write(SUMMARY)
log('README + summary written; total runtime', round(time.time() - T0, 1), 's')
for c in CL:
    print(f"{c['id']:4s} {c['kind']:17s} {c['verdict']:17s} {c['statement'][:120]}")
print('best per regime:', json.dumps(BEST, indent=None))
print('floor-feasible bests:', json.dumps(BEST_FLOOR, indent=None))

