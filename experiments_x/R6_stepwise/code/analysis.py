"""R6 step-wise routing analysis (replay of recorded continuations; FLOPs and batch-1 seconds bases).

Reference B = always-L (large model).  Tasks: HumanEval+ (163) and MBPP+ (378), scored by the EvalPlus plus tests.
Fit on a source-stratified half of the tasks ('probing'), evaluate on the other half ('test'); both folds are reported.
usage: python analysis.py --pair A|B
"""
import argparse, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ap = argparse.ArgumentParser(); ap.add_argument('--pair', required=True); ap.add_argument('--nboot', type=int, default=1000)
a = ap.parse_args()
PAIR = a.pair; S, L = PAIRS[PAIR]; D = f'{RUNS}/pair{PAIR}'
T0 = time.time()
def log(*x): print(f'[{time.time()-T0:6.1f}s]', *x, flush=True)

# ------------------------------------------------------------------ model arithmetic
ARCH = {'Qwen3-0.6B': dict(L=28, qd=2048, ne=0.440e9, head=0.156e9), 'Qwen3-1.7B': dict(L=28, qd=2048, ne=1.409e9, head=0.311e9),
        'Qwen3-4B': dict(L=36, qd=4096, ne=3.633e9, head=0.389e9), 'Qwen3-14B': dict(L=40, qd=5120, ne=13.212e9, head=0.778e9)}
LAT = {m: json.load(open(f'{RUNS}/latency_{m}.json')) for m in (S, L)}

def flops(m, a_, b_, decode):
    """FLOPs to process positions [a_, b_) with model m: 2*N_nonemb per token, attention 4*layers*qdim*pos,
    and the LM head on decoded tokens (prefill computes logits for its last token only)."""
    A = ARCH[m]; n = b_ - a_
    if n <= 0: return 0.0
    f = 2 * A['ne'] * n + 2 * A['L'] * A['qd'] * (b_ ** 2 - a_ ** 2)
    f += 2 * A['head'] * (n if decode else 1)
    return f

def secs(m, n, decode):
    return (LAT[m]['decode_s_per_token'] if decode else LAT[m]['prefill_s_per_token']) * max(0, n)

class Cost:
    """Accumulate (FLOPs, seconds) of prefill / decode segments."""
    def __init__(self): self.f = 0.0; self.s = 0.0
    def pre(self, m, a_, b_): self.f += flops(m, a_, b_, False); self.s += secs(m, b_ - a_, False); return self
    def dec(self, m, a_, b_): self.f += flops(m, a_, b_, True); self.s += secs(m, b_ - a_, True); return self
    def proj(self, n_tok, params): self.f += 2.0 * params * n_tok; return self

# ------------------------------------------------------------------ load data
TASKS = load_tasks(); TID = [t['task_id'] for t in TASKS]; SRC = np.array([t['source'] for t in TASKS]); N = len(TASKS)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_path(S))
P0 = np.array([len(tok(chat_prefix(tok, t['user']), add_special_tokens=False)['input_ids']) for t in TASKS])

def load(name):
    p = f'{D}/out_{name}.jsonl'
    return {r['job_id']: r for r in read_jsonl(p)} if os.path.exists(p) else {}
def scores(name):
    p = f'{D}/score_{name}.jsonl'
    return {r['job_id']: float(r['passed']) for r in read_jsonl(p)} if os.path.exists(p) else {}
OUT = {}; SC = {}
for nm in ['A_S', 'A_L', 'B_L', 'B_S', 'C_S', 'c2c', 'live']:
    OUT.update(load(nm)); SC.update(scores(nm))
log('rows', len(OUT), 'scores', len(SC))
def gen_len(j): r = OUT[j]; return len(r['full_ids']) - r['n_prefix']

sroot = [OUT[f'{PAIR}|S|{t}|root'] for t in TID]; lroot = [OUT[f'{PAIR}|L|{t}|root'] for t in TID]
YS = np.array([SC[f'{PAIR}|S|{t}|root'] for t in TID]); YL = np.array([SC[f'{PAIR}|L|{t}|root'] for t in TID])
lines = [[st['n_tok'] for st in r['steps']] for r in sroot]; cum = [np.concatenate([[0], np.cumsum(x)]) for x in lines]
conf = [[st['lp_min'] for st in r['steps']] for r in sroot]
llines = [[st['n_tok'] for st in r['steps']] for r in lroot]; lcum = [np.concatenate([[0], np.cumsum(x)]) for x in llines]
HAS_C2C = any('|C2C|' in j for j in OUT)
C2C_PARAMS = next((r['proj_params'] for j, r in OUT.items() if '|C2C|' in j), 0)

# ------------------------------------------------------------------ split
FOLD = split_folds(TASKS)

# ------------------------------------------------------------------ per-task realizations of each action
def c_sroot(i): return Cost().pre(S, 0, P0[i]).dec(S, P0[i], P0[i] + cum[i][-1])
def c_lroot(i): return Cost().pre(L, 0, P0[i]).dec(L, P0[i], P0[i] + lcum[i][-1])

def handoff(i, k, s_lines):
    """S wrote s_lines lines (>= k; the last s_lines-k are discarded), L continues from S prefix k."""
    t = TID[i]; c = Cost().pre(S, 0, P0[i]).dec(S, P0[i], P0[i] + cum[i][s_lines]) if s_lines > 0 else Cost()
    if k == 0:
        j = f'{PAIR}|L|{t}|root'; c.pre(L, 0, P0[i]).dec(L, P0[i], P0[i] + gen_len(j))
    else:
        j = f'{PAIR}|L|{t}|fromS{k}'; e = P0[i] + cum[i][k]; c.pre(L, 0, e).dec(L, e, e + gen_len(j))
    return SC[j], c

def intervene1(i, k, s_lines):
    """S wrote s_lines lines, the last s_lines-k discarded; L writes ONE line from S prefix k; S resumes."""
    t = TID[i]; c = Cost().pre(S, 0, P0[i]).dec(S, P0[i], P0[i] + cum[i][s_lines]) if s_lines > 0 else Cost()
    if k == 0:
        m1 = llines[i][0]; c.pre(L, 0, P0[i]).dec(L, P0[i], P0[i] + m1)
        if len(llines[i]) == 1: return YL[i], c
        j = f'{PAIR}|S|{t}|fromL1'; e = P0[i] + m1
        c.pre(S, (P0[i] if s_lines > 0 else 0), e).dec(S, e, e + gen_len(j)); return SC[j], c
    j1 = f'{PAIR}|L1|{t}|fromS{k}'; r1 = OUT[j1]; m1 = gen_len(j1); e0 = P0[i] + cum[i][k]
    c.pre(L, 0, e0).dec(L, e0, e0 + m1)
    if r1['reason'] != 'step_limit': return SC[j1], c
    j = f'{PAIR}|S|{t}|fromS{k}L1'; e = e0 + m1
    c.pre(S, e0, e).dec(S, e, e + gen_len(j)); return SC[j], c

def lplan(i, jn):
    """L writes the first jn lines, S continues."""
    t = TID[i]
    if jn >= len(llines[i]): return YL[i], c_lroot(i)
    j = f'{PAIR}|S|{t}|fromL{jn}'; e = P0[i] + lcum[i][jn]
    c = Cost().pre(L, 0, P0[i]).dec(L, P0[i], e).pre(S, 0, e).dec(S, e, e + gen_len(j)); return SC[j], c

def c2c(i, k, s_lines):
    """S wrote s_lines lines (last s_lines-k discarded); L reads prompt + S prefix k, the fuser projects L's cache into S's,
    S re-reads the context (as implemented) and continues."""
    t = TID[i]; j = f'{PAIR}|C2C|{t}|k{k}'
    if j not in OUT or j not in SC: return None
    c = Cost().pre(S, 0, P0[i]).dec(S, P0[i], P0[i] + cum[i][s_lines]) if s_lines > 0 else Cost().pre(S, 0, P0[i])
    e = P0[i] + cum[i][k]
    # L must read the whole context (it has not seen it); the fuser runs over every context token; S's own cache of the
    # context already exists (the released implementation recomputes it, which we do not charge); S then decodes.
    c.pre(L, 0, e).proj(e, C2C_PARAMS).dec(S, e, e + gen_len(j)); return SC[j], c

def first_trigger(i, tau):
    """First S-root line q (1-based) with lp_min < tau, or None."""
    for q, v in enumerate(conf[i], 1):
        if v < tau: return q
    return None

# ------------------------------------------------------------------ policy families (per task (V, FLOPs, seconds))
def realize(fn):
    V = np.zeros(N); F = np.zeros(N); Sx = np.zeros(N); ok = np.ones(N, bool)
    for i in range(N):
        out = fn(i)
        if out is None: ok[i] = False; continue
        v, c = out; V[i] = v; F[i] = c.f; Sx[i] = c.s
    return dict(V=V, F=F, S=Sx, ok=ok)

lp_prob = np.concatenate([conf[i] for i in range(N) if FOLD[i] == 0])
TAUS = tau_grid(PAIR)
POL = {}
POL['always_S'] = realize(lambda i: (YS[i], c_sroot(i)))
POL['always_L (reference)'] = realize(lambda i: (YL[i], c_lroot(i)))
def casc_oracle(i):
    c = c_sroot(i)
    if YS[i] >= 1: return YS[i], c
    cl = c_lroot(i); c.f += cl.f; c.s += cl.s; return YL[i], c
POL['cascade S->L, oracle acceptance'] = realize(casc_oracle)
def casc_conf(tau):
    def f(i):
        c = c_sroot(i)
        if min(conf[i]) >= tau: return YS[i], c
        cl = c_lroot(i); c.f += cl.f; c.s += cl.s; return YL[i], c
    return f
FAM = {'task_cascade_conf': {}, 'handoff_conf': {}, 'handoff_conf_keep': {}, 'intervene1_conf': {}, 'handoff_fixed': {}, 'lplan_fixed': {},
       'c2c_conf': {}, 'c2c_fixed': {}}
for tau in TAUS:
    FAM['task_cascade_conf'][tau] = realize(casc_conf(tau))
    def hc(i, tau=tau):
        q = first_trigger(i, tau)
        return (YS[i], c_sroot(i)) if q is None else handoff(i, q - 1, q)
    FAM['handoff_conf'][tau] = realize(hc)
    def hk(i, tau=tau):
        q = first_trigger(i, tau)
        if q is None or q >= len(conf[i]): return YS[i], c_sroot(i)
        return handoff(i, q, q)
    FAM['handoff_conf_keep'][tau] = realize(hk)
    def iv(i, tau=tau):
        q = first_trigger(i, tau)
        return (YS[i], c_sroot(i)) if q is None else intervene1(i, q - 1, q)
    FAM['intervene1_conf'][tau] = realize(iv)
    if HAS_C2C:
        def cc(i, tau=tau):
            q = first_trigger(i, tau)
            return (YS[i], c_sroot(i)) if q is None else c2c(i, q - 1, q)
        FAM['c2c_conf'][tau] = realize(cc)
FAM['prompt_routing'] = {}
for he_m in ('S', 'L'):
    for mb_m in ('S', 'L'):
        FAM['prompt_routing'][f'humaneval+:{he_m},mbpp+:{mb_m}'] = realize(
            lambda i, he_m=he_m, mb_m=mb_m: (YS[i], c_sroot(i)) if (he_m if SRC[i] == 'humaneval+' else mb_m) == 'S' else (YL[i], c_lroot(i)))
for k in [1, 2, 3, 4, 6, 8]:
    FAM['handoff_fixed'][k] = realize(lambda i, k=k: (YS[i], c_sroot(i)) if k >= len(lines[i]) else handoff(i, k, k))
    FAM['lplan_fixed'][k] = realize(lambda i, k=k: lplan(i, k))
    if HAS_C2C:
        FAM['c2c_fixed'][k] = realize(lambda i, k=k: (YS[i], c_sroot(i)) if k >= len(lines[i]) else c2c(i, k, k))
if HAS_C2C:
    POL['C2C root (S with L-fused prompt cache)'] = realize(lambda i: c2c(i, 0, 0))
    def plain_root(i):
        j = f'{PAIR}|C2C|{TID[i]}|k0'; r = [v for kk, v in OUT.items() if False]
        return None
    PLAIN = {r['job_id']: r for r in read_jsonl(f'{D}/out_c2cplain_k0.jsonl')} if os.path.exists(f'{D}/out_c2cplain_k0.jsonl') else {}
    PSC = {r['job_id']: float(r['passed']) for r in read_jsonl(f'{D}/score_c2cplain_k0.jsonl')} if os.path.exists(f'{D}/score_c2cplain_k0.jsonl') else {}
    if PLAIN and PSC:
        def pr(i):
            j = f'{PAIR}|C2C|{TID[i]}|k0'
            if j not in PLAIN or j not in PSC: return None
            n = PLAIN[j]['n_gen']; return PSC[j], Cost().pre(S, 0, P0[i]).dec(S, P0[i], P0[i] + n)
        POL['S root, HF implementation (C2C control, no fusion)'] = realize(pr)
HAS_LIVE = any('|live|' in j for j in OUT)
if HAS_LIVE:
    FAM['intervene_live'] = {}
    for tau in TAUS:
        def lv(i, tau=tau):
            j = f'{PAIR}|live|{tau:.6g}|{TID[i]}'
            if j not in OUT or j not in SC: return None
            c = Cost()
            for m, kind, a_, b_ in OUT[j]['segments']:
                (c.pre if kind == 'pre' else c.dec)(m, a_, b_)
            return SC[j], c
        FAM['intervene_live'][tau] = realize(lv)
if any('|C2Cplain|' in j or j.endswith('|k0') and OUT[j].get('model') == 'c2c-plain' for j in OUT):
    pass
FAM = {k: v for k, v in FAM.items() if v}
log('policies realized; taus', [round(t, 3) for t in TAUS])

# ------------------------------------------------------------------ evaluation helpers
def boot_w(idx, B, seed):
    r = np.random.default_rng(seed); n = len(idx)
    return np.vstack([np.ones((1, n)), r.multinomial(n, np.full(n, 1 / n), size=B).astype(float)])
def ci(v):
    v = np.asarray(v, float); w = v[1:][np.isfinite(v[1:])]
    return [float(v[0]), float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))]
REF = 'always_L (reference)'

def stats(p, idx, W, basis):
    X = p['F'] if basis == 'flops' else p['S']
    R = POL[REF]; RX = R['F'] if basis == 'flops' else R['S']
    M = W @ np.column_stack([p['V'][idx], X[idx], R['V'][idx], RX[idx]]); n = W.sum(1)
    EV, ED, vb, cb = M[:, 0] / n, M[:, 1] / n, M[:, 2] / n, M[:, 3] / n
    return dict(EV=EV, ED=ED, vbar=vb, cbar=cb, r=(EV / ED) / (vb / cb), dV=EV - vb, cost_ratio=ED / cb)

def eta(p, idx, basis):
    X = p['F'] if basis == 'flops' else p['S']
    return p['V'][idx].sum() / X[idx].sum(), p['V'][idx].mean()

def select(fam, idx, basis, eps=None):
    """Dinkelbach-style selection within a finite policy family on the fitting tasks: max eta, subject to E[V] >= vbar_L - eps."""
    vbar = POL[REF]['V'][idx].mean(); best = None
    for key, p in fam.items():
        if not p['ok'][idx].all(): continue
        e, v = eta(p, idx, basis)
        if eps is not None and v < vbar - eps - 1e-12: continue
        if best is None or e > best[1]: best = (key, e)
    return best

RES = dict(pair=PAIR, small=S, large=L, n_tasks=N, n_by_source={s: int((SRC == s).sum()) for s in set(SRC)},
           taus=TAUS, tau_quantiles=QS, latency={m: {k: LAT[m][k] for k in ('c0_s', 'prefill_s_per_token', 'decode_s_per_token', 'r2')} for m in LAT},
           c2c_projector_params=C2C_PARAMS, folds={})
RES['whole_sample'] = {nm: dict(score=float(p['V'].mean()), flops_rel=float(p['F'].mean() / POL[REF]['F'].mean()),
                                sec_rel=float(p['S'].mean() / POL[REF]['S'].mean())) for nm, p in POL.items() if p['ok'].all()}
RES['whole_sample_families'] = {f: {str(k): dict(score=float(p['V'][p['ok']].mean()), flops_rel=float(p['F'][p['ok']].mean() / POL[REF]['F'][p['ok']].mean()),
                                                 sec_rel=float(p['S'][p['ok']].mean() / POL[REF]['S'][p['ok']].mean()), n_ok=int(p['ok'].sum()))
                                    for k, p in fam.items()} for f, fam in FAM.items()}
# fraction of S-root lines triggering at each tau, and the mean number of lines per task
RES['trigger_rate_line'] = {str(t): float(np.mean(lp_prob < t)) for t in TAUS}
RES['trigger_rate_task'] = {str(t): float(np.mean([first_trigger(i, t) is not None for i in range(N)])) for t in TAUS}

for fold in (0, 1):
    fit = np.where(FOLD == fold)[0]; test = np.where(FOLD != fold)[0]
    W = boot_w(test, a.nboot, SEED + fold)
    FR = {}
    for basis in ('flops', 'sec'):
        rows = {}
        base = {nm: stats(p, test, W, basis) for nm, p in POL.items() if p['ok'][test].all()}
        for nm, st in base.items():
            rows[nm] = dict(kind='fixed', r=ci(st['r']), score=ci(st['EV']), dV=ci(st['dV']), cost_ratio=ci(st['cost_ratio']))
        for fam_name, fam in FAM.items():
            for eps in (None, 0.0, 0.02, 0.05):
                sel = select(fam, fit, basis, eps)
                if sel is None:
                    rows[f'{fam_name} [eps={eps}]'] = dict(kind='selected', feasible=False); continue
                key = sel[0]; p = fam[key]
                if not p['ok'][test].all():
                    rows[f'{fam_name} [eps={eps}]'] = dict(kind='selected', param=key, note='missing test rows'); continue
                st = stats(p, test, W, basis)
                d_ref = st['dV']; comp = {}
                for bn in ('cascade S->L, oracle acceptance', 'always_S') + (('C2C root (S with L-fused prompt cache)',) if HAS_C2C else ()):
                    if bn in base:
                        comp[bn] = dict(d_r=ci(st['r'] - base[bn]['r']), d_dV=ci(st['dV'] - base[bn]['dV']),
                                        d_cost_ratio=ci(st['cost_ratio'] - base[bn]['cost_ratio']))
                rows[f'{fam_name} [eps={eps}]'] = dict(kind='selected', param=key, fit_eta_rel=float(sel[1] / eta(POL[REF], fit, basis)[0]),
                                                      fit_score=float(p['V'][fit].mean()), r=ci(st['r']), score=ci(st['EV']), dV=ci(d_ref),
                                                      cost_ratio=ci(st['cost_ratio']), floor_rate=None if eps is None else float(np.mean(st['dV'][1:] >= -eps - 1e-12)),
                                                      vs=comp)
        # per-task oracle over the step-wise menu (in-sample ceiling on test): Dinkelbach over per-task actions
        menus = [POL['always_S'], POL[REF]] + [p for fam in FAM.values() for p in fam.values()] + [p for nm, p in POL.items() if nm.startswith('C2C')]
        Vm = np.column_stack([p['V'][test] for p in menus]); Xm = np.column_stack([(p['F'] if basis == 'flops' else p['S'])[test] for p in menus])
        okm = np.column_stack([p['ok'][test] for p in menus]); Vm = np.where(okm, Vm, -1e9)
        lam = eta(POL[REF], test, basis)[0]
        for _ in range(100):
            g = np.argmax(Vm - lam * Xm, axis=1); new = Vm[np.arange(len(test)), g].sum() / Xm[np.arange(len(test)), g].sum()
            if abs(new - lam) < 1e-12 * lam: break
            lam = new
        Vo = Vm[np.arange(len(test)), g]; Xo = Xm[np.arange(len(test)), g]
        orc = dict(V=np.zeros(N), F=np.zeros(N), S=np.zeros(N), ok=np.ones(N, bool))
        orc['V'][test] = Vo; orc['F' if basis == 'flops' else 'S'][test] = Xo
        other = 'S' if basis == 'flops' else 'F'
        orc[other][test] = np.column_stack([p[other][test] for p in menus])[np.arange(len(test)), g]
        st = stats(orc, test, W, basis)
        rows['per-task oracle over step-wise menu (ceiling)'] = dict(kind='oracle', r=ci(st['r']), score=ci(st['EV']), dV=ci(st['dV']), cost_ratio=ci(st['cost_ratio']))
        # unconstrained oracle restricted to task-level actions {S, L, cascade}
        FR[basis] = rows
    RES['folds'][f'fit_on_fold{fold}'] = FR
    log(f'fold {fold} done')

# ------------------------------------------------------------------ live multi-intervention vs replayed single intervention
if HAS_LIVE:
    LV = {}
    for tau in TAUS:
        js = [f'{PAIR}|live|{tau:.6g}|{t}' for t in TID]
        if not all(j in OUT and j in SC for j in js): continue
        n_live = np.array([OUT[j]['n_interventions'] for j in js]); pred = np.array([sum(v < tau for v in conf[i]) for i in range(N)])
        p1 = FAM['intervene1_conf'][tau]; pl = FAM['intervene_live'][tau]
        LV[str(tau)] = dict(pred_interventions_from_S_root=float(pred.mean()), live_interventions=float(n_live.mean()),
                            share_L_lines_live=float(np.mean([OUT[j]['authors'].count('L') / max(1, len(OUT[j]['authors'])) for j in js])),
                            score_live=float(pl['V'].mean()), score_one_intervention_replay=float(p1['V'].mean()), score_S=float(YS.mean()), score_L=float(YL.mean()),
                            flops_rel_live=float(pl['F'].mean() / POL[REF]['F'].mean()), flops_rel_one=float(p1['F'].mean() / POL[REF]['F'].mean()),
                            sec_rel_live=float(pl['S'].mean() / POL[REF]['S'].mean()), sec_rel_one=float(p1['S'].mean() / POL[REF]['S'].mean()))
    RES['live_vs_replay'] = LV

# ------------------------------------------------------------------ local comparison at S-root histories (eq:transfer-decision / X10-like signs)
def pos_bucket(k): return 0 if k == 0 else (1 if k <= 2 else (2 if k <= 5 else (3 if k <= 10 else 4)))
cq = np.percentile(lp_prob, [10, 30, 60])
def conf_bucket(v): return int(np.searchsorted(cq, v))
HIST = []   # (task i, k, state, per-action (V, F, S) for T = text handoff to L, R = S continues, C = C2C transfer)
for i in range(N):
    K = len(lines[i])
    for k in range(0, K):
        st = (SRC[i], pos_bucket(k), -1 if k == 0 else conf_bucket(conf[i][k - 1]))
        vt, ct = handoff(i, k, k)
        # remaining cost of R = S continues from history k (S prompt prefill sunk unless k == 0)
        cr = Cost().dec(S, P0[i] + cum[i][k], P0[i] + cum[i][-1]) if k > 0 else c_sroot(i)
        cs = Cost().pre(S, 0, P0[i]).dec(S, P0[i], P0[i] + cum[i][k]) if k > 0 else Cost()   # sunk S work, removed from T
        row = dict(i=i, k=k, state=st, VT=vt, FT=ct.f - cs.f, ST=ct.s - cs.s, VR=YS[i], FR=cr.f, SR=cr.s)
        if HAS_C2C:
            o = c2c(i, k, k)
            if o is not None:
                row.update(VC=o[0], FC=o[1].f - cs.f, SC=o[1].s - cs.s)
        HIST.append(row)
log('histories', len(HIST))

def sign_test(fit_mask, test_idx_set, lam, basis, A='T', Bn='R'):
    """State-level value-cost comparison of action A vs Bn at price lam: gain = (V_A - V_B) + lam (D_B - D_A).
    Prediction from fitting tasks; realized state mean with a task bootstrap on test tasks; decided if the CI excludes 0."""
    key = 'F' if basis == 'flops' else 'S'
    rows = [h for h in HIST if f'V{A}' in h and f'V{Bn}' in h]
    g = np.array([(h[f'V{A}'] - h[f'V{Bn}']) + lam * (h[f'{key}{Bn}'] - h[f'{key}{A}']) for h in rows])
    states = sorted(set(h['state'] for h in rows), key=str)
    ti = np.array([h['i'] for h in rows]); sidx = np.array([states.index(h['state']) for h in rows])
    fitm = np.array([FOLD[i] == fit_mask for i in ti])
    out = []
    rngb = np.random.default_rng(SEED + 99)
    test_tasks = np.array(sorted(test_idx_set))
    Wb = rngb.multinomial(len(test_tasks), np.full(len(test_tasks), 1 / len(test_tasks)), size=500).astype(float)
    wpos = {t: j for j, t in enumerate(test_tasks)}
    for s in range(len(states)):
        mf = fitm & (sidx == s); mt = (~fitm) & (sidx == s)
        if mf.sum() < 5 or mt.sum() < 5: continue
        pred = g[mf].mean()
        tt = np.array([wpos[t] for t in ti[mt]]); gt = g[mt]
        w = Wb[:, tt]; den = w.sum(1); bm = np.where(den > 0, (w * gt).sum(1) / np.maximum(den, 1e-12), np.nan)
        lo, hi = np.nanpercentile(bm, [2.5, 97.5]); real = gt.mean()
        out.append(dict(state=str(states[s]), n_fit=int(mf.sum()), n_test=int(mt.sum()), pred=float(pred), real=float(real), lo=float(lo), hi=float(hi),
                        decided=bool(lo > 0 or hi < 0), agree=bool(np.sign(pred) == np.sign(real))))
    dec = [o for o in out if o['decided']]
    def agree_rule(rule):
        return float(np.mean([rule(o) == (o['real'] > 0) for o in dec])) if dec else None
    n_pos = sum(o['real'] > 0 for o in dec)
    return dict(n_states=len(out), n_decided=len(dec), n_decided_positive=int(n_pos), state_dependent=bool(0 < n_pos < len(dec)),
                agree_best_constant=(max(n_pos, len(dec) - n_pos) / len(dec)) if dec else None,
                agree_value_cost=agree_rule(lambda o: o['pred'] > 0),
                agree_always_A=agree_rule(lambda o: True), agree_never_A=agree_rule(lambda o: False),
                agree_low_conf_only=agree_rule(lambda o: ('-1)' not in o['state']) and o['state'].endswith(', 0)')),
                weighted_loss_value_cost=float(np.mean([max(0, -o['real']) if o['pred'] > 0 else max(0, o['real']) for o in out])) if out else None,
                states=out)

SIGN = {}
for fold in (0, 1):
    test_set = set(np.where(FOLD != fold)[0])
    for basis in ('flops', 'sec'):
        fit = np.where(FOLD == fold)[0]; etaB = eta(POL[REF], fit, basis)[0]
        for mult in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 2.0):
            SIGN[f'fold{fold}/{basis}/lam={mult}etaB/T_vs_R'] = sign_test(fold, test_set, mult * etaB, basis, 'T', 'R')
            if HAS_C2C:
                SIGN[f'fold{fold}/{basis}/lam={mult}etaB/C_vs_R'] = sign_test(fold, test_set, mult * etaB, basis, 'C', 'R')
                SIGN[f'fold{fold}/{basis}/lam={mult}etaB/C_vs_T'] = sign_test(fold, test_set, mult * etaB, basis, 'C', 'T')
RES['local_comparisons'] = SIGN

# ------------------------------------------------------------------ transfer budget (eq:transfer-budget) for C2C vs text handoff, whole sample
if HAS_C2C:
    tb = {}
    for basis in ('flops', 'sec'):
        key = 'F' if basis == 'flops' else 'S'; etaB = eta(POL[REF], np.arange(N), basis)[0]
        rows = [h for h in HIST if 'VC' in h]
        dV_CT = np.mean([h['VC'] - h['VT'] for h in rows]); dV_CR = np.mean([h['VC'] - h['VR'] for h in rows])
        DT = np.mean([h[f'{key}T'] for h in rows]); DC = np.mean([h[f'{key}C'] for h in rows]); DR = np.mean([h[f'{key}R'] for h in rows])
        tb[basis] = dict(n_histories=len(rows), mean_V_C=float(np.mean([h['VC'] for h in rows])), mean_V_T=float(np.mean([h['VT'] for h in rows])),
                         mean_V_R=float(np.mean([h['VR'] for h in rows])), mean_D_C_rel=float(DC / DT), mean_D_R_rel=float(DR / DT),
                         A_C_vs_T_at_etaB=float(dV_CT + etaB * (DT - DC)), A_C_vs_R_at_etaB=float(dV_CR + etaB * (DR - DC)))
    RES['c2c_transfer_summary'] = tb

# ------------------------------------------------------------------ dominance by task-level randomization between S and L (chord)
def chord_gap(p, idx, basis):
    X = p['F'] if basis == 'flops' else p['S']; RS = POL['always_S']; RL = POL[REF]
    xs = (RS['F'] if basis == 'flops' else RS['S'])[idx].mean(); xl = (RL['F'] if basis == 'flops' else RL['S'])[idx].mean()
    vs, vl = RS['V'][idx].mean(), RL['V'][idx].mean(); x = X[idx].mean(); v = p['V'][idx].mean()
    lam = (x - xs) / (xl - xs); chord = vs + lam * (vl - vs) if lam <= 1 else vl
    return dict(score=float(v), cost_rel=float(x / xl), chord_score_at_cost=float(chord), gap=float(v - chord))
CH = {}
allidx = np.arange(N)
for basis in ('flops', 'sec'):
    rows = {}
    for nm, p in POL.items():
        if p['ok'].all(): rows[nm] = chord_gap(p, allidx, basis)
    for f, fam in FAM.items():
        for k, p in fam.items():
            if p['ok'].all(): rows[f'{f}:{k}'] = chord_gap(p, allidx, basis)
    CH[basis] = dict(rows=rows, n_above_chord=sum(r['gap'] > 0.005 for r in rows.values()), n_total=len(rows),
                     best_gap=max(rows.items(), key=lambda kv: kv[1]['gap'])[0])
RES['chord_dominance_whole_sample'] = CH
# per-task complementarity and max-value oracles
menu_step = [('always_S', POL['always_S']), (REF, POL[REF])] + [(f'{f}:{k}', p) for f, fam in FAM.items() for k, p in fam.items() if f != 'prompt_routing']
menu_step += [(nm, p) for nm, p in POL.items() if nm.startswith('C2C')]
Vall = np.column_stack([np.where(p['ok'], p['V'], 0) for _, p in menu_step])
only_step = (YS == 0) & (YL == 0) & (Vall.max(1) > 0)
RES['complementarity'] = dict(n_tasks=N, S_pass=int(YS.sum()), L_pass=int(YL.sum()), either_root=int(((YS > 0) | (YL > 0)).sum()),
                              any_stepwise_action=int((Vall.max(1) > 0).sum()), solved_only_by_stepwise=int(only_step.sum()),
                              L_pass_but_S_prefix_handoff_all_fail=int(sum(1 for i in range(N) if YL[i] > 0 and len(lines[i]) > 1 and
                                                                          all(SC[f'{PAIR}|L|{TID[i]}|fromS{k}'] == 0 for k in range(1, len(lines[i]))))),
                              frac_handoffs_below_L=float(np.mean([SC[f'{PAIR}|L|{TID[i]}|fromS{k}'] < YL[i] for i in range(N) for k in range(1, len(lines[i]))])),
                              frac_handoffs_above_L=float(np.mean([SC[f'{PAIR}|L|{TID[i]}|fromS{k}'] > YL[i] for i in range(N) for k in range(1, len(lines[i]))])),
                              mean_handoff_pass_by_k={str(k): float(np.mean([SC[f'{PAIR}|L|{TID[i]}|fromS{k}'] for i in range(N) if len(lines[i]) > k])) for k in (1, 2, 3, 5, 8, 12)},
                              mean_L_pass_same_tasks_by_k={str(k): float(np.mean([YL[i] for i in range(N) if len(lines[i]) > k])) for k in (1, 2, 3, 5, 8, 12)},
                              mean_S_pass_same_tasks_by_k={str(k): float(np.mean([YS[i] for i in range(N) if len(lines[i]) > k])) for k in (1, 2, 3, 5, 8, 12)})
for basis in ('flops', 'sec'):
    X = np.column_stack([(p['F'] if basis == 'flops' else p['S']) for _, p in menu_step]); X = np.where(np.column_stack([p['ok'] for _, p in menu_step]), X, np.inf)
    best = Vall.max(1); cand = np.where(Vall >= best[:, None] - 1e-12, X, np.inf); g = np.argmin(cand, 1)
    task_level = np.where(YL >= YS, np.where(YS >= YL, np.minimum(X[:, 0], X[:, 1]), X[:, 1]), X[:, 0])
    RES.setdefault('max_value_oracles', {})[basis] = dict(stepwise_score=float(best.mean()), stepwise_cost_rel=float(X[np.arange(N), g].mean() / X[:, 1].mean()),
                                                        tasklevel_score=float(np.maximum(YS, YL).mean()), tasklevel_cost_rel=float(task_level.mean() / X[:, 1].mean()))

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, float) and not math.isfinite(o): return None
    return o
json.dump(clean(RES), open(f'{D}/analysis_results.json', 'w'), indent=1)
log('ANALYSISDONE')

