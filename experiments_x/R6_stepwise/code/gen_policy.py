"""Live line-level intervention policy (R2R-like at line granularity), both models loaded in one process.

For every active task the small model S writes the next line.  If that line's minimum chosen-token logprob is below
tau, the line is discarded and the large model L writes the line instead (from the same prefix); S then resumes.
Repeats until EOS / budget.  All models share the Qwen3 tokenizer, so handoffs pass exact token ids.
Cost bookkeeping: for each model we record prefill segments [a, b) (context tokens not in that model's prefix cache,
i.e. beyond the longest common prefix with the last sequence the model processed) and decode segments.

usage: python gen_policy.py --pair A --taus t1 t2 ... --out runs/pairA/out_live.jsonl
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ap = argparse.ArgumentParser(); ap.add_argument('--pair', required=True); ap.add_argument('--taus', nargs='+', type=float, required=True)
ap.add_argument('--out', required=True); ap.add_argument('--util_s', type=float, default=0.25); ap.add_argument('--util_l', type=float, default=0.55)
a = ap.parse_args()
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer
S, L = PAIRS[a.pair]
tok = AutoTokenizer.from_pretrained(model_path(S)); NL = newline_token_ids(tok)
eng = {S: LLM(model=model_path(S), dtype='bfloat16', gpu_memory_utilization=a.util_s, enable_prefix_caching=True, max_model_len=4096, seed=0, max_num_seqs=256),
       L: LLM(model=model_path(L), dtype='bfloat16', gpu_memory_utilization=a.util_l, enable_prefix_caching=True, max_model_len=4096, seed=0, max_num_seqs=256)}
TASKS = load_tasks(); chat = {t['task_id']: tok(chat_prefix(tok, t['user']), add_special_tokens=False)['input_ids'] for t in TASKS}
done = {r['job_id'] for r in read_jsonl(a.out)} if os.path.exists(a.out) else set()
fout = open(a.out, 'a')

def lcp(x, y):
    n = min(len(x), len(y)); i = 0
    while i < n and x[i] == y[i]: i += 1
    return i

def step(model, items):
    """items: list of task states; generate one line each with `model`; returns (ids, lp_min, finish_eos)."""
    prompts, sps = [], []
    for s in items:
        ctx = s['chat'] + s['ids']
        c = lcp(s['last'][model], ctx)
        s['seg'].append((model, 'pre', max(c, 0), len(ctx)))
        prompts.append(TokensPrompt(prompt_token_ids=ctx))
        sps.append(SamplingParams(temperature=0.0, max_tokens=max(1, min(STEP_MAX_TOKENS, TOTAL_MAX_TOKENS - len(s['ids']))),
                                  stop_token_ids=NL, logprobs=1, seed=0))
    outs = eng[model].generate(prompts, sps, use_tqdm=False); res = []
    for s, o in zip(items, outs):
        c = o.outputs[0]; ids = list(c.token_ids)
        lps = [d[t].logprob for t, d in zip(c.token_ids, c.logprobs or []) if d is not None and t in d]
        ctx_len = len(s['chat']) + len(s['ids'])
        s['seg'].append((model, 'dec', ctx_len, ctx_len + len(ids)))
        s['last'][model] = s['chat'] + s['ids'] + ids
        eos = c.finish_reason == 'stop' and not (isinstance(c.stop_reason, int) and c.stop_reason in NL)
        res.append((ids, min(lps) if lps else 0.0, eos))
    return res

for tau in a.taus:
    t0 = time.time()
    st = [dict(tid=t['task_id'], chat=chat[t['task_id']], ids=[], seg=[], last={S: [], L: []}, n_int=0, n_lines=0, authors=[], fin=False, reason=None)
          for t in TASKS if f'{a.pair}|live|{tau:.6g}|{t["task_id"]}' not in done]
    while True:
        act = [s for s in st if not s['fin']]
        if not act: break
        rs = step(S, act); redo = []
        for s, (ids, lpm, eos) in zip(act, rs):
            if lpm < tau: redo.append(s); continue
            s['ids'] += ids; s['authors'].append('S'); s['n_lines'] += 1
            if eos: s['fin'] = True; s['reason'] = 'eos'
        if redo:
            rl = step(L, redo)
            for s, (ids, lpm, eos) in zip(redo, rl):
                s['ids'] += ids; s['authors'].append('L'); s['n_lines'] += 1; s['n_int'] += 1
                if eos: s['fin'] = True; s['reason'] = 'eos'
        for s in act:
            if not s['fin'] and (len(s['ids']) >= TOTAL_MAX_TOKENS or s['n_lines'] >= MAX_STEPS):
                s['fin'] = True; s['reason'] = 'budget'
    for s in st:
        fout.write(json.dumps(dict(job_id=f'{a.pair}|live|{tau:.6g}|{s["tid"]}', task_id=s['tid'], tau=tau, n_prefix=0, full_ids=s['ids'],
                                   full_text=tok.decode(s['ids']), segments=s['seg'], n_interventions=s['n_int'], authors=''.join(s['authors']),
                                   reason=s['reason'])) + '\n')
    fout.flush()
    print(f'tau {tau:.4g}: {len(st)} tasks, mean interventions {sum(s["n_int"] for s in st)/max(1,len(st)):.2f}, {time.time()-t0:.0f}s', flush=True)
print('LIVEDONE', flush=True)

