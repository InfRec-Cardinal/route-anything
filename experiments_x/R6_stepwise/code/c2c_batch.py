"""Batched C2C continuation (same model, fuser and prompt layout as c2c_gen.py, much faster).

Per job the RosettaModel prefill fuses the sharer (Qwen3-4B) cache into the receiver (Qwen3-0.6B) cache over every
context token but the last (kv_cache_index [1,0]; mode 'plain' uses [-1,0], i.e. no fusion).  The returned receiver
caches of a chunk of jobs are left-padded and stacked, and the receiver decodes greedily in one batch (the released
fuser is applied to the context only, include_response = false, exactly as in RosettaModel.generate).
Output rows match c2c_gen.py.  usage: python c2c_batch.py --jobs J --out O [--mode fused|plain] [--worker i --n_workers n]
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from rosetta.model.projector import load_projector
from rosetta.model.wrapper import RosettaModel

ap = argparse.ArgumentParser(); ap.add_argument('--jobs', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--worker', type=int, default=0); ap.add_argument('--n_workers', type=int, default=1)
ap.add_argument('--mode', default='fused', choices=['fused', 'plain']); ap.add_argument('--batch', type=int, default=48)
ap.add_argument('--reverse', action='store_true', help='process the longest contexts first (helper for a slow worker; merge dedups)')
a = ap.parse_args()
dev = torch.device('cuda')
CK = glob.glob(f'{HF_HUB}/models--nics-efc--C2C_Fuser/snapshots/*/qwen3_0.6b+qwen3_4b_Fuser/final')[0]
tok = AutoTokenizer.from_pretrained(model_path('Qwen3-0.6B'))
slm = AutoModelForCausalLM.from_pretrained(model_path('Qwen3-0.6B'), torch_dtype=torch.bfloat16, device_map={'': dev}).eval()
llm = AutoModelForCausalLM.from_pretrained(model_path('Qwen3-4B'), torch_dtype=torch.bfloat16, device_map={'': dev}).eval()
n_proj = len([f for f in os.listdir(CK) if f.startswith('projector_') and f.endswith('.pt')])
projs = []
for t in range(n_proj):
    p = load_projector(os.path.join(CK, f'projector_{t}.json')).to(dev)
    p.load_state_dict(torch.load(os.path.join(CK, f'projector_{t}.pt'), map_location=dev), strict=False)
    projs.append(p.eval())
model = RosettaModel(model_list=[slm, llm], base_model_idx=0, projector_list=projs).to(dev).eval()
model.load_projector_config(os.path.join(CK, 'projector_config.json'))
PROJ_PARAMS = int(sum(p.numel() for pr in projs for p in pr.parameters()))
TASKS = {t['task_id']: t for t in load_tasks()}
chat_ids = {tid: tok(chat_prefix(tok, t['user']), add_special_tokens=False)['input_ids'] for tid, t in TASKS.items()}
jobs = [j for i, j in enumerate(read_jsonl(a.jobs)) if i % a.n_workers == a.worker]
done = set()
if os.path.exists(a.out): done = {r['job_id'] for r in read_jsonl(a.out)}
jobs = [j for j in jobs if j['job_id'] not in done]
jobs.sort(key=lambda j: len(chat_ids[j['task_id']]) + len(j['prefix_ids']), reverse=a.reverse)
EOS = {151645, 151643}; PAD = 151643
print(f'worker {a.worker} mode {a.mode}: {len(jobs)} jobs', flush=True)
f = open(a.out, 'a'); t0 = time.time(); nd = 0

@torch.no_grad()
def prefill(j):
    ids = chat_ids[j['task_id']] + list(j['prefix_ids']); x = torch.tensor([ids], device=dev); Lx = x.shape[1]
    fuse = 1 if a.mode == 'fused' else -1
    kvi = [torch.tensor([fuse, 0], dtype=torch.long).repeat(Lx - 1, 1).unsqueeze(0).to(dev),
           torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(dev)]
    model.kv_cache_dict = dict()
    out = model.forward(kv_cache_index=kvi, input_ids=x, attention_mask=torch.ones_like(x), use_cache=True)
    pkv = out.past_key_values
    kv = [(pkv.key_cache[l].clone(), pkv.value_cache[l].clone()) for l in range(len(pkv.key_cache))]
    return Lx, int(out.logits[0, -1].argmax()), kv

@torch.no_grad()
def run_chunk(chunk):
    t1 = time.time(); pre = [prefill(j) for j in chunk]; t_pre = time.time() - t1
    B = len(chunk); lens = [p[0] for p in pre]; M = max(lens); nl = len(pre[0][2])
    cache = DynamicCache()
    for l in range(nl):
        ks, vs = [], []
        for Lx, _, kv in pre:
            k, v = kv[l]; padn = M - Lx
            if padn: k = torch.cat([k.new_zeros(k.shape[0], k.shape[1], padn, k.shape[3]), k], 2); v = torch.cat([v.new_zeros(v.shape[0], v.shape[1], padn, v.shape[3]), v], 2)
            ks.append(k); vs.append(v)
        cache.update(torch.cat(ks, 0), torch.cat(vs, 0), l)
    mask = torch.zeros(B, M, dtype=torch.long, device=dev)
    for b, Lx in enumerate(lens): mask[b, M - Lx:] = 1
    budget = [TOTAL_MAX_TOKENS - len(j['prefix_ids']) for j in chunk]
    nxt = torch.tensor([p[1] for p in pre], device=dev); pos = torch.tensor(lens, device=dev)
    gen = [[] for _ in range(B)]; fin = [False] * B
    for b in range(B):
        t_ = int(nxt[b])
        if t_ in EOS: fin[b] = True
        else: gen[b].append(t_)
        if len(gen[b]) >= budget[b]: fin[b] = True
    steps = 0
    while not all(fin):
        mask = torch.cat([mask, torch.ones(B, 1, dtype=torch.long, device=dev)], 1)
        inp = torch.where(torch.tensor(fin, device=dev), torch.full_like(nxt, PAD), nxt)
        out = slm(input_ids=inp[:, None], attention_mask=mask, position_ids=pos[:, None], past_key_values=cache, use_cache=True)
        cache = out.past_key_values; nxt = out.logits[:, -1].argmax(-1); pos = pos + 1; steps += 1
        for b in range(B):
            if fin[b]: continue
            t_ = int(nxt[b])
            if t_ in EOS: fin[b] = True; continue
            gen[b].append(t_)
            if len(gen[b]) >= budget[b]: fin[b] = True
    dt = time.time() - t1
    rows = []
    for b, j in enumerate(chunk):
        full = list(j['prefix_ids']) + gen[b]
        rows.append(dict(job_id=j['job_id'], task_id=j['task_id'], model=f'c2c-{a.mode}', n_prefix=len(j['prefix_ids']), n_context=lens[b],
                         n_gen=len(gen[b]), full_ids=full, full_text=tok.decode(full), wall_s_chunk=dt, prefill_s_chunk=t_pre, chunk=B,
                         proj_params=PROJ_PARAMS, reason='eos' if len(gen[b]) < budget[b] else 'budget', impl='batched'))
    return rows

for s in range(0, len(jobs), a.batch):
    ch = jobs[s:s + a.batch]
    for r in run_chunk(ch): f.write(json.dumps(r) + '\n')
    f.flush(); nd += len(ch)
    print(f'{nd}/{len(jobs)} {time.time()-t0:.0f}s', flush=True)
f.close(); print('C2CDONE', a.worker, len(jobs), f'{time.time()-t0:.0f}s', flush=True)

