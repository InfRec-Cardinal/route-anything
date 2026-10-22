"""C2C (Cache-to-Cache, thu-nics) continuation for pair A: receiver Qwen3-0.6B, sharer Qwen3-4B, released fuser
nics-efc/C2C_Fuser/qwen3_0.6b+qwen3_4b_Fuser (trained on OpenHermes, prompt-section fusion).

A job = (task, token-id prefix written by the receiver S so far).  The receiver re-reads chat prompt + prefix while
the sharer computes its own KV cache over the same tokens; the fuser projects the sharer cache into the receiver's
cache for every token but the last (kv_cache_index section [1,0]); the receiver then decodes greedily until EOS or
the global budget.  job_id suffix 'c2c_root' = prefix empty (the native C2C use: fused instruction section).
Batch size 1 (the wrapper's section logic reads the sharer mask of batch row 0).  Runs in the envs/c2c environment
(torch 2.6, transformers 4.52.4 as pinned by C2C).

usage: python c2c_gen.py --jobs jobs.jsonl --out out.jsonl --worker i --n_workers n
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rosetta.model.projector import load_projector
from rosetta.model.wrapper import RosettaModel

ap = argparse.ArgumentParser(); ap.add_argument('--jobs', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--worker', type=int, default=0); ap.add_argument('--n_workers', type=int, default=1)
ap.add_argument('--mode', default='fused', choices=['fused', 'plain'])
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
for p in glob.glob(os.path.join(os.path.dirname(a.out), os.path.basename(a.out).split('_w')[0] + '_w*.jsonl')) + [a.out]:
    if os.path.exists(p): done |= {r['job_id'] for r in read_jsonl(p)}
jobs = [j for j in jobs if j['job_id'] not in done]
print(f'worker {a.worker}: {len(jobs)} jobs, {n_proj} projectors, {PROJ_PARAMS/1e6:.1f}M projector params', flush=True)
eos = [151645, 151643]
f = open(a.out, 'a'); t0 = time.time()
for n, j in enumerate(jobs):
    ids = chat_ids[j['task_id']] + list(j['prefix_ids'])
    x = torch.tensor([ids], device=dev)
    L = x.shape[1]
    fuse = 1 if a.mode == 'fused' else -1
    kvi = [torch.tensor([fuse, 0], dtype=torch.long).repeat(L - 1, 1).unsqueeze(0).to(dev),
           torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(dev)]
    budget = TOTAL_MAX_TOKENS - len(j['prefix_ids'])
    torch.cuda.synchronize(); t1 = time.time()
    with torch.no_grad():
        out = model.generate(kv_cache_index=kvi, input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max(1, budget),
                             do_sample=False, eos_token_id=eos)
    torch.cuda.synchronize(); dt = time.time() - t1
    gen = out[0, L:].tolist()
    while gen and gen[-1] in eos: gen = gen[:-1]
    full = list(j['prefix_ids']) + gen
    f.write(json.dumps(dict(job_id=j['job_id'], task_id=j['task_id'], model=f'c2c-{a.mode}', n_prefix=len(j['prefix_ids']),
                            n_context=L, n_gen=len(gen), full_ids=full, full_text=tok.decode(full), wall_s=dt,
                            proj_params=PROJ_PARAMS, reason='eos' if len(gen) < budget else 'budget')) + '\n')
    if n % 20 == 0:
        f.flush(); print(f'{n+1}/{len(jobs)} {time.time()-t0:.0f}s', flush=True)
f.close(); print('C2CDONE', a.worker, len(jobs), f'{time.time()-t0:.0f}s', flush=True)

