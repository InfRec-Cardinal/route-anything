"""Batch-1 latency model on one idle H20: t(call) = c0 + c_prefill * n_prefill + c_decode * n_decode (seconds).
Random token prompts, prefix caching off, ignore_eos, 5 repetitions per configuration.  Writes runs/latency_<model>.json.
usage: python bench_latency.py --model Qwen3-4B
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--gpu_util', type=float, default=0.5)
a = ap.parse_args()
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
llm = LLM(model=model_path(a.model), dtype='bfloat16', gpu_memory_utilization=a.gpu_util, enable_prefix_caching=False,
          max_model_len=4096, seed=0)
rng = np.random.default_rng(0)
rows = []
def run(n_p, n_d):
    ids = rng.integers(1000, 100000, size=n_p).tolist()
    sp = SamplingParams(temperature=0.0, max_tokens=n_d, ignore_eos=True)
    t = time.perf_counter(); llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False); return time.perf_counter() - t
for _ in range(3): run(128, 8)                      # warm-up
for n_p in [32, 128, 512, 1024, 2048]:
    for n_d in [1, 64, 256]:
        for r in range(5):
            rows.append((n_p, n_d, run(n_p, n_d)))
X = np.array([[1.0, p, d - 1] for p, d, _ in rows]); y = np.array([t for _, _, t in rows])
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
res = dict(model=a.model, c0_s=float(coef[0]), prefill_s_per_token=float(coef[1]), decode_s_per_token=float(coef[2]),
           r2=float(1 - ((X @ coef - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()), n_obs=len(rows), rows=rows)
json.dump(res, open(f'{RUNS}/latency_{a.model}.json', 'w'), indent=1)
print('LATDONE', json.dumps({k: v for k, v in res.items() if k != 'rows'}), flush=True)

