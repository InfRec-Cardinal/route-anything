"""Build the job files of each generation phase for a model pair (S = small, L = large).

phase A : S root and L root trajectories (plus single-call copies of the roots to check that stepping is exact) (from the prompt, until EOS / budget)
phase B : L continues from every S boundary k (S->L handoff); L writes exactly one step from every S boundary
          (one-step intervention, first half); S continues from every L boundary j (L->S handoff)
phase C : S resumes after L's one intervention step (second half of the one-step intervention)
usage: python plan.py --pair A --phase A|B|C
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ap = argparse.ArgumentParser(); ap.add_argument('--pair', required=True); ap.add_argument('--phase', required=True)
a = ap.parse_args()
S, L = PAIRS[a.pair]; D = f'{RUNS}/pair{a.pair}'; os.makedirs(D, exist_ok=True)
tasks = load_tasks()

def jobs_path(role, phase): return f'{D}/jobs_{phase}_{role}.jsonl'
def out_path(role, phase): return f'{D}/out_{phase}_{role}.jsonl'

def prefixes(row):
    """(k, prefix token ids) for every interior boundary k = 1..K-1 of a finished trajectory (line boundaries)."""
    st = row['steps']; out = []; ids = list(row['full_ids'][:row['n_prefix']])
    for k in range(1, len(st)):
        ids = ids + st[k - 1]['ids']
        out.append((k, list(ids)))
    return out

if a.phase == 'A':
    for role in ('S', 'L'):
        write_jsonl(jobs_path(role, 'A'), [dict(job_id=f'{a.pair}|{role}|{t["task_id"]}|root', task_id=t['task_id'], prefix_ids=[],
                                              max_steps=None) for t in tasks])
        write_jsonl(jobs_path(role, 'A1'), [dict(job_id=f'{a.pair}|{role}|{t["task_id"]}|single', task_id=t['task_id'], prefix_ids=[],
                                               max_steps=None) for t in tasks])
elif a.phase == 'B':
    sroot = {r['task_id']: r for r in read_jsonl(out_path('S', 'A'))}
    lroot = {r['task_id']: r for r in read_jsonl(out_path('L', 'A'))}
    jl, js = [], []
    for tid, r in sroot.items():
        for k, ids in prefixes(r):
            jl.append(dict(job_id=f'{a.pair}|L|{tid}|fromS{k}', task_id=tid, prefix_ids=ids, max_steps=None))
            jl.append(dict(job_id=f'{a.pair}|L1|{tid}|fromS{k}', task_id=tid, prefix_ids=ids, max_steps=1))
    for tid, r in lroot.items():
        for j, ids in prefixes(r):
            js.append(dict(job_id=f'{a.pair}|S|{tid}|fromL{j}', task_id=tid, prefix_ids=ids, max_steps=None))
    write_jsonl(jobs_path('L', 'B'), jl); write_jsonl(jobs_path('S', 'B'), js)
    print('phase B jobs: L', len(jl), 'S', len(js))
elif a.phase == 'C':
    js = []
    for r in read_jsonl(out_path('L', 'B')):
        if '|L1|' not in r['job_id'] or r['reason'] != 'step_limit':
            continue
        k = r['job_id'].split('|')[-1]
        js.append(dict(job_id=f'{a.pair}|S|{r["task_id"]}|{k}L1', task_id=r['task_id'], prefix_ids=r['full_ids'], max_steps=None))
    write_jsonl(jobs_path('S', 'C'), js)
    print('phase C jobs: S', len(js))

