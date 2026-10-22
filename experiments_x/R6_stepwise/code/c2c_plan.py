"""C2C jobs for pair A: every S-root boundary k = 0..K-1 (k = 0: the native prompt-only fusion, 'C2C root')."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
D = f'{RUNS}/pairA'; jobs = []
for r in read_jsonl(f'{D}/out_A_S.jsonl'):
    ids = []; K = len(r['steps'])
    for k in range(0, K):
        if k > 0: ids = ids + r['steps'][k - 1]['ids']
        jobs.append(dict(job_id=f'A|C2C|{r["task_id"]}|k{k}', task_id=r['task_id'], prefix_ids=list(ids)))
jobs.sort(key=lambda j: (int(j['job_id'].rsplit('|k', 1)[1]), j['task_id']))   # roots first
write_jsonl(f'{D}/jobs_c2c.jsonl', jobs); print('c2c jobs', len(jobs))

