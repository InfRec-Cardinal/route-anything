"""Score final responses with the EvalPlus plus tests (sandbox copied from experiments_x/common/sandbox.py).

usage: python eval_tests.py --inputs a.jsonl b.jsonl ... --out scores.jsonl [--canonical]
Each input row needs job_id, task_id and full_text.  Results are cached by the task identifier and extracted source code.
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
os.environ.setdefault('EX_SANDBOX_MPLCONFIGDIR', os.path.join(ROOT, 'cache', 'mplconfig'))
os.makedirs(os.environ['EX_SANDBOX_MPLCONFIGDIR'], exist_ok=True)
import sandbox

TASKS = {t['task_id']: t for t in load_tasks()}
CACHE_P = f'{ROOT}/cache/test_cache.jsonl'

def code_for(task, text):
    code = sandbox.extract_code(text, 'python', task['entry_point'])
    if task['source'] == 'humaneval+':
        code = sandbox._with_prompt_imports(code, task['prompt'])
    return code

def run_one(arg):
    tid, code = arg
    t = TASKS[tid]
    if not code.strip():
        return dict(passed=False, frac=0.0, error='empty')
    r = sandbox.run_python_tests(code, t['tests'], entry_point=t['entry_point'] if t['source'] == 'humaneval+' else None,
                                 timeout_s=20, mem_mb=4096)
    return dict(passed=bool(r['passed']), frac=float(r.get('frac') or 0.0), error=(r.get('error') or '')[:200],
                timed_out=bool(r.get('timed_out')))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--inputs', nargs='*', default=[]); ap.add_argument('--out', required=True)
    ap.add_argument('--canonical', action='store_true'); ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--skip', default=None, help='skip rows whose job_id contains this substring')
    a = ap.parse_args()
    rows = []
    if a.canonical:
        rows = [dict(job_id='canonical|' + tid, task_id=tid, full_text='```python\n' + t['canonical'] + '\n```') for tid, t in TASKS.items()]
    for p in a.inputs:
        rows += [r for r in read_jsonl(p) if not (a.skip and a.skip in r['job_id'])]
    cache = {}
    if os.path.exists(CACHE_P):
        for r in read_jsonl(CACHE_P): cache[r['key']] = r['res']
    keyed = []
    for r in rows:
        code = code_for(TASKS[r['task_id']], r['full_text'])
        key = json.dumps([r['task_id'], code], ensure_ascii=False, separators=(',', ':'))
        keyed.append((r, code, key))
    todo = {}
    for r, code, key in keyed:
        if key not in cache and key not in todo: todo[key] = (r['task_id'], code)
    print(f'{len(rows)} responses, {len(todo)} new programs to run', flush=True)
    keys = list(todo)
    with ProcessPoolExecutor(a.workers) as ex, open(CACHE_P, 'a') as fc:
        for k, res in zip(keys, ex.map(run_one, [todo[k] for k in keys], chunksize=4)):
            cache[k] = res; fc.write(json.dumps(dict(key=k, res=res)) + '\n')
    out = [dict(job_id=r['job_id'], task_id=r['task_id'], code_key=key, **cache[key]) for r, code, key in keyed]
    write_jsonl(a.out, out)
    print('EVALDONE', len(out), 'pass rate', sum(o['passed'] for o in out) / max(1, len(out)), flush=True)

