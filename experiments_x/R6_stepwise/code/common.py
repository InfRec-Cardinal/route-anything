"""Shared paths, model registry, prompts, task loading and cost accounting for the step-wise routing study."""
import glob, json, os, re
import pandas as pd

ROOT = os.path.abspath(os.environ.get('STEPWISE_ROOT', os.path.dirname(os.path.dirname(__file__))))
MODELS_DIR = f'{ROOT}/models'
HF_HUB = os.environ.get('HF_HUB_CACHE', os.path.expanduser('~/.cache/huggingface/hub'))
RUNS = f'{ROOT}/runs'
os.makedirs(RUNS, exist_ok=True)
SEED = 20260917
# non-embedding parameters (Qwen3 technical report, Table 1 counts minus tied/untied embeddings)
MODELS = {
    'Qwen3-0.6B': dict(path='Qwen/Qwen3-0.6B', params_ne=0.44e9),
    'Qwen3-1.7B': dict(path='Qwen/Qwen3-1.7B', params_ne=1.4e9),
    'Qwen3-4B':   dict(path='Qwen/Qwen3-4B', params_ne=3.6e9),
    'Qwen3-14B':  dict(path='Qwen/Qwen3-14B', params_ne=13.2e9),
}
PAIRS = {'A': ('Qwen3-0.6B', 'Qwen3-4B'), 'B': ('Qwen3-1.7B', 'Qwen3-14B')}
STEP_MAX_TOKENS = 256
TOTAL_MAX_TOKENS = 1536
MAX_STEPS = 256

def model_path(name):
    local_path = os.path.join(MODELS_DIR, name)
    p = local_path if os.path.isdir(local_path) else MODELS[name]['path']
    if p.endswith('snapshots'):
        p = sorted(glob.glob(p + '/*'))[-1]
    return p

INSTR = ('First explain your approach briefly in a few sentences. Then give the complete, self-contained '
         'Python implementation in a single ```python code block.')

def load_tasks():
    """HumanEval+ (164) and MBPP+ (378) from the EvalPlus HF datasets; plus tests are used for scoring."""
    tasks = []
    he = pd.read_parquet(sorted(glob.glob(f'{HF_HUB}/datasets--evalplus--humanevalplus/snapshots/*/**/*.parquet', recursive=True))[0])
    for r in he.itertuples():
        user = f'Complete the following Python function.\n\n```python\n{r.prompt}```\n\n{INSTR}'
        tasks.append(dict(task_id=r.task_id, source='humaneval+', user=user, prompt=r.prompt, entry_point=r.entry_point,
                          tests=r.test, canonical=r.prompt + r.canonical_solution))
    mb = pd.read_parquet(sorted(glob.glob(f'{HF_HUB}/datasets--evalplus--mbppplus/snapshots/*/**/*.parquet', recursive=True))[0])
    for r in mb.itertuples():
        tl = list(r.test_list)
        m = re.search(r'assertion\((\w+)\(\*inp\)', r.test)
        ep = m.group(1) if m else None
        user = (f'{r.prompt}\nYour code should satisfy this test:\n{tl[0]}\n\n{INSTR}') if tl else f'{r.prompt}\n\n{INSTR}'
        imports = '\n'.join(list(r.test_imports)) if len(r.test_imports) else ''
        tasks.append(dict(task_id=f'Mbpp/{r.task_id}' if not str(r.task_id).startswith('Mbpp') else str(r.task_id), source='mbpp+',
                          user=user, prompt='', entry_point=ep, tests=(imports + '\n' + r.test) if imports else r.test,
                          canonical=r.code))
    return [t for t in tasks if t['task_id'] not in EXCLUDE]

# HumanEval/32 (find_zero): the EvalPlus plus-test harness rejects even the canonical solution in this sandbox.
EXCLUDE = {'HumanEval/32'}

def newline_token_ids(tok):
    """Ids of every non-special vocabulary token whose text contains a newline: a line-level step ends with one."""
    special = set(tok.all_special_ids)
    return [i for i in range(len(tok)) if i not in special and '\n' in tok.decode([i])]

def chat_prefix(tok, user):
    return tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)

def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(p, rows):
    tmp = p + '.tmp'
    with open(tmp, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    os.replace(tmp, p)

QS = [0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 40, 50]

def split_folds(tasks):
    """Source-stratified random halves (fold 0 / fold 1), seed SEED."""
    import numpy as np
    src = np.array([t['source'] for t in tasks]); rng = np.random.default_rng(SEED); fold = np.zeros(len(tasks), int)
    for s in sorted(set(src)):
        idx = np.where(src == s)[0]; perm = rng.permutation(idx); fold[perm[len(idx) // 2:]] = 1
    return fold

def tau_grid(pair):
    """Line-confidence thresholds: quantiles QS of the S-root lines' minimum token logprob on fold-0 tasks."""
    import numpy as np
    tasks = load_tasks(); fold = split_folds(tasks); S = PAIRS[pair][0]
    rows = {r['task_id']: r for r in read_jsonl(f'{RUNS}/pair{pair}/out_A_S.jsonl')}
    lp = np.concatenate([[st['lp_min'] for st in rows[t['task_id']]['steps']] for t, f in zip(tasks, fold) if f == 0])
    return sorted(set(float(np.percentile(lp, q)) for q in QS))

