#!/usr/bin/env python3
"""X12c: use TypeSafe Jev as a direct model-selection policy on the X12b sample."""
import argparse, concurrent.futures, json, math, os, random, sys, time, urllib.error, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EX = HERE.parent
sys.path.insert(0, str(EX))
from common import sources

DATA = Path(os.environ.get('ACROUTER_DATA', 'data/agent-as-a-router/data'))
ID_RESULTS = DATA / 'coderouterbench' / 'id_results_long.csv'
SOURCES = ['humaneval', 'he', 'mbpp', 'bcb', 'hex', 'lcb', 'ds1000']
MODELS = ['Qwen3-Max', 'kimi-k2.5', 'qwen3.5-plus', 'MiniMax-M2.7', 'glm-5', 'gpt-5.4', 'claude-sonnet-4-6', 'claude-opus-4-6']
MODEL_DESC = {
    'Qwen3-Max': 'low-cost general-purpose coding model; good default for routine tasks',
    'kimi-k2.5': 'low-cost general-purpose coding model; alternate primary',
    'qwen3.5-plus': 'mid-cost coding model; broader capacity than the low-cost primaries',
    'MiniMax-M2.7': 'mid-cost general-purpose coding model',
    'glm-5': 'mid-cost coding model',
    'gpt-5.4': 'higher-cost general-purpose coding model',
    'claude-sonnet-4-6': 'high-cost strong coding model',
    'claude-opus-4-6': 'highest-cost strongest fallback coding model',
}

SYSTEM_INSTRUCTIONS = (
    'Choose the single backend that should receive this programming benchmark task. '
    'Use only the task statement and metadata. Do not solve the task or write code. '
    'Maximize expected hidden-test correctness while keeping cost low: use a cheaper model when it is likely sufficient, '
    'and select a stronger model when the task specification, algorithmic difficulty, edge cases, or required output format warrant it.'
)


def allocate(caps, total):
    alloc = {k: 0 for k in caps}; rem = int(total); active = {k for k in caps if caps[k] > 0}
    while rem > 0 and active:
        share = rem // len(active)
        if share == 0:
            for k in sorted(active, key=lambda q: (-(caps[q] - alloc[q]), q)):
                if rem == 0: break
                if alloc[k] < caps[k]: alloc[k] += 1; rem -= 1
            break
        for k in sorted(active):
            take = min(share, caps[k] - alloc[k]); alloc[k] += take; rem -= take
            if alloc[k] >= caps[k]: active.discard(k)
    return alloc


def largest_remainder(weights, total):
    w = np.asarray(weights, float); q = w / w.sum() * total; fl = np.floor(q).astype(int)
    for i in np.argsort(-(q - fl), kind='stable')[:total - fl.sum()]: fl[i] += 1
    return fl


def make_sample(seed=20260917, n_per_split=600):
    df = pd.read_csv(ID_RESULTS)
    S = df.pivot(index='task_id', columns='model', values='score')
    C = df.pivot(index='task_id', columns='model', values='cost_usd')
    meta = df.drop_duplicates('task_id').set_index('task_id')[['split', 'dimension']].loc[S.index].copy()
    meta['source'] = meta.index.str.split('_').str[1]
    meta['cell'] = meta.dimension + '|' + meta.source
    meta['allzero'] = (S.max(axis=1) <= 0)
    pop = meta[meta.source.isin(SOURCES) & ~meta.allzero].copy()
    rng = np.random.default_rng(seed)
    def sample_split(split):
        sub = pop[pop.split == split]
        caps = {s: int((sub.source == s).sum()) for s in SOURCES}; alloc = allocate(caps, n_per_split); out = []
        for s in SOURCES:
            g = sub[sub.source == s]; cells = sorted(g.cell.unique())
            per_cell = largest_remainder([int((g.cell == c).sum()) for c in cells], alloc[s])
            for c, n in zip(cells, per_cell):
                ids = np.array(sorted(g.index[g.cell == c])); n = min(int(n), len(ids))
                if n > 0: out.extend(rng.choice(ids, n, replace=False).tolist())
        return sorted(out), alloc
    prob, alloc_p = sample_split('probing'); test, alloc_t = sample_split('id_test')
    return S, C, meta, prob, test, alloc_p, alloc_t


def truncate_mid(s, n=5000):
    if len(s) <= n: return s
    h = n // 2
    return s[:h] + f'\n... [{len(s)-n} characters omitted] ...\n' + s[-h:]


def state_for(tid, meta):
    task = sources.load_task(tid)
    prompt = task.get('prompt') or ''
    return {
        'task_id': str(tid),
        'source': str(meta.loc[tid, 'source']),
        'dimension': str(meta.loc[tid, 'dimension']),
        'task': truncate_mid(prompt),
    }


def request_one(tid, state, timeout=90, retries=4):
    key = os.environ.get('TYPESAFE_API_KEY')
    if not key:
        raise RuntimeError('TYPESAFE_API_KEY is not set')
    body = {
        'model': 'jev-latest',
        'state': state,
        'questions': {
            'route': {
                'type': 'choice',
                'instructions': SYSTEM_INSTRUCTIONS,
                'criteria': {m: MODEL_DESC[m] for m in MODELS},
            }
        },
    }
    payload = json.dumps(body, ensure_ascii=False).encode('utf-8')
    for attempt in range(retries):
        req = urllib.request.Request(
            'https://api.typesafe.ai/v1/systemone', data=payload,
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.loads(resp.read().decode('utf-8'))
            ans = (out.get('answers') or {}).get('route') or {}
            choice = ans.get('choice') if isinstance(ans, dict) else None
            return {'task_id': str(tid), 'choice': choice, 'answer': ans, 'model': out.get('model'),
                    'usage': out.get('usage') or {}, 'latency_s': time.time() - t0, 'error': None}
        except urllib.error.HTTPError as e:
            status = int(e.code)
            try: detail = e.read().decode('utf-8')[:500]
            except Exception: detail = ''
            if status not in (408, 409, 425, 429, 500, 502, 503, 504, 529) or attempt == retries - 1:
                return {'task_id': str(tid), 'choice': None, 'answer': {}, 'model': None, 'usage': {},
                        'latency_s': time.time() - t0, 'error': f'HTTP {status}: {detail}'}
        except Exception as e:
            if attempt == retries - 1:
                return {'task_id': str(tid), 'choice': None, 'answer': {}, 'model': None, 'usage': {},
                        'latency_s': time.time() - t0, 'error': type(e).__name__ + ': ' + str(e)[:300]}
        time.sleep(min(8.0, 0.5 * (2 ** attempt)) + random.random() * 0.2)
    raise AssertionError('unreachable')


def call_all(ids, meta, cache_path, workers=24, limit=0):
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                try:
                    row = json.loads(line); cache[str(row['task_id'])] = row
                except Exception: pass
    ids = [str(x) for x in ids]
    if limit: ids = ids[:limit]
    todo = [tid for tid in ids if tid not in cache]
    print(f'[jev] target={len(ids)} cached={len(ids)-len(todo)} live={len(todo)} workers={workers}', flush=True)
    if todo:
        rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(request_one, tid, state_for(tid, meta)): tid for tid in todo}
            for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                row = fut.result(); cache[str(row['task_id'])] = row; rows.append(row)
                if i % 50 == 0 or i == len(todo):
                    ok = sum(1 for r in rows if r.get('choice') in MODELS)
                    print(f'[jev] {i}/{len(todo)} completed parse_ok={ok}', flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open('w', encoding='utf-8') as f:
            for tid in ids:
                if tid in cache: f.write(json.dumps(cache[tid], ensure_ascii=False) + '\n')
    return [cache[tid] for tid in ids if tid in cache]


def summarize(rows, split_name, S, C, meta):
    rs = pd.DataFrame(rows).set_index('task_id')
    idx = [x for x in rs.index if x in S.index]
    rs = rs.loc[idx]
    chosen = rs.choice
    valid = chosen.isin(MODELS)
    out = {'split': split_name, 'n': len(rs), 'n_valid_choice': int(valid.sum()), 'parse_rate': float(valid.mean()) if len(rs) else None,
           'selection_counts': {str(k): int(v) for k, v in chosen.value_counts(dropna=False).items()}}
    for m in MODELS:
        sc = S.loc[idx, m]; co = C.loc[idx, m]
        out[f'fixed_{m}'] = {'score_mean': float(sc.mean()), 'pass_at_1': float((sc >= 1.0).mean()), 'cost_mean_usd': float(co.mean())}
    valid_ids = [i for i in idx if chosen.loc[i] in MODELS]
    if valid_ids:
        psc = np.array([float(S.loc[i, chosen.loc[i]]) for i in valid_ids])
        pco = np.array([float(C.loc[i, chosen.loc[i]]) for i in valid_ids])
        best = S.loc[valid_ids, MODELS].max(axis=1)
        cheapest = C.loc[valid_ids, MODELS].min(axis=1)
        out['jev_policy'] = {'score_mean': float(psc.mean()), 'pass_at_1': float((psc >= 1.0).mean()),
                             'cost_mean_usd': float(pco.mean()), 'score_sum': float(psc.sum()),
                             'oracle_best_score_mean': float(best.mean()), 'oracle_best_pass_at_1': float((best >= 1.0).mean()),
                             'oracle_cheapest_cost_mean_usd': float(cheapest.mean()),
                             'choice_matches_best_score_model': float(np.mean([chosen.loc[i] in S.loc[i, MODELS][S.loc[i, MODELS] == S.loc[i, MODELS].max()].index for i in valid_ids]))}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int, default=0, help='first N tasks per split for smoke'); ap.add_argument('--workers', type=int, default=24); ap.add_argument('--out', default=str(HERE)); ap.add_argument('--seed', type=int, default=20260917)
    ap.add_argument('--collect', action='store_true', help='collect new Jev choices; otherwise audit the archived full replay')
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if not args.collect:
        import runpy
        runpy.run_path(str(HERE / 'audit_official.py'), run_name='__main__')
        return
    S, C, meta, prob, test, alloc_p, alloc_t = make_sample(args.seed)
    if args.limit: prob, test = prob[:args.limit], test[:args.limit]
    rows_p = call_all(prob, meta, out / 'jev_calls_probing.jsonl', args.workers)
    rows_t = call_all(test, meta, out / 'jev_calls_id_test.jsonl', args.workers)
    rows = rows_p + rows_t
    for r in rows:
        r['split'] = 'probing' if str(r['task_id']) in set(map(str, prob)) else 'id_test'
    (out / 'jev_calls.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n', encoding='utf-8')
    usage = [r.get('usage') or {} for r in rows]; in_tok = sum(int(u.get('input_tokens') or 0) for u in usage); out_tok = sum(int(u.get('output_tokens') or 0) for u in usage)
    report = {'experiment_id':'X12c_jev_policy','model':'jev-latest','resolved_models':sorted({r.get('model') for r in rows if r.get('model')}),
              'setting':{'seed':args.seed,'n_probing':len(prob),'n_id_test':len(test),'alloc_probing':alloc_p,'alloc_id_test':alloc_t,'models':MODELS,'question_type':'choice','workers':args.workers},
              'api':{'n_calls':len(rows),'n_errors':sum(bool(r.get('error')) for r in rows),'input_tokens':in_tok,'output_tokens':out_tok,'estimated_input_cost_usd':in_tok*0.042/1e6,'wall_seconds':None},
              'probing':summarize(rows_p,'probing',S,C,meta),'id_test':summarize(rows_t,'id_test',S,C,meta)}
    (out / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__': main()

