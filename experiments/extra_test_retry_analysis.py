"""Rescore archived paired HumanEval retries on the unused EvalPlus suite.

No model requests are made. Generated programs run only through the existing
subprocess sandbox. EvalPlus overlaps the base tests, so this is a supplementary
suite evaluation, not a disjoint holdout or a new online experiment.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np


def literal_inputs(test):
    tree = ast.parse(test)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inputs' for t in node.targets):
            return ast.literal_eval(node.value)
    return []


def literal_base_calls(test, entry):
    calls = []
    for node in ast.walk(ast.parse(test)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in (entry, 'candidate'):
            try:
                calls.append([ast.literal_eval(arg) for arg in node.args])
            except (ValueError, TypeError):
                pass
    return calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, default=Path('./experiments_x'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.json'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.experiment_root))
    from common import sources, sandbox

    rows = [json.loads(line) for line in (args.experiment_root / 'X7b_inloop_live/attempts_view_a_6144.jsonl').read_text().splitlines()]
    by_job = {}
    for row in rows:
        if row['source'] in ('humaneval', 'he'):
            by_job.setdefault(row['job_id'], {})[(row['branch'], row['k'])] = row
    jobs = {key: value for key, value in by_job.items()
            if ('fresh', 2) in value and ('retry', 2) in value and not value[('root', 1)]['truth_pass']}
    connection = sqlite3.connect('file:' + str(args.experiment_root / 'common/cache/packy_cache.sqlite') + '?mode=ro', uri=True)
    tasks = {}
    items = []
    for job_id, attempts in sorted(jobs.items()):
        task = sources.load_task(attempts[('root', 1)]['task_id'])
        plus = task['meta']['tests_plus']
        assert task['meta']['tests_source'] == 'HumanEval base tests (humaneval-x Python)'
        assert plus != task['tests']
        if task['task_id'] not in tasks:
            plus_inputs = literal_inputs(plus)
            base_inputs = literal_base_calls(task['tests'], task['entry_point'])
            overlap = sum(any(inp == base for base in base_inputs) for inp in plus_inputs)
            canonical = task['raw_prompt'] + task['reference']
            reference_check = sandbox.verify_response({**task, 'tests': plus}, '```python\n' + canonical + '\n```')
            tasks[task['task_id']] = {
                'source': task['source'], 'base_source': task['meta']['tests_source'],
                'plus_source': 'evalplus/humanevalplus cached test field',
                'plus_cases': len(plus_inputs), 'base_literal_calls': len(base_inputs),
                'plus_cases_matching_literal_base_calls': overlap,
                'overlap_method': 'AST literal calls only; does not enumerate computed or loop-generated base inputs',
                'canonical_pass': bool(reference_check['passed']), 'canonical_error': reference_check['error'],
            }
        if not tasks[task['task_id']]['canonical_pass']:
            continue
        for branch, stage in [('root', 1), ('fresh', 2), ('retry', 2)]:
            row = attempts[(branch, stage)]
            request, response = connection.execute('select request,response from cache where key=?', (row['cache_key'],)).fetchone()
            request = json.loads(request)
            response = json.loads(response)
            message = response['choices'][0]['message']['content'] or ''
            assert isinstance(message, str)
            messages = request['messages']
            request_text = '\n'.join(str(m['content']) for m in messages)
            assert plus not in request_text
            items.append((job_id, row, {**task, 'tests': plus}, message, {
                'message_count': len(messages), 'full_plus_suite_in_request': False,
                'request_feedback_present': 'Your solution failed the tests.' in request_text,
                'cached_response_chars': len(message), 'logged_response_chars': len(row.get('text', '')),
            }))
    connection.close()

    def score(item):
        job_id, row, task, message, request_audit = item
        verdict = sandbox.verify_response(task, message)
        return {'job_id': job_id, 'task_id': row['task_id'], 'root_model': row['root_model'],
                'source': row['source'], 'branch': row['branch'], 'k': row['k'],
                'base_pass': bool(row['truth_pass']), 'plus_pass': bool(verdict['passed']),
                'plus_fraction': verdict['frac'], 'failure_kind': verdict['failure_kind'],
                'timed_out': verdict['timed_out'], 'evaluation_seconds': verdict['elapsed_s'],
                'error': verdict['error'], 'request_audit': request_audit}

    scored = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(score, item) for item in items]
        for future in as_completed(futures):
            scored.append(future.result())
            print(f'Rescored {len(scored)}/{len(items)}', flush=True)
    scored.sort(key=lambda row: (row['job_id'], row['k'], row['branch']))
    pairs = []
    for job_id in sorted({row['job_id'] for row in scored}):
        group = {r['branch']: r for r in scored if r['job_id'] == job_id}
        pairs.append({'job_id': job_id, 'task_id': group['root']['task_id'],
                      'source': group['root']['source'], 'root_model': group['root']['root_model'],
                      'base_root': group['root']['base_pass'], 'plus_root': group['root']['plus_pass'],
                      'base_fresh': group['fresh']['base_pass'], 'base_retry': group['retry']['base_pass'],
                      'plus_fresh': group['fresh']['plus_pass'], 'plus_retry': group['retry']['plus_pass']})
    rng = np.random.default_rng(20260923)
    summary = {}
    for model in ['pooled'] + sorted({p['root_model'] for p in pairs}):
        p = pairs if model == 'pooled' else [p for p in pairs if p['root_model'] == model]
        delta = np.array([int(x['plus_retry']) - int(x['plus_fresh']) for x in p])
        task_ids = sorted({x['task_id'] for x in p})
        totals = np.array([sum(int(x['plus_retry']) - int(x['plus_fresh']) for x in p if x['task_id'] == tid) for tid in task_ids])
        counts = np.array([sum(x['task_id'] == tid for x in p) for tid in task_ids])
        sampled = rng.integers(0, len(task_ids), size=(10000, len(task_ids)))
        boot = totals[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
        summary[model] = {'n_pairs': len(p), 'base_fresh_passes': sum(x['base_fresh'] for x in p),
                          'base_retry_passes': sum(x['base_retry'] for x in p),
                          'plus_fresh_passes': sum(x['plus_fresh'] for x in p),
                          'plus_retry_passes': sum(x['plus_retry'] for x in p),
                          'plus_root_passes': sum(x['plus_root'] for x in p),
                          'plus_paired_difference': float(delta.mean()),
                          'paired_bootstrap_unit': 'task_id (all model pairs on the sampled task retained)',
                          'paired_bootstrap_95_interval': np.quantile(boot, [.025, .975]).tolist()}
    output = {'scope': 'Archived first-failure HumanEval/he pairs with both fresh and feedback retry outputs.',
              'interpretation': 'EvalPlus was not used to produce the experiment feedback; it overlaps base tests and is not a disjoint holdout. Coverage is small and does not support replacing the five-model overall result.',
              'selection': 'Headline budget-matched view; source humaneval/he; first attempt failed base tests; both second attempts recorded.',
              'model_calls': 0, 'n_candidate_pairs': len(jobs), 'n_candidate_tasks': len(tasks),
              'n_pairs': len(pairs), 'n_unique_tasks': len({p['task_id'] for p in pairs}),
              'excluded_tasks': {key: value['canonical_error'] for key, value in tasks.items() if not value['canonical_pass']},
              'n_rescored_outputs': len(scored), 'summary': summary, 'tasks': tasks,
              'pairs': pairs, 'scores': scored}
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

