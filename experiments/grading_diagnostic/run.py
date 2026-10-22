"""Recompute the BigCodeBench grading diagnostic from matrix and saved verdicts.

The default run reads the archived verdict CSV. Optional --recheck evaluates the
same saved task/model sample using separately supplied response records and the
shared subprocess sandbox. Neither mode makes model requests.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import sys


def flag(value):
    return str(value).lower() in {'true', '1'}


def summarize(matrix, verdicts):
    scores = {}
    with matrix.open() as stream:
        for row in csv.DictReader(stream):
            if row['task_id'].split('_')[1] == 'bcb':
                scores.setdefault(row['task_id'], []).append(float(row['score']))
    summary = {
        'scope': 'Archived BigCodeBench grading diagnostic; no new model requests.',
        'bcb_tasks': len(scores),
        'bcb_allzero_tasks': sum(max(values) <= 0 for values in scores.values()),
        'sample_design': 'Archived sample: 20 library-importing and 20 other all-zero tasks; 20 non-all-zero control tasks.',
        'original_sampling_seed': 20260917,
        'groups': {},
    }
    for name in ('allzero', 'ctrl_zero', 'ctrl_one'):
        rows = [row for row in verdicts if row['kind'] == name]
        available = [row for row in rows if not flag(row['empty'])]
        passed = sum(flag(row['passed']) for row in available)
        summary['groups'][name] = {
            'tasks': len({row['task_id'] for row in rows}),
            'responses': len(rows), 'nonempty_responses': len(available),
            'passed': passed, 'pass_fraction_nonempty': passed / len(available) if available else None,
        }
    summary['interpretation'] = (
        'The local official-test recheck is inconsistent with universal model failure on the all-zero subset. '
        'It supports treating these released zeros as an evaluation limitation; it does not identify the cause of every zero or replace their released scores.'
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument('--matrix', type=Path, default=Path(os.environ.get('CODEROUTERBENCH_MATRIX', 'data/agent-as-a-router/data/coderouterbench/id_results_long.csv')))
    parser.add_argument('--verdicts', type=Path, default=here / 'sandbox_rows.csv')
    parser.add_argument('--output', type=Path, default=here / 'summary.json')
    parser.add_argument('--recheck', action='store_true')
    parser.add_argument('--response-matrix', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    with args.verdicts.open() as stream:
        verdicts = list(csv.DictReader(stream))
    if args.recheck:
        if args.response_matrix is None:
            parser.error('--recheck requires --response-matrix')
        sys.path.insert(0, str(here.parents[1] / 'experiments_x'))
        from common import sources, sandbox
        responses = json.loads(args.response_matrix.read_text())
        tasks = {row['task_id']: sources.load_task(row['task_id'], strict=True) for row in verdicts}

        def evaluate(row):
            candidates = responses.get(row['task_id'], {})
            text = candidates.get(row['model'])
            if (text is None or not str(text).strip()) and row['model'] == 'Qwen3-Max':
                text = candidates.get('\u901a\u4e49\u5343\u95eeMax', '')
            text = text or ''
            result = sandbox.verify_response(tasks[row['task_id']], text)
            return {**row, 'empty': not bool(text.strip()), 'passed': bool(result['passed']),
                    'frac': result['frac'], 'failure_kind': result['failure_kind'],
                    'elapsed_s': result['elapsed_s']}

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            verdicts = list(executor.map(evaluate, verdicts))
        updated = args.output.with_name(args.output.stem + '_verdicts.csv')
        with updated.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(verdicts[0]))
            writer.writeheader()
            writer.writerows(verdicts)
    result = summarize(args.matrix, verdicts)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

