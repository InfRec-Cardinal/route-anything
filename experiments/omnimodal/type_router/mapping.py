"""Fit a single capability-to-model lookup table from training observations."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

from model_policy import allowed, candidate_model, POLICY_VERSION, EXCLUDED_MODEL_IDS

from common import SCHEMA_VERSION


MODEL_PRIORITY = {'claude-opus-5-5': 0, 'gpt-6-astra': 1}


def model_id(candidate):
    return candidate_model(candidate)


def valid_quality(candidate):
    return (candidate.get('observed_mask') is True
            and allowed(model_id(candidate))
            and candidate.get('quality_valid_mask') is True
            and candidate.get('quality') is not None
            and candidate.get('selection_mask', True) is not False
            and candidate.get('capability_mask', True) is not False)


def mean_metric(candidates, field, valid_field, scope_field):
    scope = candidates[0].get(scope_field)
    if not scope or not all(c.get(valid_field) is True and c.get(field) is not None
                            and c.get(scope_field) == scope for c in candidates):
        return None, None
    return sum(c[field] for c in candidates) / len(candidates), scope


def model_observations(row):
    """One observation per decision/model; average frozen arms, never best-of."""
    grouped = defaultdict(list)
    for candidate in row['candidates']:
        if valid_quality(candidate) and model_id(candidate):
            grouped[model_id(candidate)].append(candidate)
    for name, candidates in grouped.items():
        cost, cost_scope = mean_metric(candidates, 'cost_usd', 'cost_valid_mask', 'cost_scope')
        tokens, token_scope = mean_metric(candidates, 'total_tokens', 'token_valid_mask', 'token_scope')
        yield name, {
            'decision_id': row['decision_id'], 'group_id': row['group_id'],
            'benchmark': row['benchmark'], 'metric_id': row.get('metric_id'),
            'scalar_spec_id': row.get('scalar_spec_id'),
            'quality': sum(c['quality'] for c in candidates) / len(candidates),
            'cost_usd': cost, 'cost_scope': cost_scope,
            'total_tokens': tokens, 'token_scope': token_scope,
            'arm_version_ids': sorted({c.get('arm_version_id', c.get('arm_id')) for c in candidates}),
        }


def weighted_mean(records, field):
    return sum(r['weight'] * r[field] for r in records) / sum(r['weight'] for r in records)


def summarize(name, records):
    by_benchmark = defaultdict(list)
    for record in records:
        by_benchmark[record['benchmark']].append(record)
    summaries = []
    for benchmark, rows in sorted(by_benchmark.items()):
        item = {
            'benchmark': benchmark, 'metric_ids': sorted({r['metric_id'] for r in rows}),
            'scalar_spec_ids': sorted({r['scalar_spec_id'] for r in rows if r['scalar_spec_id']}),
            'n': len(rows), 'group_count': len({r['group_id'] for r in rows}),
            'weight_sum': sum(r['weight'] for r in rows),
            'mean_quality': weighted_mean(rows, 'quality'),
            'arm_version_ids': sorted({a for r in rows for a in r['arm_version_ids']}),
        }
        for field, scope_field in [('cost_usd', 'cost_scope'), ('total_tokens', 'token_scope')]:
            scopes = {r[scope_field] for r in rows}
            complete = all(r[field] is not None for r in rows) and len(scopes) == 1 and None not in scopes
            item['mean_' + field] = weighted_mean(rows, field) if complete else None
            item[scope_field] = next(iter(scopes)) if complete else None
        summaries.append(item)
    result = {
        'model_id': name,
        'mean_quality': sum(s['mean_quality'] for s in summaries) / len(summaries),
        'benchmark_count': len(summaries), 'benchmarks': [s['benchmark'] for s in summaries],
        'n': len(records), 'group_count': len({r['group_id'] for r in records}),
        'weight_sum': sum(r['weight'] for r in records), 'by_benchmark': summaries,
        # Retain the exact evidence to expose unequal coverage and support tie comparisons.
        'decision_ids': sorted(r['decision_id'] for r in records),
    }
    for field in ['cost_usd', 'total_tokens']:
        values = [s['mean_' + field] for s in summaries]
        result['mean_' + field] = sum(values) / len(values) if all(v is not None for v in values) else None
    return result


def comparable_minimum(models, field, scope_field):
    """Aggregate spend is comparable only on the same task support and scopes."""
    first = models[0]
    support = first['decision_ids']
    scopes = [(b['benchmark'], b[scope_field]) for b in first['by_benchmark']]
    if any(scope is None for _, scope in scopes) or not all(
            m['decision_ids'] == support and m['mean_' + field] is not None
            and [(b['benchmark'], b[scope_field]) for b in m['by_benchmark']] == scopes
            for m in models):
        return models
    minimum = min(m['mean_' + field] for m in models)
    return [m for m in models if m['mean_' + field] == minimum]


def choose_model(models):
    maximum = max(m['mean_quality'] for m in models)
    best = [m for m in models if m['mean_quality'] == maximum]
    if len(best) == 1:
        return best[0], 'unique_quality'
    best = comparable_minimum(best, 'cost_usd', 'cost_scope')
    if len(best) == 1:
        return best[0], 'cost'
    best = comparable_minimum(best, 'total_tokens', 'token_scope')
    if len(best) == 1:
        return best[0], 'tokens'
    priority = min(MODEL_PRIORITY.get(m['model_id'].split('/')[-1], 2) for m in best)
    best = [m for m in best if MODEL_PRIORITY.get(m['model_id'].split('/')[-1], 2) == priority]
    return min(best, key=lambda m: m['model_id']), 'model_priority' if priority < 2 and len(best) == 1 else 'stable_order'


def build_table(train_rows, dimensions, model_pools=None):
    if model_pools is not None:
        model_pools = {family: [name for name in names if allowed(name)]
                       for family, names in model_pools.items()}
    evidence = defaultdict(lambda: defaultdict(list))
    support = defaultdict(list)
    input_rows = 0
    for row in train_rows:
        if row['split'] != 'train':
            raise ValueError('The fixed table accepts only optimization train rows.')
        if 'critpt' in row['benchmark'].lower():
            raise ValueError('CritPt belongs only in the special test set.')
        input_rows += 1
        observations = list(model_observations(row))
        if model_pools is not None:
            family_pool = set(model_pools.get(row.get('output_family'), []))
            observations = [(name, observation) for name, observation in observations if name in family_pool]
        for capability, weight in row['capability_targets'].items():
            if weight <= 0:
                continue
            support[capability].append(row)
            for name, observation in observations:
                evidence[capability][name].append(dict(observation, weight=weight))
    entries = []
    for dimension in dimensions:
        capability = dimension['id']
        models = [summarize(name, rows) for name, rows in sorted(evidence[capability].items())]
        rows = support[capability]
        selected, resolution = choose_model(models) if models else (None, 'no_valid_training_quality')
        entries.append({
            'benchmark': None, 'metric_id': None, 'capability_id': capability,
            'base_capability_id': dimension.get('base_capability_id', capability),
            'output_family': dimension.get('output_family'),
            'model_id': selected['model_id'] if selected else None,
            'mean_quality': selected['mean_quality'] if selected else None,
            'resolution': resolution, 'labeled_train_n': len(rows),
            'labeled_train_group_count': len({r['group_id'] for r in rows}),
            'labeled_train_benchmarks': sorted({r['benchmark'] for r in rows}),
            'model_count': len(models), 'model_statistics': models,
        })
    return {
        'schema_version': SCHEMA_VERSION, 'scope': 'global', 'source_split': 'train',
        'model_exclusion_policy': POLICY_VERSION, 'excluded_model_ids': EXCLUDED_MODEL_IDS,
        'calibration_used': False, 'dev_used': False, 'test_used': False,
        'training_row_count': input_rows,
        'output_family_model_pools': model_pools,
        'aggregation': {
            'quality': 'canonical scalar quality; capability-weighted mean within each benchmark, then equal-weight mean of observed benchmarks for each model',
            'capability_typing': 'Each global capability key identifies its output execution family; only training-defined compatible models compete within that key.',
            'multiple_arms_same_decision_model': 'arithmetic mean of observed valid frozen arms; no best-arm selection',
            'missing': 'unknown observations omitted; no zero imputation; model benchmark/task coverage may differ',
            'interpretation': 'empirical maximum on recorded training coverage, not a claim of universal ability or full shared coverage; canonical metrics differ across benchmarks',
            'ties': ['quality', 'comparable mean cost', 'comparable mean total input/output tokens', 'claude-opus-5-5', 'gpt-6-astra', 'stable model order'],
            'cost_token_comparability': 'complete values, identical decision support, and matching scope within every benchmark; same capability weights and benchmark macro aggregation as quality',
        },
        'coverage': {'capability_count': len(entries), 'mapped_count': sum(e['model_id'] is not None for e in entries),
                     'unknown_capability_ids': [e['capability_id'] for e in entries if e['model_id'] is None]},
        'entries': entries,
    }


def write_table_markdown(path, table, dimensions):
    names = {dimension['id']: dimension['name'] for dimension in dimensions}
    lines = [
        '# 全项目固定能力→模型表', '',
        '仅使用冻结 optimization train 的已观察有效质量；CritPt、calibration、dev、test 均不参与建表。', '',
        '能力 ID 包含输出执行家族；每个能力在全项目固定选择一个模型。分数先按能力权重计算各 benchmark 均值，再将该模型有记录的 benchmark 等权平均。各模型覆盖不同，分数不代表完整共同面板或普适能力。', '',
        'n 是该获选模型在此能力下的有效训练 decision 数。完整的逐 benchmark 权重、样本和 arm 证据保存在 model-table.json。', '',
        '| 能力 | 输出执行家族 | 固定模型 | train n | 宏平均质量 |',
        '| --- | --- | --- | ---: | ---: |',
    ]
    for entry in table['entries']:
        selected = next((model for model in entry['model_statistics'] if model['model_id'] == entry['model_id']), None)
        score = f"{entry['mean_quality']:.6f}" if entry['mean_quality'] is not None else '未知'
        lines.append(f"| {names[entry['capability_id']]} | {entry.get('output_family') or '未类型化'} | {entry['model_id'] or '未知'} | {selected['n'] if selected else 0} | {score} |")
    Path(path).write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True, type=Path)
    parser.add_argument('--taxonomy', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    taxonomy = json.loads(args.taxonomy.read_text())
    dimensions = taxonomy['dimensions']
    pools = {family['id']: family['model_ids'] for family in taxonomy.get('families', [])} or None
    table = build_table((json.loads(line) for line in args.train.open()), dimensions, model_pools=pools)
    table['source_train_path'] = str(args.train.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, ensure_ascii=False, indent=2) + '\n')
    write_table_markdown(args.output.with_suffix('.md'), table, dimensions)
    print(json.dumps(table['coverage'], ensure_ascii=False))


if __name__ == '__main__':
    main()
