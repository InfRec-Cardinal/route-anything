"""Paper leaf-table fit, extracted from the original preparation code.

Leaf quality contrasts retain the original paired least-squares computation.
Benchmark fixed arms use the revised paper observed-training-mean definition.
"""
import argparse
from collections import defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
from model_policy import filter_record, POLICY_VERSION, EXCLUDED_MODEL_IDS

PRIORITY = {"claude-opus-5-5": 0, "gpt-6-astra": 1}

def valid_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean_comparable(items, field, scope_field):
    scopes = {item.get(scope_field) for item in items}
    if len(scopes) == 1 and None not in scopes and all(valid_number(item.get(field)) for item in items):
        return {'value': sum(item[field] for item in items) / len(items), 'scope': next(iter(scopes))}
    return {'value': None, 'scope': None}


def paired_choice(records):
    """Estimate relative quality using within-identical-input model contrasts."""
    import numpy as np
    repeated = defaultdict(list)
    for row in records:
        row = filter_record(row)
        key = (row['group_id'], row['task_id'], row['task_text'], tuple(row['images']), row['state_text'],
               json.dumps(row.get('output_contract', {}), sort_keys=True))
        repeated[key].append(row)
    observations = []
    for members in repeated.values():
        by_model = defaultdict(list)
        for row in members:
            for name, quality in row['model_scores'].items():
                by_model[name].append((quality, row['model_costs'].get(name, {}), row['model_tokens'].get(name, {})))
        entry = {}
        for name, values in by_model.items():
            entry[name] = {'quality': sum(x[0] for x in values) / len(values)}
            for slot, field in ((1, 'cost'), (2, 'tokens')):
                parts = [{'value': x[slot].get('value'), 'scope': x[slot].get('scope')} for x in values]
                entry[name][field] = mean_comparable(parts, 'value', 'scope')
        observations.append(entry)
    models = sorted({name for item in observations for name in item})
    evidence = {'training_records': len(records), 'unique_visible_inputs': len(observations),
                'models': models, 'paired_inputs': sum(len(x) > 1 for x in observations)}
    if not models:
        return None, dict(evidence, status='no_observed_train_quality')
    if len(models) == 1:
        return models[0], dict(evidence, status='single_observed_model', comparison_verified=False)
    adjacency = {name: set() for name in models}
    matrix, target = [], []
    position = {name: i for i, name in enumerate(models)}
    for item in observations:
        pairs = list(combinations(sorted(item), 2))
        for left, right in pairs:
            adjacency[left].add(right); adjacency[right].add(left)
            weight = 1 / math.sqrt(len(pairs))
            line = np.zeros(len(models))
            line[position[left]], line[position[right]] = weight, -weight
            matrix.append(line); target.append((item[left]['quality'] - item[right]['quality']) * weight)
    visited, pending = set(), [models[0]]
    while pending:
        name = pending.pop()
        if name not in visited:
            visited.add(name); pending.extend(adjacency[name] - visited)
    if len(visited) != len(models):
        return None, dict(evidence, status='disconnected_same_input_comparisons')
    matrix.append(np.ones(len(models))); target.append(0.)
    values = np.linalg.lstsq(np.asarray(matrix), np.asarray(target), rcond=None)[0]
    best = [name for name, value in zip(models, values) if abs(value - max(values)) <= 1e-10]
    evidence['relative_quality'] = dict(zip(models, map(float, values)))
    evidence['tie_break'] = []
    for field in ('cost', 'tokens'):
        if len(best) == 1:
            break
        common = [item for item in observations if set(best) <= set(item)]
        comparable = bool(common) and all(len({item[name][field]['scope'] for name in best}) == 1
                       and all(item[name][field]['value'] is not None and item[name][field]['scope'] is not None for name in best)
                       for item in common)
        if comparable:
            means = {name: sum(item[name][field]['value'] for item in common) / len(common) for name in best}
            best = [name for name in best if abs(means[name] - min(means.values())) <= 1e-10]
            evidence['tie_break'].append({'field': field, 'same_input_count': len(common), 'means': means})
    winner = min(best, key=lambda name: (PRIORITY.get(name.split('/')[-1], 2), name))
    return winner, dict(evidence, status='same_input_paired_fit', comparison_verified=True)



def train_mean_choice(records):
    scores = defaultdict(list)
    for row in records:
        for name, value in filter_record(row)["model_scores"].items():
            if value is not None:
                scores[name].append(value)
    means = {name: sum(values) / len(values) for name, values in scores.items()}
    selected = min(means, key=lambda name: (-means[name], PRIORITY.get(name.split("/")[-1], 2), name)) if means else None
    return selected, {"mean_quality": means, "observed_training_rows": {name: len(values) for name, values in scores.items()},
                      "selection": "Observed train mean; missing rows do not disqualify an arm."}

def fit_table(records, tree):
    by_leaf, by_benchmark = defaultdict(list), defaultdict(list)
    for row in records:
        if row['split'] != 'train':
            continue
        by_benchmark[row['benchmark']].append(row)
        if row.get('leaf_id'):
            by_leaf[row['leaf_id']].append(row)
    leaf_map = {leaf['leaf_id']: None for track in tree['tracks'] for task in track['tasks'] for leaf in task['capabilities']}
    bench_map, leaf_evidence, benchmark_evidence = {}, {}, {}
    for leaf, subset in sorted(by_leaf.items()):
        leaf_map[leaf], leaf_evidence[leaf] = paired_choice(subset)
    for bench, subset in sorted(by_benchmark.items()):
        bench_map[bench], benchmark_evidence[bench] = train_mean_choice(subset)
    return {'schema': 'capability-tree-global-model-table-v1', 'scope': 'global',
            'model_exclusion_policy': POLICY_VERSION, 'excluded_model_ids': EXCLUDED_MODEL_IDS,
            'shared_across_matrix': True, 'fit_split': 'train',
            'method': 'Same-input repeat mean, then task-equal weighted least-squares model quality differences; disconnected comparisons remain unassigned.',
            'tie_break_order': ['quality', 'comparable_cost', 'comparable_tokens', 'claude-opus-5-5', 'gpt-6-astra', 'stable_model_order'],
            'leaf_to_model': leaf_map, 'benchmark_to_model': bench_map,
            'leaf_evidence': leaf_evidence, 'benchmark_evidence': benchmark_evidence}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.train.read_text().splitlines() if line.strip()]
    if any(row["split"] != "train" for row in records):
        raise ValueError("--train must contain training rows only.")
    result = fit_table(records, json.loads(args.tree.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
