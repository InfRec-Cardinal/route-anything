"""Predict one task capability, then route through the frozen capability table."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

from common import lookup_model, task_payload
from model_policy import filter_record, require_current_table


def read_rows(path):
    return [filter_record(json.loads(line)) for line in Path(path).open() if line.strip()]


def encode_task(row, tokenizer, plan):
    messages = [
        {'role': 'system', 'content': 'Identify the capability required by this task from the supplied '
                                     'candidate capabilities. Predict one capability ID.'},
        {'role': 'user', 'content': json.dumps(task_payload(row, plan['dimensions']),
                                              ensure_ascii=False, sort_keys=True, separators=(',', ':'))},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > plan['max_sequence']:
        task = row.get('decision_id') or row.get('task_id') or 'unlabelled input'
        raise ValueError(f"Task {task} uses {len(ids)} tokens, exceeding max_sequence={plan['max_sequence']}; "
                         'increase the prepared sequence limit. Capability definitions are not truncated.')
    return ids


def candidate_model(candidate):
    profile = candidate.get('profile') or {}
    return candidate.get('model_id') or profile.get('model') or profile.get('model_id')


def replay_lookup(row, capability_id, table):
    """Replay only the table-selected model. Missing evidence never selects another model."""
    entry = lookup_model(table, row['benchmark'], row.get('metric_id'), capability_id)
    model_id = entry.get('model_id') if entry else None
    result = {'selected_model_id': model_id, 'choice': None, 'selected_quality': None,
              'quality': None, 'quality_valid': False}
    if entry is None:
        return {**result, 'routing_status': 'capability_not_in_table'}
    if model_id is None:
        return {**result, 'routing_status': 'no_training_evidence_for_capability'}
    matching = [candidate for candidate in row.get('candidates', [])
                if candidate_model(candidate) == model_id
                and candidate.get('selection_mask', True)
                and candidate.get('capability_mask') is not False
                and candidate.get('availability_mask') is not False]
    if not matching:
        return {**result, 'routing_status': 'model_not_in_candidate_panel'}
    if len(matching) > 1:
        return {**result, 'routing_status': 'multiple_model_configurations'}
    candidate = matching[0]
    arm_id = candidate.get('arm_version_id') or candidate.get('arm_id')
    result['choice'] = arm_id
    if candidate.get('quality_valid_mask') is not True or candidate.get('quality') is None:
        return {**result, 'routing_status': 'selected_quality_unobserved'}
    return {**result, 'routing_status': 'observed', 'quality_valid': True,
            'selected_quality': candidate['quality'], 'quality': candidate['quality']}


def predict_tasks(model, rows, tokenizer, plan, table, device, cache=None):
    """The online path: one forward per task and one global table lookup."""
    import torch
    from model import padded_inputs

    dimensions = plan['dimensions']
    model.eval()
    with torch.no_grad():
        for row in rows:
            ids = cache[row['decision_id']] if cache is not None else encode_task(row, tokenizer, plan)
            logits = model(**padded_inputs([ids], tokenizer.pad_token_id, device)).float()[0]
            probabilities = logits.softmax(-1).cpu().tolist()
            index = int(logits.argmax())
            capability_id = dimensions[index]['id']
            entry = lookup_model(table, row['benchmark'], row.get('metric_id'), capability_id)
            yield {'predicted_capability_id': capability_id,
                   'selected_model_id': entry.get('model_id') if entry else None,
                   'lookup_status': ('mapped' if entry and entry.get('model_id') is not None else
                                     'no_training_evidence' if entry else 'capability_not_in_table'),
                   'capability_probabilities': {dimension['id']: probability
                                                for dimension, probability in zip(dimensions, probabilities)},
                   'input_tokens': len(ids)}


def predict_records(model, rows, tokenizer, plan, table, device, cache=None):
    predictions = predict_tasks(model, rows, tokenizer, plan, table, device, cache)
    for row, prediction in zip(rows, predictions):
        capability_id = prediction['predicted_capability_id']
        targets = row.get('capability_targets', {})
        best_ids = [key for key, weight in targets.items() if weight == max(targets.values())] if targets else []
        yield {
            'decision_id': row['decision_id'], 'task_id': row['task_id'],
            'group_id': row.get('group_id', row['task_id']), 'split': row['split'],
            'benchmark': row['benchmark'], 'metric_id': row.get('metric_id'),
            **prediction,
            'target_capability_id': row.get('target_capability_id'),
            'max_weight_capability_ids': best_ids,
            'capability_match': capability_id in best_ids if best_ids else None,
            'primary_capability_match': (capability_id == row['target_capability_id']
                                         if row.get('target_capability_id') is not None else None),
            'selected_target_weight': targets.get(capability_id, 0.0) if targets else None,
            **replay_lookup(row, capability_id, table),
        }


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row['benchmark'], row.get('metric_id'))].append(row)
    output = []
    for (benchmark, metric), rows in sorted(groups.items(), key=lambda item: str(item[0])):
        known = [row['quality'] for row in rows if row['quality_valid']]
        matches = [row['capability_match'] for row in rows if row['capability_match'] is not None]
        output.append({'benchmark': benchmark, 'metric_id': metric, 'tasks': len(rows),
                       'capability_accuracy': sum(matches) / len(matches) if matches else None,
                       'capability_labelled_count': len(matches),
                       'known_selected_quality_mean': sum(known) / len(known) if known else None,
                       'known_selected_quality_count': len(known), 'quality_coverage': len(known) / len(rows),
                       'routing_status_counts': dict(Counter(row['routing_status'] for row in rows))})
    return output


def evaluate(model, rows, tokenizer, plan, table, device, path, cache=None):
    started = time.perf_counter()
    records = []
    with Path(path).open('w') as stream:
        for row in predict_records(model, rows, tokenizer, plan, table, device, cache):
            records.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    return {'predictions': str(path), 'tasks': len(records),
            'input_tokens': sum(row['input_tokens'] for row in records),
            'seconds': time.perf_counter() - started, 'benchmarks': summarize(records),
            'quality_population': 'Only observed outcomes of the fixed table-selected model; coverage and '
                                  'missing/ambiguous configurations are reported, with no alternative selection.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=['train', 'dev', 'test', 'special_test'], default='test')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    if plan.get('schema_version') != 'task-capability-router-v1':
        raise ValueError('Run the new capability-router prepare step first; old quality-scoring plans are incompatible.')
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from model import CapabilityClassifier

    checkpoint = Path(args.checkpoint)
    table = require_current_table(json.loads((checkpoint / 'routing_table.json').read_text()))
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        plan['model_path'], dtype=torch.bfloat16 if args.device.startswith('cuda') else torch.float32,
        attn_implementation='sdpa', local_files_only=False)
    model = CapabilityClassifier.load_checkpoint(checkpoint, base).to(args.device)
    if model.dimensions != plan['dimensions']:
        raise ValueError('The plan taxonomy differs from the trained classifier taxonomy.')
    tokenizer = AutoTokenizer.from_pretrained(plan['model_path'], local_files_only=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = Path(plan[args.split + '_path'])
    rows_path = rows_path if rows_path.is_absolute() else plan_path.parent / rows_path
    report = evaluate(model, read_rows(rows_path), tokenizer, plan, table,
                      args.device, output / 'predictions.jsonl')
    (output / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
