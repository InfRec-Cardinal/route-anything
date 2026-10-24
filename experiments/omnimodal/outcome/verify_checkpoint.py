"""Checkpoint verification on encoded source-train/dev probes, never test data.

Each entry is {id, input_ids, weights}; optional decision_id and arm_version_id
identify separate choice groups. Without decision_id, entries form one probe
decision. Weights are the frozen aggregation weights, not supervision masks.
The primary inference contract is micro1. Repacking at the training microbatch
is a measured sensitivity, not a checkpoint acceptance condition.
"""
from collections import defaultdict
import json
from pathlib import Path
import time

import torch


def capture_initial_state(model):
    """Call before training; LoRA_B movement demonstrates an updated adapter."""
    if model.mode != 'scalar':
        raise ValueError('Use capability_router_20261018/verify_checkpoint.py for task-capability models.')
    return {
        'head': {name: value.detach().float().cpu().clone()
                 for name, value in model.quality_head.state_dict().items()},
        'adapter': {name: value.detach().float().cpu().clone()
                    for name, value in model.encoder.named_parameters()
                    if 'lora_B' in name and value.requires_grad},
    }


def _inputs(entries, pad_token_id, device):
    width = max(len(entry['input_ids']) for entry in entries)
    return {
        'input_ids': torch.tensor([
            [pad_token_id] * (width - len(e['input_ids'])) + e['input_ids']
            for e in entries], dtype=torch.long, device=device),
        'attention_mask': torch.tensor([
            [0] * (width - len(e['input_ids'])) + [1] * len(e['input_ids'])
            for e in entries], dtype=torch.long, device=device),
    }


def predict_probe(model, entries, tokenizer, device, micro_batch=1):
    """Score every encoded pair once, preserving all dimension predictions."""
    if model.mode != 'scalar':
        raise ValueError('The old per-candidate capability verifier is retired; use the new task classifier verifier.')
    records = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(entries), micro_batch):
            chunk = entries[start:start + micro_batch]
            logits = model(**_inputs(chunk, tokenizer.pad_token_id, device)).float()
            if logits.ndim != 2 or logits.shape[0] != len(chunk):
                raise ValueError('The scorer must return [pairs, dimensions] logits')
            probabilities = logits.sigmoid()
            for entry, z, p in zip(chunk, logits, probabilities):
                values = p.cpu().tolist()
                if len(values) == 1:
                    aggregate = values[0]
                else:
                    if len(entry['weights']) != len(values):
                        raise ValueError('Frozen aggregation weights do not match the output dimensions')
                    # Match the primary evaluator's fixed CPU weighted sum.
                    aggregate = sum(w * probability for w, probability in zip(entry['weights'], values))
                records.append({
                    'id': entry['id'],
                    'decision_id': entry.get('decision_id', 'probe'),
                    'arm_version_id': entry.get('arm_version_id', entry['id']),
                    'logits': z.cpu().tolist(), 'probabilities': values,
                    'aggregate_score': float(aggregate),
                    'aggregation_weights': entry['weights'],
                    'selection_mask': entry.get('selection_mask', True),
                    'pair_tokens': len(entry['input_ids']),
                })
    return records


def _choices(records):
    groups = defaultdict(list)
    for record in records:
        group = groups[record['decision_id']]
        if record['selection_mask']:
            group.append(record)
    return {group: (min(items, key=lambda r: (-r['aggregate_score'], r['arm_version_id'], r['id']))['arm_version_id']
                    if items else None)
            for group, items in groups.items()}


def compare_predictions(actual, expected, atol=1e-3, rtol=1e-3):
    """Align by pair ID; report choice changes in the fixed aggregated score."""
    a, b = ({r['id']: r for r in records} for records in (actual, expected))
    if len(a) != len(actual) or len(b) != len(expected) or a.keys() != b.keys():
        raise ValueError('Probe pair IDs must be unique and identical across comparisons')
    same, logit_delta, probability_delta, score_delta = True, 0.0, 0.0, 0.0
    for identifier in b:
        left, right = a[identifier], b[identifier]
        z, ref = torch.tensor(left['logits']), torch.tensor(right['logits'])
        if z.shape != ref.shape:
            raise ValueError('Checkpoint probe output dimensions changed')
        same = same and bool(torch.allclose(z, ref, atol=atol, rtol=rtol))
        logit_delta = max(logit_delta, float((z - ref).abs().max()))
        probability_delta = max(probability_delta, max(abs(x - y) for x, y in
                                                        zip(left['probabilities'], right['probabilities'])))
        score_delta = max(score_delta, abs(left['aggregate_score'] - right['aggregate_score']))
    actual_choices, expected_choices = _choices(actual), _choices(expected)
    changes = [{'decision_id': key, 'expected': expected_choices[key], 'actual': actual_choices[key]}
               for key in expected_choices if actual_choices[key] != expected_choices[key]]
    return {'allclose': same, 'max_abs_logit_difference': logit_delta,
            'max_abs_probability_difference': probability_delta,
            'max_abs_aggregate_difference': score_delta,
            'expected_choices': expected_choices, 'actual_choices': actual_choices,
            'choice_changes': changes, 'choice_change_count': len(changes),
            'pair_count': len(expected)}


def _updates(model, initial_state):
    head = model.quality_head.state_dict()
    adapter = dict(model.encoder.named_parameters())
    head_changes = {name: float((head[name].detach().float().cpu() - before).abs().max())
                    for name, before in initial_state['head'].items()}
    adapter_changes = {name: float((adapter[name].detach().float().cpu() - before).abs().max())
                       for name, before in initial_state['adapter'].items()}
    return {'head_max_update': max(head_changes.values(), default=0.0),
            'adapter_max_update': max(adapter_changes.values(), default=0.0),
            'head_parameter_updates': head_changes,
            'tracked_adapter_parameters': len(adapter_changes),
            'adapter_tracking': 'all trainable LoRA_B tensors captured before training'}


def verify_checkpoint(model, checkpoint, entries, tokenizer, device, initial_state,
                      train_micro_batch=4, report_path=None, atol=1e-3, rtol=1e-3):
    """Save and actually reload PEFT+head, then enforce the fixed-shape contract.

    Leaves the model in eval mode with its saved adapter and head reloaded.
    Save the returned report; a failed mandatory check writes the same evidence
    before raising. No optimizer, training, source-data or test loader is used.
    """
    from peft import PeftModel

    if not entries:
        raise ValueError('Checkpoint verification needs a nonempty source probe')
    started = time.perf_counter()
    checkpoint = Path(checkpoint)
    report_path = Path(report_path) if report_path else checkpoint.parent / 'checkpoint-verification.json'
    updates = _updates(model, initial_state)
    before = predict_probe(model, entries, tokenizer, device, 1)
    reversed_entries = list(reversed(entries))
    reverse = predict_probe(model, reversed_entries, tokenizer, device, 1)
    permutation = compare_predictions(reverse, before, atol, rtol)
    training_shape = predict_probe(model, entries, tokenizer, device, train_micro_batch)
    training_reverse = predict_probe(model, reversed_entries, tokenizer, device, train_micro_batch)
    sensitivity = {
        'micro_batch': train_micro_batch, 'hard_gate': False,
        'repacked_order': compare_predictions(training_reverse, training_shape, atol, rtol),
        'versus_primary_micro1': compare_predictions(training_shape, before, atol, rtol),
        'forward': training_shape, 'reverse': training_reverse,
        'interpretation': 'Repacking can change padding, batch shape and numerics. These measurements do not establish invariance outside this source probe.',
    }
    model.save_checkpoint(checkpoint)
    unloaded = model.encoder.unload()
    model.encoder = PeftModel.from_pretrained(unloaded, checkpoint / 'adapter', is_trainable=False)
    # Deliberately overwrite the in-memory head: agreement must require loading
    # the saved head, not merely retaining the trained instance.
    with torch.no_grad():
        for parameter in model.quality_head.parameters():
            parameter.fill_(9)
    model.load_head(checkpoint)
    after = predict_probe(model, entries, tokenizer, device, 1)
    reload_check = compare_predictions(after, before, atol, rtol)
    mandatory = {'head_updated': updates['head_max_update'] > 0,
                 'lora_updated': updates['adapter_max_update'] > 0,
                 'same_shape_micro1_permutation': permutation['allclose'] and not permutation['choice_changes'],
                 'actual_adapter_and_head_reload': reload_check['allclose'] and not reload_check['choice_changes']}
    report = {
        'status': 'passed' if all(mandatory.values()) else 'failed',
        'checkpoint': str(checkpoint), 'mandatory_checks': mandatory, **updates,
        'primary_eval_micro_batch': 1,
        'tolerance': {'atol': atol, 'rtol': rtol},
        'primary_choice_contract': 'Same-shape permutation and reload must preserve the frozen aggregated choice as well as satisfy logit tolerance.',
        'same_shape_permutation': permutation, 'actual_reload': reload_check,
        'before_reload': before, 'after_reload': after,
        'training_shape_sensitivity': sensitivity,
        'verification_actual_pair_tokens': sum(r['pair_tokens'] for part in
                                              (before, reverse, training_shape, training_reverse, after) for r in part),
        'verification_seconds': time.perf_counter() - started,
        'test_data_read': False,
        'limitations': 'The primary micro1 inference order and aggregation are frozen. Passing source probes does not prove full-data numerical invariance; different-shape sensitivity is reported without changing the acceptance tolerance.',
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if not all(mandatory.values()):
        raise RuntimeError(f'Checkpoint mandatory verification failed: {mandatory}')
    return report
