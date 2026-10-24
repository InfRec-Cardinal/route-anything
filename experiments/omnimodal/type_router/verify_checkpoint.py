"""Check actual task-classifier updates and adapter/head save-and-reload behavior."""
import json
from pathlib import Path
import time

import torch

from evaluate import predict_records


def capture_initial_state(model):
    return {
        'head': {name: value.detach().float().cpu().clone()
                 for name, value in model.capability_head.state_dict().items()},
        'adapter': {name: value.detach().float().cpu().clone()
                    for name, value in model.encoder.named_parameters()
                    if 'lora_B' in name and value.requires_grad},
    }


def compare_predictions(actual, expected):
    actual = {row['decision_id']: row for row in actual}
    expected = {row['decision_id']: row for row in expected}
    same_ids = actual.keys() == expected.keys()
    maximum = 0.0
    choices_equal = same_ids
    close = same_ids
    for identifier in actual.keys() & expected.keys():
        left, right = actual[identifier], expected[identifier]
        a = torch.tensor(list(left['capability_probabilities'].values()))
        b = torch.tensor(list(right['capability_probabilities'].values()))
        maximum = max(maximum, float((a - b).abs().max()))
        close = close and bool(torch.allclose(a, b, atol=1e-4, rtol=1e-3))
        choices_equal = choices_equal and all(left[key] == right[key]
                                              for key in ('predicted_capability_id', 'selected_model_id', 'choice'))
    return {'allclose': close, 'same_capability_and_model': choices_equal,
            'max_probability_difference': maximum, 'tasks': len(expected)}


def verify_checkpoint(model, checkpoint, rows, tokenizer, plan, table, device,
                      initial_state, cache=None, report_path=None):
    """Save and reload both trained components using train/dev tasks only."""
    from peft import PeftModel

    if any(row['split'] in ('test', 'special_test') for row in rows):
        raise ValueError('Checkpoint verification uses train/dev probes, not test tasks.')
    started = time.perf_counter()
    head = model.capability_head.state_dict()
    adapter = dict(model.encoder.named_parameters())
    head_update = max(float((head[name].detach().float().cpu() - before).abs().max())
                      for name, before in initial_state['head'].items())
    adapter_update = max((float((adapter[name].detach().float().cpu() - before).abs().max())
                          for name, before in initial_state['adapter'].items()), default=0.0)
    before = list(predict_records(model, rows, tokenizer, plan, table, device, cache))
    reordered = list(predict_records(model, list(reversed(rows)), tokenizer, plan, table, device, cache))
    permutation = compare_predictions(reordered, before)
    checkpoint = Path(checkpoint)
    model.save_checkpoint(checkpoint)
    unloaded = model.encoder.unload()
    model.encoder = PeftModel.from_pretrained(unloaded, checkpoint / 'adapter', is_trainable=False)
    with torch.no_grad():
        for parameter in model.capability_head.parameters():
            parameter.fill_(9)
    model.load_head(checkpoint)
    after = list(predict_records(model, rows, tokenizer, plan, table, device, cache))
    reloaded = compare_predictions(after, before)
    checks = {'head_updated': head_update > 0, 'lora_updated': adapter_update > 0,
              'task_order': permutation['allclose'] and permutation['same_capability_and_model'],
              'actual_adapter_and_head_reload': reloaded['allclose'] and reloaded['same_capability_and_model']}
    report = {'status': 'passed' if all(checks.values()) else 'failed',
              'checks': checks, 'head_max_update': head_update, 'adapter_max_update': adapter_update,
              'permutation': permutation, 'reload': reloaded, 'before_reload': before, 'after_reload': after,
              'seconds': time.perf_counter() - started, 'test_data_read': False}
    destination = Path(report_path) if report_path else checkpoint.parent / 'checkpoint-verification.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if not all(checks.values()):
        raise RuntimeError(f'Task-capability checkpoint verification failed: {checks}')
    return report
