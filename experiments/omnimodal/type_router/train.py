"""Train task capability classification, followed by a frozen global model lookup."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time

from common import SCHEMA_VERSION
from evaluate import encode_task, evaluate, read_rows
from model_policy import require_current_table


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def load_plan(path):
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    if plan.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Old candidate-quality plans cannot train the task-capability classifier. '
                         'Run capability_router_20261018/prepare.py to build a new plan, then this train.py '
                         'with a new output directory. Old quality-head checkpoints are incompatible.')
    for key in ('train_path', 'dev_path', 'test_path', 'special_test_path', 'table_path'):
        if key not in plan:
            continue
        value = Path(plan[key])
        plan[key] = str(value if value.is_absolute() else path.parent / value)
    return plan


def supervised_training_rows(rows):
    """Retain all data rows on disk; only labelled tasks enter optimizer batches."""
    supervised = [row for row in rows if row.get('capability_targets')
                  and row.get('capability_supervision_valid', True)]
    return supervised, {'total_rows': len(rows), 'supervised_rows': len(supervised),
                        'masked_rows': len(rows) - len(supervised)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--smoke', action='store_true', help='Small train/dev-only path check; no test evaluation')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    plan = load_plan(args.plan)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose a new empty output directory for this task-capability experiment; '
                         'historical candidate-quality results are not reusable as classifier results.')
    table = require_current_table(json.loads(Path(plan['table_path']).read_text()))
    if table.get('scope') != 'global':
        raise ValueError('The capability classifier requires the project-wide global lookup table.')

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from loss import capability_loss, capability_targets
    from model import CapabilityClassifier, padded_inputs
    from verify_checkpoint import capture_initial_state, verify_checkpoint

    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    train_rows = read_rows(plan['train_path'])
    dev_rows = read_rows(plan['dev_path'])
    supervised_train, train_counts = supervised_training_rows(train_rows)
    if args.smoke:
        supervised_train, dev_rows = supervised_train[:4], dev_rows[:4]
    train_counts['used_supervised_rows'] = len(supervised_train)
    if not supervised_train:
        raise ValueError('The prepared training set contains no capability labels.')
    tokenizer = AutoTokenizer.from_pretrained(plan['model_path'], local_files_only=False)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    cache_rows = (supervised_train if args.smoke else train_rows) + dev_rows
    cache = {row['decision_id']: encode_task(row, tokenizer, plan) for row in cache_rows}
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        plan['model_path'], dtype=torch.bfloat16 if args.device.startswith('cuda') else torch.float32,
        attn_implementation='sdpa', local_files_only=False)
    base.config.use_cache = False
    targets = [name for name, module in base.named_modules()
               if isinstance(module, torch.nn.Linear) and 'language_model.layers.' in name]
    encoder = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                                             target_modules=targets, task_type='CAUSAL_LM', bias='none'))
    model = CapabilityClassifier(encoder, base.config.text_config.hidden_size, plan['dimensions']).to(args.device)
    initial = capture_initial_state(model)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.encoder.enable_input_require_grads()
    optimizer = torch.optim.AdamW([
        {'params': [parameter for parameter in model.encoder.parameters() if parameter.requires_grad],
         'base_lr': plan['lr']},
        {'params': list(model.capability_head.parameters()), 'base_lr': plan['head_lr']},
    ], lr=plan['lr'], weight_decay=.01)
    epochs = 2 if args.smoke else plan['epochs']
    batch_size = 2 if args.smoke else plan['effective_batch']
    steps = epochs * math.ceil(len(supervised_train) / batch_size)
    dump(out / 'plan.json', {'schema_version': SCHEMA_VERSION, 'mode': 'task_capability',
                             'seed': args.seed, 'smoke': args.smoke, 'plan': plan,
                             'training_data': train_counts,
                             'optimizer_updates': steps, 'input_contract': 'one task, no candidate model input',
                             'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'device': args.device})
    rng = random.Random(args.seed)
    step, examples, input_tokens = 0, 0, 0
    curves = []
    for epoch in range(1, epochs + 1):
        ordered = list(supervised_train)
        rng.shuffle(ordered)
        for offset in range(0, len(ordered), batch_size):
            batch = ordered[offset:offset + batch_size]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            # Microbatch one task; accumulation preserves the task-level mean.
            for row in batch:
                ids = cache[row['decision_id']]
                logits = model(**padded_inputs([ids], tokenizer.pad_token_id, args.device))
                targets = capability_targets([row], plan['dimensions'], args.device)
                loss = capability_loss(logits, targets) / len(batch)
                loss.backward()
                loss_value += float(loss.detach())
                examples += 1
                input_tokens += len(ids)
            step += 1
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            fraction = (step - .5) / steps
            factor = (fraction / .05 if fraction < .05 else
                      .1 + .9 * .5 * (1 + math.cos(math.pi * (fraction - .05) / .95)))
            for group in optimizer.param_groups:
                group['lr'] = group['base_lr'] * factor
            optimizer.step()
            progress = {'step': step, 'steps': steps, 'epoch': epoch, 'examples': examples,
                        'input_tokens': input_tokens, 'loss': loss_value, 'gradient_norm': norm,
                        'lr': plan['lr'] * factor, 'head_lr': plan['head_lr'] * factor}
            dump(out / 'progress.json', progress)
            with (out / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(progress) + '\n')
            print(json.dumps(progress), flush=True)
        curves.append({'epoch': epoch, 'step': step,
                       **evaluate(model, dev_rows, tokenizer, plan, table, args.device,
                                  out / f'predictions-dev-{epoch:02d}.jsonl', cache)})
        model.save_checkpoint(out / 'latest-checkpoint')
        dump(out / 'latest-checkpoint' / 'routing_table.json', table)
        dump(out / 'latest-checkpoint' / 'training_plan.json', plan)
        dump(out / 'curve.json', curves)

    checkpoint = out / 'checkpoint'
    proof = verify_checkpoint(model, checkpoint, supervised_train[:4], tokenizer, plan, table,
                              args.device, initial, cache)
    dump(checkpoint / 'routing_table.json', table)
    dump(checkpoint / 'training_plan.json', plan)
    test, special_test = None, None
    if not args.smoke:
        test = evaluate(model, read_rows(plan['test_path']), tokenizer, plan, table, args.device,
                        out / 'predictions-test-final.jsonl')
        if plan.get('special_test_path'):
            special_test = evaluate(model, read_rows(plan['special_test_path']), tokenizer, plan, table,
                                    args.device, out / 'predictions-special-test-final.jsonl')
    result = {'status': 'completed', 'schema_version': SCHEMA_VERSION, 'mode': 'task_capability',
              'seed': args.seed, 'smoke': args.smoke, 'checkpoint': str(checkpoint),
              'training_data': train_counts,
              'training': progress, 'verification': proof, 'curve': curves, 'test': test, 'special_test': special_test,
              'checkpoint_selection': 'fixed final checkpoint; no test-based selection',
              'total_seconds': time.perf_counter() - started}
    dump(out / 'result.json', result)
    print(json.dumps({'event': 'complete', 'mode': 'task_capability', 'seed': args.seed}), flush=True)


if __name__ == '__main__':
    main()
