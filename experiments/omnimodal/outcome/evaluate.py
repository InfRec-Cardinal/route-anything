"""Supplemental scalar inference using the unchanged historical trainer functions."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--backend-trainer', default=str(Path(__file__).with_name('train.py')))
    parser.add_argument('--split', choices=['train', 'dev', 'test', 'special_test'], default='special_test')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    started = time.perf_counter()
    plan_path, trainer_path = Path(args.plan).resolve(), Path(args.backend_trainer).resolve()
    plan = json.loads(plan_path.read_text())
    checkpoint = Path(args.checkpoint).resolve()
    config = json.loads((checkpoint / 'scorer_config.json').read_text())
    if config.get('mode') != 'scalar':
        raise ValueError('Scalar supplemental evaluator requires a scalar checkpoint.')
    # Separate process avoids collisions with the classifier owner modules.
    sys.path.insert(0, str(trainer_path.parent))
    spec = importlib.util.spec_from_file_location('paired_scalar_backend', trainer_path)
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from model import SharedQualityModel

    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(plan['model_path'], local_files_only=False)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        plan['model_path'], dtype=torch.bfloat16 if args.device.startswith('cuda') else torch.float32,
        attn_implementation='sdpa', local_files_only=False)
    base.config.use_cache = False
    model = SharedQualityModel.load_checkpoint(checkpoint, base, is_trainable=False).to(args.device)
    rows_path = Path(plan[args.split + '_path'])
    if not rows_path.is_absolute():
        rows_path = plan_path.parent / rows_path
    rows = backend.read_rows(rows_path)
    cache = {(row['decision_id'], candidate['arm_version_id']): backend.encoded_pair(row, candidate, tokenizer, plan)
             for row in rows for candidate in row['candidates']}
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = backend.evaluate(model, rows, tokenizer, cache, torch, args.device, output / 'predictions.jsonl')
    report.update({'status': 'completed', 'mode': 'scalar', 'split': args.split,
                   'checkpoint': str(checkpoint), 'plan': str(plan_path), 'backend_trainer': str(trainer_path),
                   'total_seconds': time.perf_counter() - started,
                   'evaluation_micro_batch': 1, 'training_performed': False})
    (output / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
