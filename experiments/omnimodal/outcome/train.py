"""Paper outcome-quality trainer: BCE plus within-task score-gap regression."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import time
from model_policy import filter_record


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read_rows(path):
    return [filter_record(json.loads(line)) for line in Path(path).open() if line.strip()]


def inputs(pairs, tokenizer, torch, device):
    width = max(map(len, pairs))
    return {'input_ids': torch.tensor([[tokenizer.pad_token_id] * (width-len(x)) + x for x in pairs], device=device),
            'attention_mask': torch.tensor([[0] * (width-len(x)) + [1] * len(x) for x in pairs], device=device)}


def encoded_pair(row, candidate, tokenizer, plan):
    payload = {'request': row['input_text'], 'output_contract': row.get('output_contract', {}),
               'benchmark': row['benchmark'], 'metric_id': row.get('metric_id'), 'capabilities': plan['dimensions'],
               'aggregation_weights': row['weights'],
               'candidate': {'arm_version_id': candidate['arm_version_id'],
                             'profile': candidate.get('profile', {})}}
    # Both recipes see exactly the same prompt, including the same definitions.
    messages = [{'role': 'system', 'content': 'Estimate this candidate model performance for the user request under the stated evaluation capabilities.'},
                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > plan['max_sequence']:
        raise ValueError(f"Pair exceeds frozen sequence limit: {row['decision_id']}")
    return ids


def supervised(candidate):
    return candidate.get('supervision_valid') is True


def predict(model, row, tokenizer, cache, torch, device):
    scores, vectors, tokens = [], [], 0
    with torch.no_grad():
        for c in row['candidates']:
            pair = cache[(row['decision_id'], c['arm_version_id'])]
            logits = model(**inputs([pair], tokenizer, torch, device))
            probs = logits.float().sigmoid()[0].cpu().tolist()
            score = probs[0] if model.mode == 'scalar' else sum(w*p for w,p in zip(row['weights'], probs))
            scores.append(score); vectors.append(probs); tokens += len(pair)
    return scores, vectors, tokens


def evaluate(model, rows, tokenizer, cache, torch, device, path):
    model.eval()
    start = time.perf_counter()
    count, tokens, grouped = 0, 0, {}
    with Path(path).open('w') as output:
        for row in rows:
            scores, dimensions, pair_tokens = predict(model, row, tokenizer, cache, torch, device)
            candidates = row['candidates']
            selectable = [i for i,c in enumerate(candidates) if c.get('selection_mask') is True]
            observed = [i for i,c in enumerate(candidates) if c.get('quality') is not None and c.get('quality_valid_mask') is True]
            choose = min(selectable, key=lambda i:(-scores[i], candidates[i]['arm_version_id'])) if selectable else None
            complete = len(selectable) >= 2 and all(i in observed for i in selectable)
            quality = candidates[choose]['quality'] if choose is not None and choose in observed else None
            oracle = max(candidates[i]['quality'] for i in selectable) if complete else None
            prior_values = [c.get('calibration_profile', {}).get('mean_quality') for c in candidates]
            prior_values = [.5 if x is None else x for x in prior_values]
            prior_choice = min(selectable, key=lambda i:(-prior_values[i], candidates[i]['arm_version_id'])) if selectable else None
            prior_quality = candidates[prior_choice]['quality'] if complete else None
            errors = [abs(scores[i]-candidates[i]['quality']) for i in observed]
            dimension_errors = []
            if model.mode == 'capability':
                for i in range(len(candidates)):
                    for k,y in enumerate(candidates[i]['dimension_scores']):
                        if k>0 and y is not None and row['weights'][k] > 0:
                            dimension_errors.append({'dimension': k, 'absolute_error': abs(dimensions[i][k]-y)})
            record = {'decision_id': row['decision_id'], 'group_id': row['group_id'], 'benchmark': row['benchmark'],
                      'split': row['split'], 'n_candidates': len(candidates), 'n_selectable': len(selectable),
                      'candidate_ids': [c['arm_version_id'] for c in candidates], 'candidate_qualities': [c['quality'] if i in observed else None for i,c in enumerate(candidates)], 'selectable_ids': [candidates[i]['arm_version_id'] for i in selectable], 'scores': scores,
                      'dimension_predictions': dimensions if model.mode == 'capability' else None,
                      'dimension_errors': dimension_errors, 'weights': row['weights'],
                      'choice': candidates[choose]['arm_version_id'] if choose is not None else None,
                      'selected_quality': quality, 'routing_comparable': complete,
                      'quality': quality if complete else None, 'oracle': oracle,
                      'prior_choice': candidates[prior_choice]['arm_version_id'] if prior_choice is not None else None,
                      'prior_quality': prior_quality,
                      'regret': oracle-quality if complete else None,
                      'mae': sum(errors)/len(errors) if errors else None,
                      'singleton_mae': errors[0] if len(selectable)==1 and len(observed)==1 else None}
            output.write(json.dumps(record, ensure_ascii=False)+'\n')
            bucket = grouped.setdefault(row['benchmark'], [])
            bucket.append(record); count += 1; tokens += pair_tokens
    summary = {}
    for b,records in grouped.items():
        item = {'n':len(records), 'routing_n':sum(r['routing_comparable'] for r in records)}
        for key in ['quality','oracle','regret','mae','singleton_mae','prior_quality']:
            values = [r[key] for r in records if r[key] is not None]
            item[key] = sum(values)/len(values) if values else None
        summary[b] = item
    return {'predictions': str(path), 'count':count, 'pair_tokens':tokens, 'seconds':time.perf_counter()-start, 'benchmarks':summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--mode', choices=['scalar'], default='scalar')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    for key in ('train_path', 'dev_path', 'test_path'):
        value = Path(plan[key])
        plan[key] = str(value if value.is_absolute() else plan_path.parent / value)
    for stream in plan['streams'].values():
        value = Path(stream['path'])
        stream['path'] = str(value if value.is_absolute() else plan_path.parent / value)
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from peft import get_peft_model, LoraConfig
    from model import SharedQualityModel
    from verify_checkpoint import capture_initial_state, verify_checkpoint
    from loss import task_loss
    start = time.perf_counter()
    out = Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        raise RuntimeError('Completed run already exists; use its saved result')
    random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(plan['model_path'],local_files_only=False)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    train_rows, dev_rows = read_rows(plan['train_path']), read_rows(plan['dev_path'])
    all_rows = {r['decision_id']:r for r in train_rows+dev_rows}
    stream = read_rows(plan['streams'][str(args.seed)]['path'])
    if args.smoke: stream = [{'decision_id':i, 'pair_tokens':sum(len(encoded_pair(all_rows[i],c,tokenizer,plan)) for c in all_rows[i]['candidates'] if supervised(c))} for i in plan['smoke_ids']]*2
    cache = {(r['decision_id'],c['arm_version_id']):encoded_pair(r,c,tokenizer,plan) for r in all_rows.values() for c in r['candidates']}
    for entry in stream:
        r=all_rows[entry['decision_id']]
        actual=sum(len(cache[(r['decision_id'],c['arm_version_id'])]) for c in r['candidates'] if supervised(c))
        if actual != entry['pair_tokens']: raise ValueError('Frozen token budget differs')
    train_panel = [all_rows[i] for i in plan['train_panel_ids']]
    if args.smoke: train_panel=[all_rows[i] for i in plan['smoke_ids'][:4]]; dev_rows=dev_rows[:4]
    probe_rows = [all_rows[i] for i in plan['probe_ids']]
    base=Qwen3_5ForConditionalGeneration.from_pretrained(plan['model_path'], dtype=torch.bfloat16, attn_implementation='sdpa',local_files_only=False)
    base.config.use_cache=False
    targets=[n for n,m in base.named_modules() if isinstance(m,torch.nn.Linear) and 'language_model.layers.' in n]
    encoder=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=targets,task_type='CAUSAL_LM',bias='none'))
    model=SharedQualityModel(encoder,base.config.text_config.hidden_size,mode=args.mode,num_dimensions=len(plan['dimensions'])).to('cuda')
    initial=capture_initial_state(model)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.encoder.enable_input_require_grads()
    optimizer=torch.optim.AdamW([{'params':[p for p in model.encoder.parameters() if p.requires_grad],'base_lr':plan['lr']},{'params':list(model.quality_head.parameters()),'base_lr':plan['head_lr']}],lr=plan['lr'],weight_decay=.01)
    batch_size=2 if args.smoke else plan['effective_batch']; steps=math.ceil(len(stream)/batch_size)
    marks={0,math.ceil(steps/3),math.ceil(steps*2/3),steps}
    micro=plan['train_micro_batch']; evaluations=[]
    run_plan={**vars(args),'plan':plan,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'optimizer_updates':steps,'execution_started_unix':time.time()}
    dump(out/'plan.json',run_plan)
    def panels(step):
        for name,panel in [('train',train_panel),('dev',dev_rows)]:
            r=evaluate(model,panel,tokenizer,cache,torch,'cuda',out/f'predictions-{name}-{step:06d}.jsonl')
            r.update(step=step,split=name); evaluations.append(r)
            with (out/'curve.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    panels(0)
    total_tokens=0; examples=0; training_seconds=0.
    for step,offset in enumerate(range(0,len(stream),batch_size),1):
        batch=stream[offset:offset+batch_size]; model.train(); optimizer.zero_grad(set_to_none=True); loss_total=0.
        started=time.perf_counter()
        for entry in batch:
            row=all_rows[entry['decision_id']]; candidates=[c for c in row['candidates'] if supervised(c)]
            parts=[]
            for j in range(0,len(candidates),micro):
                chunk=candidates[j:j+micro]
                pairs=[cache[(row['decision_id'],c['arm_version_id'])] for c in chunk]
                parts.append(model(**inputs(pairs,tokenizer,torch,'cuda')))
                total_tokens+=sum(map(len,pairs))
            logits=torch.cat(parts,dim=0)
            term,components=task_loss(logits,candidates,row['weights'],args.mode)
            term=term/len(batch)
            term.backward(); loss_total+=term.item()
            del parts,logits,term
            examples+=1
        grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
        fraction=(step-.5)/steps
        factor=fraction/.05 if fraction<.05 else .1+.9*.5*(1+math.cos(math.pi*(fraction-.05)/.95))
        for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
        optimizer.step();torch.cuda.synchronize();training_seconds+=time.perf_counter()-started
        progress={'step':step,'steps':steps,'examples':examples,'pair_tokens':total_tokens,'loss':loss_total,'gradient_norm':grad_norm,'lr':plan['lr']*factor,'head_lr':plan['head_lr']*factor,'training_seconds':training_seconds,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
        dump(out/'progress.json',progress)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n')
        print(json.dumps(progress),flush=True)
        if step in marks:
            model.save_checkpoint(out/'latest-checkpoint')
            dump(out/'latest-checkpoint'/'training-progress.json',progress)
            panels(step)
    expected_tokens=sum(x['pair_tokens'] for x in stream)
    if total_tokens!=expected_tokens:raise ValueError('Actual pair token count differs from stream')
    entries=[{'id':r['decision_id']+'::'+c['arm_version_id'],'decision_id':r['decision_id'],'arm_version_id':c['arm_version_id'],'selection_mask':c.get('selection_mask') is True,'input_ids':cache[(r['decision_id'],c['arm_version_id'])],'weights':r['weights']} for r in probe_rows for c in r['candidates']]
    proof=verify_checkpoint(model,out/'checkpoint',entries,tokenizer,'cuda',initial,train_micro_batch=micro,report_path=out/'checkpoint-verification.json')
    dump(out/'checkpoint'/'data-plan.json',plan)
    final_test=None; full_train=None
    if not args.smoke:
        full_train=evaluate(model,train_rows,tokenizer,cache,torch,'cuda',out/'predictions-train-full-final.jsonl')
        dump(out/'full-train-evaluation.json',full_train)
        test=read_rows(plan['test_path'])
        for r in test:
            for c in r['candidates']:cache[(r['decision_id'],c['arm_version_id'])]=encoded_pair(r,c,tokenizer,plan)
        final_test=evaluate(model,test,tokenizer,cache,torch,'cuda',out/'predictions-test-final.jsonl')
    result={'status':'completed','mode':args.mode,'seed':args.seed,'smoke':args.smoke,'checkpoint':str(out/'checkpoint'),'training':progress,'verification':proof,'curve':evaluations,'test':final_test,'full_train':full_train,'total_seconds':time.perf_counter()-start,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'frozen_plan':str(Path(args.plan))}
    dump(out/'result.json',result)
    print(json.dumps({'event':'complete','mode':args.mode,'seed':args.seed}),flush=True)


if __name__=='__main__': main()
