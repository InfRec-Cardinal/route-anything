# R6: routing within a generated answer

This optional GPU experiment compares whole-answer choices, line-boundary handoffs, one-step interventions and Cache-to-Cache continuations. Pair A is Qwen3-0.6B / Qwen3-4B; pair B is Qwen3-1.7B / Qwen3-14B. Tasks come from HumanEval+ and MBPP+. The original protocol excludes HumanEval/32 because its canonical answer fails the retained sandbox.

`results/` contains archived pair summaries and latency fits. Per-response scores are distributed with the companion dataset. The historical generated token trajectories are not included: scores alone cannot reconstruct confidence thresholds, prefix lengths or inference cost. `code/analysis.py` therefore requires a fresh collection or separately supplied `out_*.jsonl` trajectories. The archived summaries can be inspected without GPUs:

```bash
python experiments_x/R6_stepwise/code/show.py experiments_x/R6_stepwise/results/pairA_analysis_results.json
python experiments_x/R6_stepwise/code/show.py experiments_x/R6_stepwise/results/pairB_analysis_results.json
```

## Environment and inputs

Use Linux with CUDA and sufficient GPU memory for the chosen Qwen models, plus PyTorch, vLLM, Transformers, NumPy, pandas and PyArrow. Install mutually compatible versions of PyTorch, CUDA and vLLM for the host. This release has not rerun GPU generation or established a new dependency lock.

The model registry uses the public `Qwen/Qwen3-*` model identifiers. A local model directory under `models/<model-name>` takes precedence. Download `evalplus/humanevalplus` and `evalplus/mbppplus` into the normal Hugging Face cache. `HF_HUB_CACHE` overrides that cache path. `STEPWISE_ROOT` overrides the experiment directory used for `models/`, `cache/` and `runs/`.

## Collection stages

Run these commands from `experiments_x/R6_stepwise`. They create new model responses and execute generated Python against the benchmark tests. The example uses pair A; for pair B substitute `B`, `Qwen3-1.7B` and `Qwen3-14B`.

```bash
python code/plan.py --pair A --phase A
python code/gen_vllm.py --model Qwen3-0.6B --jobs runs/pairA/jobs_A_S.jsonl --out runs/pairA/out_A_S.jsonl
python code/gen_vllm.py --model Qwen3-4B --jobs runs/pairA/jobs_A_L.jsonl --out runs/pairA/out_A_L.jsonl
python code/gen_vllm.py --single_call --model Qwen3-0.6B --jobs runs/pairA/jobs_A1_S.jsonl --out runs/pairA/out_A1_S.jsonl
python code/gen_vllm.py --single_call --model Qwen3-4B --jobs runs/pairA/jobs_A1_L.jsonl --out runs/pairA/out_A1_L.jsonl
python code/plan.py --pair A --phase B
python code/gen_vllm.py --model Qwen3-4B --jobs runs/pairA/jobs_B_L.jsonl --out runs/pairA/out_B_L.jsonl
python code/gen_vllm.py --model Qwen3-0.6B --jobs runs/pairA/jobs_B_S.jsonl --out runs/pairA/out_B_S.jsonl
python code/plan.py --pair A --phase C
python code/gen_vllm.py --model Qwen3-0.6B --jobs runs/pairA/jobs_C_S.jsonl --out runs/pairA/out_C_S.jsonl
```

For each output file, run `code/eval_tests.py --inputs <output> --out <matching-score-file>`: for example, `runs/pairA/out_A_S.jsonl` maps to `runs/pairA/score_A_S.jsonl`. `code/bench_latency.py --model <model-name>` measures each model's cost calibration on the current host. `code/analysis.py --pair A` computes the paired analyses after these trajectories, scores and latency fits are available. Use `code/gen_policy.py` for live line interventions; its `--taus` values must come from `common.tau_grid('A')`, which uses only fold-0 small-model root trajectories. Write these to `runs/pairA/out_live.jsonl` and score them as `runs/pairA/score_live.jsonl` before analysis.

## Optional Cache-to-Cache comparison

`code/c2c_plan.py` builds pair-A continuation jobs. `code/c2c_gen.py` and `code/c2c_batch.py` require the upstream C2C package exposing `rosetta.model`, plus `nics-efc/C2C_Fuser` in the Hugging Face cache. The recorded C2C environment used PyTorch 2.6 and Transformers 4.52.4, separately from vLLM. The released fuser is `qwen3_0.6b+qwen3_4b_Fuser/final`. Run fused and plain conditions explicitly; preserve their separate outputs and score files. Upstream models, benchmark datasets and the C2C implementation retain their own licenses.
