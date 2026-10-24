# Offline input schema

All files are UTF-8 JSON Lines with one object per line. Unused fields may be omitted. Pass fields are booleans; seeds are integers or JSON `null`. Use the original record order, model names, seeds and task IDs when reproducing the paper.

## Earlier disjoint-study directory

| File | Fields read by the analyzer | Meaning |
| --- | --- | --- |
| `splits.jsonl` | `task_id`, `eligible`, `reason` | Eligible tasks have at least two separable test units. `reason` explains an excluded task. |
| `first_attempts.jsonl` | `task_id`, `model`, `pass_F`, `pass_S`, optional `finish_reason` | First attempts from the initial five models. Failures are selected using `pass_S == false`. |
| `calls.jsonl` | `task_id`, `model`, `arm`, `seed`, `pass_S`, `api_error` | Earlier fresh and feedback retry samples. |

The split manifest originally also records `f_idx`, `s_idx`, `n_cases`, `unit`, and `source`. Retain these with the source tests to establish which units belong to F and S. The analyzer itself uses eligibility and recorded outcomes, and does not inspect test programs.

The initial model strings are `deepseek-flash`, `deepseek-v4-pro`, `kimi-k2.5`, `qwen3-max`, and `qwen3.5-plus`.

## Extension directory

| File | Fields read by the analyzer | Meaning |
| --- | --- | --- |
| `first_attempts_new.jsonl` | `task_id`, `model`, `pass_F`, `pass_S`, `api_error`; optional accounting fields below | First attempts for `glm-5` and `MiniMax-M2.7`. |
| `calls_new.jsonl` | `task_id`, `model`, `arm`, `seed`, `pass_S`, `api_error`; optional accounting fields below | Additional samples for the initial models and all retry samples for the two added models. |

Optional accounting fields are `cached`, `cost_usd`, `prompt_tokens`, `completion_tokens`, and `finish_reason`. They affect collection-accounting/truncation summaries, not pass/fail recovery. Reproducing every archived diagnostic requires the original values.

Retry selection uses these slots:

| Arm | Slot 1 | Slot 2 | Slot 3 |
| --- | --- | --- | --- |
| `fresh` | seed `2` | seed `3` | seed `4` |
| `fb_F` | seed `null` | seed `3` | seed `4` |

Slot 1 comes from `calls.jsonl` for the initial five models and from `calls_new.jsonl` for the added models. Slots 2 and 3 always come from `calls_new.jsonl`. Calls marked `api_error` are excluded. For duplicate `(task_id, model, arm, seed)` keys, the last successful row is used, following the original analysis.

## Earlier X7b denominator archive

`--x7b-attempts` takes the earlier `attempts_view_a_6144.jsonl`. Only rows with `branch == "root"` and their `task_id` and `model` are used, solely for denominator diagnostics. The retry estimates use the disjoint-study records above.

## Output

The aggregate JSON contains settings, denominator counts, seven per-model comparisons, pooled comparisons, truncation sensitivity, model heterogeneity, feedback-content subgroups, collection-accounting totals, and missing-sample counts. Recovery values and confidence intervals use a 0–1 scale; `diff_pp` multiplies the difference by 100. No task IDs, source programs, generated responses, or per-failure rows are written.
