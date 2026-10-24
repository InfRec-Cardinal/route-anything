# User-supplied input contracts

All JSONL records are UTF-8, one object per line. Keep original grouped splits
and full-input identity; benchmark task names alone may not identify identical
inputs (for example, source code history can differ). Missing scores are null,
not zero. Qualities used by the BCE trainers must be normalized to [0, 1].
Do not put model outcomes, winner labels or task IDs into encoder input text.

Run `synthetic_demo.py --out /tmp/route-anything-synthetic` for complete invented
examples of every training input contract. These examples have one leaf/type,
two text arms, and seven invented robot policies. They are not research data.

## Capability tree

`tree.json` has `tracks`, each with `track_id` and `tasks`. Each task has
`task_id`, `task_description`, and `capabilities`. Each capability has `leaf_id`
and its input-only definition (`name_zh`, `definition`, and optionally the
additional definition fields enumerated by `LEAF_FIELDS` in `train.py`).

Each split JSONL row supplies:

- Identity: `id`, `task_id`, `group_id`, `benchmark`, `split`.
- Encoder input: `task_text`, `images` (paths), `state_text`, `output_contract`.
- Scope: `track_id`, `tree_task_id`, `candidate_leaf_ids`.
- Supervision: `leaf_id`, `label_status`. A valid leaf uses `teacher_labeled`;
  a genuinely unresolved annotated example uses `unresolved_requirements`
  with null leaf. Failed/pending annotations remain masked.
- Offline evidence: `model_scores` mapping model IDs to observed quality;
  `model_costs` and `model_tokens`, each mapping model IDs to
  `{"value": number-or-null, "scope": string-or-null}`.

`fit_table.py` uses the training rows only. Identical inputs are grouped by
group/task IDs, exact text, image paths, state text and output contract. Input
assets must already be aligned: different paths to identical files are not
automatically treated as equal. Missing outcomes should be omitted from the
`model_scores` dictionary used by the source leaf fitter.

## Text outcome and type studies

Rows share `decision_id`, `task_id`, `group_id`, `benchmark`, `metric_id`,
`scalar_spec_id`, `split`, `input_text`, `output_contract`, `weights` and
`candidates`. For the scalar outcome head, `weights` is `[1.0]`.

Each candidate supplies `arm_version_id`, `model_id`, static `profile`,
`quality`, and boolean `observed_mask`, `quality_valid_mask`,
`supervision_valid`, `selection_mask`, and `capability_mask`.
The profile can identify the model and static execution configuration but
must not contain the current task's outcome. Invalid or absent qualities
must have the corresponding supervision flags false.

Type training additionally supplies `capability_targets` (normalized positive
weights by type ID), `target_capability_id`, and
`capability_supervision_valid`. A taxonomy object contains `dimensions`, each
with `id`, `name`, `definition`; optional `families` entries define `id` and
`model_ids`. Supply already frozen, output-family-aware definitions. The
release adapter does not infer or relabel types from test outcomes.

## Robot

Each row supplies `id`, `group_id`, `task_text`, `images`, `state_text`,
`output_contract`, `model_scores` (all seven binary observed outcomes), and
optional `model_latencies` (seconds; missing values omitted or null).
Image paths are absolute or relative to the data directory. A paper-style
record uses the three initial camera observations, never future frames.

`models.json` contains `model_ids` in frozen output-head order and
`fixed_model`, chosen by mean success on train only. Keep the same order in
every training run. Latencies are replay diagnostics, not prediction inputs.

## Shared-panel replay

`matched_evaluate.py` reads an independent, small interface:

```json
{"id":"synthetic-input-1","group_id":"synthetic-family-1","benchmark":"synthetic-benchmark","split":"train","candidate_models":["synthetic-a","synthetic-b"],"model_scores":{"synthetic-a":0.9,"synthetic-b":0.7}}
```

Predictions are one row per method, seed and test input:

```json
{"id":"synthetic-input-2","method":"synthetic-router","seed":42,"selected_model":"synthetic-a"}
```

Use stable model identity after explicitly resolving execution configurations
and aligning full input contexts. The source tree and robot predictions use
`selected_model`; type predictions use `selected_model_id`; outcome predictions
use an arm-version `choice`, which must be mapped through the candidate metadata.
The evaluator does not silently merge distinct model execution configurations.
The prediction file must contain every intended method and seed: missing a
prediction for one seed removes that input from the shared panel.

By default, the fixed model maximizes its mean over its observed training
outcomes. Candidates with some missing training outcomes remain eligible.
With `--opus-fallback`, each candidate instead represents the same policy used
at test: an absent or null Opus outcome uses the same input's Astra outcome.
Candidate identities are collected from the union of training rows' declared
`candidate_models` and score keys, without reading test data. Thus an Opus key
that is entirely absent in some training rows still receives the configured
fallback. The report includes effective observed training counts, means and
`fallback_train_rows`; when both outcomes are missing, that input remains
unobserved for the policy.

Test support is selected after fixing this baseline. Declared candidate panel
completeness is checked after applying the same fallback; use
`--allow-partial-candidate-panel` only to permit remaining missing unselected
outcomes. The code performs no inference, new scoring, or data download.
