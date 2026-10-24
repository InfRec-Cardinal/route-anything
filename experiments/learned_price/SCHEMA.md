# Caller-supplied inputs and aggregate outputs

All input files stay at paths selected by the caller. This release contains no historical prompt projection or quality/cost matrix. Input model IDs are literal recorded IDs; no aliasing to manuscript display names occurs.

## Normalized JSONL

Each line is one task. Keep every selected model's score and cost on every row. Scores are unit-interval outcome measurements and costs are recorded nonnegative USD amounts for that task/model execution. The historical CodeRouterBench release includes zero-cost missing runs; the convenience loader preserves them as described in the repository data notes. The historical experiments use complete panels. Preserve original row order for the RouterBench split and source model order for tie breaking; `--models` can explicitly fix the model order. The CRB convenience loader sorts task IDs and model IDs as its source pivot does.

An invented structural example (use substantially more rows for five-fold fitting and 20-neighbor estimation):

```json
{"id":"synthetic-1","split":"probing","stratum":"code|synthetic","source":"synthetic","prompt":"An invented programming task.","scores":{"reference":1.0,"cheap":0.5},"costs":{"reference":0.01,"cheap":0.001},"responses":{"reference":"Invented answer A","cheap":"Invented answer B"}}
```

| Field | Meaning |
| --- | --- |
| `id` | Optional caller identifier. It is not written to reports. |
| `prompt` | Exact text given to the feature extractor. The default leaves it unchanged. |
| `scores` | Model-ID to numeric outcome score mapping. |
| `costs` | Same model IDs, mapping to execution cost in USD. |
| `split` | CRB: `probing` or `id_test`. RouterBench ignores this field and creates its source split. |
| `stratum` | CRB: dimension plus source cell. RouterBench: `eval_name`. Used for five-fold fitting; RouterBench also uses it for its split and stratified bootstrap. |
| `source` | CRB source identifier, required only for `--strip-wrappers`. Stripping rules fit on fitting rows grouped by this field, then their union applies to all prompts. |
| `responses` | Required for `--methods frugal`: model-ID to recorded response text mapping. Each scorer uses at most the first 4,000 characters. |

The reference and optional weak model must be present in the selected model set. For ordinary runs, keep enough fitting examples in each stratum and both target classes for the source five-fold classifiers. A deliberately small smoke subset can produce a scikit-learn warning about rare strata.

## CodeRouterBench convenience loader

`--crb-data-dir DIR` reads `DIR/coderouterbench/id_results_long.csv` with fields `task_id,split,dimension,model,score,cost_usd`. It groups rows by task, derives the source from the second underscore-separated component of `task_id`, and derives the fitting stratum as `dimension|source`.

`--prompts` is required and accepts either a JSON object mapping task IDs to prompt strings, or JSONL objects with `task_id` and `prompt`. The loader follows the source behavior of using an empty prompt for missing projection entries. The public `id_tasks.jsonl` metadata alone does not provide these strings. Exact historical prompt reconstruction has not been established by this release.

`--responses` accepts a JSON object mapping task IDs to objects mapping model IDs to recorded response strings. A missing response uses the source empty-response feature. Supplying an incomplete response projection changes FrugalGPT-style scorer fitting and cannot establish the historical result.

## RouterBench convenience loader

`--routerbench-pkl FILE` reads the source pandas table. It needs `prompt`, `eval_name`, each model's score column named by the model ID, and its cost column named `MODEL|total_cost`. The table column order determines the model order. Prompts serialized as Python string lists are parsed with `ast.literal_eval` and joined by newlines, preserving the source parser. Use a pickle you already trust, or convert your own data to the normalized JSONL format.

The source split uses `eval_name` crossed with whether the selected reference's score is at least 0.5, shuffles within each sorted stratum using seed 20260926, and takes the rounded 70% count for fitting. Preserve the original table row order to reproduce membership. The CRB-only `--strip-wrappers` option is rejected for this profile.

## Aggregate report

`--out` writes a JSON object containing the profile, literal model and reference IDs, seed, fitting/test counts, action names, preprocessing choice, feature count, fitting hyperparameters, selected operating points, and aggregate metrics. It contains no task IDs, prompts, responses, per-task score/cost/prediction arrays, or trained feature/model objects.

For price and budget routes, `max_fit_efficiency` and `zero_loss_fit` each contain the selected parameter, fitting aggregate metrics, and test metrics. `zero_loss_fit` uses the original best-quality fallback if no fitting point has nonnegative quality difference, so inspect its `p_dV` before describing it as a zero-loss policy. For FrugalGPT-style routing, the report records the selected model chain, thresholds, scorer hyperparameters, aggregate test metrics, and stage return shares.

Each test metric is `[point_estimate, lower_95_percentile, upper_95_percentile]`. `r` is aggregate quality per USD divided by reference aggregate quality per USD; `dV` is mean quality minus reference mean quality; `cost_mUSD` is mean cost multiplied by 1,000; `mean_score` is mean outcome quality. `compliance` is the fraction of bootstrap replicates with `dV >= -1e-12`. This is fixed-policy bootstrap uncertainty, not a new certification procedure.

Either limit option marks the report `smoke: true`. The report's `evidence_type` is recorded-outcome replay, not newly collected model inference. The default bootstrap count is 1,000; the synthetic check uses 30. The synthetic check intentionally writes its invented JSONL fixture alongside its aggregate report; the production runner does not write task-level data.
