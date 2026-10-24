# Learned price-routing baselines

This directory extracts the CPU fit and recorded-outcome evaluation used for the learned rows associated with manuscript Tables 11 and 12. It includes quality/cost ridge regression, TF-IDF nearest neighbors, the experiment's RouteLLM-style pair classifier, and its FrugalGPT-style response-scored cascade. These are the actual experiment calculations with portable input/output handling. They do not call a model or download data.

No task prompts, responses, quality/cost matrices, fitted models, or paper results are included. Supply the inputs described in [SCHEMA.md](SCHEMA.md). `run.py` writes aggregate reports only. The separate synthetic smoke check writes invented inputs into the directory you explicitly select.

## Source protocol versus manuscript wording

The source programs `ra7-fix/R7_policy/run.py` and `ra7-fix/R8_baselines/run.py` read their supplied prompt projection unchanged. The R7 source documentation explicitly describes loader-written wrappers in these prompts. Its recorded summary agrees with the Table 11 learned-policy numbers. The current manuscript describes a unified train-fitted wrapper-stripping protocol. We have not established that the exact frozen prompt projection behind those table rows underwent that preprocessing.

The release therefore preserves the source program's default: supplied prompts are unchanged. `--strip-wrappers` explicitly selects an **alternate CRB protocol**, using the actual union-stripper from `ra9-fix/crb_robust/stripwrap.py`: learn repeated lines, sentences, and affixes within training sources at frequency 0.2 with minimum count 3, union the learned rules, and apply that union to both fitting and test prompts without test-source labels. This option changes the features and may change the results. It is not a correction of the manuscript or evidence that its stripped Table 11 rows have been reproduced. It does not strip recorded response text. RouterBench retains its own parsed source prompts and does not accept this flag.

The exact historical CRB prompt and response projections are not distributed here. The public CodeRouterBench `id_tasks.jsonl` inspected during extraction contains task and split metadata, **not prompt text**. It cannot replace `--prompts`. Reconstructing prompts with another upstream revision or loader also does not establish equality with the historical projection. The commands below run the released implementation on your supplied data; they do not promise exact manuscript table values.

## Source mapping and release boundary

| Released implementation | Original experiment source, inspected 2026-10-24 |
| --- | --- |
| CRB ridge, kNN, routing, and pair classifier | `ra7-fix/R7_policy/run.py`, learned-policy section |
| CRB response-scored chain search | `ra7-fix/R8_baselines/run.py`, FrugalGPT-style section |
| RouterBench split, numerical solver, and learned routing | `ra9-fix/rb_policy/run.py`, learned-baseline section |
| Optional CRB union stripping | `ra9-fix/crb_robust/stripwrap.py` |

The eight Python files separate source computations from local paths, data ingestion, and report writing. The source RouteLLM-style implementation is a linear TF-IDF classifier for a chosen model pair; it is not the upstream RouteLLM training package. The FrugalGPT-style implementation is the experiment's response-quality classifier and finite chain search; it is not an upstream service or pretrained release. Those method names credit the baseline designs rather than claiming upstream authorship of this implementation. NumPy, SciPy, scikit-learn, and pandas retain their respective upstream licenses.

This directory excludes later BGE/RouterDC GPU experiments, remote embeddings, inference collection, AutoMix, OOD experiments, and state/certification policies. Those are not needed to execute these extracted CPU baselines. No external repository is vendored.

## Run

From the repository root:

```sh
python -m pip install -r experiments/learned_price/requirements.txt
python experiments/learned_price/smoke.py --out runs/learned-price-synthetic
```

The smoke check uses 100 invented fitting rows and 40 invented test rows. It exercises all four baseline families, ten routing/report paths, the 15 possible chains of up to three distinct synthetic models, and RouterBench list-string parsing. It is an implementation check, not a paper experiment.

For CodeRouterBench, point `--crb-data-dir` at the directory containing `coderouterbench/id_results_long.csv`, and supply your reconstructed prompt projection:

```sh
python experiments/learned_price/run.py --profile crb \
  --crb-data-dir data/agent-as-a-router/data \
  --prompts /path/to/id_prompts.json \
  --reference claude-opus-4-6 --weak Qwen3-Max \
  --methods ridge knn routellm --out runs/crb-learned-aggregates.json
```

These reference and weak names are the **recorded source arm IDs**. The loader does not relabel historical arms to newer names appearing in the manuscript. Select IDs that exist in your input. To include the response-scored baseline, supply `--responses /path/to/id_responses.json` and add `frugal` to `--methods`.

For the source RouterBench pickle:

```sh
python experiments/learned_price/run.py --profile routerbench \
  --routerbench-pkl /path/to/routerbench_0shot.pkl \
  --reference gpt-4-1106-preview --methods ridge knn \
  --out runs/routerbench-learned-aggregates.json
```

Both profiles also accept `--input /path/to/records.jsonl`. See the schema for ordering and split rules. `--limit-fit 240 --limit-test 120 --bootstrap 30` provides a small real-input smoke run and marks the report `smoke: true`; such a subset changes the fitted policy and is not a paper result. Full fitting uses dense task-by-task kernels, so memory grows quadratically with the number of fitting rows. pandas is needed only for the pickle loader; the normalized JSONL and CRB loaders use the other three dependencies.

## Preserved fit and evaluation protocol

- **Features:** word TF-IDF 1–2 grams and character `char_wb` TF-IDF 3–5 grams, each capped at 100,000 features with minimum document frequency 2 and sublinear term frequency. Their concatenation is divided by the square root of 2. Vectorizers fit on all fitting prompts without labels; they are not refitted within each fold.
- **CRB split:** use supplied `probing` and `id_test` membership. The inspected source had 7,080 fitting and 2,919 test tasks. Five shuffled folds stratify by dimension and source with seed 20260925. Ridge alphas are 0.3, 1, 3, and 10; each target selects by out-of-fold MSE. The source uses float64 dual-kernel solves.
- **RouterBench split:** make the source 70/30 split within `eval_name` and reference-score-at-least-0.5 strata, with rounding and seed 20260926. Five fitting folds stratify by `eval_name`. Ridge alphas are 1, 3, 10, and 30; the source float32 kernel solver and iterative refinement are preserved.
- **Targets and menu:** learn score, log cost, acceptance probability, and accepted score. CRB uses direct actions and each non-reference primary with thresholds 1 and 0.5; RouterBench uses threshold 1. An accepted attempt returns the primary score; otherwise it returns the reference score and pays for both calls. Predicted cascade value is accepted-score expectation plus predicted miss probability times predicted reference score; predicted cost is primary cost plus that miss probability times reference cost. This is the source composition assumption, not a separately fitted conditional reference model. Exponentiated cost predictions receive the source fitting-set mean calibration.
- **kNN:** average actual action values and costs of 20 nearest fitting prompts. Fitting estimates exclude only the query itself; this is the source leave-one-out procedure, not five-fold cross-fitting.
- **Operating points:** price routing maximizes predicted value minus price times predicted cost. CRB searches 201 prices from 0.01 to 1,000; RouterBench searches 251 from 1 to 100,000. Budget routing uses the source bisection and 40 fractions from 0.001 to 1 of the routed batch's predicted reference mean cost. Selection uses fitting outcomes: report the maximum-efficiency point and the highest-efficiency nonnegative-quality-difference point. If no such zero-loss point exists, preserve the source fallback to the best quality difference; the returned fitting `p_dV` exposes this case.
- **Pair classifier:** predict whether the reference beats the chosen weak arm using logistic regression, select regularization by out-of-fold AUC, and select a threshold from out-of-fold quantiles using fitting outcomes. CRB uses the joined word/character features; the optional RouterBench pair classifier uses word features, as in its source.
- **Response-scored cascade:** CRB only. Each model's scorer uses prompt TF-IDF, response TF-IDF of the first 4,000 characters, and an empty-response flag. The source fits a `liblinear` logistic classifier for score at least 1, selecting among four regularizations by out-of-fold AUC. Search all ordered chains of up to three distinct models and the source 50-quantile threshold grid, maximizing fitting efficiency subject to nonnegative fitting quality difference. The final stage always returns. The replay charges each attempted model's recorded cost; scorer computation has no added charge.
- **Uncertainty:** keep the fitted policy fixed and use 1,000 paired task bootstrap replicates by default. RouterBench bootstraps within `eval_name`; CRB uses ordinary task resampling. Reports contain aggregate quality, relative efficiency, mean cost in milli-USD, quality difference, intervals, and the bootstrap fraction with nonnegative quality difference.

## Validation performed

The source extraction was checked on existing local CRB inputs without copying those inputs into this repository. All 9,999 menu value/cost rows matched the original source calculations, and a 40-prompt numerical check matched the source ridge predictions and cascade composition exactly in the tested environment. A 240/120 real-input smoke run completed for ridge, kNN, and the pair classifier. Independent synthetic checks covered all four families and the RouterBench numerical profile. The full historical fits and paper tables were not rerun. No GPU, inference, or network operation is part of these checks.
