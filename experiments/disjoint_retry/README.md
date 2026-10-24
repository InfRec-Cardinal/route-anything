# Retries with disjoint feedback and scoring tests

This is the offline analysis for **Table 14 and Appendix C.4** of *Route Anything: an attempt for AGI through a simple omni-modal router* (2026-10-24).

Each eligible task has a feedback test subset **F** and a disjoint scoring subset **S**. A first attempt enters the analysis when it fails S. Each failure receives three fresh retries and three retries with feedback from F; every retry is scored on S. The analysis averages the three retry outcomes within each failure and then averages over failures. Its 95% intervals use 2,000 paired task-cluster bootstrap resamples, preserving all model failures of a sampled task.

The paper has 282 eligible tasks and 263 first-attempt failures across seven models. Pooled recovery is 0.16477 for fresh retries and 0.30418 for feedback retries; the difference is 0.13942, with interval [0.05846, 0.22267]. `reference.json` contains the per-model results and aggregate sensitivity analyses.

## Run offline

Install the root `requirements.txt`, then provide the earlier disjoint-test archive and its seven-model extension:

```sh
python experiments/disjoint_retry/analyze.py \
  --earlier-dir /path/to/earlier_disjoint_records \
  --extension-dir /path/to/extended_retry_records \
  --x7b-attempts /path/to/attempts_view_a_6144.jsonl \
  --output runs/disjoint_retry/results.json
```

The [input schema](schema.md) lists the five required disjoint-study files and the X7b denominator archive. The X7b file is used only to count the earlier study's population. **X7b outcomes cannot replace the disjoint-study records.** The existing public replay bundle supplies that earlier X7b archive, but does not supply the five newer disjoint-study files. Those files must be provided separately with appropriate access.

The default is analysis only. The command makes no model calls, downloads nothing, does not execute generated programs, and writes aggregate results only. There is no collection mode in this package. The original collection implementation depended on its archived service configuration and benchmark reconstruction; it is not represented here as a complete portable collector. The schema and study protocol describe the inputs needed for another collection.

## Protocol and interpretation

The initial five models are DeepSeek-flash, DeepSeek-v4-pro, Kimi-K2.5, Qwen3-Max and Qwen3.5-Plus. The extension adds GLM-5 and MiniMax-M2.7 and expands every arm to three retry samples. For each of the initial models, the first retry sample is read from the earlier archive; the remaining samples come from the extension. All samples for the two added models come from the extension.

The historical collection used temperature 0.7. Feedback contains at most 800 characters from F, and F is not used to score recovery. Test units are source-specific: nontrivial assertion statements, MBPP test-list entries, or BigCodeBench test methods. Tasks need at least two separable units; F receives half rounded down and S the remainder. The recorded split manifests must be preserved. This analyzer consumes recorded pass/fail judgments and does not independently reconstruct or verify that split.

Missing retries are reported in the output. The paper's records have zero missing fresh or feedback samples. For sensitivity comparisons, the original calculation averages available samples and omits failures without observations in both compared arms. The released code retains that behavior.

## Provenance and validation

`analyze.py` preserves the statistical calculations of the original `online_deploy/scripts/part2_analyze.py`. Local paths became command-line arguments; per-failure output was removed; excluded-task identities became counts by reason. Seeds and numerical calculations are unchanged. No new task-level records, model responses, split manifests, or credentials are distributed here or uploaded to Hugging Face.

On 2026-10-24, the complete local archived records were replayed with 2,000 bootstrap resamples. All 734 numeric result fields and all retained descriptive fields matched the archived reference within floating-point tolerance. This reproduces the recorded-outcome analysis; it does not rerun the seven models.
