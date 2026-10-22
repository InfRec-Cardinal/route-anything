# BigCodeBench grading diagnostic

The released matrix assigns zero scores to all eight models on 750 of 1,097 BigCodeBench tasks. The retained verdicts come from a local recheck with official tests of 40 sampled all-zero tasks: 20 importing the relevant scientific libraries and 20 other tasks. Of 320 recorded model responses, 318 are nonempty and 91 pass locally. Among released-zero responses on 20 non-all-zero control tasks, 5 of 50 nonempty responses pass; all 107 nonempty released-positive control responses pass.

This is evidence that the all-zero subset includes evaluation failures. The diagnostic does not establish the cause of every zero, and the main replay analyses retain the released scores. The 2026-09-17 audit used seed 20260917 to select the sample. Task and model identifiers in `sandbox_rows.csv` identify the exact checked records. The CSV contains verdicts, not model prompts, responses, account records, or a shared cache.

`run.py --matrix /path/to/id_results_long.csv` recomputes the matrix count and diagnostic summary from the saved verdicts. To rerun verification of the same sample, additionally pass `--recheck --response-matrix /path/to/response_matrix.json` and configure `ACROUTER_DATA` and the local benchmark snapshots used by `experiments_x/common/sources.py`. Generated programs execute through `common/sandbox.py`; no model API is called. Local dependency or runtime differences can change the recheck, so its verdicts are saved separately from the archived CSV.

