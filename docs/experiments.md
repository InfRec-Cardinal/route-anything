# Experiment guide

Run commands from the repository root after installing `requirements.txt` and preparing the [data](data.md). Recorded aggregates and figure assets are stored with their scripts. The historical scripts generally replace their local results and README when run.

## Analytical and outcome-matrix experiments

| ID | Entry | Required input |
| --- | --- | --- |
| Analytical examples | `python scripts/reproduce.py --outdir runs/analytical` | None |
| X2 portfolio coverage | `python scripts/run_experiment.py X2` | CodeRouterBench ID matrix |
| X3 attempt threshold and information value | `python scripts/run_experiment.py X3` | CodeRouterBench ID matrix |
| X4 ideal-write memory control | `python scripts/run_experiment.py X4` | CodeRouterBench |
| X5 latency sensitivity | `python scripts/run_experiment.py X5` | CodeRouterBench matrix and OOD metadata |
| X6 allocation | `python scripts/run_experiment.py X6` | CodeRouterBench |
| X7a released chain replay | `python scripts/run_experiment.py X7a` | CodeRouterBench, RouterBench, released decisions and OOD traces |
| X9 finite-menu policy optimization | `python scripts/run_experiment.py X9` | CodeRouterBench and released ID decisions |
| X10 failure-conditioned escalation | `python scripts/run_experiment.py X10` | CodeRouterBench and OOD traces |
| X11 portfolio selection on RouterBench | `python scripts/run_experiment.py X11` | RouterBench 0-shot matrix |
| X12a released signal value | `python scripts/run_experiment.py X12a` | CodeRouterBench and released router decisions |
| X16 reference/resource sensitivity | `python scripts/run_experiment.py X16` | Outcome matrices and the experiment-specific inputs named in its script |

X5 retains historical constructed-process checks, but only its latency sensitivity is used in the paper. X4 is an ideal-write control accompanying the empirical memory study. Several scripts estimate uncertainty by resampling tasks or constructed streams; their full runs can take minutes.

## Archived calls and optional collection

| ID | Entry | Replay inputs |
| --- | --- | --- |
| X7b feedback, retry and switch | `python experiments_x/X7b_inloop_live/run.py` | Recorded attempts and original task reconstruction |
| X8 verifier operating point | `python experiments_x/X8_operating_point/run.py` | Samples, sandbox, calibration and judge caches |
| X12b paid forecast | `python experiments_x/X12b_llm_signal/run.py` | Forecast records, tasks, model costs and released decisions |
| X12c direct-choice evaluation | `python experiments_x/X12c_jev_policy/audit_official.py` | Archived choice calls and backend outcome matrices |
| X13 memory | `python experiments_x/X13_memory/run.py` | Verdicts, task features, model outcomes and constructed streams |

X7b, X8, X12b and X13 default to replaying saved records; `--collect` enables new collection. Missing records must be supplied for the selected protocol. Collection uses the configurable shared client. X12c's `run.py` and `run_two_primary.py` are collection entry points; use `audit_official.py` for the archived evaluation.

Replaying the same test suite used for feedback measures performance against that feedback suite. X8's judge price is a stated price proxy. X13 combines real outcomes and judgments with constructed arrivals and retention. These distinctions are part of the experiment, not implementation details to discard when reporting results.

## Follow-up analyses

- `experiments/paired_portfolio_analysis.py`: fitted portfolio rules on CodeRouterBench and RouterBench; takes `--code-data`, `--router-data`, and output arguments.
- `experiments/router_selection_recheck.py`: original split and ten fixed rerun splits; takes `--data`. It preserves per-split results and pooled comparisons.
- `experiments/verifier_price_analysis.py` and `verifier_price_intervals.py`: X8 price sensitivity from archived row tables and judge records.
- `experiments/ood_cost_imputation.py`: OOD missing-call cost sensitivity.
- `experiments/quality_cost_sensitivity.py`, `decision_baselines_analysis.py`, and `additional_sign_baselines.py`: resource, quality, and decision-rule comparisons.
- `experiments/grading_diagnostic/run.py` and `extra_test_retry_analysis.py`: retained grading and feedback-test diagnostics.

Use each argument-based script's `--help` for its input and output options. The legacy `run.py` scripts listed in the first table execute their analysis directly.

## Optional GPU experiment

`experiments_x/R6_stepwise/` preserves the stepwise-routing/cache-transfer code and available score/latency summaries. Generation uses open-weight Qwen3 models, vLLM and the upstream C2C model. It needs GPU resources and separately obtained checkpoints; it is outside the CPU quick start. Historical raw token continuations are unavailable, so rerunning continuation scoring requires regenerating those outputs. See the directory's release notes before running it.

## Reproducibility scope

The release preserves the original seeds and scientific calculations while making data paths configurable and collection explicit. It does not claim that all fresh-call experiments have been rerun. See [validation](validation.md) for commands actually executed during release preparation and [reproduction notes](reproducibility.md) for the original analysis history.
