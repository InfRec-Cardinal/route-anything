# Release validation

## October 24, 2026 refresh

All empirical checks below replay recorded outcomes locally. No model collection, GPU retraining, or Hugging Face upload was performed. Reproduction outputs were written to temporary directories, preserving the archived reference files.

| Check | Outcome |
| --- | --- |
| CodeRouterBench quality-floor certification | Full 200 fitting/certification splits and 1,000 paired bootstrap replicates; all 1,047 reference numeric fields matched. |
| RouterBench quality-floor certification | Full 200 halvings and 1,000 paired bootstrap replicates; all 349 certification reference numeric fields matched. A rerun after extracting the shared solver preserved the short-run results. |
| Disjoint feedback/scoring retries (Table 14) | Full recorded-outcome replay with 2,000 task-cluster bootstrap resamples; all 734 numeric fields matched. There were 263 first-attempt failures and no missing retry samples. |
| Omni-modal implementation checks | Syntax and trainer CLI checks; invented-data tests of missing-score baseline selection, identical train/test fallback semantics, shared method/seed panels, paired bootstrap differences, and leaf-table fitting passed. Type-plan generation and replay CLI also passed. |
| Quality-floor synthetic example | Price-path monotonicity, a quality-constrained fractional program, and stopping at the first failed certificate passed without datasets or network. |
| Learned-price CPU paths | Invented-data smoke: 100 fitting inputs, 40 test inputs, ten baseline paths, fifteen Frugal-style search chains, prompt parsing and aggregate output passed. This is not a full rerun of Tables 11–12. |
| Analytical regression | Four coordinate CSVs and the summary still matched the earlier archive. |
| Data download and runner | Download selection was checked with a mocked client against the legacy file inventory; explicit `--data-root` overrides dataset environment settings, while omission preserves them. |
| Release contents | The omni-modal directory contains source, Markdown and dependency text only. No original records, matrices, split assignments, fitted tables, media, or weights were included. |

Code/text numeric comparisons use relative tolerance `1e-8` and absolute tolerance `1e-10`. Full local certification runtimes were approximately 39 seconds (code) and 43 seconds (text); these are replay-analysis times, not inference latency. Missing-score tests use deliberately invented examples, not the proprietary matrix.

CI compiles Python, checks command-line entry points, reruns the analytical checks, and runs the omni-modal, quality-floor and learned-price synthetic checks on Python 3.12. It does not download datasets or model weights. GPU training, the complete private omni-modal evaluation, and unbundled image/winner-label recipes have not been rerun. Newly documented aggregate manuscript results must not be presented as newly reproduced private-matrix results.

## October 22, 2026 initial release

The following checks were executed on 2026-10-22 while preparing this release. Experiments ran in a separate copy so the archived scientific result files remained unchanged. No paid model inference was performed.

| Check | Outcome |
| --- | --- |
| Analytical figure generator | All exact finite-example and recurrence assertions passed; all four CSV coordinate tables and summary matched the archive |
| X2 portfolio coverage | Full replay completed; all 223,145 numeric result fields matched |
| X10 escalation chains | 259 chains, 1,000 task-bootstrap resamples, all four configurations; all 31,519 numeric result fields matched; three figures regenerated |
| X8 saved-record replay | 500 fitting and 334 evaluation tasks; full analysis and sensitivity stages completed without API credentials; published claims and scientific results matched |
| X12c direct-choice audit | All 3,095 archived calls analyzed; all 153 numeric audit fields matched |
| Verifier-price analysis | All 48 numeric result fields matched, including the Qwen threshold-0.5 break-even judge charge of $0.003300437 |

Numeric result comparisons used relative tolerance `1e-8` and absolute tolerance `1e-10`, with stricter comparisons for analytical coordinates. Structure and non-runtime descriptive fields were also compared. Typical local runtimes were approximately 8 seconds for X2, 139 seconds for X10, and 37 seconds for X8; these are CPU analysis runtimes, not model inference latency.

X8's omitted operational relay ledger changes four billing-unit diagnostic totals and removes six ledger metadata keys. Its quality, estimated dollar cost, latency, scientific claims, and analysis outputs match the recorded scientific results. Provider billing units are not part of the public replay contract.

The validation environment used Python 3.14.7, NumPy 2.4.2, pandas 3.0.1, SciPy 1.17.1, Matplotlib 3.10.8, scikit-learn 1.9.1, PyArrow 25.0.1, and HTTPX 0.28.1. The dependency file allows compatible versions; fonts and insignificant floating-point digits can differ across installations.

CI runs Python syntax compilation and the self-contained analytical reproduction on Python 3.12. It does not download benchmark data or call model services. Fresh inference, GPU stepwise/cache-transfer generation, and every historical bootstrap sweep have not been rerun for this release.
