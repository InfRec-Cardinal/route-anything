# Release validation

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
