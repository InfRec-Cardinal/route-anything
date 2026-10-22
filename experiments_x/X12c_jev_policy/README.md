# X12-c — Jev pre-execution Choice policy

This experiment evaluates Jev as an ex-ante dispatcher. Each request sends the task prompt, source and dimension metadata, and short descriptions of the eight released backends. Jev returns one typed `Choice`. Candidate outputs, benchmark scores, labels, and backend prices are not sent to Jev; the selected backend is scored offline against the released outcome matrix.

The archived official run uses the complete 2,919-task `id_test` split and all 176 OOD tasks: 3,095 calls, zero errors, 3,084,845 Jev input tokens, and an estimated Jev input charge of $0.12956 at $0.042 per million input tokens. The request alias `jev-latest` resolved to `jev-1.13.0`.

## Reproduce the audit

The live calls are preserved in `jev_calls_official.jsonl`; rerunning the audit does not call the API:

```sh
ACROUTER_DATA=/path/to/agent-as-a-router/data \
python3 audit_official.py
```

This writes `audit_results.json`, `summary_for_paper.md`, and `figures/fig1_jev_policy.pdf`. It uses 1,000 paired task-bootstrap resamples for the policy and fixed-backend comparisons. The original live-call script is `run.py`; it requires `TYPESAFE_API_KEY` and should only be run when a fresh API pass is intended.

The primary ID result is mean score 0.3884 and $0.000662 total cost per task, versus 0.3967 and $0.000525 for fixed Qwen3-Max. On OOD176 the Jev policy scores 0.2670 at $0.075656 per task, versus 0.3295 at $0.023903 for fixed Qwen3-Max. These are offline replays of recorded backend outcomes, not regenerated answers and not a post-output verifier experiment.

The manuscript uses `../../figures/jev_policy.tex`, a vector plot of all eight fixed backends and Jev from `audit_results.json`. It compiles directly with pgfplots; the audit's standalone PDF remains an optional reproduction output.

