# Supplementary test-suite check for archived retries

No new model calls were made. The analysis uses full responses and requests from the archived cache, and runs responses only through the existing subprocess sandbox. The original feedback and final scores used the base HumanEval tests. A separate EvalPlus suite was retained in the task metadata and was not passed to the models or used for retry feedback.

The archive contains 13 paired fresh/feedback retries on six HumanEval or HumanEval-completion tasks. Before scoring responses, each supplementary suite was checked against its bundled canonical solution. The cached HumanEval/32 suite raises a `TypeError` for that solution because its special polynomial assertion incorrectly unpacks a scalar return. That task and its four pairs are excluded; the suite was not modified. The valid analysis has nine task–model pairs on five distinct tasks, with 27 rescored outputs including the first attempts. It covers 9 of the 229 first-failure pairs in the original five-model comparison.

| Root model | Valid pairs | Fresh passes | Feedback passes |
| --- | ---: | ---: | ---: |
| Qwen3.5-Plus | 2 | 0 | 1 |
| Qwen3-Max | 2 | 0 | 1 |
| Kimi-K2.5 | 1 | 0 | 0 |
| DeepSeek-v4-pro | 1 | 0 | 0 |
| DeepSeek-flash | 3 | 0 | 2 |
| Total | 9 | 0 | 4 |

All nine initial outputs fail the supplementary suite. Four feedback retries and no fresh retries pass, matching their base-test pass counts on this valid subset. The aggregate paired difference is 4/9; a task-cluster bootstrap gives [0.20, 0.80], but five tasks provide very limited information. These results should be presented as a small supplementary check, not as a replacement for the full feedback-test evaluation or evidence about all benchmark sources.

The supplementary suites contain 12–1,006 cases per retained task and overlap the original base tests; literal-call matching finds 3–10 shared cases per retained suite. Thus these are augmented suites unused for feedback, not disjoint holdout suites. The original request archive confirms that no full supplementary suite was sent. The collection code constructs feedback solely from the base verifier's error and traceback, and those tracebacks can reveal failing assertions despite the 800-character cap.

Broader evaluation of independent scoring requires separating feedback tests from scoring tests before new retries are collected, including the MBPP and BigCodeBench tasks that account for most current failures. Existing outputs can support further external-suite rescoring where a compatible, verified unused suite exists, but splitting the already-used feedback suite afterward cannot establish an untouched scoring set.

The companion JSON records the eligible pairs, canonical checks, per-output verdicts, request checks, suite overlap, and task-cluster bootstrap. The script reproduces the analysis without making model requests.

