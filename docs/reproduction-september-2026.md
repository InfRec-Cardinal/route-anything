# September 2026 analysis archive

These notes describe the earlier code/text campaign. Section, table, and appendix labels below refer to that manuscript revision, not the October 24 manuscript. R6, allocation, and constructed-process examples are historical extensions. For current scope use [reproduction notes](reproducibility.md) and the [paper map](paper-map.md).

# Route Anything: reproduction notes

This file describes the experiments retained in the paper. Experiment identifiers preserve their archive names, while the manuscript groups them by scientific question. Algebraic checks and imposed-process simulations are not additional empirical experiments. Reproducing results requires the corresponding source datasets and call records; the analytical command does not run benchmark experiments. The analyses are exploratory, were not preregistered, and do not use a multiple-comparison correction.

## Data and versions

- CodeRouterBench: the released Agent-as-a-Router `data/coderouterbench/id_results_long.csv`, 9,999 tasks and eight models; probing 7,080 and ID Test 2,919. The accompanying `data/coderouterbench/models.json` supplies token prices. Always-Opus has ID Test mean score 0.4383 and total estimated cost $34.02.
- Released ex-ante router decisions: the nine files in `data/id/voter_decisions/`; ACRouter's released ID lookup decisions and OOD176 escalation traces are used separately.
- RouterBench: the released 0-shot outcome matrix, 36,497 prompts and 11 models. X11 uses a source-and-reference-success-stratified 70/30 split of 25,542 and 10,955 prompts. Other experiment splits follow their own scripts; they should not be silently substituted for X11's split.
- CodeRouterBench contains 148 missing Qwen3-Max runs represented by score 0 and cost 0, and 3,150 zero-latency rows. Dollar runs keep the missing outcomes unless an exclusion sensitivity is named. Most latency runs exclude invalid entries, but X9 retains the original latency entries.
- BigCodeBench has 750 all-zero tasks among 1,097. `experiments/grading_diagnostic/` retains the relevant local-verification evidence: 91 of 318 nonempty responses pass on 40 sampled all-zero tasks. Its focused script recomputes the matrix counts and verdict summary, and can optionally recheck the same sample from separately supplied response records through the subprocess sandbox.
- OOD176 has 168 cheap-model steps and one fallback step without execution records among 469 replay steps. The released matrix treats these as zero-cost failures; this does not establish that real calls would be free. X7-a's cost sensitivity imputes the model's mean recorded old112 cost, changing the chain's efficiency ratio from 2.59 [1.97, 3.54] to 1.75 [1.41, 2.23]. The provider's new64 subset has constant imputed cost per model and four models' rows originating from other source models. The old112 imputed-cost ratio is 1.22 [0.92, 1.65], so the pooled result should not imply a uniformly established advantage.

The manuscript uses the archived September 2026 experiment outputs named below. X12-c includes an archived Jev run (3,095 calls); no model calls are made while compiling this manuscript. The data filenames, population sizes, model names, split rules, and price basis identify the experimental inputs in the paper.

## Experiment entry points

The original statistics scripts use NumPy, pandas, SciPy, and Matplotlib. They read fixed matrices and write `results.json`, their experiment `README.md`, and vector figures. Historical runtimes below are statistics-script runtimes, not inference latency.

| Directory under `experiments/` | Entry | Historical runtime | Scope |
|---|---|---:|---|
| `X2_residual_coverage` | `run.py` | 7 s | Portfolio coverage |
| `X3_threshold_voi` | `run.py` | 14 s | Supplementary single-attempt thresholds and information value |
| `X4_memory_tail` | `run.py` | 47 s | Ideal-write control accompanying X13 |
| `X5_markov_latency_barrier` | `run.py` | 139 s | Archived construction checks; only latency sensitivity retained in the paper |
| `X6_allocation` | `run.py` | 33 s | Real coverage curves and allocation |

The release includes the retained analysis versions. The analysis-history section below records changes relevant to interpreting their estimates.

The retained X7–X13 and X16 analyses use directories under `experiments_x/`, with scripts, recorded results, experiment READMEs, and figures. The main-text studies combine X2 + X11, X9, X10 + X7-b, X8, X12-a + X12-b, and X13. X9 opens the main-text section on additional inference; X3, X6, X7-a, X12-c, and X16 remain supplementary. X13's full protocol is accompanied by X4 as its ideal-write control.

The five experiments containing fresh model calls are X7-b, X8, X12-b, X12-c, and X13. Their models include DeepSeek-v4-pro, DeepSeek-flash, Qwen3-Max, Qwen3.5-Plus, Kimi-K2.5, Grok-4.6, and Jev. Every Opus and Sonnet value is a recorded outcome. `experiments_x/common/` contains the configurable client, task loader, sandbox, and judge; service credentials, account ledgers, and shared response-cache databases are excluded from the release. X12-c's retained per-task outputs support its direct-choice replay. It makes 3,095 direct-choice calls over the complete 2,919-task ID Test split and OOD176, with 0 errors and 3,084,845 input tokens; its estimated input charge is $0.12956 at $0.042/M. The X8 judge's dollar charge uses the primary's blended token-price proxy; X12-b's forecast charge uses DeepSeek's peak list price; X12-c charges input tokens separately and scores backend outcomes offline.

X3 additionally writes `a_threshold_cells.csv`, `b2_refinement_swap_cells.csv`, `c_voi_band_cells.csv`, and `d_blackwell_cells.csv`. X6 writes `summary_table.csv`. The main-text tables draw on the retained experiments' recorded results; their captions identify the comparison and evaluation population.

The newly exposed X9 quality and cost columns are taken from `experiments_x/X9_optimal_routing/README.md`, Results (2) and (3): the unconstrained 21-cell rule has quality difference −0.0201 [−0.0343, −0.0056] and mean cost $0.00039452; fitting with a zero-loss floor gives −0.0044 [−0.0175, +0.0097] and $0.00045218. Its 26.6% bootstrap floor-satisfaction rate describes the held-out resamples, not an independently calibrated success probability.

## Seeds and resampling

The retained scripts under `experiments/` use seed 20260911; the X7–X13 and X16 campaign uses 20260917, subject to the experiment-specific resampling seeds below. Later direct-choice runs record their own protocol in X12-c's README.

| Experiment | Bootstrap or simulation details |
|---|---|
| X2 | 500 task resamples |
| X3 | 1,000 per-cell task resamples |
| X4 | 500 task-cluster and seed resamples; 50 record streams, seed 20260911; cluster seed 777, seed-bootstrap seed 99; 300-stream retention replay seed 555; Zipf seed 4242; arm-reveal seed 31337 |
| X5 | 500 resamples; 100-round blocks for streams; alternate audit seed 20260911 + 99 |
| X6 | 1,000 evaluation resamples and 500 fit resamples |
| X9 | 1,000 paired test-task resamples and 1,000 probing resamples |
| X11 | 500 held-out task resamples |
| X13 | 50 streams of 60,000 rounds, stationary window the last 10,000 rounds; paired stream bootstrap |
| X12-c | Full ID Test and OOD176 direct-choice replay; `audit_official.py` adds 1,000 paired task-bootstrap resamples for the policy and fixed-Qwen comparisons; Jev input usage is recorded in `results_official.json` and `audit_results.json` |

Prior archived reruns reported zero numerical mismatches with their stored outputs. This is a reproduction claim about those archived runs, not a new rerun performed while condensing the manuscript.

## Analysis history

These changes were made after initial results. The paper presents them as exploratory analysis, not preregistered confirmation.

- X3: the first C5 construction changed the primary contingent on the router signal and did not instantiate the stated fixed-refinement hypothesis. The later analysis added a non-contingent primary swap, the closed-form information comparison, and C7 baselines. The swap's conditional-value caveats remain.
- X4: after the first run, record-probability comparisons acquired a standard-error floor and a 0.005 tolerance because coverage values of 0 or 1 produced zero seed standard errors. Initial coverage was 81% of bins. Retention moved to a dedicated 300-stream replay, with FIFO measured when the queue was full; the initial FIFO figure was 0.83. The Zipf construction grew from 4,000 to 20,000 cells after first-run deviations of 0.033/0.051 at exponent 1.5. A bookkeeping correction removed 18 undefined FIFO points previously counted as failures, changing FIFO interval coverage from 90.7% to 94.8%.
- X5: B2/B5 were also reported with a variance calculation that adds test-pool uncertainty; D2 gained a per-pair inconclusive category; D2b was reported per pair as well as over all pairs; D1's summary changed from failure to partial support. An alternate draw of the same ten random half-splits changed A3's verdict, showing the original ±1%/90% rule was sensitive to sampling noise. These constructed Markov/barrier experiments are omitted from the condensed paper.
- X6: the uniform baseline's alphabetical allocation of remainder slots was replaced by an average over rotations, correcting its value by 0.0024/0.0041. The literal closed form with `d_k = 1` replaced the free-intercept fit as the headline. Both results remain worse than the probing-selected integer-optimal allocation when evaluated on held-out tasks; the paper retains that limitation.
- X2: the normalized paired-gain rule was added after inspecting the original residual-rule results. It reuses the same fitting and held-out task split and is an exploratory follow-up, not a preregistered comparison. The original residual-rule diagnostic on excluded trivial cases also received a description correction.

## Dependence and alternative readings

`experiments/router_selection_recheck.py` independently reconstructs RouterBench portfolio outcomes from the released matrix. Run it with `--data data/routerbench/routerbench_0shot.pkl`; it writes the summary, original-split comparison records, and a readable report alongside the script. Table 1 uses all ten rerun seeds, the consecutive integers 20260924 through 20260933, fixed before their results were examined; these are seed identifiers, not dates. No seeds are selected by their results. Each split uses 25,542 fitting and 10,955 held-out tasks with the original 70/30 evaluation-source-by-reference-success stratification, candidate filters, base portfolios, and five rules; a normalized-residual diagnostic is reported separately. All selections use fitting tasks only. Table 1 pools 5,539 retained split–context–base-portfolio comparisons at threshold 0.5 and 212 at threshold 1, which excludes binary-score contexts. The pooled hit rate, mean relative regret, and mean relative gain weight every retained comparison equally. The JSON also preserves equal-split means as a separate sensitivity diagnostic. The original seed 20260917 remains a control and is excluded from the main aggregate. These splits reuse the same tasks and nested contexts, so the pooled counts do not denote independent tasks or independent test sets. The report includes every rerun and both thresholds.

X4's 128 stationary-sign configurations contain four underlying cases: uninformed prior × resource basis. It serves as a control within X13; retention configurations share streams. X3's exact non-positive-margin branch uses one physical cell repeated across primary/price settings; four noisy discrepancies come from one 32-task cell. X2's binary-score dimensions duplicate the two threshold settings. Noisy-verifier CIs in X3 and the historical latency analysis condition on fixed task-level error draws. X10's sign-agreement CIs hold the decided set and fitted predictions fixed while resampling tasks. Its 57,405 decided segment comparisons share tasks and chains; the non-degenerate subset contains 34,746 comparisons.

X12-a cross-fits released binary signals within ten half-splits of the labeled test population because the release has no corresponding probing decisions. It is a diagnostic information-value estimate, not a router fitted only on probing. The dimension-conditioned capture divides task-weighted gross savings by the task-weighted perfect-information bound. X12-b instead fits its online forecast on probing and evaluates on held-out tasks.

X6's literal and free-intercept fits answer different model assumptions. The supplementary allocation analysis retains the observed failure of the common-exponent and concavity assumptions. X7-b's online feedback and final scoring use the same sandbox tests, and X8's hidden-test verifier uses the benchmark's scoring mechanism. These protocols establish their stated comparisons, rather than an independent hidden-test generalization claim.

## Portable data configuration

Run scripts from the repository root, and point them to separately obtained benchmark releases. Most original scripts accept `ACROUTER_DATA` (the directory containing `coderouterbench/` and `id/`), `ACROUTER_OUTPUTS` (released output directory), `ACROUTER_RELEASE_DECISIONS` (ID decision file), and `ROUTERBENCH_PKL` (the RouterBench matrix). New analyses expose equivalent file or directory arguments in `--help`; their output paths default to their experiment directory. The package replaces machine-specific defaults with relative locations. Task identifiers and released model names are retained unchanged.

NumPy, pandas, SciPy, Matplotlib, scikit-learn, and PyArrow support the offline analyses; the collection client additionally uses HTTPX. Source-task reconstruction reads locally available benchmark snapshots. The sandbox executes generated code in bounded subprocesses. Replaying archived API outputs additionally requires the relevant call records; aggregate results in the package do not reconstruct omitted raw response caches.

New collection requires the caller's `ROUTER_API_BASE_URL` and `ROUTER_API_KEY` environment settings. No provider address or local credential file is bundled. The historical model availability, price assumptions, and token limits describe the recorded experiments and are not guarantees about a newly configured service.

The step-wise routing and cache-transfer instance in Appendix `app:stepwise-instance` is in `experiments_x/R6_stepwise/`: generation, C2C, sandbox scoring and analysis scripts under `code/`, and per-continuation scores (gzipped JSONL), latency fits and analysis summaries under `results/`. It ran open-weight Qwen3 models with vLLM and the released `nics-efc/C2C_Fuser` checkpoint on H20 GPUs; raw token outputs are not archived.
