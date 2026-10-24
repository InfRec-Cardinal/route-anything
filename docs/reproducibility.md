# Reproducing the October 24, 2026 manuscript

The current manuscript is *Route Anything: an attempt for AGI through a simple omni-modal router*. Its main empirical study uses a proprietary omni-modal outcome matrix. This code release includes reusable implementations without distributing that matrix or its derived artifacts. The older public code/text testbeds support a separate part of the paper.

## Three kinds of reproduction

| Run | Required resources | What it establishes |
| --- | --- | --- |
| Synthetic and analytical checks | CPU dependencies only | Numerical identities and implementation behavior on artificial inputs |
| Code/text replay and quality-floor certification | Public CodeRouterBench/replay records; separately obtained RouterBench | The corresponding §6 / Appendix C calculations on recorded outcomes |
| Omni-modal training and evaluation | Separately authorized input records, original splits, media, GPU environment, model weights | Matrix-dependent results only when the exact original experimental inputs are available |

No full omni-modal benchmark rerun or GPU retraining is claimed by this release. See [validation](validation.md) for the commands actually executed. The downloader does not provide the omni-modal matrix.

## Data and populations

CodeRouterBench has eight models on 9,999 coding tasks: 7,080 probing tasks and 2,919 test tasks. The reference is `claude-opus-4-6`. The X9 state-dependent policy uses 21 dimension×source cells and 22 actions: eight direct dispatches and fourteen primary-to-reference cascades. A released score supplies oracle acceptance in this replay.

RouterBench has eleven models on 36,497 prompts, with `gpt-4-1106-preview` as the reference. Individual analyses have their own documented split and fitting rules. Do not replace a script's split with another experiment's split merely because both say 70/30.

The omni-modal matrix spans 25 benchmarks, 103,307 tasks, 374,401 scored runs, and 452 arm versions in the supplied manuscript. Its 104,050 routing decision records and its 19,814-input headroom test panel count different objects. SciCode is omitted from that complete-input headroom panel; it appears in other matched router studies. The capability-tree, robot, target-comparison, and winner-label panels must not be merged. [The paper map](paper-map.md) lists their sample counts.

## Quality floors and uncertainty

A fitted quality floor constrains training estimates. Certification fixes the candidate sequence using fitting data, tests it on independent certification inputs, stops at the first failure, and otherwise falls back to the reference. The new [quality-floor implementation](../experiments/quality_floor/README.md) follows the original code/text experiments, including their empirical-Bernstein tests and mixture-refined candidate paths.

The manuscript reports 200 random halvings of the fitting sample. These reuse the same underlying dataset and held-out test sample; 200 is not a count of independent benchmark datasets. The code result of 200/200 certified routes meeting the test floor and a median 19.4× efficiency is an empirical observation under that protocol. Formal certification also requires the proposition's matched outcomes, independent calibration sample, and deployment-law assumptions.

Quality is stored on a 0–1 scale in the code/text analyses. Main-text quality differences are generally reported in points on a 0–100 scale; Appendix C tables sometimes retain 0–1 differences. Efficiency is mean value divided by mean resource, and relative efficiency divides that ratio by the reference's ratio. Router inference overhead and list-price sensitivities are separate calculations, not silently included in every replay.

## Learned-router prompt provenance

The archived CPU TF-IDF implementation and the October manuscript do not fully agree on preprocessing. The source scripts whose recorded figures match several Table 11 rows read reconstructed prompts with their loader-written instruction wrappers; the manuscript groups those rows under a wrapper-removal heading. The [learned-price release](../experiments/learned_price/README.md) preserves the source protocol and exposes stripping as an explicit option. Enabling it defines a different run. Without the exact frozen prompt projection, a new run must not be advertised as reproducing those table entries. This update documents the discrepancy; it does not change the manuscript or invent replacement results.

## Matched baselines

Select the fixed arm on training inputs and evaluate it on the same panel as each router. Do not remove a candidate simply because it misses a few training scores. In the 11-benchmark target study, the manuscript uses the same Opus-to-Astra missing-score fallback for router and fixed baseline. On test, every compared policy must have a score under the chosen observation/fallback rule. The [omni-modal implementation notes](../experiments/omnimodal/README.md) explain which parts of this protocol are implemented and the input contract.

Bootstrap intervals are paired over the stated sampling units. Omni-modal comparisons use source groups (robot scenes for RoboTwin); some historical code/text analyses use tasks, others constructed streams. Pointwise bootstrap intervals across 124 image-routing configurations are exploratory, not multiplicity-adjusted confirmation.

## Earlier experiments and known limitations

The [September analysis archive](reproduction-september-2026.md) preserves detailed seed schedules, grading diagnostics, missing-cost conventions, original analytical checks, and analysis history. Its table and appendix references belong to the older manuscript. Current entry points and scope are in the [experiment guide](experiments.md).

CodeRouterBench's missing outcomes, zero-latency rows, grading issues, and OOD cost imputations remain relevant; updates to the paper do not retroactively repair the released upstream records. Retries and verification share benchmark test mechanisms in some protocols. The memory study combines real outcomes with simulated arrivals and retention. Fresh-call replay cannot establish provider identity beyond the archived requested model names.

## Output locations

Use a separate checkout or a script's output-directory option for new runs. Historical scripts overwrite aggregate files and figures next to their code. Private input directories and generated artifacts should live outside the tracked code tree. Nothing in these instructions uploads data, trains a model, or makes paid calls automatically.
