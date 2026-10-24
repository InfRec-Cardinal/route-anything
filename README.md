# Route Anything

**Experiment code for _Route Anything: an attempt for AGI through a simple omni-modal router_ (manuscript dated October 24, 2026).**

A router chooses which model, generator, or robot policy should run for an input. The paper studies how much routing can gain, what learned routers recover, and when routing pays through price. Its theory chooses the arm with the highest expected value minus priced cost, with a reference policy and a quality tolerance fixing the operating point.

[中文说明](README.zh-CN.md) · [Paper-to-code map](docs/paper-map.md) · [Experiment guide](docs/experiments.md) · [Data](docs/data.md) · [Reproduction notes](docs/reproducibility.md) · [Validation](docs/validation.md)

## Release scope

| Setting | What this repository provides | Data availability |
| --- | --- | --- |
| Omni-modal routing, §§3–5 and Appendix B | Released router implementations and evaluation utilities; see [implementation coverage](experiments/omnimodal/README.md) | **The omni-modal outcome matrix is proprietary and is not released.** |
| Priced code/text testbeds, §6 and Appendix C | Offline replay, quality-floor certification, collection utilities, and archived aggregate results | CodeRouterBench and the existing code/text replay bundle are public; RouterBench is obtained separately from upstream. |
| Historical extensions | Analytical examples, allocation, latency, and optional stepwise experiments | Retained for continuity; they are not all experiments in the October manuscript. |

No omni-modal matrix records, task-level derived tables, split files, embeddings, media, or trained checkpoints are included in the code release or uploaded to Hugging Face. The existing [Hugging Face replay bundle](https://huggingface.co/datasets/Lance1573/route-anything-data) covers the earlier code/text studies; it is **not** the paper's 25-benchmark omni-modal matrix. Reproducing the matrix-based empirical scores requires separately authorized data. Synthetic examples check code behavior, not those scores.

## Findings in the current manuscript

These are reported manuscript results, not claims that the public checkout recreates the private evaluation:

- On 19,814 complete-panel test inputs from 24 benchmarks, the per-input oracle scores 87.98 versus 82.60 for the training-selected fixed arm. The underlying matrix spans 25 benchmarks; SciCode is excluded from this headroom panel.
- The capability-tree router scores 79.98 versus 79.68 on its separate 3,444-input panel. Outcome regression reaches 71.55% robot success versus 42.8% for winner-label prediction.
- The fitted zero-loss code route reaches 25.5× reference efficiency with a 0.44-point test quality loss. Independent floor certification holds on test in all 200 reported splits at a median 19.4×. The empirical split count is not a universal guarantee.

Evaluation panels have different inputs and baselines. [The paper map](docs/paper-map.md) records section and table correspondence, units, and reproduction limits.

## Quick start

Use Python 3.11 or newer for CPU analyses:

```bash
git clone https://github.com/InfRec-Cardinal/route-anything.git
cd route-anything
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce.py --outdir runs/analytical
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`. The analytical command needs no dataset or API key. It checks finite examples and recurrence formulas and writes figures, four coordinate tables, and a summary. These are mathematical illustrations retained from the earlier release.

For the current manuscript's learned routers and certification studies, follow [omni-modal code](experiments/omnimodal/README.md) and [quality-floor certification](experiments/quality_floor/README.md). GPU training dependencies are separate from CPU replay.

## Reproduce public code/text experiments

Download the existing code/text records, then run an offline experiment:

```bash
python scripts/download_data.py
python scripts/run_experiment.py X2
python scripts/run_experiment.py X10
```

RouterBench must be obtained separately as described in [data instructions](docs/data.md). For an existing data directory:

```bash
python scripts/run_experiment.py X2 --data-root /path/to/data
```

The expected layout includes `data/agent-as-a-router/data/coderouterbench/` and `data/routerbench/routerbench_0shot.pkl`. Historical scripts write into their experiment folders; use a separate checkout for reruns if you want to preserve the archived outputs. See the [experiment guide](docs/experiments.md) for inputs and commands supported by each study.

Fresh collection requires an explicitly configured service through `ROUTER_API_BASE_URL` and `ROUTER_API_KEY`. Recorded model names, prices, and relay identities describe the historical experiment. Offline replay needs no service account.

## Repository layout

```text
experiments/omnimodal/      router code; no proprietary matrix or artifacts
experiments/quality_floor/  independent quality-floor certification on code/text
experiments/learned_price/  CPU learned quality/cost routers and baselines
experiments/disjoint_retry/ disjoint feedback/scoring retry analysis
experiments/               X2–X6 and follow-up code/text analyses
experiments_x/             X7–X13, X16, collection utilities, historical R6
scripts/                   analytical checks, code/text download and replay runner
numerical_results/         analytical reference coordinates
docs/                      paper map, data scope, reproduction and validation
```

Efficiency is the ratio of mean quality to mean resource use, not a mean of per-task ratios. Costs in dollars and time are separate resources. A fitted floor and an independently certified floor answer different questions; oracle verification is a replay assumption, not a deployed verifier. The repository preserves negative results and limitations alongside positive ones.

## License and citation

The project code uses the [MIT license](LICENSE). Dataset, benchmark, and model licenses remain with their upstream sources. The code license grants no access to the proprietary matrix. Use [CITATION.cff](CITATION.cff) for the current manuscript authors and title.
