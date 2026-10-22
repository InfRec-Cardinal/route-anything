# Route Anything

**Experiment code for _Route Anything: A Theory of Inference Routers_.**

When is it worth paying for another model, a verifier, a routing signal, or a stored decision? This repository evaluates those choices using reference-relative quality and resource accounting. It contains the paper's analytical examples, benchmark replays, collection scripts, and recorded aggregate results.

[中文说明](README.zh-CN.md) · [Experiment guide](docs/experiments.md) · [Data and attribution](docs/data.md) · [Reproduction notes](docs/reproducibility.md)

## Quick start

Python 3.11 or newer is recommended. CPU replay uses NumPy, pandas, SciPy, Matplotlib, and scikit-learn. GPU generation is a separate optional experiment.

```bash
git clone https://github.com/InfRec-Cardinal/route-anything.git
cd route-anything
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce.py --outdir runs/analytical
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.

The analytical command needs no dataset or API key. It writes `allocation.pdf`, `memory.pdf`, four coordinate tables, and `summary.json`. Built-in assertions check the exact finite examples and recurrence formulas. These figures evaluate specified mathematical models; they are separate from benchmark measurements.

## Reproduce benchmark experiments

Obtain the datasets using the instructions in [docs/data.md](docs/data.md), then run from the repository root:

```bash
python scripts/run_experiment.py X2
python scripts/run_experiment.py X10
```

For data stored elsewhere:

```bash
python scripts/run_experiment.py X2 --data-root /path/to/data
```

The expected layout is `data/agent-as-a-router/data/coderouterbench/` and `data/routerbench/routerbench_0shot.pkl`. The runner also accepts the existing `ACROUTER_DATA`, `ACROUTER_OUTPUTS`, and `ROUTERBENCH_PKL` environment settings. Individual scripts expose further arguments where applicable.

Historical scripts write result files and figures into their experiment folders. Use a separate checkout to keep the recorded results untouched. Fixed seeds, fitting/evaluation splits, and original result files are retained so a rerun can be compared with the archive.

## Experiments

| Question | Experiments | Execution |
| --- | --- | --- |
| Which model adds useful portfolio coverage? | X2, X11; paired and repeated-split analyses | Offline benchmark replay |
| When should a failed attempt continue or escalate? | X3, X10, X7a; X7b feedback/retry | Offline replay; optional fresh collection |
| Does verification remain useful after its cost? | X8; verifier-price analyses | Archived verdict replay; optional fresh collection |
| Which finite-menu policy satisfies a quality target? | X9 | Offline fitting and held-out evaluation |
| Is a routing signal worth buying? | X12a, X12b, X12c | Released decisions and archived call replay |
| When do stored decisions help? | X13; X4 ideal-write control | Archived verdicts and simulated arrival streams |
| Where do simple allocation or cost models fail? | X6, X16; X5 latency sensitivity | Offline analysis |
| How does stepwise routing interact with cache transfer? | R6 | Optional GPU generation and recorded score analysis |

The [experiment guide](docs/experiments.md) gives entry points and input requirements. Collection scripts require your own compatible endpoint and credentials through `ROUTER_API_BASE_URL` and `ROUTER_API_KEY`. Historical model names and prices describe the archived runs; availability depends on your provider. Offline commands do not require a service account.

## Repository layout

```text
scripts/             analytical reproduction, data download, offline runner
experiments/         original X2–X6 studies and follow-up analyses
experiments_x/       X7–X13, X16, shared collection utilities, optional R6
numerical_results/   reference coordinates for the analytical figures
docs/                experiment map, provenance, reproduction and validation
```

Per-task inputs are distributed separately from the code. Aggregate results and figure assets remain beside the corresponding experiment. This release contains the experimental implementation, rather than the manuscript editing workspace.

## Reading the results

Efficiency is the ratio of mean delivered value to mean resource use. Dollar cost, elapsed time, and quality are reported separately. The reference model and evaluated population are part of each comparison.

The analyses are exploratory. Several experiments reuse tasks or constructed streams, so comparison counts are not independent sample counts. An oracle-verifier replay does not establish deployable verification, and a fitted quality constraint does not guarantee the same quality on unseen tasks. The archive retains negative results and model-fit limitations; see the [reproduction notes](docs/reproducibility.md).

## License

The project code is released under the [MIT license](LICENSE). Third-party datasets, model weights, and benchmark tasks retain their upstream terms; see [data attribution](docs/data.md).
