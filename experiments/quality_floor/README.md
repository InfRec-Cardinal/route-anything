# Quality-floor certification on code and text

This directory implements the code/text certification experiments in Section 6.1 and Appendix C.2 of **Route Anything: an attempt for AGI through a simple omni-modal router** (2026-10-24). It reads the older CodeRouterBench and RouterBench outcomes. It contains no omni-modal outcome data, per-task predictions, or model weights, and makes no model calls.

The fitting sample determines a finite price path. An independent certification sample tests its routers in that order, stopping at the first failed test. If the first router fails, the procedure returns the reference. The quality floor is relative to that named reference and concerns aggregate expected quality; it does not promise improvement in every task category.

## Paper results and files

| Paper result | Implementation | Archived aggregate |
| --- | --- | --- |
| CodeRouterBench: 200 unstratified 3,540/3,540 probing halvings; median 15.17× Hoeffding, 18.83× empirical Bernstein, 19.38× refined empirical Bernstein; all 200 test splits meet the zero-loss floor | `crb.py`, `crb_core.py` | `reference/crb.json` |
| CodeRouterBench: the uncertified floor refitted on the fitting half meets the test floor in 12.5% of splits | Same experiment, `lpfloor_eps0` | `reference/crb.json` |
| RouterBench: 200 stratified fitting-sample halvings; median 6.70× empirical Bernstein; all 200 held-out evaluations meet the floor | `routerbench.py`, `certification.py` | `reference/routerbench.json` |
| RouterBench: fitted policies for eight families and 86 datasets, including the 7.10× family-state zero-loss route in Tables 5 and 12 | `routerbench.py`, `fitted_policies` output | Recomputed by the script |

The RouterBench script also retains Hoeffding, mixture-refined, dataset-state, and uncertified comparisons. The CodeRouterBench script retains the fitted price-path checks, three tolerances, Bonferroni controls, and state-wise controls. Its CLT rows are an asymptotic sensitivity analysis, not a finite-sample certificate.

## Run

Install the repository's `requirements.txt`. Use the released code/text input files described in `../../docs/data.md`; the CodeRouterBench input is `coderouterbench/id_results_long.csv`. With the documented default data layout:

```sh
python experiments/quality_floor/crb.py
python experiments/quality_floor/routerbench.py
```

For data stored elsewhere, point to the CodeRouterBench data directory or the locally obtained RouterBench file:

```sh
ACROUTER_DATA=/path/to/agent-as-a-router/data python experiments/quality_floor/crb.py
ROUTERBENCH_PKL=/path/to/routerbench_0shot.pkl python experiments/quality_floor/routerbench.py
```

RouterBench must be obtained separately from its upstream release. This repository does not redistribute that file. No download occurs in either experiment.

Defaults are 200 splits/halvings and 1,000 paired bootstrap replicates. Outputs go to `runs/quality_floor/crb.json` and `runs/quality_floor/routerbench.json`; set `QUALITY_FLOOR_OUT` to choose a different output file. For a short execution check on the same complete inputs:

```sh
NSPLIT=2 NBOOT=50 QUALITY_FLOOR_OUT=runs/quality_floor/crb-smoke.json python experiments/quality_floor/crb.py
RB_NHALV=2 NBOOT=50 QUALITY_FLOOR_OUT=runs/quality_floor/routerbench-smoke.json python experiments/quality_floor/routerbench.py
```

A separate example requires no datasets and no network:

```sh
python experiments/quality_floor/synthetic_demo.py
```

That example checks a two-state price path, a constrained fractional program, and the fixed-sequence stopping rule. Its synthetic numbers are not paper results.

## Provenance and validation

`crb_core.py` contains the shared definitions from the original `R8_theory/certified_floor.py`; `crb.py` retains Parts 1 and 2 of `certified_floor_v2.py`, the unstratified implementation used in the paper. `routerbench.py` retains the split, menu, policy, and certification calculations from `rb_policy/run.py`. `certification.py` exposes its data-independent solver and testing functions. Paths and output locations were made portable; scientific seeds and calculations were retained.

The full 200-split, 1,000-bootstrap runs were checked against the saved experiment outputs on 2026-10-24: 1,047 numeric fields for CodeRouterBench and 349 for RouterBench certification matched within floating-point tolerance. The archived files contain aggregate results only; RouterBench's per-halving records and bootstrap arrays are omitted from the reference file. The scripts can regenerate those local outputs.

This directory does not reproduce the nested 70/30 resplit study, prompt-predicted states, learned quality-cost routers, or the image/robot experiments. The code/text cascades use released-score acceptance and recorded fallback outcomes, so they are replay experiments rather than fresh online deployments.
