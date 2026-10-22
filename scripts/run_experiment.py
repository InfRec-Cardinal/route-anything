"""Run an offline paper experiment with consistent data locations."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "X2": "experiments/X2_residual_coverage/run.py",
    "X3": "experiments/X3_threshold_voi/run.py",
    "X4": "experiments/X4_memory_tail/run.py",
    "X5": "experiments/X5_markov_latency_barrier/run.py",
    "X6": "experiments/X6_allocation/run.py",
    "X7a": "experiments_x/X7a_regimes_offline/run.py",
    "X9": "experiments_x/X9_optimal_routing/run.py",
    "X10": "experiments_x/X10_escalation_chains/run.py",
    "X11": "experiments_x/X11_residual_coverage_routerbench/run.py",
    "X12a": "experiments_x/X12a_voi_released_signals/run.py",
    "X16": "experiments_x/X16_sensitivity/run.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    environment = os.environ.copy()
    environment.setdefault("ACROUTER_DATA", str(data_root / "agent-as-a-router/data"))
    environment.setdefault("ACROUTER_OUTPUTS", str(data_root / "agent-as-a-router/outputs"))
    environment.setdefault("ROUTERBENCH_PKL", str(data_root / "routerbench/routerbench_0shot.pkl"))
    environment.setdefault("MPLBACKEND", "Agg")
    command = [sys.executable, str(ROOT / EXPERIMENTS[args.experiment])]
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=environment))


if __name__ == "__main__":
    main()
