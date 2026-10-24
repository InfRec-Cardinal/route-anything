"""Download the legacy code/text replay bundle for the paper's Appendix C.

This downloads CodeRouterBench records and related routing experiments only.
The omni-modal outcome matrix, raw/derived records, media, splits and checkpoints
are not public and are not downloaded by this command. RouterBench's matrix
must be obtained separately from its upstream release.
"""

import argparse
from pathlib import Path

DEFAULT_REPO_ID = "Lance1573/route-anything-data"

# Keep the legacy replay layout explicit; other dataset contents are not inputs
# to this command. In particular, do not select the entire data/ directory.
REPLAY_FILES = [
    "data/agent-as-a-router/data/coderouterbench/id_results_long.csv",
    "data/agent-as-a-router/data/coderouterbench/id_probing_results_long.csv",
    "data/agent-as-a-router/data/coderouterbench/id_test_results_long.csv",
    "data/agent-as-a-router/data/coderouterbench/ood176_results_long.csv",
    "data/agent-as-a-router/data/coderouterbench/id_tasks.jsonl",
    "data/agent-as-a-router/data/coderouterbench/id_probing_tasks.jsonl",
    "data/agent-as-a-router/data/coderouterbench/id_test_tasks.jsonl",
    "data/agent-as-a-router/data/coderouterbench/ood176_tasks.jsonl",
    "data/agent-as-a-router/data/coderouterbench/models.json",
    "data/agent-as-a-router/data/id/voter_decisions/*.jsonl",
    "data/agent-as-a-router/data/matrices/phase1_acrouter_v2/response_matrix.json",
    "data/agent-as-a-router/outputs/acrouter_ood176/ood_decisions.jsonl",
    "data/agent-as-a-router/outputs/acrouter_ood176/ood_metrics.json",
    "data/agent-as-a-router/outputs/baselines_ood176/decisions/*.jsonl",
    "data/agent-as-a-router/outputs/current/id_decisions.jsonl",
    "data/agent-as-a-router/outputs/current/id_metrics.json",
    "data/agent-as-a-router/outputs/current/ood_sandbox_decisions.jsonl",
    "data/routerbench/README.md",
    "experiments_x/X7b_inloop_live/*.jsonl",
    "experiments_x/X8_operating_point/rows_probing.csv",
    "experiments_x/X8_operating_point/rows_id_test.csv",
    "experiments_x/X8_operating_point/cache/samples.json",
    "experiments_x/X8_operating_point/cache/sandbox.json",
    "experiments_x/X8_operating_point/cache/calibration.json",
    "experiments_x/X8_operating_point/cache/judge.json",
    "experiments_x/X12b_llm_signal/signal_rows.csv",
    "experiments_x/X12b_llm_signal/signal_calls_*_partial.jsonl",
    "experiments_x/X12c_jev_policy/jev_calls_official.jsonl",
    "experiments_x/X12c_jev_policy/two_primary/jev_calls_*.jsonl",
    "experiments_x/X13_memory/cache/verdicts.jsonl",
    "experiments_x/X13_memory/cache/judge_verdicts*.jsonl",
    "experiments_x/X13_memory/cache/judge_meta*.json",
    "experiments_x/common/tasks.jsonl",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id", default=DEFAULT_REPO_ID,
        help=f"explicitly select another compatible code/text replay bundle (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument("--revision", default="main", help="dataset revision (default: main)")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1],
        help="destination checkout (default: this repository)",
    )
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.root.expanduser(),
        allow_patterns=REPLAY_FILES,
    )
    print(f"Legacy code/text replay inputs are available under {args.root.expanduser().resolve()}")


if __name__ == "__main__":
    main()
