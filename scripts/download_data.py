"""Download the experiment replay files into this checkout."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repository")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.root,
        allow_patterns=["data/**", "experiments_x/**"],
    )
    print(f"Replay inputs are available under {args.root.resolve()}")


if __name__ == "__main__":
    main()
