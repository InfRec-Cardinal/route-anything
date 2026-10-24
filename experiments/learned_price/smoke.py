"""Invented-data CPU checks; no paper data, downloads or model calls."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from source_core import parse_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert parse_prompt("['a', 'b']") == ("a\nb", True)
    records = []
    for index in range(140):
        scores = {"synthetic-reference": float(index % 7 < 5),
                  "synthetic-cheap": float(index % 5 < 3), "synthetic-other": float(index % 3 < 2)}
        records.append({"id": f"synthetic-{index}", "split": "probing" if index < 100 else "id_test",
                        "stratum": "synthetic-source-" + str(index % 2), "source": "synthetic-source-" + str(index % 2),
                        "prompt": f"Synthetic wrapper line.\nSolve synthetic task group {index % 11} operation {index % 7}.",
                        "scores": scores, "costs": {"synthetic-reference": .01, "synthetic-cheap": .001,
                                                       "synthetic-other": .002},
                        "responses": {name: f"Synthetic answer type {index % 13} from {name}." for name in scores}})
    data = args.out / "synthetic-input.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in records))
    report = args.out / "synthetic-aggregates.json"
    command = [sys.executable, str(Path(__file__).with_name("run.py")), "--profile", "crb",
               "--input", str(data), "--reference", "synthetic-reference", "--weak", "synthetic-cheap",
               "--methods", "ridge", "knn", "routellm", "frugal", "--limit-fit", "100", "--limit-test", "40",
               "--bootstrap", "30", "--out", str(report)]
    subprocess.run(command, check=True)
    result = json.loads(report.read_text())
    assert result["fit_rows"] == 100 and result["test_rows"] == 40
    assert len(result["methods"]) == 10
    assert result["methods"]["frugalgpt-style"]["selected"]["searched_chains"] == 15
    assert "task_id" not in report.read_text() and "responses" not in result
    print(json.dumps({"status": "passed", "synthetic": True,
                      "checks": ["RouterBench list-string parsing", "all CPU baseline paths", "aggregate-only result"]}))


if __name__ == "__main__":
    main()
