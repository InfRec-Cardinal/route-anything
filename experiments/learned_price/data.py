"""Read explicit caller-supplied matrices/prompts; never download data."""
import csv
import json
from pathlib import Path
import numpy as np
from source_core import paper_split, parse_prompt


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_crb(data_dir, prompts_path, responses_path=None):
    if prompts_path is None:
        raise ValueError("CRB id_tasks.jsonl has metadata only; supply --prompts with reconstructed prompt text.")
    if prompts_path.suffix == ".jsonl":
        prompts = {row["task_id"]: row["prompt"] for row in read_jsonl(prompts_path)}
    else:
        prompts = json.loads(prompts_path.read_text())
    matrix = {}
    with (data_dir / "coderouterbench" / "id_results_long.csv").open() as stream:
        for value in csv.DictReader(stream):
            identifier = value["task_id"]
            row = matrix.setdefault(identifier, {"id": identifier, "split": value["split"],
                                    "stratum": value["dimension"] + "|" + identifier.split("_")[1],
                                    "source": identifier.split("_")[1],
                                    "prompt": prompts.get(identifier, ""), "scores": {}, "costs": {}})
            row["scores"][value["model"]] = float(value["score"])
            row["costs"][value["model"]] = float(value["cost_usd"])
    if responses_path:
        responses = json.loads(responses_path.read_text())
        for identifier, row in matrix.items():
            row["responses"] = responses.get(identifier, {})
    records = [matrix[identifier] for identifier in sorted(matrix)]
    return records, sorted(records[0]["scores"])


def load_routerbench(path):
    import pandas as pd
    frame = pd.read_pickle(path)
    models = [model for model in frame.columns if f"{model}|total_cost" in frame.columns]
    records = [{"id": str(index), "prompt": parse_prompt(row["prompt"])[0],
                "stratum": str(row["eval_name"]),
                "scores": {model: float(row[model]) for model in models},
                "costs": {model: float(row[f"{model}|total_cost"]) for model in models}}
               for index, row in frame.iterrows()]
    return records, models


def split_records(records, profile, reference, seed):
    if profile == "crb":
        return (np.array([i for i, row in enumerate(records) if row["split"] == "probing"]),
                np.array([i for i, row in enumerate(records) if row["split"] == "id_test"]))
    names = np.array([row["stratum"] for row in records])
    score = np.array([row["scores"][reference] for row in records])
    fit = paper_split(names, score, seed)
    return np.flatnonzero(fit), np.flatnonzero(~fit)
