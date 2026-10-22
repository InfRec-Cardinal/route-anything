# Data and provenance

## Distribution status

The code release includes aggregate results and analytical coordinates. A separate replay bundle has been prepared for Hugging Face, but its upload is pending because the Hub connection was unavailable during publication. There is no live Route Anything dataset URL in this release yet.

The analytical quick start works without the bundle. The core outcome-matrix experiments can use the public upstream releases below. Experiments that depend on the authors' archived calls need the replay bundle, or new collection using your own service.

## Public inputs

| Input | Source | Role |
| --- | --- | --- |
| CodeRouterBench | [Hugging Face](https://huggingface.co/datasets/Lance1573/CodeRouterBench) | Task metadata, model scores, token-price costs and latency |
| Agent-as-a-Router | [GitHub](https://github.com/LanceZPF/agent-as-a-router) | Released routing decisions, OOD traces and model metadata |
| RouterBench | [Upstream release](https://github.com/withmartian/alt-routing-methods/tree/public-productionize) | Zero-shot prompt-by-model outcome matrix |

CodeRouterBench contains 79,992 model/task rows for 9,999 tasks and eight models. Its released partition is 7,080 probing tasks and 2,919 ID evaluation tasks. The OOD population has 176 tasks. RouterBench contains 36,497 prompts; X11 uses its own source-by-reference-success-stratified split of 25,542 fitting and 10,955 evaluation prompts.

Keep the upstream directory layout:

```text
data/
  agent-as-a-router/
    data/
      coderouterbench/
        id_results_long.csv
        ood176_results_long.csv
        id_tasks.jsonl
        ood176_tasks.jsonl
        models.json
      id/voter_decisions/
    outputs/
  routerbench/
    routerbench_0shot.pkl
```

For example, if the Agent-as-a-Router release is already available locally, set `ACROUTER_DATA` to its `data` folder and `ACROUTER_OUTPUTS` to its released `outputs` folder. Set `ROUTERBENCH_PKL` to the 0-shot matrix. The runner's `--data-root` accepts the parent directory shown above.

## Replay bundle

The prepared bundle preserves the code's paths under `data/` and `experiments_x/`. It includes X7b attempts and recorded responses; X8 samples, sandbox results and judge verdicts; X12b forecasts; X12c direct-choice calls; X13 verdicts; and the upstream outcome tables and decision records needed to join them by task ID. The dataset card records each file and source. Optional R6 contains score records, not the historical raw token continuations.

After publication, download a dataset repository into the checkout with:

```bash
python scripts/download_data.py --repo-id OWNER/DATASET_NAME
```

Replace `OWNER/DATASET_NAME` with the published identifier. The downloader selects only `data/` and `experiments_x/`, so the dataset card does not replace this repository's README. Per-task files and caches are excluded from Git.

## Task reconstruction and new collection

The shared loader can reconstruct original tasks from local Hugging Face snapshots. It respects `HF_HUB_CACHE` and `HF_HOME_HUB`; its `FIELD_MAP` in `experiments_x/common/sources.py` records the source-specific mapping. Source reconstruction is needed by some historical collection and replay scripts. Use the individual experiment instructions for the required inputs.

New model collection uses `ROUTER_API_BASE_URL` and `ROUTER_API_KEY`. Recorded backend scores remain replayed unless the experiment explicitly reruns those backends. The R6 generation scripts require their separately documented GPU stack and model checkpoints.

## Attribution and terms

The upstream Agent-as-a-Router code and CodeRouterBench release declare MIT terms. Their attribution is retained in the replay bundle. Original benchmark tasks and model weights retain their own upstream terms. The code repository's MIT license does not replace those terms.

The inspected RouterBench snapshot did not contain an explicit dataset license. This release points to its upstream distribution instead of republishing that matrix. Obtain it from the source above and preserve its accompanying terms.

## Interpretation

The original matrix includes missing runs encoded as zero-cost failures and zero-latency rows. The scripts preserve those values unless a named sensitivity excludes or imputes them. See [reproduction notes](reproducibility.md) for population counts, grading diagnostics, price assumptions, and the distinction between fixed-matrix replay and new inference.
