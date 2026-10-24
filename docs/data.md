# Code/text replay data and provenance

## Distribution status

The public replay bundle at [Lance1573/route-anything-data](https://huggingface.co/datasets/Lance1573/route-anything-data) covers the earlier **code/text testbeds**: CodeRouterBench and related routing, retry, verification, signal and memory experiments. These support the current paper's Appendix C. RouterBench is obtained separately from its upstream source.

The paper's **omni-modal outcome matrix is not public**. Its raw and derived records, media, split files and trained checkpoints are not included in the public data release and are not downloaded by `scripts/download_data.py`. The code/text bundle does not reproduce the paper's 25-benchmark omni-modal studies.

The code/text release includes analytical coordinates and recorded aggregate results. The analytical quick start needs no dataset. Code/text experiments use the public upstream releases below; experiments that depend on the authors' archived calls also need the legacy replay bundle or new collection using your own service.

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

## Download the legacy code/text replay bundle

The legacy bundle preserves the code's paths under `data/` and `experiments_x/`. The downloader selects CodeRouterBench outcome tables, task metadata, released decisions and recorded code responses; X7b attempts; X8 samples, sandbox results and judge verdicts; X12b forecasts; X12c direct-choice calls; X13 verdicts; and the shared code-task cache. The dataset card records each file and source.

The legacy file `data/agent-as-a-router/data/matrices/phase1_acrouter_v2/response_matrix.json` contains **code-benchmark responses**. Its name does not refer to the new omni-modal outcome matrix.

Download the code/text inputs into the checkout with:

```bash
python scripts/download_data.py
```

The default repository is `Lance1573/route-anything-data`. To choose another compatible **code/text** bundle explicitly, pass `--repo-id OWNER/DATASET_NAME`; `--revision` selects its revision and `--root` selects the destination checkout.

The downloader uses an explicit list of legacy input paths rather than selecting all of `data/` or `experiments_x/`. It does not fetch the omni-modal matrix, optional R6 stepwise records, or dataset documentation over this repository's README. Per-task inputs and caches are excluded from Git. Obtain the RouterBench matrix directly from its upstream release and place it at the path shown above.

## Task reconstruction and new collection

The shared loader can reconstruct original tasks from local Hugging Face snapshots. It respects `HF_HUB_CACHE` and `HF_HOME_HUB`; its `FIELD_MAP` in `experiments_x/common/sources.py` records the source-specific mapping. Source reconstruction is needed by some historical collection and replay scripts. Use the individual experiment instructions for the required inputs.

New code/text model collection uses `ROUTER_API_BASE_URL` and `ROUTER_API_KEY`. Recorded backend scores remain replayed unless the experiment explicitly reruns those backends. Optional R6 generation uses a separate GPU stack and public model checkpoints, documented in its experiment folder; this is independent of the private omni-modal matrix.

## Attribution and terms

The upstream Agent-as-a-Router code and CodeRouterBench release declare MIT terms. Their attribution is retained in the replay bundle. Original benchmark tasks and model weights retain their own upstream terms. The code repository's MIT license does not replace those terms.

The inspected RouterBench snapshot did not contain an explicit dataset license. This release points to its upstream distribution instead of republishing that matrix. Obtain it from the source above and preserve its accompanying terms.

## Interpretation

The original CodeRouterBench matrix includes missing runs encoded as zero-cost failures and zero-latency rows. The scripts preserve those values unless a named sensitivity excludes or imputes them. See [reproduction notes](reproducibility.md) for population counts, grading diagnostics, price assumptions, and the distinction between fixed-matrix replay and new inference.
