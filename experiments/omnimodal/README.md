# Omni-modal routing experiments

This directory releases the actual model, loss, training and evaluation code used
for the October Route Anything experiments, with portable paths and imports.
It contains **code only**. The original outcome matrix, task inputs, media,
capability labels, fitted tables, predictions and checkpoints are not included.
The generated synthetic examples are invented interface checks, not a
reproduction of the paper's numerical results.

| Component | Included implementation | Paper setting |
| --- | --- | --- |
| `capability_tree/` | Multimodal leaf classifier, task-scoped classes plus unresolved class, train-only paired least-squares leaf table, training and replay | Learned capability-tree routing (157-leaf experiment) |
| `outcome/` | Shared task–candidate quality head, BCE plus four times pairwise score-gap MSE, LoRA training and checkpoint inference | Outcome-supervised target study |
| `type_router/` | Task-only soft-label capability classification and global train-only type-to-model table | Matched capability-type target study |
| `robot/` | Multimodal encoder with independent success logits, scene-mean BCE, distributed training and outcome replay | Seven-policy RoboTwin success prediction |
| `matched_evaluate.py` | Train-mean fixed baseline, shared observed-outcome panel, within-input seed averaging, group bootstrap | Portable implementation of the revised comparison rules |

The underlying trainers are extracted source, not synthetic replacements.
`prepare_plans.py`, `matched_evaluate.py` and `synthetic_demo.py` are release
adapters. They do not reconstruct the original data preparation or predictions.
The text outcome/type study uses the original **text projection** of each input;
it does not load raw images. The capability tree and robot trainers do load
images through the Qwen processor.

## Install and check without data

Python 3.10+ is required. CPU schema/replay checks need only NumPy:

```bash
python -m pip install numpy
python experiments/omnimodal/synthetic_demo.py --out /tmp/route-anything-synthetic
```

This checks that 11 missing scores among 1,122 invented training records do not
disqualify the stronger fixed model; all methods and seeds share one panel; and
the Opus-to-Astra fallback changes fixed-policy fitting when appropriate.
It also checks panel completeness after fallback and runs the original paired
least-squares leaf-table calculation. It creates
synthetic tree, text and robot schemas under the requested directory. The
synthetic robot rows have no camera images and are not robot benchmark data.

For actual training, install `requirements.txt` and use a Transformers build
that exposes `Qwen3_5ForConditionalGeneration` and the Qwen3.5 processor.
Model weights can be a local model directory or a Hub model identifier; loading
a Hub identifier may download weights. Training requires appropriate GPU memory.
No GPU training or full paper rerun was performed while preparing this release.

```bash
python -m pip install -r experiments/omnimodal/requirements.txt
```

Run each entry as a script, as below; the source trainers use local module imports.

## Capability-tree training

Supply `train.jsonl`, `dev.jsonl`, `test.jsonl`, your frozen `tree.json`, and
labels in the schema described in [SCHEMA.md](SCHEMA.md). Image paths are
absolute or relative to `--data-dir`. The taxonomy is an input; this release
does not provide the original 157 leaf definitions or labels.

```bash
python experiments/omnimodal/capability_tree/fit_table.py \
  --train /path/to/tree-data/train.jsonl --tree /path/to/tree.json \
  --out /path/to/tree-table.json
python experiments/omnimodal/capability_tree/train.py \
  --data-dir /path/to/tree-data --tree /path/to/tree.json \
  --table /path/to/tree-table.json --model-path Qwen/Qwen3.5-4B \
  --out runs/tree-base-mixed
```

Defaults retain LoRA rank 8 / alpha 16 / dropout .05, 3 epochs, effective batch
16, encoder/head learning rates 1e-4/1e-3, seed 42, maximum sequence length
8,192, at most 256 tokens per image, total image budget 2,048, and 16 images.
The classifier sees task content, observed images/state, output requirements
and scoped capability definitions. It does not receive candidate outcomes.
The selected checkpoint minimizes dev annotation cross-entropy.

Use `--benchmark NAME` for single-benchmark training. A mixed-then-individual
run starts from `--init-checkpoint runs/tree-base-mixed/checkpoint` with the
same base `--model-path`. A separately pretrained full SFT initialization is
passed as `--model-path`, with `--processor-path` if it lacks processor files.
Evaluation uses `--eval-only --init-checkpoint CHECKPOINT`. The original
multi-host scheduling and data-labeling services are not included.

Leaf fitting retains the source within-identical-input repeat means and
least-squares model contrasts. Disconnected comparisons remain unassigned.
The benchmark fallback was updated to the revised paper's highest **observed
training mean**, without a full-coverage eligibility restriction; the old
benchmark-specific fixed-model override was removed. Retired Fable models are
excluded from new fitting and historical Fable selections map to Opus.

## Matched outcome and type training

Supply the same split IDs and normalized candidate scores to both methods.
Supply the frozen, output-family-aware type taxonomy and input projections.
`prepare_plans.py` takes already prepared labels; it does not call a teacher.

```bash
python experiments/omnimodal/prepare_plans.py --mode outcome \
  --data-dir /path/to/text-data --taxonomy /path/to/taxonomy.json \
  --model Qwen/Qwen3.5-4B --out /path/to/outcome-plan
python experiments/omnimodal/outcome/train.py \
  --plan /path/to/outcome-plan/plan.json --seed 42 --out runs/outcome-42
python experiments/omnimodal/prepare_plans.py --mode type \
  --data-dir /path/to/text-data --taxonomy /path/to/taxonomy.json \
  --model Qwen/Qwen3.5-4B --out /path/to/type-plan
python experiments/omnimodal/type_router/train.py \
  --plan /path/to/type-plan/plan.json --seed 42 --out runs/type-42
```

Repeat both with seeds `42`, `13`, `2026`, and `7`. Plans retain six shuffled
epochs, effective batch 32, LoRA rank 16 / alpha 32 / dropout .05,
encoder/head learning rates 2e-5/3e-4, weight decay .01, gradient clipping 1,
5% warmup and cosine decay to 10% of peak. Each pair receives its task's
normalized score; missing outcomes are masked. Type targets are normalized
requirement weights and use soft CE. Both use the final checkpoint, not a
test-selected checkpoint. Outcome training remains a CUDA entry point.

Outcome plan creation tokenizes inputs and stores the exact token counts for
each shuffled stream. It requires the tokenizer. The release adapter uses all
training rows as its diagnostic panel; the original frozen diagnostic subset
and original stream artifacts are not distributed.
Its prompt uses the source adapter's single `total_quality` definition and
fixed weight `[1.0]`; type labels are not outcome-model inputs.

```bash
python experiments/omnimodal/outcome/evaluate.py \
  --plan /path/to/outcome-plan/plan.json --checkpoint runs/outcome-42/checkpoint \
  --split test --out runs/outcome-42/replay
python experiments/omnimodal/type_router/evaluate.py \
  --plan /path/to/type-plan/plan.json --checkpoint runs/type-42/checkpoint \
  --split test --out runs/type-42/replay
```

## Robot success prediction

Supply the original-style, task-family-disjoint train/dev/test splits, three
initial RGB cameras, state and instruction for each scene, all seven binary
policy outcomes, and `models.json` with candidate order and a fixed model
chosen from training outcomes. In the paper, train/dev/test task families were
disjoint; splitting random scenes from the same family is not equivalent.

```bash
torchrun --standalone --nproc_per_node=4 experiments/omnimodal/robot/train.py \
  --data-dir /path/to/robot-data --model-path Qwen/Qwen3.5-4B \
  --out runs/robot-success
```

For one GPU, replace the `torchrun` prefix with `python`. The original defaults
are LoRA rank 8, 3 epochs, global batch 16, encoder/head rates 1e-4/1e-3,
seed 42, sequence length 8,192, and 256 tokens per image. The seven independent
sigmoid heads use unweighted BCE averaged within scene and then across scenes.
All-success and all-failure scenes remain supervised. Dev routing success
selects the checkpoint, with lower BCE and then earlier epoch breaking ties.
Test is evaluated after selection. This replays recorded outcomes; it does not
launch a simulator or a robot. The later clean-only training experiment is
deliberately not part of this paper implementation.

## Compare predictions on one panel

Convert supplied records and predictions to the short replay schema in
[SCHEMA.md](SCHEMA.md). The evaluator selects the fixed arm from train only,
then compares all supplied methods/seeds on identical test inputs. Missing
fixed or selected outcomes remove the input for every method. The default
also requires complete declared candidate outcomes after applying the configured
fallback for the oracle panel.

```bash
python experiments/omnimodal/matched_evaluate.py \
  --records /path/to/aligned-records.jsonl \
  --predictions /path/to/model-choices.jsonl --out runs/matched.json
```

For the target study's explicitly declared Opus-to-Astra missing-outcome
fallback use `--opus-fallback`. This flag applies the same policy both when
choosing the fixed model on train and when evaluating all methods on test.
The candidate universe is collected across training rows, so an entirely
absent Opus key on a training row still uses that row's Astra outcome. Default
behavior without this flag remains the mean of each model's observed outcomes.
Panel completeness is checked after fallback, so a missing Opus outcome with
an observed Astra outcome remains eligible. Use `--allow-partial-candidate-panel`
only when the intended panel allows other missing unselected candidates; report
this option with results. The oracle is then limited to observed candidates. Seeds are
averaged within input before paired group bootstrap. Intervals condition on
the supplied predictions and do not measure retraining uncertainty. Pooled
means across different benchmark metrics are descriptive, not a universal
quality scale.

## Source and coverage boundaries

The first-party extracted sources are from the authors' `A-RouteEverything`
experiment tree; directory names identify the source revision:

- `experiments/capability_tree_router_matrix_20261019/`: model, trainer,
  routing policy, and the `paired_choice`/`fit_table` preparation functions.
- `experiments/robotwin_router_success_20261020/`: model and trainer.
- `experiments/2026-10-17/full-data-capability-vs-instance-4b/`: outcome model,
  loss, trainer and checkpoint verification.
- `experiments/capability_router_20261018/`: type model, loss, table mapping,
  trainer, evaluator and checkpoint verification.
- `experiments/2026-10-18/single-capability-vs-instance-4b/evaluate_scalar.py`:
  standalone outcome checkpoint evaluator.

PyTorch, Transformers, PEFT, NumPy and Pillow remain external dependencies
under their own licenses. Qwen weights and benchmark materials retain their
own terms. No third-party implementation was vendored into this directory.

Not packaged here: the 25-benchmark matrix ingestion/export layer, the fixed
rule-tree study, the complete 124-configuration image-router suite, the 12
winner-label full-finetuning recipes, teacher annotation APIs, private cluster
schedulers, original taxonomy/labels, model weights, and paper result files.
The image suite's source is `experiments/2026-10-12/execution-20261015/`
(including `methods_report.py`, method subdirectories and `train_scaling.py`).
The winner-label sources are `experiments/qwen35_4b_trajectory_transfer_20261017/`
and `experiments/qwen35_4b_joint_sft_20261018/`; they have additional pretrained
checkpoints and corpus dependencies. Their absence means this folder alone
does not reproduce every paper table.
