# Implementation Summary for Partner

This document explains the current LoRA GPT-2 Medium E2E project as it exists in code. It is meant as a practical onboarding guide for someone who has not read the repository yet.

The short version: the repository now contains a working from-scratch LoRA adapter implementation, GPT-2 attention injection utilities, E2E tokenization helpers, adapter-only checkpoint helpers, generation/evaluation helpers, CLI entrypoints, and unit tests. It still intentionally blocks real training from the CLI until dataset/model access and compute are approved.

## Project Goal

The project is a course replication of the core idea from "LoRA: Low-Rank Adaptation of Large Language Models" using:

- Base model: Hugging Face `gpt2-medium`.
- Task: E2E NLG restaurant data-to-text generation.
- Adaptation method: freeze GPT-2 Medium and train only low-rank LoRA matrices.
- First target setting: LoRA on GPT-2 attention query and value projections with rank `r=4`, alpha `32`, dropout `0.1`.

The code does not depend on `peft` or `loralib` for the core LoRA logic. Those libraries are only used as research references in the docs.

## Top-Level Structure

```text
lora-gpt2-medium-e2e/
  README.md
  requirements.txt
  pyproject.toml
  configs/
    e2e_gpt2_medium_lora.yaml
  docs/
    implementation_plan.md
    implementation_summary_for_partner.md
    paper_summary.md
    official_lora_repo_notes.md
    loralib_implementation_notes.md
    gpt2_e2e_reference_pipeline.md
    extension_ideas.md
  src/lora_gpt2/
    __init__.py
    config.py
    utils.py
    lora_layers.py
    inject.py
    modeling.py
    data.py
    train.py
    checkpointing.py
    generate.py
    evaluate.py
  scripts/
    prepare_e2e.py
    train.py
    generate.py
    evaluate.py
    count_params.py
    smoke_test.py
  tests/
    test_lora_layers.py
    test_inject.py
    test_data.py
    test_checkpointing.py
    test_generation.py
    test_train.py
    test_evaluate.py
```

Generated or external artifacts are ignored by git:

- `data/`: raw and processed E2E files.
- `outputs/`: run outputs, predictions, metrics.
- `checkpoints/`: model or adapter checkpoints.
- `external/`: external metric scripts, such as official E2E evaluation code.
- `.venv/`, Python caches, logs, `.env`.

## Config and Packaging

`pyproject.toml` configures the project as a standard `src/` layout package. It also tells pytest to add `src` to `PYTHONPATH` and to discover tests from `tests/`.

`requirements.txt` lists the expected experiment dependencies:

- `torch`
- `transformers`
- `datasets`
- `accelerate`
- `pyyaml`
- `tqdm`
- `numpy`
- `sacrebleu`
- `rouge-score`
- `evaluate`
- `pytest`

`configs/e2e_gpt2_medium_lora.yaml` is the central experiment config. Important sections:

- `project`: run name, output directory, seed.
- `model`: base model and tokenizer names, currently `gpt2-medium`, plus GPT-2 pad-token handling.
- `lora`: target model family, target GPT-2 `attention.c_attn` slices, rank, alpha, dropout, and merge behavior.
- `data`: raw/processed data paths, max token length, prompt template, EOS handling, and prompt-label masking.
- `training`: optimizer, learning rate, batch size, epochs, warmup, label smoothing, gradient clipping, mixed precision, and checkpoint cadence.
- `generation`: decoding settings such as beam count, max new tokens, length penalty, no-repeat n-gram size, and EOS token.
- `evaluation`: metric names and default prediction/reference file locations.

Path values in the config are resolved relative to the project root by `src/lora_gpt2/config.py`.

## Core Package Modules

### `config.py`

Loads YAML config files and attaches two convenience fields:

- `_config_path`: the config path that was loaded.
- `_project_root`: the project root inferred from the config location.

It also provides `resolve_path(config, value)` so scripts can use config-relative paths consistently.

### `utils.py`

Small shared helpers:

- RNG seeding for Python, NumPy, and PyTorch.
- Device selection, preferring CUDA when available.
- Parameter counting and trainable-parameter reporting.
- Directory creation.
- JSONL read/write helpers.

### `lora_layers.py`

This is the core from-scratch LoRA implementation.

`LoRALinear` wraps a normal `torch.nn.Linear` layer:

- Freezes the original base layer.
- Adds trainable `lora_A` and `lora_B` matrices.
- Uses scaling `alpha / rank`.
- Initializes `A` randomly and `B` to zeros, so the wrapped layer initially matches the frozen base layer exactly.
- Supports optional merge/unmerge behavior for evaluation.

`LoRAQKVConv1D` is the GPT-2-specific wrapper for Hugging Face's fused attention projection `attn.c_attn`:

- GPT-2 stores query, key, and value in one Conv1D-style projection.
- The wrapper keeps the original `c_attn` projection frozen.
- It splits the output into query/key/value slices.
- It applies LoRA updates only to enabled slices.
- The config currently enables query and value, and leaves key unchanged.
- Like `LoRALinear`, it initializes with zero delta because every `lora_B` matrix starts at zero.

This module is the most important place to understand the math of the implementation.

### `inject.py`

Owns the model surgery that inserts LoRA into a GPT-2-style model.

Main flow in `inject_lora_into_gpt2`:

1. Read rank, alpha, dropout, merge behavior, and Q/K/V target booleans from config or explicit arguments.
2. Freeze the whole model with `freeze_base_model`.
3. Iterate over `model.transformer.h`, the Hugging Face GPT-2 block list.
4. Replace each block's `block.attn.c_attn` with `LoRAQKVConv1D`.
5. Mark only parameters whose names contain `lora_` as trainable.
6. Return an `InjectionReport` with replaced-module count, total parameters, trainable parameters, and trainable parameter names.

The module also provides:

- `mark_only_lora_as_trainable`, following the official LoRA convention.
- `lora_state_dict`, which extracts adapter weights only.
- `expected_qv_lora_parameters`, used for quick sanity checks.

For GPT-2 Medium with hidden size `1024`, `24` layers, rank `4`, and query/value LoRA, the expected trainable adapter count is:

```text
24 layers * 2 slices * (4 * 1024 + 1024 * 4) = 393216 trainable parameters
```

This is close to the paper's reported 0.35M trainable-parameter scale.

### `modeling.py`

Loads Hugging Face assets and applies injection:

- `load_tokenizer(config)`: loads the configured tokenizer and sets GPT-2 padding to EOS if requested.
- `load_base_model(config)`: loads `AutoModelForCausalLM.from_pretrained`.
- `load_lora_model(config)`: loads base model and tokenizer, injects LoRA, and returns `(model, tokenizer, report)`.

Important: these functions may download GPT-2/tokenizer files if they are not already cached locally.

### `data.py`

Owns E2E parsing, prompt formatting, tokenization, datasets, and batching.

Current raw input formats:

- Text lines formatted as `context||completion`.
- JSONL records with fields like `context`/`completion`, `mr`/`target`, or `ref`.

Tokenization flow for each example:

1. Format the meaning representation with the config prompt template.
2. Optionally add a leading space to the target, which helps GPT-2 tokenization.
3. Tokenize prompt and target separately.
4. Optionally append EOS to the target.
5. Concatenate prompt tokens and target tokens.
6. Truncate to `max_length`.
7. Set labels to `-100` for prompt tokens when `mask_prompt_labels` is true.
8. Keep target token IDs as labels.

`DataCollatorForCompletionOnlyLM` dynamically pads `input_ids`, `attention_mask`, and `labels`. Label padding uses `-100`, so the loss ignores padding.

### `train.py`

Contains reusable training utilities, but not the final full experiment driver yet.

Implemented utilities:

- `create_optimizer`: builds AdamW over trainable parameters only.
- `create_linear_scheduler`: wraps Hugging Face's linear warmup/decay scheduler.
- `shifted_label_smoothed_loss`: causal LM loss with optional label smoothing and `-100` ignore labels.
- `forward_loss`: runs the model and computes the shifted loss.
- `train_one_epoch`: one epoch of gradient accumulation, clipping, optimizer/scheduler steps.
- `evaluate_loss`: validation loss without gradient updates.

The training utilities are real, but the CLI still blocks actual training. See the training guard section below.

### `checkpointing.py`

Handles adapter-only checkpoints.

`save_adapter_checkpoint(path, model, config, extra)` writes:

- `adapter_state_dict`: only keys containing `lora_`.
- `config`: lightweight config metadata.
- `extra`: optional run metadata.

`load_adapter_checkpoint(path, model, strict=False)` loads adapter weights into an already LoRA-injected model.

The intended usage is:

1. Recreate/load GPT-2.
2. Inject the same LoRA structure.
3. Load the saved adapter weights.

The checkpoint does not save the full frozen GPT-2 base model.

### `generate.py`

Generation helpers:

- `generate_continuations`: tokenizes prompt-only inputs, calls `model.generate`, slices off the prompt tokens, and returns decoded continuations.
- `extract_continuation`: removes prompt text and common stop markers from already-decoded text.
- `format_predictions`: writes stable prediction records with `id`, `prompt`, and `prediction`.

Generation uses the decoding settings from `config["generation"]`.

### `evaluate.py`

Metric helpers:

- `corpus_bleu`: quick `sacrebleu` corpus BLEU sanity check.
- `read_prediction_texts`: reads either plain text predictions or JSONL prediction records.
- `write_metric_summary`: writes metrics as JSON.
- `run_official_e2e_evaluator`: calls an external `measure_scores.py` script if installed under the configured `external/e2e-metrics` directory.

The official E2E evaluator is not vendored in this repository.

## CLI Scripts

All scripts insert `src/` into `sys.path`, so they can run from the project root without installing the package editable.

### `scripts/prepare_e2e.py`

Tokenizes raw E2E splits.

Expected command shape:

```bash
python scripts/prepare_e2e.py --config configs/e2e_gpt2_medium_lora.yaml
```

Default splits are `train`, `valid`, and `test`. It looks for raw files under `data/raw/e2e/` by default, accepting either `.jsonl` or `.txt`, and writes processed JSONL files under `data/processed/e2e_gpt2/`.

This script loads the configured tokenizer, so it may download GPT-2 tokenizer files if they are not cached.

### `scripts/train.py`

Training entrypoint with an intentional guard.

Current behavior:

- If run without `--dry-run`, it exits immediately with a message saying training is blocked.
- If run with `--dry-run`, it loads GPT-2/tokenizer, injects LoRA, builds the optimizer, prints parameter counts, and exits.

Because `--dry-run` still calls `load_lora_model`, it may download GPT-2 if not cached. Do not use it in no-download environments.

### `scripts/generate.py`

Generates predictions from a trained adapter checkpoint.

Expected command shape after a real adapter exists:

```bash
python scripts/generate.py \
  --config configs/e2e_gpt2_medium_lora.yaml \
  --split test \
  --adapter outputs/runs/e2e_lora_r4_alpha32/adapter.pt
```

Flow:

1. Load config.
2. Load GPT-2 and inject LoRA.
3. Load adapter checkpoint into the injected model.
4. Load raw E2E records for the selected split.
5. Build prompts from meaning representations.
6. Generate continuations.
7. Write JSONL predictions to the configured predictions file.

This script requires GPT-2/model access, raw data, and a trained adapter.

### `scripts/evaluate.py`

Evaluates generated predictions.

Expected command shape after predictions and references exist:

```bash
python scripts/evaluate.py --config configs/e2e_gpt2_medium_lora.yaml
```

By default it computes quick sacreBLEU and writes a `.metrics.json` file beside the predictions file.

With `--official`, it also calls the external official E2E evaluator:

```bash
python scripts/evaluate.py --config configs/e2e_gpt2_medium_lora.yaml --official
```

That requires `external/e2e-metrics/measure_scores.py` and any dependencies required by the official evaluator.

### `scripts/count_params.py`

Reports expected LoRA parameter counts without loading GPT-2:

```bash
python scripts/count_params.py --config configs/e2e_gpt2_medium_lora.yaml
```

It assumes GPT-2 Medium hidden size `1024` and `24` layers, then computes the expected query/value LoRA parameter count.

Optional model-loading mode:

```bash
python scripts/count_params.py --config configs/e2e_gpt2_medium_lora.yaml --load-model
```

Only use `--load-model` when GPT-2 downloads/cache access are allowed.

### `scripts/smoke_test.py`

Runs a no-training dry smoke test using tiny in-memory stand-ins:

```bash
python scripts/smoke_test.py --config configs/e2e_gpt2_medium_lora.yaml --dry-run
```

It verifies:

- A tiny QKV LoRA wrapper initially matches the frozen base projection.
- The tokenized dataset and collator produce matching label/input shapes.
- Adapter checkpoint save/load works.

It does not load GPT-2 or the E2E dataset.

## Tests

The test suite uses tiny fake models/tokenizers where possible, so it is safe to run without downloading GPT-2 or E2E data.

Current coverage:

- `test_lora_layers.py`: zero-initialized LoRA matches base output, disabled key slice remains unchanged, merge behavior matches unmerged eval output.
- `test_inject.py`: GPT-2-like `c_attn` modules are replaced, base parameters are frozen, and only LoRA parameters are trainable.
- `test_data.py`: E2E parsing, prompt-label masking, EOS appending, dynamic padding, and ignored label padding.
- `test_checkpointing.py`: adapter-only state dict excludes base weights and checkpoint round trip restores LoRA weights.
- `test_generation.py`: prompt/stop-marker cleanup and stable prediction record formatting.
- `test_train.py`: shifted label-smoothed loss handles ignored prompt positions.
- `test_evaluate.py`: JSONL prediction reading.

Run all tests:

```bash
python -m pytest
```

## End-to-End Data Flow

The intended full experiment flow is:

1. Put raw E2E splits under `data/raw/e2e/`.
2. Run `scripts/prepare_e2e.py` to tokenize examples into `data/processed/e2e_gpt2/`.
3. Load GPT-2 Medium and tokenizer through `modeling.py`.
4. Inject LoRA adapters into every GPT-2 attention `c_attn` projection through `inject.py`.
5. Build train/validation datasets and dataloaders from tokenized examples.
6. Train only LoRA parameters using utilities from `train.py`.
7. Save adapter-only checkpoints through `checkpointing.py`.
8. Recreate GPT-2 plus the same LoRA structure for generation.
9. Load the adapter checkpoint.
10. Generate E2E test predictions through `scripts/generate.py`.
11. Evaluate predictions with quick BLEU and, later, official E2E metrics.

The implemented code covers most of these pieces. The main missing integration is the full training CLI that wires tokenized data, dataloaders, scheduler step counts, validation, logging, and periodic adapter checkpoint saving into one approved training run.

## LoRA Injection Flow

This is the most important model path:

1. `load_lora_model(config)` in `modeling.py` loads the base GPT-2 model and tokenizer.
2. It calls `inject_lora_into_gpt2(model, config=config)`.
3. `inject_lora_into_gpt2` freezes every existing parameter.
4. It walks through `model.transformer.h`, which is Hugging Face GPT-2's list of transformer blocks.
5. For each block, it replaces `block.attn.c_attn` with `LoRAQKVConv1D`.
6. `LoRAQKVConv1D` keeps the original frozen fused QKV projection as `base_layer`.
7. It creates separate low-rank A/B matrices for enabled slices.
8. The config enables query and value LoRA, so key remains frozen and unchanged.
9. `mark_only_lora_as_trainable` ensures only parameters with `lora_` in their names can receive gradients.
10. The returned `InjectionReport` is the first sanity check that replacement count and trainable parameter count make sense.

The initial wrapped model should behave like the base model because all `lora_B` matrices are zero-initialized. Training then learns nonzero low-rank deltas.

## Training Boundary and No-Training Guard

The repository intentionally prevents accidental full training right now.

The guard lives in `scripts/train.py`:

- Running `python scripts/train.py --config ...` exits before model setup or training.
- The message tells you to use `--dry-run` or remove the guard only after explicit approval.
- Running `--dry-run` still loads GPT-2/tokenizer, injects LoRA, builds the optimizer, and prints basic sanity information.

This means the current safe no-download checks are tests, `scripts/smoke_test.py`, and `scripts/count_params.py` without `--load-model`.

Do not remove the guard until the team is ready to:

- Confirm compute budget.
- Confirm dataset location and format.
- Confirm GPT-2 cache/download policy.
- Add or approve the full training driver.
- Decide checkpoint/output naming.

## Adapter Checkpointing

The project uses adapter-only checkpoints rather than saving GPT-2 Medium.

Saving:

```text
save_adapter_checkpoint(path, model, config, extra)
```

This stores only LoRA parameters from `lora_state_dict(model)`, plus lightweight metadata. Base GPT-2 weights are intentionally excluded.

Loading:

```text
load_adapter_checkpoint(path, model, strict=False)
```

The caller must first create a model with the same LoRA structure. In practice:

1. Load GPT-2 Medium.
2. Inject LoRA with the same config.
3. Load adapter weights.

This pattern keeps checkpoint files small and matches the LoRA paper's practical deployment story.

## Generation and Evaluation Flow

Generation starts from a trained adapter:

1. `scripts/generate.py` loads config.
2. It loads GPT-2 and injects LoRA.
3. It loads the adapter checkpoint.
4. It reads raw E2E records for a split.
5. It formats prompts from meaning representations.
6. It calls `generate_continuations`.
7. It writes JSONL records with `id`, `prompt`, and `prediction`.

Evaluation reads those predictions:

1. `scripts/evaluate.py` reads the configured prediction file.
2. It reads the configured reference file.
3. It computes quick sacreBLEU.
4. It writes a JSON metrics summary.
5. If `--official` is passed, it calls the external official E2E evaluator.

One practical detail to check before real evaluation: `scripts/evaluate.py` currently reads references with field name `completion`, while the config points to `data/processed/e2e_gpt2/references_test.txt`. Before final evaluation, make sure the reference file format matches what `read_prediction_texts` expects, or adjust the script/reference export path.

## Safe Verification Commands

From the project root:

```bash
python -m pytest
```

Runs the unit tests. This should not download GPT-2 or datasets.

```bash
python scripts/smoke_test.py --config configs/e2e_gpt2_medium_lora.yaml --dry-run
```

Runs a tiny no-training smoke test. This should not download GPT-2 or datasets.

```bash
python scripts/count_params.py --config configs/e2e_gpt2_medium_lora.yaml
```

Prints the expected GPT-2 Medium query/value LoRA trainable parameter count without loading GPT-2.

Avoid these until downloads or local cache access are approved:

```bash
python scripts/prepare_e2e.py --config configs/e2e_gpt2_medium_lora.yaml
python scripts/train.py --config configs/e2e_gpt2_medium_lora.yaml --dry-run
python scripts/count_params.py --config configs/e2e_gpt2_medium_lora.yaml --load-model
python scripts/generate.py --config configs/e2e_gpt2_medium_lora.yaml --adapter <adapter.pt>
```

Those commands may require GPT-2 files, raw E2E data, or a trained adapter.

## What Remains Before Real Training

Before running the actual experiment, the main remaining work is:

- Obtain or point to the E2E raw train/validation/test splits under `data/raw/e2e/`.
- Decide whether raw files will be official text format or JSONL, and keep one consistent format.
- Run and inspect tokenization output from `scripts/prepare_e2e.py`.
- Create/export the exact reference file format expected by evaluation.
- Wire the full training CLI: processed datasets, dataloaders, scheduler total steps, validation loop, logging, checkpoint cadence, and adapter saving.
- Decide whether to use mixed precision through `torch.cuda.amp` or `accelerate`.
- Add a small integration test around the future training driver that still uses tiny fake data/model pieces.
- Confirm GPT-2 Medium can be loaded from cache or downloaded in the approved environment.
- Run `scripts/train.py --dry-run` only after model access is allowed.
- Remove or bypass the no-training guard only after explicit approval.
- Run the first real training job and save adapter-only checkpoints.
- Generate validation/test predictions from the saved adapter.
- Install or vendor the official E2E metrics under `external/e2e-metrics/` if official BLEU/NIST/METEOR/ROUGE-L/CIDEr numbers are needed.
- Add ablation configs after the baseline works, such as rank, alpha, target slices, dropout, or merge behavior.

## Recommended Reading Order

For onboarding, read code in this order:

1. `configs/e2e_gpt2_medium_lora.yaml` to understand the intended experiment.
2. `src/lora_gpt2/lora_layers.py` to understand the LoRA math.
3. `src/lora_gpt2/inject.py` to understand how GPT-2 is modified and frozen.
4. `tests/test_lora_layers.py` and `tests/test_inject.py` to see the most important invariants.
5. `src/lora_gpt2/data.py` and `tests/test_data.py` to understand the E2E conditioning setup.
6. `src/lora_gpt2/checkpointing.py` and `tests/test_checkpointing.py` to understand adapter-only persistence.
7. `scripts/smoke_test.py` for a compact non-training end-to-end sanity check.
8. `scripts/train.py`, `src/lora_gpt2/train.py`, `scripts/generate.py`, and `scripts/evaluate.py` for the remaining experiment workflow.

The most important invariant throughout the project is: GPT-2 base weights stay frozen, and only `lora_` parameters are trainable.