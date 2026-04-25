# Implementation Plan

## Current Implementation Status

This document is now partly the original implementation roadmap and partly a snapshot of current progress. The project path has changed from `finalproject/...` to `CS4782-final-project/...`, and the scaffold, code, and tests described below now exist.

For the current code walkthrough, see `docs/implementation_summary_for_partner.md`.

Implemented now:

- LoRA GPT-2 Medium scaffold, source code, and tests are present in the repository.
- Generation config now uses `max_new_tokens=64`, `length_penalty=0.8`, and `eos_token_id=628`.
- Preprocessing now uses the raw E2E context directly (`prompt_template="{mr}"`) and appends GPT-2's `50256` end-of-text token after the context, matching the official LoRA NLG pipeline more closely than the earlier instruction-style prompt.
- `scripts/train.py` is guarded and currently only supports `--dry-run`.

### Known Deviations / Not Yet Run

- Training has not been run yet.
- A real training run is the next implementation step, but the training guard in `scripts/train.py` should only be deliberately changed after approval.
- The sections below still describe the intended full training and evaluation design, not completed experiment results.

## 1. Project Scope

Build a from-scratch LoRA implementation for GPT-2 Medium and train it on the E2E NLG data-to-text benchmark. The base model should come from an open-source GPT-2 Medium checkpoint, but the LoRA module, parameter freezing, training loop, decoding path, and experiment logging should be implemented in this repository.

Non-goals for the first version:

- Do not pretrain GPT-2.
- Do not download large model or dataset artifacts during scaffolding.
- Do not rely on `peft` or `loralib` for the core LoRA logic.
- Do not aim to exactly reproduce every paper baseline.

## 2. Reproduction Target

Primary target:

- Base model: `gpt2-medium`.
- Dataset: E2E NLG Challenge.
- Adaptation: LoRA on attention query and value projections.
- Rank: `r=4`.
- Alpha: `32`.
- Dropout: `0.1`.
- Optimizer: AdamW.
- Learning rate: `2e-4`.
- Weight decay: `0.01`.
- Epochs: `5`.
- Warmup steps: `500`.
- Schedule: linear decay.
- Batch size: `8`, using gradient accumulation if necessary.
- Decode: beam size `10`, length penalty `0.9`, no-repeat n-gram size `4`.

Paper target for GPT-2 Medium on E2E:

- Full fine-tuning: 354M trainable params, BLEU 68.2, NIST 8.62, METEOR 46.2, ROUGE-L 71.0, CIDEr 2.47.
- LoRA: 0.35M trainable params, BLEU 68.9, NIST 8.69, METEOR 46.4, ROUGE-L 71.3, CIDEr 2.51.

For a course project, success can be defined as:

- Correctly train only LoRA parameters.
- Reach plausible validation loss decline.
- Produce fluent E2E generations.
- Get within a reasonable band of reported metrics after matching preprocessing and decoding.
- Demonstrate parameter/memory savings and at least two ablations.

## 3. Architecture

### 3.1 Base Model

Use Hugging Face `transformers` to load GPT-2 Medium:

```text
AutoModelForCausalLM.from_pretrained("gpt2-medium")
AutoTokenizer.from_pretrained("gpt2-medium")
```

Set tokenizer padding carefully:

- GPT-2 has no default pad token.
- Use `tokenizer.pad_token = tokenizer.eos_token`.
- Mask padding tokens in the loss.

Freeze all pretrained parameters:

```text
for param in model.parameters():
    param.requires_grad = False
```

Then inject LoRA modules and mark only adapter parameters trainable.

### 3.2 LoRA Module

Implement a small module that computes:

```text
base_out = frozen_linear(x)
lora_out = dropout(x) @ A.T @ B.T * (alpha / r)
output = base_out + lora_out
```

For a generic linear layer:

- `A`: shape `r x in_features`.
- `B`: shape `out_features x r`.
- Initialize `A` with Kaiming or Gaussian initialization.
- Initialize `B` with zeros.
- Store `scaling = alpha / r`.

The wrapper should support:

- Train mode with dropout.
- Eval mode without dropout.
- Optional `merge()` for inference by adding `B @ A * scaling` into the base weight.
- Optional `unmerge()` for continued training.

### 3.3 GPT-2 Attention Injection

GPT-2 uses a combined query/key/value projection, usually `attn.c_attn`, with output size `3 * hidden_size`.

Preferred approach:

- Wrap `attn.c_attn` in a module that preserves the original frozen projection.
- Add LoRA only to the query slice and value slice.
- Leave the key slice unchanged.

Pseudo-output:

```text
base_qkv = frozen_c_attn(x)
delta_q = lora_q(x)
delta_v = lora_v(x)
base_q, base_k, base_v = split(base_qkv)
q = base_q + delta_q
k = base_k
v = base_v + delta_v
return concat(q, k, v)
```

This matches the paper's `r_q = r_v = 4` setting while avoiding a full rewrite of GPT-2 attention.

Important implementation detail:

- Hugging Face GPT-2 uses `Conv1D`, whose stored weight layout differs from `nn.Linear`.
- Test shapes carefully. A unit test should confirm that a zero-initialized LoRA wrapper exactly matches the base layer output before training.

### 3.4 Parameter Counting

Implement utility functions:

- `count_total_parameters(model)`
- `count_trainable_parameters(model)`
- `list_trainable_parameters(model)`

Expected first-run trainable count:

- Approximately 0.35M to 0.40M parameters for GPT-2 Medium query/value LoRA with `r=4`.
- If the count is much larger, some base parameters were not frozen.
- If the count is much smaller, not all layers received LoRA.

## 4. Data Pipeline

### 4.1 Dataset Access

Use either:

- Hugging Face `datasets` if the E2E dataset is available and stable in the current environment.
- The official E2E NLG Challenge files if required by evaluation scripts.

Keep raw data out of git. Store it under `data/raw/` or a configurable path ignored by git.

### 4.2 Input Formatting

Convert meaning representations into deterministic textual prompts. Example:

```text
Meaning representation: name : The Eagle | food : French | priceRange : moderate | area : riverside
Prompt/context: "name : The Eagle | food : French | priceRange : moderate | area : riverside" + GPT-2 end-of-text token `50256`
Target: "The Eagle is a moderately priced French restaurant by the riverside."
```

Use one consistent format across train/validation/test.

### 4.3 Tokenization

For each example:

1. Tokenize prompt.
2. Tokenize target plus EOS.
3. Concatenate prompt and target tokens.
4. Truncate or skip examples exceeding `max_length`.
5. Labels should be `-100` for prompt tokens and padding tokens.
6. Labels should equal token IDs for target tokens.

This trains conditional generation while using GPT-2's causal language modeling head.

### 4.4 Batching

Use dynamic padding in a custom collator:

- Pad `input_ids`.
- Pad `attention_mask`.
- Pad `labels` with `-100`.
- Optionally record prompt lengths for generation.

## 5. Training Loop

### 5.1 Core Loop

Implement:

```text
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / grad_accum_steps
        loss.backward()
        optimizer.step() every grad_accum_steps
        scheduler.step()
        zero_grad()
```

Recommended features:

- Mixed precision with `torch.cuda.amp` when CUDA is available.
- Gradient accumulation to emulate batch size 8 if memory is tight.
- Gradient clipping, e.g. max norm 1.0.
- Periodic validation loss.
- Save adapter-only checkpoints.
- Save config and tokenizer metadata with each run.

### 5.2 Label Smoothing

The paper uses label smoothing `0.1` for E2E. Hugging Face causal LM loss does not apply label smoothing automatically. Options:

1. Implement cross-entropy with label smoothing over shifted logits/labels.
2. Start without label smoothing for a minimal baseline, then add it as an ablation.

The report should clearly state whether label smoothing was used.

### 5.3 Checkpointing

Save only:

- LoRA adapter weights.
- Config.
- Training state if resuming is needed.

Do not save the full GPT-2 Medium checkpoint per run unless explicitly required.

Suggested layout:

```text
outputs/
  runs/
    e2e_lora_r4_alpha32_seed0/
      adapter.pt
      config.yaml
      metrics.jsonl
      generations_valid.txt
```

## 6. Decoding And Evaluation

### 6.1 Generation

For validation/test:

- Feed only the prompt.
- Generate with `model.generate`.
- Use beam size 10.
- Use length penalty 0.9.
- Use no-repeat n-gram size 4.
- Set max new tokens based on E2E target length distribution, e.g. 80-120.
- Decode only generated continuation tokens, not the prompt.

### 6.2 Metrics

Use standard E2E NLG metrics where possible:

- BLEU.
- NIST.
- METEOR.
- ROUGE-L.
- CIDEr.

Potential tooling:

- Official E2E NLG evaluation scripts.
- `sacrebleu` for a simpler BLEU sanity check.
- `evaluate` or local scripts for ROUGE-L.

Because official E2E scores can depend on tokenization and script versions, record:

- Evaluation script source.
- Preprocessing details.
- Exact generated output files.
- Command used for metrics.

## 7. Proposed Files To Implement

### Core Package

```text
src/lora_gpt2/lora_layers.py
```

Contains:

- `LoRALinear`.
- GPT-2 `Conv1D`-compatible LoRA wrapper.
- Merge/unmerge helpers.

```text
src/lora_gpt2/inject.py
```

Contains:

- `inject_lora_into_gpt2(model, config)`.
- Target module selection.
- Trainable parameter marking.

```text
src/lora_gpt2/data.py
```

Contains:

- E2E dataset loading.
- Prompt formatting.
- Tokenization.
- Data collator.

```text
src/lora_gpt2/train.py
```

Contains:

- Training loop.
- Validation loop.
- Optimizer/scheduler setup.
- Checkpoint saving.

```text
src/lora_gpt2/generate.py
```

Contains:

- Prompt-only generation.
- Decoding cleanup.
- Output formatting for E2E metrics.

```text
src/lora_gpt2/evaluate.py
```

Contains:

- Metric wrappers.
- Official script integration.
- JSON summary output.

```text
src/lora_gpt2/utils.py
```

Contains:

- Seeding.
- Parameter counting.
- Config loading.
- Device utilities.

### CLI Scripts

```text
scripts/prepare_e2e.py
scripts/train.py
scripts/generate.py
scripts/evaluate.py
scripts/count_params.py
```

### Tests

```text
tests/test_lora_layers.py
tests/test_inject.py
tests/test_data_collator.py
```

Critical tests:

- With `B=0`, LoRA wrapper output equals frozen base output.
- Only LoRA parameters require gradients after injection.
- Parameter count is near expected.
- Prompt tokens are masked with `-100`.
- Generation strips prompt text correctly.

## 8. Ablation Plan

Minimum ablations:

1. Rank sweep: `r in {1, 2, 4, 8, 16}`.
2. Target modules:
  - Query only.
  - Value only.
  - Query + value.
  - Query + key + value + output if time allows.
3. Alpha scaling:
  - `alpha in {8, 16, 32, 64}` at fixed `r=4`.
4. LoRA dropout:
  - `0.0` vs `0.1`.

Optional ablations:

- Label smoothing on/off.
- Smaller base model `gpt2` as a compute fallback.
- Adapter-only checkpoint size comparisons.
- Merge vs unmerged inference latency sanity check.

## 9. Compute Expectations

GPT-2 Medium has about 354M parameters, so the frozen model still needs memory for activations even though optimizer states are small.

Expected hardware:

- Best: one GPU with 16GB+ VRAM.
- Possible: 12GB GPU with smaller batch size, gradient accumulation, mixed precision, and shorter max length.
- CPU-only training is not practical for full 5-epoch reproduction.

Memory-saving settings:

- Use `fp16` or `bf16` if supported.
- Batch size 1-2 with gradient accumulation to effective batch size 8.
- Use max sequence length based on E2E data, not an unnecessarily large context window.
- Freeze base parameters before creating the optimizer.
- Optimizer should receive only trainable LoRA parameters.

Time expectation:

- Smoke test on a tiny subset: minutes.
- One full E2E training run: likely a few hours on a single modern GPU, depending on sequence length and batch size.
- Full ablation suite: plan for multiple GPU-hours; prioritize rank sweep if time is limited.

## 10. Risks And Mitigations

### Risk: Incorrect GPT-2 `Conv1D` Weight Orientation

Mitigation:

- Write shape tests.
- Compare zero-LoRA wrapper output to original layer output.
- Inspect GPT-2 `Conv1D` forward logic before coding.

### Risk: Accidentally Training Base GPT-2

Mitigation:

- Freeze model before injection.
- Print and log all trainable parameter names.
- Assert trainable count is below 1M for `r=4`.

### Risk: Loss Trains On Prompt Tokens

Mitigation:

- Unit test label construction.
- Manually inspect one tokenized example.
- Track prompt length separately.

### Risk: Evaluation Script Mismatch

Mitigation:

- Save generated outputs and references.
- Run a simple BLEU sanity check first.
- Clearly document official metric scripts and versions.

### Risk: Compute Too Limited For GPT-2 Medium

Mitigation:

- Implement a tiny-subset smoke test.
- Add a `model_name: gpt2` fallback config for debugging only.
- Use gradient accumulation and mixed precision.
- If necessary, report full-quality experiments on fewer seeds and more ablations on smaller subsets.

### Risk: Reproduction Numbers Differ

Mitigation:

- Report implementation details precisely.
- Compare trends, parameter efficiency, and qualitative generations.
- Run at least 3 seeds only if compute allows.

## 11. Milestone Schedule

### Milestone 1: Paper And Baseline Setup

- Finalize paper summary and implementation plan.
- Confirm E2E dataset access.
- Load GPT-2 Medium and tokenizer.
- Build prompt formatting and tokenization.
- Create tiny-subset training dataloader.

Deliverable: data pipeline notebook or script with inspected examples.

### Milestone 2: LoRA Core Implementation

- Implement LoRA wrapper.
- Inject into GPT-2 attention query/value paths.
- Freeze base model and verify trainable parameters.
- Add unit tests for zero-update equivalence and trainable counts.

Deliverable: parameter count report and passing unit tests.

### Milestone 3: Training Smoke Test

- Train on 100-1000 examples.
- Verify loss decreases.
- Generate a few validation examples.
- Save adapter-only checkpoint.

Deliverable: short qualitative sample table and smoke-test logs.

### Milestone 4: Full E2E Run

- Train `r=4`, `alpha=32`, dropout `0.1` on full E2E.
- Decode validation/test with paper settings.
- Run metrics.
- Compare against paper targets.

Deliverable: main reproduction table.

### Milestone 5: Ablations

- Rank sweep over at least `{1, 4, 8, 16}`.
- Target module ablation if time allows.
- Alpha or dropout ablation if time allows.

Deliverable: ablation plots/tables and interpretation.

### Milestone 6: Final Report

- Explain LoRA intuition and math.
- Describe implementation details.
- Present reproduction results.
- Discuss failure modes and deviations from paper.
- Include generated examples.
- Reflect on parameter efficiency and compute.

Deliverable: final course report and runnable code.

## 12. Recommended First Coding Order

1. `utils.py`: config loading, seeding, parameter counting.
2. `data.py`: prompt formatting, tokenization, collator.
3. `lora_layers.py`: generic LoRA and GPT-2 `c_attn` wrapper.
4. `inject.py`: replace attention modules and freeze parameters.
5. Tests for data and LoRA correctness.
6. `train.py`: one-batch overfit and tiny-subset training.
7. `generate.py`: prompt-only generation.
8. `evaluate.py`: metrics integration.

This order reduces risk by validating the LoRA math and data labels before spending GPU time.