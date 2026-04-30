# LoRA GPT-2 Medium on E2E NLG

Course final project scaffold for reimplementing the core idea from "LoRA: Low-Rank Adaptation of Large Language Models" on a practical data-to-text task.

The planned experiment freezes `gpt2-medium`, injects trainable low-rank adapters into selected attention projections, trains on the E2E NLG restaurant dataset, and evaluates generation quality with the standard E2E metrics.

## Project Goals

- Reimplement LoRA from scratch in PyTorch rather than relying on `peft` or `loralib`.
- Use open-source GPT-2 Medium as the frozen base model.
- Train only low-rank adapter parameters on E2E NLG.
- Reproduce the paper's GPT-2/E2E setup closely enough for a course project.
- Run focused ablations on rank, target layers, alpha scaling, and trainable parameter count.

## Current Status

This folder is a starter scaffold. It contains planning documents and a draft config, but it does not download models, download datasets, or start training jobs.

## Saved Research Notes

- `docs/official_lora_repo_notes.md`: overview of the official Microsoft LoRA repo and what is useful for GPT-2 Medium on E2E.
- `docs/loralib_implementation_notes.md`: detailed notes on `loralib` mechanics to reimplement from scratch.
- `docs/gpt2_e2e_reference_pipeline.md`: official GPT-2/E2E preprocessing, training, generation, decoding, and evaluation reference.

## Suggested Structure

```text
lora-gpt2-medium-e2e/
  README.md
  requirements.txt
  configs/
    e2e_gpt2_medium_lora.yaml
  docs/
    paper_summary.md
    implementation_plan.md
    official_lora_repo_notes.md
    loralib_implementation_notes.md
    gpt2_e2e_reference_pipeline.md
  src/lora_gpt2/
    # future implementation modules
  scripts/
    # future CLI entrypoints
  tests/
    # future unit tests
```

## Setup Placeholder

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Planned Commands

These are placeholders for the implementation described in `docs/implementation_plan.md`.

```bash
# Prepare/tokenize E2E after dataset access is configured.
python scripts/prepare_e2e.py --config configs/e2e_gpt2_medium_lora.yaml

# Train LoRA adapters only.
python scripts/train.py --config configs/e2e_gpt2_medium_lora.yaml

# Generate model outputs with beam search.
python scripts/generate.py --config configs/e2e_gpt2_medium_lora.yaml --split test

# Run E2E metrics.
python scripts/evaluate.py --config configs/e2e_gpt2_medium_lora.yaml
```

## Main References

- Paper: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- Released code: [https://github.com/microsoft/LoRA](https://github.com/microsoft/LoRA)
- Dataset paper: [https://arxiv.org/abs/1706.09254](https://arxiv.org/abs/1706.09254)

## Important Notes

- The paper reports GPT-2 Medium LoRA on E2E with about 0.35M trainable parameters versus 354M for full fine-tuning.
- The recommended first target is LoRA on attention query and value projections with rank `r=4`, scaling `alpha=32`, dropout `0.1`, AdamW, 5 epochs, warmup 500 steps, linear LR schedule, and beam size 10.
- External E2E metric scripts may require separate setup, especially METEOR and CIDEr.

