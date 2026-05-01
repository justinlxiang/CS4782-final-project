# `lora_gpt2` Package

From-scratch implementation package for the CS 4782 LoRA reproduction and
poster extensions.

## Core modules

- `lora_layers.py` implements `LoRALinear` and GPT-2's fused-attention
  `LoRAConv1D` wrapper for Q/V adapters.
- `inject.py` attaches LoRA modules to a frozen `GPT2LMHeadModel`.
- `train.py`, `generate.py`, and `evaluate.py` provide the single-task
  training, decoding, and metric pipeline.
- `data.py` handles E2E, DART, and WebNLG formatting/tokenization.
- `config.py`, `checkpointing.py`, `modeling.py`, and `utils.py` keep shared
  setup, save/load, model construction, and reproducibility helpers.

## Poster extensions

- `adalora_lite.py` adds gated rank directions and sensitivity-based pruning
  to the fixed 192-active-unit budget.
- `mole_layers.py`, `mole_inject.py`, and `mole_diagnostics.py` implement and
  inspect the top-1 Mixture-of-LoRA-Experts router used for the cross-task
  E2E/DART/WebNLG experiment.
