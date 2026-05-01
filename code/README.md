# `code/` — implementation

This is the implementation half of the CS 4782 final project re-implementing
**LoRA** on GPT-2 Medium for E2E / DART / WebNLG, plus our **Depth-Weighted
LoRA**, **AdaLoRA-Lite**, and **MoLE** extensions.

The repo's top-level [`README.md`](../README.md) is the canonical project
overview (paper, chosen result, full results tables, full reproduction
steps). This file is a short developer-facing tour of the `code/` folder.

## Layout

```text
code/
├── README.md                  <- (this file)
├── requirements.txt
├── pyproject.toml
├── configs/                   <- one YAML per experiment (paths use ../data/...)
│   ├── e2e_gpt2_medium_lora.yaml
│   ├── e2e_gpt2_medium_adalora_lite.yaml
│   ├── dart_gpt2_medium_lora.yaml
│   ├── webnlg_gpt2_medium_lora.yaml
│   └── mole_e2e_dart_webnlg.yaml
├── src/lora_gpt2/             <- from-scratch library
│   ├── lora_layers.py         <- LoRALinear & LoRAConv1D (GPT-2 c_attn)
│   ├── inject.py              <- attaches LoRA modules in place to a frozen GPT-2
│   ├── adalora_lite.py        <- gated rank + sensitivity pruning + cubic schedule
│   ├── mole_layers.py         <- top-1 router-gated mixture of LoRA experts
│   ├── mole_inject.py / mole_diagnostics.py
│   ├── data.py                <- E2E / DART / WebNLG tokenization
│   ├── generate.py            <- official-style beam search decoder
│   ├── evaluate.py            <- BLEU / NIST / METEOR / ROUGE / CIDEr wrappers
│   ├── checkpointing.py / config.py / modeling.py / utils.py / train.py
├── scripts/                   <- thin CLIs around `src/`
│   ├── prepare_e2e.py / prepare_dart_raw.py / prepare_webnlg_raw.py
│   ├── train.py / train_mole.py
│   ├── generate.py / generate_mole.py
│   ├── evaluate.py / eval_matrix_generate.py / eval_matrix_evaluate.py
│   ├── export_adalora_lite_static.py
│   ├── make_figures.py / count_params.py / smoke_test.py
├── tests/                     <- pytest unit tests
├── outputs/                   <- training run artifacts (gitignored)
└── colab_*.ipynb              <- one Colab notebook per experiment
```

## Local CLI quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Prepare data (assumes you have already populated ../data/raw/e2e/).
# All configs reference ../data/... so paths resolve to the repo-root data folder.
python scripts/prepare_e2e.py --config configs/e2e_gpt2_medium_lora.yaml

# Train, generate, evaluate
python scripts/train.py    --config configs/e2e_gpt2_medium_lora.yaml
python scripts/generate.py --config configs/e2e_gpt2_medium_lora.yaml --split test
python scripts/evaluate.py --config configs/e2e_gpt2_medium_lora.yaml
```

For the AdaLoRA-Lite / Late-Ramp / MoLE variants, swap the config file and
the train script (`train_mole.py` for MoLE). The Colab notebooks in this
folder run the same commands end-to-end on a Colab A100 GPU.

## Paper-aligned defaults

The default E2E config reproduces the GPT-2 Medium / E2E LoRA setup used in
the poster: LoRA `r=4` on Q,V, `alpha=32`, dropout `0.1`, AdamW at `2e-4`, 5
epochs, beam=10, and length penalty `0.9`.

The extension configs and notebooks cover:

- Depth-weighted late-ramp LoRA through the rank-pattern notebooks.
- AdaLoRA-Lite via `configs/e2e_gpt2_medium_adalora_lite.yaml`.
- MoLE via `configs/mole_e2e_dart_webnlg.yaml` and the `*_mole.py` scripts.
