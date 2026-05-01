# Scripts

Thin command-line entrypoints around `code/src/lora_gpt2`.

## Data preparation

- `prepare_e2e.py` downloads/tokenizes the E2E NLG files into `../data/`.
- `prepare_dart_raw.py` converts DART v1.1.1 JSON into JSONL and tokenizes it.
- `prepare_webnlg_raw.py` converts the WebNLG challenge JSON into JSONL and tokenizes it.

## Training and generation

- `train.py` trains uniform LoRA, rank-pattern, late-ramp, and AdaLoRA-Lite configs.
- `generate.py` runs the paper-aligned beam decoder for single-task adapters.
- `train_mole.py` trains the MoLE router over frozen E2E/DART/WebNLG LoRA experts.
- `generate_mole.py` generates with the routed MoLE model.

## Evaluation and reporting

- `evaluate.py` wraps the official-style metrics used for the poster tables.
- `eval_matrix_generate.py` and `eval_matrix_evaluate.py` build the cross-task table from Fig. 11.
- `export_adalora_lite_static.py` exports the final active-rank allocation.
- `make_figures.py` and `count_params.py` produce supporting plots/parameter reports.
- `smoke_test.py` runs a small end-to-end sanity check.