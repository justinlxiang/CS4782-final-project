# Data Directory

This folder holds the **raw** and **tokenized** versions of the three data-to-text
benchmarks our experiments train and evaluate on.

```text
data/
  raw/              # original release files for each benchmark (downloaded)
    e2e/            # train.txt / valid.txt / test.txt   (Microsoft LoRA mirror)
    dart/           # dart-v1.1.1-full-{train,dev,test}.json + *.jsonl after prep
    webnlg/         # train.json / dev.json / test.json + *.jsonl after prep
  processed/        # tokenized JSONL ready for GPT-2 Medium (created by scripts)
    e2e_gpt2/
    dart_gpt2/
    webnlg_gpt2/
```

The repository **does not** check the dataset files in (they are large and have
their own licenses). Instead, every Colab notebook under `code/` regenerates
this directory in two steps:

1. **Download** — `curl` the original release files into `data/raw/<task>/`.
2. **Tokenize** — `python code/scripts/prepare_e2e.py | prepare_dart_raw.py |
   prepare_webnlg_raw.py --config code/configs/<task>_gpt2_medium_lora.yaml`.

Each task's notebook (e.g. `code/colab_train_lora.ipynb`,
`code/colab_train_lora_dart.ipynb`, `code/colab_train_lora_webnlg.ipynb`,
`code/colab_train_mole.ipynb`, `code/colab_eval_matrix.ipynb`) contains the
exact commands and runs them automatically when you press *Run all* on Colab.

## Manual download (if you skip the notebooks)

From the repo root:

```bash
# E2E NLG (restaurant attributes -> description)
mkdir -p data/raw/e2e
curl -L -o data/raw/e2e/train.txt https://raw.githubusercontent.com/microsoft/LoRA/main/examples/NLG/data/e2e/train.txt
curl -L -o data/raw/e2e/valid.txt https://raw.githubusercontent.com/microsoft/LoRA/main/examples/NLG/data/e2e/valid.txt
curl -L -o data/raw/e2e/test.txt  https://raw.githubusercontent.com/microsoft/LoRA/main/examples/NLG/data/e2e/test.txt

# DART v1.1.1 (open-domain triples -> sentence)
mkdir -p data/raw/dart
curl -L -o data/raw/dart/dart-v1.1.1-full-train.json https://raw.githubusercontent.com/Yale-LILY/dart/master/data/v1.1.1/dart-v1.1.1-full-train.json
curl -L -o data/raw/dart/dart-v1.1.1-full-dev.json   https://raw.githubusercontent.com/Yale-LILY/dart/master/data/v1.1.1/dart-v1.1.1-full-dev.json
curl -L -o data/raw/dart/dart-v1.1.1-full-test.json  https://raw.githubusercontent.com/Yale-LILY/dart/master/data/v1.1.1/dart-v1.1.1-full-test.json

# WebNLG 2017 challenge split (RDF triples -> paragraph)
mkdir -p data/raw/webnlg
curl -L -o data/raw/webnlg/train.json https://raw.githubusercontent.com/microsoft/LoRA/main/examples/NLG/data/webnlg_challenge_2017/train.json
curl -L -o data/raw/webnlg/dev.json   https://raw.githubusercontent.com/microsoft/LoRA/main/examples/NLG/data/webnlg_challenge_2017/dev.json
curl -L -o data/raw/webnlg/test.json  https://raw.githubusercontent.com/microsoft/LoRA/main/examples/NLG/data/webnlg_challenge_2017/test.json
```

Then tokenize for GPT-2 Medium:

```bash
cd code
python scripts/prepare_e2e.py        --config configs/e2e_gpt2_medium_lora.yaml
python scripts/prepare_dart_raw.py   --config configs/dart_gpt2_medium_lora.yaml
python scripts/prepare_webnlg_raw.py --config configs/webnlg_gpt2_medium_lora.yaml
```

All configs point at `../data/raw/<task>` and
`../data/processed/<task>_gpt2` relative to `code/`, so commands run from
inside `code/` resolve dataset paths to this folder. No symlink is
required (clean cross-platform behaviour, including Windows).

## Dataset stats

| Task   | Train | Valid | Test  | Source |
|--------|------:|------:|------:|--------|
| E2E    | 42,061 | 4,672 | 4,693 | [Microsoft/LoRA mirror][e2e] of the SIGDIAL 2017 release |
| DART   | 62,659 | 6,980 | 12,551 | [Yale-LILY/dart v1.1.1][dart] |
| WebNLG | 18,025 | 2,258 | 4,928 | [Microsoft/LoRA mirror][webnlg] of the WebNLG 2017 challenge |

[e2e]: https://github.com/microsoft/LoRA/tree/main/examples/NLG/data/e2e
[dart]: https://github.com/Yale-LILY/dart
[webnlg]: https://github.com/microsoft/LoRA/tree/main/examples/NLG/data/webnlg_challenge_2017

## Licenses

- **E2E NLG**: CC BY-SA 4.0 (Novikova, Dušek & Rieser, 2017).
- **DART**: MIT (Yale-LILY).
- **WebNLG**: CC BY-NC-SA 4.0 (WebNLG Challenge 2017).

We redistribute *no* dataset content from this repo; only download URLs and
tokenization scripts.
