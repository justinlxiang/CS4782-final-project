# Official Microsoft LoRA Repo Notes

These notes summarize the official Microsoft LoRA repository as a reference for this course project. The project should reimplement the core behavior in its own code rather than copying source files verbatim.

Official repository: [microsoft/LoRA](https://github.com/microsoft/LoRA)

## What The Repo Contains

- [`README.md`](https://github.com/microsoft/LoRA/blob/main/README.md): project overview, installation quickstart, GPT-2 and GLUE result tables, LoRA checkpoint links, citation, and implementation guidance.
- [`loralib/`](https://github.com/microsoft/LoRA/tree/main/loralib): the small PyTorch package implementing LoRA layers and checkpoint utilities.
- [`examples/NLG/`](https://github.com/microsoft/LoRA/tree/main/examples/NLG): the GPT-2 natural-language generation example. This is the most relevant directory for GPT-2 Medium on E2E.
- [`examples/NLU/`](https://github.com/microsoft/LoRA/tree/main/examples/NLU): RoBERTa and DeBERTa examples for GLUE. Useful for comparison, but not central to this project.
- [`setup.py`](https://github.com/microsoft/LoRA/blob/main/setup.py): package metadata for `loralib`.
- [`LICENSE.md`](https://github.com/microsoft/LoRA/blob/main/LICENSE.md): MIT license.

## Important Files For This Project

### `loralib`

- [`loralib/layers.py`](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py): defines `LoRALayer`, `Linear`, `MergedLinear`, `Embedding`, and convolution variants.
- [`loralib/utils.py`](https://github.com/microsoft/LoRA/blob/main/loralib/utils.py): defines `mark_only_lora_as_trainable()` and `lora_state_dict()`.
- [`loralib/__init__.py`](https://github.com/microsoft/LoRA/blob/main/loralib/__init__.py): re-exports the layer and utility APIs.

### GPT-2/E2E NLG Example

- [`examples/NLG/README.md`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/README.md): official train, generate, decode, and evaluate commands for GPT-2 Medium on E2E.
- [`examples/NLG/src/model.py`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/model.py): custom GPT-2 implementation with LoRA inserted into attention.
- [`examples/NLG/src/gpt2_ft.py`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/gpt2_ft.py): training loop and checkpointing.
- [`examples/NLG/src/gpt2_beam.py`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/gpt2_beam.py): beam-search generation.
- [`examples/NLG/src/gpt2_decode.py`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/gpt2_decode.py): converts generated token ids into reference and prediction text files.
- [`examples/NLG/src/data_utils.py`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/data_utils.py): fine-tuning dataset, padding, query construction, and completion-only loss mask.
- [`examples/NLG/src/format_converting_e2e.py`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/format_converting_e2e.py): converts raw E2E `context||completion` text into JSONL.
- [`examples/NLG/src/gpt2_encode.py`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/gpt2_encode.py): GPT-2 BPE tokenization for formatted E2E examples.
- [`examples/NLG/create_datasets.sh`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/create_datasets.sh): official preprocessing sequence.
- [`examples/NLG/eval/download_evalscript.sh`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/eval/download_evalscript.sh): downloads E2E and other NLG evaluation scripts.

## What Matters For GPT-2 Medium + E2E

The official GPT-2 example adapts only the attention query and value projections. GPT-2 stores query, key, and value as a fused projection, so the example uses `lora.MergedLinear` with `enable_lora=[True, False, True]`. This means:

- Query gets LoRA.
- Key stays frozen without LoRA.
- Value gets LoRA.
- The output projection, MLP layers, embeddings, and layer norms are not LoRA targets in the main reported setup.

For GPT-2 Medium, the official example config uses:

- `n_embd=1024`
- `n_layer=24`
- `n_head=16`
- LoRA rank `r=4`
- LoRA alpha `32`
- LoRA dropout `0.1`
- E2E sequence length `512`
- Completion generation length `64`

The root README reports GPT-2 Medium LoRA on E2E at about `70.4` BLEU with roughly `0.35M` trainable parameters. This is a useful target, but exact reproduction may vary with environment, tokenizer details, metric setup, random seed, and hardware.

## Safe To Use Almost Directly As Reference

- The official hyperparameter values from `examples/NLG/README.md`.
- The high-level command sequence: preprocess, train, beam-generate, decode, evaluate.
- The target-layer choice: fused attention Q/V only.
- The LoRA configuration: `r=4`, `alpha=32`, dropout `0.1`.
- The idea of saving LoRA-only checkpoints and loading them after the pretrained GPT-2 weights.
- The use of the standard E2E metric scripts for evaluation.

## Should Reimplement In This Project

- LoRA layer classes and low-rank update math.
- Trainable parameter filtering.
- LoRA-only state dict saving and loading.
- GPT-2 target-layer replacement or wrapping.
- E2E preprocessing and completion-only loss masking.
- Training, generation, decoding, and evaluation entrypoints.
- Tests for LoRA correctness and checkpoint behavior.

## Likely Unnecessary For The Course Project

- `examples/NLU/` RoBERTa/DeBERTa GLUE code.
- WebNLG preprocessing/evaluation unless used as an extension. (DART is now supported via `prepare_dart_raw.py` + `dart_gpt2_medium_lora.yaml`.)
- `Embedding` and convolution LoRA variants, unless doing ablations beyond the main GPT-2 setup.
- Azure, Kubernetes, Horovod, and distributed infrastructure in the official example.
- R-Drop and cutoff augmentation utilities in the NLU example.
- Downloading official LoRA checkpoints, unless comparing against released weights.

## Citation And Academic Honesty

The official repo is MIT licensed, but this course project should not submit copied `loralib` or `examples/NLG` code as original work. Use the official repository as a behavioral reference, cite it clearly, and write project-specific implementations.

Suggested citation:

```bibtex
@inproceedings{
hu2022lora,
title={Lo{RA}: Low-Rank Adaptation of Large Language Models},
author={Edward J Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=nZeVKeeFYf9}
}
```

Also cite the official implementation: [https://github.com/microsoft/LoRA](https://github.com/microsoft/LoRA).
