# GPT-2 Medium E2E Reference Pipeline

These notes summarize the official GPT-2/E2E pipeline from [`examples/NLG`](https://github.com/microsoft/LoRA/tree/main/examples/NLG) in the Microsoft LoRA repository. They are intended as a project reference, not as code to copy directly.

Primary reference: [`examples/NLG/README.md`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/README.md)

## Pipeline Overview

The official GPT-2 NLG example follows this sequence:

1. Format raw E2E data into JSONL.
2. Tokenize context and completion with GPT-2 BPE.
3. Fine-tune GPT-2 Medium with LoRA adapters on attention Q/V projections.
4. Generate test outputs with beam search.
5. Decode token ids into text predictions and references.
6. Run the standard E2E NLG evaluation scripts.

Relevant files:

- [`examples/NLG/create_datasets.sh`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/create_datasets.sh)
- [`examples/NLG/src/format_converting_e2e.py`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/format_converting_e2e.py)
- [`examples/NLG/src/gpt2_encode.py`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/gpt2_encode.py)
- [`examples/NLG/src/data_utils.py`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/data_utils.py)
- [`examples/NLG/src/gpt2_ft.py`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/gpt2_ft.py)
- [`examples/NLG/src/gpt2_beam.py`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/gpt2_beam.py)
- [`examples/NLG/src/gpt2_decode.py`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/gpt2_decode.py)
- [`examples/NLG/eval/download_evalscript.sh`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/eval/download_evalscript.sh)

## Data Preprocessing

The official repo includes E2E raw files under [`examples/NLG/data/e2e`](https://github.com/microsoft/LoRA/tree/main/examples/NLG/data/e2e):

- `train.txt`
- `valid.txt`
- `test.txt`

Each raw line is treated as:

```text
context||completion
```

The official preprocessing does two stages:

1. `format_converting_e2e.py` converts raw lines into JSONL records with string fields:

```json
{"context": "...", "completion": "..."}
```

2. `gpt2_encode.py` tokenizes both fields with GPT-2 BPE and writes JSONL records with token-id arrays:

```json
{"context": [token_ids], "completion": [token_ids]}
```

The official script adds GPT-2 BOS/EOS token id `50256` during encoding. It also prefixes the completion string with a leading space before tokenizing, which is common for GPT-2 BPE behavior.

## Dataset And Loss Masking

The official `FT_Dataset` in `data_utils.py` builds examples with:

- `conditions = context`
- `input = context + completion`, padded/truncated to `seq_len`
- `target = shifted input`
- `mask = 0` over context tokens and `1` over completion tokens
- `query = context`, used during generation

For the default conditional language modeling objective, loss is computed only on the completion. This project should preserve that behavior.

## Model Configuration

The official GPT-2 Medium model card corresponds to:

```text
model_card = gpt2.md
n_embd = 1024
n_layer = 24
n_head = 16
```

The official LoRA setup for E2E uses:

```text
lora_dim = 4
lora_alpha = 32
lora_dropout = 0.1
target = attention Q and V projections only
```

In `examples/NLG/src/model.py`, this target behavior is implemented by replacing the fused QKV attention projection with `lora.MergedLinear(..., enable_lora=[True, False, True])`.

## Training Reference

Official E2E training script: [`examples/NLG/src/gpt2_ft.py`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/gpt2_ft.py)

Important official hyperparameters from the README:

```text
train_data = ./data/e2e/train.jsonl
valid_data = ./data/e2e/valid.jsonl
train_batch_size = 8
grad_acc = 1
valid_batch_size = 4
seq_len = 512
model_card = gpt2.md
clip = 0.0
lr = 0.0002
weight_decay = 0.01
correct_bias = true
adam_beta2 = 0.999
scheduler = linear
warmup_step = 500
max_epoch = 5
save_interval = 1000
lora_dim = 4
lora_alpha = 32
lora_dropout = 0.1
label_smooth = 0.1
random_seed = 110
```

The official training loop:

- Loads a pretrained GPT-2 Medium checkpoint.
- Injects LoRA into the GPT-2 attention layers.
- Calls `mark_only_lora_as_trainable()` when `lora_dim > 0`.
- Uses AdamW with linear warmup/decay.
- Saves LoRA-only checkpoints at interval checkpoints using `lora.lora_state_dict(model)`.
- Reports validation loss and perplexity.

For this project, use the same values as the first baseline unless resource limits require smaller batch sizes or fewer epochs. If changing batch size or gradient accumulation, document the effective batch size difference.

## Generation Reference

Official generation script: [`examples/NLG/src/gpt2_beam.py`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/gpt2_beam.py)

Important official generation settings:

```text
data = ./data/e2e/test.jsonl
batch_size = 1
seq_len = 512
eval_len = 64
model_card = gpt2.md
lora_dim = 4
lora_alpha = 32
beam = 10
length_penalty = 0.8
no_repeat_ngram_size = 4
repetition_penalty = 1.0
eos_token_id = 628
```

Generation uses the tokenized context as the prompt and writes JSONL predictions containing each example id and generated token ids.

Project caveat: confirm whether `eos_token_id=628` is appropriate for the exact tokenizer and decoded E2E format used in this project. The standard GPT-2 end-of-text token is `50256`, but the official E2E command uses `628` for stopping generation.

## Decoding Reference

Official decoding script: [`examples/NLG/src/gpt2_decode.py`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/gpt2_decode.py)

The decoder:

- Loads the GPT-2 vocabulary files.
- Reads generated token ids from the beam output JSONL.
- Reads formatted test examples with string references.
- Groups multiple references by context.
- Decodes predictions.
- Strips text after `<|endoftext|>` and double-newline boundaries.
- Writes one prediction file and one reference file for metric scripts.

For E2E, the official output files are:

```text
e2e_ref.txt
e2e_pred.txt
```

## Evaluation Reference

The official E2E evaluation command is:

```text
python eval/e2e/measure_scores.py e2e_ref.txt e2e_pred.txt -p
```

The official repo obtains that evaluator by running [`examples/NLG/eval/download_evalscript.sh`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/eval/download_evalscript.sh), which clones [`tuetschek/e2e-metrics`](https://github.com/tuetschek/e2e-metrics).

Expected metrics from the E2E evaluator typically include BLEU, NIST, METEOR, ROUGE-L, and CIDEr, depending on local dependency setup. METEOR and CIDEr can be more fragile because they may require extra Java or package dependencies.

## Official Result Target

The root README reports:

```text
GPT-2 Medium full fine-tuning: 354.92M trainable params, E2E BLEU 68.2
GPT-2 Medium LoRA: about 0.35M trainable params, E2E BLEU 70.4 +/- 0.1
```

For the course project, the more important reproduction goals are:

- correct LoRA mechanics,
- correct trainable parameter count scale,
- reasonable E2E generation quality,
- documented comparison to the official result.

Exact BLEU may vary by environment, dependency versions, random seeds, metric scripts, and generation details.

## Project Caveats

- Do not depend on `loralib` or `peft` for the main implementation if the goal is reimplementation from scratch.
- Do not copy the official GPT-2 model/training code into the submission. Recreate the behavior in project-specific modules.
- If using Hugging Face `gpt2-medium`, carefully handle GPT-2 `Conv1D` weight orientation for `c_attn`.
- Save adapter-only checkpoints for the main LoRA result.
- Record all deviations from the official command, especially batch size, gradient accumulation, maximum epochs, tokenizer source, stopping token, and metric script version.
- Keep official code links in the docs so reviewers can distinguish reference behavior from project code.
