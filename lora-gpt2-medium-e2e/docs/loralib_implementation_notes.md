# `loralib` Implementation Notes

These notes describe how the official [`loralib`](https://github.com/microsoft/LoRA/tree/main/loralib) package works, with emphasis on behavior this project should reproduce in its own implementation.

Primary references:

- [`loralib/layers.py`](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)
- [`loralib/utils.py`](https://github.com/microsoft/LoRA/blob/main/loralib/utils.py)
- [`examples/NLG/src/model.py`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/model.py)

## Core LoRA Contract

For a frozen dense layer with pretrained weight `W`, LoRA learns a low-rank update:

```text
W_adapted = W + scale * B @ A
scale = lora_alpha / r
```

The base weight stays frozen. Only the low-rank matrices are trained. For a standard linear layer, the effective forward pass is:

```text
base_output = x @ W.T + bias
lora_output = dropout(x) @ A.T @ B.T * (lora_alpha / r)
output = base_output + lora_output
```

This project should match that behavior, while writing the code independently.

## `LoRALayer`

The official `LoRALayer` is a small mixin that stores shared LoRA configuration:

- `r`: rank of the low-rank update. If `r=0`, LoRA is disabled.
- `lora_alpha`: scaling numerator.
- `lora_dropout`: optional dropout applied before the low-rank path.
- `merged`: whether the LoRA delta has been folded into the base weight.
- `merge_weights`: whether eval/train mode should merge/unmerge LoRA weights.

Implementation note: the mixin is not itself an `nn.Module`; concrete classes inherit both a PyTorch layer type and `LoRALayer`.

## `Linear`

The official [`Linear`](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py) class subclasses `nn.Linear` and adds:

- `lora_A`: shape `(r, in_features)`.
- `lora_B`: shape `(out_features, r)`.
- `scaling`: `lora_alpha / r`.
- Frozen pretrained `weight` when `r > 0`.
- Optional `fan_in_fan_out` handling for layers that store weights transposed relative to `nn.Linear`.

Initialization:

- `lora_A` is initialized with Kaiming uniform.
- `lora_B` is initialized to zeros.
- Because `lora_B` starts at zero, the initial LoRA branch contributes zero and the model initially behaves like the pretrained base model.

Forward behavior:

- If LoRA is active and not merged, compute the frozen base linear result and add the low-rank update.
- If LoRA is disabled or already merged, run the normal linear operation.

## `MergedLinear`

`MergedLinear` handles fused projections such as a single attention layer that produces query, key, and value together. This is the key class for the official GPT-2 example.

Important behavior:

- `enable_lora` selects which slices of the output projection get LoRA.
- For GPT-2 QKV, the official example uses `enable_lora=[True, False, True]`.
- This adapts query and value while leaving key unchanged.
- `out_features` must be divisible by `len(enable_lora)`.
- The official implementation builds LoRA matrices only for enabled slices, then pads the resulting delta back into the full fused projection shape.
- It uses grouped `conv1d` internally as a compact way to compute separate low-rank updates for each enabled slice.

For GPT-2 Medium, each attention block has a fused projection from hidden size `1024` to `3 * 1024`. With `enable_lora=[True, False, True]`, LoRA learns separate rank-`r` updates for the query and value thirds of the projection.

## GPT-2 Target Pattern

In [`examples/NLG/src/model.py`](https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/model.py), attention uses:

- `c_attn = lora.MergedLinear(...)`
- `r=config.lora_attn_dim`
- `lora_alpha=config.lora_attn_alpha`
- `lora_dropout=config.lora_dropout`
- `enable_lora=[True, False, True]`
- `fan_in_fan_out=True`
- `merge_weights=False`

The `fan_in_fan_out=True` setting matters because the official GPT-2 implementation uses a custom `Conv1D` style projection that stores weights as `(fan_in, fan_out)` rather than the usual PyTorch linear shape.

If this project uses Hugging Face GPT-2, inspect the actual `Conv1D` module shape before injecting LoRA. Hugging Face GPT-2 also uses a `Conv1D`-style `c_attn`, so the same conceptual transposed-weight issue applies.

## Scaling

The official scaling is:

```text
scaling = lora_alpha / r
```

For the official GPT-2 Medium E2E setup:

```text
r = 4
lora_alpha = 32
scaling = 8
```

This scaling multiplies the low-rank update, not the frozen base path.

## Dropout

For linear LoRA layers, dropout is applied to the input of the LoRA path only:

```text
dropout(x) @ A.T @ B.T
```

The base pretrained path does not use this LoRA dropout. Official GPT-2 Medium E2E uses `lora_dropout=0.1`.

## Merge And Unmerge

The official layers can merge LoRA weights into the frozen base weight during evaluation:

- Calling `model.eval()` triggers merging if `merge_weights=True`.
- Calling `model.train()` unmerges if weights were merged.
- Merging removes extra LoRA computation during inference.
- Unmerging restores separate trainable LoRA matrices for training.

The GPT-2 NLG example passes `merge_weights=False` for the attention `MergedLinear`, so it keeps the LoRA path separate even during eval. For this project, it is still useful to implement and test merge/unmerge because it is part of the official `loralib` behavior.

Recommended tests:

- A merged forward pass should numerically match an unmerged forward pass.
- Calling `train()` after `eval()` should not permanently change outputs.
- Repeated eval/train toggles should not keep adding or subtracting the LoRA delta.

## Trainable Parameter Marking

The official [`mark_only_lora_as_trainable()`](https://github.com/microsoft/LoRA/blob/main/loralib/utils.py) iterates over `model.named_parameters()`:

- If the parameter name does not contain `lora_`, set `requires_grad=False`.
- If `bias='none'`, no biases are trained.
- If `bias='all'`, every parameter with `bias` in its name is trainable.
- If `bias='lora_only'`, only biases inside LoRA layers are trainable.

For the main project baseline, use `bias='none'` unless intentionally running a bias-training ablation.

## LoRA-Only State Dict

The official [`lora_state_dict()`](https://github.com/microsoft/LoRA/blob/main/loralib/utils.py) returns a filtered state dict:

- `bias='none'`: only keys containing `lora_`.
- `bias='all'`: keys containing `lora_` or `bias`.
- `bias='lora_only'`: LoRA keys plus biases associated with LoRA layers.

Recommended checkpoint behavior for this project:

1. Load pretrained GPT-2 Medium.
2. Inject LoRA modules.
3. Train only LoRA parameters.
4. Save only LoRA parameters for adapter checkpoints.
5. At inference, recreate the base model, inject the same LoRA structure, load the pretrained weights, then load the LoRA state dict with non-strict loading.

## Other Layer Types

The official repo also implements:

- `Embedding`: LoRA for embeddings.
- `ConvLoRA`, `Conv1d`, `Conv2d`, `Conv3d`: LoRA for convolution weights.

These are not needed for the main GPT-2 Medium on E2E baseline. They can be ignored unless the project adds extension experiments.

## Course Project Implementation Checklist

- Implement a base LoRA mixin or helper that stores rank, alpha, dropout, scaling, and merge state.
- Implement LoRA for the GPT-2 attention projection, including fused QKV support.
- Match the official Q/V-only target behavior.
- Freeze every non-LoRA parameter for the baseline.
- Save only LoRA parameters by default.
- Add unit tests for zero initial delta, trainable parameter counts, state dict filtering, and merge/unmerge equivalence.
- Document any intentional differences from the official implementation.
