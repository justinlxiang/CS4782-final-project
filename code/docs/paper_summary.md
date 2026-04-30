# Paper Summary: LoRA

Paper: "LoRA: Low-Rank Adaptation of Large Language Models" by Hu et al., arXiv:2106.09685v2.

## 1. High-Level Intuition

Large pretrained language models are useful because one shared model can be adapted to many downstream tasks. The problem is that ordinary fine-tuning updates every parameter. For a model with hundreds of millions or billions of parameters, each task then needs a full model-sized checkpoint and optimizer states during training.

LoRA's core idea is simple: keep the pretrained model fixed and learn only a small, low-rank "update" for selected weight matrices. Instead of learning a full dense change to a weight matrix, LoRA represents that change as the product of two skinny matrices. This is much cheaper because the number of trainable values scales with `r * (input_dim + output_dim)` instead of `input_dim * output_dim`, where `r` is a small rank such as 4 or 8.

Intuitively, LoRA assumes that adapting a large language model to a specific task does not require moving the model in every possible parameter direction. It only needs to amplify or modify a small number of task-relevant directions already present in the pretrained model. The paper calls this a low "intrinsic rank" of adaptation.

LoRA also has a practical deployment advantage. Since its update is just a linear matrix product, the learned low-rank update can be merged into the frozen base weight after training. That means LoRA can avoid the extra inference latency of adapter modules that add new sequential layers.

## 2. Technical Summary

### 2.1 Problem Setting

The paper studies adaptation of a pretrained language model `P_phi(y | x)` to downstream tasks with input-output pairs `(x, y)`. For generation tasks, the training objective is conditional language modeling: maximize the probability of target tokens given an input prompt and previous target tokens.

Full fine-tuning initializes from pretrained weights `phi_0` and learns an unconstrained update `delta_phi` over all model parameters. LoRA instead freezes `phi_0` and learns a much smaller set of task-specific parameters.

### 2.2 Low-Rank Reparameterization

For a dense pretrained weight matrix `W0` with shape `d_out x d_in`, LoRA represents the adaptation update as:

```text
W = W0 + delta_W
delta_W = B A
A has shape r x d_in
B has shape d_out x r
r << min(d_in, d_out)
```

The forward pass for an input vector or activation `x` becomes:

```text
h = W0 x + (alpha / r) * B A x
```

Only `A` and `B` are trained. The pretrained `W0` remains frozen.

Initialization:

- `A` is initialized with a random Gaussian distribution.
- `B` is initialized to zero.
- This makes the LoRA branch initially output zero, so the model starts exactly as the pretrained model.

Scaling:

- The paper uses a scale factor `alpha / r`.
- `alpha` acts like a LoRA-specific gain. It lets the implementation vary `r` without changing the effective update scale too abruptly.

### 2.3 Why It Saves Parameters

A full update to a `d_out x d_in` matrix has `d_out * d_in` trainable parameters. LoRA has:

```text
r * d_in + d_out * r = r * (d_in + d_out)
```

For a square `1024 x 1024` projection and rank `r=4`, full fine-tuning uses 1,048,576 parameters for that matrix, while LoRA uses only 8,192.

### 2.4 Where LoRA Is Applied

The paper applies LoRA primarily to Transformer attention projection matrices:

- `Wq`: query projection
- `Wk`: key projection
- `Wv`: value projection
- `Wo`: attention output projection

Its strongest practical default is to adapt query and value projections. For GPT-2 experiments, the paper reports `r_q = r_v = 4`, meaning LoRA is inserted into the query and value portions of attention and the key projection remains frozen.

### 2.5 Relationship To Other Adaptation Methods

LoRA is compared against:

- Full fine-tuning: high quality, but expensive storage and optimizer memory.
- Adapter layers: parameter-efficient, but add extra sequential computation and can add inference latency.
- Prefix/prompt tuning: parameter-efficient, but consumes context length and may underperform on some tasks.
- BitFit and partial fine-tuning: train fewer existing parameters, but can be less flexible.

The paper emphasizes that LoRA:

- Trains far fewer parameters.
- Avoids storing full model checkpoints per task.
- Can be merged into base weights for zero additional inference latency.
- Can be combined with methods such as prefix tuning.

### 2.6 Empirical Findings

Across RoBERTa, DeBERTa, GPT-2, and GPT-3, LoRA is competitive with or better than full fine-tuning and other parameter-efficient approaches.

The paper's central empirical claim is not only that LoRA is efficient, but that low ranks are enough. In GPT-3 experiments, ranks as small as 1 or 2 can work well for some tasks. For GPT-2 Medium on E2E, the appendix reports that performance peaks around rank 4 for BLEU and around rank 16 for validation loss, with larger ranks providing little benefit.

### 2.7 Low-Rank Analysis

The paper studies the learned adaptation matrices with singular vector and subspace similarity analyses. The key observation is that the most important singular directions are shared between lower-rank and higher-rank LoRA runs. Extra rank often adds directions that are less useful or noisier.

The paper also compares LoRA updates against the original pretrained weights. Its interpretation is that the learned low-rank update does not merely repeat the largest singular directions of `W0`; instead, it amplifies task-relevant directions that were present but not emphasized by pretraining.

## 3. Project-Relevant Notes For GPT-2 Medium + E2E

### 3.1 Target Model

The course project should use `gpt2-medium` as the frozen base model.

Relevant GPT-2 Medium dimensions:

- Layers: 24 Transformer blocks.
- Hidden size: 1024.
- Attention projection implementation in Hugging Face GPT-2: `c_attn`, a combined query/key/value projection.
- Output projection: `c_proj`.

The paper's GPT-2 Medium full fine-tuning baseline has about 354M trainable parameters.

### 3.2 Dataset

The E2E NLG Challenge dataset is a restaurant-domain data-to-text dataset.

Paper-reported scale:

- About 42,000 training examples.
- About 4,600 validation examples.
- About 4,600 test examples.
- Inputs are slot-value pairs.
- Outputs are natural-language restaurant descriptions.
- One input can have multiple reference outputs.

The project should format each example as a conditional generation prompt:

```text
Input: name[The Eagle], food[French], area[riverside], ...
Output: The Eagle is a French restaurant by the riverside ...
```

For GPT-2 causal language modeling, concatenate prompt and target, but mask the prompt positions in the loss so the model is trained to predict only the output text.

### 3.3 LoRA Placement For First Reproduction

The recommended first implementation follows the paper:

- Freeze all pretrained GPT-2 parameters.
- Insert LoRA into the query and value projections of every self-attention block.
- Leave key, output projection, MLP, layer norm, embeddings, and biases frozen.
- Use rank `r=4` for both query and value.
- Use LoRA alpha `32`.
- Use LoRA dropout `0.1`.

Because Hugging Face GPT-2 stores query, key, and value as one combined `Conv1D` layer named `c_attn`, the implementation has two practical options:

1. Replace `c_attn` with a wrapper that computes the frozen base projection plus LoRA updates for only the query and value slices.
2. Rewrite/split the projection into explicit query/key/value projections inside a custom GPT-2 attention module.

The first option is lower-risk for a course project because it preserves the pretrained model's structure.

### 3.4 Trainable Parameter Counts

For one square projection with `d_in = d_out = 1024` and `r=4`:

```text
LoRA params = r * (d_in + d_out)
            = 4 * (1024 + 1024)
            = 8,192
```

For query and value in one GPT-2 Medium block:

```text
2 * 8,192 = 16,384 trainable params
```

Across 24 blocks:

```text
24 * 16,384 = 393,216 trainable params
```

The paper reports about 0.35M trainable parameters for GPT-2 Medium LoRA on E2E. The exact count can differ slightly depending on implementation details, but the expected range is roughly 0.35M to 0.40M, far below the 354M full fine-tuning baseline.

### 3.5 Training Setup From The Paper

For GPT-2 LoRA on E2E/WebNLG/DART, the appendix reports:

- Optimizer: AdamW.
- Weight decay: 0.01 for E2E.
- Dropout: 0.1 for E2E.
- Batch size: 8.
- Epochs: 5.
- Warmup steps: 500.
- LR schedule: linear.
- Label smoothing: 0.1 for E2E.
- Learning rate: 0.0002.
- Adaptation: `r_q = r_v = 4`.
- LoRA alpha: 32.

### 3.6 Decoding Setup From The Paper

For E2E generation:

- Beam size: 10.
- Length penalty: 0.9.
- No-repeat n-gram size: 4.

The project should save generated outputs in a stable text format that can be consumed by the E2E evaluation scripts.

### 3.7 Evaluation Metrics

The paper reports standard E2E NLG metrics:

- BLEU: n-gram precision with brevity penalty.
- NIST: information-weighted n-gram overlap.
- METEOR: alignment-based metric with recall emphasis.
- ROUGE-L: longest-common-subsequence overlap.
- CIDEr: consensus-based metric using TF-IDF weighted n-grams.

Reported GPT-2 Medium E2E results from the paper:


| Method           | Trainable params | BLEU | NIST | METEOR | ROUGE-L | CIDEr |
| ---------------- | ---------------- | ---- | ---- | ------ | ------- | ----- |
| Full fine-tuning | 354M             | 68.2 | 8.62 | 46.2   | 71.0    | 2.47  |
| LoRA             | 0.35M            | 68.9 | 8.69 | 46.4   | 71.3    | 2.51  |


These numbers should be treated as reproduction targets, not guaranteed outcomes, because library versions, dataset preprocessing, decoding details, and random seeds can change results.

### 3.8 Rank Ablation Targets

The appendix reports a rank sweep on GPT-2 Medium/E2E after 26,000 steps:


| Rank | Val loss | BLEU  | NIST   | METEOR | ROUGE-L | CIDEr  |
| ---- | -------- | ----- | ------ | ------ | ------- | ------ |
| 1    | 1.23     | 68.72 | 8.7215 | 0.4565 | 0.7052  | 2.4329 |
| 2    | 1.21     | 69.17 | 8.7413 | 0.4590 | 0.7052  | 2.4639 |
| 4    | 1.18     | 70.38 | 8.8439 | 0.4689 | 0.7186  | 2.5349 |
| 8    | 1.17     | 69.57 | 8.7457 | 0.4636 | 0.7196  | 2.5196 |
| 16   | 1.16     | 69.61 | 8.7483 | 0.4629 | 0.7177  | 2.4985 |
| 32   | 1.16     | 69.33 | 8.7736 | 0.4642 | 0.7105  | 2.5255 |


For the course project, a smaller ablation over `r in {1, 2, 4, 8, 16}` is enough if compute is limited.

## 4. Takeaways For The Final Project

- The most faithful minimal reproduction is GPT-2 Medium with LoRA on query/value projections for E2E.
- The implementation challenge is mostly engineering: correctly wrapping GPT-2 attention, freezing all base weights, masking prompt tokens in the loss, and matching decoding/evaluation.
- A strong report can compare parameter counts, GPU memory, training time, validation loss, and generation metrics.
- The most meaningful ablations are rank, target projections, alpha scaling, and possibly LoRA dropout.
- The expected story is that a sub-million-parameter adapter can match or slightly exceed full fine-tuning on E2E-style generation metrics.

