# Extension Ideas: Budgeted LoRA Ranks and Gated LoRA Experts

This note sketches two course-project extensions for the GPT-2 Medium + E2E NLG LoRA replication. The main recommendation is to implement the fixed-budget non-uniform rank allocation first, then treat the gated two-LoRA variant as optional if the baseline and evaluation scripts are already stable.

## Baseline Context

The scaffold targets GPT-2 Medium on E2E NLG with the base model frozen and LoRA injected into the query and value slices of each GPT-2 attention `c_attn` projection.

For a frozen linear projection `W in R^{d_out x d_in}`, standard LoRA uses

```text
y = W x + Delta W x
Delta W x = (alpha / r) B A x
A in R^{r x d_in}, B in R^{d_out x r}
```

The LoRA trainable parameter count for one adapted projection is

```text
params = r * (d_in + d_out)
```

For GPT-2 Medium, `d_in = d_out = 1024`, `n_layer = 24`, and the planned baseline adapts query and value in every layer with `r = 4`. This gives

```text
24 layers * 2 projections/layer * 4 * (1024 + 1024) = 393,216 LoRA parameters
```

This is about `0.39M` trainable adapter parameters, close to the LoRA paper's reported `0.35M` scale depending on exact accounting.

## Related Work and Terminology

### Idea 1: Budgeted Non-Uniform LoRA Ranks

Useful terms:

- Adaptive-rank LoRA
- Rank allocation
- Layer-wise rank pattern
- Sensitivity-based rank allocation
- Budgeted PEFT
- Dynamic rank adaptation
- Rank pruning / rank scheduling

Closest papers and systems:

- LoRA: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021. Introduces the fixed-rank low-rank update used as the baseline. URL: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- AdaLoRA: Zhang et al., "Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning", ICLR 2023. This is the closest match. It argues that uniform LoRA rank allocation overlooks that different weight matrices have different importance, uses an SVD-style parameterization, and prunes singular values according to importance scores. URL: [https://arxiv.org/abs/2303.10512](https://arxiv.org/abs/2303.10512)
- DyLoRA: Valipour et al., "DyLoRA: Parameter-Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation", EACL 2023. Trains LoRA blocks so that multiple ranks can be used without retraining, reducing rank search cost. It evaluates on language generation tasks including E2E, DART, and WebNLG. URL: [https://aclanthology.org/2023.eacl-main.239/](https://aclanthology.org/2023.eacl-main.239/)
- PEFT `rank_pattern` / `alpha_pattern`: Hugging Face PEFT exposes per-module rank and alpha patterns, which is useful terminology even if this project implements LoRA from scratch. AdaLoRA support also exposes allocated rank patterns.

For this course project, the proposed variant is simpler than AdaLoRA: choose a fixed non-uniform rank schedule before the main training run, keep the total trainable parameter budget equal to the uniform `r=4` baseline, and compare metrics.

### Idea 2: Gated or Mixture-of-LoRA Updates

Useful terms:

- Gated LoRA
- Mixture of LoRA experts
- LoRA experts
- Conditional LoRA
- Input-dependent adapter routing
- Adapter fusion
- LoRA composition
- MoE adapters / mixture-of-adapters

Closest papers and systems:

- AdapterFusion: Pfeiffer et al., "AdapterFusion: Non-Destructive Task Composition for Transfer Learning", 2020/2021. Not LoRA-specific, but it is an important adapter-composition predecessor that learns to combine multiple adapters. URL: [https://arxiv.org/abs/2005.00247](https://arxiv.org/abs/2005.00247)
- LoraHub: Huang et al., "LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition", COLM 2024. Composes multiple trained LoRA modules with learned coefficients using a few examples, but the composition is closer to task-level/few-shot module merging than per-input gating inside one fine-tune. URL: [https://arxiv.org/abs/2307.13269](https://arxiv.org/abs/2307.13269)
- Mixture of LoRA Experts (MoLE): Wu et al., 2024. Treats trained LoRAs as experts and learns layer-wise gating/composition weights. URL: [https://arxiv.org/abs/2404.13628](https://arxiv.org/abs/2404.13628)
- MixLoRA: "Enhancing Large Language Models Fine-Tuning with LoRA-based Mixture of Experts", 2024. Inserts multiple LoRA experts and uses a router/top-k style MoE mechanism, mainly for multi-task settings. URL: [https://arxiv.org/abs/2404.15159](https://arxiv.org/abs/2404.15159)
- Gated LoRA: "Gated LoRA: Dual-Purpose Projections for Low-Rank Adaptation", 2026-era work. It uses input-dependent gating over rank-1 LoRA directions and reuses the LoRA down-projection to avoid adding router parameters. URL: [https://openreview.net/pdf?id=ZiBDVotA7g](https://openreview.net/pdf?id=ZiBDVotA7g)

The proposed course variant is much smaller than full MoE LoRA: train two LoRA branches per adapted projection and blend them with a sigmoid gate computed from the current hidden state.

## Idea 1: Fixed-Budget Non-Uniform Rank Allocation

### Intuition

Uniform LoRA gives every adapted query/value projection the same rank, but not every layer or projection may need the same adaptation capacity. Earlier transformer layers often capture more general token-level features, while later layers are closer to task-specific generation behavior. For E2E data-to-text, value projections may also matter more than query projections because they directly affect the content mixed into attention outputs.

The extension asks whether the same total adapter budget works better when concentrated in more useful locations.

### Math

Let `l` index layers and `m` index adapted modules, here `m in {q, v}`. Standard LoRA uses one shared rank `r = 4`:

```text
y_{l,m} = W_{l,m} x + s B_{l,m} A_{l,m} x
s = alpha / r
```

The non-uniform version uses rank `r_{l,m}`:

```text
y_{l,m} = W_{l,m} x + s_{l,m} B_{l,m} A_{l,m} x
A_{l,m} in R^{r_{l,m} x d_in}
B_{l,m} in R^{d_out x r_{l,m}}
```

The trainable budget is

```text
sum_{l,m} r_{l,m} * (d_in + d_out)
```

Because all query/value slices in GPT-2 Medium have the same shape, matching the baseline is equivalent to matching total rank units:

```text
sum_{l=0}^{23} (r_{l,q} + r_{l,v}) = 24 * (4 + 4) = 192
```

Use `r_{l,m} = 0` to skip a LoRA module. For stable scaling, prefer keeping the effective scale constant:

```text
s_{l,m} = alpha_base / r_base = 32 / 4 = 8
```

Equivalently, set `alpha_{l,m} = 8 * r_{l,m}` for nonzero ranks. This avoids making lower-rank modules automatically receive larger update scaling.

### Same-Budget Rank Schedules

Recommended first schedule:

```text
Depth-heavy Q/V schedule:
layers 0-7:   r_q = 2, r_v = 2
layers 8-15:  r_q = 4, r_v = 4
layers 16-23: r_q = 6, r_v = 6
```

Per projection, the rank sum is `8*2 + 8*4 + 8*6 = 96`, equal to `24*4`.

Other simple schedules:

```text
Late-concentrated schedule:
layers 0-15:  r_q = 2, r_v = 2
layers 16-23: r_q = 8, r_v = 8
```

```text
Value-heavy schedule:
all layers: r_q = 2, r_v = 6
```

```text
Late + value-heavy schedule:
layers 0-7:   r_q = 1, r_v = 3
layers 8-15:  r_q = 2, r_v = 6
layers 16-23: r_q = 3, r_v = 9
```

The last schedule also averages to `r_q + r_v = 8` per layer, so it matches the baseline budget.

### How to Allocate Ranks Without Expensive Search

Start with one hand-designed schedule, then optionally add one data-informed schedule. Avoid a combinatorial search.

Practical allocation methods:

- Depth heuristic: assign low ranks to early layers and high ranks to later layers. This is the easiest and most defensible first experiment.
- Module-type heuristic: assign more rank to value than query, e.g. `q=2, v=6`, because value updates directly alter the information passed through attention.
- Warmup gradient proxy: train the uniform `r=4` baseline for a small number of steps, log per-adapter gradient norms, then allocate ranks proportional to an EMA of each module's score.
- Sensitivity proxy: after warmup, score adapter `(l,m)` by a simple product such as `||A||_F ||grad_A||_F + ||B||_F ||grad_B||_F`. This is inspired by AdaLoRA's importance-based budget allocation but much easier to implement.
- Magnitude proxy: after a short uniform run, score each adapter by `||B A||_F` or by the singular-value concentration of `B A`. Modules with larger learned updates receive higher final rank.
- Dropout/ablation proxy: temporarily disable each layer's LoRA update on a validation minibatch and measure validation loss increase. This is more expensive but still much cheaper than full rank search.

If using a data-informed schedule, freeze the chosen ranks before the real comparison run. Report that the schedule was selected by a short warmup probe.

### Feasibility

This is the better main extension for the course project.

Implementation complexity is low:

- Extend the LoRA wrapper to accept a rank per module.
- Store a `rank_pattern` keyed by layer and target slice.
- Skip modules with rank `0`.
- Add a trainable parameter counter to verify budget matching.

Training stability should be close to baseline because the forward path is still ordinary LoRA. Compute and memory are essentially the same when the total rank budget is matched.

Main risk: it may not improve BLEU. That is still presentable if the comparison is parameter-matched and shows that the uniform rank allocation is hard to beat on E2E.

## Idea 2: Two LoRA Branches With an Input-Dependent Gate

### Intuition

Standard LoRA applies the same adapter update to every token and every input. A gated two-branch variant gives the model two low-rank update directions and lets the hidden state decide how much of each to use. For E2E, the hope is that different input meaning representations or generated tokens may benefit from different adapter behavior, e.g. content planning versus surface realization.

### Math

For one projection, use two LoRA branches:

```text
Delta_1(x) = s B_1 A_1 x
Delta_2(x) = s B_2 A_2 x
g(x) = sigmoid(router(x))
y = W x + g(x) Delta_1(x) + (1 - g(x)) Delta_2(x)
```

For a token-wise scalar gate at layer `l` and position `t`:

```text
g_{l,t} = sigmoid(w_l^T x_{l,t} + b_l)
y_{l,t} = W_l x_{l,t}
          + g_{l,t} s B_{l,1} A_{l,1} x_{l,t}
          + (1 - g_{l,t}) s B_{l,2} A_{l,2} x_{l,t}
```

To match the baseline LoRA parameter budget, use two rank-2 branches instead of one rank-4 branch:

```text
branch ranks: r_1 = 2, r_2 = 2
LoRA params = (r_1 + r_2) * (d_in + d_out)
            = 4 * (d_in + d_out)
```

The gate adds parameters, so either report them separately or slightly reduce ranks where possible. With a scalar token-wise gate per layer shared by query and value:

```text
gate params = 24 layers * (1024 weights + 1 bias) = 24,600
```

That is about 6 percent overhead relative to the `393,216` LoRA baseline. If the gate is per query/value module, it is about `49,200` parameters.

For scaling, keep the same effective LoRA multiplier as baseline:

```text
s = alpha_base / r_base = 8
```

For rank-2 branches, this means using `alpha_branch = 16` if the implementation stores scale as `alpha_branch / r_branch`.

### Gate Architecture Options

Safest option:

```text
Token-wise scalar gate, shared by q/v within a layer:
g_{l,t} = sigmoid(w_l^T x_{l,t} + b_l)
```

Why it is safest:

- It is causal-safe because each token's gate only uses that token's current hidden state.
- It has small parameter overhead.
- It avoids sequence-level pooling that could accidentally leak future target tokens during causal LM training.
- It is easier to analyze: log average gate values by layer and token position.

Other options:

- Sequence-wise scalar gate: compute one gate per example and layer from the prompt representation. This is interpretable, but using all sequence positions during training can leak future tokens unless the gate is computed only from the prompt or from causal prefix pooling.
- Per-module scalar gate: separate gates for query and value. More flexible, but doubles gate parameters and logging.
- Vector gate per hidden dimension: `g_{l,t} in R^{d_out}`. This is much more expressive, but a full `d -> d` router is too expensive. A diagonal or low-rank gate is possible but adds implementation complexity.
- Softmax over two experts: replace sigmoid with `softmax([a_1(x), a_2(x)])`. This generalizes to more than two branches but needs more router code.

Recommended if attempted:

```text
Two rank-2 LoRA branches per q/v projection.
One token-wise scalar gate per transformer layer, shared by q and v.
Initialize gate bias to 0 so g approx 0.5.
Use the same dropout as baseline on branch inputs.
Add a tiny regularizer only if the gate collapses immediately.
```

### Feasibility

This is a good stretch idea, but it is riskier than non-uniform ranks.

Implementation complexity is medium:

- The LoRA wrapper must compute two branches.
- The gate must receive the right hidden state and broadcast over the projection output.
- Input-dependent LoRA cannot be merged into the frozen base weights for evaluation.
- Fair parameter accounting is trickier because the router adds parameters.

Training stability risks:

- The gate may collapse to always selecting one branch.
- The gate may hover near `0.5`, making the method behave like one rank-4 LoRA with extra complication.
- With small E2E data, the router can overfit or add variance without improving BLEU.
- If sequence-wise gates are implemented incorrectly, they can leak future tokens in causal training.

Compute cost is still reasonable for two rank-2 branches, but it adds extra forward operations and prevents merged inference.

## Experimental Plan

### Stage 1: Reproduction Baseline

Train the planned baseline:

```text
model: gpt2-medium
dataset: E2E NLG
targets: attention query and value slices
rank: r_q = r_v = 4 in all 24 layers
alpha: 32
effective scale: 8
dropout: 0.1
```

Report:

- Trainable parameter count.
- Validation loss curve.
- BLEU, NIST, METEOR, ROUGE-L, CIDEr with the same decoding script.
- GPU memory and approximate wall-clock time if available.

### Stage 2: Fixed-Budget Non-Uniform Ranks

Run at least one same-budget schedule:

```text
Depth-heavy:
layers 0-7:   q=2, v=2
layers 8-15:  q=4, v=4
layers 16-23: q=6, v=6
```

Optional second schedule:

```text
Value-heavy:
all layers: q=2, v=6
```

Keep all other settings identical to baseline. The most important control is identical decoding and evaluation.

### Stage 3: Optional Gated Two-LoRA Variant

Only attempt this after the baseline and fixed-rank schedule are working.

Recommended setup:

```text
two branches per adapted projection
r_1 = 2, r_2 = 2
token-wise scalar gate per layer
gate shared by q and v
branch scale = 8
same optimizer, dropout, preprocessing, decoding, and evaluation
```

Report extra diagnostics:

- Mean and standard deviation of gate values per layer.
- Fraction of gates below `0.2` or above `0.8`.
- Whether the gate collapses to one branch.
- Trainable parameter count including and excluding gate parameters.

### Suggested Run Matrix

Minimal:

```text
1. Uniform LoRA r=4 baseline
2. Depth-heavy rank schedule, same budget
```

Better:

```text
1. Uniform LoRA r=4 baseline, seed 0
2. Depth-heavy same-budget schedule, seed 0
3. Value-heavy same-budget schedule, seed 0
4. Repeat best two settings with seed 1, and seed 2 if compute allows
```

Stretch:

```text
5. Gated two-branch LoRA, rank 2 + rank 2, seed 0
```

## Pitfalls and Honest Presentation

Key pitfalls:

- BLEU may not improve because the E2E task is small and uniform LoRA is already strong.
- Small differences may be seed noise, especially with generation metrics.
- Different trainable parameter counts make results hard to interpret.
- Different decoding settings can overwhelm the effect of the LoRA variant.
- Gated LoRA cannot be merged into the base model, so compare inference latency honestly.
- Gate architectures can leak target information if sequence pooling is not causal-safe.
- A warmup-chosen rank schedule should be described as using a small allocation probe, not as a purely hand-designed schedule.

How to present if the extension does not beat baseline:

- Emphasize parameter-matched comparison and controlled evaluation.
- Report trainable parameters, memory, and runtime for each method.
- Show that non-uniform allocation is easy to implement and stable even if it does not improve metrics.
- For gated LoRA, show gate statistics and discuss whether it learned meaningful conditional behavior.
- Frame the result as evidence about E2E/GPT-2 Medium rather than a universal conclusion about LoRA.

## Recommendation

Use fixed-budget non-uniform rank allocation as the main extension. It has a clear connection to AdaLoRA and budgeted PEFT, but it is simple enough for a course replication: no extra router, no merging issues, no major training instability, and clean parameter matching.

The most feasible first experiment is:

```text
Uniform baseline:
all layers q=4, v=4

Depth-heavy same-budget variant:
layers 0-7:   q=2, v=2
layers 8-15:  q=4, v=4
layers 16-23: q=6, v=6
```

If time remains, add the value-heavy schedule `q=2, v=6`. Attempt the gated two-LoRA variant only after the baseline and rank-pattern experiments are producing reliable metrics.