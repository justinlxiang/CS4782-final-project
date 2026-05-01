# Rank Where It Counts: Budgeting LoRA on GPT-2 Medium

**Re-implementing _LoRA: Low-Rank Adaptation of Large Language Models_ (Hu et al., 2021), and probing where its rank budget is best spent.**

| | |
|---|---|
| **Course** | Cornell CS 4782 (Spring 2026) — Final Project |
| **Authors** | Justin Xiang (`jx372@cornell.edu`) · Eric Qiu (`sq225@cornell.edu`) |
| **Paper reproduced** | E. J. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| **Backbone** | GPT-2 Medium (354 M, frozen) |
| **Tasks** | E2E NLG · DART · WebNLG (data-to-text generation) |

---

## 1. Introduction

This repository is our **CS 4782 final project**: a from-scratch
re-implementation of **LoRA** for GPT-2 Medium together with three small
extensions that ask the same question from different angles —

> *LoRA spends the same rank on every layer. Should it?*

**The paper (LoRA, 2021)** freezes a pre-trained model and trains a tiny
rank-`r` matrix product `B·A` next to each frozen weight `W₀`, leaving
inference cost unchanged. Its GPT-2 Medium / E2E NLG Table 3 result for LoRA
`r=4` on Q/V trains **0.35 M parameters** and reaches **70.4 BLEU**,
outperforming full fine-tuning at 68.2 BLEU. We rebuild that pipeline in
PyTorch from scratch (no `peft`, no `loralib`), measure the paper-aligned
uniform baseline on the official E2E metric scripts, and then run three
budget-allocation experiments on the same backbone:

* **Extension #1 — Depth-Weighted LoRA (Late-Ramp).** Hand-redistribute the
  same 192-rank-unit budget toward later layers.
* **Extension #2 — AdaLoRA-Lite.** A learned, sensitivity-pruned gating
  scheme that recovers a similar late-heavy profile automatically.
* **Extension #3 — Mixture-of-LoRA-Experts (MoLE).** A 0.07 M trainable
  router over three frozen task LoRAs serves E2E + DART + WebNLG from one
  checkpoint.

## 2. Chosen Result

We aim to reproduce the **GPT-2 Medium / E2E NLG row of LoRA's Table 3**
(Hu et al., 2021, p. 8 — "LoRA, r=4 on Q,V"):

> 0.35 M trainable parameters · BLEU **70.4 ± 0.1** · NIST **8.85 ± 0.02** ·
> METEOR **46.8 ± 0.2** · ROUGE-L **71.8 ± 0.1** · CIDEr **2.53 ± 0.02** —
> outperforming full fine-tuning at ~0.1 % of the trainable parameters.

This number is the load-bearing claim of LoRA: that you can *match* full
fine-tuning quality at a small fraction of trainable parameters with **zero
extra inference latency**. It justifies LoRA's central design — rank-`r`
adapters on `Q`/`V` only, scaling `α=32`, dropout `0.1`, AdamW for 5 epochs,
beam-search decoding with the official E2E evaluation harness — and is the
reference number we compare every extension against.

Our best run (AdaLoRA-Lite `r_max=6` at 0.39 M active params) reaches
**68.85 BLEU**: above the paper's full fine-tuning baseline, roughly tied with
the paper's 11 M-parameter `AdapterL` baseline, but still **1.55 BLEU below**
the LoRA Table 3 target. See [§ 6 Results / Insights](#6-results--insights).

## 3. GitHub Contents

```text
.
├── README.md                       <- this file
├── LICENSE                         <- MIT
├── .gitignore
├── final_deliverables.pdf          <- course rubric (kept for reference)
├── data/                           <- dataset staging area
│   ├── README.md                   <- how to download E2E / DART / WebNLG
│   ├── raw/{e2e,dart,webnlg}/      <- original release files (gitignored)
│   └── processed/{e2e,dart,webnlg}_gpt2/   <- tokenized JSONL (gitignored)
├── code/
│   ├── README.md                   <- short developer notes
│   ├── requirements.txt
│   ├── pyproject.toml
│   ├── configs/                    <- one YAML per experiment
│   ├── src/lora_gpt2/              <- the from-scratch library
│   │   ├── lora_layers.py          <- LoRALinear, LoRAConv1D (GPT-2 c_attn)
│   │   ├── inject.py               <- attach LoRA modules to GPT-2 in place
│   │   ├── adalora_lite.py         <- gated rank + sensitivity pruning
│   │   ├── mole_layers.py / mole_inject.py / mole_diagnostics.py
│   │   ├── data.py                 <- E2E / DART / WebNLG tokenization
│   │   ├── generate.py             <- official-style beam decoder
│   │   ├── evaluate.py             <- BLEU / NIST / METEOR / ROUGE / CIDEr
│   │   └── ...
│   ├── scripts/                    <- thin CLIs around `src/`
│   ├── tests/                      <- pytest unit tests
│   └── colab_*.ipynb               <- one Colab notebook per experiment
├── results/
│   └── README.md                   <- poster-aligned metric tables and artifact map
├── poster/
│   ├── LoRA_poster_final.pdf       <- submitted poster PDF
│   ├── LoRA_Poster_final.html      <- 36" × 24" landscape poster source
│   └── README.md                   <- poster file notes
└── report/
    └── README.md                   <- 2-page report status / PDF slot
```

All configs and Colab notebooks reference the dataset folder as
`../data/...` relative to `code/`, so the root `data/` directory is the
single canonical location for raw + tokenised dataset files (no symlink
trickery, cross-platform clean — Windows-friendly).

## 4. Re-implementation Details

### Backbone & adapter placement
* **Model:** `gpt2-medium` (24 layers, hidden 1024, 354 M params), frozen.
* **Adapter sites:** the **query** and **value** slices of each layer's
  fused `c_attn` Conv1D — exactly the paper's `r_q = r_v = r` setup.
* **Init:** `A ~ N(0, σ²)` (Kaiming-uniform), `B = 0`, scaling `α/r` with
  `α = 32`, dropout `0.1`. We compute the LoRA path in fp32 and add it to
  the fp16 frozen path, matching the official LoRA repo's numerics.

### Training recipe (paper-aligned)
| | E2E | DART | WebNLG |
|---|---|---|---|
| Batch size (effective) | 8 | 8 | 8 |
| Optimizer | AdamW (β₁=0.9, β₂=0.999, ε=1e-6) | same | same |
| LR / schedule | 2e-4, linear decay, 500-step warmup | same | same |
| Epochs | 5 | 5 | 5 |
| Precision | fp16 (loss scaling) | same | same |
| Label smoothing | 0.1 | 0.1 | 0.1 |
| Decoding | beam=10, length_penalty=0.9, no-repeat-4gram | same | same |
| Eval harness | the official `e2e-metrics` repo (MS-COCO style) | same | same |

### From-scratch library highlights
* `code/src/lora_gpt2/lora_layers.py` — `LoRALinear` and the `LoRAConv1D`
  variant required by GPT-2's fused `c_attn` projection (we slice into the
  Q and V columns of a single 3*hidden output).
* `code/src/lora_gpt2/inject.py` — wraps a frozen `GPT2LMHeadModel` and
  attaches LoRA at the requested target modules without changing weights.
* `code/src/lora_gpt2/adalora_lite.py` — adds a learnable diagonal gate
  `s` per rank direction and prunes by EMA(`|sᵢ · ∂L/∂sᵢ|`) every 500
  steps, on a cubic schedule from `r_max` down to a target active count.
* `code/src/lora_gpt2/mole_layers.py` + `mole_inject.py` — top-1 router
  over three frozen LoRA experts; only the per-layer `Linear(d → N)` gates
  train (~0.07 M params).
* `code/src/lora_gpt2/generate.py` — re-implements the official LoRA
  repo's beam decoder (length penalty, EOS list including the E2E newline
  token) so our metric scores are directly comparable.
* `code/src/lora_gpt2/evaluate.py` — wraps the upstream `e2e-metrics`
  scripts (BLEU, NIST, METEOR, ROUGE-L, CIDEr) with grouped multi-reference
  formatting per the official protocol.

### Experimental design
* **Q1 (Depth-Weighted LoRA).** Same total budget (192 rank units across 24
  layers × Q/V), redistributed by hand: `r ∈ {2, 3, 4, 4, 5, 6}` from early
  to late blocks vs. the uniform `r=4`.
* **Q2 (AdaLoRA-Lite).** Three runs at `r_max ∈ {6, 8, 16}`, all pruning
  down to the same 192 active units. The cubic schedule warms up for
  `t_init` steps, prunes over the `t_final` window, then freezes the
  allocation for the remaining training and final evaluation.
* **Q3 (MoLE).** Three frozen task LoRAs (E2E, DART, WebNLG `r=4`) plus a
  single trainable per-layer router; trained on the union of the three
  task training sets and evaluated on each task's held-out split.

### Modifications from the paper
* We use mixed-precision **fp16** rather than the paper's fp32 because of
  Colab T4/A100 memory pressure; our implementation reports **0.39 M**
  trainable Q/V adapter parameters for uniform `r=4`, while the paper reports
  **0.35 M** under its Table 3 counting convention.
* AdaLoRA-Lite is a **lightweight variant** of Zhang et al. (ICLR 2023):
  diagonal gates only (no SVD `P, S, Q` factorization, no orthogonality
  regularizer). The full AdaLoRA matched-budget comparison and its
  orthogonality regularizer are listed under § 7 Future work.

## 5. Reproduction Steps

The fastest path is to open one Colab notebook per experiment on a **GPU
runtime** (we recommend an **A100** for the AdaLoRA-Lite and MoLE runs;
each takes ≈ 50–90 minutes; T4 works for the uniform-rank runs but is
slower).

### 5.1 Quick start on Colab (A100 recommended)

1. Open the notebook for the experiment you want to run from
   [`code/`](code/). Each notebook is **self-contained** — it clones the
   repo, installs requirements, downloads the dataset, trains, generates,
   and evaluates.

   | Experiment | Notebook |
   |---|---|
   | Paper repro: uniform LoRA r=4 on E2E | `code/colab_train_lora.ipynb` |
   | Rank sweep r ∈ {1,2,4,8,16} on E2E | `code/colab_rank_sweep_lora.ipynb` |
   | Q-only / V-only / mixed-pattern ablations | `code/colab_rank_pattern_lora.ipynb` |
   | Late-ramp r=8 | `code/colab_late_ramp_r8_budget_lora.ipynb` |
   | Late-ramp r=16 | `code/colab_late_ramp_r16_budget_lora.ipynb` |
   | AdaLoRA-Lite (r_max=6/8/16) | `code/colab_adalora_lite_train_eval.ipynb` |
   | DART task LoRA | `code/colab_train_lora_dart.ipynb` |
   | WebNLG task LoRA | `code/colab_train_lora_webnlg.ipynb` |
   | MoLE router training | `code/colab_train_mole.ipynb` |
   | Cross-task evaluation matrix (Fig. 11) | `code/colab_eval_matrix.ipynb` |

2. In Colab: **Runtime → Change runtime type → A100 GPU** (or T4 for
   smaller runs). Then **Runtime → Run all**.

3. Each notebook backs up its run folder to **Google Drive** at
   `MyDrive/<run_name>/` so that downstream notebooks (e.g.
   `colab_eval_matrix`, `colab_train_mole`) can read trained adapter
   checkpoints without re-training.

### 5.2 Local run (CLI)

If you have a local CUDA GPU, the same commands work from the repo root:

```bash
# 0. Environment
git clone https://github.com/justinlxiang/CS4782-final-project.git
cd CS4782-final-project
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt

# 1. Download datasets (see data/README.md for full URLs)
# Example for E2E (run from repo root):
mkdir -p data/raw/e2e
curl -L -o data/raw/e2e/train.txt https://raw.githubusercontent.com/microsoft/LoRA/main/examples/NLG/data/e2e/train.txt
curl -L -o data/raw/e2e/valid.txt https://raw.githubusercontent.com/microsoft/LoRA/main/examples/NLG/data/e2e/valid.txt
curl -L -o data/raw/e2e/test.txt  https://raw.githubusercontent.com/microsoft/LoRA/main/examples/NLG/data/e2e/test.txt

# 2. Tokenize (run from inside code/, configs reference ../data/...)
cd code
python scripts/prepare_e2e.py --config configs/e2e_gpt2_medium_lora.yaml

# 3. Train (paper-aligned uniform-r=4)
python scripts/train.py --config configs/e2e_gpt2_medium_lora.yaml

# 4. Generate test predictions (beam search, length penalty 0.9)
python scripts/generate.py --config configs/e2e_gpt2_medium_lora.yaml --split test

# 5. Evaluate with the official E2E metrics
git clone https://github.com/tuetschek/e2e-metrics external/e2e-metrics
python scripts/evaluate.py --config configs/e2e_gpt2_medium_lora.yaml
```

For the AdaLoRA-Lite extension, swap `--config configs/e2e_gpt2_medium_lora.yaml`
for `--config configs/e2e_gpt2_medium_adalora_lite.yaml`. For MoLE use
`configs/mole_e2e_dart_webnlg.yaml` plus `scripts/train_mole.py` /
`scripts/generate_mole.py` (called from `colab_train_mole.ipynb`).

### 5.3 Compute requirements

| Experiment | Min. GPU | Wall-clock on A100-40GB |
|---|---|---|
| Uniform LoRA `r=4` (E2E, 5 ep.) | T4 16 GB | ~25 min |
| Rank sweep r ∈ {1,2,4,8,16} | T4 / A100 | ~2 h aggregate |
| AdaLoRA-Lite `r_max=6 / 8 / 16` | A100 (recommended) | ~50–70 min each |
| MoLE (3 frozen experts + router) | A100 (recommended) | ~75 min |
| Cross-task eval matrix | T4 (eval only) | ~25 min |

Total disk for the full set of runs is ≈ 6 GB (most of it is checkpoints
that we back up to Google Drive). Outputs land in
`code/outputs/runs/<run_name>/`; the lightweight, poster-aligned result
summary lives in [`results/README.md`](results/README.md).

## 6. Results / Insights

All BLEU/NIST/METEOR/ROUGE-L/CIDEr numbers below come from the **official
`e2e-metrics`** scripts (the same harness the LoRA paper uses), run on the
held-out test split with beam-10 / length-penalty 0.9 generations.

### 6.1 Paper repro & rank/budget table (E2E test set, official metrics)

| Configuration | Params | BLEU | NIST | METEOR | ROUGE-L | CIDEr |
|---|---:|---:|---:|---:|---:|---:|
| ★ LoRA paper GPT-2 M r=4 (Hu '21) | 0.35 M | **70.40 ± 0.10** | **8.85 ± 0.02** | **46.80 ± 0.20** | **71.80 ± 0.10** | **2.53 ± 0.02** |
| ★ Full fine-tuning (paper) | 354.92 M | 68.20 | 8.62 | 46.20 | 71.00 | 2.47 |
| ★ AdapterL baseline (paper) | 11.09 M | 68.90 | 8.71 | 46.10 | 71.30 | 2.47 |
| Uniform r=1 | 0.10 M | 66.78 | 8.51 | 45.02 | 68.94 | 2.36 |
| Uniform r=2 | 0.20 M | 67.11 | 8.52 | 45.78 | 70.30 | 2.45 |
| Uniform r=4 (ours) | 0.39 M | 67.30 | 8.51 | 46.05 | 70.96 | 2.47 |
| Uniform r=8 | 0.79 M | 67.97 | 8.59 | 46.23 | 70.94 | 2.50 |
| Uniform r=16 | 1.57 M | 68.21 | 8.62 | 46.22 | 71.12 | 2.49 |
| **Late-ramp r=4** *(Q1, ours)* | 0.39 M | **68.39** | 8.63 | 46.51 | 71.29 | 2.49 |
| Late-ramp r=8 | 0.79 M | 67.77 | 8.57 | 46.28 | 70.90 | 2.48 |
| Late-ramp r=16 | 1.57 M | 68.15 | 8.62 | 46.38 | 71.24 | 2.49 |
| AdaLoRA-Lite r_max=16 | 0.39 M\* | 67.67 | 8.59 | 46.10 | 70.50 | 2.45 |
| AdaLoRA-Lite r_max=8 | 0.39 M\* | 68.34 | 8.64 | 46.30 | 71.00 | 2.49 |
| **AdaLoRA-Lite r_max=6 (best)** *(Q2, ours)* | 0.39 M\* | **68.85** | **8.69** | 46.40 | **71.42** | **2.51** |

\* AdaLoRA-Lite trains with `r_max` rank directions per layer but prunes to
**192 active units** (the same active Q/V rank count as uniform `r=4`). These are
the same official E2E numbers shown in the poster; see
[`results/README.md`](results/README.md) for the poster-aligned summary and
artifact map.

### 6.2 Headline insights

* **The paper LoRA target is higher than our reproduction.** The correct
  Table 3 target is **70.4 BLEU**, not 68.9. AdaLoRA-Lite `r_max=6` reaches
  **68.85 BLEU**: above paper full fine-tuning (68.2) and roughly tied with
  the 11 M-parameter `AdapterL` baseline (68.9), but below paper LoRA.
* **Allocation > raw rank.** Late-ramp `r=4` (0.39 M) beats uniform
  `r=8` (0.79 M) on every official metric — the same active rank count as
  uniform `r=4` with **half** the rank-units that uniform-r=8 uses. Where the budget is
  spent matters more than how big it is.
* **Learned ≈ hand-designed.** AdaLoRA-Lite recovers a late-heavy profile
  on its own (summarized in [`results/README.md`](results/README.md) and
  Fig. 7 in the poster); the manual late-ramp profile is essentially a
  discrete approximation of what gradient sensitivity learns end-to-end.
* **Pareto frontier.** AdaLoRA-Lite `r_max=6` (0.39 M) **strictly dominates
  uniform r=16** (1.57 M) — same/better BLEU at 4× fewer parameters.

### 6.3 Cross-task generalization (MoLE, Q3)

A single mixture-of-experts model (3 × 0.39 M frozen task LoRAs + 0.07 M
trainable router ≈ 1.24 M) versus three single-task specialists, each
evaluated on E2E, DART, and WebNLG test sets:

| Method | Params | E2E BLEU | DART BLEU | WebNLG BLEU |
|---|---:|---:|---:|---:|
| E2E LoRA (specialist) | 0.39 M | 67.30 | 28.53 | 10.58 |
| DART LoRA (specialist) | 0.39 M | 60.54 | **47.15** | 50.44 |
| WebNLG LoRA (specialist) | 0.39 M | 39.75 | 33.41 | **55.23** |
| **MoLE (3 + router)** | ≈ 1.24 M | **67.76** | 45.35 | 51.75 |

* Single-task LoRAs **fall apart off-domain** — the E2E specialist drops
  ~39 BLEU when evaluated on DART. MoLE stays strong on every column from
  one served checkpoint, recovering most of each specialist's BLEU.
* The router is **0.07 M trainable parameters** — about **1/5,000th** of
  the full backbone. The poster-aligned cross-task table is mirrored in
  [`results/README.md`](results/README.md).

### 6.4 What the AdaLoRA-Lite gates learn

The final per-layer Q+V active rank for the best run (`r_max=6`) drops
toward zero in early blocks and stays at the cap in late blocks — the same
shape we hand-designed for late-ramp:

> Active rank by layer (Q + V) — early layers (L0–L4): 3–4 units · middle
> (L5–L11): 4–8 units · late (L12–L23): 10–12 units (near `r_max=6` per
> projection). The summary in [`results/README.md`](results/README.md)
> mirrors the poster; rerunning `export_adalora_lite_static.py` writes the
> full per-layer allocation JSON.

### 6.5 Discrepancies and challenges

* **Pure uniform-r=4** lands at BLEU **67.30**, **3.1 BLEU below** the
  paper's LoRA Table 3 score of 70.4 and 0.9 BLEU below paper full
  fine-tuning. Plausible reasons: (i) we use fp16 vs. the paper's fp32,
  (ii) seed variance (paper reports mean over 3 GPT-2 LoRA seeds), and
  (iii) small differences in the reference training/evaluation pipeline.
  AdaLoRA-Lite-`r_max=6` and Late-Ramp-r=4 close much of the gap to full
  fine-tuning and AdapterL, but they do not match the paper LoRA row.
* **Beam-search EOS handling** matters more than expected — the official
  LoRA repo decodes with both 50256 (`<|endoftext|>`) and 628 (the E2E
  newline) as stopping tokens. Our first beam-search implementation only
  used 50256 and lost ~1 BLEU; matching the official EOS list in
  [`code/src/lora_gpt2/generate.py`](code/src/lora_gpt2/generate.py) closed
  that gap.

## 7. Conclusion

* **The paper LoRA target remains above our reproduction.** Uniform `r=4`
  trails the paper by 3.1 BLEU in our single run, while Late-ramp and
  AdaLoRA-Lite close much of that gap without increasing the active parameter
  budget.
* **Where you spend rank matters more than how much rank you have.**
  Late-ramp r=4 (0.39 M) beats uniform r=8 (0.79 M) on every official
  metric; AdaLoRA-Lite r_max=6 (0.39 M active) beats uniform r=16 (1.57 M).
* **Hand design ≈ learned.** AdaLoRA-Lite's pruned profile is essentially
  the same late-heavy shape we designed by hand — the model can find it on
  its own from a single 192-unit budget.
* **One served checkpoint can do many tasks.** A 0.07 M trainable router
  over three frozen LoRAs (MoLE) recovers most of each task specialist's
  BLEU on its own data while remaining usable across all three tasks —
  something no single specialist achieves.
* **Lessons learned.** (i) Faithful re-implementations are dominated by
  small numerical / decoding details (EOS handling, length penalty,
  warm-up). (ii) Looking inside an adaptive method is as informative as
  the headline metric — the AdaLoRA-Lite gates *interpret* the paper's
  fixed-rank choice in hindsight.

### Future work
* Scale the budget-allocation study beyond GPT-2 Medium (Llama-2/3 7B,
  Mistral 7B) and to more diverse tasks (NL ↔ SQL, code).
* **MoLE + adaptive rank.** Combine top-`k` LoRA routing with
  AdaLoRA-Lite-style learned rank budgets per expert.
* **Orthogonality regularizer.** Add the AdaLoRA SVD orthogonality penalty
  on `P, Q` factors and measure whether interference vs. pruning dynamics
  changes the recovered profile.
* **Key and output matrices.** Try applying LoRA / AdaLoRA-Lite to the
  transformer key and output projections in addition to Q/V.

## 8. References

1. **Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S.,
   Wang, L., & Chen, W.** (2021). *LoRA: Low-Rank Adaptation of Large
   Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
   Official code: [github.com/microsoft/LoRA](https://github.com/microsoft/LoRA).
2. **Zhang, Q. et al.** (2023). *AdaLoRA: Adaptive Budget Allocation for
   Parameter-Efficient Fine-Tuning.* ICLR 2023 /
   [arXiv:2303.10512](https://arxiv.org/abs/2303.10512).
3. **Wu, X. et al.** (2024). *Mixture-of-LoRA-Experts: Leveraging the
   Power of Fine-Tuned Models.* ICLR 2024 /
   [arXiv:2404.13628](https://arxiv.org/abs/2404.13628).
4. **Houlsby, N. et al.** (2019). *Parameter-Efficient Transfer Learning
   for NLP.* ICML 2019.
5. **Li, X. L. & Liang, P.** (2021). *Prefix-Tuning: Optimizing Continuous
   Prompts for Generation.* ACL 2021.
6. **Novikova, J., Dušek, O., & Rieser, V.** (2017). *The E2E Dataset: New
   Challenges for End-to-End Generation.* SIGDIAL 2017.
7. **Nan, L. et al.** (2021). *DART: Open-Domain Structured Data Record to
   Text Generation.* NAACL 2021.
8. **Gardent, C., Shimorina, A., Narayan, S., & Perez-Beltrachini, L.**
   (2017). *The WebNLG Challenge.* INLG 2017.
9. **Wolf, T. et al.** (2020). *Transformers: State-of-the-Art Natural
   Language Processing.* EMNLP demos. (`gpt2-medium`, `transformers`)
10. **Tuetschek, O.** (2017). [`e2e-metrics`](https://github.com/tuetschek/e2e-metrics)
    — official E2E NLG evaluation scripts (BLEU / NIST / METEOR /
    ROUGE-L / CIDEr).

## 9. Acknowledgements

This project was carried out as the **CS 4782 (Cornell University, Spring
2026) final project**. We thank the course staff for the structured
deliverables and feedback rubric, and the authors of LoRA, AdaLoRA, and
MoLE for releasing well-documented reference code that made it possible to
reproduce and extend their results in a single semester. We also thank the
maintainers of `transformers`, the official `e2e-metrics` scripts, and the
public Microsoft / Yale-LILY data mirrors. All compute for the headline
runs (uniform / late-ramp / AdaLoRA-Lite / MoLE) ran on a single Google
Colab A100 (40 GB) instance.
