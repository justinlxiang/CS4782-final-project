# Poster Sections Draft

## Title and Authors
- Suggested title: Adaptive Rank Allocation for LoRA Fine-Tuning on GPT-2 Medium
- Subtitle option: Reproducing LoRA on E2E NLG and extending it with budgeted rank schedules / AdaLoRA-Lite
- Authors: add group member names, Cornell University, CS 4782 Final Project

## Introduction / Background / Motivation
- Full fine-tuning adapts every parameter, which is expensive to train, store, and deploy for each downstream task.
- LoRA freezes the pretrained model and learns a low-rank update, Delta W = B A, inside selected Transformer projections.
- The LoRA paper reports that GPT-2 Medium can match full fine-tuning on E2E NLG with about 0.35M trainable parameters instead of 354M.
- Our goal was to reproduce the GPT-2 Medium + E2E LoRA result and test whether rank should be distributed uniformly or adaptively across layers.

## Methodology
- Model and data: frozen `gpt2-medium` on the E2E NLG restaurant data-to-text benchmark.
- Baseline: LoRA adapters on the query and value slices of GPT-2 attention `c_attn`; base model weights remain frozen.
- Training setup: rank sweeps over uniform LoRA, official-style beam decoding, and standard E2E metrics: BLEU, NIST, METEOR, ROUGE-L, CIDEr.
- Extension 1: same-budget non-uniform rank schedules, including late-ramp, early-ramp, query-heavy, and value-heavy allocations.
- Extension 2: AdaLoRA-Lite starts with a larger maximum rank, scores rank directions during training, and prunes to a fixed target budget of 192 rank units.

## Results
- Uniform LoRA reproduces the paper trend: very small ranks work surprisingly well, and `r=4` is a strong low-parameter baseline.
- Same-budget late-ramp rank allocation improved rank-4 BLEU from 67.30 to 68.39, suggesting later GPT-2 layers benefited more from adaptation capacity on E2E.
- Query-heavy and value-heavy schedules did not beat late-ramp, so depth allocation mattered more than projection-type allocation in these runs.
- Best AdaLoRA-Lite run (`rmax6`, target 192 rank units) reached BLEU 68.85 and ROUGE-L 71.42, close to the LoRA paper target of BLEU 68.9 / ROUGE-L 71.3.
- AdaLoRA-Lite used the same active rank budget as uniform `r=4`: 192 active rank units, about 0.39M effective adapter parameters.
- The `rmax6` implementation allocated 0.59M trainable tensor parameters during adaptive training; pruned/masked components are present in the tensors but inactive in the final budget.

## Conclusion
- LoRA successfully compresses task adaptation for GPT-2 Medium: sub-million-parameter adapters can approach full fine-tuning quality on E2E NLG.
- Rank allocation is meaningful: distributing the same rank budget toward later layers outperformed uniform rank at `r=4`.
- AdaLoRA-Lite gave the strongest result, showing that simple adaptive pruning can recover paper-level generation quality at the same final active rank budget as uniform `r=4`.
- Main limitation: results are from a single task and mostly single-seed runs, so small metric differences should be interpreted cautiously.

## Future Work
- Repeat the best configurations over multiple random seeds to separate real effects from generation metric variance.
- Add validation loss and wall-clock/memory plots to compare efficiency beyond final generation quality.
- Try official AdaLoRA's SVD-style parameterization and compare it against this simpler importance-pruning version.
- Test rank allocation on WebNLG or DART to see whether late-layer concentration generalizes beyond E2E.

## References
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
- Zhang, Q., Chen, M., Bukharin, A., Karampatziakis, N., He, P., Cheng, Y., Chen, W., & Zhao, T. (2023). AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning. ICLR 2023 / arXiv:2303.10512.
- Novikova, J., Dusek, O., & Rieser, V. (2017). The E2E Dataset: New Challenges for End-to-End Generation. SIGDIAL 2017 / arXiv:1706.09254.
