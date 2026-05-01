# Results

This directory records the lightweight result tables used by the poster. Full
training outputs, checkpoints, and external metric clones are regenerated under
`code/outputs/` by the notebooks/scripts and are not duplicated here.

## E2E Official Metrics

| Configuration | Params | BLEU | NIST | METEOR | ROUGE-L | CIDEr |
|---|---:|---:|---:|---:|---:|---:|
| LoRA paper GPT-2 M r=4 | 0.35M | 70.40 ± 0.10 | 8.85 ± 0.02 | 46.80 ± 0.20 | 71.80 ± 0.10 | 2.53 ± 0.02 |
| Full fine-tuning, paper | 354.92M | 68.20 | 8.62 | 46.20 | 71.00 | 2.47 |
| AdapterL baseline, paper | 11.09M | 68.90 | 8.71 | 46.10 | 71.30 | 2.47 |
| Uniform r=1 | 0.10M | 66.78 | 8.51 | 45.02 | 68.94 | 2.36 |
| Uniform r=2 | 0.20M | 67.11 | 8.52 | 45.78 | 70.30 | 2.45 |
| Uniform r=4, ours | 0.39M | 67.30 | 8.51 | 46.05 | 70.96 | 2.47 |
| Uniform r=8 | 0.79M | 67.97 | 8.59 | 46.23 | 70.94 | 2.50 |
| Uniform r=16 | 1.57M | 68.21 | 8.62 | 46.22 | 71.12 | 2.49 |
| Late-ramp r=4 | 0.39M | 68.39 | 8.63 | 46.51 | 71.29 | 2.49 |
| Late-ramp r=8 | 0.79M | 67.77 | 8.57 | 46.28 | 70.90 | 2.48 |
| Late-ramp r=16 | 1.57M | 68.15 | 8.62 | 46.38 | 71.24 | 2.49 |
| AdaLoRA-Lite rmax=16 | 0.39M active | 67.67 | 8.59 | 46.10 | 70.50 | 2.45 |
| AdaLoRA-Lite rmax=8 | 0.39M active | 68.34 | 8.64 | 46.30 | 71.00 | 2.49 |
| AdaLoRA-Lite rmax=6, best | 0.39M active | 68.85 | 8.69 | 46.40 | 71.42 | 2.51 |

AdaLoRA-Lite trains with extra rank directions but prunes to 192 active rank
units, the same active budget as uniform `r=4`.

The paper's matching GPT-2 Medium LoRA row is **70.4 BLEU**, not 68.9. The
68.9 score is the paper's larger `AdapterL` baseline.

## Tight-Budget Ablations

| Configuration | E2E BLEU |
|---|---:|
| Query-heavy r=4 pattern | 67.09 |
| Uniform r=4 | 67.30 |
| Value-heavy r=4 pattern | 67.42 |
| Early-ramp r=4 | 67.89 |
| Late-ramp r=4 | 68.39 |
| AdaLoRA-Lite rmax=6 | 68.85 |

## MoLE Cross-Task Matrix

| Method | Params | E2E BLEU | DART BLEU | WebNLG BLEU |
|---|---:|---:|---:|---:|
| E2E LoRA specialist | 0.39M | 67.30 | 28.53 | 10.58 |
| DART LoRA specialist | 0.39M | 60.54 | 47.15 | 50.44 |
| WebNLG LoRA specialist | 0.39M | 39.75 | 33.41 | 55.23 |
| MoLE, 3 experts + router | about 1.24M | 67.76 | 45.35 | 51.75 |

MoLE uses three frozen uniform `r=4` LoRA experts plus a trainable router of
about 0.07M parameters. It is separate from the late-ramp and AdaLoRA-Lite
budget-allocation experiments.

## Regenerating Artifacts

Run the Colab notebooks or CLI commands from the root README. Generated
checkpoints, logs, predictions, and figures land in `code/outputs/runs/`; if
you want to keep a smaller reviewed subset in GitHub, mirror it under this
directory using the run names from the corresponding notebook/config.
