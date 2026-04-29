# Poster Asset Manifest

Use these as the main figure candidates for a 36x24 landscape poster. All figures are high-resolution PNGs.

1. `01_key_results_bleu_rouge.png`: best overview figure; compares paper target, uniform reproduction, late-ramp extension, and best AdaLoRA-Lite.
2. `02_uniform_rank_sweep_bleu.png`: LoRA reproduction rank sweep over ranks 1, 2, 4, 8, and 16.
3. `03_rank_allocation_ablation_bleu.png`: same-budget non-uniform rank experiments; late-ramp r=4 is the clearest extension result.
4. `04_adalora_lite_variants_bleu_params.png`: AdaLoRA-Lite variants, active rank budget, and allocated trainable tensor counts.
5. `05_late_ramp_rank_distribution.png`: simple visual explanation of the hand-designed late-ramp rank schedule.
6. `06_adalora_lite_best_rank_distribution.png`: final adaptive rank distribution for the best AdaLoRA-Lite run.
7. `07_lora_low_rank_update_diagram.png`: conceptual LoRA diagram for background/methodology.
8. `08_adalora_lite_workflow_diagram.png`: conceptual AdaLoRA-Lite workflow diagram for methodology/extension.
9. `09_compact_results_table.png`: optional compact table image if you have poster space.
10. `compiled_poster_metrics.csv`: source metrics compiled from all six result folders.
11. `poster_sections_bullets.md`: concise bullets for the required poster sections.

Recommended layout: use `07` and `08` in methodology, `01` as the main results anchor, `03` and `06` as extension evidence, and `02`/`04` as smaller supporting figures.

Note: AdaLoRA-Lite `rmax6` targets the same active final rank budget as uniform `r=4` (192 active rank units, about 0.39M effective adapter parameters). The 0.59M value is the overcomplete trainable tensor allocation before/with masking, not the final active budget.
