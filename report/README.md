# Report

The 2-page final project report (`group_topic_2page_report.pdf`) will be added
here by the May 12 deadline per the CS 4782 final-deliverable instructions.

While drafting, the report follows the rubric's six sections:

1. **Introduction** — problem, paper title/authors, paper contributions.
2. **Chosen Result** — the GPT-2 Medium / E2E NLG row from the LoRA paper
   (Hu et al., 2021, Table 3): LoRA `r=4` on Q/V at 0.35 M trainable
   parameters and 70.4 BLEU.
3. **Methodology** — our from-scratch LoRA reimplementation, the AdaLoRA-Lite
   and Depth-Weighted LoRA extensions, and the Mixture-of-LoRA-Experts router.
4. **Results & Analysis** — official E2E metrics, the Pareto frontier, and
   the cross-task MoLE generalization study.
5. **Reflections** — lessons learned and open directions.
6. **References** — full citations for LoRA, AdaLoRA, MoLE, E2E, DART, WebNLG.

The poster (this repo's `poster/`) summarizes the same content visually, and
the repo-root [`README.md`](../README.md) contains the longer version of the
methodology + results presented here.
