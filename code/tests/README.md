# Tests

Pytest coverage for the implementation used by the poster experiments.

## Covered areas

- LoRA layer math and zero-initialized equivalence to the frozen base layer.
- GPT-2 injection, trainable-parameter filtering, and checkpointing.
- E2E/DART/WebNLG preprocessing and prompt-label masking.
- Beam-generation text extraction and metric formatting.
- AdaLoRA-Lite gates, pruning schedules, and static allocation export support.
- MoLE router/expert wiring and generation-time routing behavior.

Run from `code/`:

```bash
pytest
```
