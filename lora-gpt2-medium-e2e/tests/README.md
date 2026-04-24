# Test Placeholder

Recommended first tests:

- Zero-initialized LoRA wrapper matches the frozen base layer output.
- GPT-2 injection leaves only LoRA parameters trainable.
- Prompt tokens are masked with `-100` in labels.
- Generated text extraction removes the prompt and keeps only the continuation.
