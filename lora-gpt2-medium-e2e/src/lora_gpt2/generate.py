"""Generation helpers for E2E prompts."""

from __future__ import annotations

from typing import Any, Sequence

import torch


def extract_continuation(full_text: str, prompt: str) -> str:
    """Remove the prompt prefix and common GPT-2/E2E stop markers."""
    continuation = full_text[len(prompt) :] if full_text.startswith(prompt) else full_text
    for marker in ("<|endoftext|>", "\n\n"):
        if marker in continuation:
            continuation = continuation.split(marker, 1)[0]
    return continuation.strip()


@torch.no_grad()
def generate_continuations(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    config: dict[str, Any],
    device: torch.device,
) -> list[str]:
    """Generate continuations for prompt-only E2E inputs."""
    generation = config["generation"]
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    model.eval()
    generated = model.generate(
        **encoded,
        max_new_tokens=int(generation["max_new_tokens"]),
        num_beams=int(generation["num_beams"]),
        length_penalty=float(generation["length_penalty"]),
        no_repeat_ngram_size=int(generation["no_repeat_ngram_size"]),
        repetition_penalty=float(generation.get("repetition_penalty", 1.0)),
        early_stopping=bool(generation.get("early_stopping", True)),
        eos_token_id=generation.get("eos_token_id"),
        pad_token_id=tokenizer.pad_token_id,
    )

    continuations = []
    prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
    for row, prompt_len in zip(generated, prompt_lengths):
        continuation_ids = row[int(prompt_len) :]
        continuations.append(tokenizer.decode(continuation_ids, skip_special_tokens=True).strip())
    return continuations


def format_predictions(prompts: Sequence[str], generations: Sequence[str]) -> list[dict[str, str]]:
    """Create serializable prediction records."""
    return [
        {"id": str(index), "prompt": prompt, "prediction": generation}
        for index, (prompt, generation) in enumerate(zip(prompts, generations))
    ]
