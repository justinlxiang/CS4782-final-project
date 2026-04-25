"""Generation helpers for E2E prompts."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch.nn import functional as F


def extract_continuation(full_text: str, prompt: str) -> str:
    """Remove the prompt prefix and common GPT-2/E2E stop markers."""
    continuation = full_text[len(prompt) :] if full_text.startswith(prompt) else full_text
    for marker in ("<|endoftext|>", "\n\n"):
        if marker in continuation:
            continuation = continuation.split(marker, 1)[0]
    return continuation.strip()


def encode_generation_prompts(
    tokenizer: Any,
    prompts: Sequence[str],
    append_bos_to_context: bool = False,
    padding_side: str = "left",
) -> dict[str, torch.Tensor]:
    """Tokenize generation prompts, optionally appending GPT-2's BOS separator."""
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be either 'left' or 'right'.")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    rows = [list(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    if append_bos_to_context and tokenizer.eos_token_id is not None:
        rows = [row + [int(tokenizer.eos_token_id)] for row in rows]

    max_len = max(len(row) for row in rows)
    input_ids = []
    attention_mask = []
    for row in rows:
        pad_len = max_len - len(row)
        if padding_side == "left":
            input_ids.append([int(pad_token_id)] * pad_len + row)
            attention_mask.append([0] * pad_len + [1] * len(row))
        else:
            input_ids.append(row + [int(pad_token_id)] * pad_len)
            attention_mask.append([1] * len(row) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def _eos_token_ids(value: Any) -> list[int]:
    """Normalize one or more EOS token ids from config."""
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    return [int(item) for item in value]


def _calc_banned_ngram_tokens(
    generated: torch.Tensor,
    no_repeat_ngram_size: int,
    cur_len: int,
) -> list[list[int]]:
    """Return next-token bans for already-seen generated ngrams."""
    if no_repeat_ngram_size <= 0 or cur_len + 1 < no_repeat_ngram_size:
        return [[] for _ in range(generated.size(0))]

    banned_tokens: list[list[int]] = []
    for row in generated.tolist():
        generated_ngrams: dict[tuple[int, ...], list[int]] = {}
        for ngram in zip(*[row[i:] for i in range(no_repeat_ngram_size)]):
            prefix = tuple(ngram[:-1])
            generated_ngrams.setdefault(prefix, []).append(ngram[-1])
        start_idx = cur_len + 1 - no_repeat_ngram_size
        current_prefix = tuple(row[start_idx:cur_len])
        banned_tokens.append(generated_ngrams.get(current_prefix, []))
    return banned_tokens


def _enforce_repetition_penalty(
    scores: torch.Tensor,
    generated: torch.Tensor | None,
    repetition_penalty: float,
) -> None:
    """Apply CTRL-style repetition penalty in place."""
    if generated is None or repetition_penalty == 1.0:
        return
    for row_idx in range(scores.size(0)):
        for token_id in set(generated[row_idx].tolist()):
            if scores[row_idx, token_id] < 0:
                scores[row_idx, token_id] *= repetition_penalty
            else:
                scores[row_idx, token_id] /= repetition_penalty


def _postprocess_next_token_scores(
    scores: torch.Tensor,
    generated: torch.Tensor | None,
    cur_len: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    min_length: int,
    eos_token_ids: Sequence[int],
) -> torch.Tensor:
    """Mirror the filtering done by the official LoRA beam script."""
    _enforce_repetition_penalty(scores, generated, repetition_penalty)
    if cur_len < min_length:
        for eos_token_id in eos_token_ids:
            scores[:, eos_token_id] = -float("inf")
    if generated is not None and no_repeat_ngram_size > 0:
        banned_tokens = _calc_banned_ngram_tokens(
            generated,
            no_repeat_ngram_size=no_repeat_ngram_size,
            cur_len=cur_len,
        )
        for row_idx, row_bans in enumerate(banned_tokens):
            if row_bans:
                scores[row_idx, row_bans] = -float("inf")
    return scores


def _reorder_past_key_values(past_key_values: Any, beam_idx: torch.Tensor) -> Any:
    """Reorder Hugging Face GPT-2 KV cache after beam selection."""
    return tuple(
        tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
        for layer_past in past_key_values
    )


def _next_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return position ids for the final token in an extended attention mask."""
    return attention_mask.long().cumsum(dim=-1)[:, -1:].sub(1).clamp_min(0)


def _decode_generated_ids(
    tokenizer: Any,
    token_ids: Sequence[int],
    eos_token_ids: Sequence[int],
) -> str:
    """Decode generated token ids, cutting at EOS or double-newline markers."""
    trimmed = []
    for token_id in token_ids:
        if int(token_id) in eos_token_ids:
            break
        trimmed.append(int(token_id))
    text = tokenizer.decode(trimmed, skip_special_tokens=True)
    for marker in ("<|endoftext|>", "\n\n"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


@torch.no_grad()
def official_style_beam_search_continuations(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    config: dict[str, Any],
    device: torch.device,
) -> list[str]:
    """Generate with a Hugging Face implementation of Microsoft's LoRA beam loop."""
    generation = config["generation"]
    num_beams = int(generation["num_beams"])
    max_new_tokens = int(generation["max_new_tokens"])
    length_penalty = float(generation["length_penalty"])
    repetition_penalty = float(generation.get("repetition_penalty", 1.0))
    no_repeat_ngram_size = int(generation["no_repeat_ngram_size"])
    min_length = int(generation.get("min_length", 0))
    eos_token_ids = _eos_token_ids(generation.get("eos_token_id"))

    encoded = encode_generation_prompts(
        tokenizer,
        prompts,
        append_bos_to_context=bool(config["data"].get("append_bos_to_context", False)),
        padding_side="left",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    batch_size = input_ids.size(0)

    beam_input_ids = input_ids.repeat_interleave(num_beams, dim=0)
    beam_attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)
    position_ids = beam_attention_mask.long().cumsum(dim=-1).sub(1).clamp_min(0)

    model.eval()
    outputs = model(
        input_ids=beam_input_ids,
        attention_mask=beam_attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values

    beam_scores = torch.zeros(batch_size, num_beams, dtype=torch.float, device=device)
    batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * num_beams
    generated: torch.Tensor | None = None
    best_scores = torch.full((batch_size,), -float("inf"), dtype=torch.float, device=device)
    best_sequences: list[list[int] | None] = [None] * batch_size

    for step in range(max_new_tokens):
        logits = _postprocess_next_token_scores(
            logits,
            generated,
            cur_len=step,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_length=min_length,
            eos_token_ids=eos_token_ids,
        )
        log_probs = F.log_softmax(logits, dim=-1)
        vocab_size = log_probs.size(-1)

        if step == 0:
            next_scores = log_probs.view(batch_size, num_beams, vocab_size)[:, 0, :]
        else:
            next_scores = beam_scores.unsqueeze(-1) + log_probs.view(
                batch_size,
                num_beams,
                vocab_size,
            )
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)

        next_scores, next_tokens = torch.topk(
            next_scores,
            num_beams,
            dim=1,
            largest=True,
            sorted=True,
        )
        beam_id = next_tokens // vocab_size
        token_id = (next_tokens % vocab_size).view(-1, 1)
        beam_idx = (beam_id + batch_offsets).view(-1)

        past_key_values = _reorder_past_key_values(past_key_values, beam_idx)
        beam_scores = next_scores
        if generated is None:
            generated = token_id.detach()
        else:
            generated = torch.cat((generated[beam_idx], token_id.detach()), dim=1)

        flat_scores = beam_scores.view(-1)
        flat_last_tokens = generated[:, -1]
        current_length = generated.size(1)

        for flat_idx in range(generated.size(0)):
            batch_idx = flat_idx // num_beams
            is_eos = int(flat_last_tokens[flat_idx]) in eos_token_ids
            if is_eos:
                candidate_score = flat_scores[flat_idx] / (current_length**length_penalty)
                if candidate_score > best_scores[batch_idx]:
                    best_scores[batch_idx] = candidate_score
                    best_sequences[batch_idx] = generated[flat_idx].tolist()
                flat_scores[flat_idx] = -float("inf")

        for flat_idx in range(generated.size(0)):
            batch_idx = flat_idx // num_beams
            candidate_score = flat_scores[flat_idx] / (current_length**length_penalty)
            if candidate_score > best_scores[batch_idx]:
                best_scores[batch_idx] = candidate_score
                best_sequences[batch_idx] = generated[flat_idx].tolist()

        if step == max_new_tokens - 1:
            break

        selected_attention_mask = beam_attention_mask[beam_idx]
        beam_attention_mask = torch.cat(
            (
                selected_attention_mask,
                torch.ones(
                    selected_attention_mask.size(0),
                    1,
                    dtype=selected_attention_mask.dtype,
                    device=device,
                ),
            ),
            dim=1,
        )
        outputs = model(
            input_ids=token_id,
            attention_mask=beam_attention_mask,
            position_ids=_next_position_ids(beam_attention_mask),
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

    return [
        _decode_generated_ids(tokenizer, sequence or [], eos_token_ids)
        for sequence in best_sequences
    ]


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
    decoder = generation.get("decoder", "official_beam")
    if decoder == "official_beam":
        return official_style_beam_search_continuations(model, tokenizer, prompts, config, device)
    if decoder != "hf_generate":
        raise ValueError(f"Unknown generation decoder: {decoder}")

    encoded = encode_generation_prompts(
        tokenizer,
        prompts,
        append_bos_to_context=bool(config["data"].get("append_bos_to_context", False)),
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
    input_length = encoded["input_ids"].shape[1]
    for row in generated:
        continuation_ids = row[input_length:]
        continuations.append(tokenizer.decode(continuation_ids, skip_special_tokens=True).strip())
    return continuations


def format_predictions(prompts: Sequence[str], generations: Sequence[str]) -> list[dict[str, str]]:
    """Create serializable prediction records."""
    return [
        {"id": str(index), "prompt": prompt, "prediction": generation}
        for index, (prompt, generation) in enumerate(zip(prompts, generations))
    ]
