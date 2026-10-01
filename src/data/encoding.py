"""
src/data/encoding.py

Shared prompt/completion encoding for generator SFT and BoN retraining.

Fixes two defects in the previous builders:

1. No EOS. The Gemma tokenizer adds no EOS, and neither builder appended
   one, so the generator was never taught to stop and every completion ran
   to the token cap. EOS is now appended to the completion and included in
   the loss.
2. Boundary mismatch. The old code tokenized prompt + completion as one
   string and masked the first len(tokenize(prompt)) ids, which can be off
   by a token where the prompt and completion meet. Here the prompt and
   completion are tokenized separately and concatenated, so the mask is
   exact and matches what the sampler feeds the model at generation time
   (the prompt alone, tokenized the same way).

Examples that do not fit in max_length are dropped, not truncated. A
truncated completion has no EOS, and training on those teaches the model
that text can end mid-sentence.
"""

from __future__ import annotations


def encode_prompt_completion(
    tokenizer,
    prompt: str,
    completion: str,
    max_length: int,
    append_eos: bool = True,
):
    """
    Returns dict(input_ids, attention_mask, labels), padded to max_length,
    or None if prompt + completion (+ EOS) does not fit.
    """
    prompt_ids = tokenizer(
        prompt, add_special_tokens=True, truncation=False,
    )["input_ids"]
    comp_ids = tokenizer(
        completion, add_special_tokens=False, truncation=False,
    )["input_ids"]
    if append_eos:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has no eos_token_id")
        comp_ids = comp_ids + [tokenizer.eos_token_id]

    n = len(prompt_ids) + len(comp_ids)
    if n > max_length or len(comp_ids) < 2:
        return None

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    pad_len = max_length - n

    return {
        "input_ids": prompt_ids + comp_ids + [pad_id] * pad_len,
        "attention_mask": [1] * n + [0] * pad_len,
        "labels": [-100] * len(prompt_ids) + comp_ids + [-100] * pad_len,
    }


def encode_many(tokenizer, prompts, completions, max_length, append_eos=True):
    """Encodes a list of pairs; returns (columns dict, n_dropped)."""
    cols = {"input_ids": [], "attention_mask": [], "labels": []}
    dropped = 0
    for p, c in zip(prompts, completions):
        enc = encode_prompt_completion(tokenizer, p, c, max_length, append_eos)
        if enc is None:
            dropped += 1
            continue
        for k in cols:
            cols[k].append(enc[k])
    return cols, dropped
