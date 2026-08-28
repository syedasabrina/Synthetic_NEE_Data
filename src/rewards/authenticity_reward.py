from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class AuthenticityReward:
    """
    Computes an authenticity reward for candidate synthetic BIPs
    using BIPDomainSFT perplexity as the signal, combined with a
    repetition penalty.

    Raw perplexity alone is gameable: the lowest-perplexity text under
    a domain LM is often repetitive boilerplate, not genuinely varied
    authentic writing. PPO will find this shortcut if it is not
    guarded against. Two mitigations are applied:

    1. Batch-relative normalization -- reward reflects how authentic
       a candidate is RELATIVE to others in the same batch, rather
       than an absolute exp(-nll) value that clusters near a fixed
       point for any coherent English text.
    2. Repetition penalty -- a distinct-bigram ratio multiplies the
       perplexity-based reward, so repetitive low-perplexity text is
       penalized rather than rewarded.

    This model is always frozen -- it is never updated during PPO.
    """

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-7B",
        adapter_path: str = "models/BIPDomainSFT",
        device: str = "cuda",
    ):
        self.device = device

        print("Loading BIPDomainSFT authenticity reward model...")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        print("AuthenticityReward ready.")

    def _repetition_score(self, text: str) -> float:
        """
        Distinct-bigram ratio in [0, 1]. Higher means less repetitive.
        A candidate that repeats the same phrase gets a low score
        here, which multiplies down its final reward regardless of
        how low its perplexity is.
        """
        tokens = text.split()
        if len(tokens) < 2:
            return 0.5  # too short to judge; neutral score
        bigrams = list(zip(tokens, tokens[1:]))
        if len(bigrams) == 0:
            return 0.5
        return len(set(bigrams)) / len(bigrams)

    @torch.no_grad()
    def _compute_nlls(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        """
        Computes per-example mean NLL for a list of texts, batched
        for GPU throughput. Returns a numpy array of raw NLL values
        (not yet converted to reward).
        """
        all_nlls = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            labels = inputs["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
            )

            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(
                reduction="none",
                ignore_index=-100,
            )
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())

            mask = (shift_labels != -100).float()
            example_nlls = (token_losses * mask).sum(dim=1) / mask.sum(dim=1)
            all_nlls.extend(example_nlls.float().cpu().numpy().tolist())

        return np.array(all_nlls)

    @torch.no_grad()
    def score_batch(
        self,
        texts: list[str],
        batch_size: int = 8,
        normalize: bool = True,
    ) -> list[float]:
        """
        Returns an authenticity reward in roughly [0, 1] for each
        candidate, combining batch-normalized perplexity with a
        repetition penalty.

        normalize=True (default, used during PPO): reward reflects
        relative standing within the batch via z-score + sigmoid.
        This avoids the compressed, uninformative signal that raw
        exp(-nll) produces when most candidates cluster near the
        same perplexity.

        normalize=False: raw exp(-nll) per example, useful for
        one-off diagnostic scoring outside a batch context (e.g. the
        test script comparing a handful of hand-written examples).
        """
        nlls = self._compute_nlls(texts, batch_size=batch_size)

        if normalize and len(nlls) > 1:
            mean, std = nlls.mean(), nlls.std() + 1e-6
            z = (nlls - mean) / std
            # lower NLL should give higher reward, so invert sign
            base_reward = 1.0 / (1.0 + np.exp(z))  # sigmoid(-z)
        else:
            base_reward = np.exp(-nlls)

        rep_scores = np.array([self._repetition_score(t) for t in texts])
        combined = base_reward * rep_scores

        return combined.tolist()

    @torch.no_grad()
    def score(self, texts: list[str]) -> list[float]:
        """
        Simple non-batched scoring for quick manual testing. Uses
        raw exp(-nll) x repetition penalty, no batch normalization,
        since a single-example call has no batch to normalize against.
        """
        return self.score_batch(texts, batch_size=1, normalize=False)