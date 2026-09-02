from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class AuthenticityReward:
    """
    Computes an authenticity reward for candidate synthetic BIPs using
    a CONTRASTIVE perplexity signal, plus a repetition penalty.

    Why contrastive: raw perplexity under BIPDomainSFT measures general
    fluency, not domain membership. Calibration showed off-domain text
    ("The weather today is sunny...") scoring 0.63 while real BIP text
    scored 0.61 -- short, clean, predictable English is easy for any
    language model regardless of domain. Optimizing that signal would
    push PPO toward generic fluent prose.

    The contrastive signal measures how much EASIER the text is for the
    domain-adapted model than for the base model:

        delta = nll_base - nll_finetuned

    Generic fluent text has delta near zero, since both models predict
    it equally well. Real BIP text has large positive delta, because
    fine-tuning specifically lowered its loss. That difference is what
    "sounds like a principal wrote it" actually means.

    Both models are frozen -- neither is updated during PPO.
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

        # single base model load; the PEFT adapter can be toggled on and
        # off, so we get both the base and fine-tuned distributions
        # without paying for two full 7B models in memory
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            device_map=device,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        print("AuthenticityReward ready (contrastive mode).")

    def _repetition_score(self, text: str) -> float:
        """
        Distinct-bigram ratio in [0, 1]. Higher means less repetitive.
        Multiplies down candidates that repeat phrases, which would
        otherwise score well on any perplexity-based signal.
        """
        tokens = text.split()
        if len(tokens) < 2:
            return 0.3  # too short to judge, and short generic text
                        # is exactly the failure mode we are guarding
                        # against, so do not award a neutral score
        bigrams = list(zip(tokens, tokens[1:]))
        if not bigrams:
            return 0.3
        return len(set(bigrams)) / len(bigrams)

    @torch.no_grad()
    def _nll_batch(self, texts: list[str], batch_size: int) -> np.ndarray:
        """
        Per-example mean NLL under whichever adapter state is currently
        active on self.model. Caller controls adapter state.
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

            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(
                reduction="none", ignore_index=-100,
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
        return_components: bool = False,
    ):
        """
        Returns an authenticity reward per candidate.

        The reward combines a contrastive domain-fit signal with a
        repetition penalty:

            delta   = nll_base - nll_finetuned   (higher = more domain-like)
            fit     = sigmoid(delta)  or  batch z-scored sigmoid
            reward  = fit * repetition_score

        normalize=True (used during PPO): delta is z-scored within the
        batch before the sigmoid, so the reward reflects relative
        standing among candidates the generator produced in that step.
        This gives PPO a well-spread signal instead of values bunched
        near a fixed point.

        normalize=False: absolute sigmoid(delta), for comparing
        hand-written examples across separate calls.
        """
        # fine-tuned distribution
        nll_ft = self._nll_batch(texts, batch_size)

        # base distribution: disable the LoRA adapter
        with self.model.disable_adapter():
            nll_base = self._nll_batch(texts, batch_size)

        delta = nll_base - nll_ft

        if normalize and len(delta) > 1:
            std = delta.std()
            if std < 1e-6:
                fit = np.full_like(delta, 0.5)
            else:
                z = (delta - delta.mean()) / std
                fit = 1.0 / (1.0 + np.exp(-z))
        else:
            # absolute mode: delta is typically small in magnitude, so
            # scale before the sigmoid to spread the output range
            fit = 1.0 / (1.0 + np.exp(-delta * 4.0))

        rep = np.array([self._repetition_score(t) for t in texts])
        reward = fit * rep

        if return_components:
            return {
                "reward": reward.tolist(),
                "nll_finetuned": nll_ft.tolist(),
                "nll_base": nll_base.tolist(),
                "delta": delta.tolist(),
                "fit": fit.tolist(),
                "repetition": rep.tolist(),
            }
        return reward.tolist()

    @torch.no_grad()
    def score(self, texts: list[str]) -> list[float]:
        """
        Absolute (non batch-normalized) scoring, for diagnostics.
        """
        return self.score_batch(texts, batch_size=4, normalize=False)