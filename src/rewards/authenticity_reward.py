from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class AuthenticityReward:
    """
    Contrastive authenticity reward for candidate synthetic BIPs.

    Raw perplexity under BIPDomainSFT measures general fluency, not
    domain membership. Calibration showed off-domain text ("The weather
    today is sunny...") scoring 0.63 against 0.61 for a real BIP --
    short, clean English is easy for any language model regardless of
    what it was fine-tuned on.

    The contrastive signal measures how much the domain fine-tune
    specifically helped:

        delta = nll_base - nll_finetuned

    Generic text has delta near zero because both models predict it
    equally well. Real BIP text has large positive delta because
    fine-tuning lowered its loss specifically. Measured deltas:

        real BIP (Element3)   0.93
        real BIP (Element6)   0.61
        too short             0.18
        repetitive            0.17
        off-domain weather    0.11

    Two guards sit on top of the contrastive score:

    Repetition penalty. Repeated phrases are trivially predictable, so
    they score well on any perplexity signal. A distinct-bigram ratio
    multiplies the reward down.

    Degenerate-text floor. Contrastive scoring discards absolute
    perplexity, so text neither model can predict (nll_ft 7.53 for
    "We monitored our goals") still scores on delta alone. A ceiling on
    nll_finetuned catches that.

    Both models frozen -- neither is updated during training.
    """

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-7B",
        adapter_path: str = "models/BIPDomainSFT",
        device: str = "cuda",
        nll_ceiling: float = 5.0,
        delta_scale: float = 3.0,
        delta_midpoint: float = 0.45,
    ):
        """
        nll_ceiling: candidates whose fine-tuned NLL exceeds this are
        treated as degenerate and heavily penalized. Real BIPs sit
        around 2.2-2.3; the "too short" probe hit 7.5.

        delta_scale / delta_midpoint: shape the absolute-mode sigmoid.
        Previously sigmoid(delta * 4.0) put delta=0.11 at 0.61, so
        off-domain text kept a high floor. Centering on 0.45 -- between
        the off-domain cluster (~0.15) and real BIPs (~0.6-0.9) -- maps
        near-zero delta close to zero and real BIPs above 0.6.
        """
        self.device = device
        self.nll_ceiling = nll_ceiling
        self.delta_scale = delta_scale
        self.delta_midpoint = delta_midpoint

        print("Loading BIPDomainSFT authenticity reward model...")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # one base model load; the PEFT adapter toggles on and off, so
        # both distributions come from a single 7B in memory
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
        tokens = text.split()
        if len(tokens) < 2:
            return 0.2
        bigrams = list(zip(tokens, tokens[1:]))
        if not bigrams:
            return 0.2
        return len(set(bigrams)) / len(bigrams)

    @torch.no_grad()
    def _nll_batch(self, texts: list[str], batch_size: int) -> np.ndarray:
        """
        Per-example mean NLL under whichever adapter state is active.
        Caller controls adapter state.
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
        Authenticity reward per candidate.

            delta    = nll_base - nll_finetuned
            fit      = sigmoid((delta - midpoint) * scale)   [absolute]
                       or batch z-scored sigmoid             [normalized]
            floor    = 0 if nll_finetuned > nll_ceiling else 1
            reward   = fit * repetition * floor

        normalize=True (used in training loops): delta is z-scored
        within the batch, so the reward reflects relative standing
        among candidates produced in the same step. This gives a
        well-spread signal rather than values bunched near a point.

        normalize=False: absolute sigmoid, for comparing hand-written
        probes across separate calls.
        """
        nll_ft = self._nll_batch(texts, batch_size)

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
            fit = 1.0 / (1.0 + np.exp(
                -(delta - self.delta_midpoint) * self.delta_scale
            ))

        rep = np.array([self._repetition_score(t) for t in texts])
        floor = (nll_ft <= self.nll_ceiling).astype(float)

        reward = fit * rep * floor

        if return_components:
            return {
                "reward": reward.tolist(),
                "nll_finetuned": nll_ft.tolist(),
                "nll_base": nll_base.tolist(),
                "delta": delta.tolist(),
                "fit": fit.tolist(),
                "repetition": rep.tolist(),
                "floor": floor.tolist(),
            }
        return reward.tolist()

    @torch.no_grad()
    def score(self, texts: list[str]) -> list[float]:
        """Absolute (non batch-normalized) scoring, for diagnostics."""
        return self.score_batch(texts, batch_size=4, normalize=False)