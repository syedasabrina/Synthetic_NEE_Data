from __future__ import annotations

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class AuthenticityReward:
    """
    Contrastive authenticity reward for candidate synthetic BIPs.

    Raw perplexity under BIPDomainSFT measures general fluency, not domain
    membership: off-domain text ("The weather today is sunny...") scored
    0.63 against 0.61 for a real BIP. The contrastive signal measures how
    much the domain fine-tune specifically helped:

        delta = nll_base - nll_finetuned

    Generic text has delta near zero. Real BIP text has large positive
    delta because the fine-tune lowered its loss specifically.

    Changes from the previous version:

    1. Element prefix. BIPDomainSFT trained on "ElementN: text". Probes
       carried that prefix and BoN candidates did not, so absolute levels
       were not comparable. Pass `elements` and every text is scored as
       "ElementN: text", the form the model was trained on.
    2. One pass. score_candidates returns both the batch-relative and the
       absolute reward from a single adapter-on and adapter-off pass. The
       old loop ran the same two passes twice per prompt.
    3. The repetition factor and the NLL ceiling are unchanged. The ceiling
       rejects text neither model can predict.

    The relative reward is z-scored within the candidates passed in, so it
    can only rank them. It cannot say whether the best one is good. Use
    `delta` (absolute) with a floor taken from held-out real BIPs for that
    (scripts/score_real_reference.py).

    Both models frozen. Neither is updated during training.
    """

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-7B",
        adapter_path: str = "models/BIPDomainSFT",
        device: str = "cuda",
        nll_ceiling: float = 5.0,
        delta_scale: float = 3.0,
        delta_midpoint: float = 0.45,
        max_length: int = 768,
    ):
        self.device = device
        self.nll_ceiling = nll_ceiling
        self.delta_scale = delta_scale
        self.delta_midpoint = delta_midpoint
        self.max_length = max_length

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

    @staticmethod
    def _repetition_score(text: str) -> float:
        tokens = text.split()
        if len(tokens) < 2:
            return 0.2
        bigrams = list(zip(tokens, tokens[1:]))
        return len(set(bigrams)) / len(bigrams)

    @staticmethod
    def with_prefix(texts: list[str], elements: list[str] | None) -> list[str]:
        if elements is None:
            return list(texts)
        if len(elements) != len(texts):
            raise ValueError("elements and texts must have the same length")
        return [f"{e}: {t}" for e, t in zip(elements, texts)]

    @torch.no_grad()
    def _nll_batch(self, texts: list[str], batch_size: int) -> np.ndarray:
        """
        Per-example mean NLL under whichever adapter state is active.
        Caller controls adapter state.
        """
        all_nlls = []
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(self.device)

            labels = inputs["input_ids"].clone()
            labels[inputs["attention_mask"] == 0] = -100

            # labels are not passed to the model: the loss is computed
            # below, so letting the model compute it too was wasted work
            logits = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
            ).view(shift_labels.size())

            mask = (shift_labels != -100).float()
            example_nlls = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_nlls.extend(example_nlls.cpu().numpy().tolist())

        return np.array(all_nlls)

    @torch.no_grad()
    def score_candidates(
        self,
        texts: list[str],
        elements: list[str] | None = None,
        batch_size: int = 4,
    ) -> dict:
        """
        One adapter-on and one adapter-off pass. Returns lists:

            delta, nll_finetuned, nll_base, repetition, floor,
            fit_abs, fit_rel, reward_abs, reward_rel

        reward_rel is z-scored within `texts` (ranking only). reward_abs
        uses the fixed sigmoid and is comparable across calls.
        """
        scored = self.with_prefix(texts, elements)

        nll_ft = self._nll_batch(scored, batch_size)
        with self.model.disable_adapter():
            nll_base = self._nll_batch(scored, batch_size)
        delta = nll_base - nll_ft

        fit_abs = 1.0 / (1.0 + np.exp(
            -(delta - self.delta_midpoint) * self.delta_scale
        ))
        if len(delta) > 1:
            std = delta.std()
            if std < 1e-6:
                fit_rel = np.full_like(delta, 0.5)
            else:
                z = (delta - delta.mean()) / std
                fit_rel = 1.0 / (1.0 + np.exp(-z))
        else:
            fit_rel = fit_abs

        rep = np.array([self._repetition_score(t) for t in texts])
        floor = (nll_ft <= self.nll_ceiling).astype(float)

        return {
            "delta": delta.tolist(),
            "nll_finetuned": nll_ft.tolist(),
            "nll_base": nll_base.tolist(),
            "repetition": rep.tolist(),
            "floor": floor.tolist(),
            "fit_abs": fit_abs.tolist(),
            "fit_rel": fit_rel.tolist(),
            "reward_abs": (fit_abs * rep * floor).tolist(),
            "reward_rel": (fit_rel * rep * floor).tolist(),
        }

    @torch.no_grad()
    def score_batch(
        self,
        texts: list[str],
        batch_size: int = 8,
        normalize: bool = True,
        return_components: bool = False,
        elements: list[str] | None = None,
    ):
        """
        Backward-compatible wrapper around score_candidates.

        normalize=True returns the batch-relative reward, False the
        absolute one. return_components keeps the old key names, with
        'reward' selected by `normalize`.
        """
        c = self.score_candidates(texts, elements=elements, batch_size=batch_size)
        use_rel = normalize and len(texts) > 1
        reward = c["reward_rel"] if use_rel else c["reward_abs"]
        if return_components:
            return {
                "reward": reward,
                "nll_finetuned": c["nll_finetuned"],
                "nll_base": c["nll_base"],
                "delta": c["delta"],
                "fit": c["fit_rel"] if use_rel else c["fit_abs"],
                "repetition": c["repetition"],
                "floor": c["floor"],
            }
        return reward

    @torch.no_grad()
    def score(self, texts: list[str], elements: list[str] | None = None) -> list[float]:
        """Absolute (non batch-normalized) scoring, for diagnostics."""
        return self.score_batch(texts, batch_size=4, normalize=False,
                                elements=elements)
