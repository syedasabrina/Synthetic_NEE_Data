from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys



class AuthenticityReward:
    """
    Computes an authenticity reward for candidate synthetic BIPs
    using BIPDomainSFT perplexity as the signal.

    Lower perplexity under BIPDomainSFT means the candidate looks
    more like real principal writing. The reward is the negative
    normalized mean per-token log-likelihood, scaled to [0, 1].

    This model is always frozen -- it is never updated during PPO.
    """

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-7B",
        adapter_path: str = "models/BIPDomainSFT",
        device: str = "cuda",
    ):
        self.device = device

        print(f"Loading BIPDomainSFT authenticity reward model...")
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

        # freeze all parameters -- this model never trains
        for param in self.model.parameters():
            param.requires_grad = False

        print("AuthenticityReward ready.")

    @torch.no_grad()
    def score(self, texts: list[str]) -> list[float]:
        """
        Given a list of candidate BIP texts, returns a reward score
        in [0, 1] for each. Higher score = more authentic.

        Uses mean per-token negative log-likelihood (NLL) as the
        perplexity proxy. Lower NLL = higher reward.
        """
        nlls = []

        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False,
            ).to(self.device)

            # labels = input_ids so loss = per-token NLL
            outputs = self.model(
                **inputs,
                labels=inputs["input_ids"],
            )
            # outputs.loss is mean NLL across tokens
            nlls.append(outputs.loss.item())

        # convert NLL to reward: lower NLL = higher reward
        # use exp(-nll) to map to (0, 1] range
        rewards = [float(np.exp(-nll)) for nll in nlls]
        return rewards

    @torch.no_grad()
    def score_batch(self, texts: list[str], batch_size: int = 8) -> list[float]:
        """
        Batched version of score() for efficiency during PPO training.
        Pads within each batch for fast GPU throughput.
        """
        all_rewards = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            # compute per-example NLL manually when batching
            # because HF loss averages across the whole batch
            labels = inputs["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
            )

            # get per-token logits and compute per-example NLL
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

            # mean NLL per example (ignoring padding)
            mask = (shift_labels != -100).float()
            example_nlls = (token_losses * mask).sum(dim=1) / mask.sum(dim=1)
            rewards = [float(np.exp(-nll.item())) for nll in example_nlls]
            all_rewards.extend(rewards)

        return all_rewards
    

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    reward_model = AuthenticityReward(
        base_model_name="Qwen/Qwen2.5-7B",
        adapter_path="models/BIPDomainSFT",
        device="cpu",  # use cpu for testing on normal node
    )

    # test with a real BIP-like text vs generic text
    test_texts = [
        "Element6: Throughout the school year, we monitored our "
        "building improvement objectives by collecting MAP data "
        "every eight weeks and reviewing results with our grade "
        "level teams during collaborative planning time.",

        "The weather today is sunny with a high of 75 degrees. "
        "We recommend bringing sunscreen if you plan to be outside "
        "for extended periods of time.",

        "Element3: Our objectives are aligned to the district CSIP "
        "goals around literacy and mathematics proficiency. We "
        "established SMART goals for each building objective with "
        "measurable baseline data from last year's assessments.",
    ]

    print("Testing individual scoring:")
    rewards = reward_model.score(test_texts)
    for text, reward in zip(test_texts, rewards):
        print(f"  reward={reward:.4f} | {text[:60]}...")

    print("\nTesting batch scoring:")
    batch_rewards = reward_model.score_batch(test_texts, batch_size=2)
    for text, reward in zip(test_texts, batch_rewards):
        print(f"  reward={reward:.4f} | {text[:60]}...")