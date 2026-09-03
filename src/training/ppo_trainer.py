from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from transformers import Gemma4ForConditionalGeneration, AutoTokenizer
from peft import PeftModel

from src.rewards.authenticity_reward import AuthenticityReward
from src.rewards.rubric_reward import RubricReward
from src.generation.sampler import (
    ElementCycler,
    build_generation_prompt,
    sample_prompt_spec,
)


class ValueHead(nn.Module):
    """
    Scalar value estimate per token position, the PPO baseline.

    Kept separate from the policy trunk: the LoRA adapter is the only
    trainable part of the policy, and coupling the value function to it
    makes both objectives compete for the same small parameter budget.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size // 4)
        self.out = nn.Linear(hidden_size // 4, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.dense(hidden_states))
        return self.out(x).squeeze(-1)


class CustomPPO:
    """
    PPO for a LoRA policy with a programmatic (non-neural) reward.

    TRL's PPOTrainer cannot be used. Its get_reward() helper does:

        lm_backbone = getattr(model, model.base_model_prefix)
        output = lm_backbone(input_ids=..., output_hidden_states=True)
        reward_logits = model.score(output.hidden_states[-1])

    It hands the reward model token IDs and expects a transformer with
    a .score head, running the forward pass itself. Our reward decodes
    to text, runs a 7B model twice with the adapter toggled, computes a
    bigram ratio, then runs a separate 4B judge and parses a digit.
    That is a Python function, so the rollout loop has to be ours.

    Device placement: policy on cuda:0, authenticity on cuda:1, judge
    on cuda:2. All four on one A100 OOM'd on step 1.

    Known risk: PPO fails silently when subtly wrong. check_divergence
    below tests the standard failure modes each step so a bad run shows
    up within a few hundred steps rather than at the end.
    """

    def __init__(
        self,
        base_model_name: str = "google/gemma-4-E4B-it",
        sft_checkpoint: str = "models/GeneratorSFT",
        output_dir: str = "models/PPOGenerator",
        anchor_df=None,
        gold_df=None,
        alpha: float = 0.5,
        kl_coef: float = 0.1,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        gamma: float = 1.0,
        lam: float = 0.95,
        learning_rate: float = 1.41e-5,
        batch_size: int = 2,
        ppo_epochs: int = 2,
        max_new_tokens: int = 256,
        temperature: float = 0.9,
        seed: int = 42,
        policy_device: str = "cuda:0",
        auth_device: str = "cuda:1",
        rubric_device: str = "cuda:2",
        gradient_checkpointing: bool = True,
    ):
        self.device = policy_device
        self.output_dir = Path(output_dir)
        self.alpha = alpha
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.gamma = gamma
        self.lam = lam
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.anchor_df = anchor_df

        os.makedirs(self.output_dir, exist_ok=True)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        n_gpu = torch.cuda.device_count()
        print(f"Visible GPUs: {n_gpu}")
        if n_gpu < 3:
            print(f"WARNING: only {n_gpu} GPU(s); all models on cuda:0. "
                  f"Expect OOM.")
            policy_device = auth_device = rubric_device = "cuda:0"
            self.device = policy_device

        print(f"Placement: policy={policy_device}  auth={auth_device}  "
              f"rubric={rubric_device}")

        self.tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = Gemma4ForConditionalGeneration.from_pretrained(
            base_model_name, dtype=torch.bfloat16, device_map=policy_device,
        )
        self.policy = PeftModel.from_pretrained(
            base, sft_checkpoint, is_trainable=True
        )

        if gradient_checkpointing:
            self.policy.gradient_checkpointing_enable()
            self.policy.enable_input_require_grads()

        # the KL reference is the same model with the adapter disabled,
        # so no second copy of the weights is needed
        cfg = self.policy.config
        hidden = getattr(
            getattr(cfg, "text_config", cfg), "hidden_size", None
        )
        if hidden is None:
            hidden = cfg.hidden_size
        print(f"Value head hidden size: {hidden}")
        self.value_head = ValueHead(hidden).to(policy_device).to(torch.bfloat16)

        trainable = [p for p in self.policy.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable + list(self.value_head.parameters()),
            lr=learning_rate,
        )
        print(f"Trainable policy params: "
              f"{sum(p.numel() for p in trainable):,}")

        print("Loading reward models...")
        self.auth = AuthenticityReward(device=auth_device)
        few_shot = RubricReward.build_few_shot_examples(gold_df, max_examples=1)
        self.rubric = RubricReward(
            device=rubric_device, few_shot_examples=few_shot,
            batch_size=batch_size,
        )

        self.element_cycler = ElementCycler(
            anchor_df["Element_numberX"].unique().tolist(), self.rng
        )

    @torch.no_grad()
    def _rollout(self, specs: list[dict]):
        prompts = [
            build_generation_prompt(
                s["element"], s["target_score"],
                s["rubric_text"], s["anchor_text"],
            )
            for s in specs
        ]

        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=1024,
        ).to(self.device)

        gen = self.policy.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.95,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )

        seqs = gen.sequences
        prompt_len = enc["input_ids"].shape[1]

        texts = [
            self.tokenizer.decode(s[prompt_len:], skip_special_tokens=True).strip()
            for s in seqs
        ]

        attn = (seqs != self.tokenizer.pad_token_id).long()

        out = self.policy(
            input_ids=seqs, attention_mask=attn, output_hidden_states=True,
        )
        logits = out.logits[:, :-1, :]
        targets = seqs[:, 1:]
        token_logprobs = torch.gather(
            torch.log_softmax(logits.float(), dim=-1),
            2, targets.unsqueeze(-1),
        ).squeeze(-1)

        values = self.value_head(out.hidden_states[-1][:, :-1, :])

        with self.policy.disable_adapter():
            ref_out = self.policy(input_ids=seqs, attention_mask=attn)
            ref_token_logprobs = torch.gather(
                torch.log_softmax(ref_out.logits[:, :-1, :].float(), dim=-1),
                2, targets.unsqueeze(-1),
            ).squeeze(-1)

        # only completion tokens are actions; prompt tokens are not
        mask = torch.zeros_like(token_logprobs)
        mask[:, prompt_len - 1:] = 1.0
        mask = mask * attn[:, 1:].float()

        return {
            "sequences": seqs,
            "attention_mask": attn,
            "texts": texts,
            "logprobs": token_logprobs,
            "ref_logprobs": ref_token_logprobs,
            "values": values,
            "mask": mask,
            "prompt_len": prompt_len,
        }

    def _compute_rewards(self, texts, specs):
        valid = [i for i, t in enumerate(texts) if len(t.split()) >= 10]
        auth = np.zeros(len(texts))
        rub = np.zeros(len(texts))

        if valid:
            vt = [texts[i] for i in valid]
            va = self.auth.score_batch(vt, batch_size=4, normalize=True)
            vr = self.rubric.score(
                vt,
                [specs[i]["element"] for i in valid],
                [specs[i]["target_score"] for i in valid],
            )
            for j, i in enumerate(valid):
                auth[i] = va[j]
                rub[i] = vr[j]

        combined = self.alpha * auth + (1 - self.alpha) * rub
        return combined, auth, rub

    def _gae(self, rewards, values, mask):
        """
        Generalized advantage estimation.

        The reward is one scalar per sequence placed at the final
        completion token; intermediate tokens get zero, so the value
        function carries credit assignment backward.
        """
        adv = torch.zeros_like(values)
        lastgae = torch.zeros(values.shape[0], device=values.device)

        for t in reversed(range(values.shape[1])):
            nextval = values[:, t + 1] if t + 1 < values.shape[1] \
                else torch.zeros_like(values[:, t])
            nextmask = mask[:, t + 1] if t + 1 < mask.shape[1] \
                else torch.zeros_like(mask[:, t])
            delta = rewards[:, t] + self.gamma * nextval * nextmask - values[:, t]
            lastgae = delta + self.gamma * self.lam * nextmask * lastgae
            adv[:, t] = lastgae

        return adv, adv + values

    def step(self, specs: list[dict]) -> dict:
        roll = self._rollout(specs)
        combined, auth, rub = self._compute_rewards(roll["texts"], specs)

        mask = roll["mask"]
        seq_lens = mask.sum(dim=1).long()

        kl = (roll["logprobs"] - roll["ref_logprobs"]) * mask
        kl_per_seq = kl.sum(dim=1)

        reward_t = torch.zeros_like(roll["logprobs"])
        for i in range(len(specs)):
            last = max(int(seq_lens[i].item()) - 1, 0)
            pos = min(roll["prompt_len"] - 1 + last, reward_t.shape[1] - 1)
            reward_t[i, pos] = combined[i]

        reward_t = reward_t - self.kl_coef * kl

        with torch.no_grad():
            adv, returns = self._gae(reward_t, roll["values"].float(), mask)
            n = mask.sum().clamp(min=1)
            mean = (adv * mask).sum() / n
            var = (((adv * mask) - mean) ** 2 * mask).sum() / n
            adv = (adv - mean) / (var.sqrt() + 1e-8)

        old_logprobs = roll["logprobs"].detach()

        stats = {}
        for _ in range(self.ppo_epochs):
            out = self.policy(
                input_ids=roll["sequences"],
                attention_mask=roll["attention_mask"],
                output_hidden_states=True,
            )
            targets = roll["sequences"][:, 1:]
            new_logprobs = torch.gather(
                torch.log_softmax(out.logits[:, :-1, :].float(), dim=-1),
                2, targets.unsqueeze(-1),
            ).squeeze(-1)

            new_values = self.value_head(
                out.hidden_states[-1][:, :-1, :]
            ).float()

            ratio = torch.exp(new_logprobs - old_logprobs)
            pg1 = -adv * ratio
            pg2 = -adv * torch.clamp(
                ratio, 1 - self.clip_range, 1 + self.clip_range
            )
            pg_loss = (torch.max(pg1, pg2) * mask).sum() / mask.sum().clamp(min=1)
            v_loss = (((new_values - returns) ** 2) * mask).sum() \
                / mask.sum().clamp(min=1)

            loss = pg_loss + self.vf_coef * v_loss

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.policy.parameters() if p.requires_grad]
                + list(self.value_head.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()

            clipfrac = ((ratio - 1.0).abs() > self.clip_range).float()
            clipfrac = (clipfrac * mask).sum() / mask.sum().clamp(min=1)

            stats = {
                "pg_loss": float(pg_loss.item()),
                "value_loss": float(v_loss.item()),
                "grad_norm": float(grad_norm),
                "clip_frac": float(clipfrac.item()),
            }

        stats.update({
            "reward_combined": float(np.mean(combined)),
            "reward_auth": float(np.mean(auth)),
            "reward_rubric": float(np.mean(rub)),
            "kl": float(kl_per_seq.mean().item()),
            "mean_gen_words": float(
                np.mean([len(t.split()) for t in roll["texts"]])
            ),
            "distinct_frac": float(
                np.mean([
                    len(set(t.split())) / max(len(t.split()), 1)
                    for t in roll["texts"]
                ])
            ),
        })
        return stats


def check_divergence(history: list[dict], window: int = 50) -> list[str]:
    """
    Flags the standard silent PPO failure modes. PPO can run to
    completion while learning nothing, so these are checked during
    training rather than inspected afterward.
    """
    if len(history) < window * 2:
        return []

    recent = history[-window:]
    prior = history[-2 * window:-window]
    warns = []

    r_now = np.mean([h["reward_combined"] for h in recent])
    r_before = np.mean([h["reward_combined"] for h in prior])
    if abs(r_now - r_before) < 0.005:
        warns.append(
            f"reward flat: {r_before:.4f} -> {r_now:.4f} over {window} steps"
        )

    kl_now = np.mean([h["kl"] for h in recent])
    if kl_now > 50:
        warns.append(f"KL exploding: {kl_now:.1f}; lower lr or raise kl_coef")

    v_now = np.mean([h["value_loss"] for h in recent])
    v_before = np.mean([h["value_loss"] for h in prior])
    if v_now > v_before * 1.5:
        warns.append(f"value loss rising: {v_before:.4f} -> {v_now:.4f}; "
                     f"the baseline is not fitting")

    d_now = np.mean([h["distinct_frac"] for h in recent])
    if d_now < 0.35:
        warns.append(f"generation collapsing: distinct token fraction "
                     f"{d_now:.3f}; likely reward hacking via repetition")

    c_now = np.mean([h["clip_frac"] for h in recent])
    if c_now > 0.5:
        warns.append(f"clip fraction {c_now:.3f}; updates too large")

    w_now = np.mean([h["mean_gen_words"] for h in recent])
    if w_now < 15:
        warns.append(f"generations collapsing to {w_now:.1f} words")

    return warns