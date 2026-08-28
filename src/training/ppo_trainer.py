from __future__ import annotations

import os
import json
from pathlib import Path

import torch
import numpy as np
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import Gemma4ForConditionalGeneration, AutoTokenizer

from trl.experimental.ppo import PPOTrainer, AutoModelForCausalLMWithValueHead
from trl import PPOConfig as TRLPPOConfig

from src.utils.config import PPOConfig
from src.rewards.authenticity_reward import AuthenticityReward
from src.rewards.rubric_reward import RubricReward
from src.rewards.combined_reward import CombinedReward


def build_ppo_prompt(element: str, target_score: int, rubric_text: str, anchor_text: str) -> str:
    """
    Builds the PPO training prompt. Unlike the SFT warmup, this
    includes the target score explicitly -- this is where score
    conditioning is introduced, reinforced entirely through the
    reward signal rather than through supervised labels.
    """
    return f"""You are a school principal writing a Building Improvement Plan.

Element: {element}
Target score: {target_score}
Rubric criteria for this score: {rubric_text}

Reference example on a similar topic:
{anchor_text}

Generate a BIP response for this element that earns a score of
{target_score} according to the rubric criteria above. Write in your
own words, addressing a similar theme to the reference:
"""


def sample_ppo_batch(
    anchor_df,
    rubric_class,
    batch_size: int,
    score_weights: dict[int, float] | None = None,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """
    Samples a batch of (element, target_score, anchor) tuples for one
    PPO step. score_weights controls how often each score level is
    sampled -- default oversamples 2 and 4 since 0 anchors are scarce.
    """
    rng = rng or np.random.default_rng()
    score_weights = score_weights or {0: 0.1, 2: 0.35, 4: 0.55}

    scores = list(score_weights.keys())
    weights = list(score_weights.values())

    batch = []
    for _ in range(batch_size):
        target_score = rng.choice(scores, p=weights)
        # filter anchor pool to rows that have this score, any element
        pool = anchor_df[anchor_df["score"] == target_score]
        if len(pool) == 0:
            # fallback: use score 4 anchors if the sampled score has none
            pool = anchor_df[anchor_df["score"] == 4]
        row = pool.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        element = row["Element_numberX"]
        anchor_text = row["Text"]
        rubric_text = rubric_class.RUBRIC[element][target_score]
        batch.append({
            "element": element,
            "target_score": int(target_score),
            "anchor_text": anchor_text,
            "rubric_text": rubric_text,
        })
    return batch


class PPOPipeline:
    """
    Wraps TRL's PPOTrainer with the dual reward model setup described
    in the project scope. Handles prompt construction, reward
    computation, and diagnostic logging every step.
    """

    def __init__(
        self,
        config: PPOConfig,
        anchor_df,
        device: str = "cuda",
    ):
        self.config = config
        self.anchor_df = anchor_df
        self.device = device

        print(f"Loading generator from SFT checkpoint: {config.sft_checkpoint}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.sft_checkpoint)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # base model with value head required by PPOTrainer
        base_model = Gemma4ForConditionalGeneration.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=config.lora.target_modules,
            bias=config.lora.bias,
            lora_dropout=config.lora.lora_dropout,
        )

        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            base_model,
            peft_config=lora_config,
        )

        # load the SFT-trained LoRA weights as the starting point
        # this is critical -- PPO on a cold model is unstable
        self.model.pretrained_model.load_adapter(
            config.sft_checkpoint, adapter_name="default"
        )

        trl_ppo_config = TRLPPOConfig(
            model_name=config.model_name,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            mini_batch_size=config.mini_batch_size,
            ppo_epochs=config.ppo_epochs,
            seed=config.seed,
            log_with="wandb",
        )

        self.trainer = PPOTrainer(
            config=trl_ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
        )

        print("Loading reward models...")
        self.auth_reward = AuthenticityReward(device=device)
        self.rubric_reward = RubricReward(device=device)
        self.combined_reward = CombinedReward(
            authenticity_reward=self.auth_reward,
            rubric_reward=self.rubric_reward,
            alpha=config.alpha,
            beta=config.beta,
        )

        self.rng = np.random.default_rng(config.seed)
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

    def run(self, max_steps: int | None = None):
        """
        Main PPO training loop. Logs combined reward, authenticity
        reward, rubric reward, and KL divergence every step so
        instability is visible immediately rather than discovered
        after a long run completes.
        """
        max_steps = max_steps or self.config.max_steps
        diagnostics = []

        for step in range(max_steps):
            batch = sample_ppo_batch(
                self.anchor_df,
                self.rubric_reward,
                batch_size=self.config.batch_size,
                rng=self.rng,
            )

            prompts = [
                build_ppo_prompt(
                    b["element"], b["target_score"],
                    b["rubric_text"], b["anchor_text"],
                )
                for b in batch
            ]

            query_tensors = [
                self.tokenizer(p, return_tensors="pt", truncation=True,
                                max_length=768).input_ids[0].to(self.device)
                for p in prompts
            ]

            response_tensors = self.trainer.generate(
                query_tensors,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.8,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            responses = [
                self.tokenizer.decode(r, skip_special_tokens=True)
                for r in response_tensors
            ]

            elements = [b["element"] for b in batch]
            target_scores = [b["target_score"] for b in batch]

            reward_outputs = self.combined_reward.score(
                responses, elements, target_scores,
            )
            rewards = [torch.tensor(o.final) for o in reward_outputs]

            stats = self.trainer.step(query_tensors, response_tensors, rewards)

            # log diagnostics every step -- cheap and catches
            # instability early rather than after hours of wasted compute
            mean_auth = float(np.mean([o.authenticity for o in reward_outputs]))
            mean_rubric = float(np.mean([o.rubric for o in reward_outputs]))
            mean_combined = float(np.mean([o.combined for o in reward_outputs]))
            kl = stats.get("objective/kl", None)

            diagnostics.append({
                "step": step,
                "mean_authenticity": mean_auth,
                "mean_rubric": mean_rubric,
                "mean_combined": mean_combined,
                "kl": kl,
            })

            if step % 10 == 0:
                print(f"step={step} auth={mean_auth:.4f} "
                      f"rubric={mean_rubric:.4f} combined={mean_combined:.4f} "
                      f"kl={kl}")

            if step % self.config.save_every == 0 and step > 0:
                ckpt_dir = f"{self.config.output_dir}/checkpoint-{step}"
                self.model.save_pretrained(ckpt_dir)
                self._save_diagnostics(diagnostics)

        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        self._save_diagnostics(diagnostics)
        print(f"PPO training complete. Saved to {self.config.output_dir}")

    def _save_diagnostics(self, diagnostics: list[dict]):
        path = Path(self.config.log_dir) / "ppo_diagnostics.json"
        with open(path, "w") as f:
            json.dump(diagnostics, f, indent=2)