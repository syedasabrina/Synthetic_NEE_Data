from __future__ import annotations

import gc
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoTokenizer, Gemma4ForConditionalGeneration,
    Trainer, TrainingArguments, default_data_collator,
)

from src.data.encoding import encode_many
from src.generation.sampler import (
    CandidateSampler,
    build_generation_prompt,
    element_for_index,
    sample_prompt_spec,
)
from src.rewards.authenticity_reward import AuthenticityReward
from src.rewards.rubric_reward import RubricReward
from src.utils.text_quality import QualityGate, text_metrics


# ── pure helpers (no models, unit-testable) ──────────────────────────

def repetition(text: str) -> float:
    """
    Distinct-bigram ratio in [0, 1]. Only useful for catching loops: word
    salad repeats no bigrams and scores ~1.0, so this is NOT a quality
    measure. Quality is QualityGate.
    """
    w = text.split()
    if len(w) < 2:
        return 0.0
    b = list(zip(w, w[1:]))
    return len(set(b)) / len(b)


def anchor_overlap(candidate: str, anchor: str) -> float:
    """Word-level Jaccard overlap, a guard against copying the anchor."""
    a = set(candidate.lower().split())
    b = set(anchor.lower().split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class SelectionConfig:
    """
    How candidates are chosen once they have passed the gates.

    min_judge_reward: hard gate on judge agreement. 1.0 requires the blind
    judge to name the target score exactly. 0.5 also admits one level off.
    None disables the gate (the old behaviour, under which 7.4% of
    accepted rows in rounds 2 to 4 had judge reward 0).

    min_auth_delta: absolute floor on authenticity delta, taken from
    held-out real BIPs (scripts/score_real_reference.py). None disables it.
    Without a floor, the batch z-score means the best of 8 poor candidates
    still wins.

    rank_by: "auth" ranks the survivors by authenticity, "combined" by
    alpha * auth + (1 - alpha) * judge reward.
    """
    keep_top_k: int = 2
    min_judge_reward: float | None = 1.0
    min_auth_delta: float | None = None
    rank_by: str = "auth"
    alpha: float = 0.5


def gate_reasons(cand: dict, anchor_text: str, gate: QualityGate,
                 min_repetition: float, leak_threshold: float) -> list[str]:
    reasons = gate.check(cand["text"], stopped=cand.get("stopped"))
    if "short" in reasons:
        return reasons
    if repetition(cand["text"]) < min_repetition:
        reasons.append("loop")
    if anchor_overlap(cand["text"], anchor_text) > leak_threshold:
        reasons.append("anchor_leak")
    if cand.get("auth_floor_ok") is False:
        reasons.append("nll_ceiling")
    return reasons


def select_candidates(cands: list[dict], sel: SelectionConfig) -> list[int]:
    """Indices of the candidates to keep, best first."""
    eligible = []
    for j, c in enumerate(cands):
        if c["gate"] or not c.get("scored"):
            continue
        if (sel.min_judge_reward is not None
                and c["rubric_reward"] < sel.min_judge_reward):
            continue
        if (sel.min_auth_delta is not None
                and c["auth_delta"] < sel.min_auth_delta):
            continue
        eligible.append(j)

    if sel.rank_by == "combined":
        def key(j):
            c = cands[j]
            return sel.alpha * c["auth_rel"] + (1 - sel.alpha) * c["rubric_reward"]
    else:
        def key(j):
            return cands[j]["auth_rel"]

    eligible.sort(key=key, reverse=True)
    return eligible[:sel.keep_top_k]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def read_records(path: Path) -> list[dict]:
    """Reads candidates.jsonl, skipping a partial last line from a killed job."""
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def finalize_round(
    output_dir: str | Path,
    n_prompts: int,
    gate: QualityGate,
    selection: SelectionConfig,
    min_repetition: float = 0.60,
    leak_threshold: float = 0.85,
) -> pd.DataFrame:
    """
    Builds accepted.jsonl and round_stats.json from candidates.jsonl.

    Gate reasons and selection are recomputed from the stored candidate
    text and scores, so the same call re-selects with new thresholds and
    needs no GPU (run_bon_round.py --reselect).
    """
    output_dir = Path(output_dir)
    records = [r for r in read_records(output_dir / "candidates.jsonl")
               if r["i"] < n_prompts]

    rows = []
    gate_counts = Counter()
    n_cands = n_scored = 0
    pool_match, pool_delta, pool_auth_abs = [], [], []
    sel_match, sel_delta = [], []
    zero_accept = 0

    for rec in records:
        cands = rec["candidates"]
        for c in cands:
            n_cands += 1
            c["gate"] = gate_reasons(c, rec["anchor_text"], gate,
                                     min_repetition, leak_threshold)
            for r in c["gate"]:
                gate_counts[r] += 1
            c["selected"] = False
            if c.get("scored"):
                n_scored += 1
                if not c["gate"]:
                    pool_match.append(float(c["rubric_reward"] == 1.0))
                    pool_delta.append(c["auth_delta"])
                    pool_auth_abs.append(c["auth_abs"])

        chosen = select_candidates(cands, selection)
        if not chosen:
            zero_accept += 1
        for j in chosen:
            c = cands[j]
            c["selected"] = True
            sel_match.append(float(c["rubric_reward"] == 1.0))
            sel_delta.append(c["auth_delta"])
            rows.append({
                "prompt_idx": rec["i"],
                "element": rec["element"],
                "target_score": rec["target_score"],
                "anchor_score": rec.get("anchor_score"),
                "anchor_fallback": rec.get("anchor_fallback"),
                "rubric_text": rec["rubric_text"],
                "anchor_text": rec["anchor_text"],
                "prompt": rec["prompt"],
                "completion": c["text"],
                "judge_pred": c["judge_pred"],
                "auth_delta": c["auth_delta"],
                "auth_reward": c["auth_rel"],
                "auth_reward_abs": c["auth_abs"],
                "rubric_reward": c["rubric_reward"],
                "combined_reward": (selection.alpha * c["auth_rel"]
                                    + (1 - selection.alpha) * c["rubric_reward"]),
                "stopped": c["stopped"],
                "repetition": repetition(c["text"]),
                "n_words": c["n_words"],
            })

    df = pd.DataFrame(rows)

    stats = {
        "n_prompts_requested": n_prompts,
        "n_prompts_done": len(records),
        "n_accepted": len(df),
        "prompts_with_zero_accepted": zero_accept,
        "n_candidates": n_cands,
        "n_candidates_scored": n_scored,
        "gate_fail_counts": dict(gate_counts),
        "selection": asdict(selection),
        "gate": asdict(gate),
        # Selection lift: the gated pool is the "random pick" baseline,
        # the accepted rows are the "best-of-n pick".
        "gated_pool_n": len(pool_match),
        "gated_pool_judge_match": float(np.mean(pool_match)) if pool_match else None,
        "gated_pool_mean_auth_delta": float(np.mean(pool_delta)) if pool_delta else None,
        "gated_pool_mean_auth_abs": float(np.mean(pool_auth_abs)) if pool_auth_abs else None,
        "accepted_judge_match": float(np.mean(sel_match)) if sel_match else None,
        "accepted_mean_auth_delta": float(np.mean(sel_delta)) if sel_delta else None,
    }
    if len(df):
        m = [text_metrics(t) for t in df["completion"]]
        stats.update({
            "accepted_mean_words": float(df["n_words"].mean()),
            "accepted_ends_punct": float(np.mean([x["ends_punct"] for x in m])),
            "accepted_mean_run_together_per_100": float(np.mean([x["rt"] for x in m])),
            "accepted_mean_title_case": float(np.mean([x["tc"] for x in m])),
            "accepted_mean_marks_per_100": float(np.mean([x["marks"] for x in m])),
            "accepted_by_target": {str(k): int(v) for k, v in
                                   df["target_score"].value_counts().items()},
            "accepted_by_element": {str(k): int(v) for k, v in
                                    df["element"].value_counts().items()},
            "accepted_by_element_target": {
                f"{e}|{t}": int(v) for (e, t), v in
                df.groupby(["element", "target_score"]).size().items()
            },
        })

    with open(output_dir / "round_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=_json_default)
    df.to_json(output_dir / "accepted.jsonl", orient="records", lines=True)

    print("\nRound summary")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return df


# ── the round ────────────────────────────────────────────────────────

class BestOfNRound:
    """
    One round of best-of-n generation, scoring and selection.

    For each prompt: sample N candidates, run the quality gate, score every
    candidate with the authenticity model and the blind judge, then keep
    the top k among candidates that pass the gate AND (by default) whose
    blind judge prediction equals the target score.

    Everything is written to candidates.jsonl as it goes: all N candidates
    per prompt with gate reasons, judge prediction and raw output, and the
    authenticity components. A killed job resumes from the last complete
    prompt. Prompt specs are a pure function of (seed, prompt index), so a
    resumed run is identical to an uninterrupted one.

    Device placement: generator, authenticity model and judge are three
    large models. Holding all three plus optimizer state on one A100
    caused OOM at retrain, so release() frees them before retraining.
    """

    def __init__(
        self,
        anchor_df,
        gold_df,
        generator_adapter: str,
        output_dir: str,
        n_candidates: int = 8,
        selection: SelectionConfig | None = None,
        gate: QualityGate | None = None,
        min_repetition: float = 0.60,
        anchor_similarity_threshold: float = 0.85,
        score_gated_out: bool = True,
        elements: list[str] | None = None,
        score_weights: dict[int, float] | None = None,
        max_prompt_tokens: int = 640,
        sampler_kwargs: dict | None = None,
        seed: int = 42,
        generator_device: str = "cuda:0",
        auth_device: str = "cuda:1",
        rubric_device: str = "cuda:2",
    ):
        self.output_dir = Path(output_dir)
        self.n_candidates = n_candidates
        self.selection = selection or SelectionConfig()
        self.gate = gate or QualityGate()
        self.min_repetition = min_repetition
        self.leak_threshold = anchor_similarity_threshold
        self.score_gated_out = score_gated_out
        self.score_weights = score_weights
        self.max_prompt_tokens = max_prompt_tokens
        self.seed = seed

        if elements:
            anchor_df = anchor_df[anchor_df["Element_numberX"].isin(elements)]
        self.anchor_df = anchor_df
        self.elements = sorted(anchor_df["Element_numberX"].unique().tolist())

        os.makedirs(self.output_dir, exist_ok=True)
        self._check_run_config(generator_adapter, sampler_kwargs or {})

        n_gpu = torch.cuda.device_count()
        print(f"Visible GPUs: {n_gpu}")
        if n_gpu < 3:
            print(f"WARNING: only {n_gpu} GPU(s); placing all models on "
                  f"cuda:0. Run generation with --skip_retrain and "
                  f"retrain separately with --retrain_only.")
            generator_device = auth_device = rubric_device = "cuda:0"
        print(f"Placement: generator={generator_device}  "
              f"auth={auth_device}  rubric={rubric_device}")

        self.auth = AuthenticityReward(device=auth_device)

        few_shot = RubricReward.build_few_shot_examples(gold_df, max_examples=1)
        self.rubric = RubricReward(
            device=rubric_device,
            few_shot_examples=few_shot,
            batch_size=n_candidates,
        )
        self.sampler = CandidateSampler(
            adapter_path=generator_adapter,
            device=generator_device,
            **(sampler_kwargs or {}),
        )

    # -- run bookkeeping ------------------------------------------------

    def _check_run_config(self, generator_adapter: str, sampler_kwargs: dict):
        """
        Resuming with different settings would mix incompatible candidates
        in one file. Refuse unless the settings match the first run.
        """
        cfg = {
            "generator_adapter": generator_adapter,
            "seed": self.seed,
            "n_candidates": self.n_candidates,
            "elements": self.elements,
            "score_weights": self.score_weights,
            "max_prompt_tokens": self.max_prompt_tokens,
            "sampler_kwargs": sampler_kwargs,
        }
        path = self.output_dir / "run_config.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old != json.loads(json.dumps(cfg, default=_json_default)):
                raise RuntimeError(
                    f"{path} differs from the current settings. Use a new "
                    f"--output directory, or delete candidates.jsonl and "
                    f"run_config.json to start over.\nold: {old}\nnew: {cfg}"
                )
        else:
            path.write_text(json.dumps(cfg, indent=2, default=_json_default))

    def _load_done(self, path: Path) -> set[int]:
        records = read_records(path)
        # rewrite without any partial trailing line so appends stay valid
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r, default=_json_default) + "\n")
        return {r["i"] for r in records}

    def release(self):
        """Frees the generator and both reward models before retraining."""
        for attr in ("sampler", "auth", "rubric"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        torch.cuda.empty_cache()
        print("Released generation and reward models.")

    # -- one prompt -----------------------------------------------------

    def _process_prompt(self, i: int) -> dict:
        rng = np.random.default_rng([self.seed, i])
        element = element_for_index(self.elements, self.seed, i)
        spec = sample_prompt_spec(
            self.anchor_df, RubricReward, element, rng, self.score_weights
        )
        target = spec["target_score"]

        prompt = build_generation_prompt(
            element, target, spec["rubric_text"], spec["anchor_text"],
            tokenizer=self.sampler.tokenizer,
            max_prompt_tokens=self.max_prompt_tokens,
        )
        samples = self.sampler.sample_detailed(prompt, n=self.n_candidates)

        cands = []
        for s in samples:
            c = {
                "text": s["text"],
                "n_words": len(s["text"].split()),
                "n_tokens": s["n_tokens"],
                "stopped": s["stopped"],
                "scored": False,
                "selected": False,
            }
            c["gate"] = gate_reasons(c, spec["anchor_text"], self.gate,
                                     self.min_repetition, self.leak_threshold)
            cands.append(c)

        to_score = [
            j for j, c in enumerate(cands)
            if "short" not in c["gate"]
            and (self.score_gated_out or not c["gate"])
        ]
        if to_score:
            texts = [cands[j]["text"] for j in to_score]
            els = [element] * len(texts)
            a = self.auth.score_candidates(texts, elements=els, batch_size=4)
            preds, raws = self.rubric.predict(texts, els, return_raw=True)
            for k, j in enumerate(to_score):
                c = cands[j]
                c.update({
                    "scored": True,
                    "judge_pred": preds[k],
                    "judge_raw": raws[k],
                    "rubric_reward": self.rubric._compute_reward(preds[k], target),
                    "auth_delta": a["delta"][k],
                    "auth_rel": a["reward_rel"][k],
                    "auth_abs": a["reward_abs"][k],
                    "nll_ft": a["nll_finetuned"][k],
                    "auth_floor_ok": bool(a["floor"][k]),
                })
                c["gate"] = gate_reasons(c, spec["anchor_text"], self.gate,
                                         self.min_repetition, self.leak_threshold)

        for j in select_candidates(cands, self.selection):
            cands[j]["selected"] = True

        return {
            "i": i,
            "element": element,
            "target_score": target,
            "anchor_score": spec["anchor_score"],
            "anchor_fallback": spec["anchor_fallback"],
            "anchor_person": spec["anchor_person"],
            "rubric_text": spec["rubric_text"],
            "anchor_text": spec["anchor_text"],
            "prompt": prompt,
            "candidates": cands,
        }

    # -- main loop ------------------------------------------------------

    def run(self, n_prompts: int = 500, log_every: int = 25) -> pd.DataFrame:
        cand_path = self.output_dir / "candidates.jsonl"
        done = self._load_done(cand_path)
        if done:
            print(f"Resuming: {len(done)} prompts already in {cand_path}")

        n_new = n_acc = 0
        with open(cand_path, "a") as f:
            for i in range(n_prompts):
                if i in done:
                    continue
                rec = self._process_prompt(i)
                f.write(json.dumps(rec, default=_json_default) + "\n")
                f.flush()
                n_new += 1
                n_acc += sum(c["selected"] for c in rec["candidates"])
                if n_new % log_every == 0:
                    print(f"prompt {i + 1}/{n_prompts}  new={n_new}  "
                          f"accepted_this_session={n_acc}", flush=True)

        return finalize_round(
            self.output_dir, n_prompts, self.gate, self.selection,
            self.min_repetition, self.leak_threshold,
        )


# ── retraining ───────────────────────────────────────────────────────

def build_retrain_dataset(
    accepted_df: pd.DataFrame,
    tokenizer,
    max_length: int = 1280,
) -> Dataset:
    """
    Tokenizes accepted candidates for the next generator fine-tune. Prompt
    tokens are masked; EOS is appended to every completion and included in
    the loss (see src/data/encoding.py). Rows that do not fit are dropped.
    """
    cols, dropped = encode_many(
        tokenizer,
        accepted_df["prompt"].tolist(),
        accepted_df["completion"].tolist(),
        max_length=max_length,
    )
    print(f"Retrain dataset: {len(cols['input_ids'])} examples, "
          f"{dropped} dropped for length")
    if not cols["input_ids"]:
        raise ValueError("No retrain examples fit in max_length.")
    return Dataset.from_dict(cols)


def retrain_generator(
    accepted_df: pd.DataFrame,
    base_model_name: str,
    prev_adapter_path: str,
    output_dir: str,
    num_train_epochs: int = 2,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.05,
    max_seq_length: int = 1280,
    seed: int = 42,
) -> None:
    """
    Continues LoRA training on the previous adapter with this round's
    accepted candidates.

    Caution: each round that retrains on the previous round's output
    compounds any defect in that output. In rounds 1 to 4 the run-together
    rate in the last 100 words went 1.9, 5.0, 8.5, 10.2. With the gates on
    this is much safer, but a single selection round from a clean SFT
    checkpoint avoids the issue entirely.
    """
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(prev_adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = Gemma4ForConditionalGeneration.from_pretrained(
        base_model_name, dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, prev_adapter_path, is_trainable=True)
    model.print_trainable_parameters()

    dataset = build_retrain_dataset(accepted_df, tokenizer, max_length=max_seq_length)

    effective_batch = per_device_train_batch_size * gradient_accumulation_steps
    steps_per_epoch = max(1, len(dataset) // effective_batch)
    total_steps = steps_per_epoch * num_train_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    print(f"Retrain total steps: {total_steps}, warmup: {warmup_steps}")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="wandb",
        run_name=f"BoNRetrain-{Path(output_dir).name}",
        seed=seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=dataset,
        data_collator=default_data_collator,
    )
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Retrained generator saved to {output_dir}")
