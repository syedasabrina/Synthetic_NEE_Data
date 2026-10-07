#!/usr/bin/env python
"""
Fine-tunes Gemma to score a BIP on the rubric.

What the model is taught: given the rubric for one element and a BIP, reply
with 0, 2 or 4. The prompt is the one the judge already uses, with no worked
examples in it, so an untuned Gemma, a Gemma given 21 worked examples, and a
fine-tuned Gemma all see the same kind of question and differ only in how
they were prepared.

Training data (--data):
  synthetic        the best-of-n dataset (accepted.jsonl). The label is the
                   score the generator was asked for, which the judge agreed with.
  judge_real       real BIPs labelled by the judge (scripts/label_real_with_judge.py).
                   Control: real text, the judge's labels.
  supervisor_real  real BIPs with the supervisor scores. The labels this
                   project argues against.

All three are capped to the same number of rows (--max_rows) so that none of
them wins by having more data. BIPs from the held-out "practice" principals
are never trained on. After training, the model is scored on the practice set
(real BIPs from those principals, labelled by the judge) and the result is
written to val_metrics.json. That number is for choosing between training
settings. The gold set is never used to choose anything.

Smoke test (a few minutes):
    python scripts/train_scorer.py --data judge_real --smoke --output models/_smoke_scorer
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

LEVELS = (0, 2, 4)
FAR_LEVEL = {0: 4, 2: 0, 4: 0}


# ── data ────────────────────────────────────────────────────────────

def load_training_rows(args, exclude_person_ids) -> pd.DataFrame:
    """Returns a frame with columns text, element, label (points 0, 2 or 4)."""
    from src.training.assessor import load_condition_data

    if args.data == "synthetic":
        d = pd.read_json(args.synthetic, lines=True)
        df = pd.DataFrame({"text": d["completion"], "element": d["element"],
                           "label": d["target_score"].astype(int)})
    elif args.data == "judge_real":
        if args.match_synthetic:
            t, e, s = load_condition_data(
                "judge_real", args.synthetic, None, args.seed,
                args.judge_labelled, exclude_person_ids=exclude_person_ids)
            df = pd.DataFrame({"text": t, "element": e, "label": s})
        else:
            jl = pd.read_json(args.judge_labelled, lines=True)
            jl = jl[jl["judge_pred"].isin(LEVELS)]
            jl = jl[~jl["PersonId"].astype(str).isin({str(p) for p in exclude_person_ids})]
            df = pd.DataFrame({"text": jl["text"], "element": jl["element"],
                               "label": jl["judge_pred"].astype(int)})
    elif args.data == "supervisor_real":
        from src.data.corpus import for_anchor_pool, load
        a = for_anchor_pool(load(args.raw))
        a = a[~a["PersonId"].astype(str).isin({str(p) for p in exclude_person_ids})]
        df = pd.DataFrame({"text": a["Text"], "element": a["Element_numberX"],
                           "label": a["score"].astype(int)})
    else:
        raise ValueError(args.data)

    df = df.reset_index(drop=True)
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
    return df


# ── prompts and encoding ────────────────────────────────────────────

def make_prompt_builder(tokenizer):
    """The judge's zero-shot prompt, rendered through the chat template."""
    from src.rewards.rubric_reward import RubricReward

    j = object.__new__(RubricReward)
    j.few_shot_examples = {}
    j.max_example_chars = 900
    j.use_chat_template = True
    j.tokenizer = tokenizer
    j.chat_template_kwargs = {}
    return lambda element, text: j._format(j._build_prompt(element, text))


def completion_template(tokenizer) -> str:
    """
    What the chat template appends after the assistant's answer (the end-of-turn
    marker), read from the template itself so nothing is hard-coded. Returns a
    string that starts with the placeholder answer "2".
    """
    user = [{"role": "user", "content": "x"}]
    p = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    f = tokenizer.apply_chat_template(
        user + [{"role": "assistant", "content": "2"}], tokenize=False)
    if not f.startswith(p):
        raise RuntimeError("The chat template does not extend the generation prompt with "
                           "the answer, so the training labels cannot be built safely.")
    tail = f[len(p):]
    if not tail.startswith("2"):
        raise RuntimeError(f"Unexpected assistant rendering: {tail!r}")
    return tail


def encode_example(tokenizer, prompt, completion, max_length):
    """Prompt tokens are masked; the answer and end-of-turn marker are trained on."""
    bos = getattr(tokenizer, "bos_token", None)
    add_special = not (bos and prompt.startswith(bos))
    p = tokenizer(prompt, add_special_tokens=add_special)["input_ids"]
    c = tokenizer(completion, add_special_tokens=False)["input_ids"]
    n = len(p) + len(c)
    if n > max_length:
        return None
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    k = max_length - n
    return {"input_ids": p + c + [pad] * k,
            "attention_mask": [1] * n + [0] * k,
            "labels": [-100] * len(p) + c + [-100] * k}


def build_dataset(tokenizer, df, max_length):
    from datasets import Dataset

    build = make_prompt_builder(tokenizer)
    tail = completion_template(tokenizer)
    cols = {"input_ids": [], "attention_mask": [], "labels": []}
    dropped = 0
    first = None
    for text, element, label in zip(df["text"], df["element"], df["label"]):
        prompt = build(element, text)
        completion = str(int(label)) + tail[1:]
        if first is None:
            first = (prompt, completion)
        enc = encode_example(tokenizer, prompt, completion, max_length)
        if enc is None:
            dropped += 1
            continue
        for k in cols:
            cols[k].append(enc[k])
    print(f"Training examples: {len(cols['input_ids'])}  dropped for length: {dropped}")
    if not cols["input_ids"]:
        raise RuntimeError("No training example fits in max_length.")
    print("\n--- check: end of the first training prompt, then the trained answer ---")
    print(first[0][-260:])
    print("ANSWER:", repr(first[1]))
    print("---\n")
    return Dataset.from_dict(cols)


# ── practice-set scoring ────────────────────────────────────────────

def score_texts(model, tokenizer, builder, texts, elements, batch_size=8, max_length=1536):
    """Greedy answers parsed to 0/2/4 (None if unparseable); skips over-long prompts."""
    import torch
    from src.rewards.rubric_reward import RubricReward

    tokenizer.padding_side = "left"
    model.eval()
    device = next(model.parameters()).device
    bos = getattr(tokenizer, "bos_token", None)
    preds, keep = [], []
    prompts = [builder(e, t) for e, t in zip(elements, texts)]
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=False,
                        add_special_tokens=not (bos and batch[0].startswith(bos))).to(device)
        if enc["input_ids"].shape[1] > max_length:
            # score the short prompts of the batch one by one instead of cutting any off
            singles = []
            for i, p in enumerate(batch):
                one = tokenizer([p], return_tensors="pt", truncation=False,
                                add_special_tokens=not (bos and p.startswith(bos))).to(device)
                if one["input_ids"].shape[1] <= max_length:
                    singles.append((start + i, one))
            for idx, one in singles:
                with torch.no_grad():
                    out = model.generate(**one, max_new_tokens=6, do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id)
                txt = tokenizer.decode(out[0][one["input_ids"].shape[1]:], skip_special_tokens=True)
                preds.append(RubricReward._parse_score(txt.strip())); keep.append(idx)
            continue
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=6, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        for i, seq in enumerate(out):
            txt = tokenizer.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            preds.append(RubricReward._parse_score(txt.strip())); keep.append(start + i)
    return preds, keep


def practice_metrics(preds, truth) -> dict:
    from src.evaluation.gold_metrics import qwk

    truth = np.asarray(truth)
    parsed = np.array([p is not None for p in preds])
    filled = np.array([p if p is not None else FAR_LEVEL[int(t)] for p, t in zip(preds, truth)])
    return {"n": int(len(truth)), "parsed": int(parsed.sum()),
            "exact": float((filled == truth).mean()), "qwk": qwk(truth, filled)}


# ── main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=["synthetic", "judge_real", "supervisor_real"])
    ap.add_argument("--synthetic", default="models/BoN_v2_full/accepted.jsonl")
    ap.add_argument("--judge_labelled", default="data/derived/judge_labelled_real.jsonl")
    ap.add_argument("--raw", default="data/raw/bips.csv")
    ap.add_argument("--match_synthetic", action="store_true",
                    help="judge_real only: copy the synthetic set's (element, score) counts")
    ap.add_argument("--max_rows", type=int, default=3000, help="0 = no cap")
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=1536)
    ap.add_argument("--practice_fraction", type=float, default=0.1,
                    help="share of principals held out as the practice set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the plumbing")
    args = ap.parse_args()

    if args.smoke:
        args.max_rows, args.epochs = 48, 1

    from src.training.assessor import real_validation_split
    from src.utils.config import GeneratorSFTConfig
    from src.training.generator_sft import setup_generator_model_and_tokenizer
    from transformers import Trainer, TrainingArguments, default_data_collator

    start = time.time()
    p_texts, p_elements, p_idx, practice_ids = real_validation_split(
        args.judge_labelled, args.practice_fraction)
    p_truth = [LEVELS[i] for i in p_idx]
    if args.smoke:
        p_texts, p_elements, p_truth = p_texts[:24], p_elements[:24], p_truth[:24]
    print(f"Practice set: {len(p_texts)} real BIPs from {len(practice_ids)} held-out principals")

    df = load_training_rows(args, practice_ids)
    print(f"Training rows: {len(df)}   label counts:\n{df['label'].value_counts().sort_index().to_string()}")

    config = GeneratorSFTConfig(
        output_dir=args.output, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate, max_seq_length=args.max_length, seed=args.seed,
    )
    model, tokenizer = setup_generator_model_and_tokenizer(config)
    dataset = build_dataset(tokenizer, df, args.max_length)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output) / "run_args.json", "w") as f:
        json.dump({**vars(args), "n_train": len(dataset), "n_practice": len(p_texts)}, f, indent=2)

    eff = args.batch_size * args.grad_accum
    total = max(1, len(dataset) // eff) * args.epochs
    targs = TrainingArguments(
        output_dir=args.output, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate, warmup_steps=int(total * 0.05),
        max_grad_norm=1.0, bf16=True, fp16=False, logging_steps=10,
        save_strategy="no", report_to="wandb",
        run_name=f"Scorer-{args.data}-lr{args.learning_rate}", seed=args.seed,
        dataloader_num_workers=2, remove_unused_columns=False,
    )
    Trainer(model=model, args=targs, train_dataset=dataset,
            data_collator=default_data_collator).train()

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved scorer adapter to {args.output}")

    try:
        builder = make_prompt_builder(tokenizer)
        preds, keep = score_texts(model, tokenizer, builder, p_texts, p_elements)
        m = practice_metrics(preds, [p_truth[i] for i in keep])
        m["skipped_too_long"] = len(p_texts) - len(keep)
        with open(Path(args.output) / "val_metrics.json", "w") as f:
            json.dump(m, f, indent=2)
        print(f"Practice set (n={m['n']}, parsed {m['parsed']}): exact {m['exact']:.3f}  QWK {m['qwk']:.3f}")
    except Exception as e:  # the adapter is already saved
        print(f"WARNING: practice-set scoring failed ({e!r}); the adapter is saved.")

    print(f"\nDone in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
