#!/usr/bin/env python
"""
Decoding probe for the BoN generator. Rebuilt from the thread handoff spec.

Question it answers: where does the garbled text come from?

  - decoder constraints (repetition_penalty / no_repeat_ngram_size applied
    over the prompt as well as the completion), or
  - a damaged adapter, or
  - the rounds compounding degraded output.

It generates from the same prompts under three settings and prints
sentence-mark rate, run-together-word rate, Title Case share and
ends-on-punctuation, next to the real-corpus reference.

Settings
  current       temperature 0.9, top_p 0.95, repetition_penalty 1.15, no_repeat_ngram_size 4
  no_penalties  same, repetition_penalty 1.0, no n-gram ban
  cooler        temperature 0.7, top_p 0.9, no penalties

Run once per adapter (models/BoN_round3 and models/GeneratorSFT), one A100 each:

  sbatch --partition=gpuq --qos=gpu --time=02:00:00 \
    --export=ALL,SCRIPT=scripts/probe_decoding.py,ARGS="--adapter models/BoN_round3 --pool models/BoN_round3/accepted.jsonl --out results/probe_round3.jsonl" \
    scripts/train.slurm

  sbatch --partition=gpuq --qos=gpu --time=02:00:00 \
    --export=ALL,SCRIPT=scripts/probe_decoding.py,ARGS="--adapter models/GeneratorSFT --pool models/BoN_round3/accepted.jsonl --out results/probe_sft.jsonl" \
    scripts/train.slurm

Use the same --pool and --seed for both so the prompts are identical.

Metrics come from src/utils/text_quality.py (checked by recomputing them on
accepted_bon1-4.jsonl; they land within about 10% of the handoff's pool numbers):
  marks per 100 words      count of '.', '!' and '?' characters in the window
  run-together per 100     matches of [a-z][A-Z][a-z] in the window
  Title Case share         share of whitespace tokens where str.istitle() is True
  ends on punctuation      text ends in . ! or ? optionally followed by quotes/brackets
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.text_quality import text_metrics  # noqa: E402

SETTINGS = {
    "current": dict(
        temperature=0.9, top_p=0.95,
        repetition_penalty=1.15, no_repeat_ngram_size=4,
    ),
    "no_penalties": dict(
        temperature=0.9, top_p=0.95,
        repetition_penalty=1.0, no_repeat_ngram_size=0,
    ),
    "cooler": dict(
        temperature=0.7, top_p=0.9,
        repetition_penalty=1.0, no_repeat_ngram_size=0,
    ),
}

# Real-corpus reference from the handoff.
REAL = {
    "ends_punct": 0.74,
    "first_marks": 6.4, "first_rt": 0.33, "first_tc": 0.16,
    "last_marks": 7.2, "last_rt": 0.46, "last_tc": 0.165,
}

# ── prompt selection ────────────────────────────────────────────────

def pick_prompts(pool_path, max_prompt_words, seed):
    """One prompt per element, prompts under max_prompt_words words."""
    import random
    rows = []
    with open(pool_path) as f:
        for line in f:
            r = json.loads(line)
            if len(r["prompt"].split()) < max_prompt_words:
                rows.append(r)
    rng = random.Random(seed)
    rng.shuffle(rows)
    chosen = {}
    for r in rows:
        chosen.setdefault(r["element"], r)
    return [chosen[e] for e in sorted(chosen)]


# ── generation ──────────────────────────────────────────────────────

def stop_ids(sampler):
    return set(sampler.eos_ids)


def generate(sampler, prompt, n, cfg):
    """
    Mirrors CandidateSampler.sample (same truncation, same pad id, same decode)
    but also returns, per sample, whether a stop token was emitted.
    """
    import torch
    inputs = sampler.tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024,
    ).to(sampler.device)
    prompt_len = inputs["input_ids"].shape[1]
    stops = stop_ids(sampler)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = sampler.model.generate(
            **inputs,
            max_new_tokens=sampler.max_new_tokens,
            do_sample=True,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            repetition_penalty=cfg["repetition_penalty"],
            no_repeat_ngram_size=cfg["no_repeat_ngram_size"],
            num_return_sequences=n,
            eos_token_id=sampler.eos_ids,
            pad_token_id=sampler.pad_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    samples = []
    for seq in out:
        gen = seq[prompt_len:].tolist()
        stopped, length = False, len(gen)
        for i, tok in enumerate(gen):
            if tok in stops:
                stopped, length = True, i
                break
        text = sampler.tokenizer.decode(
            gen[:length], skip_special_tokens=True
        ).strip()
        samples.append(dict(text=text, n_tokens=length, stopped=stopped))
    return samples, elapsed, prompt_len


# ── reporting ───────────────────────────────────────────────────────

def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(records):
    by = {}
    for r in records:
        by.setdefault(r["setting"], []).append(r)

    header = (f"{'setting':<14}{'n':>3}{'words':>7}{'stop':>6}{'endP':>6}"
              f"{'f_mark':>8}{'f_rt':>6}{'f_tc':>6}"
              f"{'l_mark':>8}{'l_rt':>6}{'l_tc':>6}{'s/call':>8}")
    print("\n" + header)
    print("-" * len(header))
    for name, rs in by.items():
        calls = {(r["element"], r["call_seconds"]) for r in rs}
        print(f"{name:<14}{len(rs):>3}"
              f"{mean([r['n_words'] for r in rs]):>7.0f}"
              f"{mean([r['stopped'] for r in rs]):>6.2f}"
              f"{mean([r['ends_punct'] for r in rs]):>6.2f}"
              f"{mean([r['first_marks'] for r in rs]):>8.2f}"
              f"{mean([r['first_rt'] for r in rs]):>6.2f}"
              f"{mean([r['first_tc'] for r in rs]):>6.2f}"
              f"{mean([r['last_marks'] for r in rs]):>8.2f}"
              f"{mean([r['last_rt'] for r in rs]):>6.2f}"
              f"{mean([r['last_tc'] for r in rs]):>6.2f}"
              f"{mean([c[1] for c in calls]):>8.1f}")
    print(f"{'REAL corpus':<14}{'':>3}{'':>7}{'':>6}"
          f"{REAL['ends_punct']:>6.2f}"
          f"{REAL['first_marks']:>8.2f}{REAL['first_rt']:>6.2f}{REAL['first_tc']:>6.2f}"
          f"{REAL['last_marks']:>8.2f}{REAL['last_rt']:>6.2f}{REAL['last_tc']:>6.2f}")
    print("\nstop = share of samples that emitted a stop token before the cap")
    print("endP = share ending on . ! or ?   f_/l_ = first/last 100 words")
    print("mark = sentence-ending characters per 100 words, rt = run-together words per 100, "
          "tc = Title Case share")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--pool", required=True,
                    help="accepted.jsonl to draw prompts from")
    ap.add_argument("--out", required=True, help="JSONL with every sample")
    ap.add_argument("--n", type=int, default=4, help="samples per prompt per setting")
    ap.add_argument("--max_prompt_words", type=int, default=450)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=400,
                    help="400 matches the probe spec for the old adapters; use 512 for the rebuilt generator")
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS),
                    choices=list(SETTINGS))
    ap.add_argument("--show", action="store_true",
                    help="print the last 250 characters of the first sample per call")
    args = ap.parse_args()

    import torch
    from src.generation.sampler import CandidateSampler

    prompts = pick_prompts(args.pool, args.max_prompt_words, args.seed)
    print(f"Prompts: {[(p['element'], p['target_score']) for p in prompts]}", flush=True)

    sampler = CandidateSampler(adapter_path=args.adapter,
                               max_new_tokens=args.max_new_tokens)
    print(f"Stop token ids: {sorted(stop_ids(sampler))}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    records = []
    with open(args.out, "w") as fout:
        for p in prompts:
            for name in args.settings:
                torch.manual_seed(args.seed)
                samples, secs, prompt_len = generate(
                    sampler, p["prompt"], args.n, SETTINGS[name]
                )
                print(f"{p['element']} target={p['target_score']} {name:<13}"
                      f"prompt_tokens={prompt_len} {secs:.1f}s", flush=True)
                for i, s in enumerate(samples):
                    rec = dict(
                        adapter=args.adapter, setting=name,
                        element=p["element"], target_score=p["target_score"],
                        sample=i, call_seconds=secs, prompt_tokens=prompt_len,
                        n_tokens=s["n_tokens"], stopped=s["stopped"],
                        text=s["text"], **text_metrics(s["text"]),
                    )
                    records.append(rec)
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                if args.show and samples:
                    print("   ...", samples[0]["text"][-250:].replace("\n", " "),
                          flush=True)

    print(f"\nAdapter: {args.adapter}")
    summarize(records)


if __name__ == "__main__":
    main()
