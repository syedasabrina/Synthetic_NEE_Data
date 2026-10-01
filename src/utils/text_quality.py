"""
src/utils/text_quality.py

Text-quality metrics and the hard gate used to keep degraded text out of
the accepted pool.

Why this exists: the distinct-bigram ratio averaged 0.9993 on the accepted
rows of rounds 1 to 4 because word salad repeats no bigrams. The metrics
here are the ones that separate the pools from real BIPs: sentence-ending
marks per 100 words, run-together words per 100 words and Title Case
share, measured on the whole text and on the last 100 words.

The defaults (run-together 7, Title Case 0.70, marks 1.0) come from
scripts/sweep_gate.py on the 11,953 real BIPs: 3.9% of real BIPs rejected
(Element 5: 6.4%), 20% of the old degraded pool still passing. Title Case
is a weak criterion for Element 5, where citations and proper nouns put
24% of real BIPs above 0.40. This gate only removes gross outliers; the stop
test, the judge and the authenticity model do the fine selection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_RT = re.compile(r"[a-z][A-Z][a-z]")
_END = re.compile(r"[.!?][\"')\]]*\s*$")


def window_metrics(words: list[str]) -> dict:
    """Marks per 100 words, run-together per 100 words, Title Case share."""
    n = len(words)
    if n == 0:
        nan = float("nan")
        return {"marks": nan, "rt": nan, "tc": nan}
    marks = sum(w.count(".") + w.count("!") + w.count("?") for w in words)
    rt = len(_RT.findall(" ".join(words)))
    tc = sum(1 for w in words if w.istitle())
    return {"marks": marks * 100 / n, "rt": rt * 100 / n, "tc": tc / n}


def text_metrics(text: str) -> dict:
    words = text.split()
    whole = window_metrics(words)
    first = window_metrics(words[:100])
    last = window_metrics(words[-100:])
    return {
        "n_words": len(words),
        "ends_punct": bool(_END.search(text.strip())),
        "marks": whole["marks"], "rt": whole["rt"], "tc": whole["tc"],
        "first_marks": first["marks"], "first_rt": first["rt"], "first_tc": first["tc"],
        "last_marks": last["marks"], "last_rt": last["rt"], "last_tc": last["tc"],
    }


@dataclass
class QualityGate:
    """
    Returns the list of reasons a candidate fails; an empty list means pass.

    require_stop: the generator emitted an end-of-sequence token before the
    token cap. This is the truncation test. It needs a generator trained
    with EOS, so it rejects everything from the old adapters.

    require_end_punct is off by default. Only 74% of real BIPs end on
    sentence punctuation (lists and fragments are common), so requiring it
    would push the pool away from real style. Turn it on only if truncation
    still slips through.
    """
    min_words: int = 10
    max_rt_per_100: float = 7.0
    max_title_case: float = 0.70
    min_marks_per_100: float = 1.0
    require_stop: bool = True
    require_end_punct: bool = False

    def check(self, text: str, stopped: bool | None = None) -> list[str]:
        m = text_metrics(text)
        if m["n_words"] < self.min_words:
            return ["short"]
        reasons = []
        if self.require_stop and stopped is False:
            reasons.append("no_stop")
        if self.require_end_punct and not m["ends_punct"]:
            reasons.append("no_end_punct")
        if max(m["rt"], m["last_rt"]) > self.max_rt_per_100:
            reasons.append("run_together")
        if max(m["tc"], m["last_tc"]) > self.max_title_case:
            reasons.append("title_case")
        if min(m["marks"], m["last_marks"]) < self.min_marks_per_100:
            reasons.append("few_sentence_marks")
        return reasons
