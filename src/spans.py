"""Map named semantic spans of a prompt to token index ranges.

Patching results are only interpretable if we can say *which part of the
prompt* a position belongs to, and surface variants have different lengths, so
we cannot average raw positions across variants. We therefore locate spans by
character offset and convert to token ranges with the fast tokenizer's
offset mapping.
"""
from typing import Dict, List, Tuple


def token_spans(tok, text: str, named: List[Tuple[str, str]]) -> Dict[str, List[int]]:
    """named: [(span_name, substring)] searched left-to-right, non-overlapping.
    Returns span_name -> list of token indices."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    out, cursor = {}, 0
    for name, sub in named:
        i = text.find(sub, cursor)
        if i < 0:
            raise ValueError(f"span {name!r} substring not found: {sub[:40]!r}")
        j = i + len(sub)
        cursor = j
        idx = [k for k, (a, b) in enumerate(offsets)
               if b > a and a < j and b > i]
        out[name] = idx
    # everything not claimed, in order, becomes 'other'
    claimed = set(k for v in out.values() for k in v)
    out["other"] = [k for k in range(len(offsets)) if k not in claimed]
    return out


def parity_english_spans(tok, text: str, rewarded_word: str, other_word: str,
                         instr_sentence: str, system_present: bool):
    """Span decomposition for the canonical english-incentive prompt."""
    named = []
    if system_present:
        named.append(("system", "system\n"))
    named += [
        ("instr", instr_sentence),
        ("scoring_pre", "Scoring: your answer will be scored automatically. "
                        "You receive 1 point if your digit is"),
        ("rewarded_word", " " + rewarded_word),
        ("mid", ", and 0 points if your digit is"),
        ("other_word", " " + other_word),
        ("fmt_head", "\n\n"),          # start of the format instruction
        ("answer_slot", "ANSWER: "),
    ]
    return token_spans(tok, text, named)
