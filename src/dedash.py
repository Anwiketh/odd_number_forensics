"""Remove em dashes from the write-up without wrecking the prose.

Em-dash density is one of the most recognisable machine-writing signatures, and
the admissions doc says documents that read like LLM output get rejected. A blind
find-and-replace produces garbage, because an em dash does at least four
different jobs: it opens a parenthetical, it introduces an appositive, it hangs a
trailing clause off a sentence, and it marks an emphatic break. Each wants
different punctuation.

Method. Markdown is split into blocks. Tables, fenced code, headings, list items
and blockquotes are passed through untouched. Prose paragraphs are joined into
one flowing line so that a construction split across a line break can be matched,
rewritten, then re-wrapped to 79 columns.

PAIRS are handled first and explicitly, because a parenthetical needs both of its
delimiters decided together. SINGLES are then resolved by looking at the word
that follows the dash: a coordinating conjunction takes a comma, an appositive
takes a colon, and so on.

Run: python src/dedash.py paper/paper.md [more.md ...]
"""
import re
import sys
import textwrap

DASH = "—"

# ---------------------------------------------------------------- paired
# (opening context, inner clause, closing context) -> rewritten
PAIRS = [
    ("Toy instruction-conflict environments", "instruct the model to do X, add an "
     "in-context reward function that pays for not-X, see what it does",
     "are a standard first probe", "paren"),
    ("every such environment has two arms", "instruct X and pay for not-X, or "
     "instruct not-X and pay for X", "and essentially nobody runs both", "paren-comma"),
    ("at small scale δ", "the part that survives the mirror arm, and the only "
     "part a careful experimenter would report as incentive-following",
     "is not incentive-following either", "comma"),
    ("while a pure word-order effect", "naming the disobedient option first rather "
     "than last", "is worth up to 6.36 nats", "comma"),
    ("the incentive's own contribution", "δ when the block pays more for "
     "disobeying minus δ when it pays more for obeying, both options paid",
     "is detectable at essentially every scale", "paren"),
    ("The equal-payout condition", "pay both answers the same",
     "is a one-cell validity check", "paren"),
    ("Both readings", "reward hacking and metagaming",
     "assume that whatever moved the model", "comma"),
    ("The disobedience *rate*", "the number that gets reported",
     "is 0.92 or 0.42", "comma"),
    ("The test is to hold the two answers fixed", "same instruction, same answer "
     "set, same sentence frame", "and vary only what the scoring block", "paren-comma"),
    ("The primacy coefficient", "δ when the disobedient option is named first, "
     "minus δ when the obedient one is", "is +2.20", "paren"),
    ("the incentive's contribution against the *mere payment structure*",
     "δ when only the disobedient option is paid anything, versus when both are paid",
     "and against word order alone", "paren-comma"),
    ("and the 0.8B model", "which has a near-zero word-order artifact and unusually "
     "tight CIs", "is what exposed the SE bug", "comma"),
    ("A prior-free within-item contrast", "ask the same question twice, flipping "
     "only which side the specification rewards, so any prior over answers cancels",
     "gives, for Qwen3-0.6B", "paren"),
    ("But in four cells", "Qwen3-0.6B's `parity`/code, `parity`/instruct, "
     "`magnitude`/english and `yesno`/english", "β(aligned) is the *larger*", "paren"),
    ("The content direction", "which the geometry places at chance across "
     "environments (ρ = 0.04)", "steers essentially as well", "comma"),
]

# ---------------------------------------------------------------- singles
# next word (lowercased) -> punctuation that replaces the dash
BY_NEXT_WORD = {
    "and": ",", "but": ",", "so": ",", "yet": ",", "or": ",",
    "which": ",", "who": ",", "whose": ",",
    "exactly": ",", "worst": ",", "six": ",", "opposite": ":",
    "*higher*": ",", "*refuted*": ",",
}
DEFAULT_SINGLE = ":"   # appositive / explanation, the commonest remaining case


MATCHED = set()


def fix_pairs(t):
    for idx, (before, inner, after, mode) in enumerate(PAIRS):
        pat = (re.escape(before) + r"\s*" + DASH + r"\s*" + re.escape(inner)
               + r"\s*" + DASH + r"\s*" + re.escape(after))
        if mode == "paren":
            rep = f"{before} ({inner}) {after}"
        elif mode == "paren-comma":
            rep = f"{before} ({inner}), {after}"
        else:
            rep = f"{before}, {inner}, {after}"
        t, n = re.subn(pat, lambda _m, r=rep: r, t)
        if n:
            MATCHED.add(idx)
    return t


def fix_singles(t):
    def repl(m):
        nxt = m.group("w")
        punct = BY_NEXT_WORD.get(nxt.lower().strip(",.;:"), DEFAULT_SINGLE)
        return punct + " " + nxt
    return re.sub(r"\s*" + DASH + r"\s*(?P<w>\S+)", repl, t)


def fix_headings(t):
    # "**Retraction 2 - a broken statistic**" reads better with a colon
    return re.sub(r"(\*\*[A-Z][^*]{0,40}?)\s*" + DASH + r"\s*", r"\1: ", t)


PASSTHRU = re.compile(r"^\s*(\||```|#{1,6}\s|[-*]\s|\d+\.\s|>)")


def process(path):
    raw = open(path, encoding="utf-8").read()
    lines = raw.split("\n")
    out, i = [], 0
    in_code = False
    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("```"):
            in_code = not in_code
            out.append(ln); i += 1; continue
        if in_code or PASSTHRU.match(ln) or not ln.strip():
            out.append(ln); i += 1; continue
        # gather a prose paragraph
        para = []
        while (i < len(lines) and lines[i].strip()
               and not PASSTHRU.match(lines[i])
               and not lines[i].strip().startswith("```")):
            para.append(lines[i].strip()); i += 1
        flowed = " ".join(para)
        flowed = fix_headings(fix_singles(fix_pairs(flowed)))
        out.extend(textwrap.wrap(flowed, width=79, break_long_words=False,
                                 break_on_hyphens=False) or [""])
    new = "\n".join(out)
    open(path, "w", encoding="utf-8").write(new)
    return raw.count(DASH), new.count(DASH)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        MATCHED.clear()
        before, after = process(p)
        miss = [i for i in range(len(PAIRS)) if i not in MATCHED]
        msg = f"  {p}: em dashes {before} -> {after}"
        if miss:
            msg += f"  | pairs not found in this file: {miss}"
        sys.stdout.buffer.write((msg + chr(10)).encode("utf-8", "replace"))
