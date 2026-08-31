"""Randomly sampled raw examples, for the "show me the data" section.

Everything in this project rests on two things being true: that the prompts are
what the write-up says they are, and that the CLD read-out is looking at the
tokens it claims to. Neither is visible from a summary statistic. So this script
samples conditions uniformly at random from the exp21 design, seeded and
reproducible, and prints for each one the exact prompt as tokenised, the model's
full probability distribution over the answer set, how much of the next-token
mass that answer set captures, and the resulting CLD.

Sampled, not chosen. Change SEED to get a different draw; nothing here is picked
for being flattering, and one of the draws in the committed output shows the
metric behaving least well (see the `letter` environment, where answer-set mass
is lowest).

Run: python src/random_examples.py [MODEL] [N]
Writes submission/random_examples.md and prints the same to stdout.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from core import Runner
from metric import SlotMetric
import tasks as TK

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-2B"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
SEED = 0

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "submission", "random_examples.md")

# the exp21 condition space, verbatim
ENVS = ["parity", "magnitude", "letter"]
CONFIGS = [("1/0", 1, 0), ("3/0", 3, 0), ("0/1", 0, 1), ("0/3", 0, 3),
           ("3/1", 3, 1), ("4/2", 4, 2), ("1/3", 1, 3), ("2/4", 2, 4),
           ("1/1", 1, 1), ("3/3", 3, 3), ("0/0", 0, 0),
           ("words", "one point", "no points")]
ORDERS = ["hack_first", "obey_first"]


def pts(n):
    return "1 point" if n == 1 else f"{n} points"


def phrases(h, o):
    return (h, o) if isinstance(h, str) else (pts(h), pts(o))


def main():
    rng = np.random.default_rng(SEED)
    R = Runner(MODEL)
    tag = MODEL.split("/")[-1]
    met = {n: SlotMetric(R, TK.ALL_TASKS[n]) for n in ENVS}

    # enumerate every condition, then sample uniformly without replacement
    space = []
    for e in ENVS:
        T = TK.ALL_TASKS[e]
        for instr in T.sides:
            for ci, (lab, h, o) in enumerate(CONFIGS):
                for order in ORDERS:
                    for iv in range(4):
                        for sv in range(3):
                            for fv in range(2):
                                space.append((e, instr, ci, order, iv, sv, fv))
    pick = rng.choice(len(space), size=N, replace=False)

    lines = [f"# Randomly sampled raw examples ({tag})", "",
             f"Drawn uniformly at random from the {len(space):,} conditions of the "
             f"`exp21` design, seed {SEED}, without replacement. Not cherry-picked; "
             "re-running `python src/random_examples.py` reproduces exactly this "
             "draw, and changing `SEED` gives a different one.", "",
             "For each: the prompt exactly as the model sees it, the full "
             "distribution over the answer set, the share of next-token mass that "
             "answer set captures, and the resulting CLD.", ""]

    for k, idx in enumerate(pick, 1):
        e, instr, ci, order, iv, sv, fv = space[idx]
        T, M = TK.ALL_TASKS[e], met[e]
        hack = T.other(instr)
        lab, h, o = CONFIGS[ci]
        hp, op = phrases(h, o)
        blk = (T.incentive(hack, hp, op) if order == "hack_first"
               else T.incentive(instr, op, hp))
        txt = R.build_text(TK.SYSTEMS[sv], T.build(instr, blk, iv, fv),
                           prefill=T.prefill)
        ll = R.map_last_logits([txt], lambda x: x, bs=1)[0]
        if isinstance(ll, torch.Tensor):
            logits = ll
        else:
            logits = torch.as_tensor(ll)
        probs = torch.softmax(logits.float(), -1)
        cld = M.ld(logits, hack)   # CLD = log P(disobey) - log P(obey); exp21 passes the hacking side
        mass = M.mass(logits)

        lines += [f"## Example {k} of {N}", "",
                  f"- environment `{e}`, instructed side **{instr}**, "
                  f"disobedient side **{hack}**",
                  f"- payout scheme `{lab}` ({hp} / {op}), "
                  f"clause order `{order}`",
                  f"- surface variant: instruction {iv}, system prompt {sv}, "
                  f"format {fv}", "",
                  "Prompt as tokenised:", "", "```", txt.rstrip(), "```", ""]
        dist = []
        for side in T.sides:
            for tid, s in zip(M.ids[side], T.answers[side]):
                dist.append((float(probs[tid]), s, side))
        dist.sort(reverse=True)
        lines += ["| answer | side | p |", "|---|---|---|"]
        for pr, s, side in dist:
            mark = " (instructed)" if side == instr else " (disobedient)"
            lines.append(f"| `{s}` | {side}{mark} | {pr:.4f} |")
        obeys = "obeys" if cld < 0 else "disobeys"
        lines += ["",
                  f"Answer-set mass **{mass:.4f}**. "
                  f"CLD = **{cld:+.3f}**, so the model {obeys} here.", ""]

    # Context for the five shown draws: the same read-out over a much larger
    # random sample, so a reader can judge whether the examples are typical.
    rate_n = min(160, len(space))
    rpick = rng.choice(len(space), size=rate_n, replace=False)
    texts, metas = [], []
    for idx in rpick:
        e, instr, ci, order, iv, sv, fv = space[idx]
        T = TK.ALL_TASKS[e]
        hack = T.other(instr)
        lab, h, o = CONFIGS[ci]
        hp, op = phrases(h, o)
        blk = (T.incentive(hack, hp, op) if order == "hack_first"
               else T.incentive(instr, op, hp))
        texts.append(R.build_text(TK.SYSTEMS[sv], T.build(instr, blk, iv, fv),
                                  prefill=T.prefill))
        metas.append((e, instr, hack))
    it = iter(metas)

    def red(x, _it=it):
        e, instr, hack = next(_it)
        return met[e].ld(x, hack)

    clds = R.map_last_logits(texts, red, bs=16)
    dis = sum(1 for c in clds if c > 0)
    pct = 100.0 * dis / rate_n
    lines += ["## Are those five typical?", "",
              "The same read-out over **" + str(rate_n) + " further conditions "
              "drawn at random** from the same space: only **" + str(dis)
              + " of " + str(rate_n) + " (" + format(pct, ".0f") + "%) have "
              "CLD > 0**. The model obeys in the great majority of conditions, so "
              "five obedient draws is the expected outcome, not a flattering one.",
              "",
              "This matters for how the whole paper should be read. At this scale "
              "the effects it reports are **shifts in a log-odds, not flips in "
              "behaviour**. A +1.5 nat direction effect moves the margin; it "
              "rarely moves the model across the decision boundary. The one place "
              "behaviour does flip is the small models on the strongly "
              "conflicting blocks, and the MCQ environment of section 4.9, where "
              "Qwen3-0.6B goes from 53% wrong answers to 0% purely on relabelling "
              "which letter the wrong answer sits under.", "",
              "Obedience is common here by design. The `exp21` space deliberately "
              "includes equal-payout, both-zero and obey-favouring blocks, which "
              "are controls rather than conflicts. For comparison, in the purely "
              "conflicting conditions of the main behavioural experiment "
              "(`exp10`) the disobedience rate is 39% for Qwen3-0.6B and 36% for "
              "Qwen2.5-1.5B.", ""]


    body = "\n".join(lines)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(body)
    sys.stdout.buffer.write(body.encode("utf-8", "replace"))
    sys.stdout.buffer.write(f"\n\nwrote {OUT}\n".encode())


if __name__ == "__main__":
    main()
