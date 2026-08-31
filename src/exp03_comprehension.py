"""exp03: a prior-free comprehension probe.

My first probe (exp02 Part B) was confounded: it asked "which of a or b scores
higher" and read the digit logits, but the model has a large a-priori
preference among digits, which swamps the grader signal. The fix is a
WITHIN-ITEM CONTRAST: for the identical question, flip only which parity the
grader rewards, and measure how much the answer moves. Any prior over digits
or over answer positions cancels exactly.

    comprehension score  =  mean over items of
        [ logit(rewarded-under-A) - logit(rewarded-under-B) ]_{grader=A}
      - [ same quantity ]_{grader=B}

This is >0 iff the model's answer tracks the grader definition. It is
reported in nats and is directly comparable across models and across the
code/English modalities.

We also run a second, harder probe: an *arithmetic* readout, where the model
must apply the reward function to a specific number ("reward(7) = ?").
"""
import sys, os, statistics as st, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *
import factorial as FX

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
TAG = MODEL.split("/")[-1]
R = Runner(MODEL)

PAIRS = [(7, 8), (3, 4), (5, 6), (1, 2), (9, 8), (5, 2), (3, 6), (1, 4)]
SYSTEMS = ["You are a helpful assistant.", None]

# ---------------- Probe 1: which digit scores higher (contrast) ------------
print(f"{TAG}\n--- probe 1: comparative (within-item contrast, nats) ---")
p1 = {}
for style in ["code", "english", "instruct"]:
    diffs = []
    for (a, b) in PAIRS:
        for sysmsg in SYSTEMS:
            vals = {}
            for p in ["odd", "even"]:
                blk = FX.FRAMES[style](p)
                q = (f"{blk}\n\nWhich of these two digits should you output, "
                     f"{a} or {b}? Reply with nothing except:\nANSWER: <digit>")
                ll = R.forward(R.build(sysmsg, q)).logits[0, -1].float()
                vals[p] = float(ll[R.digit_tokens[a]] - ll[R.digit_tokens[b]])
            # a is odd in every pair listed above except where noted; use parity
            sign = +1 if a % 2 == 1 else -1
            diffs.append(sign * (vals["odd"] - vals["even"]))
    p1[style] = (st.mean(diffs), st.stdev(diffs) / math.sqrt(len(diffs)))
    print(f"  {style:10s} contrast = {p1[style][0]:+6.2f} +- {p1[style][1]:.2f}")

# ---------------- Probe 2: evaluate the reward function -------------------
# "def reward(n): return 1.0 if n%2==1 else 0.0 ; what is reward(7)?"
# Answer is 1 or 0 -- both single digit tokens, and we contrast across parity
# of n, so digit priors cancel.
print("--- probe 2: evaluate reward(n) for a concrete n (nats) ---")
p2 = {}
for style in ["code", "english"]:
    diffs = []
    for p in ["odd", "even"]:
        blk = FX.FRAMES[style](p)
        for n in [1, 2, 3, 4, 5, 6, 7, 8]:
            for sysmsg in SYSTEMS:
                q = (f"{blk}\n\nHow many points does the digit {n} receive? "
                     "Reply with nothing except:\nANSWER: <number>")
                ll = R.forward(R.build(sysmsg, q)).logits[0, -1].float()
                d = float(ll[R.digit_tokens[1]] - ll[R.digit_tokens[0]])
                # correct answer is 1 point iff n has the rewarded parity
                good = (n % 2) == FX.MOD[p]
                diffs.append(d if good else -d)
    p2[style] = (st.mean(diffs), st.stdev(diffs) / math.sqrt(len(diffs)))
    print(f"  {style:10s} score    = {p2[style][0]:+6.2f} +- {p2[style][1]:.2f}")

save({"model": MODEL, "probe1_comparative": p1, "probe2_evaluate": p2},
     f"comprehension_{TAG}.json")
print("saved.")
