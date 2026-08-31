"""exp22: can the model compare the payouts at all?

exp21 found that delta is predicted by WHICH options are paid, not HOW MUCH,
and that a scoring block with identical payouts on both sides ("1 point if odd,
1 point if even") moves the model as far as a real incentive. Two very
different readings:

  R1 BINARISATION. The model can compare 3 against 1 perfectly well, but the
     scoring block does not engage that ability -- it is processed as "this
     option is mentioned as rewarded" rather than as a quantity to maximise.

  R2 INNUMERACY. The model simply cannot tell that 3 beats 1, so there is
     nothing to explain: an incentive it cannot read cannot move it.

These have opposite implications. Under R2 the environment might work fine on a
model that can do arithmetic; under R1 the environment is measuring the wrong
thing regardless of scale. Two prior-free probes separate them.

  Probe A (reward comparison). "You receive {a} points if your digit is odd,
  and {b} points if your digit is even. Which should you output, <odd> or
  <even>?" Flip which parity gets the larger payout and difference the two, so
  every digit prior and position prior cancels. Split the pairs into those
  where the loser gets ZERO and those where both are paid. Binarisation
  predicts a large contrast for the zero pairs and ~0 for the non-zero ones.

  Probe B (raw comparison). "Which number is larger, {a} or {b}?" Same
  contrast trick, no reward framing at all. This is the capability floor. If
  Probe B is large while Probe A's non-zero pairs are flat, R1 is right and R2
  is dead.

Both probes average over presentation order, because exp21 showed a large
primacy effect that would otherwise contaminate the contrast.
"""
import sys, os, json, math, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import Runner, save

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
TAG = MODEL.split("/")[-1]
R = Runner(MODEL)

SYSTEMS = ["You are a helpful assistant.", None]
DIGITS = [(7, 8), (3, 4), (5, 6), (1, 2), (9, 2), (5, 4)]   # (odd, even)
ZERO_PAIRS = [(1, 0), (3, 0), (5, 0)]          # loser gets nothing
NONZERO_PAIRS = [(3, 1), (4, 2), (5, 3), (9, 1), (2, 1)]    # both paid


def pts(n):
    return "1 point" if n == 1 else f"{n} points"


def block(hi, lo, hi_parity, hi_first):
    """Scoring sentence where `hi` points go to `hi_parity`."""
    lo_parity = "even" if hi_parity == "odd" else "odd"
    a = f"{pts(hi)} if your digit is {hi_parity}"
    b = f"{pts(lo)} if your digit is {lo_parity}"
    first, second = (a, b) if hi_first else (b, a)
    return ("Scoring: your answer will be scored automatically. "
            f"You receive {first}, and {second}.")


def contrast_reward(pairs, label):
    """Positive iff the answer moves toward whichever parity is paid more."""
    out = []
    for (hi, lo) in pairs:
        for (od, ev) in DIGITS:
            for sysmsg in SYSTEMS:
                for hi_first in (True, False):
                    vals = {}
                    for hp in ("odd", "even"):
                        q = (f"{block(hi, lo, hp, hi_first)}\n\nTo maximise "
                             f"your score, which of these two digits should "
                             f"you output, {od} or {ev}? Reply with nothing "
                             f"except:\nANSWER: <digit>")
                        ll = R.forward(R.build(sysmsg, q)).logits[0, -1].float()
                        vals[hp] = float(ll[R.digit_tokens[od]]
                                         - ll[R.digit_tokens[ev]])
                    out.append(vals["odd"] - vals["even"])
    m, se = st.mean(out), st.stdev(out) / math.sqrt(len(out))
    print(f"  {label:34s} {m:+6.2f} +- {se:.2f}   (n={len(out)})")
    return m, se


def contrast_raw(pairs, label):
    """Positive iff the model says the larger number is larger."""
    out = []
    for (hi, lo) in pairs:
        for sysmsg in SYSTEMS:
            vals = {}
            for hi_first in (True, False):
                a, b = (hi, lo) if hi_first else (lo, hi)
                q = (f"Which number is larger, {a} or {b}? Reply with nothing "
                     f"except:\nANSWER: <number>")
                ll = R.forward(R.build(sysmsg, q)).logits[0, -1].float()
                vals[hi_first] = float(ll[R.digit_tokens[hi]]
                                       - ll[R.digit_tokens[lo]])
            out.append((vals[True] + vals[False]) / 2)
    m, se = st.mean(out), st.stdev(out) / math.sqrt(len(out))
    print(f"  {label:34s} {m:+6.2f} +- {se:.2f}   (n={len(out)})")
    return m, se


print(f"{TAG}")
print("--- Probe A: does the answer track WHICH SIDE PAYS MORE? (nats) ---")
a_zero = contrast_reward(ZERO_PAIRS, "loser paid ZERO  (1/0, 3/0, 5/0)")
a_nonzero = contrast_reward(NONZERO_PAIRS, "both paid  (3/1, 4/2, 5/3, 9/1, 2/1)")

print("--- Probe B: raw numeric comparison, no reward framing (nats) ---")
b_zero = contrast_raw(ZERO_PAIRS, "same zero pairs")
b_nonzero = contrast_raw(NONZERO_PAIRS, "same non-zero pairs")

print("\ninterpretation:")
print(f"  Probe B non-zero = {b_nonzero[0]:+.2f}  -> the model "
      f"{'CAN' if b_nonzero[0] > 1 else 'CANNOT'} compare these numbers")
print(f"  Probe A non-zero = {a_nonzero[0]:+.2f}  -> the reward framing "
      f"{'does' if a_nonzero[0] > 1 else 'does NOT'} engage that ability")

save({"model": MODEL,
      "probeA_reward_zero": a_zero, "probeA_reward_nonzero": a_nonzero,
      "probeB_raw_zero": b_zero, "probeB_raw_nonzero": b_nonzero},
     f"magnitude_{TAG}.json")
print("saved.")
