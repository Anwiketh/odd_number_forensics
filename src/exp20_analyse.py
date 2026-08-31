"""exp20 follow-up analysis: separate the copying effect from the payout effect.

`exp20_copying.py` prints beta against numeral-parity imbalance pooled over all
ten schemes, and gets r = +0.21 in `parity` -- which looks like a null. It is
not. The pooled correlation is diluted by a much larger factor the original
design did not control for:

  (a) THE SCHEME TABLE DOES NOT HOLD THE INCENTIVE FIXED. `Task.incentive(side,
      hi, lo)` attaches `hi` to the side passed in, and exp20 passes the hacking
      side -- so `hi` is the hacking payout and `lo` the obedient one. Five of
      the ten schemes (1/2, 3/5, 5/7, 2/4, 4/6) therefore pay MORE FOR OBEYING,
      the opposite of what the module docstring claims. See `direction()`.

  (b) delta is 1.4-2.8x larger whenever the losing option pays literally zero
      ("0 points" / "no points") than when it pays some positive number. That
      single factor dominates every other difference between schemes and is why
      the docstring's "delta should be ~constant across numeral choices" control
      fails (sd = 1.78 nats in `parity`).

Restricting the copying test to the seven schemes with a non-zero low payout
removes (b), and (a) turns out to be balanced across the imbalance groups --
each of {even-heavy, odd-heavy} contains exactly one scheme that pays more for
hacking -- so the residual correlation is not a direction artefact.

Run after `exp20_copying.py`. Reads results/copying_<tag>.json.
"""
import json, os, sys, statistics as st
import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results")
TAG = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-0.6B"

# scheme -> (payout for hacking, payout for obeying, n_odd - n_even in numerals)
SCHEMES = {
    "1/0": (1, 0, 0), "3/0": (3, 0, 0), "1/2": (1, 2, 0),
    "3/5": (3, 5, 2), "5/7": (5, 7, 2), "9/3": (9, 3, 2),
    "2/4": (2, 4, -2), "4/6": (4, 6, -2), "8/2": (8, 2, -2),
    "words": (1, 0, 0),
}
ENVS = ["parity", "magnitude", "letter"]


def direction(s):
    hi, lo, _ = SCHEMES[s]
    return "hack" if hi > lo else ("obey" if lo > hi else "equal")


def load(tag):
    with open(os.path.join(RESULTS, f"copying_{tag}.json"), encoding="utf-8") as f:
        d = json.load(f)
    rows = d
    if isinstance(d, dict):
        for v in d.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "beta" in v[0]:
                rows = v
                break
    return {(r["env"], r["scheme"]): r for r in rows}


def main():
    by = load(TAG)
    zero = [s for s in SCHEMES if SCHEMES[s][1] == 0]
    nonz = [s for s in SCHEMES if SCHEMES[s][1] != 0]

    print(f"=== {TAG} ===\n")
    print("A. delta by whether the LOSING option pays zero")
    print(f"   {'env':10s} {'lo==0':>7s} {'lo!=0':>7s} {'ratio':>7s}")
    for e in ENVS:
        z = st.mean(by[(e, s)]["delta"] for s in zero)
        n = st.mean(by[(e, s)]["delta"] for s in nonz)
        print(f"   {e:10s} {z:+7.2f} {n:+7.2f} {z / n:6.2f}x")

    print("\nB. delta by WHICH SIDE PAYS MORE (restricted to lo != 0)")
    print(f"   {'env':10s} {'hack>obey':>10s} {'obey>hack':>10s} {'diff':>7s}")
    for e in ENVS:
        h = st.mean(by[(e, s)]["delta"] for s in nonz if direction(s) == "hack")
        o = st.mean(by[(e, s)]["delta"] for s in nonz if direction(s) == "obey")
        print(f"   {e:10s} {h:+10.2f} {o:+10.2f} {h - o:+7.2f}")

    print("\nC. copying test: beta vs numeral parity imbalance (lo != 0 only)")
    for e in ENVS:
        x = np.array([SCHEMES[s][2] for s in nonz], float)
        y = np.array([by[(e, s)]["beta"] for s in nonz], float)
        r = float(np.corrcoef(x, y)[0, 1])
        slope = float(np.polyfit(x, y, 1)[0])
        print(f"   {e:10s} r = {r:+.3f}  slope = {slope:+.3f} nats/numeral")
        for imb, lab in [(-2, "even-heavy"), (0, "balanced  "), (2, "odd-heavy ")]:
            v = [by[(e, s)]["beta"] for s in nonz if SCHEMES[s][2] == imb]
            d = {direction(s) for s in nonz if SCHEMES[s][2] == imb}
            print(f"{'':14s}{lab} imb={imb:+d}: beta = {st.mean(v):+.2f}"
                  f"   (payout directions present: {sorted(d)})")

    print("\nD. spelled-out control -- does removing the numerals collapse beta?")
    print(f"   {'env':10s} {'beta 1/0':>9s} {'beta words':>11s} {'drop':>6s}  words 95% CI")
    for e in ENVS:
        a, w = by[(e, "1/0")], by[(e, "words")]
        drop = (1 - w["beta"] / a["beta"]) * 100
        ci = w["beta_ci"]
        tag = "includes 0" if ci[0] <= 0 <= ci[1] else "EXCLUDES 0"
        print(f"   {e:10s} {a['beta']:+9.2f} {w['beta']:+11.2f} {drop:5.0f}%  "
              f"[{ci[0]:+.2f}, {ci[1]:+.2f}] {tag}")


if __name__ == "__main__":
    main()
