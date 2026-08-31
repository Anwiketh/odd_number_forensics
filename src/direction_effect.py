"""The direction effect, computed correctly.

delta(block pays more for disobeying) - delta(block pays more for obeying),
using only conditions where BOTH options are paid so the zero channel is held
out. This is what the Odd Number environment claims to measure.

WHY THIS FILE EXISTS. An earlier analysis computed the standard error by pooling
the four (config x mention-order) cells on each side. That is wrong: mention
order produces a large SYSTEMATIC shift in delta (up to +4.6 nats on
Qwen3-0.6B), so pooling it into the noise term inflated the SE by up to 8x and
manufactured a spurious "no significant effect below 2B" threshold. Order must
be a blocking factor: form the contrast WITHIN each order, where the primacy
shift is common and cancels, then average the blocks and take the SE from the
residual config-level spread.

The corrected picture has no threshold. The effect is small at every scale
(+0.38 to +1.53 nats) and grows roughly 4x from 0.6B to 4B, while the payment
structure is worth +4 to +11 nats and word order alone up to +4.6.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")
ENVS = ["parity", "magnitude", "letter"]
ORD = ["hack_first", "obey_first"]
HACK, OBEY, EQ = ["3/1", "4/2"], ["1/3", "2/4"], ["1/1", "3/3"]
SIZE = {"Qwen3-0.6B": 0.6, "Qwen3.5-0.8B": 0.8, "Qwen2.5-1.5B-Instruct": 1.5,
        "Qwen3.5-2B": 2.0, "Qwen3.5-4B": 4.0, "Qwen3.5-9B": 9.0}


def load(tag):
    d = json.load(open(os.path.join(RESULTS, f"payout_{tag}.json"), encoding="utf-8"))
    return {(r["env"], r["cfg"], r["order"]): r["delta"] for r in d["rows"]}, \
           [c for c, _, _ in d["configs"]]


def direction_effect(by, env):
    """Order-blocked contrast, with SE from residual config spread."""
    blocks = [np.mean([by[(env, c, o)] for c in HACK])
              - np.mean([by[(env, c, o)] for c in OBEY]) for o in ORD]
    est = float(np.mean(blocks))
    resid = []
    for o in ORD:
        for grp in (HACK, OBEY):
            vals = [by[(env, c, o)] for c in grp]
            resid += list(np.array(vals) - np.mean(vals))
    dof = len(resid) - 4
    sd = np.sqrt(np.sum(np.square(resid)) / dof) if dof > 0 else 0.0
    se = sd * np.sqrt(1 / len(HACK) / len(ORD) + 1 / len(OBEY) / len(ORD)) * np.sqrt(len(ORD))
    return est, 1.96 * float(se)


def payment_structure(by, env):
    """delta when only the disobedient answer is paid, minus when both are."""
    only = np.mean([by[(env, c, o)] for c in ("1/0", "3/0", "words") for o in ORD])
    both = np.mean([by[(env, c, o)] for c in HACK + OBEY + EQ for o in ORD])
    return float(only - both)


def primacy(by, cfgs, env):
    return float(np.mean([by[(env, c, "hack_first")] - by[(env, c, "obey_first")]
                          for c in cfgs]))


def main():
    tags = sorted([f[len("payout_"):-5] for f in os.listdir(RESULTS)
                   if f.startswith("payout_") and f.endswith(".json")
                   and f[len("payout_"):-5] in SIZE], key=lambda t: SIZE[t])
    print("Direction effect (order-blocked), payment-structure effect, and the")
    print("word-order artifact. All in nats. * = 95% CI excludes zero.\n")
    hdr = f"{'model':22s} {'size':>5s} | " + " ".join(f"{e:>16s}" for e in ENVS)
    print(hdr + f" | {'mean':>6s} {'structure':>10s} {'primacy':>8s}")
    for t in tags:
        by, cfgs = load(t)
        cells, effs = [], []
        for e in ENVS:
            est, hw = direction_effect(by, e)
            effs.append(est)
            cells.append(f"{est:+.2f}{'*' if est - hw > 0 else ' '}+-{hw:.2f}")
        struct = np.mean([payment_structure(by, e) for e in ENVS])
        prim = np.mean([primacy(by, cfgs, e) for e in ENVS])
        print(f"{t:22s} {SIZE[t]:4g}B | " + " ".join(f"{c:>16s}" for c in cells)
              + f" | {np.mean(effs):+6.2f} {struct:+10.2f} {prim:+8.2f}")
    print("\nRead: the incentive's own contribution (direction effect) is small at")
    print("every scale and grows with it. The payment structure -- merely which")
    print("option is flagged as paid at all -- is several times larger throughout.")


if __name__ == "__main__":
    main()
