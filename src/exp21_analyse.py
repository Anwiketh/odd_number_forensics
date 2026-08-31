"""exp21 analysis: does the model read the payouts, or only binarise them?

The pattern in the raw table suggests delta is a function of WHICH options are
paid at all -- a two-bit fact -- plus a primacy term for which option is named
first, with the actual point values contributing nothing. This script tests
that as a model comparison rather than by eyeballing.

  BINARY   delta ~ hack_is_paid + obey_is_paid + hack_named_first
  AMOUNTS  delta ~ (the same) + (hack_pay - obey_pay) + log ratio of payouts

AMOUNTS strictly contains BINARY, so its R^2 cannot be lower. The question is
whether telling the model how much each option pays buys any explanatory power
over merely telling it whether each option pays. Reported as adjusted R^2 (the
extra terms are free parameters) and as the F-test for the two added terms.
"""
import os, sys, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")
ENVS = ["parity", "magnitude", "letter"]


def load(tag):
    with open(os.path.join(RESULTS, f"payout_{tag}.json"), encoding="utf-8") as f:
        d = json.load(f)
    pay = {lab: ((1, 0) if isinstance(h, str) else (h, o))
           for lab, h, o in d["configs"]}
    return d["rows"], pay


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    n, k = X.shape
    r2 = 1 - ss_res / ss_tot
    adj = 1 - (1 - r2) * (n - 1) / (n - k)
    return b, ss_res, r2, adj


def main():
    tags = sorted(f[len("payout_"):-5] for f in os.listdir(RESULTS)
                  if f.startswith("payout_") and f.endswith(".json"))
    for tag in tags:
        rows, pay = load(tag)
        print(f"\n{'='*72}\n{tag}\n{'='*72}")

        for env in ENVS:
            sub = [r for r in rows if r["env"] == env]
            h = np.array([pay[r["cfg"]][0] for r in sub], float)
            o = np.array([pay[r["cfg"]][1] for r in sub], float)
            first = np.array([r["order"] == "hack_first" for r in sub], float)
            y = np.array([r["delta"] for r in sub], float)
            one = np.ones_like(y)

            Xb = np.column_stack([one, (h > 0).astype(float),
                                  (o > 0).astype(float), first])
            Xa = np.column_stack([Xb, h - o,
                                  np.log((h + 1) / (o + 1))])
            bb, ssb, r2b, adjb = ols(Xb, y)
            ba, ssa, r2a, adja = ols(Xa, y)
            df1, df2 = Xa.shape[1] - Xb.shape[1], len(y) - Xa.shape[1]
            F = ((ssb - ssa) / df1) / (ssa / df2) if ssa > 0 else float("inf")

            print(f"\n--- {env} (n = {len(y)} conditions) ---")
            print(f"  BINARY   (is each option paid? who is named first?)"
                  f"   R2 = {r2b:.3f}   adj = {adjb:.3f}")
            print(f"  AMOUNTS  (+ payout difference and log ratio)        "
                  f"   R2 = {r2a:.3f}   adj = {adja:.3f}")
            print(f"  adding the actual point values buys "
                  f"{100*(r2a-r2b):+.1f} pp of R2   (F({df1},{df2}) = {F:.2f})")
            print(f"  fitted:  delta = {bb[0]:+.2f} "
                  f"{bb[1]:+.2f}*[disobedient option is paid] "
                  f"{bb[2]:+.2f}*[obedient option is paid] "
                  f"{bb[3]:+.2f}*[disobedient named first]")

            # the four payment patterns, and the no-incentive control
            print("   payment pattern            mean delta   configs")
            for (hp, op), nm in [((1, 0), "only DISOBEDIENT paid"),
                                 ((0, 1), "only obedient paid"),
                                 ((1, 1), "both paid"),
                                 ((0, 0), "neither paid")]:
                sel = [r for r, hh, oo in zip(sub, h, o)
                       if (hh > 0) == hp and (oo > 0) == op
                       and r["order"] == "hack_first"]
                if sel:
                    labs = sorted({r["cfg"] for r in sel})
                    print(f"   {nm:26s} {np.mean([r['delta'] for r in sel]):+7.2f}"
                          f"   {', '.join(labs)}")
            eq = [r for r in sub if r["direction"] == 0 and r["cfg"] != "0/0"
                  and r["order"] == "hack_first"]
            iv = [r for r in sub if r["direction"] != 0 and r["order"] == "hack_first"
                  and pay[r["cfg"]][0] > 0 and pay[r["cfg"]][1] > 0]
            if eq and iv:
                print(f"   NO-INCENTIVE control (1/1, 3/3): "
                      f"{np.mean([r['delta'] for r in eq]):+.2f}    "
                      f"real incentive, both paid (3/1, 4/2, 1/3, 2/4): "
                      f"{np.mean([r['delta'] for r in iv]):+.2f}")


if __name__ == "__main__":
    main()
