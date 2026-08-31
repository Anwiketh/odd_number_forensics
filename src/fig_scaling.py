r"""Three things across scale. The one the environment claims to measure is the
smallest of them at every size tested.

A. COMPREHENSION (exp22). How much of the model's demonstrated ability to
   compare two numbers survives when they are framed as a reward. Climbs from
   nothing to ~80% and saturates.

B. WORD-ORDER ARTIFACT (exp21). delta when the disobedient option is named
   first minus when named last. Pure presentation artifact. Large at 0.6B,
   negligible from 0.8B on.

C. THE INCENTIVE vs THE MERE PAYMENT STRUCTURE. Solid: the direction effect --
   delta when the block pays more for disobeying minus when it pays more for
   obeying, both options paid, order-blocked. Dashed: the structure effect --
   delta when only the disobedient option is paid at all, minus when both are.
   The thing the environment claims to measure is the solid line; it is 2-12x
   smaller than the dashed one everywhere.

Writes figures/fig10_scaling.png.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from viz import CAT, INK, INK2, MUTED, BASELINE, despine, zeroline
from direction_effect import (load, direction_effect, payment_structure,
                              primacy, SIZE, ENVS, RESULTS)

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "figures")


def main():
    ptags = sorted([f[len("payout_"):-5] for f in os.listdir(RESULTS)
                    if f.startswith("payout_") and f.endswith(".json")
                    and f[len("payout_"):-5] in SIZE], key=lambda t: SIZE[t])
    ctags = sorted([f[len("magnitude_"):-5] for f in os.listdir(RESULTS)
                    if f.startswith("magnitude_") and f.endswith(".json")
                    and f[len("magnitude_"):-5] in SIZE], key=lambda t: SIZE[t])
    PS, CS = [SIZE[t] for t in ptags], [SIZE[t] for t in ctags]

    comp = []
    for t in ctags:
        m = json.load(open(os.path.join(RESULTS, f"magnitude_{t}.json"), encoding="utf-8"))
        comp.append(100 * m["probeA_reward_nonzero"][0] / m["probeB_raw_nonzero"][0])

    prim = {e: [] for e in ENVS}
    dirn, dhw, struct = [], [], []
    for t in ptags:
        by, cfgs = load(t)
        for e in ENVS:
            prim[e].append(primacy(by, cfgs, e))
        ds = [direction_effect(by, e) for e in ENVS]
        dirn.append(np.mean([d[0] for d in ds]))
        dhw.append(np.mean([d[1] for d in ds]) / np.sqrt(len(ENVS)))
        struct.append(np.mean([payment_structure(by, e) for e in ENVS]))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

    ax[0].plot(CS, comp, "-o", color=CAT[3], lw=2.2, zorder=3)
    for x, y in zip(CS, comp):
        ax[0].annotate(f"{abs(y):.0f}%", (x, y), textcoords="offset points",
                       xytext=(0, 9), ha="center", fontsize=8.5, color=INK2)
    ax[0].axhline(100, ls=":", lw=1.2, color=BASELINE)
    ax[0].text(CS[-1], 100, "full engagement ", fontsize=7.5, color=MUTED,
               va="bottom", ha="right")
    ax[0].set_ylim(-12, 118)
    ax[0].set_ylabel("magnitude sensitivity in a reward frame\n(% of same model asked directly)")
    ax[0].set_title("A. does the model READ the reward?\n     climbs, then saturates", loc="left")

    for k, e in enumerate(ENVS):
        ax[1].plot(PS, prim[e], "-o", color=CAT[k], lw=1.9, label=e, zorder=3)
    ax[1].set_ylabel("$\delta$(disobedient named 1st) $-$ $\delta$(named last), nats")
    ax[1].set_title("B. pure word-order artifact\n     dominates at 0.6B, gone by 0.8B", loc="left")
    ax[1].legend(fontsize=7.5)

    ax[2].errorbar(PS, dirn, yerr=dhw, fmt="-o", color=CAT[0], lw=2.2, capsize=3,
                   zorder=4, label="the INCENTIVE\n(pays more for disobeying\nvs for obeying)")
    ax[2].plot(PS, struct, "--s", color=CAT[1], lw=2.0, zorder=3,
               label="mere PAYMENT STRUCTURE\n(only disobedient paid\nvs both paid)")
    ax[2].set_ylabel("effect on $\delta$, nats")
    ax[2].set_title("C. the environment claims to measure the solid line\n"
                    "     it is 2-12x smaller than the dashed one", loc="left")
    ax[2].legend(fontsize=7, loc="upper left")
    ax[2].set_ylim(bottom=min(-0.5, min(dirn) - 0.5))

    for j, a in enumerate(ax):
        T = CS if j == 0 else PS
        a.set_xscale("log"); a.set_xticks(T)
        a.set_xticklabels([f"{s:g}B" for s in T], fontsize=8.5)
        a.set_xticks([], minor=True)
        a.tick_params(axis="x", which="minor", length=0)
        a.set_xlabel("model size (params, log scale)")
        despine(a)
    zeroline(ax[1]); zeroline(ax[2])

    fig.suptitle("Comprehension scales and the word-order artifact vanishes — but the "
                 "incentive's own contribution stays small at every size.",
                 x=0.006, ha="left", fontsize=12, fontweight="semibold",
                 color=INK, y=1.03)
    fig.tight_layout()
    out = os.path.join(FIG, "fig10_scaling.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
    print("")
    print(f"{'model':22s} {'size':>5s} {'comprehension':>14s}")
    for t, c in zip(ctags, comp):
        print(f"{t:22s} {SIZE[t]:4g}B {abs(c):13.0f}%")
    print("")
    print(f"{'model':22s} {'size':>5s} {'incentive':>10s} {'structure':>10s} {'ratio':>7s} {'primacy':>8s}")
    for i, t in enumerate(ptags):
        pr = np.mean([prim[e][i] for e in ENVS])
        print(f"{t:22s} {SIZE[t]:4g}B {dirn[i]:+10.2f} {struct[i]:+10.2f} "
              f"{struct[i]/dirn[i]:6.1f}x {pr:+8.2f}")


if __name__ == "__main__":
    main()
