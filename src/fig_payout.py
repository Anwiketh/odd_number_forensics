"""Figure for exp21: what the scoring block actually does.

Three claims, one panel each.

A. delta does not track the stated payout difference. Left of the vertical line
   the block pays MORE for obeying, so incentive-following predicts delta < 0
   there. Almost nothing is.

B. delta is close to a step function of a two-bit fact -- which of the two
   options is paid ANYTHING. Amounts do not enter.

C. the control that matters: conditions where both answers are worth exactly
   the same ("1 point if odd, 1 point if even"), so there is no incentive at
   all, against conditions with a genuine one.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from viz import CAT, INK, INK2, MUTED, BASELINE, GRID, despine, zeroline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
ZERO_C, BOTH_C, EQ_C = CAT[1], CAT[0], CAT[3]
ENVS = ["parity", "magnitude", "letter"]
MK = {"parity": "o", "magnitude": "^", "letter": "s"}
PATTERNS = [((1, 0), "only the\ndisobedient\nanswer paid"),
            ((1, 1), "both\npaid"),
            ((0, 0), "neither\npaid"),
            ((0, 1), "only the\nobedient\nanswer paid")]


def load(tag):
    with open(os.path.join(RESULTS, f"payout_{tag}.json"), encoding="utf-8") as f:
        d = json.load(f)
    pay = {lab: ((1, 0) if isinstance(h, str) else (h, o))
           for lab, h, o in d["configs"]}
    return d["rows"], pay


def boot(vals, rng, n=5000):
    v = np.asarray(vals, float)
    if len(v) < 2:
        return float(v.mean()), 0.0, 0.0
    m = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    mu = float(v.mean())
    return mu, mu - float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)) - mu


def main():
    tags = sorted(f[len("payout_"):-5] for f in os.listdir(RESULTS)
                  if f.startswith("payout_") and f.endswith(".json"))
    if not tags:
        print("no results/payout_*.json -- run exp21_payout.py first")
        return
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(tags), 3, figsize=(15.5, 4.3 * len(tags)),
                             gridspec_kw=dict(width_ratios=[1.25, 1.15, 0.8],
                                              hspace=0.48, wspace=0.30))
    axes = np.atleast_2d(axes)

    for row, tag in enumerate(tags):
        rows, pay = load(tag)

        # ---------------- A: delta vs stated payout difference -------------
        # Restricted to the canonical presentation order (disobedient option
        # named first), which is how the Odd Number block is actually written.
        ax = axes[row, 0]
        sub = [r for r in rows if r["order"] == "hack_first"]
        X = np.array([pay[r["cfg"]][0] - pay[r["cfg"]][1] for r in sub], float)
        Y = np.array([r["delta"] for r in sub], float)
        Z = np.array([r["zero"] for r in sub])
        lo, hi = -4.0, 4.0
        for e in ENVS:
            m0 = np.array([r["env"] == e for r in sub])
            j = {"parity": -0.13, "magnitude": 0.0, "letter": 0.13}[e]
            for m, c in [(m0 & ~Z, BOTH_C), (m0 & Z, ZERO_C)]:
                eq = m & (X == 0)
                ax.scatter(X[m & ~eq] + j, Y[m & ~eq], s=32, marker=MK[e],
                           color=c, edgecolor=INK, lw=0.4, zorder=3, alpha=.92)
                if eq.any():
                    ax.scatter(X[eq] + j, Y[eq], s=44, marker=MK[e], color=EQ_C,
                               edgecolor=INK, lw=0.6, zorder=4)
        # two fits: does delta track the payout gap within each family?
        notes = []
        for m, c, nm in [(~Z, BOTH_C, "both paid"), (Z, ZERO_C, "one pays 0")]:
            s, b = np.polyfit(X[m], Y[m], 1)
            xx = np.linspace(lo + .4, hi - .4, 8)
            ax.plot(xx, s * xx + b, color=c, lw=1.8, alpha=.85, zorder=2)
            notes.append((c, f"{nm}: slope {s:+.2f} nats/point"))
        pad = 0.34 * (Y.max() - Y.min())
        ax.set_xlim(lo, hi)
        ax.set_ylim(min(Y.min() - pad, -2.5), Y.max() + pad)
        ax.axvline(0, color=BASELINE, lw=1.0, zorder=1)
        y1 = ax.get_ylim()[1]
        for i, (c, t) in enumerate(notes):
            ax.text(lo + 0.15, y1 - (0.055 + i * 0.075) * (y1 - ax.get_ylim()[0]),
                    t, fontsize=8, color=c, va="top", ha="left",
                    fontweight="bold")
        ax.set_xlabel("points for the DISOBEDIENT answer − points for the obedient one")
        ax.set_ylabel("$\\delta$   (nats toward disobeying)")
        ax.set_title(f"{tag}\nA. within a payment pattern, the amounts do nothing",
                     loc="left")
        despine(ax)
        zeroline(ax)
        hs = [plt.Line2D([], [], ls="", marker="o", color=c, mec=INK, mew=.4)
              for c in (ZERO_C, BOTH_C, EQ_C)]
        ax.legend(hs, ["one option pays 0", "both options paid",
                       "EQUAL pay = no incentive"], loc="lower right",
                  fontsize=7.5)

        # ---------------- B: delta by binary payment pattern ---------------
        ax = axes[row, 1]
        w = 0.26
        for k, e in enumerate(ENVS):
            sub = [r for r in rows if r["env"] == e and r["order"] == "hack_first"]
            mus, los, his = [], [], []
            for (hp, op), _ in PATTERNS:
                v = [r["delta"] for r in sub
                     if (pay[r["cfg"]][0] > 0) == hp and (pay[r["cfg"]][1] > 0) == op]
                m, l, u = boot(v, rng)
                mus.append(m); los.append(l); his.append(u)
            ax.bar(np.arange(4) + (k - 1) * w, mus, w,
                   yerr=[los, his], color=[ZERO_C, BOTH_C, MUTED, ZERO_C],
                   alpha=[1.0, .74, .5][k], edgecolor=INK, linewidth=0.4,
                   error_kw=dict(ecolor=INK2, lw=0.9, capsize=1.8))
        ax.set_xticks(range(4))
        ax.set_xticklabels([p[1] for p in PATTERNS], fontsize=7.5)
        ax.set_ylabel("$\\delta$   (nats)")
        ax.set_title("B. but WHICH option is paid at all does\n"
                     "     (bar shade = parity / magnitude / letter)", loc="left")
        despine(ax)
        zeroline(ax)

        # ---------------- C: the no-incentive control ----------------------
        ax = axes[row, 2]
        xs, hsv, es, cs = [], [], [], []
        for i, e in enumerate(ENVS):
            sub = [r for r in rows if r["env"] == e and r["order"] == "hack_first"]
            for j, f in enumerate([
                    lambda r: r["direction"] == 0 and r["cfg"] != "0/0",
                    lambda r: r["direction"] != 0 and pay[r["cfg"]][0] > 0
                    and pay[r["cfg"]][1] > 0]):
                m, l, u = boot([r["delta"] for r in sub if f(r)], rng)
                xs.append(i * 2.4 + j); hsv.append(m); es.append([l, u])
                cs.append([EQ_C, BOTH_C][j])
        ax.bar(xs, hsv, 0.82, yerr=np.array(es).T, color=cs, edgecolor=INK,
               linewidth=0.4, error_kw=dict(ecolor=INK2, lw=1.0, capsize=2))
        ax.set_xticks([i * 2.4 + 0.5 for i in range(len(ENVS))])
        ax.set_xticklabels(ENVS, fontsize=8)
        ax.set_ylabel("$\\delta$   (nats)")
        ax.set_title("C. no incentive at all\n     vs a real one", loc="left")
        despine(ax)
        zeroline(ax)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.28)
        hs = [plt.Rectangle((0, 0), 1, 1, fc=c, ec=INK, lw=.4)
              for c in (EQ_C, BOTH_C)]
        ax.legend(hs, ["equal pay (no incentive)", "real incentive"],
                  loc="upper center", fontsize=7, ncol=1)

    fig.suptitle("exp21 — payout direction × zero-ness × mention order. "
                 "Every condition is the same sentence naming the same two options.",
                 x=0.006, ha="left", fontsize=11.5, fontweight="semibold",
                 color=INK, y=1.002)
    out = os.path.join(FIG, "fig9_payout.png")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
