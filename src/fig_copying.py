"""Figure for exp20: what the reward block's *wording* does.

Panel A -- delta by scheme, split on whether the losing option pays literally
zero. This is the headline: the incentive is semantically identical in every
scheme, yet delta is 1.4-2.8x larger whenever the loser pays "0 points" /
"no points" than when it pays some positive number.

Panel B -- the direction control. Among the non-zero schemes, does delta care
which side actually pays more? Barely.

Panel C -- the copying test. beta against the parity imbalance of the printed
numerals, in `parity` (where those numerals are legal answers) against `letter`
(where they are not). The slope is the effect size; the letter slope is the
negative control.

Writes figures/fig8_copying.png. Run after exp20_copying.py.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from viz import CAT, INK, INK2, MUTED, BASELINE, despine, zeroline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# scheme -> (pays for hacking, pays for obeying, n_odd - n_even among numerals)
S = {"1/0": (1, 0, 0), "3/0": (3, 0, 0), "1/2": (1, 2, 0), "3/5": (3, 5, 2),
     "5/7": (5, 7, 2), "9/3": (9, 3, 2), "2/4": (2, 4, -2), "4/6": (4, 6, -2),
     "8/2": (8, 2, -2), "words": (1, 0, 0)}
ORDER = ["1/0", "3/0", "words", "1/2", "3/5", "5/7", "9/3", "2/4", "4/6", "8/2"]
ENVS = ["parity", "magnitude", "letter"]
ZERO_C, NONZERO_C = CAT[1], CAT[0]      # orange = loser paid zero, blue = not


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


def models():
    return sorted(f[len("copying_"):-5] for f in os.listdir(RESULTS)
                  if f.startswith("copying_") and f.endswith(".json"))


def main():
    tags = models()
    if not tags:
        print("no results/copying_*.json -- run exp20_copying.py first")
        return
    data = {t: load(t) for t in tags}
    nz = [s for s in ORDER if S[s][1] != 0]

    fig = plt.figure(figsize=(14.5, 3.5 * len(tags)))
    gs = fig.add_gridspec(len(tags), 3, width_ratios=[1.5, 0.85, 1.0],
                          hspace=0.70, wspace=0.62)

    for row, tag in enumerate(tags):
        by = data[tag]

        # ---- Panel A: delta per scheme, coloured by zero-vs-nonzero loser ----
        ax = fig.add_subplot(gs[row, 0])
        w, xs = 0.26, np.arange(len(ORDER))
        for k, e in enumerate(ENVS):
            vals = [by[(e, s)]["delta"] for s in ORDER]
            cols = [ZERO_C if S[s][1] == 0 else NONZERO_C for s in ORDER]
            ax.bar(xs + (k - 1) * w, vals, w, color=cols,
                   edgecolor=INK, linewidth=0.4,
                   alpha=[1.0, 0.72, 0.48][k])
        for i, s in enumerate(ORDER):
            if S[s][1] == 0:
                ax.axvspan(i - 0.5, i + 0.5, color=ZERO_C, alpha=0.07, zorder=0)
        ax.set_xticks(xs)
        ax.set_xticklabels(ORDER, fontsize=8)
        ax.set_ylabel("$\\delta$  (nats)")
        ax.set_xlabel("payout scheme  (hacking side / obedient side)")
        ax.set_title(f"{tag} — A. the incentive is identical in every bar",
                     loc="left")
        despine(ax)
        zeroline(ax)
        ax.text(0.985, 0.98, "shaded: loser pays literally zero",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=ZERO_C)
        ax.text(0.985, 0.895, "bar shade = parity / magnitude / letter",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=MUTED)
        ax.set_ylim(top=max(by[(e, s)]["delta"] for e in ENVS
                            for s in ORDER) * 1.30)

        # ---- Panel B: zero-vs-nonzero, and the direction control ----------
        ax = fig.add_subplot(gs[row, 1])
        labs, vals, cols = [], [], []
        for e in ENVS:
            z = np.mean([by[(e, s)]["delta"] for s in ORDER if S[s][1] == 0])
            n = np.mean([by[(e, s)]["delta"] for s in nz])
            h = np.mean([by[(e, s)]["delta"] for s in nz if S[s][0] > S[s][1]])
            o = np.mean([by[(e, s)]["delta"] for s in nz if S[s][1] > S[s][0]])
            labs += [f"{e}: zero vs non-zero", f"{e}: hack- vs obey-favouring"]
            vals += [z - n, h - o]
            cols += [ZERO_C, MUTED]
        y = np.arange(len(vals))
        ax.barh(y, vals, color=cols, edgecolor=INK, linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(labs, fontsize=7)
        ax.invert_yaxis()
        lo, hi = min(vals + [0]), max(vals)
        ax.set_xlim(lo - 0.35 * (hi - lo) - 0.6, hi + 0.30 * (hi - lo))
        ax.set_xlabel("difference in $\\delta$  (nats)")
        ax.set_title("B. what actually moves $\\delta$", loc="left")
        despine(ax)
        zeroline(ax, horizontal=False)
        span = hi - lo
        for yi, v in zip(y, vals):
            ax.text(v + (0.03 if v >= 0 else -0.03) * span, yi, f"{v:+.2f}",
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=7, color=INK2)

        # ---- Panel C: the copying test, with its negative control ---------
        ax = fig.add_subplot(gs[row, 2])
        for e, c, mk in [("parity", CAT[0], "o"), ("letter", CAT[2], "s")]:
            x = np.array([S[s][2] for s in nz], float)
            yv = np.array([by[(e, s)]["beta"] for s in nz], float)
            slope, icept = np.polyfit(x, yv, 1)
            jit = 0.10 if e == "parity" else -0.10
            ax.scatter(x + jit, yv - yv.mean(), s=26, color=c, marker=mk,
                       zorder=3, edgecolor=INK, linewidth=0.4,
                       label=f"{e}: slope {slope:+.3f}")
            xx = np.linspace(-2.4, 2.4, 10)
            ax.plot(xx, slope * xx + (icept - yv.mean()), color=c, lw=1.6,
                    alpha=0.85, zorder=2)
        ax.set_xticks([-2, 0, 2])
        ax.set_xlabel("(# odd numerals) − (# even numerals) printed")
        ax.set_ylabel("$\\beta$, centred  (nats)")
        ax.set_title("C. copying test + numeral-free control", loc="left")
        ax.legend(loc="upper left", fontsize=7.5)
        despine(ax)
        zeroline(ax)

    fig.suptitle("exp20 — varying only how the reward is *written*, never what it pays",
                 x=0.008, ha="left", fontsize=11.5, fontweight="semibold",
                 color=INK, y=1.005)
    out = os.path.join(FIG, "fig8_copying.png")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
