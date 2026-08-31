r"""Figure for exp24: the causal test, done with the shared component removed.

A. Why exp16 could not have worked. The cosine between the two leave-one-out
   directions, by layer. Near zero early, then 0.25 to 0.73 through the middle of
   the network, which is where exp16 steered. Two directions that overlap that
   much cannot be compared as though they were alternatives.

B. Which layers are causally effective at all, measured by the ceiling: the
   direction fitted on the held-out environment itself. This is what the gate in
   panel C selects on, and it is a property of the site rather than of the
   transferred vectors.

C. The test. Effect on delta CLD relative to a norm-matched random direction,
   over layers whose ceiling exceeds +1 nat, using only alphas that keep
   answer-set mass at or above 0.9. Conflict-orthogonal-to-content retains most
   of the ceiling; content-orthogonal-to-conflict sits on the floor.

Writes figures/fig11_ortho.png.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from viz import CAT, INK, INK2, MUTED, BASELINE, despine, zeroline
from exp24_gated import load, eff, GATE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
TAGS = ["Qwen3-0.6B", "Qwen3.5-2B"]
SHOW = [("own_conflict", "ceiling\n(fitted on held-out env)", CAT[5]),
        ("conf_perp_cont", "conflict $\\perp$ content\n(the real test)", CAT[0]),
        ("bisector_shared", "bisector\n(the shared part)", CAT[3]),
        ("loo_content", "content, raw\n(what exp16 used)", CAT[1]),
        ("cont_perp_conf", "content $\\perp$ conflict\n(should be nothing)", CAT[7])]


def main():
    have = [t for t in TAGS if os.path.exists(
        os.path.join(ROOT, "results", f"steerortho_{t}.json"))]
    if not have:
        print("no results/steerortho_*.json")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4),
                           gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.35]))

    for k, tag in enumerate(have):
        d, rows, mm, base = load(tag)
        envs = sorted({r["held"] for r in rows})
        col = CAT[k]

        # A: cosine by layer
        by = {}
        for c in d["cosines"]:
            by.setdefault(c["layer"], []).append(c["cos"])
        xs = sorted(by)
        ax[0].plot([x / d["layers"][-1] for x in xs],
                   [np.mean(by[x]) for x in xs], "-o", color=col, lw=2.0,
                   label=tag, zorder=3)
        for x in xs:
            ax[0].scatter([x / d["layers"][-1]] * len(by[x]), by[x], s=12,
                          color=col, alpha=0.35, zorder=2)

        # B: ceiling by layer
        ceil = []
        for L in xs:
            v = [eff(rows, base, mm, L, "own_conflict", h) for h in envs]
            v = [x for x in v if x is not None]
            ceil.append(np.mean(v) if v else np.nan)
        ax[1].plot([x / d["layers"][-1] for x in xs], ceil, "-o", color=col,
                   lw=2.0, label=tag, zorder=3)

        # C: gated effect over random
        gated = [L for L, c in zip(xs, ceil) if c > GATE]
        res = {}
        for v, _, _ in SHOW + [("random", "", "")]:
            vals = [eff(rows, base, mm, L, v, h) for L in gated for h in envs]
            got = [x for x in vals if x is not None]
            res[v] = np.mean(got) if got else np.nan
        rnd = res["random"]
        w = 0.38
        pos = np.arange(len(SHOW)) + (k - 0.5) * w
        ax[2].barh(pos, [res[v] - rnd for v, _, _ in SHOW], w, color=col,
                   edgecolor=INK, linewidth=0.4,
                   label=f"{tag}  (layers {gated})")

    ax[0].axhline(0, color=BASELINE, lw=1.0)
    ax[0].set_xlabel("relative depth (layer / final layer)")
    ax[0].set_ylabel("cos(loo_conflict, loo_content)")
    ax[0].set_title("A. why exp16 could not work\n     the two directions overlap",
                    loc="left")
    ax[0].legend(fontsize=8)

    ax[1].axhline(GATE, ls="--", lw=1.2, color=MUTED)
    ax[1].text(0.99, GATE, f" gate: +{GATE:g} nat ", fontsize=7.5, color=MUTED,
               ha="right", va="bottom", transform=ax[1].get_yaxis_transform())
    ax[1].set_xlabel("relative depth (layer / final layer)")
    ax[1].set_ylabel("ceiling effect on $\\delta$ CLD (nats)")
    ax[1].set_title("B. which sites are causally effective\n     "
                    "the ceiling, by depth", loc="left")
    ax[1].legend(fontsize=8)
    zeroline(ax[1])

    ax[2].set_yticks(np.arange(len(SHOW)))
    ax[2].set_yticklabels([lab for _, lab, _ in SHOW], fontsize=8)
    ax[2].invert_yaxis()
    ax[2].set_xlabel("effect over a norm-matched random direction (nats)")
    ax[2].set_title("C. the test, with the shared component removed\n"
                    "     mass-guarded, ceiling-gated layers", loc="left")
    ax[2].legend(fontsize=7.5, loc="lower right")
    zeroline(ax[2], horizontal=False)

    for a in ax:
        despine(a)
    fig.suptitle("exp24: once the component shared with content is projected out, "
                 "the conflict direction still steers and content does not.",
                 x=0.006, ha="left", fontsize=12, fontweight="semibold",
                 color=INK, y=1.03)
    fig.tight_layout()
    out = os.path.join(FIG, "fig11_ortho.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
