"""Generate every figure in the paper from results/*.json."""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from viz import CAT, DIVERGING, SEQ, INK, INK2, MUTED, BASELINE, despine, zeroline
from analyse import table, decompose, load_behaviour, SIDES, RESULTS

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "figures")
os.makedirs(FIG, exist_ok=True)
ENVS = ["parity", "magnitude", "yesno", "letter"]
FRAMES = ["code", "english", "instruct"]
PRETTY = {"parity": "parity\n(even vs odd)", "magnitude": "magnitude\n(<=4 vs >=5)",
          "yesno": "yesno\n(YES vs NO)", "letter": "letter\n(A vs B)"}
FR = {"code": "reward fn (code)", "english": "reward fn (prose)",
      "instruct": "plain instruction"}


def have(name):
    return os.path.exists(os.path.join(RESULTS, name))


def models():
    return sorted(f[len("behaviour_"):-5] for f in os.listdir(RESULTS)
                  if f.startswith("behaviour_"))


# ---------------------------------------------------------------- Figure 1
def fig1_arm_asymmetry():
    ms = models()
    fig, axes = plt.subplots(1, len(ms), figsize=(3.6 * len(ms), 3.4),
                             sharey=True, squeeze=False)
    for ax, tag in zip(axes[0], ms):
        rows = load_behaviour(tag)
        labels, a0, a1 = [], [], []
        for e in ENVS:
            for f in FRAMES:
                d = decompose(rows, e, f)
                if not d:
                    continue
                labels.append(f"{e[:4]}/{f[:4]}")
                a0.append(d["hack_rate_arm0"])
                a1.append(d["hack_rate_arm1"])
        x = np.arange(len(labels))
        w = 0.38
        ax.bar(x - w / 2 - 0.01, a0, w, color=CAT[0], label="arm A instructed",
               zorder=3)
        ax.bar(x + w / 2 + 0.01, a1, w, color=CAT[1], label="arm B instructed",
               zorder=3)
        for xi, (u, v) in enumerate(zip(a0, a1)):
            if abs(u - v) >= 0.5:
                ax.annotate("", xy=(xi, max(u, v) + 0.06),
                            xytext=(xi, min(u, v) + 0.02),
                            arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1))
                ax.text(xi, max(u, v) + 0.09, f"{abs(u-v):.2f}", ha="center",
                        fontsize=7, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_title(tag, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="x", visible=False)
        despine(ax)
    axes[0][0].set_ylabel("disobedience rate")
    axes[0][0].legend(loc="upper left", ncols=1)
    fig.suptitle("The same environment gives opposite answers depending on "
                 "which side you instruct", fontsize=11, y=1.04)
    fig.savefig(os.path.join(FIG, "fig1_arm_asymmetry.png"))
    plt.close(fig)
    print("fig1 ok")


# ---------------------------------------------------------------- Figure 2
def fig2_decomposition():
    ms = models()
    fig, axes = plt.subplots(1, len(ms), figsize=(3.6 * len(ms), 3.6),
                             sharey=True, squeeze=False)
    for ax, tag in zip(axes[0], ms):
        rows = load_behaviour(tag)
        labels, dl, dlo, dhi, bl, blo, bhi = [], [], [], [], [], [], []
        for e in ENVS:
            for f in FRAMES:
                d = decompose(rows, e, f)
                if not d:
                    continue
                labels.append(f"{e[:4]}/{f[:4]}")
                dl.append(d["delta"]); dlo.append(d["delta"] - d["delta_ci"][0])
                dhi.append(d["delta_ci"][1] - d["delta"])
                bl.append(d["beta"]); blo.append(d["beta"] - d["beta_ci"][0])
                bhi.append(d["beta_ci"][1] - d["beta"])
        y = np.arange(len(labels))
        ax.errorbar(dl, y - 0.16, xerr=[dlo, dhi], fmt="o", color=CAT[0],
                    ms=6, lw=1.6, capsize=2.5, label=r"$\delta$  incentive-following",
                    zorder=3)
        ax.errorbar(bl, y + 0.16, xerr=[blo, bhi], fmt="s", color=CAT[1],
                    ms=6, lw=1.6, capsize=2.5, label=r"$\beta$  content bias",
                    zorder=3)
        zeroline(ax, horizontal=False)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("effect on log-odds of disobeying (nats)")
        ax.set_title(tag, fontsize=9)
        ax.grid(axis="y", visible=False)
        despine(ax)
    axes[0][0].legend(loc="lower right")
    fig.suptitle("Decomposing the conflict effect into a polarity-invariant and "
                 "a polarity-specific part", fontsize=11, y=1.02)
    fig.savefig(os.path.join(FIG, "fig2_decomposition.png"))
    plt.close(fig)
    print("fig2 ok")


# ---------------------------------------------------------------- Figure 3
def fig3_comprehension():
    pts = []
    for tag in models():
        f = os.path.join(RESULTS, f"comprehension_{tag}.json")
        if not os.path.exists(f):
            continue
        comp = json.load(open(f, encoding="utf8"))
        rows = load_behaviour(tag)
        for style in ["code", "english"]:
            c = comp["probe2_evaluate"].get(style)
            if c is None:
                continue
            for e in ENVS:
                d = decompose(rows, e, style)
                if d:
                    pts.append((c[0], d["delta"], style, tag, e))
    if not pts:
        print("fig3 skipped (no comprehension data)")
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    mk = {"code": "o", "english": "s"}
    seen = set()
    for x, y, style, tag, e in pts:
        i = models().index(tag)
        lab = tag if tag not in seen else None
        seen.add(tag)
        ax.scatter(x, y, marker=mk[style], s=52, color=CAT[i],
                   edgecolor="white", linewidth=0.8, zorder=3, label=lab)
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    if len(xs) > 2:
        b, a = np.polyfit(xs, ys, 1)
        xr = np.linspace(xs.min(), xs.max(), 20)
        ax.plot(xr, a + b * xr, color=MUTED, lw=1.4, ls="--", zorder=2)
        r = np.corrcoef(xs, ys)[0, 1]
        ax.text(0.03, 0.96, f"r = {r:.2f}  (n = {len(xs)})", transform=ax.transAxes,
                va="top", fontsize=9, color=INK2)
    zeroline(ax); zeroline(ax, horizontal=False)
    ax.set_xlabel("reward-spec comprehension (nats, prior-free contrast)")
    ax.set_ylabel(r"incentive-following $\delta$ (nats)")
    ax.set_title("Measured hacking tracks whether the model can read the spec")
    ax.legend(title="circle = code spec, square = prose spec", loc="lower right",
              title_fontsize=7)
    despine(ax)
    fig.savefig(os.path.join(FIG, "fig3_comprehension.png"))
    plt.close(fig)
    print("fig3 ok")


# ---------------------------------------------------------------- Figure 4
def fig4_patch(tag="Qwen3-0.6B"):
    f = os.path.join(RESULTS, f"patch_{tag}.json")
    if not os.path.exists(f):
        print("fig4 skipped")
        return
    d = json.load(open(f, encoding="utf8"))
    arms = list(d["arms"])
    fig, axes = plt.subplots(len(arms), 1, figsize=(11, 3.1 * len(arms)),
                             squeeze=False)
    for ax, arm in zip(axes[:, 0], arms):
        A = d["arms"][arm]
        S, nL = A["S"], d["n_layers"]
        m = np.array(A["denoise"]).reshape(nL, S)
        base, src = A["base_aligned"], A["base_conflict"]
        norm = (m - base) / (src - base)
        v = float(np.abs(norm).max())
        im = ax.imshow(norm, aspect="auto", cmap=DIVERGING, vmin=-v, vmax=v,
                       origin="lower", interpolation="nearest")
        ax.set_ylabel(f"layer\n(instructed {arm})")
        toks = A["tokens"]
        step = max(1, S // 45)
        ax.set_xticks(range(0, S, step))
        ax.set_xticklabels([toks[i].strip()[:7] for i in range(0, S, step)],
                           rotation=90, fontsize=5.5)
        for p in A["diff_pos"]:
            ax.axvline(p, color=INK, lw=0.8, ls=":", alpha=0.7)
        ax.grid(False)
        fig.colorbar(im, ax=ax, pad=0.01,
                     label="normalised effect\n(0 = aligned, 1 = conflict)")
    fig.suptitle("Denoising activation patching: writing conflict-run residuals "
                 "into the aligned run", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_patch.png"))
    plt.close(fig)
    print("fig4 ok")


# ---------------------------------------------------------------- Figure 5
def fig5_geometry(tag="Qwen3-0.6B"):
    f = os.path.join(RESULTS, f"null_{tag}.json")
    if not os.path.exists(f):
        print("fig5 skipped (no null_*.json)")
        return
    d = json.load(open(f, encoding="utf8"))
    L = d["layers"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5), sharey=True)

    def mean_over(dd, pref=None):
        ks = [k for k in dd if pref is None or k.startswith(pref)]
        return [float(np.mean([dd[k][l] for k in ks])) for l in L]

    # panel 1: raw cross-environment cosine, conflict vs content
    ax = axes[0]
    for i, kind in enumerate(["conflict", "content"]):
        y = mean_over(d[kind]["cross_env"], "english:")
        ax.plot(L, y, color=CAT[i], label=f"$v_{{{kind}}}$")
    ax.axhline(d["chance"], color=MUTED, ls="--", lw=1.2)
    ax.axhline(-d["chance"], color=MUTED, ls="--", lw=1.2)
    ax.text(L[-1], d["chance"], "  chance", color=MUTED, fontsize=7, va="bottom",
            ha="right")
    zeroline(ax)
    ax.set_title("raw cross-environment cosine")
    ax.set_ylabel("cosine")

    # panel 2: split-half reliability (the ceiling)
    ax = axes[1]
    for i, kind in enumerate(["conflict", "content"]):
        ax.plot(L, mean_over(d[kind]["within"]), color=CAT[i],
                label=f"$v_{{{kind}}}$")
    zeroline(ax)
    ax.set_title("split-half reliability (ceiling)")

    # panel 3: attenuation-corrected rho
    ax = axes[2]
    for i, kind in enumerate(["conflict", "content"]):
        ax.plot(L, mean_over(d[kind]["rho_env"], "english:"), color=CAT[i],
                label=f"$v_{{{kind}}}$ across environments")
        ax.plot(L, mean_over(d[kind]["rho_frame"]), color=CAT[i], ls=":",
                label=f"$v_{{{kind}}}$ across framings")
    ax.axhline(1.0, color=MUTED, ls="--", lw=1.0)
    zeroline(ax)
    ax.set_title(r"attenuation-corrected $\rho$")
    ax.set_ylim(-1.2, 1.4)
    ax.legend(fontsize=7, loc="lower right")

    for ax in axes:
        ax.set_xlabel("layer")
        despine(ax)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("The polarity-invariant direction is shared across "
                 "environments and framings; the content direction is not",
                 fontsize=11, y=1.03)
    fig.savefig(os.path.join(FIG, "fig5_geometry.png"))
    plt.close(fig)
    print("fig5 ok")


# ---------------------------------------------------------------- Figure 6
def fig6_transfer(tag="Qwen3-0.6B"):
    f = os.path.join(RESULTS, f"steer_{tag}.json")
    if not os.path.exists(f):
        print("fig6 skipped (no steer_*.json)")
        return
    d = json.load(open(f, encoding="utf8"))
    rows = d["rows"]
    held = sorted(set(r["held"] for r in rows), key=ENVS.index)
    fig, axes = plt.subplots(2, len(held), figsize=(3.0 * len(held), 5.4),
                             sharey="row", squeeze=False,
                             gridspec_kw={"height_ratios": [2.4, 1]})
    styles = {"loo_conflict": (CAT[0], "-", "o", "held-out $v_{conflict}$"),
              "own_conflict": (CAT[2], ":", "D", "own $v_{conflict}$"),
              "loo_content": (CAT[1], "-", "s", "held-out $v_{content}$"),
              "random": (MUTED, "--", "^", "random (damage floor)")}
    for j, h in zip(range(len(held)), held):
        ax, ax2 = axes[0][j], axes[1][j]
        b = [r for r in rows if r["held"] == h
             and r["vec"] == "baseline"][0]["cld_mean"]
        for v, (col, ls, mk, lab) in styles.items():
            sub = sorted((r for r in rows if r["held"] == h and r["vec"] == v),
                         key=lambda r: r["alpha"])
            if not sub:
                continue
            xs = [r["alpha"] for r in sub] + [0.0]
            ys = [r["cld_mean"] - b for r in sub] + [0.0]
            o = np.argsort(xs)
            ax.plot(np.array(xs)[o], np.array(ys)[o], color=col, ls=ls,
                    marker=mk, ms=4.5, label=lab)
            ax2.plot([r["alpha"] for r in sub], [r["mass"] for r in sub],
                     color=col, ls=ls, marker=mk, ms=3.5)
        zeroline(ax); zeroline(ax, horizontal=False)
        ax.set_title(f"held out: {h}", fontsize=9)
        ax2.set_xlabel(r"$\alpha$  (units of mean $\Vert h\Vert$)")
        ax2.set_ylim(0, 1.05)
        despine(ax); despine(ax2)
    axes[0][0].set_ylabel(r"$\Delta$ CLD (nats)")
    axes[1][0].set_ylabel("answer-set mass")
    axes[0][-1].legend(loc="upper left", fontsize=7)
    fig.suptitle(f"Leave-one-environment-out steering at layer {d['layer']}. "
                 f"Bottom row: has the perturbation broken the model?",
                 fontsize=11, y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_transfer.png"))
    plt.close(fig)
    print("fig6 ok")


# ---------------------------------------------------------------- Figure 7
def fig7_cot(tag="Qwen3-0.6B"):
    f = os.path.join(RESULTS, f"cot_{tag}.json")
    if not os.path.exists(f):
        print("fig7 skipped")
        return
    d = json.load(open(f, encoding="utf8"))["rows"]
    tasks = sorted(set(r["task"] for r in d))
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.4),
                             squeeze=False)
    for ax, t in zip(axes[0], tasks):
        sides = sorted(set(r["instr_side"] for r in d if r["task"] == t))
        for i, s in enumerate(sides):
            sub = [r for r in d if r["task"] == t and r["instr_side"] == s]
            for r in sub:
                x = np.linspace(0, 1, len(r["traj"]))
                ax.plot(x, r["traj"], color=CAT[i], alpha=0.22, lw=1.0)
            n = max(len(r["traj"]) for r in sub)
            grid = np.linspace(0, 1, 25)
            M = np.stack([np.interp(grid, np.linspace(0, 1, len(r["traj"])),
                                    r["traj"]) for r in sub])
            ax.plot(grid, M.mean(0), color=CAT[i], lw=2.6,
                    label=f"instructed {s} (n={len(sub)})")
        zeroline(ax)
        ax.set_xlabel("fraction of the CoT revealed")
        ax.set_title(t, fontsize=9)
        ax.legend(loc="best", fontsize=7)
        despine(ax)
    axes[0][0].set_ylabel("CLD if forced to answer here (nats)")
    fig.suptitle("Commitment trajectories: how much of the decision is made "
                 "before the reasoning", fontsize=11, y=1.03)
    fig.savefig(os.path.join(FIG, "fig7_cot.png"))
    plt.close(fig)
    print("fig7 ok")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7"]
    fns = {"1": fig1_arm_asymmetry, "2": fig2_decomposition,
           "3": fig3_comprehension, "4": fig4_patch, "5": fig5_geometry,
           "6": fig6_transfer, "7": fig7_cot}
    for w in which:
        try:
            fns[w]()
        except Exception as e:
            print(f"fig{w} FAILED: {type(e).__name__}: {e}")
