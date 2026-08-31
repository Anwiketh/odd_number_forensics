"""Analyse exp24: does the conflict direction steer once the shared part is removed?

effect(vec) = mean(delta CLD at alpha > 0) - mean(delta CLD at alpha < 0),
averaged over held-out environments, using ONLY alphas whose answer-set mass is
at least MASS_MIN. Points where the perturbation has pushed mass off the answer
set are not evidence about anything and are excluded, with the count reported.
"""
import json, os, sys, io
import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")
ORDER = ["own_conflict", "loo_conflict", "conf_perp_cont", "bisector_shared",
         "loo_content", "cont_perp_conf", "random"]
LABEL = {"own_conflict": "own_conflict (ceiling)",
         "loo_conflict": "loo_conflict (raw, as exp16)",
         "conf_perp_cont": "conflict PERP content  <-- the real test",
         "bisector_shared": "bisector (the shared part)",
         "loo_content": "loo_content (raw control)",
         "cont_perp_conf": "content PERP conflict",
         "random": "random (damage floor)"}


def main(tag):
    d = json.load(open(os.path.join(RES, f"steerortho_{tag}.json"), encoding="utf-8"))
    rows, mass_min = d["rows"], d["mass_min"]
    o = io.StringIO()
    o.write(f"=== {tag} ===  mass guard >= {mass_min}\n")
    cs = [c["cos"] for c in d["cosines"]]
    o.write(f"cos(loo_conflict, loo_content): min {min(cs):+.3f} "
            f"max {max(cs):+.3f}\n\n")

    base = {(r["layer"], r["held"]): r["cld"] for r in rows if r["vec"] == "baseline"}
    best = {}
    for layer in d["layers"]:
        o.write(f"layer {layer:3d}   " + "".join(f"{v[:13]:>15s}" for v in ORDER) + "\n")
        cells, drops = [], []
        for v in ORDER:
            eff, nkept, ntot = [], 0, 0
            for held in sorted({r["held"] for r in rows}):
                pos = [r for r in rows if r["layer"] == layer and r["held"] == held
                       and r["vec"] == v and r["alpha"] > 0]
                neg = [r for r in rows if r["layer"] == layer and r["held"] == held
                       and r["vec"] == v and r["alpha"] < 0]
                ntot += len(pos) + len(neg)
                pk = [r["cld"] - base[(layer, held)] for r in pos if r["mass"] >= mass_min]
                nk = [r["cld"] - base[(layer, held)] for r in neg if r["mass"] >= mass_min]
                nkept += len(pk) + len(nk)
                if pk and nk:
                    eff.append(np.mean(pk) - np.mean(nk))
            cells.append(np.mean(eff) if eff else float("nan"))
            drops.append(ntot - nkept)
            best.setdefault(v, []).append(cells[-1])
        o.write("  effect   " + "".join(f"{c:+15.2f}" for c in cells) + "\n")
        o.write("  dropped  " + "".join(f"{n:>15d}" for n in drops) + "\n\n")

    o.write("Mean over layers:\n")
    for v in ORDER:
        vals = [x for x in best[v] if not np.isnan(x)]
        o.write(f"  {LABEL[v]:36s} {np.mean(vals):+6.2f}\n")
    rnd = np.nanmean(best["random"])
    o.write("\nAgainst the damage floor:\n")
    for v in ORDER:
        if v == "random":
            continue
        vals = [x for x in best[v] if not np.isnan(x)]
        o.write(f"  {LABEL[v]:36s} {np.mean(vals) - rnd:+6.2f} over random\n")
    sys.stdout.buffer.write(o.getvalue().encode("utf-8", "replace"))


if __name__ == "__main__":
    for t in (sys.argv[1:] or ["Qwen3-0.6B"]):
        main(t)
