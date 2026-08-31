"""exp24, gated: does the conflict direction steer once the shared part is gone?

Only layers where the CEILING works are used. The ceiling is own_conflict, the
direction fitted on the held-out environment itself; if that does not move CLD,
the site is not causally effective for any direction and tells us nothing about
transfer. The gate is therefore a statement about the site, chosen without
reference to the transferred vectors.

Only alphas whose answer-set mass is >= mass_min contribute. A cell with no
surviving alpha is reported as unmeasurable rather than as zero.
"""
import json, os, sys, io
import numpy as np

RES = "results"
ORDER = ["own_conflict", "loo_conflict", "conf_perp_cont", "bisector_shared",
         "loo_content", "cont_perp_conf", "random"]
GATE = 1.0


def load(tag):
    d = json.load(open(f"{RES}/steerortho_{tag}.json", encoding="utf-8"))
    rows, mm = d["rows"], d["mass_min"]
    base = {(r["layer"], r["held"]): r["cld"] for r in rows if r["vec"] == "baseline"}
    return d, rows, mm, base


def eff(rows, base, mm, layer, v, held):
    pos = [r for r in rows if r["layer"] == layer and r["held"] == held
           and r["vec"] == v and r["alpha"] > 0 and r["mass"] >= mm]
    neg = [r for r in rows if r["layer"] == layer and r["held"] == held
           and r["vec"] == v and r["alpha"] < 0 and r["mass"] >= mm]
    if not pos or not neg:
        return None
    return (np.mean([r["cld"] - base[(layer, held)] for r in pos])
            - np.mean([r["cld"] - base[(layer, held)] for r in neg]))


def report(tag, o):
    d, rows, mm, base = load(tag)
    envs = sorted({r["held"] for r in rows})
    cs = [c["cos"] for c in d["cosines"]]
    o.write(f"=== {tag} ===\n")
    o.write(f"cos(loo_conflict, loo_content) ranges {min(cs):+.2f} to {max(cs):+.2f}\n")

    gated = []
    for L in d["layers"]:
        vals = [eff(rows, base, mm, L, "own_conflict", h) for h in envs]
        vals = [v for v in vals if v is not None]
        if vals and np.mean(vals) > GATE:
            gated.append(L)
    o.write(f"layers where the ceiling exceeds +{GATE:.1f}: {gated} of {d['layers']}\n\n")
    if not gated:
        o.write("  no causally effective layer found; nothing to test.\n\n")
        return None

    o.write(f"{'vector':24s} {'effect':>8s} {'vs random':>10s} {'cells':>7s}\n")
    res = {}
    for v in ORDER:
        vals = [eff(rows, base, mm, L, v, h) for L in gated for h in envs]
        got = [x for x in vals if x is not None]
        res[v] = (np.mean(got) if got else float("nan"), len(got), len(vals))
    rnd = res["random"][0]
    for v in ORDER:
        m, n, tot = res[v]
        o.write(f"  {v:22s} {m:+8.2f} {m - rnd:+10.2f} {n:>4d}/{tot}\n")

    o.write("\nper held-out environment (gated layers pooled)\n")
    o.write(f"{'env':12s}{'ceiling':>10s}{'conf_perp':>11s}{'cont_perp':>11s}{'random':>9s}\n")
    per = {}
    for h in envs:
        cells = []
        for v in ["own_conflict", "conf_perp_cont", "cont_perp_conf", "random"]:
            got = [x for x in (eff(rows, base, mm, L, v, h) for L in gated) if x is not None]
            cells.append(np.mean(got) if got else None)
        per[h] = cells
        o.write(f"  {h:10s}" + "".join(
            ("      n/a" if c is None else f"{c:+11.2f}") for c in cells) + "\n")
    o.write("\n")
    return res, per, gated


def main():
    o = io.StringIO()
    out = {}
    for tag in ["Qwen3-0.6B", "Qwen3.5-2B"]:
        if os.path.exists(f"{RES}/steerortho_{tag}.json"):
            out[tag] = report(tag, o)
    o.write("=" * 62 + "\nVERDICT\n" + "=" * 62 + "\n")
    for tag, r in out.items():
        if not r:
            continue
        res, per, gated = r
        cp, cpn, _ = res["conf_perp_cont"]
        tp, _, _ = res["cont_perp_conf"]
        rnd = res["random"][0]
        ceil = res["own_conflict"][0]
        holds = (cp - rnd) > 0.5 and (tp - rnd) < 0.5
        o.write(f"{tag}: conflict_perp {cp - rnd:+.2f} over random, "
                f"content_perp {tp - rnd:+.2f} over random, ceiling {ceil - rnd:+.2f}\n")
        o.write(f"  specificity after orthogonalisation: "
                f"{'HOLDS' if holds else 'DOES NOT HOLD'}\n")
        ok = [h for h, c in per.items()
              if c[1] is not None and c[3] is not None and c[1] - c[3] > 0.5]
        na = [h for h, c in per.items() if c[1] is None]
        o.write(f"  environments where it holds: {ok or 'none'}"
                + (f"; unmeasurable: {na}" if na else "") + "\n")
    sys.stdout.buffer.write(o.getvalue().encode("utf-8", "replace"))


if __name__ == "__main__":
    main()
