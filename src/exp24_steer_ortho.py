"""exp24: the causal test done properly, with orthogonalisation.

WHY exp16 WAS NOT A FAIR TEST. exp16 fitted v_conflict = (d_s0 + d_s1)/2 and
v_content = (d_s0 - d_s1)/2 per environment, averaged each across three
environments, and steered the held-out fourth. It found that the *content*
direction steered about as well as the conflict direction (+2.50 against +2.96),
which reads as a failure of specificity. But the two averaged directions are not
guaranteed to be distinct: if both retain a large shared component, something
like "a scoring block is present and salient", then steering with either moves
that shared thing and the comparison says nothing about conflict versus content.

Four changes.

  1. ORTHOGONALISE. Steer with each direction projected off the other, and also
     with the bisector, which isolates the shared part. If conflict-orthogonal
     still steers and content-orthogonal does not, specificity holds after all.
     If only the bisector steers, the causal effect lives in the shared
     component, which is a sharper claim than exp16's.
  2. MASS GUARD. exp16's largest effects sit where answer-set mass is 0.56, i.e.
     where the perturbation has broken the model rather than changed its mind.
     Only alphas with mass >= MASS_MIN enter any average, and the count that
     survives is reported.
  3. LAYER SWEEP. exp16 fixed layer 16 because that is where the *geometry*
     peaks. Causal efficacy need not peak in the same place.
  4. GPU, and a second model. float32 throughout, which has been verified to
     reproduce the CPU numbers exactly.

The cosine between the two leave-one-out directions is printed first, because it
is the diagnostic for the hypothesis above and belongs in the write-up whichever
way the rest comes out.

Run: FORENSICS_DEVICE=cuda FORENSICS_DTYPE=float32 \
     python src/exp24_steer_ortho.py Qwen/Qwen3-0.6B 32
"""
import os
import sys
import time
import statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from core import Runner, save
from metric import SlotMetric
import tasks as TK

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
BS = int(sys.argv[2]) if len(sys.argv) > 2 else 32
TAG = MODEL.split("/")[-1]

NAMES = ["parity", "magnitude", "yesno", "letter"]
ALPHAS = [-0.35, -0.2, -0.1, 0.1, 0.2, 0.35]   # small only; big alphas break it
MASS_MIN = 0.90
LAYER_FRACS = [0.30, 0.45, 0.57, 0.70, 0.85]

R = Runner(MODEL)
MET = {n: SlotMetric(R, TK.ALL_TASKS[n]) for n in NAMES}
LAYERS = sorted({max(1, int(round(f * R.n_layers))) for f in LAYER_FRACS})
print(f"{TAG}: device={R.device} dtype={R.dtype} n_layers={R.n_layers}")
print(f"layers swept: {LAYERS}")


def texts(tname, instr_side, block_side):
    T = TK.ALL_TASKS[tname]
    return [R.build_text(TK.SYSTEMS[sv],
                         T.build(instr_side, T.incentive(block_side), iv, fv),
                         prefill=T.prefill)
            for iv in range(4) for sv in range(3) for fv in range(2)]


def unit(v):
    return v / v.norm()


def fit(layer):
    """Per-environment conflict and content directions at one layer."""
    VC, VT, norms = {}, {}, []
    for n in NAMES:
        T = TK.ALL_TASKS[n]
        ds = []
        for instr in T.sides:
            hk = T.other(instr)
            a = R.batch_resid(texts(n, instr, hk), [layer], bs=BS)[:, 0]
            b = R.batch_resid(texts(n, instr, instr), [layer], bs=BS)[:, 0]
            norms += [float(x) for x in a.norm(dim=-1)]
            ds.append((a - b).mean(0))
        VC[n] = (ds[0] + ds[1]) / 2
        VT[n] = (ds[0] - ds[1]) / 2
    return VC, VT, st.mean(norms)


def build_vectors(VC, VT, held):
    """The seven steering directions for one held-out environment."""
    others = [n for n in NAMES if n != held]
    u_conf = unit(torch.stack([unit(VC[n]) for n in others]).mean(0))
    u_cont = unit(torch.stack([unit(VT[n]) for n in others]).mean(0))
    cos = float((u_conf @ u_cont).clamp(-1, 1))

    # conflict with anything shared with content removed, and vice versa
    conf_perp = u_conf - (u_conf @ u_cont) * u_cont
    cont_perp = u_cont - (u_cont @ u_conf) * u_conf
    torch.manual_seed(1234 + NAMES.index(held))
    vecs = {
        "loo_conflict": u_conf,
        "loo_content": u_cont,
        "conf_perp_cont": unit(conf_perp),
        "cont_perp_conf": unit(cont_perp),
        "bisector_shared": unit(u_conf + u_cont),
        "own_conflict": unit(VC[held]),
        "random": unit(torch.randn_like(VC[held])),
    }
    return vecs, cos


def main():
    t0 = time.time()
    rows, cosines = [], []

    for layer in LAYERS:
        VC, VT, rnorm = fit(layer)
        print(f"\nlayer {layer}: mean ||resid|| = {rnorm:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)

        for held in NAMES:
            T, M = TK.ALL_TASKS[held], MET[held]
            vecs, cos = build_vectors(VC, VT, held)
            cosines.append({"layer": layer, "held": held, "cos": cos})
            print(f"  held={held:10s} cos(loo_conflict, loo_content) = {cos:+.3f}",
                  flush=True)

            ev, meta = [], []
            for s in T.sides:
                tt = texts(held, s, T.other(s))
                ev += tt
                meta += [s] * len(tt)

            base = R.batch_last_logits(ev, bs=BS)
            b0 = st.mean(st.mean(M.ld(base[i], T.other(s))
                                 for i, m in enumerate(meta) if m == s)
                         for s in T.sides)
            m0 = st.mean(M.mass(base[i]) for i in range(len(ev)))
            rows.append({"layer": layer, "held": held, "vec": "baseline",
                         "alpha": 0.0, "cld": b0, "mass": m0, "cos": cos})

            for vname, V in vecs.items():
                for a in ALPHAS:
                    L = R.batch_with_add(ev, [layer], V, a * rnorm, bs=BS)
                    cld = st.mean(st.mean(M.ld(L[i], T.other(s))
                                          for i, m in enumerate(meta) if m == s)
                                  for s in T.sides)
                    mm = st.mean(M.mass(L[i]) for i in range(len(ev)))
                    rows.append({"layer": layer, "held": held, "vec": vname,
                                 "alpha": a, "cld": cld, "mass": mm, "cos": cos})

    save({"model": MODEL, "device": str(R.device), "dtype": str(R.dtype),
          "layers": LAYERS, "alphas": ALPHAS, "mass_min": MASS_MIN,
          "cosines": cosines, "rows": rows}, f"steerortho_{TAG}.json")
    print(f"\nsaved. total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
