"""exp16: leave-one-environment-out steering, done properly.

exp13's first pass compared raw difference-in-means vectors at shared
coefficients. That is not a fair comparison: the vectors have different norms,
so "coefficient -6" was a much larger perturbation for one vector than
another, and the random control was matched to only one of them. Two fixes:

  1. every steering vector is unit-normalised;
  2. the coefficient is expressed in units of the model's own mean residual
     norm at that layer, so `alpha = 0.5` means "add half a typical residual"
     for every vector alike.

We also report the answer-set probability mass under steering. Steering that
pushes mass off the answer set has broken the model rather than changed its
mind, and any CLD read there is meaningless. The random direction at matched
alpha is the reference for "how much does a perturbation of this size damage
the model regardless of direction".

Vectors compared, all fitted at the same layer:
  loo_conflict  polarity-invariant direction, fitted on the OTHER three
                environments  -- the transfer test
  own_conflict  fitted on the held-out environment itself -- the ceiling
  loo_content   polarity-specific direction from the other three -- should
                not transfer
  random        norm-matched Gaussian -- the damage floor
"""
import sys, os, time, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from core import Runner, save, RESULTS
from metric import SlotMetric
import tasks as TK

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
BS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
LAYER = int(sys.argv[3]) if len(sys.argv) > 3 else 16
TAG = MODEL.split("/")[-1]
R = Runner(MODEL)
NAMES = ["parity", "magnitude", "yesno", "letter"]
MET = {n: SlotMetric(R, TK.ALL_TASKS[n]) for n in NAMES}
ALPHAS = [-1.0, -0.6, -0.35, -0.2, 0.0, 0.2, 0.35, 0.6, 1.0]


def texts(tname, instr_side, block_side):
    T = TK.ALL_TASKS[tname]
    return [R.build_text(TK.SYSTEMS[sv],
                         T.build(instr_side, T.incentive(block_side), iv, fv),
                         prefill=T.prefill)
            for iv in range(4) for sv in range(3) for fv in range(2)]


# ---- refit the directions at LAYER (cheap: one layer only) ---------------
print(f"fitting directions at layer {LAYER}...", flush=True)
VC, VT = {}, {}
t0 = time.time()
resid_norms = []
for n in NAMES:
    T = TK.ALL_TASKS[n]
    ds = []
    for instr_side in T.sides:
        hk = T.other(instr_side)
        a = R.batch_resid(texts(n, instr_side, hk), [LAYER], bs=BS)[:, 0]
        b = R.batch_resid(texts(n, instr_side, instr_side), [LAYER], bs=BS)[:, 0]
        resid_norms += [float(x) for x in a.norm(dim=-1)]
        ds.append((a - b).mean(0))
    VC[n] = (ds[0] + ds[1]) / 2
    VT[n] = (ds[0] - ds[1]) / 2
    print(f"  {n} ||v_conf||={VC[n].norm():.2f} ||v_cont||={VT[n].norm():.2f} "
          f"({time.time()-t0:.0f}s)", flush=True)

RNORM = st.mean(resid_norms)
print(f"mean ||resid|| at layer {LAYER} = {RNORM:.2f}", flush=True)
unit = lambda v: v / v.norm()

rows = []
for held in NAMES:
    others = [n for n in NAMES if n != held]
    T = TK.ALL_TASKS[held]
    torch.manual_seed(1234 + NAMES.index(held))
    vecs = {
        "loo_conflict": unit(torch.stack([unit(VC[n]) for n in others]).mean(0)),
        "own_conflict": unit(VC[held]),
        "loo_content": unit(torch.stack([unit(VT[n]) for n in others]).mean(0)),
        "random": unit(torch.randn_like(VC[held])),
    }
    ev, meta = [], []
    for s in T.sides:
        tt = texts(held, s, T.other(s))
        ev += tt
        meta += [s] * len(tt)

    base_L = R.batch_last_logits(ev, bs=BS)
    per0 = {s: st.mean(MET[held].ld(base_L[i], T.other(s))
                       for i, m in enumerate(meta) if m == s) for s in T.sides}
    b0 = st.mean(per0.values())
    m0 = st.mean(MET[held].mass(base_L[i]) for i in range(len(ev)))
    rows.append({"held": held, "vec": "baseline", "alpha": 0.0, "layer": LAYER,
                 "cld_mean": b0, "mass": m0,
                 **{f"cld_{s}": per0[s] for s in T.sides}})
    print(f"\n  held={held} baseline CLD={b0:+.3f} mass={m0:.3f}", flush=True)

    for vname, V in vecs.items():
        line = f"   {vname:14s}"
        for a in ALPHAS:
            if a == 0.0:
                continue
            L = R.batch_with_add(ev, [LAYER], V, a * RNORM, bs=BS)
            per = {s: st.mean(MET[held].ld(L[i], T.other(s))
                              for i, m in enumerate(meta) if m == s)
                   for s in T.sides}
            mm = st.mean(MET[held].mass(L[i]) for i in range(len(ev)))
            rows.append({"held": held, "vec": vname, "alpha": a,
                         "layer": LAYER, "cld_mean": st.mean(per.values()),
                         "mass": mm, **{f"cld_{s}": per[s] for s in T.sides}})
            line += f" {a:+.2f}:{st.mean(per.values())-b0:+5.2f}"
        print(line, flush=True)

save({"model": MODEL, "layer": LAYER, "resid_norm": RNORM,
      "alphas": ALPHAS, "rows": rows}, f"steer_{TAG}.json")
print("saved.", flush=True)
