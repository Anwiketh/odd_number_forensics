"""exp15: how shared is the conflict direction, and is it about *rewards*?

A raw cross-environment cosine is meaningless without references.

  chance      -- two random unit vectors in d dims have E|cos| ~
                 sqrt(2/(pi*d)); for d=1024 that is 0.025.
  reliability -- our estimates are noisy, which CAPS any cosine they can
                 reach. Fit the direction twice within the same environment on
                 disjoint halves of the surface variants; that cosine,
                 r_within, is the ceiling. We report the attenuation-corrected

                     rho(A,B) = r_cross(A,B) / sqrt(r_within(A) r_within(B))

We also test an alternative explanation that matters a lot for how the result
should be read. v_conflict might encode

  (a) "the context contains a *stated incentive* pointing away from the
      instruction"  -- a reward-specific representation, or
  (b) "the context contains *any* pointer away from the instruction" /
      "I am about to disobey"  -- a generic conflict axis.

To distinguish them we fit the direction twice more, once from the `english`
reward frame and once from the plain `instruct` frame (a second instruction,
no points, no grader, no stake). If those two directions are as aligned as two
split-halves of the same frame, there is no separate reward representation
here: a stated reward is just another conflicting pointer.
"""
import sys, os, time, itertools, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from core import Runner, save
import tasks as TK

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
BS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
TAG = MODEL.split("/")[-1]
R = Runner(MODEL)
LAYERS = list(range(R.n_layers))
NAMES = ["parity", "magnitude", "yesno", "letter"]
HALVES = {"h1": [0, 1], "h2": [2, 3]}
FRAMES = ["english", "instruct"]


def blk(T, frame, side):
    return T.incentive(side) if frame == "english" else T.instruct2(side)


def texts(tname, frame, instr_side, block_side, ivs):
    T = TK.ALL_TASKS[tname]
    return [R.build_text(TK.SYSTEMS[sv],
                         T.build(instr_side, blk(T, frame, block_side), iv, fv),
                         prefill=T.prefill)
            for iv in ivs for sv in range(3) for fv in range(2)]


print("fitting split-half directions...", flush=True)
VC, VT = {}, {}
t0 = time.time()
for n in NAMES:
    T = TK.ALL_TASKS[n]
    for frame in FRAMES:
        for hname, ivs in HALVES.items():
            ds = []
            for instr_side in T.sides:
                hk = T.other(instr_side)
                a = R.batch_resid(texts(n, frame, instr_side, hk, ivs), LAYERS, bs=BS)
                b = R.batch_resid(texts(n, frame, instr_side, instr_side, ivs),
                                  LAYERS, bs=BS)
                ds.append((a - b).mean(0))
            VC[(n, frame, hname)] = (ds[0] + ds[1]) / 2
            VT[(n, frame, hname)] = (ds[0] - ds[1]) / 2
        print(f"  {n}/{frame} ({time.time()-t0:.0f}s)", flush=True)

cs = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1)
F = lambda t: [float(x) for x in t]
out = {"model": MODEL, "d_model": R.d_model,
       "chance": math.sqrt(2 / (math.pi * R.d_model)), "layers": LAYERS}


def analyse(V, kind):
    res = {}
    # reliability: split-half within (env, frame)
    within = {f"{n}|{fr}": F(cs(V[(n, fr, 'h1')], V[(n, fr, 'h2')]))
              for n in NAMES for fr in FRAMES}
    res["within"] = within

    def crosspair(ka, kb):
        na, fa = ka; nb, fb = kb
        vals = torch.stack([cs(V[(na, fa, ha)], V[(nb, fb, hb)])
                            for ha in HALVES for hb in HALVES]).mean(0)
        den = [math.sqrt(max(within[f'{na}|{fa}'][l], 1e-6) *
                         max(within[f'{nb}|{fb}'][l], 1e-6)) for l in LAYERS]
        return F(vals), [float(vals[l]) / den[l] for l in LAYERS]

    # cross-environment, same frame
    res["cross_env"], res["rho_env"] = {}, {}
    for fr in FRAMES:
        for a, b in itertools.combinations(NAMES, 2):
            c, r = crosspair((a, fr), (b, fr))
            res["cross_env"][f"{fr}:{a}|{b}"] = c
            res["rho_env"][f"{fr}:{a}|{b}"] = r
    # cross-frame, same environment
    res["cross_frame"], res["rho_frame"] = {}, {}
    for n in NAMES:
        c, r = crosspair((n, "english"), (n, "instruct"))
        res["cross_frame"][n] = c
        res["rho_frame"][n] = r
    out[kind] = res

    def rowstat(d, l):
        v = [d[k][l] for k in d]
        return sum(v) / len(v)

    print(f"\n=== {kind} ===  (layer: mean within | mean cross-env rho | "
          f"mean cross-frame rho)")
    for l in LAYERS:
        print(f"  L{l:2d}  {rowstat(within, l):+5.2f}   "
              f"{rowstat(res['rho_env'], l):+5.2f}   "
              f"{rowstat(res['rho_frame'], l):+5.2f}")


analyse(VC, "conflict")
analyse(VT, "content")
print(f"\nchance |cos| for d={R.d_model}: {out['chance']:.3f}")
save(out, f"null_{TAG}.json")
print("saved.", flush=True)
