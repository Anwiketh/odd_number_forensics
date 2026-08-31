"""exp13: does the conflict direction generalise across environments?

Two measurements.

(1) GEOMETRY. Fit (v_conflict, v_content) independently in each of the four
    conflict environments and compute the cross-environment cosine matrix at
    every layer. If parity/magnitude/yesno/letter all place "the incentive
    disagrees with the instruction" along the same direction, the off-diagonal
    cosines for v_conflict are large while those for v_content are not
    (v_content is by construction about task-specific content).

(2) CAUSALITY. Leave-one-environment-out. Fit v_conflict on three
    environments, steer with it in the held-out fourth, and ask whether it
    moves conflict resolution there. Compare against v_content fitted the same
    way, a norm-matched random direction, and the no-steer baseline.

    Suppression (-v) is the safety-relevant direction: can we turn OFF
    incentive-following in an environment the vector never saw?
"""
import sys, os, time, statistics as st, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from core import Runner, save, RESULTS
from metric import SlotMetric
import tasks as TK

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
BS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
STEER_LAYER = int(sys.argv[3]) if len(sys.argv) > 3 else -1   # -1 = pick later
TAG = MODEL.split("/")[-1]
R = Runner(MODEL)
LAYERS = list(range(R.n_layers))
NAMES = ["parity", "magnitude", "yesno", "letter"]
MET = {n: SlotMetric(R, TK.ALL_TASKS[n]) for n in NAMES}
NSYS, NFMT = 3, 2


def texts(tname, instr_side, block_side, frame):
    T = TK.ALL_TASKS[tname]
    out = []
    for iv in range(4):
        for sv in range(NSYS):
            for fv in range(NFMT):
                if frame == "none":
                    blk = ""
                elif frame == "english":
                    blk = T.incentive(block_side)
                else:
                    blk = T.instruct2(block_side)
                out.append(R.build_text(TK.SYSTEMS[sv],
                                        T.build(instr_side, blk, iv, fv)))
    return out


# ------------------------------------------------------------ (1) geometry
print("fitting per-environment directions...", flush=True)
VC, VT = {}, {}
t0 = time.time()
for n in NAMES:
    T = TK.ALL_TASKS[n]
    ds = []
    for instr_side in T.sides:
        hack = T.other(instr_side)
        h_conf = R.batch_resid(texts(n, instr_side, hack, "english"), LAYERS, bs=BS)
        h_algn = R.batch_resid(texts(n, instr_side, instr_side, "english"), LAYERS, bs=BS)
        ds.append((h_conf - h_algn).mean(0))          # [L, D]
    VC[n] = (ds[0] + ds[1]) / 2
    VT[n] = (ds[0] - ds[1]) / 2
    print(f"  {n} ({time.time()-t0:.0f}s)", flush=True)

cosmat = {"conflict": {}, "content": {}}
for kind, V in [("conflict", VC), ("content", VT)]:
    for a, b in itertools.combinations(NAMES, 2):
        c = torch.nn.functional.cosine_similarity(V[a], V[b], dim=-1)
        cosmat[kind][f"{a}|{b}"] = [float(x) for x in c]
    print(f"\n{kind} cross-environment cosine, by layer:")
    for k, v in cosmat[kind].items():
        print(f"  {k:22s} " + " ".join(f"{x:+.2f}" for x in v), flush=True)

# also: how aligned is v_conflict with v_content within an environment?
within = {n: [float(x) for x in torch.nn.functional.cosine_similarity(
    VC[n], VT[n], dim=-1)] for n in NAMES}

# ---------------------------------------------------------- (2) causality
# choose the steering layer as the one maximising the mean off-diagonal
# conflict cosine, unless supplied on the command line.
mean_off = [st.mean(cosmat["conflict"][k][l] for k in cosmat["conflict"])
            for l in LAYERS]
if STEER_LAYER < 0:
    STEER_LAYER = max(LAYERS, key=lambda l: mean_off[l])
print(f"\nsteering layer = {STEER_LAYER} "
      f"(mean off-diag conflict cos = {mean_off[STEER_LAYER]:+.3f})", flush=True)

COEFFS = [-6, -4, -2, -1, 0, 1, 2, 4]
rows = []
t0 = time.time()
for held in NAMES:
    others = [n for n in NAMES if n != held]
    T = TK.ALL_TASKS[held]
    loo_conf = torch.stack([VC[n][STEER_LAYER] for n in others]).mean(0)
    loo_cont = torch.stack([VT[n][STEER_LAYER] for n in others]).mean(0)
    torch.manual_seed(hash(held) % 10000)
    rnd = torch.randn_like(loo_conf)
    rnd = rnd / rnd.norm() * loo_conf.norm()
    vecs = {"loo_conflict": loo_conf, "loo_content": loo_cont, "random": rnd,
            "own_conflict": VC[held][STEER_LAYER]}
    # evaluation prompts: the CONFLICT condition of the held-out environment
    ev, meta = [], []
    for instr_side in T.sides:
        tt = texts(held, instr_side, T.other(instr_side), "english")
        ev += tt
        meta += [instr_side] * len(tt)
    for vname, V in vecs.items():
        for c in COEFFS:
            if c == 0 and vname != "loo_conflict":
                continue
            L = (R.batch_last_logits(ev, bs=BS) if c == 0 else
                 R.batch_with_add(ev, [STEER_LAYER], V, c, bs=BS))
            per = {}
            for s in T.sides:
                idx = [i for i, m in enumerate(meta) if m == s]
                per[s] = st.mean(MET[held].ld(L[i], T.other(s)) for i in idx)
            rows.append({"held": held, "vec": vname, "coeff": c,
                         "layer": STEER_LAYER,
                         "cld_mean": st.mean(per.values()), **{
                             f"cld_{s}": per[s] for s in T.sides}})
    b = [r for r in rows if r["held"] == held and r["coeff"] == 0][0]
    print(f"\n  held={held} baseline CLD={b['cld_mean']:+.3f}", flush=True)
    for vname in vecs:
        line = "   " + f"{vname:14s}"
        for c in COEFFS:
            m = [r for r in rows if r["held"] == held and r["vec"] == vname
                 and r["coeff"] == c]
            line += f" {c:+d}:{m[0]['cld_mean']-b['cld_mean']:+6.2f}" if m else ""
        print(line, flush=True)

save({"model": MODEL, "steer_layer": STEER_LAYER, "cos": cosmat,
      "within_conflict_content_cos": within, "mean_offdiag": mean_off,
      "steer": rows}, f"transfer_{TAG}.json")
torch.save({"VC": VC, "VT": VT}, os.path.join(RESULTS, f"dirs_{TAG}.pt"))
print("saved.", flush=True)
