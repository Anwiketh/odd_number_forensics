"""exp12: is there a task-general "go with the incentive" direction?

This is the decisive experiment of the study.

Construction. For each instruction polarity we take the difference-in-means of
the final-position residual stream between the conflict and aligned prompts:

    d_even = E[h | instr=even, incentive->odd ] - E[h | instr=even, incentive->even]
    d_odd  = E[h | instr=odd,  incentive->even] - E[h | instr=odd,  incentive->odd ]

Both differences correspond to the same *relational* change ("the incentive now
disagrees with the instruction") but to OPPOSITE *content* changes (toward odd
vs toward even). Therefore

    v_conflict = (d_even + d_odd) / 2     # relational part; content cancels
    v_content  = (d_even - d_odd) / 2     # content part; relational cancels

Predictions that distinguish the two accounts. Add each vector to prompts that
contain NO incentive block at all, in both polarities, and measure

    CLD = log P(disobedient side) - log P(instructed side).

  * If the model really has a polarity-invariant "follow the incentive over the
    instruction" feature, +v_conflict raises CLD in BOTH polarities.
  * If the apparent hack is a content feature ("odd"), then v_content raises
    CLD in the even arm and LOWERS it in the odd arm, and v_conflict does
    little.

We also test generalisation: does v_conflict, fitted on parity only, modulate
conflict resolution in magnitude / yesno / letter, environments that share no
content with parity?

Baselines (required, else the result is uninterpretable):
  - random direction of matched norm
  - v_content (the content-only direction)
  - a "mean-shift" control: the difference between two aligned prompts that
    differ only in surface phrasing (should do nothing)
"""
import sys, os, time, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from core import Runner, save
from metric import SlotMetric
import tasks as TK

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
BS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
TAG = MODEL.split("/")[-1]
R = Runner(MODEL)
LAYERS = list(range(R.n_layers))
T = TK.PARITY
M = SlotMetric(R, T)

# ---------------------------------------------------------------- fit set
# 4 instruction phrasings x 3 systems x 2 formats = 24 prompts per cell.
def parity_texts(instr_side, inc_side, frame="english"):
    out = []
    for iv in range(4):
        for sv in range(3):
            for fv in range(2):
                blk = (T.incentive(inc_side) if frame == "english"
                       else T.instruct2(inc_side))
                out.append(R.build_text(TK.SYSTEMS[sv],
                                        T.build(instr_side, blk, iv, fv)))
    return out


def parity_none(instr_side):
    out = []
    for iv in range(4):
        for sv in range(3):
            for fv in range(2):
                out.append(R.build_text(TK.SYSTEMS[sv],
                                        T.build(instr_side, "", iv, fv)))
    return out


print("caching residuals for direction fitting...", flush=True)
t0 = time.time()
H = {}
for instr_side in ["even", "odd"]:
    for align in ["aligned", "conflict"]:
        inc = instr_side if align == "aligned" else T.other(instr_side)
        H[(instr_side, align)] = R.batch_resid(
            parity_texts(instr_side, inc), LAYERS, bs=BS)   # [N, L, D]
        print(f"  {instr_side}/{align} {time.time()-t0:.0f}s", flush=True)

d_even = (H[("even", "conflict")] - H[("even", "aligned")]).mean(0)   # [L, D]
d_odd = (H[("odd", "conflict")] - H[("odd", "aligned")]).mean(0)
v_conflict = (d_even + d_odd) / 2
v_content = (d_even - d_odd) / 2

torch.manual_seed(0)
v_random = torch.randn_like(v_conflict)
v_random = v_random / v_random.norm(dim=-1, keepdim=True) * \
    v_conflict.norm(dim=-1, keepdim=True)

cos = torch.nn.functional.cosine_similarity(d_even, d_odd, dim=-1)  # [L]
print("\ncos(d_even, d_odd) by layer:",
      " ".join(f"{l}:{float(c):+.2f}" for l, c in enumerate(cos)), flush=True)
print("||v_conflict||/||v_content|| by layer:",
      " ".join(f"{l}:{float(a/b):.2f}" for l, (a, b) in
               enumerate(zip(v_conflict.norm(dim=-1), v_content.norm(dim=-1)))),
      flush=True)

# ---------------------------------------------------------- steering sweep
# Evaluate on NO-INCENTIVE prompts (nothing in context mentions a grader), so
# any movement is attributable to the injected vector alone.
eval_txt = {s: parity_none(s) for s in ["even", "odd"]}
base = {}
for s in ["even", "odd"]:
    L = R.batch_last_logits(eval_txt[s], bs=BS)
    base[s] = [M.ld(L[i], T.other(s)) for i in range(L.shape[0])]
    print(f"baseline CLD instr={s}: {st.mean(base[s]):+.3f}", flush=True)

VECS = {"conflict": v_conflict, "content": v_content, "random": v_random}
COEFFS = [-2.0, 1.0, 2.0, 4.0]
SWEEP_LAYERS = LAYERS[::2] + [LAYERS[-1]]     # every 2nd layer, keep the last
sweep = []
t0 = time.time()
for name, V in VECS.items():
    for l in SWEEP_LAYERS:
        # norm-matched coefficient: coeff is in units of ||v_conflict[l]||
        for c in COEFFS:
            row = {"vec": name, "layer": l, "coeff": c}
            for s in ["even", "odd"]:
                L = R.batch_with_add(eval_txt[s], [l], V[l], c, bs=BS)
                row[s] = st.mean(M.ld(L[i], T.other(s))
                                 for i in range(L.shape[0]))
                row[s + "_delta"] = row[s] - st.mean(base[s])
            row["invariant"] = min(row["even_delta"], row["odd_delta"])
            sweep.append(row)
        best = max((r for r in sweep if r["vec"] == name and r["layer"] == l),
                   key=lambda r: r["invariant"])
        print(f"  {name:9s} L{l:2d} best coeff={best['coeff']:+.1f} "
              f"d_even={best['even_delta']:+.2f} d_odd={best['odd_delta']:+.2f} "
              f"min={best['invariant']:+.2f} ({time.time()-t0:.0f}s)",
              flush=True)

save({"model": MODEL,
      "cos_d_even_d_odd": [float(x) for x in cos],
      "norm_conflict": [float(x) for x in v_conflict.norm(dim=-1)],
      "norm_content": [float(x) for x in v_content.norm(dim=-1)],
      "baseline": {s: st.mean(base[s]) for s in base},
      "sweep": sweep}, f"directions_{TAG}.json")
torch.save({"v_conflict": v_conflict, "v_content": v_content},
           os.path.join(os.path.dirname(os.path.dirname(
               os.path.abspath(__file__))), "results", f"vecs_{TAG}.pt"))
print("saved.", flush=True)
