"""exp01: the behavioural battery.

For every condition x every surface variant, one forward pass, record PLD and
the full digit distribution. Output: results/battery_<model>.json
"""
import sys, os, time, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *
from conditions import CONDITIONS, build_user, all_variants

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
TAG = MODEL.split("/")[-1]

R = Runner(MODEL)
variants = list(all_variants())
print(f"{TAG}: {len(CONDITIONS)} conditions x {len(variants)} variants = "
      f"{len(CONDITIONS)*len(variants)} forwards")

rows = []
t0 = time.time()
for ci, cond in enumerate(CONDITIONS):
    fam = CONDITIONS[cond][0]
    for vi, (task, system, fmt) in enumerate(variants):
        user = build_user(cond, task, fmt)
        ids = R.build(system, user)
        r = R.run(ids)
        rows.append({
            "cond": cond, "family": fam, "variant": vi,
            "task": task, "system": system, "fmt": fmt,
            "pld": r["pld"], "digit_mass": r["digit_mass"],
            "argmax": r["argmax_digit"], "n_tok": r["n_tok"],
            "dist": r["dist"],
        })
    m = [x["pld"] for x in rows if x["cond"] == cond]
    hack = sum(1 for x in rows if x["cond"] == cond and x["argmax"] in ODD_DIGITS)
    print(f"  [{ci+1:2d}/{len(CONDITIONS)}] {cond:24s} "
          f"PLD={st.mean(m):+7.3f} +-{st.stdev(m):5.3f}  "
          f"P(odd)={1/(1+math.exp(-st.mean(m))):.3f}  "
          f"hack_argmax={hack}/{len(variants)}  ({time.time()-t0:.0f}s)")

save({"model": MODEL, "n_variants": len(variants), "rows": rows},
     f"battery_{TAG}.json")
print("saved.")
