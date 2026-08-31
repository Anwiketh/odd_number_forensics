"""Diagnose the answer slot for every environment.

Control C in summary.py found that the `letter` and `yesno` answer sets hold
~0 of the next-token mass, which invalidates CLD in those environments. Find
out what the model actually wants to emit after 'ANSWER: '.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from core import Runner
import tasks as TK

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
R = Runner(MODEL)
print(MODEL)
for n, T in TK.ALL_TASKS.items():
    s = T.sides[0]
    user = T.build(s, T.incentive(T.other(s)), 0, 0)
    txt = R.build_text("You are a helpful assistant.", user,
                       prefill=T.prefill)
    ll = R.batch_last_logits([txt], bs=1)[0]
    p = torch.softmax(ll, -1)
    top = torch.topk(p, 10)
    print(f"\n--- {n}  (slot='{T.slot}') prefill tail: "
          f"{R.tok.decode(R.tok(txt, add_special_tokens=False).input_ids[-4:])!r}")
    print("   top:", [(repr(R.tok.decode([int(i)])), round(float(v), 3))
                      for v, i in zip(top.values, top.indices)])
    for side, strs in T.answers.items():
        for a in strs:
            for cand in [a, " " + a, a.lower(), a.capitalize(), " " + a.lower()]:
                ids = R.tok.encode(cand, add_special_tokens=False)
                if len(ids) == 1:
                    print(f"     {side:6s} {cand!r:10s} id={ids[0]:6d} "
                          f"p={float(p[ids[0]]):.4f}")
