"""Diagnose the answer slot: what token does the model actually want to emit,
and which prefill puts ~all probability mass on a single digit token?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *

R = Runner("Qwen/Qwen3-0.6B")
user = make_prompt(REWARD_ODD)

for prefill in ["ANSWER:", "ANSWER: ", "\nANSWER:", "ANSWER:  "]:
    ids = R.build(SYSTEM, user, prefill=prefill)
    out = R.forward(ids)
    ll = out.logits[0, -1].float()
    p = torch.softmax(ll, -1)
    top = torch.topk(p, 8)
    print(f"\nprefill={prefill!r}  last3toks={[R.tok.decode([t]) for t in ids[0,-3:]]}")
    print("  top:", [(repr(R.tok.decode([int(i)])), round(float(v), 3))
                     for v, i in zip(top.values, top.indices)])
    print("  digit_mass:", round(R.mass_on_digits(ll), 4))
