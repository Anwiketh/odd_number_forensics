"""Correctness tests for the batched patching machinery.

T1: patching the LAST layer at every position with source activations must
    reproduce the source logits exactly (resid_post of the final layer fully
    determines the output).
T2: batched patching must agree with one-at-a-time patching.
T3: patching with the run's OWN activations must be a no-op.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from core import Runner
from metric import SlotMetric
import tasks as TK

R = Runner("Qwen/Qwen3-0.6B")
T = TK.PARITY
M = SlotMetric(R, T)
SYS = "You are a helpful assistant."
a_txt = R.build_text(SYS, T.build("even", T.incentive("even"), 0, 0))
c_txt = R.build_text(SYS, T.build("even", T.incentive("odd"), 0, 0))
a = R.tok(a_txt, return_tensors="pt", add_special_tokens=False).input_ids
c = R.tok(c_txt, return_tensors="pt", add_special_tokens=False).input_ids
assert a.shape == c.shape
S = a.shape[1]
read = lambda ll: M.ld(ll, "odd")
ra, rc = R.cache_resid(a), R.cache_resid(c)
base_a = read(R.forward(a).logits[0, -1].float())
base_c = read(R.forward(c).logits[0, -1].float())
print(f"S={S} CLD aligned={base_a:+.4f} conflict={base_c:+.4f}")

assert ra.shape == (R.n_layers, S, R.d_model), ra.shape

# T1
last = R.n_layers - 1
got = R.run_with_patch(a, [(last, list(range(S)), rc[last])])
print(f"T1 patch-all-final-layer -> {got:+.4f} (want {base_c:+.4f})  "
      f"{'PASS' if abs(got-base_c) < 1e-3 else 'FAIL'}")

# T3
got = R.run_with_patch(a, [(last, list(range(S)), ra[last])])
print(f"T3 self-patch            -> {got:+.4f} (want {base_a:+.4f})  "
      f"{'PASS' if abs(got-base_a) < 1e-3 else 'FAIL'}")

# T3b: self-patch at an early layer, all positions
got = R.run_with_patch(a, [(3, list(range(S)), ra[3])])
print(f"T3b self-patch L3        -> {got:+.4f} (want {base_a:+.4f})  "
      f"{'PASS' if abs(got-base_a) < 1e-3 else 'FAIL'}")

# T2
jobs = [(l, p) for l in [0, 5, 14, 27] for p in [10, 40, S - 1]]
b = R.batched_patch(a, rc, jobs, read, bs=5)
s = [R.run_with_patch(a, [(l, [p], rc[l][p])]) for (l, p) in jobs]
mx = max(abs(x - y) for x, y in zip(b, s))
print(f"T2 batched vs serial max|diff| = {mx:.2e}  "
      f"{'PASS' if mx < 1e-3 else 'FAIL'}")
for (l, p), x, y in zip(jobs, b, s):
    print(f"   L{l:2d} p{p:3d} batched={x:+.4f} serial={y:+.4f}")
