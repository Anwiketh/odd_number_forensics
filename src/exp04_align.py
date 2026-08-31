"""Check that the aligned/conflict minimal pair is token-aligned.

For activation patching we need two prompts of identical length whose tokens
differ in as few positions as possible. frame_english(odd) and
frame_english(even) differ only by swapping the two parity words, so IF " odd"
and " even" are each a single token the pair differs in exactly 2 positions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *
import factorial as FX

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
R = Runner(MODEL)
for w in [" odd", " even", "odd", "even", " higher", " lower"]:
    print(f"{w!r:10s} -> {R.tok.encode(w, add_special_tokens=False)}")

task_text = FX.TASKS["even"][0]
fmt = FX.FORMAT_VARIANTS[0]
a = R.build("You are a helpful assistant.",
            FX.build_user(task_text, FX.frame_english("even"), "after", fmt))
b = R.build("You are a helpful assistant.",
            FX.build_user(task_text, FX.frame_english("odd"), "after", fmt))
print(f"\nlens: aligned={a.shape[1]} conflict={b.shape[1]}")
if a.shape == b.shape:
    diff = (a[0] != b[0]).nonzero().flatten().tolist()
    print("differing positions:", diff)
    for p in diff:
        print(f"  pos {p}: {R.tok.decode([a[0,p]])!r} -> {R.tok.decode([b[0,p]])!r}")
print("\nposition map (index: token):")
print(" ".join(f"{i}:{R.tok.decode([t])!r}" for i, t in enumerate(a[0])))
