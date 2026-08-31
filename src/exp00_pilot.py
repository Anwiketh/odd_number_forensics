"""exp00: sanity pilot. Does the phenomenon reproduce, and is the single-token
metric well-behaved? Fail-fast check before investing in anything else."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"

t0 = time.time()
R = Runner(MODEL)
print(f"loaded {MODEL} in {time.time()-t0:.1f}s | layers={R.n_layers} d={R.d_model}")
print("digit tokens:", {d: (t, repr(R.tok.decode([t]))) for d, t in R.digit_tokens.items()})

for label, block in [("no_incentive", NO_INCENTIVE),
                     ("reward_odd", REWARD_ODD),
                     ("reward_even", REWARD_EVEN)]:
    user = make_prompt(block)
    ids = R.build(SYSTEM, user)
    t = time.time()
    r = R.run(ids)
    dt = time.time() - t
    dist = {d: round(v, 3) for d, v in r["dist"].items() if v > 0.01}
    print(f"\n[{label}] ntok={r['n_tok']} fwd={dt:.2f}s")
    print(f"  PLD={r['pld']:+.3f}  P(odd|digit)={1/(1+math.exp(-r['pld'])):.3f} "
          f"digit_mass={r['digit_mass']:.3f} argmax={r['argmax_digit']}")
    print(f"  dist={dist}")

print("\n--- full prompt (reward_odd) ---")
print(R.tok.decode(R.build(SYSTEM, make_prompt(REWARD_ODD))[0]))
