"""Serial driver -- CPU is the bottleneck, so never run two of these at once."""
import subprocess, sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

JOBS = []
for spec in sys.argv[1:]:
    script, model, *rest = spec.split(",")
    JOBS.append([PY, "-u", os.path.join(HERE, script), model] + rest)

for j in JOBS:
    print(f"\n{'='*70}\n>>> {' '.join(j[2:])}\n{'='*70}", flush=True)
    t = time.time()
    r = subprocess.run(j, env={**os.environ, "OMP_NUM_THREADS": "4",
                               "HF_HUB_OFFLINE": "1"})
    print(f"<<< rc={r.returncode} in {time.time()-t:.0f}s", flush=True)
