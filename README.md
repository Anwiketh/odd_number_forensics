# Which Side Did You Ask For?

Separating incentive-following from content bias in instruction-conflict
("reward hacking") environments.

A model-forensics study of the *Odd Number* environment from
[Concrete Problems in Model Forensics], generalised to four conflict
environments and four open-weight models. Everything runs on CPU.

## The one-paragraph version

Toy reward-hacking environments instruct the model to do X and offer an
in-context reward for not-X. They have two arms — you can also instruct not-X
and reward X — and essentially nobody runs both. Running both lets you split
the measured effect into a polarity-invariant part (real incentive-following)
and a polarity-specific part (the model just likes one of the answers). The
second part is not small, and in the worst case we measured it takes the
reported disobedience rate from 100% to 0% for the same model and the same
incentive.

## Layout

```
src/
  core.py            model wrapper: batched forwards, batched activation
                     patching, resid_post capture, steering hooks
  tasks.py           the four binary conflict environments
  metric.py          the CLD (conflict logit difference) read-out
  conditions.py      exploratory prompt battery (exp01)
  factorial.py       the first two-arm design (exp02)
  spans.py           prompt-span -> token-index mapping for patching plots

  exp00_pilot.py       does the phenomenon reproduce at all?
  exp00b_slot.py       answer-slot tokenisation diagnostics
  exp01_battery.py     25-condition exploratory battery, one arm
  exp02_factorial.py   first two-arm factorial + naive comprehension probe
  exp03_comprehension.py  prior-free reward-spec comprehension probes
  exp10_behaviour.py   MAIN behavioural experiment (4 envs x 2 arms x frames)
  exp11_patch.py       residual-stream activation patching, layer x position
  exp12_directions.py  polarity-invariance test for a conflict direction
  exp13_transfer.py    cross-environment geometry + leave-one-out steering
  exp14_cot.py         CoT commitment trajectories
  exp15_null.py        split-half reliability, chance level, frame specificity
  test_patch.py        correctness tests for the patching machinery

  analyse.py     the (delta, beta, kappa) decomposition with bootstrap CIs
  summary.py     paper tables + the three validity controls
  figures.py     every figure
  viz.py         plotting style
  run_pipeline.py  serial driver (CPU is the bottleneck; never parallelise)

results/   raw json per experiment, plus tables.md
figures/   png
paper/     paper.md, EXECUTIVE_SUMMARY.md
```

## Reproducing

Requirements: `torch` (CPU is fine), `transformers>=4.51`, `numpy`,
`matplotlib`. No TransformerLens/nnsight dependency — `core.py` uses plain
forward hooks so it runs anywhere `transformers` does.

From scratch on Windows:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install "transformers>=4.51" numpy matplotlib accelerate
```

Verified working on Python 3.12.10 / torch 2.13.0+cpu / **transformers 5.16.1**.
The v5 major bump does not break `core.py` — `test_patch.py` passes 4/4 and
`exp03` reproduces its stored values to the reported precision — but v5 does
deprecate `torch_dtype` in favour of `dtype`, which is currently a warning only.

```bash
python src/test_patch.py                       # verify the patching machinery
python src/exp10_behaviour.py Qwen/Qwen3-0.6B 16
python src/exp03_comprehension.py Qwen/Qwen3-0.6B
python src/exp13_transfer.py Qwen/Qwen3-0.6B 16
python src/summary.py
python src/figures.py
```

or drive several in sequence:

```bash
python src/run_pipeline.py "exp10_behaviour.py,Qwen/Qwen3-0.6B,16" "exp03_comprehension.py,Qwen/Qwen3-0.6B"
```

Wall-clock on 4 CPU cores: `exp10` ≈ 25 min/model, `exp13` ≈ 60 min,
`exp11` ≈ 90 min.

## Two things worth stealing even if you ignore the rest

**1. Force the decision into one token.** Every environment ends with a strict
`ANSWER: <slot>` format and we prefill up to `ANSWER: `. The whole behaviour is
then one forward pass and a logit difference, so a design that would need
thousands of rollouts needs thousands of *forwards*, and the metric is
continuous instead of binary. Check that the answer set captures the next-token
mass (ours averages 0.994; the 1% of conditions below 0.90 are worth inspecting
individually) or the metric is measuring a thin slice of an open distribution.

**2. Always run the mirror arm.** It costs exactly 2x and it is the difference
between a number and a number you can interpret.

## Gotcha we hit, in case it saves you an hour

HuggingFace's `output_hidden_states` does **not** give you `resid_post` of the
last layer: `hidden_states[i]` is the *input* to layer `i`, and
`hidden_states[n_layers]` is the state *after* `model.norm`. Patching
`hidden_states[l+1]` therefore silently corrupts the final layer. `core.py`
captures with forward hooks instead; `test_patch.py::T3` is the self-patch
no-op test that catches it.
