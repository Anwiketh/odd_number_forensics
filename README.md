# What Is the Odd Number Environment Actually Measuring?

A forensic audit of a reward-hacking probe. Six open-weight models, 0.6B–9B,
everything on CPU.

**Start here:** [`submission/MATS12_OddNumber_Submission.docx`](submission/) —
executive summary, full write-up and all seven figures in one document. Or read
[`paper/EXECUTIVE_SUMMARY.md`](paper/EXECUTIVE_SUMMARY.md) (591 words).

---

## The result

The *Odd Number* environment instructs a model to output an **even** digit, adds
a scoring block paying for **odd**, observes that it answers odd, and reports
that as reward hacking. Holding the two answers fixed and varying only what the
block *says* about them, across six models (nats):

| | 0.6B | 0.8B | 1.5B | 2B | 4B |
|---|---|---|---|---|---|
| **the incentive** (pays more for disobeying vs for obeying) | +0.38 | +0.39 | +0.89 | +0.80 | +1.53 |
| **mere payment structure** (only disobedient paid vs both paid) | +4.39 | +0.90 | +6.18 | +2.06 | +5.46 |
| **word order alone** (disobedient named first vs last) | +4.60 | −0.09 | +1.21 | +0.16 | +0.45 |

The quantity the environment is named after is its **smallest** component at
every size tested — 2–12× under the mere fact of which option is flagged as
paid, and at 0.6B an order of magnitude under the order of two clauses in one
sentence.

What scales with model size is **reward comprehension** (magnitude sensitivity
inside a reward frame, as a fraction of the same model's sensitivity to the same
comparison asked directly: 0% → 9% → 13% → 29% → 80% → 76%) and the
disappearance of the word-order artifact. Not incentive-following.

**It is not an artifact of toy tasks.** Rebuilt on a two-option MCQ the model can
answer unaided, where "disobeying" means knowingly asserting something false for
points, the confound is *worse*: κ = 0.30–0.38 against a median of 0.21 in the
toy environments, and Qwen3-0.6B gives a knowingly false answer in **53%** of
variants when the wrong option is labelled (A) and **0%** when the same wrong
answer is labelled (B).

## Two things worth stealing even if you ignore the rest

**1. Always run the mirror arm.** Every instruction-conflict environment has two
arms — instruct X and pay for not-X, or instruct not-X and pay for X — and
essentially nobody runs both. Running both splits the effect into a
polarity-invariant δ and a content bias β that would have appeared whichever way
the incentive pointed. It costs exactly 2× and it is the difference between a
number and a number you can interpret.

**2. Pay both answers the same.** One extra condition. If a block reading
"1 point if odd, 1 point if even" moves your model as far as a real incentive
does, you are not measuring incentive-following. This is the cheapest validity
check in the paper and it would have caught the problem immediately.

## Layout

```
paper/
  EXECUTIVE_SUMMARY.md   591 words, the 60-second version
  paper.md               the full write-up
  FORM_ANSWERS.md        application-form summary answers
  AUDIT.md               six rounds of self-audit -- read this one
submission/
  MATS12_OddNumber_Submission.docx    everything, figures embedded
src/
  core.py                model wrapper: batched forwards, activation patching,
                         resid_post capture via forward hooks, steering
  tasks.py metric.py     the four binary conflict environments; the CLD read-out
  exp10_behaviour.py     MAIN behavioural experiment (4 envs x 2 arms x frames)
  exp20_copying.py       numeral-copying channel
  exp21_payout.py        MAIN: payout direction x zero-ness x mention order
  exp22_magnitude.py     can the model compare the payouts at all?
  exp17_mcq.py           the non-toy replication
  exp15_null.py          split-half reliability, chance, frame specificity
  exp16_steer.py         leave-one-out steering (negative result)
  direction_effect.py    the corrected estimator -- use this, not ad-hoc SEs
  build_submission.py    assembles the .docx
results/   raw json per experiment
figures/   png
```

## Reproducing

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install "transformers>=4.51" numpy matplotlib accelerate
.venv/Scripts/python src/test_patch.py          # verify the patching machinery
.venv/Scripts/python src/exp21_payout.py Qwen/Qwen3.5-2B 32
.venv/Scripts/python src/exp22_magnitude.py Qwen/Qwen3.5-2B
.venv/Scripts/python src/direction_effect.py
.venv/Scripts/python src/fig_scaling.py
```

Verified on Python 3.12.10 / torch 2.13.0+cpu / transformers 5.16.1. `exp21` is
3600 forwards: ~13 min at 0.6B, ~30 min at 2B, ~82 min at 4B.

**Machine note:** anything over ~10 GB of weights thrashes on a 31 GB box and
slows *superlinearly* — 9B ran 5.2× slower per forward than 4B against a 2.25×
parameter ratio. Measure throughput on the short `exp22` before committing to a
long `exp21`.

## Gotchas already paid for

- HuggingFace `output_hidden_states` does **not** give `resid_post` of the last
  layer: `hidden_states[i]` is the *input* to layer `i`, and
  `hidden_states[n_layers]` is post-`model.norm`. `core.py` uses forward hooks
  instead; `test_patch.py::T3` is the regression test.
- Answer-slot prefill is per-environment. Digits want `ANSWER: ` (trailing
  space); words and letters want `ANSWER:` and you score the space-prefixed
  token `' A'`. Getting it backwards puts ~0 mass on the answer set and the
  metric becomes a ratio of two negligible numbers. Our first pass held 0.0001
  of the mass in one environment and produced a spuriously dramatic headline.
- Check the answer set holds the probability mass. Ours averages 0.994.
- Don't build a summary statistic that averages over the manipulated variable,
  or that folds a systematic factor into the noise term. This project shipped
  two of those and both looked *unusually clean* — see `paper/AUDIT.md`.
