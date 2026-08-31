# Status and pick-up notes

Current as of 2026-08-31. The project is finished and submittable; what follows
is the state of each piece and what a next session would do.

## Deliverables

| file | state |
|---|---|
| `paper/EXECUTIVE_SUMMARY.md` | 600 words, at the cap. No em dashes. |
| `paper/paper.md` | full write-up, 13 sections, 7.5k words |
| `paper/FORM_ANSWERS.md` | application-form answers; one section deliberately blank |
| `paper/AUDIT.md` | six rounds of self-audit |
| `submission/MATS12_OddNumber_Submission.docx` | all of the above with eight figures embedded, opens in Google Docs |

The blank section in `FORM_ANSWERS.md` is "1 to 3 pieces of evidence that you'd
be able to do good research". That is about the applicant's background and cannot
be written from the repo.

## The result, in one table

Effect on δ, in nats, from `src/direction_effect.py`:

| | 0.6B | 0.8B | 1.5B | 2B | 4B |
|---|---|---|---|---|---|
| the incentive (pays more for disobeying vs. obeying) | +0.38 | +0.39 | +0.89 | +0.80 | +1.53 |
| mere payment structure (only disobedient paid vs. both) | +4.39 | +0.90 | +6.18 | +2.06 | +5.46 |
| word order alone (disobedient named first vs. last) | +4.60 | −0.09 | +1.21 | +0.16 | +0.45 |

Reward comprehension, from `exp22`: 0%, 9%, 13%, 29%, 80%, 75% across
0.6B/0.8B/1.5B/2B/4B/9B.

## Read this before touching the analysis

Three of this project's own headline claims were retracted, two of them because
of summary statistics that looked unusually clean and were structurally
incapable of showing anything else. Full detail in `paper/AUDIT.md` rounds 3 to 6
and paper §12.1. The short version:

1. "Models read a reward as a one-bit predicate." True below 2B, refuted at 4B.
2. "δ(real incentive) − δ(equal pay) is zero in 9 of 9 cells." Broken: equal pay
   is the arithmetic midpoint of the two payout directions, so averaging over
   direction cancels the effect by construction.
3. "Incentive-following switches on near 2B." Broken: the standard error pooled
   the four (config × mention-order) cells, and mention order is a large
   systematic effect (+4.60 nats at 0.6B), which inflated the SE up to eightfold.
   Order-blocked, there is no threshold.

**Use `src/direction_effect.py` for any claim about δ.** It blocks on mention
order and derives the SE from residual config-level spread. Do not hand-roll an
SE over the raw condition cells.

## Experiments, and their state

| script | state |
|---|---|
| `exp10_behaviour.py` | done, 2 models. Polarity decomposition, controls A/B/C |
| `exp15_null.py` | done. Direction geometry, reliability, chance |
| `exp16_steer.py` | done. **Negative**: specificity fails and mass collapses |
| `exp17_mcq.py` | done, 2 models. The non-toy replication, §4.9 |
| `exp20_copying.py` | done, 2 models. Numeral-copying channel |
| `exp21_payout.py` | done, 5 models (0.6B to 4B). The main factorial |
| `exp22_magnitude.py` | done, 6 models (0.6B to 9B). Capability probe |

`exp11` (activation patching to localise) and `exp14` (CoT causality) were
deliberately not run and their scaffolds were deleted: both are on the admissions
doc's explicit common-mistakes list. They remain in git history.

## Optional remaining work

1. `exp21` at 9B. Killed on CPU: 18 GB of weights on a 31 GB box left 0.9 GB free
   and ran 5.2x slower per forward than 4B against a 2.25x parameter ratio, with
   about 7.5 hours remaining. The RTX 5080 in this machine does not rescue it
   either, at 16 GB VRAM against 36 GB for float32 or 18 GB for bfloat16.
2. Re-derive the §3 polarity numbers with order blocked, for consistency with
   §4.8. They use a different estimator and were not affected by the SE bug, but
   it would be tidier.
3. `exp18` (metagaming) never run; it is not referenced as a result anywhere.

## Environment

`.venv` in the project root. Python 3.12.10, transformers 5.16.1, numpy 2.5.2.
24 logical cores, 31 GB RAM, RTX 5080 Laptop with 16 GB VRAM. `test_patch.py`
passes 4/4.

`core.Runner` defaults to cpu/float32 and accepts `FORENSICS_DEVICE` and
`FORENSICS_DTYPE`. float32 is deliberate, not an accident of running on CPU: the
Qwen3.5-2B factorial was re-run on the GPU at float32 and reproduces the CPU
output to every reported digit, in 131 s against 1825 s.

**Machine note:** anything over roughly 10 GB of weights thrashes on 31 GB of RAM
and slows superlinearly rather than proportionally. Measure per-forward
throughput on the short `exp22` before committing to a long `exp21`.

## Gotchas already paid for

- HuggingFace `output_hidden_states` does not give `resid_post` of the last
  layer. `hidden_states[i]` is the input to layer `i` and
  `hidden_states[n_layers]` is post-`model.norm`. `core.py` uses forward hooks;
  `test_patch.py::T3` is the regression test.
- Answer-slot prefill is per-environment. Digits want `ANSWER: ` with the
  trailing space; words and letters want `ANSWER:` and you score the
  space-prefixed token `' A'`. Getting it backwards puts almost no mass on the
  answer set and makes CLD a ratio of two negligible numbers. `SlotMetric`
  handles it and Control C catches it.
- CLD is always `log P(disobey) − log P(obey)`. An earlier version measured
  relative to the incentivised side, which silently flipped the sign in aligned
  cells. `analyse.cell_vectors` recomputes from `compliance` and asserts
  consistency.
- On the original 4-core machine, two concurrent experiment processes halved each
  other, which is why `run_pipeline.py` existed. This machine has 24 cores, so
  two or three concurrent runs are fine if you cap `OMP_NUM_THREADS`.
