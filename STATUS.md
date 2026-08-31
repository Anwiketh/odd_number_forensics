# Status / pick-up notes

Last updated mid-project. Read this first.

## Where things stand (2026-08-30, end of day)

**The paper has a new spine.** The headline is no longer the polarity confound —
that is now §3, a supporting result. The centre is **§4: δ, the quantity that
survives the mirror arm, is not incentive-following either.**

The three deliverables are written and every number in them has been
independently re-derived from `results/`:

- `paper/EXECUTIVE_SUMMARY.md` — **597 words**, under the 600-word cap
- `paper/paper.md` — full write-up, restructured around §4
- `paper/FORM_ANSWERS.md` — application-form summary answers

One section of `FORM_ANSWERS.md` is deliberately left blank: *"1–3 pieces of
evidence that you'd be able to do good research"* is about the applicant's
background and cannot be written from the repo.

### Environment (this machine, unlike the one that produced the old results)

Python 3.12.10 + `.venv` in the project root; torch 2.13.0+cpu, transformers
**5.16.1**, numpy 2.5.2. The transformers v5 major bump does not break
`core.py`: `test_patch.py` passes 4/4 and `exp03` reproduced its stored values
exactly before any new experiment was run. 24 logical cores, so 2–3 concurrent
runs are fine (cap `OMP_NUM_THREADS`).

### Deliberately NOT run, and why

`exp11` (activation patching to localise) and `exp14` (CoT causality) are on the
admissions doc's explicit *common mistakes* list — "using patching to show which
heads/layers are used in a task", "showing that chain of thought causally
impacts the final answer". Their sections were cut from the paper rather than
filled. `exp17` (MCQ) and `exp18` (metagaming) remain unrun; the "your
environments are toys" objection is handled honestly in §10 instead.

`exp16` (leave-one-out steering) is the one causal follow-up **worth** running —
it is not on the avoid list and would make §8 causal instead of correlational.
It is written and tested. That is the top remaining item.

## Numbers audit — done 2026-08-30

Every specific value in `paper.md` and `EXECUTIVE_SUMMARY.md` was re-checked
against `results/`. Most held. The corrections applied are listed in
`paper/AUDIT.md`; originals are preserved as `paper/*.md.bak-preaudit`.

The one that matters: §3.6 claimed p("1") = 0.48 under the canonical prose
reward. **The real value is 0.71** (mean over 24 variants, `reward_odd_english`
in `results/battery_Qwen3-0.6B.json`). 0.48 appears nowhere in `results/`. The
correction strengthens the copying argument rather than weakening it.

## Done and trustworthy

| result | file | status |
|---|---|---|
| Polarity decomposition (δ, β, κ), 2 models × 4 envs × 5 framings | `results/behaviour_*.json`, `results/tables.md` | **done** |
| Controls A (task validity), B (conflict-specificity), C (answer-set mass) | `results/tables.md` | **done** |
| Comprehension probe, Qwen3-0.6B | `results/comprehension_Qwen3-0.6B.json` | **done** |
| Direction geometry: reliability, chance, cross-env ρ, cross-framing ρ | `results/null_Qwen3-0.6B.json` | **done** |
| Figures 1, 2, 3, 5 | `figures/` | **done** |
| Patching machinery correctness tests | `src/test_patch.py` | **4/4 pass on transformers 5.16.1** |
| Numeral copying, Qwen3-0.6B | `results/copying_Qwen3-0.6B.json` | **done 2026-08-30** |
| **Payout factorial (MAIN RESULT), both models** | `results/payout_*.json`, `figures/fig9_payout.png` | **done 2026-08-30** |
| **Magnitude-capability probe, both models** | `results/magnitude_*.json` | **done 2026-08-30** |

## Not done — exactly where to pick up

1. **`exp20` (numeral copying) is now DONE for Qwen3-0.6B** — 1584 prompts,
   348 s, answer-set mass 1.00/1.00/0.986. The OOM is confirmed fixed by
   `core.map_last_logits`. §3.6 is written up from the real numbers. Two things
   to know:

   - **The scheme table has a design defect.** `Task.incentive(side, hi, lo)`
     attaches `hi` to the side it is handed, and `exp20` hands it the *hacking*
     side — so five of the ten schemes (1/2, 3/5, 5/7, 2/4, 4/6) pay **more for
     obeying**, contradicting the module docstring's claim that "the high payout
     always goes to the conflicting side". This is why the "δ should be
     constant" control fails. It does **not** contaminate the copying result
     (the inverted schemes are balanced across the imbalance groups), and it
     accidentally produced the best finding in the experiment. Fix by crossing
     direction with numerals deliberately rather than re-running blind.
   - Use `python src/exp20_analyse.py Qwen3-0.6B` for the breakdown; the
     headline printed by `exp20_copying.py` itself (r = +0.21, "looks null") is
     misleading because it pools the zero-payout and non-zero-payout schemes.

   **Replicated on Qwen2.5-1.5B-Instruct** (766 s). Everything holds and the
   copying slope is 3× larger there (+0.326 vs +0.106 nats per unit imbalance
   in `parity`, against +0.030 and +0.016 in the numeral-free control). The
   zero-payout δ effect appears in all six model × environment cells.

   Open question the current scheme set **cannot** answer: the canonical "1
   point / 0 points" has parity imbalance zero, so the imbalance channel
   predicts no net push — yet p("1") = 0.71. The driver must be "copy the
   numeral on the *rewarded* side", but every non-zero-imbalance scheme also
   carries an odd numeral on the rewarded side, so the two are perfectly
   confounded. **Add 2/1, 4/3, 6/5 (even numeral on the rewarded side, odd on
   the other) to break it.** This is the highest-value small experiment left.

2. **Fill §5.4 (steering), §5.5 (CoT), §5.6 (patching)** once
   `results/steer_*.json`, `results/cot_*.json`, `results/patch_*.json` exist.
   Figures 6, 4, 7 are already written in `src/figures.py` and will render as
   soon as those files appear.

3. **Add SmolLM2-1.7B and Qwen3-1.7B to Tables 1–2** — just re-run
   `python src/summary.py && python src/figures.py` after their `exp10`/`exp03`
   land; both scripts auto-discover every `behaviour_*.json`.

4. **`exp17` (MCQ) and `exp18` (metagaming) are written but unrun.** exp17 is
   the answer to "your environments are toys" and is referenced in §6 (related
   work, MCQ position bias) — worth prioritising if time is short. exp18 is
   referenced in §3.4 and is currently a placeholder paragraph.

5. **Re-check the abstract's numbers at the end.** The abstract quotes
   "ρ ≈ 0.49", "0.04 nats", "median κ ≈ a fifth to a third" — all currently
   correct, but they will need a pass once more models are in.

## Two things a reader/reviewer will attack, and where they are handled

- *"Unequal headroom between arms manufactures β."* It does not, because CLD is
  a log-odds and a fixed logit push moves it by a fixed amount regardless of
  starting point. Stated in §3.1 and in the `summary.py` docstring.
- *"Your environments are degenerate toys."* `exp17` (MCQ where hacking means
  answering knowably wrong) is the reply. Unrun — see item 4.

## Gotchas already paid for (don't re-discover these)

- HF `output_hidden_states` does **not** give `resid_post` of the last layer:
  `hidden_states[n_layers]` is post-`model.norm`. `core.cache_resid` uses
  forward hooks instead; `test_patch.py::T3` is the regression test.
- Answer-slot prefill is per-environment: digits want `ANSWER: ` (trailing
  space), words/letters want `ANSWER:` (no trailing space, and you score the
  space-prefixed token `' A'`). Getting it backwards puts ~0 mass on the answer
  set and CLD becomes meaningless. `SlotMetric` handles it; Control C catches it.
- Sign convention: CLD is always `log P(disobey) − log P(obey)`. An earlier
  version measured relative to the *incentivised* side, which silently flipped
  the sign in `aligned` cells. `analyse.cell_vectors` recomputes from
  `compliance` and asserts consistency.
- Don't run two experiment processes at once *on the original 4-core box* —
  they halve each other, which is why `run_pipeline.py` exists. This machine
  has **24 logical cores**, so 2–3 concurrent runs are fine here provided you
  cap threads per process (`$env:OMP_NUM_THREADS = 8`); torch otherwise grabs
  everything and the processes contend anyway.
- Don't pipe a background PowerShell job through `Select-Object`; it buffers the
  whole stream and you see nothing until the job exits.
