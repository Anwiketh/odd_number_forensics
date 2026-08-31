# Numbers audit: 2026-08-30

> **Addendum, same day.** After this audit, two new experiments were run
> (`exp21_payout.py`, `exp22_magnitude.py`) and the paper was restructured
> around them. Every number in the new §4 was re-derived from
> `results/payout_*.json` and `results/magnitude_*.json` by an independent
> script after the write-up was drafted; all checks passed. The new experiments
> exist *because* of defect (4) below: see "What the defect bought" at the end.

Every specific value in `paper.md` and `EXECUTIVE_SUMMARY.md` re-checked
against `results/`. Originals preserved as `paper.md.bak-preaudit` and
`EXECUTIVE_SUMMARY.md.bak-preaudit`.

No experiments were run for this audit: none can be, on this machine (see
`STATUS.md`). Everything below is recomputed from the existing JSON in
`results/`.

## What held up

- Comprehension probe, Table 2: all six values match
`comprehension_Qwen3-0.6B.json` exactly (0.043→0.04, 2.208→2.21, 4.120→4.12,
0.182→0.18, 0.595→0.60).
- Geometry, §5.2 and the abstract. Recomputed from `null_Qwen3-0.6B.json`:
mean attenuation-corrected cross-environment ρ for `v_conflict` peaks at
**0.487 at layer 16** (paper says "+0.49, layer 16"); `v_content` at the same
layer is **0.044** (paper says ≈0.04, "+0.00 to +0.06"); chance is 0.0249 for
d=1024 (paper says 0.025). Cross-framing ρ for `v_conflict` runs 0.56–0.74. All
correct.
- The δ/β/κ decomposition in Table 1, and the sign-agreement count (β's sign
disagrees in exactly 2 of 12 environment × framing combinations).
- §3.3's "share of the prose-reward effect" column: 147%, 47%, 21%, −10% all
recompute correctly from the δ values.
- Control B's β(aligned) column: independently recomputed from
`behaviour_*.json` and matches `tables.md` in all 24 cells.
- §3.2's headline: +8.57/+4.30 nats, 0.92/0.42 disobedience, δ=+6.43, κ=0.25.
- The 4×3×2 = 24 surface-variant count, and the ~864-row design size.

## Corrections applied

**1. §3.6 / Exec-summary 4: p("1") was wrong, and understated.** Paper said
**0.48**. That value appears nowhere in `results/`. Recomputed from
`battery_Qwen3-0.6B.json`, condition `reward_odd_english` (the canonical prose
wording), mean over 24 variants: **p("1") = 0.714**, max 0.982, against
**0.013** in the `clean` no-block baseline. Total odd-digit mass in that
condition is 0.868, so the single printed numeral accounts for 82% of the
"hacking" mass. Corrected to 0.71 and expanded.

Note `src/exp20_copying.py`'s own docstring cites a *third* value (a pilot pair
of 0.20 → 0.007). That pilot used `core.REWARD_ODD`, which is the **code**
form, not the prose form the docstring quotes above it. The battery gives
p("1") = 0.022 for the code form. The docstring should be corrected too: left
alone here because it is not a paper claim.

**2. §3.6: an unbacked result stated as fact.** "Rewriting the same reward as
'one point … no points' collapses that" was written in the indicative but
`exp20` has never completed and `results/` has no spelled-out condition.
Reworded as the prediction it is, and the section now says explicitly that the
experiment is unrun. Same for §1's "removes a large part of the effect".

**3. §4.1: obedience rate wrong.** Paper said Qwen2.5-1.5B obeys "output NO"
"only half the time". Recomputed: **0.88** (7 of 8 variants). The `magnitude`
failure is 0.75. Corrected, and the fact that both failures rest on 1–2
variants out of 8 is now stated.

**4. §4.2: example numbers matched no row.** Paper quoted Qwen2.5-1.5B on
`letter`/prose as β(conflict) = +3.48 [+2.77, +4.09], β(aligned) = −0.14
[−0.32, +0.07]. `tables.md` says +3.21 [+2.53, +3.92] and −0.73 [−1.23, −0.25].
The −0.14 point estimate belongs to `letter`/**code**, with a different CI.
Replaced with two real cells.

**5. §4.2: "exceptions" understated.** Paper named one exception (`magnitude`
for Qwen2.5). Counting all 24 Control-B cells: β(aligned) is
comparable-or-larger than β(conflict) in **seven**, and strictly larger in
four: Qwen3-0.6B's `parity`/code, `parity`/instruct, `magnitude`/english and
`yesno`/english. Two of those are the canonical `parity` environment, which is
worth saying out loud. Now stated.

**6. §7 limitation 2: claim not supported, and the truth is worse.** Paper said
that on `yesno` "an aligned instruction moves one arm roughly twice as far as
the other". Recomputed: on `yesno`/instruct the two arms move −3.19 and −2.73
(a 1.2× ratio, not 2×). But on `yesno`/english for Qwen3-0.6B they move **+2.92
and −1.12**: a block that *agrees* with the instruction pushes one arm toward
disobedience. Qwen2.5-1.5B on `magnitude`/instruct does the same (+0.33 vs
−4.00). Replaced with the real, stronger finding.

**7. §2.2: answer-set mass overstated, and cross-reference wrong.** Paper said
"mean 0.999 across every condition we ran; §5.1". Recomputed over all 1728
behaviour rows: **0.9922** (Qwen3-0.6B) and **0.9960** (Qwen2.5-1.5B), i.e.
0.994 overall, with 1.04% of conditions below 0.90 and a minimum of 0.0011. The
cross-reference should be **§4.3**, not §5.1 (§5.1 is the directions
construction). Both fixed; the same 0.999 claim is fixed in `README.md`, and
Exec-summary's "mass 1.000 after the fix" is fixed.

**8. §1 contaminant 1: valid-cell and all-cell statistics were mixed.** The "17
of 22" count is the valid-cell figure but the κ IQR/max (0.13–0.29, 0.44) and
the median rate gap (0.23) quoted alongside it are the *all-cell* figures.
Valid-cell values are IQR 0.12–0.27, max 0.39, median gap 0.25. Now stated
separately. (`EXECUTIVE_SUMMARY.md` was already internally consistent here: it
quotes 22/28 with the all-cell stats.)

**9. Abstract: "0.03% of the probability mass".** The two failing environments
held 0.0001 (0.01%) and 0.033 (3.3%). "0.03%" matches neither. Changed to "as
little as 0.01%".

**10. Abstract: cross-framing ρ.** Abstract said "0.6–0.74", §5.2/§5.3 and the
data say **0.56**–0.74. Unified.

**11. Dangling cross-references to a section that does not exist.** §6 and §7
both cited "§5.4" for the MCQ environment; §5.4 is the steering section, and
the MCQ work (`exp17`) has no section and has never been run. Both repointed to
`exp17` and marked unrun. Related: "five environments" in §7, §9 and the
Exec-summary counted the unrun MCQ env: corrected to four.

**12. Control naming was inconsistent across three documents.** `tables.md`
uses Control A/B/C, `paper.md` §4 uses §4.1/§4.2/§4.3 but §7 and §8 said
"Control 2", and `EXECUTIVE_SUMMARY.md` attributed the exchangeability check to
"Control A" (it is Control B). Unified on Control A/B/C with section numbers.

**13. Unrun experiments now marked as unrun.** §3.4 (`exp18`), §5.4 (`exp16`
steering), §5.5 (`exp14` CoT) and §5.6 (`exp11` patching) all read as though
results existed, ending with "Results in `results/…`". None of those files
exist. Each now says the experiment has not been run. §5.4 additionally notes
that until it runs, §5.2's geometry is correlational only.

**14. §2.5: model list.** Listed four models; only two have results. Now says
which two the numbers come from and that the other two are queued.

## Left alone deliberately

- **The two CI variants in `tables.md`.** Table 1 and Control B report slightly
different bootstrap CIs for the same cells (e.g. `letter`/english β = +6.66 is
[+5.76, +7.51] in one and [+5.75, +7.55] in the other). Consistent with
independent reseeds of a 5000-resample bootstrap, not an error, but the paper
should quote one canonical run. Worth pinning the seed in `analyse.py`.
- **`src/exp20_copying.py`'s docstring pilot numbers** (0.20 / 0.007): wrong
form of the reward, but it is a code comment, not a paper claim.
- **§3.6's `1/2` scheme** is labelled `n_odd − n_even = +1 − 1 = 0` in
`SCHEMES`, but "1 point"/"2 points" is one odd and one even numeral, so 0 is
right. No change needed; noting it because the adjacent `3/0` entry uses the
same `+1 - 1` expression where the numerals are 3 (odd) and 0 (even), also 0.
Both fine.

## Addendum: `exp20` has since been run (2026-08-30)

A Python toolchain was installed on this machine after the audit above, so
`exp20` finally ran (1584 prompts, 348 s, Qwen3-0.6B). §3.6 is now written from
real results rather than a prediction. Three things came out of it:

- The copying channel is **real**, and replicates on Qwen2.5-1.5B-Instruct:
β tracks numeral parity imbalance with slope +0.106 (Qwen3-0.6B) and +0.326
(Qwen2.5) nats per unit in `parity`, against +0.016 and +0.030 in the
numeral-free `letter` control. It does not explain the canonical wording, whose
own imbalance is zero: see §3.6.
- The spelled-out control **does not collapse β** (−35% in `parity`, −7% in
`letter`, no drop in `magnitude`), so correction 2 above was the right call:
the original "collapses that" claim would have been refuted by the project's
own experiment.
- `exp20` has a **design defect**: five of ten schemes pay more for obeying,
contradicting its docstring. Documented in the docstring and `STATUS.md`. It
does not contaminate the copying result and it produced the experiment's best
finding: δ is up to 2.8× larger when the losing option pays zero, while
reversing which side pays more moves δ by 0.01 nats.

## Still unverifiable without running anything

§5.2's split-half reliability ranges ("0.72–0.98" and "0.74–1.00") are quoted
as ranges without saying over what: layers, environments, or framings. The
per-layer means I recomputed are consistent with them but I could not pin the
exact aggregation. Worth making explicit in the caption.

---

## What the defect bought (added 2026-08-30, after exp21/exp22)

Defect (4): `exp20`'s scheme table silently paying more for *obeying* in five
of ten schemes: is the reason this project has a main result.

Because half the schemes had an inverted incentive, the run contained an
accidental payout-direction manipulation. Reading it back showed that delta was
essentially unchanged when the incentive reversed: Qwen2.5 `parity` gave delta
= +6.22 with hacking paid 9 against obeying paid 3, and +6.44 with obeying paid
4 against hacking paid 2. Incentive-following predicts a sign flip. That
observation prompted `exp21_payout.py`, which crosses payout direction,
zero-ness and mention order properly and adds the equal-payout control the
original design lacked, and `exp22_magnitude.py`, which rules out innumeracy.

Both are now the centre of the paper (section 4). The lesson is not that the
bug was lucky: it cost a confounded headline that would have been wrong in
print. It is that reading the raw per-condition table, rather than the summary
the analysis script printed, is what surfaced both the defect and the finding.

### Verification of the new results

Recomputed independently of the write-up:

- both-paid slope: +0.085 (Qwen3-0.6B), +0.067 (Qwen2.5): quoted as +0.09/+0.07
- one-pays-zero slope: +2.121, +2.048: quoted as > 2
- Qwen3 `parity`: 3/1 = +1.88, 1/3 = +2.28, 1/1 = +2.14: all exact
- primacy per environment: +2.20/+6.36/+5.25 and +0.29/+1.11/+2.23: quoted range
+0.29 to +6.36
- equal:real ratios, order-averaged: 1.17/0.96/0.95 and 0.98/0.97/1.04: quoted
as 0.95-1.17, median 0.98
- exp22: raw comparison +4.24 +- 0.50 and +9.36 +- 0.90; reward-framed
-0.01 +- 0.10 and +1.20 +- 0.89, i.e. 0% and 13%

### Known soft spots in the new work, stated plainly

- The primacy effect is real but varies from +0.29 to +6.36 nats across cells.
We report the range rather than an average, because an average would imply a
stability the data does not have.
- `exp22` Probe B has only n = 6 and n = 10 (pairs x system prompts). The effect
is 8+ standard errors from zero so the conclusion is safe, but the probe is
small.
- Qwen2.5's reward-framed magnitude contrast is +1.20 +- 0.89 (t = 1.35). We
describe it as "not distinguishable from zero", not as zero.
- `letter` answer-set mass has a minimum of 0.055 in a small number of exp21
conditions (mean 0.991). Those conditions are retained; the mean is reported.

---

## Round 3: the scale-up, and two retractions (2026-08-31)

### Retraction 1: the headline was over-generalised

The draft concluded "an in-context reward is read as a one-bit predicate, not a
quantity." True of Qwen3-0.6B and Qwen2.5-1.5B; **false at Qwen3.5-4B**, where
the direction effect is significant in all three environments (+0.74 [+0.26,
+1.22], +1.29 [+0.23, +2.35], +2.55 [+1.65, +3.46]) against 0 of 6 significant
cells below 2B. Reward comprehension (exp22) runs 0% -> 13% -> 80%.

This experiment was run precisely because "your models are too dumb for the
task" is on the admissions doc's common-mistakes list and was the strongest
available objection to the result. It cost the claim, and produced a better
one.

### Retraction 2: a statistic that cancels by construction

To test whether the incentive contributes at all, an earlier analysis compared
delta under a real incentive against delta under an equal-payout block, and
found them **indistinguishable in all nine model x environment cells**, with
confidence intervals that tightened from +-6.18 at 0.6B to +-0.40 at 4B. It
looked like an unusually clean null and was nearly the paper's headline.

It is an artifact. Equal pay is the arithmetic midpoint of "pays more for
disobeying" and "pays more for obeying". Averaging a real incentive over both
directions therefore cancels the effect the comparison is meant to detect, and
the contrast returns ~0 whether or not the model is following the incentive.

Caught by noticing the null disagreed with the direction-resolved numbers in
the same printout: `letter` at 4B showed hack-more +6.76, equal +5.57,
obey-more +4.44 -- an obvious ordering -- while the real-minus-equal contrast
reported -0.73 +- 1.08. The correct statistic is the direction contrast; the
broken one is retained in paper section 4.5 as a worked example.

### Verification of the new numbers

Recomputed independently of the write-up:

- direction effect, both-paid only, order-averaged, 95% CI from condition-level
variance -- 0.6B: +0.16 / -0.03 / +1.01 (none significant); 1.5B: +0.10 / +2.10
/ +0.47 (none significant); 4B: +0.74 / +1.29 / +2.55 (all significant)
- equal pay sits between the two directions in **9 of 9** cells, which is the
positive control for the contrast being meaningful
- amounts term in the regression: F(2,18) = 0.33 / 1.59 / **7.24** at 4B; the
letter value is the first significant one anywhere
- primacy: +4.60 -> +1.21 -> +0.45 nats (mean over environments)
- exp22 at 4B: +7.30 +- 0.21 reward-framed vs +9.08 +- 0.20 direct = 80%
- exp21 4B answer-set mass: 0.9998 / 0.9999 / 0.9856 (letter min 0.264)

### exp16 steering: reported as a failure

Small-alpha (|a| <= 0.35) response averaged over four held-out environments:
own_conflict +5.46 (ceiling), loo_conflict +2.96, loo_content +2.50, random
-0.11. Transfer beats random; **specificity does not hold** -- the content
direction, at chance in the geometry (rho = 0.04), steers as well as the
conflict direction. Answer-set mass falls to 0.61 (yesno) and 0.56 (letter)
where the effects are largest, so the intervention is partly breaking the
model. Reported in section 8.1 as an unresolved disagreement rather than as
support.

## Round 4: 9B, partial (2026-08-31)

`exp22` completed on **Qwen3.5-9B**: reward-framed +4.95 +- 0.13 against direct
+6.55 +- 0.16 = **75%**. Comprehension curve is therefore 0% / 13% / 80% / 75%
across 0.6B / 1.5B / 4B / 9B -- a step between 1.5B and 4B followed by a
plateau, which is a stronger shape than a monotone three-point rise.

Note the 9B model's absolute scores are LOWER than the 4B model's on both
probes (+4.95 vs +7.30 reward-framed, +6.55 vs +9.08 direct). Only the ratio is
compared, and the ratio is the scale-free quantity, but this is worth stating
rather than hiding.

`exp21` on 9B was **launched and killed**, not completed. Measured cause: 18 GB
of bf16 weights on a 31 GB machine left 0.9 GB free RAM, and the resulting
thrashing produced **5.2x slower per-forward throughput than 4B** against a
2.25x parameter ratio (measured on exp22, which both models run unbatched).
Projected remaining runtime ~7.5 h with a real out-of-memory risk, and exp21
writes its JSON only at the end, so a crash would have produced nothing. Killed
by decision rather than by failure.

Consequence for the write-up: the comprehension curve has four points, the
behavioural direction effect has three. Both paper 4.8 and the limitations say
so explicitly. No four-point behavioural curve is implied anywhere.

## Round 5: Qwen3.5-2B (2026-08-31)

Ran to fill the interval where the switch happens. Both experiments completed
cleanly in 30 min (exp21) with no memory pressure -- 2B is ~4GB, so the
thrashing that killed 9B does not apply.

- comprehension: **29.2%** (reward-framed +1.71 +- 0.04, direct +5.86 +- 0.26).
Full curve 0% / 13% / 29% / 80% / 75% across 0.6B / 1.5B / 2B / 4B / 9B.
- direction effect: **+0.46 [+0.30,+0.62], +1.10 [+0.24,+1.96], +0.83
[+0.60,+1.06]** -- significant in **3 of 3** environments.
- primacy: **+0.16** nats, the smallest of any model.
- answer-set mass 0.9999 / 0.9998 / 0.9891.

**This relocated the switch.** The earlier draft put it "between 1.5B and 4B"
because 4B was the smallest model tested that showed an effect. 2B already
shows 3/3 significance, so the switch is between **1.5B and 2B**. Corrected in
the paper, exec summary and form answers.

**And it revealed a dissociation.** At 2B the behavioural direction effect is
significant while comprehension is only 29%; at 4B comprehension is 80% and the
effect has roughly doubled. So the behavioural signal appears BEFORE
comprehension is high and then grows with it, rather than both crossing at one
point. The earlier "step then plateau" description of comprehension was also
too strong -- with 2B in place the curve is a graded climb, steepest between 2B
and 4B.

**Caveat now stated in all three documents:** only 2B/4B/9B are one family
(Qwen3.5). The 0.6B and 1.5B models are Qwen3 and Qwen2.5, so the 1.5B-to-2B
step where the switch sits is confounded with a family change. The switch
location is approximate.

## Round 6: Qwen3.5-0.8B, and a standard-error bug that invented a threshold

Ran 0.8B to get an in-family point below 2B. It completed in 16 min (exp21) and
produced comprehension 8.5%, direction effects +0.13/+0.23/+0.81, and a
word-order artifact of **-0.09** nats -- the only model with essentially none.

That last number is what exposed the bug. With near-zero primacy, 0.8B had CI
half-widths of ~0.1 nats while Qwen3-0.6B had ~4.6. "Small models are noisy"
suddenly looked like it tracked model family rather than size, which did not
make sense, so we checked the SE construction.

**The bug.** The direction effect contrasts (3/1, 4/2) against (1/3, 2/4). We
had pooled all four (config x mention-order) cells on each side and taken the
variance across them. But mention order produces a large SYSTEMATIC shift in
delta -- +4.60 nats on Qwen3-0.6B -- so that variance was dominated by a real
effect, not noise. The SE was inflated up to 8x, and worst precisely on the
models with the biggest order effects.

**What it cost.** The published claim "0 of 6 cells significant below 2B, 3 of
3 at 4B, therefore a capability threshold near 2B" was an artifact. With order
as a blocking factor the counts are 1/3, 3/3, 2/3, 3/3, 3/3 across
0.6B/0.8B/1.5B/2B/4B -- **no threshold**. The effect is detectable at
essentially every scale and simply small everywhere.

| model | mean effect | blocked CI +- | old pooled CI +- | blocked sig |
|---|---|---|---|---|
| Qwen3-0.6B | +0.38 | 0.58 | 4.59 | 1/3 |
| Qwen3.5-0.8B | +0.39 | 0.12 | 0.25 | 3/3 |
| Qwen2.5-1.5B | +0.89 | 0.31 | 2.70 | 2/3 |
| Qwen3.5-2B | +0.80 | 0.14 | 0.41 | 3/3 |
| Qwen3.5-4B | +1.53 | 0.44 | 0.81 | 3/3 |

**The fix, and the lesson.** `src/direction_effect.py` now computes the
contrast within each mention order and derives the SE from residual
config-level spread. The paper was rewritten to lead with an **effect-size
ratio** -- the incentive against the mere payment structure, 2-12x at every
scale -- because that claim does not depend on any SE construction and so
cannot be broken this way again.

This is the third retraction in the project and the second of exactly this
kind: a summary statistic that folded a systematic factor into the noise term.
The first was section 4.5's equal-pay ratio, which cancelled by construction.
Both looked cleanest when they were most wrong.
