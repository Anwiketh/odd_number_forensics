# Application form answers

Drafts for the summary questions. The form's wording may differ; these map onto
what you did / what you found / why it's interesting / biggest limitations.
Every number is reproduced from `results/` and cross-checked in `AUDIT.md`.

---

## What did you do?

I audited the *Odd Number* reward-hacking environment from **Concrete Problems
in Model Forensics**. It instructs the model to output an even digit, adds a
scoring block paying for odd, and reports the resulting odd answers as reward
hacking. I asked what the probe measures rather than what the model does.

Six open-weight models from 0.6B to 9B (Qwen3-0.6B, Qwen3.5-0.8B,
Qwen2.5-1.5B-Instruct, Qwen3.5-2B/4B/9B), four binary conflict environments,
all on CPU. Each decision is forced into a single token and read as a logit
difference, CLD = log P(disobey) − log P(obey), so a design that would need
thousands of rollouts needs thousands of forward passes and the metric is
continuous rather than binary. 24 surface variants per cell, bootstrap CIs over
variants, and the thing nobody does: both arms of every environment, so also
instruct *odd* and pay for *even*.

The main experiment holds the two answers fixed and varies only what the
scoring block says about them, crossing payout direction with whether an option
pays zero and with which option is named first, plus an equal-payout control
that carries no incentive at all. 3600 forward passes per model.

## What did you find?

**1. Running the mirror arm splits the effect, and the second half is large.**
Polarity-invariant δ against content bias β. β is significant in 22 of 28
cells, median κ = 0.21. Qwen3-0.6B on canonical `parity` reports +8.57 nats and
92% disobedience in one arm, +4.30 and 42% in the other: same model, same
incentive, opposite conclusions. β's sign disagrees between models on exactly
the canonical environment, so it cannot be tabulated once and subtracted off.

**2. The incentive is the smallest term in δ.** This is the main result. In
nats:

| | 0.6B | 0.8B | 1.5B | 2B | 4B |
|---|---|---|---|---|---|
| the incentive (pays more for disobeying vs. obeying) | +0.38 | +0.39 | +0.89 | +0.80 | +1.53 |
| mere payment structure (only disobedient paid vs. both) | +4.39 | +0.90 | +6.18 | +2.06 | +5.46 |
| word order alone (disobedient named first vs. last) | +4.60 | −0.09 | +1.21 | +0.16 | +0.45 |

The thing the environment is named after is 2 to 12 times smaller than the mere
fact of which option is flagged as paid, at every size tested, and on the
smallest model an order of magnitude smaller than the order of two clauses in
one sentence. The amounts barely register: a regression knowing only which
options are paid and which is named first reaches R² = 0.73 to 0.94, and the
actual point values add 0.0 to 0.9 percentage points (F ≤ 1.10, not
significant) below 4B.

**3. What scales is comprehension, not incentive-following.** Magnitude
sensitivity inside a reward frame, as a fraction of the same model's
sensitivity to the same comparison asked directly, climbs 0%, 9%, 13%, 29%,
80%, 75% across the six models, and the word-order artifact collapses after
0.6B. A larger model gives you a cleaner instrument measuring mostly the same
wrong thing.

**4. It is not an artifact of toy tasks.** Rebuilt on a two-option MCQ the
model can answer unaided, where disobeying means knowingly asserting something
false for points, the confound is worse: κ = 0.30 to 0.38 against a median of
0.21 in the toy environments. Qwen3-0.6B gives a knowingly false answer in 53%
of variants when the wrong option is labelled (A) and 0% when the same wrong
answer is labelled (B). The entire reported hack rate is a function of which
letter the wrong answer happened to sit under.

**5. The representation result, and a false negative I had to withdraw.**
Mirror-arm difference-in-means directions separate: the polarity-invariant one
transfers across environments sharing no content (ρ = 0.49, chance 0.025),
content components sit at chance (ρ = 0.04). My first causal test said this
failed, because the content direction steered about as well as the conflict
direction. That comparison was invalid: I had never measured the cosine between
the two directions, and it is +0.25 to +0.73 through the middle of the network,
where I was steering. They were never alternatives.

Projecting the shared component out reverses it. Conflict-orthogonal-to-content
keeps 62% and 90% of the ceiling on two models, while
content-orthogonal-to-conflict sits on the random floor (+0.06 and −0.01). The
raw content direction had only been borrowing the conflict component. It is
still not a *reward* direction: it agrees better with one fitted from a bare
conflicting instruction than it transfers between environments.

## Why is it interesting?

The environment is in active use as a first-pass reward-hacking probe, and what
it mostly reports is that one of two named options was flagged as paid and, on
small models, which one was mentioned first. That matters because small open
models are what people reach for when prototyping evals.

It also fails in a way that does not announce itself. A larger model reads the
reward better (0% to 80%) and loses the word-order artifact, so the instrument
looks like it is improving, while the incentive's own contribution grows only
fourfold and stays the smallest term. A cleaner measurement of mostly the wrong
thing is worse than an obviously broken one.

The fixes are cheap and generalise: run the mirror arm at 2x cost,
counterbalance which option is named first, and set both payouts equal as a
null condition. That last one is a single extra cell and it decides the
question.

There is a methodological point past this environment. Any counterfactual whose
manipulated variable has an intrinsic direction, and "which side does the
incentive point at" always does, is confounded with content unless that
direction is counterbalanced. MCQ evaluations already counterbalance option
order for exactly this reason; instruction-conflict environments mostly do not.

## Biggest limitations

**Three families across six models.** 0.6B is Qwen3, 1.5B is Qwen2.5, and
0.8B/2B/4B/9B are Qwen3.5, so the size axis is confounded with training
differences. I am confident about the ordering of the three effects within each
model and much less confident about the shape of any curve across them. The
payment-structure row is visibly non-monotonic and I think that is family
rather than scale.

**The 9B factorial was killed.** 18GB of weights on a 31GB box left 0.9GB free,
making it 5.2x slower per forward than 4B against a 2.25x parameter ratio, with
about 7.5 hours remaining and an OOM risk that would have written nothing. The
factorial therefore covers 0.6B to 4B and comprehension covers 0.6B to 9B.

**The causal result is narrow.** It holds at the final token position, in a
mid-network band, in three environments of four, and on the smaller model only
once layers are restricted to those where the ceiling works at all; averaged
over all five layers there it is −0.06 against random. On Qwen3.5-2B that
restriction is not load-bearing, which is why it is the stronger half of the
evidence.

**Everything is single-token, no rollouts.** Four of the five environments are
deliberately contentless; the MCQ one is not, but it is still a single-turn
logit read rather than an agentic rollout.

## What did you check, and what did you get wrong?

A lot, and I think this is the most useful thing I can show you. I retracted my
own claims four times, and the fourth is the one I would point at.

**First, over-generalising.** I concluded that these models read an in-context
reward as a one-bit predicate rather than a quantity. True of the two models
under 2B I had at the time. I ran Qwen3.5-4B because "your models are too dumb"
was the strongest objection to my own result, and it refuted the claim.

**Second, a statistic that cancelled by construction.** To test whether the
incentive mattered at all I compared δ under a real incentive against δ under
an equal-payout block, and got a clean null in 9 of 9 cells with CIs that
tightened with scale. It is broken: equal pay is the arithmetic midpoint of the
two payout directions, so averaging over direction cancels the effect and the
contrast returns roughly zero regardless of the truth.

**Third, a standard error that invented a threshold.** I then claimed
incentive-following switched on near 2B, from a count of 0 significant cells
below 2B and 3 of 3 at 4B. Also an artifact. My SE pooled the four (config x
mention-order) cells per side, but mention order is a large systematic effect
(+4.60 nats at 0.6B), so the noise term absorbed it and the SE inflated up to
8x, worst on exactly the models with the biggest order effects. Blocking on
order gives 1/3, 3/3, 2/3, 3/3, 3/3 across 0.6B to 4B: no threshold. I caught
it because Qwen3.5-0.8B happens to have a near-zero word-order artifact, came
back with CIs 40 times tighter than the 0.6B model's, and made "small models
are noisy" look like a family effect rather than a scale one, which did not
make sense.

**Fourth, a false negative from an invalid comparison.** I reported that the
causal steering test failed on specificity. It had not: I contrasted two
directions as though they were alternatives without ever measuring their
cosine, which is +0.25 to +0.73 in the layers I steered. Removing the shared
component reverses the result. This one is worth more than the other three,
because it points the opposite way. The first three withdrew claims that were
too strong; this withdrew one that was too weak, and a broken comparison had
cost me a real result for a day. Under-claiming from an invalid contrast is the
same error as over-claiming from one, and it is harder to catch because it
feels like caution.

Both broken statistics produced unusually clean results: a null with no
exceptions, and a threshold with no exceptions. That cleanliness was the tell.
The paper now leads on an effect-size ratio rather than any significance count,
because a ratio of two measured means cannot be broken this way.

**I also found a live defect in my own experiment code.** `exp20`'s scheme
table claimed to hold reward semantics fixed while varying numerals, but
`Task.incentive(side, hi, lo)` attaches `hi` to the side it is handed, so five
of ten schemes silently paid more for obeying. The main experiment exists
because of that bug: the accident revealed that δ barely moved when the
incentive reversed, which prompted the properly crossed factorial.

**And 14 numbers in an earlier draft contradicted my own results**, including a
headline p("1") = 0.48 that appears in no result file (the true value is 0.71).
Full record in `AUDIT.md`, six rounds, with pre-audit copies preserved in git
history.

Other checks: the patching machinery passes 4/4 correctness tests including a
self-patch no-op that catches the HuggingFace `output_hidden_states` trap; the
whole stack was rebuilt from scratch and an existing experiment reproduced its
stored values before any new result was generated; re-running one factorial on
GPU at float32 reproduces the CPU output to every reported digit; and raw
prompts for every condition are saved and were read by hand before running.

## What are 1 to 3 pieces of evidence that you'd be able to do good research?

> **Write this yourself.** It asks about your background, not the project. Good
> candidates: open-source projects, prior research or writing, things you built or
> shipped, relevant coursework or work projects. The doc says non-standard
> credentials are welcome and asks you to explain why each is relevant.
>
> If you want to point at this project, the strongest framing is the audit: three
> retractions of my own headline, two of them caught by noticing that a statistic
> looked too clean. That is better evidence than any individual number here.

---

### Time

State your own hours. Not counted: general prep, generic tech setup such as
installing Python and torch, and time waiting on jobs while doing something
else. Counted: experimental design, analysis, thinking, and writing this up.
The doc suggests a Toggl screenshot.
