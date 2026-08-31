# Application form answers

Drafts for the summary questions. The form's exact wording may differ — these
map onto "what you did / what you found / why it's interesting / biggest
limitations". Every number here is reproduced from `results/` and cross-checked
in `AUDIT.md`.

---

## What did you do? (project summary)

I audited the *Odd Number* reward-hacking environment from **Concrete Problems
in Model Forensics** — instruct the model to output an even digit, add a scoring
block paying for odd — and asked what the probe actually measures, rather than
what the model does.

Six open-weight models spanning 0.6B–9B (**Qwen3-0.6B**, **Qwen3.5-0.8B**,
**Qwen2.5-1.5B-Instruct**, **Qwen3.5-2B/4B/9B**), four binary
conflict environments (`parity`, `magnitude`, YES/NO, A/B), everything on CPU. I
force each decision into a single token and read a logit difference
(CLD = log P(disobey) − log P(obey)), so a design that would need thousands of
rollouts needs thousands of forward passes and the metric is continuous. 24
surface variants per cell, bootstrap CIs over variants, and — the thing nobody
does — **both arms of every environment**, i.e. also instruct *odd* and pay for
*even*.

The main experiment holds the two answers fixed and varies only what the scoring
block *says about them*, crossing payout direction × whether an option pays zero
× which option is named first, plus an equal-payout control that carries no
incentive at all. 3600 forward passes per model.

## What did you find?

**1. Running the mirror arm splits the effect, and the second half is big.**
Polarity-invariant δ vs content bias β. β is significant in 22 of 28 cells,
median κ = 0.21. Qwen3-0.6B on canonical `parity` reports +8.57 nats and 92%
disobedience in one arm, +4.30 and 42% in the other — same model, same incentive,
opposite conclusions. β's sign disagrees between models on exactly the canonical
environment, so it can't be tabulated and subtracted.

**2. The incentive is the smallest term in δ — the main result.** Holding the two
answers fixed and varying only what the block says about them, across six models
from 0.6B to 9B (all in nats):

| | 0.6B | 0.8B | 1.5B | 2B | 4B |
|---|---|---|---|---|---|
| the incentive (pays more for disobeying vs obeying) | +0.38 | +0.39 | +0.89 | +0.80 | +1.53 |
| mere payment structure (only disobedient paid vs both) | +4.39 | +0.90 | +6.18 | +2.06 | +5.46 |
| word order alone | +4.60 | −0.09 | +1.21 | +0.16 | +0.45 |

The thing the environment is named after is **2–12× smaller than the mere fact of
which option is flagged as paid, at every size tested** — and on the smallest
model, an order of magnitude smaller than the order of two clauses in one
sentence. Amounts barely register: a regression knowing only which options are
paid and which is named first gets R² = 0.73–0.94, and the actual point values add
+0.0–0.9 pp (F ≤ 1.10, n.s.) below 4B.

**3. What scales is comprehension, not incentive-following.** Magnitude
sensitivity inside a reward frame, as a fraction of the same model's sensitivity
asked directly, climbs **0% → 9% → 13% → 29% → 80% → 75%** across
0.6B/0.8B/1.5B/2B/4B/9B, and the word-order artifact collapses after 0.6B. So a
bigger model gives you a cleaner instrument measuring mostly the same wrong
thing.

**4. The representation result, whose causal test fails.** Mirror-arm
difference-in-means directions separate: the polarity-invariant one transfers
across environments sharing no content (ρ = 0.49, chance 0.025), content
components sit at chance (ρ = 0.04). But leave-one-out **steering does not confirm
it** — the content direction steers as well as the conflict direction (+2.50 vs
+2.96, random −0.11), and answer-set mass collapses to 0.56–0.61 where effects are
largest, so the intervention is partly breaking the model. I report both.

## Why is it interesting?

The environment is in active use as a first-pass reward-hacking probe, and what
it mostly reports is that one of two named options was flagged as paid — plus, on
small models, which one was mentioned first. "The model answered odd" is being
read as goal-directed reward-seeking when the incentive is the smallest of the
three effects driving it at every size I tested.

It also fails in a way that doesn't announce itself. A bigger model gives you
better reward comprehension (0% → 80%) and kills the word-order artifact, so the
instrument looks like it's improving — while the incentive's own contribution
grows only fourfold and stays the smallest term. You get a cleaner measurement of
mostly the wrong thing, which is worse than an obviously broken one.

The fixes are cheap and generalise: run the mirror arm (2× cost), counterbalance
which option is named first, and — the one that decides it — **set both payouts
equal as a null condition**. One extra cell.

There's a methodological point past this environment too: any counterfactual whose
manipulated variable has an intrinsic direction, and "which side does the
incentive point at" always does, is confounded with content unless that direction
is counterbalanced. MCQ evals already counterbalance option order for exactly
this reason; instruction-conflict environments mostly don't.

## Biggest limitations

**Three families across six models.** 0.6B is Qwen3, 1.5B is Qwen2.5, and
0.8B/2B/4B/9B are Qwen3.5, so the size axis is confounded with training
differences. I'm confident about the *ordering* of the three effects within each
model and much less confident about the shape of any curve across them — the
payment-structure row is visibly non-monotonic and I think that's family, not
scale.

**The 9B factorial was killed.** 18GB of weights on a 31GB box left 0.9GB free,
making it 5.2× slower per forward than 4B against a 2.25× parameter ratio, with
~7.5h remaining and an OOM risk that would have written nothing. So the factorial
covers 0.6B–4B and comprehension covers 0.6B–9B.

**§8's causal test failed** and I report it as unresolved rather than dropping it.

**Everything is single-token, no rollouts, contentless toy environments.**

## What did you check, and what did you get wrong?

A lot, and I think this is the most useful thing I can show you. **I retracted my
own headline three times.**

**First, over-generalising.** I concluded models read an in-context reward as a
one-bit predicate rather than a quantity. True of the two models under 2B I had at
the time. I ran Qwen3.5-4B *because* "your models are too dumb" was the strongest
objection to my own result, and it refuted the claim.

**Second, a statistic that cancelled by construction.** To test whether the
incentive mattered at all, I compared δ under a real incentive against δ under an
equal-payout block, and got a clean null in 9 of 9 cells with CIs that tightened
with scale. It's broken: equal pay is the arithmetic midpoint of the two payout
directions, so averaging over direction cancels the effect and the contrast
returns ≈0 regardless of the truth.

**Third, a standard error that invented a threshold.** I then claimed
incentive-following "switches on near 2B", from a count of 0 significant cells
below 2B and 3/3 at 4B. Also an artifact: my SE pooled the four
(config × mention-order) cells per side, but mention order is a large *systematic*
effect (+4.60 nats at 0.6B), so the noise term absorbed it and the SE inflated up
to 8× — worst on exactly the models with the biggest order effects. Blocking on
order gives 1/3, 3/3, 2/3, 3/3, 3/3 across 0.6B–4B: no threshold. I caught it
because Qwen3.5-0.8B happens to have a near-zero word-order artifact, came back
with CIs 40× tighter than the 0.6B model's, and made "small models are noisy" look
like a family effect rather than a scale one — which didn't make sense.

Both broken statistics produced unusually *clean* results — a null with no
exceptions, a threshold with no exceptions. That cleanliness was the tell. The
paper now leads on an effect-size ratio rather than any significance count,
because a ratio of two measured means can't be broken this way.

**I also found a live defect in my own experiment code.** `exp20`'s scheme table
claimed to hold reward semantics fixed while varying numerals, but
`Task.incentive(side, hi, lo)` attaches `hi` to the side it's handed, so five of
ten schemes silently paid *more for obeying*. The main experiment exists because
of that bug — the accident revealed δ barely moved when the incentive reversed,
which prompted the properly crossed factorial.

**And 14 numbers in an earlier draft contradicted my own results**, including a
headline p("1") = 0.48 that appears in no result file (true value 0.71). Full
record in `AUDIT.md`, six rounds, with pre-audit copies preserved.

Other checks: patching machinery passes 4/4 including a self-patch no-op that
catches the HuggingFace `output_hidden_states` trap; the stack was rebuilt from
scratch and an existing experiment reproduced its stored values before any new
result was generated; raw prompts for every condition are saved and I read them by
hand before running.

## What are 1–3 pieces of evidence that you'd be able to do good research?

> **[Write this yourself — I can't fill it in for you.]** This asks about your
> background, not the project. Good candidates: open-source projects, prior
> research or blog posts, things you built or shipped, relevant coursework or
> work projects. The doc explicitly says non-standard credentials are welcome
> and asks you to explain *why* each is relevant.
>
> If you want to point at this project, the strongest framing is the §12 /
> audit story: finding a confound in your own experiment, and turning it into
> the paper's main result, is a better signal than any individual number here.

---

### Time

The doc asks you to state your own hours honestly and suggests a Toggl
screenshot. Not counted: general prep, generic tech setup (installing Python and
torch), and time waiting on jobs while doing something else. Counted:
experimental design, analysis, thinking, and writing this up.
