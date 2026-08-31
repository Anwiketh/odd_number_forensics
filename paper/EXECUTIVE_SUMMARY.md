# What is the Odd Number environment actually measuring?

## The problem

The *Odd Number* environment from **Concrete Problems in Model Forensics** is a
standard first probe for reward hacking: instruct an **even** digit, add a block
paying for **odd**, observe that the model answers odd, and read that as reward
beating instruction. I audited the instrument, not the model. What it
reports is dominated by things that are not the incentive.

**Setup.** Four binary conflict environments; six models, 0.6B to 9B (Qwen3-0.6B,
Qwen3.5-0.8B, Qwen2.5-1.5B, Qwen3.5-2B/4B/9B). Each decision is forced into one
token, so behaviour is a logit difference: CLD = log P(disobey) − log P(obey),
answer-set mass 0.994. 24 surface variants per cell, bootstrap CIs, both arms run.

## Findings

**1. Nobody runs the mirror arm.** Running both splits the effect into a
polarity-invariant part δ and a content bias β that would have appeared whichever
way the incentive pointed. β is significant in 22 of 28 cells, median κ = 0.21,
and its *sign* disagrees across models on the canonical environment, so it must
be measured, not tabulated. The same model with the same incentive
disobeys 92% or 42% depending on which arm you ran.

**2. The incentive is the smallest term in δ.** Holding the two answers fixed, I
varied only what the block *says*: payout direction, crossed with
whether an option pays zero, crossed with which is named first, plus an
equal-payout control carrying no incentive (3600 forwards per model). In nats,
each row against its own baseline:

| | 0.6B | 0.8B | 1.5B | 2B | 4B |
|---|---|---|---|---|---|
| **the incentive** (pays more for disobeying) | +0.38 | +0.39 | +0.89 | +0.80 | **+1.53** |
| **payment structure** (only disobedient paid) | +4.39 | +0.90 | +6.18 | +2.06 | +5.46 |
| **word order alone** (disobedient named first) | **+4.60** | −0.09 | +1.21 | +0.16 | +0.45 |

The quantity the environment is named after is 2 to 12 times smaller than the
payment structure at every size, and at 0.6B an order of magnitude smaller than
clause order. Amounts barely register: a regression knowing only *which* options
are paid and which is named first reaches R² = 0.73 to 0.94, and the point values
add 0.0 to 0.9 percentage points (F ≤ 1.10, n.s.) below 4B.

**3. What scales is comprehension, not incentive-following.** Magnitude
sensitivity inside a reward frame, as a fraction of the same model's sensitivity
asked directly, climbs 0%, 9%, 13%, 29%, 80%, 75% across the six models, and the
word-order artifact collapses after 0.6B. A larger model is a cleaner instrument
measuring the same wrong thing.

**4. The mechanism, and a failure.** Mirror-arm difference-in-means directions
separate: the polarity-invariant one transfers across environments sharing no
content (ρ = 0.49, chance 0.025), content directions sit at chance (ρ = 0.04).
The causal follow-up does not confirm this. Leave-one-out steering beats a
norm-matched random control (+2.96 vs. −0.11), but the *content* direction steers
about as well (+2.50), and answer-set mass falls to 0.56 where effects are
largest, so the intervention is partly breaking the model.

## Takeaway

The environment reports "the model reward hacks" when it mostly measures that one
of two named options was flagged as paid and, on small models, named first. The
one-cell fix: pay both answers the same. If that moves your model as far as a
real incentive does, you are not measuring incentive-following.

## Biggest limitations

CPU-only, single-token decisions, no rollouts, contentless environments. The six
models span three families, so no step isolates scale. The 9B factorial
was killed under memory thrashing, and §4's causal test failed. I retracted my
own headline three times: twice for over-generalising, once for a standard error
that pooled a systematic factor into the noise term and invented a threshold that
was not there. All three are in `AUDIT.md`.
