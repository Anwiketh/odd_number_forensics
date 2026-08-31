# What is the Odd Number environment actually measuring?

## The problem

The *Odd Number* environment from **Concrete Problems in Model Forensics** is a
standard first probe for reward hacking: instruct an **even** digit, add a block
paying for **odd**, observe the model answers odd, read that as preferring reward
over instruction. I audited the instrument, not the model. The number it reports
is dominated by things that are not the incentive.

**Setup.** Four binary conflict environments plus an MCQ one; six models,
0.6B–9B (Qwen3-0.6B, Qwen3.5-0.8B, Qwen2.5-1.5B, Qwen3.5-2B/4B/9B). Decisions
forced into one token, so behaviour is a logit difference — CLD = log P(disobey)
− log P(obey), answer-set mass 0.994. 24 surface variants/cell, bootstrap CIs,
**both polarity arms always run**.

## Findings

**1. Nobody runs the mirror arm.** Running both splits the effect into
polarity-invariant **δ** and a content bias **β** that appears whichever way the
incentive points. β is significant in 22/28 cells, median κ = |β|/(|δ|+|β|) =
0.21, and its *sign* disagrees across models, so it must be measured. The same
model with the same incentive disobeys 92% or 42% depending on which arm you ran.

**2. The incentive is the smallest term in δ — the main result.** Holding the two
answers fixed, I varied only what the block *says*: payout direction × whether an
option pays zero × which option is named first, plus an equal-payout control
carrying no incentive (3600 forwards/model). In nats:

| | 0.6B | 0.8B | 1.5B | 2B | 4B |
|---|---|---|---|---|---|
| **the incentive** (pays more for disobeying vs obeying) | +0.38 | +0.39 | +0.89 | +0.80 | +1.53 |
| **mere payment structure** (only disobedient paid vs both) | +4.39 | +0.90 | +6.18 | +2.06 | +5.46 |
| **word order alone** | **+4.60** | −0.09 | +1.21 | +0.16 | +0.45 |

The quantity the environment is named after is **2–12× smaller than which option
is flagged as paid, at every size tested** — and at 0.6B, smaller than the order
of two clauses in one sentence. Amounts barely register: a regression knowing
only *which* options are paid and which is named first reaches R² = 0.73–0.94;
the point values add +0.0–0.9 pp (F ≤ 1.10, n.s.).

**3. What scales is comprehension, not incentive-following.** Magnitude
sensitivity inside a reward frame, as a fraction of the same model's sensitivity
asked directly, climbs **0% → 9% → 13% → 29% → 80% → 75%**, and the word-order
artifact collapses after 0.6B. A bigger model is a cleaner instrument measuring
mostly the same wrong thing.

**4. It is worse outside the toys.** In a two-option MCQ the model can answer
unaided, with the reward on the **wrong** option — so hacking means knowingly
asserting something false — the polarity axis *is* MCQ position bias. κ runs
**0.30–0.38**, above the 0.21 toy median. Qwen3-0.6B answers knowingly wrong in
**53%** of variants when the wrong option is labelled (A) and **0%** when it is
labelled (B).

**5. Mechanism.** Mirror-arm difference-in-means directions do separate
(ρ = 0.49 cross-environment vs 0.04 for content, chance 0.025), but leave-one-out
steering fails to confirm it: the content direction steers as well as the
conflict one and answer-set mass falls to 0.56, so the intervention is partly
breaking the model. Reported as unresolved.

## Takeaway

**The one-cell fix: pay both answers the same.** If that moves your model as far
as a real incentive, you aren't measuring incentive-following.

## Biggest limitations

Single-token, no rollouts. Six models across three families, so no step isolates
scale. The 9B factorial was killed under memory thrashing. I retracted my own
headline three times — twice for over-generalising, once for a standard error
that folded a systematic effect into the noise and invented a threshold. And
`exp17` shipped with answer-set mass 0.000, printing confident numbers; it now
refuses to report below 0.90. All in `AUDIT.md`.
