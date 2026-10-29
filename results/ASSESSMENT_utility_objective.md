# Bounded CPU assessment: what would distinguish a represented objective from belief, computation or prompt policy (07:30)
Consolidation of earlier reading (docs/literature/utility_papers_read_2026-10-29.md); no new experiment. Depth: Utility Engineering 2502.08640 PART (Sections 3-6.3 earlier; today only keyword search of Section 7, details not read); Incoherent Values 2606.21102 PART; 2605.08556 FULL earlier (today's pdf fetch returned no content, so nothing re-verified); Planted in Pretraining PART; Value Drifts PART.

## Strongest existing evidence, scoped
- Behavioural coherence of forced-choice A/B preferences rises with scale (UE: fewer cycles, Thurstonian fit, expected-utility property). Counter: 2606.21102 finds within-ladder accuracy 97.3% but strict monotonicity against external comparisons only 59.5% mean (range 10.1-80.1%).
- Linear probes on hidden states of single outcome statements predict fitted utilities, near chance for small models (UE Fig 8). This is decodability from statement text; no text-feature baseline seen in the parts read; not causal use.
- Utility control in UE (Section 7): training a model toward citizen-assembly utilities; I did not read its design, so I make no claim whether it is an internal-state intervention or fine-tuning on outputs.
- 2605.08556: elicited belief vs decision dissociate in clinical diagnosis; prompt steering reaches a target in 12-31% of settings. Behavioural, no internals.
- Planted in Pretraining: bias vectors cluster by backbone more than instruction data (two backbones, LoRA). Value Drifts: stance drift is set mostly by SFT; stances are not utilities.

## What would distinguish the readings (design criteria, not a proposal to run)
1. Represented objective vs belief: need an internal-state edit that changes choices while a belief report (consequence knowledge) stays fixed, plus the reverse. WWOC shows belief report intact with wrong action (52 trials in RBG-4) and rescue when reports are written first (RBG-6), but with no qualifying causal edit (no patch site qualified).
2. Objective vs computation: need an input-matched control where the same decodable signal does not drive the choice (probe-vs-text baseline; edit of the probed direction with per-prompt KL validity <1 nat).
3. Objective vs prompt policy: need preference measures invariant across framing, and a base-vs-post-trained comparison (2606.21102 has one GLM pair: 10.1% vs 79.5% strict monotonicity; one family, accuracy for base not reported).

## Concrete unanswered question, if any
Whether UE-style probe-decodable utilities are causally used for choice, tested against a text-feature probe baseline. I have no evidence that this is unanswered in the literature (the search was bounded; I read UE only partly), so I do not claim novelty, and with the 27B J-space edits already null in the tested grid (a null in that grid does not show no valid editor exists) I do not propose a run.
## Verdict
Already answered by earlier reading: no new direction is established; this restates the gap. Done.
