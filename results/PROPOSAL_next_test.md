# PROPOSAL (draft, not launched, not a finding): inversion depth vs semantic attraction

Status: hypothesis-stage. No GPU run is authorised or started. All numbers cited are existing repo results.

## What already exists (so this does not redo it)
Read depth: repo FINDINGS and preregistrations FULL for v5 RBG-2/4/6 and semantic_action_mechanisms (summary level for later sections).
- RBG-4 (frozen; thinking off, constrained argmax): opaque tokens 384/384 correct; opposed meaningful assertion 79.17% -> 55.21% with redundant history. A consequence table removes the history harm; a prose rule amplifies it.
- RBG-6: letting the model write both consequence reports first rescues action to 95.83% (swapped-report replay reverses 43/48), so written intermediate steps are already known to fix it here.
- Semantic action mechanisms: irrelevant private-state sentence changes action (31% opposed vs 100% matching). No neural patch site qualified.
=> The opaque control already shows that a single contrarian inversion is NOT hard when no wording attracts. So "inversion is hard in one pass" cannot be the whole story at k=1. Remaining open question: does the opaque condition degrade as inversions stack (composition depth), and is the meaningful-wording gap a separate factor?

## Prior art (depth)
- Balesni, Korbak, Evans, 2411.16353 (PART: abstract + intro + Sections 2-3 text): fully synthetic two-hop facts fail without CoT, succeed when facts co-occur or one is natural. Lesson: guard against shortcuts and artifact failures. Overlap: composition without CoT. Difference: ours is action-policy composition with provided in-context rules, not learned facts.
- Biran et al., Hopping Too Late (ABS+PART: abstract, intro): bridge resolved early, second hop late, back-patching helps. Overlap: depth limits. Not tested here (no internals).
- Pfau et al. filler tokens, Press et al. compositionality gap: seen via citation in 2411.16353 only (ABS-cited).
Not claimed: that this is novel. In-context rule composition depth on a frontier-size open model may overlap unread work.

## Competing explanations
H_depth: accuracy falls with the number of stacked inversions even for opaque tokens.
H_attract: errors come from the meaningful claim matching the goal; opaque stays near 100% at any depth.
H_parity: model counts "flips" mod 2 (closed form), no composition. Predicts accuracy depends only on net parity, not on k.
H_literal: model picks the literal-matching claim; correct only when net map is identity.

## Design (to lock before running)
Chain of 3 hop statements fixed in all items (equal proposition count and near-equal length): each hop is either "same" or "flipped" (relay says the opposite). Vary number of flips f in {0,1,2,3}; net map is inversion iff f odd. Each hop statement carries equal-length non-flip filler text in "same" hops so token length is matched within +-5%.
Factors: f (0-3) x token type (opaque vs meaningful claim) x label/list order (balanced) x 32 fresh games. Roughly 1024 prompts, forward-only, no generation, thinking off, constrained argmax (matches the frozen readout).
Checks:
1. Expected action computed by two independent implementations (symbolic composition and a brute-force payoff simulator); disagreement aborts.
2. Closed-form baseline: report accuracy of a "count flips mod 2" oracle (100% by construction) and of "literal" and "always first" baselines on the same items.
3. Position, class balance, per-class recall reported.
4. Separate per-hop one-step accuracy (single hop prompts) as competence gate. Failing the gate gates interpretation only.
Predicted signatures (locked before run):
- f=1 vs f=3 (both net inversion): H_depth predicts f=3 lower; H_parity predicts equal.
- f=2 (net identity): H_literal predicts high accuracy; H_depth predicts a drop vs f=0.
- Meaningful minus opaque gap constant across f supports a separate attraction factor; gap growing with f supports interaction.
Interpretation limits: one model (Qwen3.6-27B), one generator, forced-choice readout. Result would describe behaviour only. A null does not exclude other depth manipulations. No mechanism claim.

## Cost
Forward-only ~1024 prompts, est $0.2-0.4 including model load (estimate; bill unread). Needs main's alignment before launch. Budget: user stated $6.17 remaining at 06:56:38; est spent ~$11.3 of $18.29.
