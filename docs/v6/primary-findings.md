# V6-ES-1 primary findings

This is a descriptive report of the pinned Qwen3.6-27B run. The dataset,
model revision, and family gates were frozen before execution. These results do
not establish a causal mechanism, deception, or a general safety property.

## What ran

| Measurement | Locked result | Interpretation |
|---|---:|---|
| Black-box generation | 0.931 accuracy, 1.000 parse rate | Free text can be parsed reliably on the locked rows. |
| Forced-choice logits | 0.945 pooled accuracy | Pooled accuracy hides family failures. |
| Trajectory controls | 32 core rows | Order effects are large enough to motivate a dedicated follow-up. |
| Activation discovery | Not run | Stopped by the preregistered behavioral family gate. |

## The main dissociation

The model handled explicit provenance claims perfectly: `provenance` accuracy
and semantic claim-content invariance were both `1.000` across source, truth
label, time, and surface. The same model did not reliably report the
agent-indexed state when the claim had to be reconstructed from the role
description: `higher_order` accuracy was `0.454`, with `self_belief` `0.472`,
`modeled_other_belief` `0.500`, and `other_model_of_self` `0.389`.

This is a useful control result. It argues against the simple explanation
“the model cannot read the nonce concepts or the report labels.” It is
consistent with a narrower failure to maintain or retrieve role-indexed state,
but it is also consistent with target wording, recursion load, or prompt-length
effects.

## Strategic results

In `core_tom`, hidden actual-belief invariance was `1.000`, so the run did not
show leakage from the hidden oracle field. However, paired modeled-belief
identification was only `0.625` (`30/48` clusters), below the `0.75` gate. On
the locked split it was `6/16`. The descriptive split by receiver policy was
`0.846` for literal games versus `0.364` for contrarian games across all splits.
That policy contrast is not itself causal because policy was not flipped within
the same payoff game; it is a strong candidate for the next controlled test.

The safety/evaluation family showed a related dissociation. Evaluation framing
alone produced `0.000` ordinary/audited discordance, while the explicit safety
penalty switch was only `0.429` (`3/7` eligible pairs). Thus this run gives no
evidence of audit-sensitive deception. It instead suggests that payoff-neutral
evaluation wording was ignored, while explicit risk reweighting was not yet
reliably implemented.

## Trajectory result: answer scaffolding

The direct locked core baselines were weak for reports (`12/32`, `0.375`) and
actions (`22/32`, `0.688`). Revealing the model's previous action before asking
for a report made the report correct on `32/32` rows: 20 direct errors were
corrected and no correct reports regressed. Revealing the model's previous
report before asking for an action raised action accuracy to `29/32` (`0.906`):
9 action errors were corrected and 2 correct actions regressed.

This is not evidence that the first answer caused a belief update. The previous
answer may simply scaffold the label mapping, repeat task structure, or expose a
useful intermediate token sequence. It is a novel, cheap hypothesis: these
models may be able to mutually bootstrap action/report outputs in a trajectory
without maintaining a stable agent-indexed ledger.

## High-value follow-ups

1. **Within-game policy counterfactual.** Clone every payoff game and flip only
   literal versus contrarian receiver policy. Keep concept labels, response
   labels, action order, and token length balanced. Require the same modeled-
   belief switch and add a policy-description permutation control. This tests
   whether the policy contrast is genuine rather than a game-template mix.

2. **Scaffold source control.** Cross the preceding answer with four sources:
   model-generated, oracle-correct, oracle-wrong, and random legal label. Run
   action→report and report→action with matched token budgets. A gain only for
   model-generated answers would support self-scaffolding; a gain for any label
   would indicate formatting or in-context priming.

3. **Recursion and access ladder.** Replace the verbose visibility sentences
   with token-length-matched private/public/common-knowledge codes. Cross
   recursion depth 0/1/2 with target identity and role permutation. This
   separates higher-order retrieval from verbosity and publicness wording.

4. **Safety dose-response.** Sweep the explicit risk penalty around the exact
   action-switch boundary, then cross it with ordinary/audited context and
   uncertainty about the receiver policy. Report logit margins and action
   switches, not just accuracy. This separates failure to parse a safety clause
   from failure to reweight a parsed hazard.

5. **Mechanistic reopening only after behavioral repair.** If a new frozen
   protocol clears the family gates, use discovery/validation/locked probes with
   unrelated-game, same-role/different-value, retention, and output-KL controls.
   Do not use the current readable provenance representation as a causal target.

The raw records, hashes, costs, and workflow IDs are in
[`../../results/v6_epistemic_strategic/`](../../results/v6_epistemic_strategic/).
