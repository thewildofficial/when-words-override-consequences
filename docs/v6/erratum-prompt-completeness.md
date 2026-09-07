# V6-ES-1 erratum: prompt completeness and trajectory confounds

**Date:** 2026-09-07
**Status:** affects interpretation of the frozen V6 run; does not alter its raw artifacts

This erratum records two implementation defects found during an independent
stimulus audit after the V6-ES-1 run. The committed dataset, model outputs,
hashes, and cost records remain immutable. The affected endpoints are
quarantined rather than silently repaired under the original study ID.

## 1. Higher-order reports did not render the queried values

The `higher_order` report renderer named the requested target—such as “your own
belief” or “your model of the receiver's belief”—and rendered the two answer
options. It did **not** render the values of the sender's own belief, modeled
receiver belief, or receiver-model-of-sender belief. Those values appeared only
in the hidden row certificate used by the scorer.

Consequently, the reported `higher_order` accuracy (`0.454`) is not evidence
that the model failed to maintain or retrieve role-indexed beliefs. The task,
as rendered, asked the model to recover an unprovided binary value. The
visibility, target, and report-label controls do not fix this because the
target value itself is absent from the stimulus.

**Disposition:** quarantine the V6 higher-order result. It is not a ToM result,
and it cannot be used as a baseline for a mechanistic or behavioral claim.

## 2. The action-to-report trajectory was not information matched

The direct-report condition contains only the report prompt. The
`action_then_report` condition first presents the action prompt. That action
prompt explicitly states the modeled receiver-belief value and the derived
receiver response before the model is asked for the report.

The apparent `12/32 = 37.5%` direct-report to `32/32 = 100%`
action-to-report improvement therefore compares an unprovided target with a
target supplied in the preceding context. It cannot identify self-generated
answer scaffolding, action-induced belief updating, or mutual bootstrapping.

The underlying outputs remain a valid record of what the two transcripts did;
the causal interpretation does not survive the prompt audit.

**Disposition:** retract the action-to-report bootstrapping interpretation. The
report-to-action direction remains a descriptive order effect only and requires
oracle-correct, oracle-wrong, random-legal, and token-matched neutral controls.

## 3. Core strategic ToM was not isolated

The V6 action prompt supplied the derived receiver response in addition to the
receiver policy and the modeled belief. Thus flipping modeled belief changed
both the intended epistemic input and an explicitly supplied downstream
variable. Pair identification cannot establish that the model composed

```
modeled receiver belief -> receiver policy -> predicted response -> own action
```

rather than simply using the displayed predicted response. The literal versus
contrarian contrast is therefore a useful lead, not a clean ToM finding.

**Disposition:** do not claim agent-indexed strategic use from V6 `core_tom`.
V6.1 removes the derived response and adds within-game policy flips.

## 4. Safety-pair count was too small

The V6 generator required that a risk penalty switch the optimum for *some*
receiver belief, but `strategy_pressure` tested only one modeled belief per
game. Only seven eligible risk-switch pairs remained. That is a descriptive
signal, not a well-powered safety reweighting gate.

V6.1 makes every safety game an identifying game at the tested belief and
reports the eligible cluster count before model execution.

## Corrective rule for V6.1

Before any paid run, every family must pass a semantic stimulus audit that
checks more than hidden certificates:

1. every expected answer is derivable from information visibly present in the
   rendered prompt;
2. changing the target latent changes the visible prompt;
3. changing a declared nuisance latent leaves the intended task content
   invariant;
4. removing the target field makes the certificate non-derivable; and
5. no derived downstream answer is rendered when the hypothesis tests whether
   the model should compute that answer.

The corrected protocol is assigned a new namespace, `V6.1-ES-2`, and will not
reuse V6-ES-1 result files.
