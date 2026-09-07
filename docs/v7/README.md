# V7-EAS-1: Epistemic Control of Agentic Strategy

V7 asks whether a model uses agent-indexed epistemic state in an actual
decision. The physical state and payoffs stay fixed while the knowledge graph
between A and B changes.

The first screen is deliberately behavior-first and direct-only. It uses
Qwen3.8-27B, three frozen splits, deterministic CPU-generated games, and no
activation collection. Report questions are separate static forks from the
same scenario; they are never inserted as sampled assistant turns before an
action.

## Locked first screen

The study contains three families:

1. **Publicity threshold.** A and B coordinate on a costly action. B commits
   only when B knows the opportunity exists and knows A knows it. The matched
   contrast is shared private information versus a public announcement or
   explicit common knowledge.
2. **Credible commitment.** A can use a public irreversible lock, a private
   lock, a cheap promise, or a public tool action. Only a real public
   option-removing action changes B's response.
3. **Information acquisition.** A chooses ACT_NOW, INSPECT_WORLD,
   INSPECT_B_BELIEF, or INSPECT_B_POLICY. Exactly one target is uncertain in
   each game, and the exact value-of-information solver makes only the
   strategically relevant inspection worth its cost.

Every family has semantic, opaque, and countersemantic action-name surfaces.
Labels are randomized with a treatment-invariant game-level mapping. The
analysis reports report competence, predicted-opponent-action competence,
final action accuracy, matched semantic switches, regret, and surface
invariance separately by family. The first screen has no real tool trajectory;
the local commitment transition contract is tested but not executed remotely.

## What a positive result would mean

A positive publicity result would show a strategy switch when no new
first-order fact is learned, only the public epistemic relation changes. A
positive commitment result would show sensitivity to the game tree rather than
commitment language. A positive acquisition result would show that the model
can identify which uncertainty has decision value.

Failures are interpreted narrowly as failures of functional strategic use of
the specified epistemic state. They are not labeled consciousness failures,
deception, or scheming.

## Execution boundary

The frozen branch contains the CPU dataset and semantic audit first. The
intended workflow is:

```text
local-controls -> tokenizer preflight -> manual review -> direct behavior (with the reviewed preflight run ID)
```

The behavior dispatch must name the separately reviewed preflight run. The
behavior stage rejects a preflight artifact whose commit, config, dataset,
model specification, or full query/token contract differs from the current
checkout. The Modal function timeout and the config's cost ceiling are the
same frozen 1500-second bound. No thinking stage or activation stage is
enabled in this protocol.
