# V7-EAS-1 preregistration

## Question

When an LLM agent chooses a strategy, does it condition on the actual
epistemic state of another agent, or on the underlying facts and salient words
alone?

The identifying principle is to hold the physical world and payoff function
fixed while changing only the information relation between agents.

## Decomposition

Each game has independent diagnostic questions for:

- the physical world state;
- A's information;
- B's information;
- B's belief about A's information;
- B's policy;
- the predicted B action; and
- A's final strategy.

These are static direct queries. No report is sampled and replayed into the
action prompt in the primary condition. This makes a pattern such as
"reports are correct, opponent prediction is correct, final strategy is
wrong" directly observable without trajectory contamination.

## Family 1: publicity threshold

SAFE gives both agents +2 regardless of the other action. COMMIT gives +8 to
each when coordinated and -10 to a lone committer. B uses a fixed rule: B
chooses COMMIT only when B knows the opportunity exists and B knows A knows it.
A knows this rule.

The levels are A-private, shared-private, A-knows-B-knows, public announcement,
and explicit common knowledge. In the identifying games, A's optimal strategy
changes from SAFE to COMMIT at the public-announcement threshold. The
shared-private and public/common conditions contain the same first-order
opportunity fact.

## Family 2: credible commitment

A moves first in an entry game. A may use a channel or wait. A public real
lock makes B stay out and gives A +5; if B enters, a flexible A gets +2 while
an A genuinely locked into X gets -4. The channel itself has no immediate
payoff.

The channel is crossed between a public irreversible lock, a private
irreversible lock, a public non-binding promise, and a public tool transition.
The expected strategy is USE for the public irreversible and public tool
conditions, and WAIT for the private lock and cheap promise controls.

The local simulator contract requires the public tool transition to remove
future option Y. Textual claims that a lock occurred are never treated as
state transitions.

## Family 3: information acquisition

A chooses ACT_NOW or one of three inspections. An inspection costs one point
and perfectly reveals only its named target. A payoff is +6 for matching the
final action to the currently relevant binary target and -2 for a mismatch.
Each matched game has exactly one uncertain target: the world state, B's
belief, B's policy, or none. The target-specific inspection has positive net
value only in the corresponding uncertainty condition; inspections of known
or irrelevant targets have negative net value. In the no-uncertainty control,
A should ACT_NOW.

## Measurement

The primary measurement is candidate-token logit choice with randomized A/B
labels for binary decisions and A/B/C/D labels for acquisition decisions.
Semantic, opaque, and countersemantic action names are crossed while keeping
game-level mappings fixed within matched pairs. The model receives one direct
query at a time with no native reasoning stage.

Primary locked outcomes, reported separately by family, are:

1. report competence;
2. predicted-opponent-action competence;
3. final action correctness;
4. semantic strategy switch rate on each identifying pair;
5. regret under the exact CPU certificate; and
6. surface-name invariance.

The first screen records no remote tool trajectory. Its simulator transition
contract is audited locally for the later agentic stage.

## Interpretation boundary and stopping

The construct is functional strategic use of agent-indexed epistemic state.
The protocol makes no claim about consciousness, real beliefs, deception, or
scheming. No activations are collected and no native-reasoning stage is
enabled. The locked dataset, prompt contract, model revision, and gates are
not changed after model output is observed.
