# Explaining the semantic action gap

Research synthesis and proposed discriminating experiments, 11 October 2026.
This note preserves the **pre-run rationale** written against checkout
`80787bf5382aeb7e16bd08dd8b7bc0d7977ba634`; its historical raw-data summaries
are exploratory. The subsequently completed study is documented separately in
the [final findings](../../results/semantic_action_mechanisms/FINDINGS.md) and
[run/artifact map](../../results/semantic_action_mechanisms/README.md).
Private-state and layout/binding interventions changed behavior; the fourteen-site
patch sweep selected no site, and held-out inference remained unopened.
Historical endpoints and preregistrations remain unchanged.

The leading working hypothesis is **competition between an outcome-based
computation and a context-triggered assertion shortcut**. An action query can
favor the shortcut while a consequence query elicits the correct mapping.
The unresolved question is where this competition occurs: task interpretation,
inverse lookup, utility comparison, or binding the selected action to A/B.
These explanations can coexist. None is an identified internal mechanism yet.

## Evidence that an explanation must accommodate

| Observation | Constraint on a theory |
|---|---|
| [RBG-4](../../results/v5_inverse_evidence/FINDINGS.md): opposed assertion/prose actions go from 36/48 without history to 13/48 with four correct demonstrations; opaque actions stay correct | A generic inability to understand contrarian policies or an undifferentiated context-length cost is insufficient |
| Independent consequence reports remain highly accurate | The mapping can be recovered under a consequence query; this does not establish its representation or use on the action branch |
| Table actions remain 40/48 under both history conditions | Format or explicit label binding protects thresholded accuracy; margins still fall, and individual choices do change |
| [RBG-3](../../results/v5_semantic_localization/FINDINGS.md): quoted strings and labeled buttons are correct in the history-free table scaffold, while seven assertion errors survive | Mere presence of the goal word is insufficient in that scaffold; assertion semantics or its interaction with framing matters |
| [RBG-6](../../results/v5_full_action_trajectory/FINDINGS.md): retaining correct reports strongly rescues action; swapped reports strongly misdirect it | Visible intermediate consequences can control action under the tested instruction; content, extra turns, and the instruction to use reports must be separated |
| [RBG-5B](../../results/v5_mechanistic_decomposition_b/README.md): 2/50 repairs and failed identity-margin control | The selected residual interchange did not identify selective transport; this does not exclude distributed, earlier, later, or query-dependent mechanisms |

The first two observations establish an operational report/action gap. They do
not establish that a single trajectory simultaneously computed the correct
consequences and ignored them. The action and report questions select different
computations, and reports may reconstruct the answer from the prompt.

### Additional clue: dependence on the private true state

Recomputed from the immutable RBG-4 raw file on 11 October:

| Assertion actions, prose mapping | No history | Redundant history | Mean correct-minus-wrong logit margin, before → after |
|---|---:|---:|---:|
| Target equals the private true state (`aligned`) | 44/48 | 48/48 | 3.036 → 5.398 |
| Target differs from the private true state (`opposed`) | 36/48 | 13/48 | 1.859 → −1.438 |

Each row aggregates two frames on 24 bases, drawn from 12 concept pairs; these
are not 48 independent games. This is a post-hoc descriptive contrast, with no
new significance claim. The same contrarian rule applies in both rows. For an
opposed goal the optimal assertion is **true**, because the receiver reverses it.
For an aligned goal the optimal assertion is **false**. Thus the hard condition
requires a truthful assertion to produce a false outcome.

This is compatible with a shortcut such as “to induce a false state, make a false
assertion.” It also admits other truth/goal-dependent shortcuts. A pure
goal-word-matching account predicts conflict in both conditions and needs an
additional moderator to explain this asymmetry. A blanket preference for honest
assertions is also insufficient: the aligned condition succeeds while requiring
a false assertion. Device trials show the effect too, so an exclusively social
or deception-specific explanation is insufficient.

The private true state has no causal role in the certified action→outcome map or
payoff once the target is specified. This makes it a valuable intervention:
change it while holding the correct action fixed.

Raw source: `results/v5_inverse_evidence/raw/behavior_v1.json`, SHA-256
`4e7ba1d0dc16e1fe231d7fe312e77dbede14e33e5db2eca2c6e030ed9237b4f9`.
The subsequent [layout/binding factorial](../../results/semantic_action_mechanisms/FINDINGS.md#layout-and-explicit-ab-binding-both-contribute)
separates those two treatments behaviorally. Equal aggregate accuracies alone
do not establish that individual choices or margins stay unchanged.

## What the literature contributes

This is a targeted primary-source review, checked on 11 October 2026, not a
systematic literature search or a claim of priority. Evidence in other models
and tasks motivates hypotheses for our model; it does not identify our mechanism.

| Primary paper | Relevant evidence | Implication here |
|---|---|---|
| [Wei et al., Larger language models do in-context learning differently (2023)](https://arxiv.org/abs/2303.03846) | Flipped and arbitrary labels reveal competition between semantic priors and demonstrated mappings; instruction tuning can strengthen reliance on priors | Compare meaningful, opaque, and countersemantic actions under the identical transition map; a base/instruct comparison is a later test of training origin |
| [Min et al., Rethinking the Role of Demonstrations (2022)](https://arxiv.org/abs/2202.12837) | In their classification and multiple-choice settings, demonstration format, input distribution, and label space can matter more than label correctness | Correct demonstrations can change task recognition and response habits without improving the intended computation |
| [McKenzie et al., Inverse Scaling (2023), §§3.3–3.4](https://arxiv.org/html/2306.09479v2) | Easy distractor tasks and correctly labeled but misleading demonstrations can dominate the intended task | Repetition-induced shortcut competition is plausible; our history manipulation is not itself a model-scale result |
| [Xu et al., Code over Words (2026)](https://arxiv.org/html/2601.18352v2) | In mutable-rule puzzles, models can follow familiar semantics despite conflicting rules; code-grounded representations improve their tested setting | Closely related behavioral precedent for semantic inertia; code, tables, label binding, and external execution are different treatments |
| [Sobotka et al., Why Do LLMs Struggle in Strategic Play? (2026), §§3.5–3.6](https://arxiv.org/html/2605.00226v1) | Internal-belief steering has limited action influence; externalized beliefs influence actions more strongly but do not consistently improve payoffs | Closest belief/action precedent. Test availability, use, and externalization separately rather than inferring knowledge from reports |
| [Feng and Steinhardt, How do Language Models Bind Entities in Context? (2023/ICLR 2024)](https://arxiv.org/abs/2310.17191) | Causal interventions identify representations binding entities to attributes | Read and manipulate which action causes which outcome, not just whether both outcome words are decodable |
| [Todd et al., Function Vectors (2023/ICLR 2024)](https://arxiv.org/abs/2310.15213) | Demonstrations can induce compact task representations with causal effects in studied transformers | Motivates a task-routing hypothesis; it does not establish such a vector in this Qwen checkpoint |
| [Abdelsalam et al., A Mechanistic Lens on Semantic Conflicts (2026)](https://arxiv.org/html/2607.05587v1) | Token-aligned cue/code contrasts and bidirectional patches identify sparse intermediate carrier sites before output readout | Prefer minimal counterfactual contrasts and intermediate sites to broad table→prose residual swaps. Their stimuli were selected to evoke conflict, limiting prevalence inference |
| [Zhang and Nanda, Towards Best Practices of Activation Patching (2023/ICLR 2024)](https://arxiv.org/abs/2309.16042) | Metrics and contrast/corruption choices materially change localization results | Validate the numerical instrument and freeze the causal estimand before interpreting patch success or failure |

Two tempting analogies need limits. [The Reversal Curse](https://arxiv.org/abs/2309.12288)
concerns reversing associations learned through training; its authors explicitly
distinguish in-context reversal, which models can perform. Our forward-report /
inverse-choice asymmetry therefore needs its own test. [Turpin et al.](https://arxiv.org/abs/2305.04388)
show that explanations can omit influences and rationalize biased choices;
this supports intervention-based validation, not a conclusion that our accurate
consequence reports are deceptive or that their CoT mechanism explains our gap.

## Ranked working hypotheses

| Hypothesis | Why it is plausible | Prediction that distinguishes it |
|---|---|---|
| **1. Truth/goal-dependent assertion shortcut** | Opposed and aligned goals react differently to the same mapping and history; nonassertive surfaces fare better in the localization scaffold | Changing an irrelevant private state changes action while mapping and target stay fixed; removing it reduces the asymmetry selectively for assertions |
| **2. Inverse lookup or composition is not recruited by direct choice** | Reporting an outcome of a named action is easier than finding the action for a target outcome and translating it to a label; explicit reports rescue | Forward lookup succeeds but inverse lookup/semantic choice fails; supplying consequences or utilities repairs downstream choice beyond matched neutral text |
| **3. Action-to-label binding fails** | Tables explicitly associate A/B with consequences; prose does not | Adding labels to prose helps as much as adding labels to tables; semantic action selection is correct while A/B choice remains wrong |
| **4. History changes task routing or associations** | Four correct episodes selectively hurt prose assertions; demonstrations repeat the same descriptions in a fixed sequence | History order, whether actions or outcomes are mentioned last, or label attachment predicts specific errors after counts and length are controlled |
| **5. Query/recency effects explain report rescue** | Report transcripts change both content and the effective decision prompt | Neutral extra turns or an instruction-only treatment accounts for rescue; otherwise consequence content has an additional effect |

The priority order reflects explanatory plausibility and information gained per
test, not numerical posterior probabilities. Post-training provenance and
checkpoint generality remain open. Native reasoning may improve the behavior,
but that alone would not distinguish these hypotheses.

## First experiment: irrelevant-state intervention

For each base, fix target, mapping, action strings, A/B assignments, option order,
and payoff. Vary only the private-state sentence:

1. The private state equals the target.
2. The private state is the other value.
3. The sentence is absent.
4. A neutral sentence occupies the corresponding position as a length control.

Cross assertion versus opaque surfaces and no-history versus redundant-history
conditions. Start with the established prose scaffold and original checkpoint;
confirm on fresh bases. Include a device-frame replication. Balance semantic
word identity and verify the token lengths of the state-flip pair. The missing
and neutral sentences are imperfect token/position controls, so the matched
state flip is the cleaner primary contrast.

**Primary readout:** within-base change in correct-action margin when the
irrelevant state flips, and its assertion-minus-opaque interaction. Correct
action and payoff do not change. Report accuracy and unconditional action accuracy
are secondary readouts. Any effect violates the task's specified invariance;
the surface interaction implicates assertion-related interference. It would
still not establish a specific learned circuit or its training origin.

If state removal does little and state flips are inert, lower the priority of
the truth-sensitive script and test goal matching and binding more directly.
If only removal helps, distinguish deletion/position effects from state semantics.

## Second experiment: layout versus binding, with intermediate endpoints

Use a 2×2 rendering factorial, preserving the same facts, clause order, option
presentation, and final instruction:

| | Mapping names action descriptions only | Mapping names A/B and action descriptions |
|---|---|---|
| Prose | Original-style conditional sentences | Same sentences with `Option A/B` attached |
| Table | Description→outcome rows without A/B | Original-style labeled rows |

Cross history none/redundant; use assertions for the primary diagnostic and
opaque actions as competence controls. Retain both truth/goal relations rather
than examining only the known failing cell. A minimal rendering example is:

- Unlabeled prose: `If you claim RED, the receiver chooses BLUE; if you claim BLUE, the receiver chooses RED.`
- Labeled prose: `If you choose A (claim RED), the receiver chooses BLUE; if you choose B (claim BLUE), the receiver chooses RED.`
- Unlabeled table: `claim RED → BLUE; claim BLUE → RED`.
- Labeled table: `A (claim RED) → BLUE; B (claim BLUE) → RED`.

On independent forks from each scenario, separately ask for the outcome of
each action, each action's payoff, the best action **by claim content**, and
the best A/B label. Use validated semantic-output parsing or full-sequence
scoring when answers span multiple tokens; do not compare unnormalized
multi-token string scores with single-token A/B scores.

Then test oracle-supplied consequences, utilities, and chosen semantic action as
distinct scaffold interventions, each with matched neutral-content controls and
the same final action instruction. Supplying the chosen semantic action makes
the decision easier by design; it tests the remaining binding step, not
spontaneous planning.

Interpret the factorial and endpoint pattern jointly:

- Labeled prose rescues and unlabeled tables do not: explicit binding is favored.
- Unlabeled tables also rescue: layout/structure contributes independently.
- Semantic action is wrong despite correct forward reports: failure occurs before
  final output-label binding; inverse lookup, utility selection, or a shortcut remain.
- Semantic action is right but A/B is wrong: output binding becomes the leading locus.
- Correct intermediates outperform neutral controls: targeted externalization
  matters; a broad “extra text helps” explanation is insufficient.

Different query forks establish competence under different queries, not the
presence of those intermediate computations in direct-action activations.

## Third experiment: give competing shortcuts different answers

Binary contrarian games cannot identify a goal-matching error signature: the
only wrong action necessarily has the goal's wording. Construct four-action
games with an arbitrary permutation mapping and distinct roles:

- the action whose consequence achieves the target;
- an incorrect action whose assertion text matches the target;
- an incorrect action whose assertion text matches the true state;
- an incorrect action mentioned last in the history.

Keep these roles distinct in the primary diagnostic cells, rotate labels and
positions independently, and balance action/outcome frequencies in history.
Score each error category and correct-versus-each-distractor margins. Include
same-state opaque controls so added choice complexity is measured. The opaque
condition must be competent before interpreting assertion-specific error types.

Test history order separately from frequency: permute the same correct examples;
compare balanced repeated examples with matched non-demonstration mentions and
irrelevant text. A target-matching excess, last-example dependence, and generic
performance loss now make different predictions. A broad “always lie” heuristic
does not specify a unique action when several assertions are false; do not claim
that four options alone uniquely identifies it.

## Causal mechanism work after behavioral localization

Capture the shared scenario before any diagnostic query, and capture declared
post-query states separately. Train low-capacity probes for the action→outcome
relation, target, action content, and output label with held-out vocabulary and
templates. Decodability is evidence about the readout, not causal use.

For a relation test, use a factorial of mapping and target, including pairs with
the same correct action but different mappings and pairs with the same mapping
but different correct actions. A donor mapping intervention must produce the
action predicted by **donor mapping + recipient target + recipient label map**,
which can differ from the donor's answer. This is more identifying than repair
alone and helps exclude copying a donor answer token.

Use the behavioral results to choose one small discovery intervention:
relation-binding sites if layout/binding dominates, or changed-state/history
carrier sites if task routing dominates. Freeze the intervention before fresh
confirmation. Require contemporaneous no-hook, no-op, and identity parity;
reverse, wrong-game, target-preservation, and answer-copy controls; and a
prespecified margin/repair endpoint. Match execution conditions so numerical
drift is small relative to the proposed effect. The old failed identity control
cannot be repaired by simply selecting a different interpretive metric.

For this hybrid checkpoint, do not assume that patching one residual token or
copying a standard Transformer KV cache replaces all recurrent state. A negative
result bounds the tested intervention and site, not all internal implementations.

Use bases as the paired sampling unit, with concept-family/template holdouts and
uncertainty reflecting shared lexical families. Balance label permutations within
bases; the original random assignments were not perfectly balanced. Freeze one
primary contrast per stage, size confirmation from whole-base pilot variance,
and treat remaining interactions as exploratory or adjust multiplicity. Preserve
unconditional outcomes rather than filtering primary analysis on correct reports.

The recommended first run is the irrelevant-state test, followed closely by the
layout/binding factorial. They directly manipulate two currently entangled
variables while retaining the known game. A successful explanation should predict
which manipulation changes behavior and which intermediate computation fails,
before we invest in another activation sweep.

## Reproduce the added descriptive counts

Run from the repository root; this reads raw data and prints counts without
changing any result artifact:

```python
import json
from collections import defaultdict

groups = defaultdict(list)
with open("results/v5_inverse_evidence/raw/behavior_v1.json") as f:
    for row in json.load(f)["rows"]:
        if row["surface_kind"] == "assertion" and row["mapping_format"] == "prose":
            groups[row["incentive"], row["history"]].append(row)
for key, rows in sorted(groups.items()):
    margins = []
    for row in rows:
        correct = row["expected_action"]
        wrong = "B" if correct == "A" else "A"
        logits = row["action_result"]["legal_logits"]
        margins.append(logits[correct] - logits[wrong])
    print(key, sum(r["action_correct"] for r in rows), len(rows),
          round(sum(margins) / len(margins), 3))
```
