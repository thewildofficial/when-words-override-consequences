# V5 literature positioning: inverse evidence and semantic action capture

## Second-pass update

The [expanded primary-source comparison](../next-sprint/literature.md) adds work
on follow-up monitoring, latent reasoning, probe trajectories, and belief-verified
lie detection. [The code/data audit](../next-sprint/audit.md) qualifies the current
evidence: independent consequence reports do not establish an unchanged internal
state on the action trajectory; RBG-6 adds an explicit report-use instruction;
and the RBG-5B candidate readout omitted final normalization. No successful
causal localization is established. The proposed next experiment is
[content-controlled report reactivity](../next-sprint/experiments.md).

The Story Shapes the Agent and INTENT-AS-A-TOOL entries below are retained from
the previous review; direct source retrieval failed in this second pass, so
their detailed comparisons are not newly verified. The next-sprint proposal does
not depend on them.

## Closest open literature

**A Mechanistic Lens on Semantic Conflicts: Using Activation Patching to
Understand LLM Behavior** is the closest mechanistic precedent found after
RBG-4. It uses token-aligned triplets, bidirectional residual patching, raw and
normalized output-preference changes, and denominator safeguards to localize
competition between meaningful cues and executable code. RBG-5 adopts those
methodological safeguards but studies redundant correct evidence, independent
consequence reports, and naturally successful table/opaque donors in a strategic
action task.

- https://arxiv.org/abs/2607.05587

**Towards Best Practices of Activation Patching in Language Models** and **How
to use and interpret activation patching** show that localization depends on the
contrast, corruption, metric, and patch granularity. RBG-5 therefore freezes its
raw logit-margin endpoint, source/destination pairing, semantic anchors, reverse
patches, and denominator handling before locked execution.

- https://arxiv.org/abs/2309.16042
- https://arxiv.org/abs/2404.15255

**Causal Abstractions of Neural Networks** motivates using interchange
interventions to test whether a neural state has the counterfactual role of a
high-level variable. RBG-5 makes the narrower sufficiency claim licensed by a
single natural residual replacement rather than claiming a full abstraction.

- https://arxiv.org/abs/2106.02997

**Function Vectors in Large Language Models** provides a candidate account on
which demonstrations write a compact task state into middle-layer residual
processing. This motivates—but does not establish—the persistent-policy-state
hypothesis for RBG-5.

- https://arxiv.org/abs/2310.15213

Probe methodology supplies an important negative constraint. **Designing and
Interpreting Probes with Control Tasks** and **Amnesic Probing** distinguish
decodable information from information the model behaviorally uses. RBG-5 uses
group-held-out and shuffled-label controls and assigns all causal weight to the
locked interchange rather than probe accuracy.

- https://aclanthology.org/D19-1275/
- https://aclanthology.org/2021.tacl-1.10/

**Broken Links Between Observations, Beliefs, and Actions** reports that internal
game-state beliefs can exceed verbal reports and that beliefs often weakly control
best-response actions. It establishes the broad strategic belief/action gap and
first-item bias. RBG-4 instead manipulates redundant correct evidence and
representation format while holding the causal game fixed, and finds action harm
with unchanged consequence reports.

- https://arxiv.org/abs/2605.00226

**Code over Words: Overcoming Semantic Inertia via Code-Grounded Reasoning** shows
that natural-language semantic priors can override counterfactual in-context rules
in `Baba Is You`, and that code grounding mitigates the effect. RBG-4 is consistent
with semantic inertia but isolates an assertion-based strategic decision and an
inverse-evidence effect: four additional correct demonstrations worsen action, and
a table rather than executable code removes the harm.

- https://arxiv.org/abs/2601.18352

**The Story Shapes the Agent: Narrative Priors in LLM Behavior** holds decision
structure fixed and shows that narrative framing drives action patterns, sometimes
against task success. RBG-4 similarly uses structural isomorphs, but additionally
measures both option consequences and intervenes on redundant evidence and policy
format within the same base games.

- https://arxiv.org/abs/2607.18566

**Larger Language Models Do In-Context Learning Differently** studies flipped and
semantically unrelated labels, showing competition between semantic priors and
in-context mappings. RBG-4 extends this tension from classification labels to
causal strategic action under false-belief goals.

- https://arxiv.org/abs/2303.03846

## Distinctive empirical conjunction

The literature reviewed so far contains each neighboring idea—semantic inertia,
narrative action priors, in-context label conflict, and strategic belief/action
gaps—but we have not found a paper reporting this exact conjunction:

1. structurally matched strategic actions with explicit consequences;
2. independent reports showing that both consequences remain available;
3. a randomized addition of correct, redundant evidence that worsens action;
4. semantic specificity relative to opaque actions; and
5. elimination of the harm by tabularizing the same causal rule.

This is a defensible novelty target, not proof of priority. A broader systematic
search and author-level comparison are required before a paper claims “first.”

## Contribution framing

The current evidence supports the following behavioral framing:

> In the tested games, additional correct demonstrations worsen meaningful
> action accuracy while independent consequence reports remain accurate.
> Tabular/opaque formats and report-containing trajectories improve behavior.
> Semantic binding or retrieval changes are candidate explanations; the internal
> mechanism and preservation of the action trajectory's knowledge state are unproven.

This links cognitive-control/Stroop-style interference to alignment evaluation:
evaluators may add demonstrations expecting greater task understanding, yet make
meaningful strategic action less reliable even as verbal checks remain excellent.

## Action-representation invariance: tool use and scheming evaluations

The safety-facing open problem is whether a consequential model action is stable
under a semantics-preserving renaming of the available action interface. A model
that selects causal outcomes should choose the correspondingly renamed action
when names change but transitions, payoffs, information, and goals do not. A model
whose policy is captured by the semantic action surface may instead change the
outcome it selects despite accurately reporting every action's consequence.

Recent agentic-safety work makes this a live measurement question. **Evaluating
and Understanding Scheming Propensity in LLM Agents** finds large changes under
tool and oversight variations and argues that measured propensity belongs to a
model-prompt-scaffold system rather than a model in isolation. **Agentic
Misalignment in Summer 2026** holds judged transcripts fixed and reverses the
downstream training consequence of `COMPLIANT` and `NON_COMPLIANT`; several
Claude judges change their verdicts, but the study does not cross consequence
direction with opaque or countersemantic verdict names.

- https://arxiv.org/abs/2603.01608
- https://alignment.anthropic.com/2026/agentic-misalignment-summer-2026/

Tool-use work supplies the adjacent interface result without the strategic
construct-validity test. **ToolRobustBench** diagnoses failures under interface,
intent, observation, and runtime perturbations. **INTENT-AS-A-TOOL** treats the
probability of calling specially named intent tools as a fine-grained signal of
agentic misalignment. **Formal Verification of Agentic Systems over Operational
Data** identifies identifier-renaming equivariance as a condition for transferring
system properties and shows that an LLM policy can violate it. These results make
action naming scientifically important, but do not independently verify action
consequence knowledge or causally localize a semantic-versus-consequence conflict.

- https://arxiv.org/abs/2608.23635
- https://arxiv.org/abs/2608.27348
- https://arxiv.org/abs/2608.03609

The proposed bridge is therefore a crossed, outcome-scored evaluation: truthful
semantic names, opaque aliases, and countersemantic aliases for identical tool
transitions, accompanied by independent consequence reports. RBG-4 supplies the
controlled behavioral precursor; RBG-5B tested whether a naturally successful
table/opaque residual state can causally repair the matched semantic failure.
The locked causal endpoint failed, and its readout needs the normalization
correction described above. Neither study establishes scheming or a stable
consequence-directed internal policy. They motivate a direct test of whether
action surfaces used in safety evaluations preserve outcome-level behavior.

The current novelty target is the conjunction of:

1. causally identical action or tool interfaces under systematic relabeling;
2. independent evidence that the model knows each action's real consequence;
3. outcome-level rather than function-name-level scoring;
4. held-out causal intervention on violations of that invariance; and
5. explicit application to the construct validity of agentic safety evaluations.

This remains a literature-shaped hypothesis rather than a priority claim. The
direct tool-calling and motivated-mislabeling translations are prospective
follow-ups and are not evidence from the present game corpus.


## Related work added 2026-10-24: label fixation and the say-do gap

Both papers were read from their arXiv full texts (not only abstracts); sections on methods, results and limitations were checked. Neither is replicated here.

**In-Context Fixation** (Liu, May 2026, [arXiv 2605.08295](https://arxiv.org/abs/2605.08295)). On few-shot single-token classification with base models (0.8B-8B), demonstrations whose labels are all the same (or drawn from a small nonsense set) collapse accuracy to 12% or below. The authors argue the trigger is the content of the label slot, and report activation patching on Pythia-1B recovering 98.4% of the gap and a logit lens showing the correct answer encoded mid-stack, then overridden.
- Shared: a correct internal answer that fails to reach the output; input-driven override of semantics.
- Different: the trigger there is label unanimity or a restricted label vocabulary. The RBG-4 "redundant" history is a balanced mapping demonstration (actions in the order 0,1,1,0 with their receiver responses), so unanimity is absent. The levers we have evidence for are the irrelevant private-state sentence and mapping layout/binding, which that paper does not test. Their circuit localization succeeded; ours did not (and the RBG-5B control failed).
- Open: whether label-slot fixation contributes to our history effects is untested here. A homogeneous-versus-balanced history control would test it.

**Large Language Models Often Say One Thing and Do Another** (Xu et al., Mar 2025, [arXiv 2503.07003](https://arxiv.org/abs/2503.07003)). The Words and Deeds Consistency Test pairs a word question with a deed question across 12 LLMs and finds widespread inconsistency. Aligning only words or only deeds with SFT/DPO affects the other poorly and unpredictably, which the authors read as the two being guided by different knowledge.
- Shared: a dissociation between what a model says and what it does, treated as an evaluation problem.
- Different: their items concern opinions and values, where the "right" deed is not defined by a payoff. Here payoffs are exact, the correct action is verifiable, and a word-versus-opaque-label contrast isolates wording. Their training-intervention arm has no counterpart here.
