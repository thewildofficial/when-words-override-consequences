# V6 controls and interpretation rules

The key discipline is to keep three levels of claim separate:

1. **content:** the model can state a belief/report correctly;
2. **behavioral use:** the model's action changes under the relevant epistemic
   counterfactual; and
3. **causal mechanism:** an internal state can be intervened on to change that
   action while preserving unrelated outputs.

Passing a lower level never licenses the next level automatically.

## Control matrix

| Threat | Control | What a residual effect means |
|---|---|---|
| World-label shortcut | Keep hidden actual receiver belief independent of the modeled belief in `core_tom` | Possible oracle leakage or failure to use the receiver model |
| Self/other role collapse | Three report targets with identical `X/Y` output format | Role-indexed report confusion, not necessarily absent representations |
| Publicness confounded with semantics | Same target value crossed with private/public/common-knowledge visibility | Access/publicness gating or higher-order failure |
| Source wording confounded with claim content | Five sources crossed with the same statement value | Provenance authority or recency sensitivity |
| Truth status confounded with report target | Truth is an audit factor; task asks for source claim content | Truth-status priming, not a changed correct answer |
| Token/position lure | Randomized action, response, and report label bindings | Binding or positional shortcut |
| Evaluation framing | Ordinary and audited text explicitly preserve payoff, policy, and information | Evaluation-context sensitivity; not proof of deceptive intent |
| Safety effect mistaken for incompetence | Utility-only and risk-sensitive certificates are computed separately | Correct safety reweighting if only the exact certificate changes |
| Probe interpreted as use | Activation probes are discovery-only and split by game ID | Readability; causal use remains untested |
| Causal patch artifact | Natural counterfactual, unrelated-game, same-role, retention, and KL controls | Only a surviving locked effect can support a mechanistic claim |

## Alternative explanations to keep alive

### A high report score but no strategic action effect

The model may encode an explicit belief proposition but treat it as an
ornamental report target. Other possibilities are weak payoff computation,
action-label priors, or the prompt failing to induce a genuine agent model.

### Action changes when actual hidden belief changes

This is the oracle-leakage failure mode. It could also arise from the model
inferring the hidden state from a correlated nonce or from a dataset-construction
mistake. The exact certificates and prompt audit must be checked before calling
it cognition.

### Action changes under audit framing

The first interpretation is context sensitivity, reward-model anticipation, or
formatting caution. It is not “the model knows it is being tested,” and still
less is it deception, without a held-out deployment/test contrast and causal
monitorability evidence.

### Source- or time-specific report effects

These may reflect legitimate source reliability heuristics, recency weighting,
surface parsing, or attention allocation. Since the V6 provenance task asks for
literal claim content, a residual effect is a communication/provenance finding;
it should not be summarized as simple factual hallucination.

### A readable activation with no patch effect

This is scientifically useful: the state may be downstream, redundant, or
readable but not action-controlling. The locked causal gate is allowed to fail;
the probe is not tuned after the fact.

## Stage gates and stopping rules

Each model has an independent artifact namespace. Within a model:

1. CPU hash/factorial audit must pass.
2. Tokenizer preflight must pass for every action/report/trajectory query.
3. Behavioral results are scored by family, not only overall.
4. A family that misses its gate is not promoted to its mechanistic follow-up.
5. Activation discovery must use held-out validation and locked splits.
6. `activation_locked` stays sealed until intervention coordinates and all
   retention controls are frozen in a new decision-log entry.

The Qwen3.8 replication has an additional model-level parity lock. A parity
failure stops only Qwen3.8; it does not alter Qwen3.6 results or justify changing
the shared prompt protocol.

## Literature position

V6 is motivated by several complementary lines rather than treating any one
paper as a complete theory of model minds:

- [Defining Knowledge: Bridging Epistemology and Large Language Models](https://aclanthology.org/2024.emnlp-main.900.pdf)
  argues that “knowledge” needs explicit epistemological definitions.
- [Standards for Belief Representations in LLMs](https://arxiv.org/html/2405.21030v2)
  emphasizes accuracy, coherence, uniformity, and causal use rather than probe
  accuracy alone.
- [Language Models Represent Beliefs of Self and Others](https://arxiv.org/html/2402.18496v1)
  provides a direct precedent for agent-indexed belief readouts and causal
  manipulation in ToM tasks.
- [Language Models Use Lookbacks to Track Beliefs](https://arxiv.org/html/2505.14685v2)
  motivates testing whether role/state binding is an address-like mechanism.
- [K-Level Reasoning](https://arxiv.org/abs/2402.01521) and
  [HI-ToM](https://aclanthology.org/2023.findings-emnlp.717/) motivate explicit
  higher-order and strategic belief conditions.
- [LM Agents May Fail to Act on Their Own Risk Knowledge](https://arxiv.org/pdf/2508.13465)
  motivates keeping risk knowledge/report and risk-sensitive action separate.
- [Reasoning models don’t always say what they think](https://www.anthropic.com/research/reasoning-models-dont-say-think)
  motivates retaining a black-box/free-generation path and not treating textual
  reasoning as a faithful internal trace.

These sources motivate hypotheses; they do not establish that V6's synthetic
state variables are human-like beliefs.
