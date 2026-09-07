# Literature and novel directions for V6.1

The V6 failure mode is best treated as a measurement problem at the boundary
between epistemic representation and strategic control. The literature gives
several reasons not to equate a correct verbal answer with a stable belief or
with policy use.

## Relevant anchors

| Area | What the literature contributes | V6.1 consequence |
|---|---|---|
| False-belief measurement | Bloom and German showed that classic false-belief tasks can be solved with non-mentalistic strategies, so a response format is not by itself a ToM measure. [Bloom & German (2000)](https://pubmed.ncbi.nlm.nih.gov/10980256/) | Make every target value visible and cross role-indexed nuisances rather than treating a certificate-only answer as evidence. |
| Strategic ToM | Hedden and Zhang separated forming a mental model of another player from translating that model into an action in matrix games. [Hedden & Zhang (2002)](https://pubmed.ncbi.nlm.nih.gov/12086711/) | Cross belief and policy within the same game; measure prediction and action separately. |
| Higher-order reasoning | Stepwise second-order ToM training improves strategic turn-taking, but the task includes learning and recursion demands. [Verbrugge et al. (2018)](https://www.cambridge.org/core/journals/judgment-and-decision-making/article/stepwise-training-supports-strategic-secondorder-theory-of-mind-in-turntaking-games/7A46351EAA49C264DDD10A9BA8E4EC20) | Use a finite level-k ladder with non-degenerate sequences and separate depth from report wording. |
| Level-k behavior | Beauty-contest behavior is commonly modeled as iterated strategic reasoning rather than a single undifferentiated “ToM” ability. [Nagel (1995)](https://www.cs.princeton.edu/courses/archive/spr07/cos444/papers/nagel95.pdf) | Treat recursion depth as a falsifiable computation, not as a pooled score. |
| Symbolic language-model ToM | SymbolicToM finds that language models can show structured belief reasoning while still being sensitive to task construction. [Sclar et al. (2023)](https://aclanthology.org/2023.acl-long.780.pdf) | Keep semantic ledger accuracy, action policy, and free-generation format as distinct endpoints. |
| Perception → belief | Recent work argues that belief reasoning in models should be decomposed into perception, belief formation, and belief use. [Perceptions to Beliefs (2024)](https://arxiv.org/abs/2407.06004) | H1/H2 separate visible state binding from policy composition; V6.1 does not call either a latent mechanism. |
| Evidence and source monitoring | Source-monitoring theory distinguishes remembering content from remembering where it came from. [Johnson, Hashtroudi & Lindsay (1993)](https://doi.org/10.1037/0033-2909.114.1.3) | H3 crosses evidence source and independence; H7 crosses oracle, random, and endogenous provenance. |
| Epistemic vigilance | Epistemic communication requires evaluating source reliability and relevance, not merely counting repeated statements. [Sperber et al. (2010)](https://doi.org/10.1111/J.1468-0017.2010.01394.X) | The independent-versus-copied contrast tests whether repetition is treated as evidence. |
| Inverse planning and pragmatic inference | Human observers infer hidden goals from actions; RSA formalizes speaker/listener reasoning over informative utterances. [Baker, Saxe & Tenenbaum (2009)](https://pubmed.ncbi.nlm.nih.gov/19729154/), [Goodman & Frank (2016)](https://langcog.stanford.edu/papers_new/goodman-2016-tics.pdf) | A future sender/receiver game can test whether reports are informative signals or merely labels. |
| Cheap talk | Strategic messages can convey information despite misaligned interests, but only under restrictions on preferences and beliefs. [Crawford & Sobel (1982)](https://www.edegan.com/pdfs/Crawford%20Sobel%20%281982%29%20-%20Strategic%20Information%20Transmission.pdf) | Do not call a model “deceptive” from a wrong report alone; require receiver response and sender-benefit evidence. |
| Dynamic epistemic logic | Public announcements and epistemic actions update agents’ information states; the update itself is a separate object from the prior state. [Stanford Encyclopedia of Philosophy: Dynamic Epistemic Logic](https://plato.stanford.edu/archives/fall2025/entries/dynamic-epistemic/), [Baltag (2002)](https://ir.cwi.nl/pub/2069/2069D.pdf) | The action→report comparison must hold the report prompt fixed and identify exactly what the preceding turn adds. |
| Long-context and trajectory effects | Lookback behavior shows that models can use earlier context in ways that need not imply a stable, explicit internal ledger. [Language Models use Lookbacks (2025)](https://arxiv.org/abs/2505.14685) | H7 includes matched non-endogenous scaffolds and treats trajectory gains as descriptive unless source/content controls separate. |
| Goal misgeneralization | A capable policy can optimize a proxy or training objective rather than the intended goal. [Langosco et al. (2022)](https://proceedings.mlr.press/v162/langosco22a.html) | H6 distinguishes actual penalty reweighting from evaluation-word sensitivity; no real-world safety claim follows. |
| Evaluation awareness and scheming | MACHIAVELLI and later evaluation-awareness work motivate testing whether behavior changes under evaluation, but behavioral changes have multiple explanations. [MACHIAVELLI (2023)](https://arxiv.org/abs/2304.03279), [SAD (2024)](https://arxiv.org/abs/2407.04694), [Needham (2025)](https://arxiv.org/pdf/2505.23836) | Audit wording is payoff-neutral and paired with explicit real consequences; any cue effect is labeled evaluation sensitivity, not deception. |
| Alignment faking | Alignment-faking results illustrate why apparent compliance under evaluation is not enough to establish motive or situational awareness. [Greenblatt et al. (2024)](https://arxiv.org/abs/2412.14093) | A future safety extension needs a receiver, an incentive, and a benefit from influencing it; V6.1 explicitly forbids motive labels. |

## Novel, mechanistically useful directions after V6.1

These are follow-ups, not hidden V6.1 endpoints.

1. **Ledger retention versus policy composition.** Give the same model a
   complete ledger, insert a distractor task, then ask for the action. Compare
   retention loss against a matched lexical-load control. If a readable ledger
   survives reporting but not action, the candidate mechanism is a routing or
   retrieval bottleneck rather than missing semantic knowledge.
2. **Counterfactual belief-policy interchange.** Patch or otherwise intervene
   on the representation associated with modeled belief while holding the
   policy description fixed, then swap only policy wording. A genuine
   composition mechanism should show an interaction, not two independent main
   effects. This requires a new natural counterfactual activation protocol.
3. **Provenance-weighted belief revision.** Use equal-content messages from a
   trusted sender, adversarial sender, and the model’s own prior answer. Vary
   independence and payoff incentives orthogonally. The key novel prediction is
   a three-way interaction: source reliability × independence × receiver
   consequence.
4. **Epistemic action selection.** Let the model choose whether to ask a
   clarifying question, inspect, or act, then score both immediate utility and
   posterior calibration. This distinguishes “knows what information would
   help” from merely answering a value-of-information prompt.
5. **Strategic communication with receiver response.** Build a cheap-talk game
   in which the sender can benefit from inducing a receiver action, while the
   receiver has its own payoff and a reliability prior. Wrong reports become
   interesting only when they are optimal under the sender’s objective and
   produce the predicted receiver response.
6. **Evaluation invariance under real stakes.** Randomize whether evaluation is
   mentioned, but keep real penalties and rewards identical. Then repeat with
   hidden versus public audit probability. The mechanistic target is a
   state-dependent gate on evaluation context, not a binary “deception neuron.”

The rule for all of these is the same: preregister the semantic certificate,
the nuisance controls, and the promotion gate before looking at activations or
free-generation stories.
