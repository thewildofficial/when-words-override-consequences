# V6.2 preregistration: reasoning capability control

Study ID: `V6.2-RCC-1`
Status: frozen before Qwen3.8 model execution
Branch: `research/v6-2-reasoning-capability-control`

## Question

V6.1 found cases where reports were accurate while downstream actions failed
to track manipulations that should change the correct action. The strongest
locked result was approximately 100% ledger-report accuracy versus a 12.5%
modeled-receiver-belief action-pair rate. A capability alternative remains:
Qwen3.6-27B may simply be bad at the required multi-step symbolic computation.

V6.2 asks whether the report-to-action dissociation survives (a) a stronger
Qwen3.8 checkpoint with the same direct measurement and (b) native reasoning
enabled on the same Qwen3.8 checkpoint.

No V6.1 artifact, gate, raw result, or interpretation is modified. The V6.1
generated dataset is regenerated from its pinned config, verified against its
committed manifest, and used only as an immutable source.

## Conditions

| Condition | Model | Reasoning | Primary output |
|---|---|---|---|
| Reference | Qwen3.6-27B | off | Existing V6.1 locked direct A/B logits |
| Direct | Qwen3.8-27B, revision pinned in config | off | Candidate-token A/B logits over all 2,496 locked rows |
| Thinking pilot | Qwen3.8-27B | native thinking on | Full generated text; exact `FINAL: A/B` parser |
| Thinking diagnostic | Qwen3.8-27B | native thinking on | Full generated text; primary final-choice scoring |

The Qwen3.8 thinking text is retained for later descriptive analysis only. It
is not a preregistered correctness signal and does not license a mechanistic
claim by itself. Missing, malformed, or ambiguous final lines count as failed
choices.

## Frozen data and selection

The V6.1 source contains 7,488 deterministic rows, including 2,496 locked
rows. The direct condition uses every locked row. The reasoning diagnostic is
selected before any Qwen3.8 output:

- every locked H1 ledger row, including all named report targets and modeled-
  belief action pairs;
- every locked H2 policy-composition row, including all literal/contrarian
  identifying pairs;
- every locked H3 report row with four observations, prior-directed evidence,
  an independent/copy expected-value contrast, and both explicit-rule and
  provenance-only prompt modes;
- every locked H5 positive/negative VOI matched pair and every same-matrix
  cost/reliability threshold cell.

This produces 640 diagnostic rows. The pilot is a separate deterministic
subset: the first complete ledger game, first policy game, one identifying H3
pair in each prompt mode, one positive/negative VOI pair, and one complete
same-matrix threshold cell, all sorted by condition ID.

The committed subset manifest records the explicit selected IDs and hashes.
The config and source hashes are:

- V6.1 source config SHA256:
  `c2c272a2fb7f3feeceb3651b85b334bfcd9177b05353fb1ee5e86c9b0ff1b9a5`
- V6.1 source dataset SHA256:
  `14aea862bd5f195b11ebe77f97fb6be857717bd10c47375e9c11cd773214d77a`
- Qwen3.8 revision:
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`

## Preflight and semantic invariants

Before any paid model stage, the workflow runs the full test suite and CPU
controls. The Qwen3.8 tokenizer preflight then records the exact tokenizer and
resolved revision, validates that direct A/B labels are scoreable completions,
and checks that thinking-on rendering contains the tokenizer's native thinking
marker while thinking-off rendering does not.

The direct and thinking prompts share the same task content, labels, game
state, payoff table, and treatment. Only the answer interface changes from
`Return only A or B` to the bounded reasoning instruction ending in
`FINAL: A` or `FINAL: B`. Template-specific rendering and token IDs are
recorded but are not cross-model semantic failures.

The controls also require:

1. exact source/config/subset hashes;
2. no treatment-dependent A/B mapping in identifying pairs;
3. complete H1/H2/H3/H5 identifying coverage;
4. a parser that accepts one legal final line and rejects missing, ambiguous,
   wrong-label, or inline occurrences;
5. a constant-A or constant-B policy failing every paired identifying endpoint;
6. no new correct downstream answer in the thinking interface;
7. game-level clustering definitions retained for analysis; and
8. zero model forwards during the CPU control stage.

## Primary endpoints

V6.2 does not create an omnibus pass/fail score. It reports the following
comparisons directly.

### H1 — Ledger binding

Report accuracy by named target; modeled-belief action-pair rate requires both
rows to be correct and the selected semantic action to switch; actual-belief
invariance is reported as a nuisance control.

### H2 — Policy composition

Literal and contrarian rows must both be correct and must switch semantic
action within each game and modeled-belief cell.

### H3 — Evidence independence

The four-independent-observation and four-copied-observation reports must
produce a selected-choice contrast in both explicit-rule and provenance-only
prompt modes. Both-correct-and-switched values are reported separately.

### H5 — Active information

Positive-VOI versus negative-VOI pairs must switch correctly. The same payoff
matrix must also cross correctly when inspection cost changes and when signal
reliability changes.

All confirmatory summaries first aggregate within `game_id`; bootstrap draws
resample games. Reasoning text is retained, but only the final parsed choice is
used for thinking-mode primary behavior.

## Interpretation matrix

| Result | Interpretation |
|---|---|
| Qwen3.8-direct substantially improves | V6.1 was at least partly capability-limited |
| Qwen3.8-direct remains poor | Simple checkpoint scaling does not explain the dissociation |
| Thinking rescues identifying pairs | Reliable multi-step computation was needed before information controlled action |
| Thinking improves accuracy but identifying pairs remain poor | Generic reasoning competence is not a sufficient explanation |
| Reasoning text derives the right intermediate state but final choice is wrong | A follow-up on composition/action selection is motivated; no mechanism is inferred here |

## Cost and execution

The branch-level authorization is USD 8.00, substantially below the remaining
approximately USD 25 Modal balance. The worst-case buffered ceiling of all
registered stages is checked in code. Each stage estimates cost before launch,
reads the persistent `cost_ledger.jsonl`, and refuses to launch if the
authorization would be exceeded. Actual elapsed time and measured/buffered
cost are recorded after completion.

The workflow has separate `workflow_dispatch` choices for CPU controls,
Qwen3.8 preflight, full direct behavior, thinking pilot, and thinking
diagnostic. There is no automatic sequence. Every later Actions run restores
the ledger from an explicitly supplied prior run ID. No multi-GPU or 100B+
model is permitted.

Artifacts are content-hashed and include the exact config, subset manifest,
preflight, raw direct logits, raw thinking generations, machine-readable
analysis, compact summaries, cost ledger, and model-run manifests. GitHub's
30-day artifact window is not treated as permanent archival storage.
