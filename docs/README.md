# Documentation guide

The `docs/` folder holds the reasoning around the experiments. The `results/`
folder holds what the experiments produced. This folder explains why a test
was designed, what was allowed to change, and what each result can support.

## Current review and next sprint

Start with [the next-sprint guide](next-sprint/README.md): it joins the surviving
claims, second-pass corrections, literature comparison, and proposed $30 plan.
The versioned documents below retain the historical information state.

## Start with the research question

- [`research-spec.md`](research-spec.md) is the original research ladder: toy behavior, lens readout, causal intervention, fact preservation, and transfer.
- [`literature-review.md`](literature-review.md) explains what earlier work already shows and what would count as a genuinely interesting new result.
- [`analysis-plan.md`](analysis-plan.md) explains how raw rows become tables, plots, effect sizes, and uncertainty intervals.
- [`compute-budget.md`](compute-budget.md) records the GPU budget and why the experiment was staged.

## Naming guide

Use the following hierarchy when adding a new experiment or writing about an
old one:

1. **Research stage** names the scientific question: Stage 1 toy mechanism,
   Stage 2 controlled composition, or Stage 3 naturalistic behavior.
2. **Protocol generation** names a chronological research program: V1
   foundation, V2 causal instrument, V3 intervention replication, V4 rationale
   interference, and V5 revealed-belief/inverse-evidence games.
3. **Study** names one concrete question inside a protocol: V5-RBG-4 is the
   inverse-evidence replication.
4. **Phase** names a step inside one study: V3-JI1 Phase C is the fresh corpus
   freeze.
5. **Gate** names a pass/fail checkpoint: V2-H0 is the original causal gate.
6. **Recovery phase** names work opened after a failed gate: H0R-B and H0R-C.
7. **Prompt revision** names wording changes inside one run: P1–P6, whose old
   raw filenames use `v1`–`v6`.

The raw folder names and historical artifact filenames remain unchanged. The
canonical names above are the names used in new prose and README headings. In
particular, **Stage 2 is not V2**: the 27B experiment is a Stage 1 scale-up,
while V2 is the second protocol generation.

## Protocol map

| Protocol | Plain-language purpose | Reader entry point |
|---|---|---|
| V1 | Establish competence, lens integrity, and observational candidates. | [`../results/README.md`](../results/README.md#the-experiment-trail) |
| V2 | Test and diagnose the causal J-space instrument; run controlled workspace follow-ons. | [`v2/README.md`](v2/README.md) |
| V3 | Characterize a cleaner J-space intervention replication. | [`v3/README.md`](v3/README.md) |
| V4 | Test rationale-induced relational-report interference behaviorally. | [`v4/README.md`](v4/README.md) |
| V5 | Test action/report dissociation, inverse evidence, and its natural-interchange mechanism in contrarian games. | [`v5/README.md`](v5/README.md) |
| V6 | Test agent-indexed epistemic state use, higher-order ToM, provenance, and strategic safety/evaluation controls. | [`v6/README.md`](v6/README.md) |

V4 and V5 are later behavioral follow-ons. They do not reopen the failed V2
causal-instrument branch and do not count as formally completed Stage 2 or
Stage 3 mechanistic studies.

## Protocol V2: the locked causal branch

Protocol V2 was the stricter version of the Stage 1 causal-instrument work. It asked the intervention to pass a
positive control before any strategic-reporting task could be opened.

| Document | In plain language |
|---|---|
| [`v2/research-spec.md`](v2/research-spec.md) | The full V2 question and order of operations. |
| [`v2/preregistration.md`](v2/preregistration.md) | What H0 tested before results were opened. |
| [`v2/hypotheses.md`](v2/hypotheses.md) | The claims H1–H8 that remained locked behind H0. |
| [`v2/method-parity.md`](v2/method-parity.md) | A checklist showing which implementation details matched the reference. |
| [`v2/final-report.md`](v2/final-report.md) | Why the original H0 gate failed. |
| [`v2/h0r-preregistration.md`](v2/h0r-preregistration.md) | The recovery plan written after H0 failed, without changing H0. |
| [`v2/h0r-analysis-plan.md`](v2/h0r-analysis-plan.md) | How the recovery candidate was selected. |
| [`v2/h0r-diagnostic-report.md`](v2/h0r-diagnostic-report.md) | What the burned-data diagnosis found. |
| [`v2/h0r-final-report.md`](v2/h0r-final-report.md) | Why the fresh recovery test also stopped the branch. |
| [`v2/decision-log.md`](v2/decision-log.md) | The append-only record of decisions, evidence already seen, and unopened tests. |

## How certainty is recorded

The documents distinguish four states:

- **Observed:** directly measured in a committed run.
- **Exploratory:** useful for choosing what to test next, but not a held-out claim.
- **Prospective:** tested after the protocol or corpus was frozen.
- **Not run:** deliberately unopened because an earlier gate failed.

That vocabulary is important here. For example, H0R-B found a strong direction
on burned country controls, but H0R-C was the fresh test that could validate the
protocol. H0R-C did not pass, so H0R-D and the **original V2 mechanistic
strategic task** remain “not run.” Later V4 and V5 behavioral studies are
separate follow-ons and are documented in their own protocol trees.

[Back to the project README](../README.md) · [Back to the results map](../results/README.md)
