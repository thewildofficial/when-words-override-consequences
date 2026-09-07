# Results: the evidence map

This folder is the project’s answer key. Each experiment has its own README
with four things a reader should be able to find quickly:

1. what question the experiment asked;
2. what was tested and what was held constant;
3. what happened, including surprising or inconclusive parts; and
4. where the raw rows, summaries, figures, and code live.

The results are ordered by the decisions they enabled, not by folder name.
Folder names are historical storage paths and are kept unchanged so scripts,
manifests, and citations remain reproducible. The canonical experiment name is
the bold label in the table.

**Second-pass review:** [audit and corrections](../docs/next-sprint/audit.md).
RBG-6 counts/uncertainty and RBG-5B J-lens interpretation are qualified there.

## The experiment trail

| Canonical name | Question | Result | Folder |
|---|---|---|---|
| **Stage 1A — 4B behavioral screen** | Could the 4B model do the tiny reporting task? | No. One balanced cell failed, so the model was retired. | [`4b_behavioral_screen/`](4b_behavioral_screen/README.md) |
| **Stage 1B — 27B scaled behavioral gate** | Could the 27B model do the same task reliably? | Yes: 192/192 rows. | [`27b_behavioral_gate/`](27b_behavioral_gate/README.md) |
| **Stage 1C — released-lens integrity** | Did the released lens fit the 27B model? | Yes, on structural and reference checks. | [`27b_lens_integrity/`](27b_lens_integrity/README.md) |
| **Stage 1D — observational policy-signal pilot** | Could we see a policy-related signal before editing the model? | Yes, but this was observational and exploratory. | [`27b_observational_pilot/`](27b_observational_pilot/README.md) |
| **Stage 1E.1 — V2 workspace mapping** | Which layers did the neutral, task-independent rule choose? | Layers 36–43. | [`v2_workspace_mapping/`](v2_workspace_mapping/README.md) |
| **Stage 1E.2 — V2 causal gate (H0)** | Did the preregistered V2 intervention pass its positive control? | No. The full-band/all-position topology produced 0/90 target top-1 answers. | [`v2_smoke_tests/`](v2_smoke_tests/README.md) |
| **Stage 1E.3 — V2 recovery diagnosis (H0R-B)** | Could a diagnostic explain the H0 failure without touching fresh controls? | It found a gentler local protocol and froze it. | [`v2_h0r_diagnostic/`](v2_h0r_diagnostic/README.md) |
| **Stage 1E.4a — H0R-C corpus eligibility check** | Did the first fresh argument corpus support a fair test? | No. Baseline was 60.8%, so intervention was never opened. | [`v2_h0r_argument_validation/`](v2_h0r_argument_validation/README.md) |
| **Stage 1E.4b — H0R-C prospective validation** | Did the replacement corpus pass the frozen causal test? | Baseline passed, but target top-1 conversion failed at 4/125. | [`v2_h0r_argument_validation_v2/`](v2_h0r_argument_validation_v2/README.md) |
| **V2-SR-1 — state/report dissociation** | Is state information still readable when the reporting rule changes? | Residual probes were positive; the J-space endpoint failed. | [`v2_stage1/`](v2_stage1/README.md) |
| **V2-SWA-1 — strategic workspace atlas** | Does a generic optimization signal identify the specific strategy? | Generic signal replicated; strategy decoding did not. | [`v2_workspace_atlas/`](v2_workspace_atlas/README.md) |
| **V2-ST-1 — strategic J-Lens trajectories** | Can top-k lens trajectories expose the decisive pathway? | Generic task semantics and late answer preparation only. | [`v2_strategic_trajectories/`](v2_strategic_trajectories/README.md) |
| **V3-JI1 — J-space intervention replication** | Can a fresh corpus support a cleaner intervention test? | Burned engineering gate passed; fresh corpus freeze failed before intervention. | [`v3_jspace_interventions/`](v3_jspace_interventions/README.md) |
| **V4-SES-1 — strategic epistemic search** | Does retaining a correct rationale hurt a later relational report? | Yes, replicated behaviorally; the internal mechanism is unknown. | [`v4_strategic_epistemic_search/`](v4_strategic_epistemic_search/README.md) |
| **V5-RBG-1 — receiver-policy competence gate** | Can the model use a demonstrated receiver policy in a game? | No; competence gate failed. | [`v5_revealed_belief_games/`](v5_revealed_belief_games/README.md) |
| **V5-RBG-2 — semantic action-outcome override** | Can consequence reports stay correct while meaningful actions fail? | Yes: a 42.19-point semantic action gap. | [`v5_semantic_override/`](v5_semantic_override/README.md) |
| **V5-RBG-3 — semantic localization** | Does the large RBG-2 effect survive fresh surface controls? | No; the large effect attenuated, but residual errors were rehearsal-repaired. | [`v5_semantic_localization/`](v5_semantic_localization/README.md) |
| **V5-RBG-4 — inverse-evidence replication** | Do redundant correct demonstrations harm strategic action? | Yes, prospectively, for prose assertions; tables removed the harm. | [`v5_inverse_evidence/`](v5_inverse_evidence/README.md) |
| **V5-RBG-5 — mechanistic decomposition** | Can matched natural activations repair the prose/assertion failure? | Behavior replicated; report-gap gate failed, so activations stayed closed. | [`v5_mechanistic_decomposition/`](v5_mechanistic_decomposition/README.md) |
| **V5-RBG-5B — powered natural-residual follow-up** | Can a fresh natural residual repair held-out failures? | No: 2/50 repairs (4%), exact `p=.429`; readout completed but candidate normalization needs correction. | [`v5_mechanistic_decomposition_b/`](v5_mechanistic_decomposition_b/README.md) |
| **V5-RBG-6 — endogenous full action trajectory** | Does seeing the model's own consequence reports control later action? | Yes, behaviorally: 0/48 primary dissociations; self-generated action rescue of 58.33 points. | [`v5_full_action_trajectory/`](v5_full_action_trajectory/README.md) |
| **Report-reactivity — ask-first ceiling & poisoned self-talk** | On fresh nonce games, do consequence reports rescue action — or can lied transcripts steer the press? | Direct/self/control at ceiling (rescue unidentifiable); swapped accuracy ~0.83, worst under wordy opposed strategic framing. | [`report_reactivity/`](report_reactivity/README.md) |
| **Report-reactivity — harder games null** | Do redundant correct demos break the Direct ceiling so ask-first rescue becomes measurable? | No: Direct stayed at 1.0; swapped still leaked (~0.79). | [`report_reactivity/`](report_reactivity/README.md) |
| **Report-reactivity — mid-trajectory ask** | Does a mid-game consequence question change the next press, or only elicit a report? | Asking does nothing (self−control contrast 0.0); a lied mid-answer rewrites ~60% of second presses (flip 0.59375). C19–C21. | [`report_reactivity/`](report_reactivity/README.md) |
| **V7-EAS-1 — epistemic control of agentic strategy** | Does an agent use public knowledge, real commitment, and opponent-relevant information value in strategy? | CPU preflight passed; 3,744 rows frozen; no model forwards yet. | [`v7_epistemic_agentic_strategy/`](v7_epistemic_agentic_strategy/README.md) |

## How to read a result

The project uses several kinds of evidence, and they answer different
questions:

- A **behavioral gate** asks whether the model can do the task at all.
- A **lens check** asks whether the measuring instrument is connected correctly.
- An **observational pilot** asks whether a pattern is visible. It cannot show that the pattern causes behavior.
- A **causal intervention** changes an internal state and compares the result with controls.
- A **prospective gate** is a locked test on data that was not used to choose the intervention.

The project treats a failed gate as information. A failed result is not silently
reframed as success, and an exploratory result is not promoted to a
confirmatory claim.

## The naming rule

The canonical hierarchy is:

```text
Research Stage 1: Toy mechanism
├── Stage 1A–1D: behavioral, lens, and observational runs
└── Stage 1E: causal-instrument validation under Protocol V2
    ├── 1E.1: workspace mapping
    ├── 1E.2: H0 causal gate
    ├── 1E.3: H0R-B recovery diagnosis
    └── 1E.4: H0R-C prospective validation
Research Stage 2: Controlled composition — not reached
Research Stage 3: Naturalistic behavior — not reached
```

Later protocol generations sit beside this ladder rather than replacing it:

```text
V1: exploratory foundation
V2: causal-instrument branch and strategic workspace follow-ons
V3: J-space intervention replication
V4: strategic epistemic / rationale-interference behavior
V5: revealed-belief and inverse-evidence games
```

`V1` through `V5` are protocol generations, not Stage 1 through Stage 5.
`V2-E1` and `V2-E2` are historical study labels; the reader-facing names are
`V2-SWA-1` and `V2-ST-1`. `v1`–`v6` in the 4B artifact names are historical
prompt revisions. `H0` is the original V2 gate, and `H0R-A` through `H0R-D` are
recovery phases.

## Evidence folders

Inside each stage folder:

- `raw/` contains the rows produced by the run;
- `summaries/` contains tables derived from those rows;
- `figures/` contains plots derived from the summaries or raw data;
- `run_manifest.json` or a matching manifest records model, code, and run details when available.

The deeper explanation of the plans and stop rules is in [`../docs/README.md`](../docs/README.md).

[Back to the project README](../README.md)
