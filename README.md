# When Words Override Consequences

**The one-sentence version:** we check whether an AI chooses actions by what they *do*, or by what they *say*.

## Claims that currently survive

Status as of the audit at commit `56e5aea`. "Supported" means the artifact exists and the limit column is part of the claim. Details: [claim ledger](docs/next-sprint/claim-ledger.md), [audit](docs/next-sprint/audit.md).

| Claim | Standing | Limit |
|---|---|---|
| Qwen3.6-27B reports consequences ~99% accurately, yet action accuracy drops for meaningful actions under an opposed incentive | Supported (prospective RBG-4) | Original model plus one contrarian-receiver game template |
| Removing the irrelevant private-state sentence: 15/48 -> 47/48 (opposed, assertion, prose) | Supported at discovery level; token-matched follow-up reproduced it (11/48 vs 48/48) | 12 lexical families, repeated conditions, descriptive family bootstrap, one model |
| Explicit A/B labels and table layout each help (prose 31 -> 67%, table 56 -> 83%) | Supported behaviorally | Does not isolate a neural binding representation |
| The drop appears in Mistral-Small-24B, GLM-4.5-Air, Qwen3-30B-Instruct | Qualitative only | All comparators failed the competence gates; floor effects possible |
| Own consequence reports before acting give 46/46 correct (RBG-6) | Shows dependence on the transcript only | The reported-trajectory arm also adds turns, text and an instruction to use the reports ([audit A3](docs/next-sprint/audit.md)) |
| Copying a "good" state into failing runs repairs 2/50 (RBG-5B) | **Not established either way** | The identity control failed (shift 0.5 vs 1e-4 tolerance, [audit A2](docs/next-sprint/audit.md)); this does not show that no mechanism exists |
| Candidate J-lens readout aggregates (old 54.69%) | Quarantined | Missing final normalization; see [errata](#errata-and-quarantine-status) |
| Token-aligned follow-up: 14 patch sites, 0 repairs of 4 selected failures | Weak negative | Tiny, mostly A-first sample; no held-out confirmation |
| V6-ES-1 higher-order results | Quarantined | Values were absent from the rendered prompts |


## The game in 30 seconds

Imagine two buttons:

- `SAY RED`
- `SAY BLUE`

You're told the other side is contrarian: pressing `SAY RED` makes them pick blue, and vice versa. You want them to pick red. The right move is `SAY BLUE`.

We ask the model two things separately: *"what will each button do?"* and *"which button gets you what you want?"* The weird, interesting moments are when it answers the first question perfectly — then presses the wrong button because its *words* match the goal instead of its *consequences*.

To make sure it's really about words, we replay the identical game with the words stripped out (`X` / `Y`) or with a plain consequence table. Details: [V5 research status](docs/v5/RESEARCH_STATUS.md).

## What we found (3 things)

1. **It knows, but doesn't always act.** The model reports consequences at 98.96% accuracy — yet adding four prior-round demonstrations of the action-to-response mapping (balanced across the two actions, not label-unanimous) drops its action accuracy from 79% to 55% for wordy actions. Tables and `X`/`Y` buttons are immune. ([Frozen RBG-4 findings](results/v5_inverse_evidence/FINDINGS.md))
2. **Action depends on the transcript, including its own reports.** In the report-then-act trajectory the model chooses correctly 46/46 times. That trajectory also adds extra turns, text and an instruction to use the reports, so this shows transcript dependence. It does not show that the reports themselves are the cause. ([Audit A3](docs/next-sprint/audit.md))
3. **One internal repair test failed, and its control failed too.** Copying a "good" internal state into failing runs fixed 2/50 cases. The identity control for that test did not pass, so this does not show that no internal mechanism exists. ([RBG-5B preregistration](docs/v5/mechanistic-decomposition-b-preregistration.md), [audit A2](docs/next-sprint/audit.md))

## What this is NOT

Not consciousness. Not deception. Not "all AI safety tests are broken." It's one narrow, preregistered check: *do represented consequences actually steer the choice when the wording points the other way?* Full boundary: [literature positioning](docs/v5/literature-positioning.md). Closest related work (label fixation, say-do consistency): [related-work note](docs/v5/literature-positioning.md#related-work-added-2026-10-24-label-fixation-and-the-say-do-gap).

## Scoreboard

Every row below was decided *before* the data was opened. Failures count as results here.

| Step | Question in plain English | Verdict |
|---|---|---|
| 4B screen | Can the small model even play? | No; retired |
| 27B behavioral gate | Can the big model play reliably? | Yes, 192/192 |
| J-space instrument | Can our brain-edit tool reliably work? | No: 0/90, then 4/125 |
| Strategic atlas | Is the brain signal strategy-specific? | No, generic only |
| RBG-1 – RBG-3 | Does the know/act gap show up and replicate? | Yes, then weakened |
| RBG-4 | Do *additional mapping demonstrations* make action worse? | Yes (prospective) |
| RBG-5 | Can we open the activations? | Behavior yes; one gate missed by 0.2 pts |
| RBG-5B | Can a good brain state repair failures? | Not established: 2/50 repaired, `p=.429`, identity control failed |
| RBG-6 | Do its own reports rescue the action? | Transcript dependence only (0/48 dissociation); confounded, see finding 2 |

Full ledger: [results/README.md](results/README.md) · Every decision: [decision log](docs/v5/decision-log.md).

## Latest mechanism investigation — 11 October 2026

Changing a task-irrelevant private-state sentence while keeping the correct
action fixed strongly changes the model's choice. In the opposed-state,
assertion/prose condition, removing that sentence raised accuracy from
**15/48 (31.25%) to 47/48 (97.92%)**. Explicit A/B labels and table layout each
helped independently; together they reached 40/48 (83.33%). These are discovery
results from twelve lexical families, with repeated conditions within families.

A token-matched follow-up reproduced the state effect. Fourteen internal patch
sites produced **zero repairs of four selected failures**; no site qualified for
held-out confirmation. The small, consistently A-first patch sample limits that
negative result. We have localized a behavioral cause of errors, while the
neural mechanism and the distinction between truth-role and lexical interference
remain open.

Read the [findings and figure](results/semantic_action_mechanisms/FINDINGS.md),
[artifact map and offline reproduction](results/semantic_action_mechanisms/README.md),
and [theory and literature](docs/next-sprint/semantic-action-mechanisms.md).
The completed study reserved $4.75216032 under its $5 admission cap, including
failed attempts; this is a reservation ledger, not a provider invoice.

## Errata and quarantine status

| Item | Status | Record |
|---|---|---|
| Candidate J-lens readout (omitted final normalization) | Calculator fixed, parity gate passed on the real model; old numbers quarantined; archived scores not recomputed | [erratum](results/research_audit/rbg5b_readout_erratum.json), [parity](results/research_audit/readout_parity_af81d642.json) |
| RBG-5B identity control | Failed; causal interpretation limited | [audit A2](docs/next-sprint/audit.md) |
| RBG-6 trajectory confounds | Open; matched-control design proposed, not run | [audit A3](docs/next-sprint/audit.md), [design](docs/next-sprint/experiments.md) |
| V6-ES-1 | Run retained; higher-order values missing from prompts, trajectory supplied missing information, policy test supplied a derived response; affected interpretations quarantined | [erratum](docs/v6/erratum-prompt-completeness.md) |

The main results in the table above are not affected by the readout erratum.

Next up is a proposed ~$30 follow-up asking whether reports *reveal* or *change* the information behind actions: [the next-sprint plan](docs/next-sprint/README.md) ([audit](docs/next-sprint/audit.md) · [literature](docs/next-sprint/literature.md) · [design](docs/next-sprint/experiments.md)). The corrective [V6.1-ES-2 protocol](docs/v6.1/README.md) has a passing CPU audit and is the active research branch; its model stages are dispatched manually through GitHub Actions.

The prospective [V6.2 reasoning-capability control](docs/v6.2/README.md) is
also CPU-frozen on its dedicated branch. It compares Qwen3.8 direct logits
with Qwen3.8 native-reasoning final choices against the completed Qwen3.6
reference. **No model forwards have occurred on the V6.2 branch yet.** Its
Modal stages are separately authorized and manually dispatched; the branch
ceiling is USD 8.00.

The separate [V7-EAS-1 protocol](docs/v7/README.md) tests whether an agent
uses another agent's epistemic state in coordination, commitment, and
information-acquisition decisions. Its CPU preflight is frozen and passed; no
model forwards have occurred on that branch yet.

## Go deeper

| You want... | Go here |
|---|---|
| The original research ladder | [research spec](docs/research-spec.md) |
| Why each test was designed this way | [docs guide](docs/README.md) |
| Frozen datasets and thresholds | [configs](configs/README.md) |
| The code behind games and analyses | [src/jspace_policy](src/jspace_policy/README.md) |
| Offline analyses and figures | [scripts](scripts/README.md) |
| Test guardrails | [tests](tests/README.md) |
| Why this matters for tool-using agents | [safety note](docs/v5/literature-positioning.md) |
| V6-ES-1 audit and retained run | [V6 documentation](docs/v6/README.md) · [erratum](docs/v6/erratum-prompt-completeness.md) |
| New epistemic/ToM/strategy branch | [V6.1-ES-2 protocol](docs/v6.1/README.md) |
| Reasoning-capability control | [V6.2 protocol](docs/v6.2/README.md) |
| V7 epistemic agentic strategy | [V7-EAS-1 protocol](docs/v7/README.md) · [preflight record](results/v7_epistemic_agentic_strategy/README.md) |
| Semantic action mechanisms | [findings](results/semantic_action_mechanisms/FINDINGS.md) · [protocol](experiments/semantic_action_mechanisms/README.md) · [offline reproduction](results/semantic_action_mechanisms/README.md#offline-reproduction) |
Reproduce the lightweight checks (committed GPU outputs are immutable; these verify everything else):

```bash
uv sync --extra dev --extra modal
uv run python -m pytest
```

Paid GPU runs are gated: manual GitHub Actions workflows, frozen hashes, overwrite refusal, hard budget ceiling. A failed gate stops later compute.

## Selected references

- Gurnee et al., [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/).
- Heimersheim and Nanda, [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255).
- Geiger et al., [Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997).
- Wang et al., [The Story Shapes the Agent](https://arxiv.org/abs/2607.18566).
- Hopman et al., [Evaluating and Understanding Scheming Propensity in LLM Agents](https://arxiv.org/abs/2603.01608).
- Lynch et al., [Agentic Misalignment in Summer 2026](https://alignment.anthropic.com/2026/agentic-misalignment-summer-2026/).
