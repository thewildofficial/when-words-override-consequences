# V6.1 — Epistemic Strategic Repair

V6.1 is a corrective follow-up to V6-ES-1. It preserves the research
question—whether a model keeps agent-indexed beliefs, evidence, strategy, and
real consequences distinct—but repairs the stimulus-construction failures
identified in the V6 audit.

Current status: the protocol and generated dataset pass the CPU semantic
audit. No model has been run in this branch yet.

The parent erratum is [V6 prompt completeness and trajectory
confounds](../v6/erratum-prompt-completeness.md). The V6 raw artifacts remain
immutable, but the affected `higher_order`, action→report, and core policy
interpretations are quarantined by that erratum.

## Frozen namespace

| Object | Canonical name |
|---|---|
| Study | `V6.1-ES-2` |
| Branch | `research/v6-1-epistemic-repair` |
| Python module | `jspace_policy.v6_1_epistemic_repair` |
| Config/data | `configs/v6.1/epistemic_repair/` |
| Results | `results/v6_1_epistemic_repair/` |
| Modal app | `jspace-v6-1-epistemic-repair` |

The primary model is the pinned `Qwen/Qwen3.6-27B` checkpoint. Qwen3.8-27B is
registered separately, but is disabled until its own model-specific tokenizer
preflight passes and its semantic prompt/label contract matches V6.1. Different
chat templates and token IDs are recorded per model and are not themselves a
replication failure. It has no shared gate or pooled result.

## What the CPU audit froze

The committed generator produces 7,488 deterministic rows at control time:
2,496 per split across discovery, validation, and locked. The locked split is
2,496 rows. The generated row table is intentionally ignored by git; the
manifest pins its hash, and the CPU control step regenerates and verifies it
before any Modal stage.

| Family | Total rows | Locked rows | Primary question |
|---|---:|---:|---|
| `ledger_binding` | 480 | 160 | Are complete, visible role-indexed values kept distinct? |
| `policy_composition` | 144 | 48 | Is receiver belief composed with a within-game policy flip? |
| `evidence_update` | 4,320 | 1,440 | Do independent observations count differently from copies, including provenance-only prompts? |
| `recursive_strategy` | 192 | 64 | Does explicit level-k recursion affect prediction and action? |
| `active_information` | 864 | 288 | Is inspection chosen by certified value of information rather than superficial cues, including cost/reliability thresholds? |
| `monitoring_goal` | 1,152 | 384 | Do real penalties matter while audit wording stays inert, including lexical conflict? |
| `scaffold_order` | 336 | 112 | Does preceding content matter beyond turn/label priming, with a real prior-report trajectory? |

The CPU control record reports:

- config SHA256: `c2c272a2fb7f3feeceb3651b85b334bfcd9177b05353fb1ee5e86c9b0ff1b9a5`
- dataset SHA256: `14aea862bd5f195b11ebe77f97fb6be857717bd10c47375e9c11cd773214d77a`
- all seven structural and semantic family gates: passed
- GPU seconds: `0`
- model forward passes: `0`

## Repairs that matter for interpretation

1. Every ledger report prompt visibly contains the value of the requested
   field. The hidden certificate is used only to audit the generator.
2. Policy-composition prompts provide the receiver policy and modeled belief,
   but not the already-derived receiver response. Literal and contrarian
   policies are crossed within the same payoff game.
3. Action→report is compared with a direct report using the exact same report
   prompt. The only extra content is the preceding action turn, whose answer
   is recorded separately.
4. Active-information rows pair positive-VOI and negative-VOI payoff profiles
   within the same uncertainty, stake, cost, and reliability cell. A separate
   `voi_threshold` control reuses one payoff matrix while cost and signal
   reliability cross net VOI through zero. The gate requires all three
   within-pair inspect/act switches; a superficial payoff-magnitude heuristic
   cannot satisfy it.
5. Safety rows cross both modeled beliefs and force a penalty switch in every
   game/belief cell. Audit cues are payoff-, policy-, and information-neutral.
   The `lexical_conflict` surface intentionally calls the penalized action
   “protective” and the unpenalized action “hazardous”; it is an adversarial
   wording control, not a safety conclusion.
6. Independent and copied evidence are crossed at counts 0, 1, 2, and 4. The
   four-observation cells have both an explicit-rule calibration and a
   provenance-only prompt that exposes shared event IDs plus a visible
   log-odds model; the dependence-sensitive step is identifying repeated rows
   as records of the same event.
7. Every primary matched comparison uses one deterministic A/B permutation per
   game-level namespace. The hash includes the game ID and namespace, never
   the treatment value, and the CPU audit checks pairwise invariance and
   across-game balance.
8. Fixed scaffold rows are content-only controls. Only the `self_generated`
   rows receive an earlier sampled report as a genuine assistant turn; no
   fixed-source row is interpreted as evidence of source provenance.

All confirmatory estimates first aggregate matched endpoints within each
`game_id`; bootstrap draws resample games, not lower-level cells. Paired
sign-flip tests are run only after the corresponding family gate passes.

## Measurement boundary

V6.1 has two behavioral measurements:

- forced-choice next-token logits, which estimate the model's preference over
  the two legal labels;
- locked-split free generation, which measures parseability and observable
  behavior separately.

Neither mode establishes a mechanism. Activation collection is intentionally
sealed. A positive family gate is a prerequisite for a later mechanistic
protocol; a failed gate is retained as a family-specific behavioral result.
The study also forbids a deception label without both receiver behavior and a
sender-benefit test.

See the [preregistration](preregistration.md), [literature and future
directions](literature.md), and [machine-readable control audit](../../results/v6_1_epistemic_repair/control_audit.json).

## Reproduction

```bash
uv sync --extra dev --extra modal
uv run python scripts/run_v6_1_epistemic_repair_controls.py --freeze-dataset
uv run pytest -q tests/test_v6_1_epistemic_repair.py
uv run ruff check .
```

The paid stages are exposed only through the manual
[V6.1 GitHub Actions workflow](../../.github/workflows/v6-1-epistemic-repair.yml).
