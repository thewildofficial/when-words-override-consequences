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
registered separately, but is disabled until its own tokenizer and rendered
prompt preflight exactly matches the V6.1 contract. It has no shared gate or
pooled result.

## What the CPU audit froze

The committed generator produces 6,144 deterministic rows at control time:
2,048 per split across discovery, validation, and locked. The locked split is
2,048 rows. The generated row table is intentionally ignored by git; the
manifest pins its hash, and the CPU control step regenerates and verifies it
before any Modal stage.

| Family | Total rows | Locked rows | Primary question |
|---|---:|---:|---|
| `ledger_binding` | 480 | 160 | Are complete, visible role-indexed values kept distinct? |
| `policy_composition` | 144 | 48 | Is receiver belief composed with a within-game policy flip? |
| `evidence_update` | 3,456 | 1,152 | Do independent observations count differently from copies? |
| `recursive_strategy` | 192 | 64 | Does explicit level-k recursion affect prediction and action? |
| `active_information` | 384 | 128 | Is inspection chosen by certified value of information? |
| `monitoring_goal` | 1,152 | 384 | Do real penalties matter while audit wording stays inert? |
| `scaffold_order` | 336 | 112 | Does preceding content matter beyond turn/label priming? |

The CPU control record reports:

- config SHA256: `f0afb9d073a46cedf2b9685dfa12b4d9affc32781fb99c6759822afbcda37c12`
- dataset SHA256: `fe5b63bb7bbe31c95fe35d700b8b8cefa6de7a409cac0207ece79e909ebc669a`
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
4. Safety rows cross both modeled beliefs and force a penalty switch in every
   game/belief cell. Audit cues are payoff-, policy-, and information-neutral.
5. Independent and copied evidence are explicitly crossed at counts 0, 1, 2,
   and 4, with an integer scoring certificate visible in the prompt.

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
