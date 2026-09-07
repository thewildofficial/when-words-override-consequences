# V6.2 — Reasoning Capability Control

**Current status: CPU controls complete; no model forwards have occurred on
this branch yet.**

V6.2 tests the main alternative explanation for the V6.1 result: perhaps
Qwen3.6-27B could retrieve the relevant state but could not execute the
multi-step symbolic computation needed to turn it into an action. The control
therefore compares:

| Condition | Measurement |
|---|---|
| Qwen3.6-27B, thinking off | Existing V6.1 direct A/B-logit reference |
| Qwen3.8-27B, thinking off | Full locked V6.1 replication with direct logits |
| Qwen3.8-27B, thinking on | Validation-only pilot plus locked diagnostic, scored from `FINAL: A/B` only |

The Qwen3.8 revision is pinned to
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The V6.1 source config and
dataset are independently hash-checked before subset construction:

- source config: `c2c272a2fb7f3feeceb3651b85b334bfcd9177b05353fb1ee5e86c9b0ff1b9a5`
- source dataset: `14aea862bd5f195b11ebe77f97fb6be857717bd10c47375e9c11cd773214d77a`
- source rows: 7,488 total; 2,496 locked
- V6.2 thinking diagnostic subset: 640 locked rows across H1, H2, H3, and H5
- V6.2 thinking pilot: 34 validation rows, never drawn from the locked diagnostic set

The branch-level authorization is USD 8.00, below the approximately USD 25
Modal balance. Thinking is frozen to `reasoning_effort=xhigh`, sampled
generation (`temperature=1.0`, `top_p=0.95`, `top_k=20`, repetition penalty
`1.0`, seed `3838`). Stages are manual and separately budget-admitted; no
workflow stage automatically launches a later stage. GitHub Actions artifacts
are retained for 30 days only and are not archival storage.

## Frozen namespace

| Object | Canonical name |
|---|---|
| Study | `V6.2-RCC-1` |
| Branch | `research/v6-2-reasoning-capability-control` |
| Python controls | `jspace_policy.v6_2_reasoning_capability` |
| Config | `configs/v6.2/reasoning_capability_control/` |
| Results | `results/v6_2_reasoning_capability_control/` |
| Modal app | `jspace-v6-2-reasoning-capability-control` |

The complete preregistration is [preregistration.md](preregistration.md). The
generated subset manifest and CPU audit are committed before any paid stage.
Existing V6.1 files, raw artifacts, gates, and interpretation are not edited.
Thinking is frozen to native `reasoning_effort=xhigh`, sampled generation
(`temperature=1.0`, `top_p=0.95`, `top_k=20`, seed `3838`), and exact
stage-specific preflight/protocol bindings.

The Qwen3.6 H3 cross-model reference uses the same strict
both-correct-and-semantic-switch endpoint as V6.2. It is a deterministic
secondary reanalysis of the archived V6.1 raw artifact, preserved with source
hashes in [`v61_reference_endpoints.json`](../../results/v6_2_reasoning_capability_control/v61_reference_endpoints.json).

## Reproduction

```bash
uv run --extra dev --extra modal pytest -q
uv run python scripts/run_v6_2_reasoning_controls.py --freeze-subsets --refresh-generated
uv run ruff check .
```

Paid stages are exposed only through the manual
[V6.2 GitHub Actions workflow](../../.github/workflows/v6-2-reasoning-capability-control.yml):

1. `local-controls`
2. `preflight-qwen38`
3. `qwen38-thinking-pilot` (validation only)
4. `qwen38-direct` (locked)
5. `qwen38-thinking-diagnostic` (locked)

Each later launch requires the exact predecessor Actions run ID. The preflight
also pins the protocol commit and all hashes; later stages fail closed if the
current commit or restored preflight differs.

Direct-versus-thinking paired sign-flip inference is run whenever the same
prospectively defined endpoint cells exist in both locked artifacts. The
absolute family gates are reported independently and do not control whether
that treatment-effect test is run.
