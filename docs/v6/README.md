# V6 — Epistemic Strategic Experiments

> **Interpretation warning:** An independent post-run audit found stimulus
> defects in the higher-order, core policy, action→report, and safety-pair
> endpoints. Read [the erratum](erratum-prompt-completeness.md) before using
> the historical findings. The repaired follow-up is
> [V6.1-ES-2](../v6.1/README.md).

V6 studies whether a language model keeps agent-indexed beliefs distinct and
uses them in strategic decisions. The study namespace is deliberately stable:

| Object | Canonical name |
|---|---|
| Study | `V6-ES-1` |
| Python module | `jspace_policy.epistemic_strategic_experiments` |
| Config/data | `configs/v6/epistemic_strategic/` |
| Results | `results/v6_epistemic_strategic/` |
| Modal app | `jspace-v6-epistemic-strategic` |
| Branch | `research/epistemic-strategic-experiments` |

The primary model is the pinned `Qwen/Qwen3.6-27B` checkpoint already used by
the repository. Qwen3.8-27B is registered separately at a pinned Hugging Face
revision, but remains opt-in until its exact primary preflight contract matches
(rendered prompts, prompt token IDs, and candidate token IDs). The two models
never share a behavioral or mechanistic gate.

## What is frozen locally

The CPU-only generator produces 1,740 rows across three splits:

| Family | Rows | Primary question |
|---|---:|---|
| `core_tom` | 96 | Does the modeled receiver belief control action while hidden actual belief does not? |
| `higher_order` | 108 | Can self, modeled-other, and other-model-of-self reports remain distinct across visibility? |
| `provenance` | 1,440 | Does source, truth status, time, or surface alter a report of the same claim? |
| `strategy_pressure` | 96 | Does safety reweighting change action while audit framing alone does not? |

The CPU protocol result is:

- dataset SHA256: `e5192e32126fcdfe5a606708a98a8264753092871bce3f427de8d5254d681995`
- config SHA256: `8eed1a0c09ad37f329137e119f6ac7a973a86592a618b1de074d1d7553cb3706`
- all four family structural gates: passed
- GPU seconds: `0`
- model forward passes: `0`

The primary Qwen3.6 model run is now retained in
[`../../results/v6_epistemic_strategic/`](../../results/v6_epistemic_strategic/).
Black-box and forced-choice behavior were both highly format-compliant, but
the preregistered family gate failed for `core_tom`, `higher_order`, and
`strategy_pressure`; `provenance` passed. Therefore activation discovery was
not run. This is a heterogeneous behavioral result, not evidence for a single
pooled capability.

See [`epistemic-strategic-preregistration.md`](epistemic-strategic-preregistration.md)
for hypotheses and promotion thresholds, [`controls-and-interpretation.md`](controls-and-interpretation.md)
for confounds and alternative explanations, and
[`../../results/v6_epistemic_strategic/control_audit.json`](../../results/v6_epistemic_strategic/control_audit.json)
for the machine-readable audit. The descriptive primary findings and next
experiments are in [`primary-findings.md`](primary-findings.md).

## Measurement modes

The branch keeps three evidence levels separate:

| Mode | Access | Claim allowed |
|---|---|---|
| `blackbox` | free-generation text only | observable behavior and parseability |
| `behavior` | one-step candidate-token logits, no activations | forced-choice behavioral preference |
| `activation_discovery` | residual stream at the decision position | observational readability on held-out splits |
| `activation_locked` | currently sealed | only after a fresh natural counterfactual patch protocol is frozen |

The black-box path and logit path used the same locked rows, but their outputs
are reported separately. Neither path can establish that a decoded belief is
causally used.

## Commands

Local protocol and controls:

```bash
uv run python scripts/run_v6_epistemic_strategic_controls.py --freeze-dataset
uv run pytest -q tests/test_epistemic_strategic_experiments.py
uv run ruff check .
```

Modal stages are manual. Dependent stages materialize or verify their
model-specific preflight artifact, while behavioral and mechanistic gates still
fail closed:

```bash
uv run --extra modal modal run modal_v6_epistemic_strategic.py::preflight
uv run --extra modal modal run modal_v6_epistemic_strategic.py::blackbox
uv run --extra modal modal run modal_v6_epistemic_strategic.py::behavior
uv run --extra modal modal run modal_v6_epistemic_strategic.py::trajectory
uv run --extra modal modal run modal_v6_epistemic_strategic.py::activation_discovery
```

For the registered Qwen3.8 replication, run only its preflight first:

```bash
uv run --extra modal modal run modal_v6_epistemic_strategic.py::preflight \
  --model-key qwen38_27b
```

Paid stages reject the disabled replication model until that preflight is
reviewed and the config is explicitly promoted. The locked causal command is
intentionally sealed:

```bash
uv run --extra modal modal run modal_v6_epistemic_strategic.py::activation_locked
# expected: fail-closed protocol error until patch coordinates and controls are frozen
```

The GitHub Actions wrapper is
[`.github/workflows/v6-epistemic-strategic.yml`](../../.github/workflows/v6-epistemic-strategic.yml).
Use `gh workflow run` only for the stage that has passed its local prerequisite;
the workflow uploads the immutable result directory.
