# Semantic action mechanism investigation

Protocol: [`experiments/semantic_action_mechanisms`](../../experiments/semantic_action_mechanisms/README.md).
Branch: `codex/semantic-action-mechanisms`.

| Run | Status | Interpretation |
|---|---|---|
| [38140480392](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/38140480392) | CPU preparation failed before model inference | Tokenizer-only image omitted the optional Jinja2 dependency required for chat templates; no behavioral or causal result |
| [38140636684](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/38140636684) | Behavior scored; causal alignment gate failed | Irrelevant state and independent layout/binding effects; only five token-matched families, so no activation patch or held-out causal test |

The first run's protocol, provenance, failure and full $0.01635312 CPU reservation
are retained. The implementation repair adds pinned Jinja2 and a CPU image
contract check; it changes no stimulus, statistical rule, patch site, or gate.
The shared `reservations.jsonl` is restored from the Actions artifact before any
repaired run, so fresh jobs cannot reset earlier spend admission accounting.

Raw outputs and prepared token payloads use immutable run directories and gzip
JSON. Successful causal stages additionally preserve exact BF16 states and
per-site checkpoints on the `semantic-action-mechanisms-artifacts` Modal volume.
