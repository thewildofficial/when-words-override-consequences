# Semantic action mechanism investigation

Protocol: [`experiments/semantic_action_mechanisms`](../../experiments/semantic_action_mechanisms/README.md).
Branch: `codex/semantic-action-mechanisms`.

Status: **completed behavioral discovery and negative neural-site selection**.
The behavioral screen used twelve historical lexical families; its repeated
targets and label permutations are not independent samples. Removing the
opposed-state sentence raised assertion/prose accuracy from 15/48 to 47/48.
Explicit labels and layout also helped independently. The aligned follow-up
reproduced the state effect, but all fourteen patch sites repaired 0/4 selected
failures. No held-out inference occurred. The full interpretation and sampling
limits are in [FINDINGS.md](FINDINGS.md).

| Run | Status | Interpretation |
|---|---|---|
| [38140480392](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/38140480392) | CPU preparation failed before model inference | Tokenizer-only image omitted the optional Jinja2 dependency required for chat templates; no behavioral or causal result |
| [38140636684](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/38140636684) | Behavior scored; causal alignment gate failed | Irrelevant state and independent layout/binding effects; only five token-matched families, so no activation patch or held-out causal test |
| [38143348343](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/38143348343) | Token-matched behavior scored; discovery timed out after 13/14 sites | State effect reproduced with exact length/position matching; preserved 312 completed patch records and exact BF16 states; no complete site-selection verdict yet |
| [38144881756](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/38144881756) | Recovery CPU validation failed before Modal dispatch | Pytest console-entrypoint import path mismatch; no additional reservation or inference |
| [38145570702](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/38145570702) | Recovery setup failed before GPU admission | File loader received a string instead of a Path; no additional reservation or inference; covered by the new loading-path test |
| [38146041969](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/38146041969) | Final-site attempt timed out in weight loading | No final-site inference or capture; failed 120-second reservation retained |
| [38147047034](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/38147047034) | Full fourteen-site discovery recovered and completed | No qualifying site; 0/4 repairs at every site; zero identity error and bitwise cross-run reproduction; held-out confirmation withheld |

Final synthesis: [FINDINGS.md](FINDINGS.md). All reservations total
**$4.75216032**, including failed compute and unknown charges, below the same
$5 admission ceiling. This total is a conservative reservation, not an invoice.

The first run's protocol, provenance, failure and full $0.01635312 CPU reservation
are retained. The implementation repair adds pinned Jinja2 and a CPU image
contract check; it changes no stimulus, statistical rule, patch site, or gate.
The shared `reservations.jsonl` is restored from the Actions artifact before any
repaired run, so fresh jobs cannot reset earlier spend admission accounting.

Raw outputs and prepared token payloads use immutable run directories and gzip
JSON. Successful causal stages additionally preserve exact BF16 states and
per-site checkpoints on the `semantic-action-mechanisms-artifacts` Modal volume.

## Artifact map

Paths in this table are relative to this directory. Run-specific `protocol.json`
and `provenance.json` retain the frozen rules, input hashes and executing commit.

| Evidence | Durable files |
|---|---|
| Parent behavioral factorial | `semantic-mechanisms-qwen36-gha-38140636684/{prepare_raw.json.gz,behavior_raw.json.gz,behavior_analysis.json}` |
| Behavioral figure | [`semantic-mechanisms-qwen36-gha-38140636684/behavior_mechanisms.png`](semantic-mechanisms-qwen36-gha-38140636684/behavior_mechanisms.png) |
| Token-matched corpus and behavior | `semantic-mechanisms-qwen36-gha-38143348343/{prepared.json.gz,behavior_raw.json.gz,behavior_analysis.json}` |
| First thirteen patch sites and exact states | `semantic-mechanisms-qwen36-gha-38143348343/{recovered_sites.json.gz,discovery_states.pt,recovery_manifest.json}` |
| Completed fourteen-site sweep | `semantic-mechanisms-qwen36-gha-38147047034/{complete_discovery_raw.json.gz,discovery_analysis.json,site_freeze.json}` |
| Missing-site capture and cross-run numerical audit | `semantic-mechanisms-qwen36-gha-38147047034/{finish_raw.json.gz,final_site_states.pt,cross_run_numerical_audit.json}` |
| Final machine-readable status | [`final_summary.json`](final_summary.json) |
| All retained compute reservations | [`reservations.jsonl`](reservations.jsonl) |

There is no scored held-out artifact: `site_freeze.json` records `site: null`.
Preparation includes locked stimuli, which remained unscored. Failure artifacts
and full reservations are retained rather than replaced by later successes.

## Offline reproduction

Run from the repository root. These commands analyze committed scores on the
CPU; they do not load the model or dispatch a Modal function. The Modal extra
is needed to import the archived analysis functions in the runner modules.

```sh
uv sync --extra dev --extra modal
uv run --extra dev --extra modal python -m pytest -q
uv run --extra modal python - <<'PY'
import gzip
import json
from pathlib import Path

from jspace_policy.semantic_action_mechanisms import analyze
from modal_semantic_action_mechanisms import patch_analysis
from modal_semantic_mechanisms_aligned import aligned_analysis

root = Path("results/semantic_action_mechanisms")

def read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)

parent = root / "semantic-mechanisms-qwen36-gha-38140636684"
prepared = read(parent / "prepare_raw.json.gz")["discovery"]
behavior = analyze(prepared["dataset"], read(parent / "behavior_raw.json.gz"))
assert behavior == read(parent / "behavior_analysis.json")

aligned = root / "semantic-mechanisms-qwen36-gha-38143348343"
prepared = read(aligned / "prepared.json.gz")["discovery"]
behavior = aligned_analysis(prepared, read(aligned / "behavior_raw.json.gz"))
assert behavior == read(aligned / "behavior_analysis.json")

final = root / "semantic-mechanisms-qwen36-gha-38147047034"
raw = read(final / "complete_discovery_raw.json.gz")
patches = patch_analysis(raw)
assert patches == read(final / "discovery_analysis.json")
assert len(raw["records"]) == 336 and len(patches["sites"]) == 14
assert patches["selected"] is None
assert all(site["failures"] == 4 and site["repairs"] == 0 for site in patches["sites"])
reservation = sum(json.loads(line)["ceiling_usd"] for line in
                  (root / "reservations.jsonl").read_text().splitlines())
assert abs(reservation - read(root / "final_summary.json")["reservation_total_usd"]) < 1e-9
print("All three archived analyses reproduced; 14 sites, no selected site.")
print(f"Retained reservation: ${reservation:.8f}")
PY
```

The figure script refuses to overwrite an existing image. To regenerate it,
copy the parent's two raw gzip files into a fresh temporary directory and pass
that directory to `uv run python scripts/plot_semantic_action_mechanisms.py`.
Keep the committed figure and raw artifacts intact.

## Model and execution record

The checkpoint is `Qwen/Qwen3.6-27B`, revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. Scoring uses BF16, thinking off,
cache reuse off and singleton forwards after the declared batch-parity fallback.
Patch layers are 8, 16, 24, 32, 40, 48 and 56, at `state_end` and `policy_end`.
These anchors precede the payoff/goal; no answer-token patch was used.

The complete sweep has 336 forward/reverse records across twelve discovery
families. Numerical identity error is zero. The final-site replay error is zero,
and the [cross-run audit](semantic-mechanisms-qwen36-gha-38147047034/cross_run_numerical_audit.json)
records bitwise equality for 24 baseline logit pairs and BF16 vectors.
Those checks validate this execution; they do not increase the four-failure
sample or establish that these anchors contain the relevant representation.

Original freezes and prospective recovery/sample amendments are in
[`experiments/semantic_action_mechanisms/`](../../experiments/semantic_action_mechanisms/README.md)
and its [aligned follow-up](../../experiments/semantic_action_mechanisms/aligned/README.md).
Ordinary branch pushes and merging to `main` do not trigger paid stages. The
workflow retains manual dispatch and explicit branch commit markers for the
archived runners, subject to their cumulative reservation gates. A new study
needs its own prospective design and budget; this record does not authorize it.
