# Semantic action mechanisms

Prospective investigation, 11 October 2026. Branch:
`codex/semantic-action-mechanisms`. No outcome is implied by this protocol.

**Completed-study status:** behavioral discovery and the token-aligned
fourteen-site patch sweep are archived. No site qualified; held-out inference
did not occur. See the [final findings](../../results/semantic_action_mechanisms/FINDINGS.md)
and [offline reproduction](../../results/semantic_action_mechanisms/README.md#offline-reproduction).
The protocol and amendments below retain their prospective information state.

The behavioral screen holds the correct action fixed while manipulating an
irrelevant private-state sentence. A second connected factorial separates
prose/table layout from explicit A/B binding in the consequence mapping.
Assertions and opaque actions share identical transition maps. Both targets and
both label assignments are crossed within every lexical family. There are
1,120 contexts and 3,360 queries, including a small device replication.

Discovery uses the twelve historical RBG-4 concept pairs with newly balanced
conditions. The held-out set has twelve disjoint concept pairs. Analyses cluster
on lexical family; targets, label permutations and frames are repeated measures.
Discovery contrasts are exploratory. The locked stage evaluates only the selected
matched causal contrast, not the full behavioral factorial.

## Sequential gates

1. CPU preparation validates all answer continuations, prompt offsets, hashes,
   oracle actions, and both splits before scoring.
2. Behavior requires deterministic singleton replay. Four-row batches must meet
   the frozen parity tolerance and choice agreement; otherwise the same invocation
   uses singletons. No tolerance widening or automatic retry.
3. At least 95% opaque action accuracy and 90% report accuracy are required.
   Prefer the state contrast if its assertion-minus-opaque accuracy interaction is
   at least 15 points and assertion margin difference at least 0.5 logits.
   Otherwise test binding only if its prose accuracy effect is at least 15 points
   and margin difference at least 0.5 logits. With neither effect, stop.
4. Patches use exact BF16 residuals and singleton forwards. Candidates are layers
   8, 16, 24, 32, 40, 48, 56 at `state_end`/`policy_end` for the state contrast,
   or `policy_end` for binding. Every site precedes the payoff/goal paragraph.
   The changed state and the goal are distinct variables; source and destination
   have the same oracle action. No answer/readout-token patches are included.
5. Discovery examines one predeclared target/permutation per family, without
   selecting only failures. State patches require equal token lengths and anchor
   positions, and at least eight eligible families. Capture/replay and identity
   error must be at most 1e-5 candidate logits.
6. Freeze one qualifying site using discovery, then evaluate fresh lexical
   families with forward, reverse, identity, wrong-game and consequence-report
   controls. The conjunction of locked gates is in `protocol.json`.

A positive result supports causal contribution under the selected contrast and
site. It does not identify a complete circuit, distinguish all possible shortcut
algorithms, or establish that a full-residual patch transports only one feature.
A negative single-token patch does not exclude recurrent or distributed state.
Layout and labels necessarily change some wording/length; tokenization and
alignment exclusions are recorded. Neutral text is an imperfect deletion control.

## Execution and accounting

The original checkpoint is pinned. Thinking and cache reuse are disabled.
Resources are one A100-80GB, eight CPU cores and 64 GiB memory. Timeouts are
300 seconds for CPU preparation, 1,500 each for behavior and discovery, and 900
for locked confirmation. Retries are disabled. At rates checked against
[Modal pricing](https://modal.com/pricing) on 11 October, the complete sequential
reservation is approximately **$4.45** including 20% buffer and a shutdown
allowance, within a **$5 coordinator ceiling**. Actual provider charges may
include image building/other workspace activity; this is a study admission limit,
not an account-wide billing control.

One coordinator persists all reservations before dispatch. Failed and unknown
charges retain their full reservation. If a workflow fails, download and commit
its ledger before any later dispatch; never reset accounting in a fresh job.
The run directory refuses overwrites. CPU statistics execute on the GitHub
runner after each remote return, allowing the GPU container to shut down.
Exact residuals are retained on the dedicated Modal artifacts volume; JSON
results, prepared inputs, freezes, hashes and reservations are uploaded together.

The workflow supports manual dispatch once GitHub recognizes it. On this new
branch, a deliberate push with `[run-semantic-mechanisms]` in the head commit
starts the full pipeline; `[prepare-semantic-mechanisms]` starts CPU preparation
only. Ordinary pushes skip the job. Neither marker is an automatic retry.

Local equivalent:

```sh
uv run --extra dev --extra modal python -m pytest -q tests/test_semantic_action_mechanisms.py tests/test_semantic_action_modal.py
uv run --extra modal modal run modal_semantic_action_mechanisms.py --run-id UNIQUE_RUN_ID
```

Related theory: [mechanism synthesis](../../docs/next-sprint/semantic-action-mechanisms.md).
