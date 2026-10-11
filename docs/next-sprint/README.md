# Next research sprint: monitoring before elicitation

Status: **research proposal, not a preregistration or a GPU result**. Audit of
commit `eb23897f12bf902763f43f4772bdf8b0ebd3f5e3`; literature accessed September 2026.
No new paid model run was launched during this review.

The proposed organizing question is:

> When verbal reports and consequential actions diverge, can we recover the
> relation controlling action without changing it by asking the model to explain?

Start with [the audit](audit.md), then [the literature comparison](literature.md),
then [the experimental design](experiments.md) and
[virtual tools with Qwen3.8, including the revised $30 budget](virtual-tools-qwen38.md).
Living claims: [claim ledger](claim-ledger.md) (includes C11–C13 GHA path;
C14–C15 GHA 16-base discovery baseline; C16 poisoned self-talk cell map).
Ops decisions: [decision log](decision-log.md) (RR-D001 GHA ledger).
Harder-games dial: [break the Direct ceiling](../experiments/report_reactivity/harder-games-break-the-ceiling.md)
(`history_mode=redundant`).
Novelty bar: prefer experiments with deployment/safety/ToM implications (tool use, oversight, multi-agent, HITL, self-supervision) over quirks.

For the current merged-main status, use the [research record](research-record.md).
It separates infrastructure, prepared payloads, failed instrument attempts,
discovery scores, and locked confirmation, and defines the naming fields used
for new work.
Repository cleanup and artifact-retention decisions are recorded in the
[repository hygiene note](repo-hygiene.md).

The original aspiration—understanding how internal information controls strategic
reporting—is still worthwhile. The strongest existing evidence concerns semantic
interference and report-conditioned action, rather than autonomous scheming.
J-lens readouts are a candidate instrument for that question, not the research
objective themselves.

## Completed semantic action mechanism follow-up

On 11 October 2026, a separate $5-capped investigation tested why the semantic
action gap appears. Removing an irrelevant opposed-state sentence raised
assertion/prose accuracy from 15/48 to 47/48; explicit A/B labels and table layout
each helped independently. The token-matched follow-up reproduced the state
effect, but no site in the fourteen-site patch sweep qualified for confirmation.
The patch sample contained only four failures and consistently A-first options.
Held-out inference did not occur.

Read the [completed findings](../../results/semantic_action_mechanisms/FINDINGS.md),
[offline reproduction and run history](../../results/semantic_action_mechanisms/README.md),
and [pre-run theory/literature note](semantic-action-mechanisms.md). This result
is separate from the historical $30 proposal above.

## Evidence map

| Track | Best-supported conclusion | What remains open |
|---|---|---|
| H0 / H0R / V3 | Coordinate edits can shift preferences; reliable substitution gate failed; V3 fresh intervention never ran | Whether a calibrated instrument can mediate a new, well-defined computation |
| V2 Stage 1 | State decodable with residual probes; frozen J-lens state criterion failed | Relational encoding versus the limitations of selected token scores |
| V2 atlas / trajectories | Generic optimization and late answer preparation; weak strategy specificity | Same-action, different-reason monitoring with stronger relational readouts |
| V4 | Trajectory-dependent report degradation, exploratory framing effects | Replication beyond specific lexical and formatting choices |
| V5 RBG-4 / 5B | Meaningful action vulnerable to redundant correct history; tables/opaque interfaces protect behavior | Why, and whether it transfers to real tool interfaces |
| V5 RBG-5B patch | 2/50 repairs; frozen causal endpoint and identity control failed | A reliable intervention on an identified relation |
| V5 RBG-6 | Correct report transcripts strongly improve subsequent action; swapped reports strongly misdirect it | Content, recency, instruction, and internal routing contributions |

## Organization decision

Keep frozen paths and results intact. Moving old runners breaks manifests,
workflow paths, local-file uploads, imports, and historical reproduction commands.
This directory is the new navigation layer; versioned directories remain the
historical record. A future experiment should get one stable descriptive ID,
not another ambiguous layer of V5/RBG/E2 labels. Report-reactivity live IDs:
`ask_mid_trajectory`, `rename_invariant` — see
[`experiments/report_reactivity/NAMING.md`](../experiments/report_reactivity/NAMING.md).

For new work use `experiments/report_reactivity/` for its design, config and CPU
preparation; keep a thin root Modal entrypoint if its local upload paths require
it. Put reusable model scoring/capture code in a new tested module, and immutable
outputs in `results/report_reactivity/<run-id>/`. A manifest must connect each
result to dataset, code, tokenizer, checkpoint, lens, environment and spend.

Before sharing a paper, make a single claim ledger: claim, experimental unit,
raw artifact, analysis command, discovery/confirmation status, limitations.
This is more valuable immediately than physically relocating 29,000 lines.
