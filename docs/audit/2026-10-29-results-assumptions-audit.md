# Results and assumptions audit (2026-10-29) - REVIEW DRAFT

Status: provisional. Nothing here is a finding for publication. M = measured in a committed raw artifact (re-computed from raw rows in this audit); D = reported in repo docs, raw not re-checked; H = hypothesis. Decodability is kept separate from causal use.

## A. J-lens edit failures: what the raw rows show

Source: results/v2_h0r_diagnostic/raw/{strength_sweep,layer_sweep,controls,cumulative_layers}.jsonl (re-computed here).

M1. Dose-response, single layer, argument-through-end writing (strength_sweep, n=144 per alpha, pooled over layers 32 and 34):

| alpha | target top-1 | median KL (nats) | mean target log-odds gain | mean delta-RMS ratio |
|---|---|---|---|---|
| 0.25 | 0/144 | 0.001 | 0.44 | 0.022 |
| 0.5 | 3/144 | 0.005 | 1.61 | 0.043 |
| 1.0 | 78/144 | 3.885 | 11.54 | 0.086 |
| 1.5 | 102/144 | 5.293 | 15.26 | 0.129 |
| 2.0 | 102/144 | 5.377 | 15.99 | 0.173 |

M2. layer_sweep (alpha 1.0): 850 of 1200 rows have KL >= 1 nat, and 457 of those are target top-1. Rows with KL < 0.1 had 0 top-1 of 50.

M3. Controls (alpha 0.5 only; 48 each, or 192 for the layer-34 out-of-window and similar): norm-matched random basis, unrelated same-category, direct-answer direction, identity and out-of-window layer all gave 0 top-1. No control was run at alpha >= 1.0, where flips occur.

M4. cumulative_layers (alpha 0.5, layers 36-42 and others) gave 1 top-1 per 48 for every configuration. The layer_sweep and position_sweep used alpha 1.0. So the "single layer works, band fails" contrast in the H0 summary compares different effective strengths in this diagnostic. (D: the H0 smoke test itself used alpha 1.0 for both topologies; its per-layer scale in band mode is not re-derived here.)

Interpretation (H, not established): the target top-1 flips occur only in a regime where the output distribution is changed by several nats, which is far outside the repo's own KL validity limit (D: docs/v2/h0r-diagnostic-report.md says alpha >= 1.0 exceeded it). Without a random-direction control at matched KL, flips at alpha 1.0 cannot be attributed to the semantic direction. The 27.8% H0 one-layer/all-position result should be read as unvalidated until that control exists. The repo's own gate already treated H0 as failed, so this does not change a gate decision; it weakens the "surprisingly useful local edit" reading.

Open and cheap: a KL-matched control (random and unrelated directions scaled to equal KL or equal delta-RMS at alpha 1.0) on the stored prompts. This needs GPU forward passes (model not stored in activations form), est. well under 1 USD on an A100-80GB. Not yet run.

## B. Lens vs residual dissociation

D: V2-SR-1 locked, residual state probe 1.000 / 0.992 vs J-space band score K -0.108 / -0.017. results/v2_stage1/raw/mechanistic_locked.json has 480 prompt rows with per-row band_K and cross-policy probe predictions but no stored activations. So a "better lens" comparison needs activations to be regenerated (GPU). No zero-cost lens comparison is possible from stored data for the Stage 1 prompts. results/semantic_action_mechanisms/*/{discovery_states,final_site_states}.pt are stored states from a different study (Qwen3.6) and have not been audited for this purpose.

## C. EV / value decode in Kuhn poker (X1)

Prior art, abstracts read plus partial full text: arXiv 2605.00226 (Kuhn, beliefs probed and steered), arXiv 2607.19369 (poker probes explained mostly by visible betting composition; abstract read). Consequence: any decoded-EV result must beat a public-history composition baseline, use solver action-values under a stated opponent policy, and control position bias. Novelty not established.

## D. Scope

No GPU run, no spend. Branch contains this note only.

## E. Update 2026-10-29 03:05 IST

- PR #48 (KL-matched control) result: semantic swap top-1 75/83 vs unrelated-category swap 0/83 on KL-matched accepted pairs; norm-matched random basis is inert (KL <= 0.001 nat) so no random KL match was possible. This partly retracts the section A hypothesis that alpha >= 1 flips are generic disruption; they are target-specific relative to the tested control, still in a ~5-nat regime outside the repo's 1-nat validity limit. Selection caveat: matching is conditioned on KL.
- epistemic-steering docs/evidence-ledger.md (read in full): within-task held-out probes AUROC 0.712-0.950, but leave-one-domain-out universal probe 0.4777 (a 0.9962 result was retired as train-at-eval leakage), rank-3 PLS-DA strict transfer 0.358, and the repo marks "universal correctness direction" and "hidden-state steering improves correctness" as claims on hold. Consequence for X1: any decoded-EV result needs policy- and rendering-held-out splits, since within-condition decodability is known to overstate transfer in this project's own history.
- Still unread: docs/next-sprint/claim-ledger.md, V2-E1 atlas raw rows.
