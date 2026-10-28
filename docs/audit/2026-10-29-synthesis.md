# Synthesis at ~$3.3 spent (estimates): what survives, what does not, what is open
Labels: M = measured in this repo, H = hypothesis, L = literature (reading depth in the literature map).

## Measured, scoped
- M: J-lens swap edits (Qwen3.6-27B, countries category, single layers 32/34/40, band tests) reach target top-1 on 24-25/33 prompts only at alpha >= 1 with median KL 3.9-5.4 nats; at per-prompt KL < 1 nat only 1/33 per layer (alpha grid 0.5-1.0 step 0.05). Unrelated-category control 0-1/33. Scope: this grid, model, category; no claim that no valid editor exists.
- M: Stage 1 locked readout by policy: final-position band J-lens K is positive only when the report equals the state (policy T). Under policy M, M/Q is positive and K negative. Cross-policy probe transfer fails (0.475/0.250, chance 0.25).
- M (round 2, F6 format, gate 96.5%): final-position J-lens Q-K -0.050 [-0.285, +0.186], logit lens +0.153 [-0.044, +0.358]. Inconclusive. When state = report, K=Q=+0.526.
- M: No-CoT arithmetic with offset/sum fails (36-59%) in this model, one explicit lookup passes (99%), two chained lookups fail (36.5%).
- M: Kuhn EV (4 renderings, 480 prompts) gated: best 74.6% with prefill, 71.5% explicit no-explain; wording alone moved 52.7 -> 71.5%. The model's actions barely vary with opponent policy inside (card, history) cells.

## Not established
- Whether the lens reads the state or the report (inconclusive). Whether any internal value or planning representation exists in this model. Whether a J-lens editor exists outside the tested grid. Anything causal.
- Recurring report-candidate signal at the newline after "assistant" (offset 4) in both Stage 1 and round 2: located on a template token, mechanism unknown, no control.

## Gaps and a different direction (hypotheses)
- H: the original "miscalibration" is historical lens failure in the earlier J-lens runs, and the unrecovered H0 band scaling (0/90) is still not recovered. A re-derivation from the stored archive (not new GPU) is the audit-led next step before new lens experiments.
- H: a game or task that the model does no-CoT at >= 85% with a model-computed (not visible-table) state is the missing ingredient for readout vs state tests. Candidates need a competence pilot (~$0.1 each) before any lens work.
- Another model may pass where Qwen3.6-27B does not; model choice is a gap, not a closure.
