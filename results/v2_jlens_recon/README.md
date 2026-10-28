# Historical band failure reconciliation: measured result
Design: DESIGN_LOCKED.md (d9bf929, before the run). Countries, alpha 1.0, semantic swap, same 33 baseline-correct prompts, pseudoinverse coordinate swap. 106 s in function, est ~$0.1 (public rates).

| config | top-1 | top-1 and KL<1 | median KL | median dRMS |
|---|---:|---:|---:|---:|
| band 36-42 (7 layers), all positions | 24/33 | 1 | 3.43 | 0.100 |
| band 36-42, argument_through_end | 25/33 | 1 | 3.42 | 0.143 |
| band 36-43 (8 layers), all positions | 0/33 | 0 | 0.00 | 0.096 |
| band 36-43, argument_through_end | 0/33 | 0 | 0.00 | 0.137 |
| L40, all positions | 24/33 | 1 | 3.64 | 0.119 |
| L40, argument_through_end | 24/33 | 1 | 3.64 | 0.180 |

Measured: the historical H0 result for band 36-43 (0/33 on countries, median dRMS 0.096 in the original run) is reproduced by current code, and median dRMS for that config matches the historical value (0.096). The position mask does not matter here (both masks equal within 1/33). Adding layer 43 to the 36-42 band turns 24-25/33 into 0/33 with median output KL 0.00, i.e. the output is essentially unchanged, not disrupted. The mask hypothesis is not supported. This is consistent with layer 43, or with an even number of layers, being the factor; the data here cannot tell those apart (7 vs 8 layers differ in both).
Interpretation only (hypothesis): the pseudoinverse coordinate swap is applied to the current state at every layer in the set; a swap is an involution, so an even number of applications could undo itself. That has NOT been tested. Next test (design pre-written in DESIGN_PARITY.md) separates parity from layer 43.
Scope: this model, category, alpha 1.0. No claim about other operations, alphas, categories.

## Parity test result (design: DESIGN_PARITY.md, written before the run; 87 s in function, est ~$0.1)
Countries, alpha 1.0, mask argument_through_end, 33 prompts, semantic swap:
| layers | count | includes 43 | top-1 | top-1 and KL<1 | median KL |
|---|---:|---|---:|---:|---:|
| 36-40 | 5 (odd) | no | 25/33 | 1 | 3.43 |
| 36-41 | 6 (even) | no | 0/33 | 0 | 0.01 |
| 38-43 | 6 (even) | yes | 0/33 | 0 | 0.00 |
| 37-43 | 7 (odd) | yes | 25/33 | 1 | 3.52 |
Together with the previous table (7 layers works, 8 fails, 1 layer works): the 4 configs match the pre-written parity prediction and contradict the layer-43 prediction (36-41 fails without 43; 37-43 works with 43). In these configs, odd layer counts give 24-25/33 and even layer counts give 0/33 with output KL about 0, so the even-count edits leave the output nearly unchanged.

Operation math (src/jspace_policy/jspace_interventions.py, coordinate_swap_fp32): per layer, basis M = [source_dir, target_dir] (d x 2, layer-specific), pre = h @ pinv(M)^T (2 coordinates of the current activation), expected = pre + alpha*(flip(pre) - pre), h' = h + (expected - pre) @ M^T. alpha=0 is a no-op, alpha=1 sets the two coordinates to each other's values, alpha>1 overshoots. The hook is applied to the current activation at every layer in the set. For the same basis, two alpha=1 applications compose to the identity. Across layers the bases differ and the residual stream is shared, so exact cancellation is not guaranteed by that algebra. The cancellation seen here is empirical.

What this supports and does not support:
- Supports: the repeated per-layer flip protocol contributes to the band result, in these configs, enough to turn 24-25/33 into 0/33 depending on layer-count parity.
- Does not support: that healing, off-manifold drift or other mechanisms are excluded, that the model's J-space is irrelevant to the historical failure, or that the cause is general beyond this operation, alpha, category and prompt set.
- On the gate: under the actual preregistered protocol (8-layer band, this per-layer operation) the gate failed and that record stands. This analysis shows that result depends on the protocol's layer-count parity. Whether the gate protocol should be adapted is a decision, and any adapted protocol is a new post hoc design, not a retroactive change of the preregistered outcome.
- Not tested: even-count bands at other alphas, other categories, a single-pass patch computed once from clean states (the J-space paper's protocol is not verified here), strength-matched controls for the parity comparison. The earlier KL finding (1/33 under 1 nat) stands.

## Clean-state delta vs recurrent (design: DESIGN_CLEANSTATE.md, 65d923f; 205 s in function, est ~$0.2)
Countries, 33 prompts, argument_through_end. "clean" = each layer's coordinate change computed from the unedited run's activation at that layer and added once (modal_h0r_clean.py, clean_delta). "recur" = existing protocol.
| config | alpha | top-1 | top-1 and KL<1 | median KL | median dRMS |
|---|---:|---:|---:|---:|---:|
| clean band 36-43 (8) | 0.5 / 0.75 / 1.0 | 32 / 31 / 29 | 1 / 0 / 0 | 7.41 / 7.28 / 7.04 | 0.083 / 0.124 / 0.165 |
| clean band 36-42 (7) | 0.5 / 0.75 / 1.0 | 32 / 31 / 29 | 1 / 0 / 0 | 7.35 / 7.14 / 7.05 | 0.084 / 0.125 / 0.167 |
| recurrent band 36-43 | 0.5 / 0.75 | 1 / 1 | 1 / 1 | 0.08 / 0.09 | 0.015 / 0.029 |
| recurrent band 36-42 | 0.5 / 0.75 | 1 / 1 | 1 / 1 | 0.09 / 0.09 | 0.016 / 0.032 |
| clean L40 (control) | 1.0 | 26 | 1 | 3.67 | 0.180 |
(recurrent alpha 1.0: 8 layers 0/33, 7 layers 24-25/33, from the earlier table.)
Measured: with the clean-state delta, the 7- and 8-layer bands behave the same (no parity effect) and reach the target on 29-32/33. So the repeated-flip protocol is a protocol-dependent factor for the recurrent result in these configs. The clean-delta edit is not low-KL: median KL 7 nats and at most 1/33 prompts are top-1 and under 1 nat at alpha 0.5-1.0 (stronger than the single-layer edit, KL ~3.7). Alpha below 0.5 not yet tested for clean-delta.
Caveats: same op/model/category/prompts; the clean-delta is my own variant, not verified to equal the J-space paper's protocol; top-1 flips at 7 nats are drastic output changes, not a clean edit.

## Clean-delta at low alpha and unrelated control (design: DESIGN_CLEAN_LOWALPHA.md, written before the run; 251 s in function, est ~$0.3)
Countries, 33 prompts, argument_through_end, clean-delta, bands 36-43 and 36-42, alpha 0.05-0.4 (raw: raw/cleanlow_raw.json, analysis: cleanlow_analysis.txt).
- Target top-1 turns on sharply: band 36-43 gives 1, 9, 29, 32 of 33 at alpha 0.05, 0.1, 0.15, 0.2 (band 36-42: 1, 5, 25, 32). Median per-layer-mean dRMS at the 0.15 step is 0.025 for the 8-layer band versus 0.173 for single-layer L40 at alpha 1 (a 7x difference in the mean-over-layers number), but that metric is not a total-magnitude comparison. Adding the per-layer values (norm_comparison.txt, median over prompts that reach top-1): L40 sums to delta_l2 15.1 / dRMS 0.173; band 36-43 at alpha 0.15 sums to 17.2 / 0.193 (alpha 0.1: 11.8 / 0.141); band 36-42 at 0.15 sums to 15.6 / 0.177. So at the point where top-1 flips, total intervention magnitude is about the same as the single-layer edit. An earlier draft of this README and my 04:41 message said "about 4x smaller" and "much more efficient"; that was wrong and is withdrawn: no efficiency advantage is established.
- But the flips are high-KL: at alpha 0.1 the flipping prompts have median KL 4.6 (band 36-43; 6.0 for 36-42) while non-flipping prompts have 0.43 (0.15). Per-prompt smallest KL among each prompt's top-1 events across the grid: median 5.5 nats (band 36-43), only 1/32 prompts ever flip under 1 nat. By the prespecified rule (<= 1/33 at every alpha) this is the failure region for this grid, op, model and category. Not shown: that no valid low-KL editor exists.
- Unrelated-category control, same op and clean-delta: 0/33 top-1 at alpha 0.1-0.3 but median KL 0.88 / 7.17 / 7.36, essentially the same output disruption as the semantic swap at the same alpha (semantic 0.88 at 0.1, 7.09 at 0.2, 7.29 at 0.3). So the KL magnitude is not specific to the target answer, while the top-1 flip is (semantic 9-32/33 vs 0/33).
Unit test (CPU, numpy) for the swap algebra: tests/test_clean_delta_math.py (same-basis recurrent alpha-1 applied twice is the identity, clean deltas add and do not cancel, alpha 0 is a no-op); 3 passed.
Summary of the repair: the clean-state delta removes the even-count cancellation (8-layer and 7-layer bands now agree) and reaches target top-1 at about the same total magnitude as the single-layer edit. It does not yield low-KL valid edits in the tested grid. The original recurrent results are retained unchanged in this directory and in results/v2_smoke_tests.

## Denominators
Counts "x/33" are prompts among the 33 baseline-correct (source, target) pairs in the countries category (all 33 eligible rows per config; none missing). "Ever top-1: 32" in cleanlow_analysis.txt means 32 of those 33 prompts reached top-1 at some alpha; the 1 remaining prompt never did, so the "1/32" in the report is 1 of the 32 that ever flipped, equivalently 1/33 of eligible prompts. The prespecified failure rule (<= 1/33 at every alpha) counts prompts that are top-1 and KL<1 over the 33 eligible.

## Check against the J-space paper: provenance and protocol (corrected)
Source: "Verbalizable Representations Form a Global Workspace in Language Models", Gurnee*, Sofroniew*, Pearce, Piotrowski, Kauvar, Chen, Soligo, Bogdan, Ong, Wang, Thompson, Abrahams, Kantamneni, Ameisen, Batson, Lindsey (Anthropic), arXiv 2607.15495 v1, 2026-07-16 (https://arxiv.org/abs/2607.15495). This is the original Anthropic paper that introduces the Jacobian lens and J-space, not a replication or critique (an earlier note in this project mislabelled it). Official code: https://github.com/anthropics/jacobian-lens. Companion post: https://www.anthropic.com/research/global-workspace. Reading depth: full arXiv HTML (about 380 KB of text, fetched with curl; the earlier 50 KB fetch was truncated before the appendices) keyword-searched for the patch protocol plus sections 2, 3.1 and 3.3 read; the other appendices were not read in full (PART).

Protocol as stated in the text: patch in lens coordinates, V=[v_s v_t], c = pinv(V) h, h_patched = h + V(sigma(c) - c), sigma swaps the two entries, optionally scaled by alpha; "at every token position across a band of intermediate layers" / "over a fixed range of layers" (workspace range, e.g. ablations at L38-54 on Claude models). The text I found does not say whether c is recomputed from the already-edited activation at each layer or from the clean activation, and I found no per-prompt KL or fluency constraint on the swaps. So neither recurrent nor clean-delta is shown to be the paper's implementation; the official repo was not checked for this (the fetched README only covers lens fitting and slice pages).

Reference points from the paper (Claude models, different model and lens): the multi-function country/month/animal/number swap succeeds in 76/192 trials (top-1 target) at alpha=1 and 101/192 at alpha=2, with success varying by category and concentrated where the source concept is strongly present in the lens. Our Qwen countries top-1 rates are on 33 baseline-correct prompts of one category and are not comparable. The paper also contains a bandit "repeat vs switch" plan swap (Figure 14), the closest item to the project's game question; it is an n=2 illustration in the text I read.
