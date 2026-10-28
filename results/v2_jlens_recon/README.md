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
