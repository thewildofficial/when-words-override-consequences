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
