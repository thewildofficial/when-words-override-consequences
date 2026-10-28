# Clean-delta band at low alpha + unrelated control (design before run)
Question: does the repaired (clean-delta) band edit reach target top-1 at per-prompt KL < 1 nat for some alpha < 0.5? Countries, same 33 prompts, argument_through_end, bands 36-43 and 36-42, alpha in {0.05, 0.1, 0.15, 0.2, 0.3, 0.4}; unrelated-category control (same op and clean-delta) for band 36-43 at alpha {0.1, 0.2, 0.3}.
Prespecified regions (per layer set, any alpha): success if >= 7/33 prompts (20%) are top-1 and KL<1 at some single alpha; failure if <= 1/33 (about 5%) at every alpha; between = inconclusive. Reported also: top-1 count, median KL, per-prompt validity, unrelated control at the same alpha.
Scope: this grid, op, model, category. A failure does not show no valid editor exists. Est ~$0.4.
