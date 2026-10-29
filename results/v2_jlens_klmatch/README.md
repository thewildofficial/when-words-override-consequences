# J-lens KL-matched control (review draft, exploratory)

Question: in the H0R diagnostic, target top-1 flips occurred only at alpha >= 1 (median KL ~4-5 nats), and controls had only been run at alpha 0.5. Are the flips target-specific or generic disruption?

Design (fixed in scripts/analyze_jlens_klmatch.py before the run): Qwen3.6-27B at pinned revision 6a9e13bd, lens n1000, countries set (n=33 eligible per cell), layers 32 and 34, argument_through_end, pseudoinverse coordinate swap. Semantic alpha in {1.0, 1.5}. Controls on an alpha grid 0.25-8: unrelated same-category swap, and norm-matched random basis (3 seeds). Pairs matched per (scenario, target, layer) on nearest log KL; accepted only if KL ratio in [0.8, 1.25].

Measured:
- Semantic: alpha 1.0 top-1 25/33 (L32), 24/33 (L34), median KL 5.1; alpha 1.5 31/33 both, median KL 6.5-6.9.
- Random basis: median KL <= 0.001 nats even at alpha 8, top-1 0 everywhere. It cannot reach semantic-level KL, so 0 pairs pass matching. It is inert, not a KL-matched control. The preregistered matching failed for this family.
- Unrelated same-category: reaches comparable KL (median 5.4-7.3 at alpha >= 1). 83/132 semantic rows found a match; median KL ratio 1.001. On accepted pairs: semantic top-1 75/83, unrelated 0/83; mean target log-odds gain 14.7 vs 8.6.

Caveats: acceptance depends on KL so accepted semantic rows skew to the high-alpha cells (49 unmatched rows dropped); single model and one category; one unrelated-direction control, not a distribution; the regime is still far outside the repo's KL validity limit (1 nat), so this does not validate the lens as a usable editing instrument and does not change any gate decision. Interpretation (provisional): at matched output disruption, a swap toward a different country's direction does not reach the target answer while the semantic swap does, so the alpha >= 1 flips in this setting are not explained by generic disruption alone. Decodability and causal use remain separate claims.

Cost: run 707 s on A100-80GB, estimated about $0.58 from public Modal rates (provider bill not read). Two earlier failed launches (client killed; CUDA OOM on batch size) add an estimated <= $0.3.
