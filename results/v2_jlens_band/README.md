# J-lens band vs single layer: top-1 versus output KL (review draft, descriptive)

Same model, lens, data and edit as results/v2_jlens_klmatch (Qwen3.6-27B rev 6a9e13bd, countries, 33 eligible per cell, argument_through_end, pseudoinverse swap). Semantic swap on single layers 32, 34, 40, band 32-34 and band 36-42, alpha 0.25-1.5. Unrelated-category control on layer 40 and band 36-42. Descriptive curves only; no matching procedure here. Raw: raw/band_raw.json; table: analysis.txt.

Measured (top-1 target of 33; median output KL in nats):
- alpha 1.0: single layers 25/33, 24/33, 24/33 (L32, L34, L40; median KL 5.1, 5.1, 3.6); band 32-34 24/33 (4.3); band 36-42 25/33 (3.4). Band and single layer are indistinguishable at alpha 1.0.
- alpha 1.5: single layers 31-32/33 (median KL 6.5-6.9); band 32-34 31/33 (7.0); band 36-42 3/33 with median KL 10.4 and mean delta-RMS ratio 1.27 (the summed per-layer writes over-drive the stream).
- alpha 0.75: single layers L32/L34/L40 each 9/33 at median KL 0.18, 0.19, 0.71 but mean KL 2.0, 1.8, 2.1 (heavy tail: flips concentrate on high-KL prompts). Band 36-42 and band 32-34 gave 1/33 and 3/33.
- alpha <= 0.5: at most 1/33 everywhere.
- Unrelated-category control: 0/33 at alpha 0.25-0.75 and at alpha 1.5; 1/33 at alpha 1.0 for both L40 and band 36-42, at similar KL to the semantic rows (3.5-3.9 median).

Reading (provisional): no evidence here that band editing per se cancels the single-layer effect at matched alpha; the earlier "band fails" contrast looks like a strength-scale issue (alpha applied per layer, summed over layers, saturating around alpha 1 to 1.5). Target-specific flips exist at alpha 0.75 for single layers, but only on a heavy-tailed subset whose mean KL (~2 nats) is still above the repo's 1-nat validity limit. Nothing here shows a valid, low-disruption editing instrument. Caveats: one model, one category, 33 prompts per cell, one unrelated-control family, and the band comparison varies layer count and total delta-RMS together. The original H0 band run (0/90) used its own scaling and is not re-derived here.

Cost: 317.6 s on A100-80GB, estimated about $0.26 (public Modal rates; provider bill not read).
