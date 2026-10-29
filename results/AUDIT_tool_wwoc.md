# Audit: tool-use WWOC and native-generation results (Qwen3.6-27B, rev 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9). Written 2026-10-29 ~07:05 IST.
Scope: Y6-Y14 from the overnight sprint. All claims one model, one generator family (V5-RBG-4 `inverse_evidence_games._make_row`, strategic frame, opposed incentive), greedy, thinking template off. Nothing here is a mechanism, utility or novelty claim.

## Claim table
| # | claim | evidence | n | estimate [95% game-cluster CI where computed] | caveat |
|---|---|---|---|---|---|
| 1 | A/B letter readout: worded assertions 44% vs opaque tokens 100% on fresh games | Y7 (results/y7_tool_wwoc/y7_logprob_analysis.txt) | 128/cond | 56/128=44% [35,52] vs 128/128 | forced-choice log-prob; not same cohort/measure as frozen RBG-4 79%->55% |
| 2 | Executed tool calls with worded tool names reach 93-94% vs 100% opaque; gap ~ -6 to -7 pts | Y11 119/128 (-7.0 [-10.9,-3.9]); Y12 240/256 (-6.2 [-9.8,-3.1]); Y8 corrected 120/128 (-6.3 [-10.2,-2.3], post hoc mixed caps) | 128-256 | see left | one model, one generator; name effect not independent conceptual replications; worded-name rows reason ~120 tokens first; error = target-claiming tool by construction |
| 3 | Tool-name log-prob (Y7) overstated the executed name gap (70% vs ~93%) | Y7 vs Y11/Y12 | 128 | 89/128=70% [61,77] | different readouts (name log-prob after tool prefill vs generation) |
| 4 | Neutral position-bound names (tool_one/two) + worded descriptions: ~73-75% executed | Y12 186/256, Y13 191/256 | 256 | -27.3 [-33.6,-21.5] vs opaque | name == list position by construction; position and rebinding confounded |
| 5 | Written deliberation recovers most of that loss | Y13: generic restatement 237/256 (92.6%), explicit which-tool 250/256 (97.7%) vs 191/256 | 256 | +18.0 [13.3,22.7]; +23.0 [18.4,28.1]; restate-generic +5.1 [1.6,8.6] | changes reasoning text and attention; not binding-specific; "any text" not tested beyond two prompts |
| 6 | Fixed-identity names recover about half of the neutral gap | Y14 217/256 (84.8%) worded, 209/256 (81.6%) blank desc | 256 | worded-blank +3.1 [-3.1,9.8] | cross-run vs Y13 optok 93% (different games; Y14 prose names tools in policy, Y13 prose did not); neither locked prediction (>=88% or ~75%) held |
| 7 | Native free-word beats A/B log-prob on equity-derived poker decisions (same first-60 items) | Y6 + results/y6_native_generation/same60_ab_vs_native.txt | 60/task | D3 36 vs 49 correct, D4 32 vs 41, D2 54 vs 54 | A/B plain rendering only; Fold recall deficit persists (D3 21/32, D4 13/32) |
| 8 | Step-by-step text rescues derive tasks | Y6 CoT D3 60/60 (400 cap); Y10 D2 58/60, D4 59/60 (1000 cap, post hoc, 3 unparsed) | 60/task | n/a | cap and format changed together; no matched sampling; not reasoning-causal |

## Failed or unmet predictions (pre-locked)
- Y7: tool_assertion_name <= 90% was not tested as such; forced-choice 70% but executed 93%.
- Y8: first pass 80-token cap truncated 98/384 rows; repair attempts Y8b (ran old code, 6/98 parsed) and Y9 (92 parsed). Y8 numbers are post hoc mixed-cap; Y11 is the uniform-cap replication.
- Y11: desc_named >= desc +10 pts (lookup confound hypothesis): not supported (-2.3 [-9.4,5.5]).
- Y13: binding-specific pattern (restate >> generic ~ desc) not seen.
- Y14: neither the re-binding (>=88%) nor the neutral-semantics (~75%) prediction held; result intermediate.
- Y5 base-vs-instruct: gate failed (D1 68%/59%), no inference.
- Y4 probe: inconclusive (input-number baseline AUC 1.0).

## Confounds not resolved
Position vs name re-binding in the neutral-name conditions (Y12/Y13); gold-tool identity skew in Y11 and Y14 (alpha 120 / beta 136); tool-token naming differences (send_<token> vs tool_alpha); cross-run comparisons use different games; the contrarian policy makes "wrong" identical to "target-claiming tool" so no wording-vs-consequence causal claim; no matched meaningless-description control except Y14 blank (policy text still carries the wording).

## Reproducibility manifest
- Branch experiments/jlens-kl-matched-controls, PR #48 (draft; untouched). Heads at last verified push: 506d81f (Y12), d115f61 (Y13); this commit adds Y14 + audit.
- Model Qwen/Qwen3.6-27B revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9, bf16 on Modal A100-80GB, greedy, `enable_thinking=False`, chat-template tools format.
- Modal app y1-game-battery; functions: y6_remote (native gen, caps 8/400), y7_remote (A/B logits + tool-name log-prob), y8_remote (cap 80, batch 8; later edited to 400), y9_remote (cap 400, reruns), y10_remote (cap 1000), y11_remote/y12_remote/y13_remote/y14_remote (CAP=400, batch 12 or 16). Source: modal_y1.py. Spawn/collect: run_y*_spawn.py.
- Games: Y7/Y8 200-231; Y11 232-263; Y12 264-295; Y13 296-327; Y14 328-359; frozen RBG-4 used 0-23. Generator row seeds from `_rng("v5-inverse-evidence", base)`.
- Parsers: first `<function=NAME>` for tool calls; Y6 free word = first word, CoT = last `Final answer:`; analysis scripts in scripts/analyze_y*.py (Y12-Y14 adapted by text substitution; check analysis.txt for the printed condition names).
- Manifests: results/y7_tool_wwoc/y8_canonical_manifest.json (384 rows, final sources pass1 286 / y9 92 / y8b 6), y11_manifest.json, y12_manifest.json; raw outputs in raw/ for each run.
- Locked designs and predictions: results/*/DESIGN_LOCKED.md (written before each run; addenda marked with times).
- Cost: estimates only, from elapsed time at ~$2.50/GPU-h; saved-run elapsed total ~$4.6 plus earlier un-saved work; running estimate ~$11.3 of $18.29; user stated $6.17 remaining at 06:56:38 IST (wamid.HBgMOTE4NDUzMjIzMzc3FQIAEhgUM0JBNUQ2RUY2RUZEQzY4NEIzNEIA). Not bill-verified; idle/startup time not measured.
- Literature read depth: Utility Engineering 2502.08640 PART; Incoherent Values? 2606.21102 PART; 2605.08556 FULL; Planted in Pretraining 2507.07186 PART; Value Drifts 2510.26707 PART; 2602.18971, Fed FEDS 2026-006 abstract only; 2607.15495 + repo inspected (no patch code); multi-hop papers (2411.16353, EMNLP 2024 Hopping Too Late) abstract/search snippet only.

## Unresolved
Position vs re-binding in neutral names; why token-like names help more than alpha/beta; whether the generator's 79%->55% frozen effect (no-thinking, constrained argmax) is the same measure as Y7's 44% (different cohort and readout); no sampling (T>0) runs; no second model; no check of other tool-name wordings; no human-readable audit of the free-text reasoning in the worded-name rows.
