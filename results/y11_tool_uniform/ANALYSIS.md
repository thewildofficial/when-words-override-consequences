# Y11 uniform-cap confirmatory tool-WWOC (Qwen3.6-27B, greedy, thinking off, fresh games 232-263)
Design locked in DESIGN_LOCKED.md before the run. One pass, cap 400 for EVERY row (output records cap=400), no reruns, 512 rows, 0 rows hit the cap without a call (0 no-call, 0 bad-name in any condition). Raw raw/y11.json; per-row manifest y11_manifest.json; scripts/analyze_y11.py; analysis.txt; label_baselines.txt. Mock executor: first <function=NAME> parsed, contrarian policy applied, reward 10 if target reached.
| condition | right/128 | wrong | cells k/32 (none/prose, none/table, red/prose, red/table) | median new tokens |
|---|---|---|---|---|
| tool_opaque | 128 (100%) | 0 | 32,32,32,32 | 16 |
| tool_assertion_name | 119 (93%) | 9 | 29,32,31,27 | 117 |
| tool_assertion_desc | 89 (69.5%) | 39 | 24,14,28,23 | 15 |
| tool_desc_named (tool named in prose policy) | 86 (67.2%) | 42 | 26,14,23,23 | 15 |
Game-cluster bootstrap 95% (2000 resamples over 32 games): name - opaque -7.0 pts [-10.9,-3.9]; desc - opaque -30.5 [-38.3,-22.7]; desc_named - opaque -32.8 [-45.3,-21.1]; desc_named - desc -2.3 [-9.4,5.5] (prose -4.7 [-18.8,10.9]; table identical rows by construction).
Predictions vs outcome: opaque >=95%: met. Name gap within -12..0: met (-7.0), and it replicates the Y8 corrected gap (-6.3) on fresh games with a uniform cap. desc_named >= desc +10 in prose (lookup-confound hypothesis): NOT supported (-4.7, CI spans 0); the Y8 description-gap is not explained by the missing tool name in the prose policy.
Label-skew warning (label_baselines.txt): the generator makes the correct tool 'A' in only 40/128 rows. Always-B would score 88/128 = 69%, equal to the desc accuracy. Choice counts: desc chose A 63, B 65; desc_named chose A 82, B 46 (expected-B rows: 46/88 right); name chose A 47, B 81; opaque 40/88 exactly right. So the neutral-name rows are NOT shown to differ from a chance-level positional policy by this accuracy; desc/desc_named errors look position-driven and the A/B name semantic (action_a vs action_b) is a confound. Interpretation of the desc conditions is blocked until a label-balanced rerun; it is not a WWOC finding.
Process observation (descriptive): the worded-name condition generates ~117 median tokens (reasons first, then calls); desc/opaque call within ~15 tokens. Not a causal claim.
Provisional, one model, one generator, greedy only: with exact stated consequences, worded tool names cost ~6-7 points of executed reward vs opaque names (93-94%), far less than the A/B letter forced-choice gap (44% vs 100%). No mechanism, no universal claim.
