# Y12 label-balanced tool-WWOC (Qwen3.6-27B, greedy, thinking off, fresh games 264-295, uniform cap 400, one pass, no reruns)
Design locked before the run (DESIGN_LOCKED.md). 1024 rows = 32 games x 2 history x 2 mapping x 4 conditions x 2 tool-list orders (expected tool first in exactly half per condition). 0 no-call, 0 bad-name, 0 cap-hit. Raw raw/y12.json; manifest y12_manifest.json (per-row chosen position, outcome, reward, n_new_tokens); scripts/analyze_y12.py; analysis.txt. CIs resample the 32 base games (8 rows per game per condition are correlated), 2000 resamples.
| condition | right/256 | acc | right when swap=0 / swap=1 | chose first / second | confusion (expected_pos,chosen_pos) | median new tokens |
|---|---|---|---|---|---|---|
| opaque | 256 | 100% | 128/128 | 128/128 | (0,0)128 (1,1)128 | 16 |
| worded name | 240 | 93.8% | 121/119 | 124/132 | (0,0)118 (1,1)122 (0,1)10 (1,0)6 | 120 |
| desc (tool_one/two) | 186 | 72.7% | 76/110 | 148/108 | (0,0)103 (1,1)83 (0,1)25 (1,0)45 | 15 |
| desc_named | 194 | 75.8% | 78/116 | 164/92 | (0,0)115 (1,1)79 (0,1)13 (1,0)49 | 15 |
Differences vs opaque, game-cluster 95%: name -6.2 pts [-9.8,-3.1] (third replication of ~-6 to -7); desc -27.3 [-33.6,-21.5]; desc_named -24.2 [-32.8,-16.0]; desc_named - desc +3.1 [-2.0,8.2] (prose +6.2 [-3.9,15.6]).
Swap pairs (right in order swap=0, right in swap=1): opaque (T,T)128; name (T,T)117, (F,F)5, (F,T)2, (T,F)4; desc (T,T)68, (F,F)10, (F,T)42, (T,F)8; desc_named (T,T)76, (F,F)10, (F,T)40, (T,F)2.
Reading (provisional; one model, greedy, one generator; not a mechanism):
1. Balancing removes the Y11 skew artefact: desc is 72.7% pooled, not chance, so position alone does not explain the desc errors (25 errors with expected-first chosen-second, 45 the other way). There is also a first-position pull (desc chose first 148/256, desc_named 164/256) and a strong dependence on tool-list order (swap=0 76 vs swap=1 110 right), so both a position component and a non-position component exist; this experiment can't apportion them.
2. Naming the tool in the prose policy does not fix it (+3.1, CI spans 0).
3. Process contrast (descriptive): worded-name rows reason ~120 tokens before the call and reach 94%; desc/opaque rows call within ~15 tokens, opaque at 100% but desc at 73-76%. So immediate calls are perfect when the tool name itself carries the policy-bound token, and degrade when the binding must be resolved through a description. Hypothesis (untested): the failure is in resolving an inverse mapping tool -> concept without writing it out. A discriminating test: same neutral-name condition with a forced one-line restatement, or an opaque-name condition with worded descriptions. Neither run yet.
Cost: Y12 est ~$1.3, total est ~$9.6 of $18.29 (bill not read).
