# Y7 / Y8 / Y9: WWOC (V5-RBG-4 contrarian game, fresh base games 200-231, opposed incentive) as tool use, Qwen3.6-27B, greedy, thinking template off
Design: DESIGN_LOCKED.md (locked before runs; addenda for the 80-token truncation and the discarded Y8b). Games: 32 base games x {none,redundant} x {prose,table} = 128 rows per condition (4 cells per game are correlated; CIs below resample GAMES). Tool schema = model's own chat-template tools format (read from the HF chat template as format data only).

## Y7 forced choice (format-transfer evidence only; not execution)
A/B letter: ab_opaque 128/128; ab_assertion 56/128=44% [35,52]. Tool-name log-prob after the tool-call prefill: tool_opaque 128/128; tool_assertion_name 89/128=70% [61,77]; tool_assertion_desc 84/128=66% [57,73]. Baselines: always-A 50%; always-shorter-name 40/76=53% (weak, global; does not exclude token-likelihood or name-semantics confounds).

## Y8 native generation + mock executor (parse <function=NAME>, apply the stated contrarian policy, reward 10 if target response reached)
Policy "cap80": uniform 80 new tokens, all 384 rows. The model often writes reasoning prose before the call, so many generations were cut before any call. cap80: opaque right 126, nocall 2; name right 48, wrong 7, nocall 73; desc right 68, wrong 37, nocall 23 (of 128 each).
Policy "cap80 then 400 on unparsed" (Y9, post hoc, NOT a uniform 400 cap: only the 92 no-call rows were regenerated; same prompt/template/revision/greedy; Y9 rows now run to 1674 chars and end with <|im_end|>, 92/92 parsed, which shows the new build ran; Y8b was discarded because a warm container ran old code: only 6/98 parsed, max ~80 tokens). Merged, all 384 eligible: 
- tool_opaque 128/128 right.
- tool_assertion_name 118/128=92%: wrong 8, nocall 2. Right among called: 118/126=94%.
- tool_assertion_desc 87/128=68%: wrong 37, nocall 4. Right among called: 87/124=70%.
Game-cluster bootstrap (2000 resamples, 95%) for accuracy minus opaque: name -7.8 pts [-11.7,-3.9]; desc -32.0 pts [-39.8,-24.2]. (cap80 policy: name -60.9, desc -45.3; the large cap80 gap is truncation, not wrong choices.)
Cells (merged, k/32): name none/prose 29, none/table 31, redundant/prose 30, redundant/table 28; desc 19, 20, 23, 25; opaque 32 in all cells.

## Reading (provisional; one model, one generator, 32 games)
1. The A/B letter effect is replicated on fresh games (44% vs 100%, forced choice), but in executed tool calls with worded tool NAMES the effect is small (92% vs 100%) and mostly gone in prose/redundant cells: the Y7 forced-choice name log-prob (70%) overstated the executed gap. Format transfer is partial.
2. The description-only condition has a larger executed gap (68%, 37 wrong valid calls). CONFOUND: with neutral names (action_a/b) the prose policy text names the concept surface, not the tool, so the model must map action_a/b to a concept via the description; this is a lookup load, not clean wording interference. Do not read desc vs name as "description matters more". A table with the tool name next to the surface was used where mapping format is table; prose rows reference surfaces only.
3. Nothing here says which internal quantity produces the residual gap; no mechanism claim.
Not done: all-rows uniform 400 cap (cost); a desc condition that names the tool in the prose policy; sampling at T>0; other models.
