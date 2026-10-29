# Y6 native generation (Qwen3.6-27B rev 6a9e13bd, greedy, thinking template off), 240 items = first 60 of each Y3 task, per-item option order from Y3
Design in DESIGN_LOCKED.md (locked before run). Raw: raw/y6.json; parser scripts/analyze_y6.py; numbers analysis_raw.txt. Wilson 95% CIs. Parse: free_word = first word; cot = last 'Final answer: X'.
| task | free_word | Fold/Call recall | cot (400-token cap) |
|---|---|---|---|
| D1 both stated | 60/60 [94,100] | 32/32, 28/28 | 57/60 [86,98] (3 unparsed) |
| D2 equity stated, threshold derived | 54/60=90% [80,95] | 32/32, 22/28 | 1/60: 59 UNPARSED = truncated at 400 tokens, uninterpretable |
| D3 equity derived from outs, threshold stated | 49/60=82% [70,89] | 21/32, 28/28 | 60/60=100% [94,100] |
| D4 both derived | 41/60=68% [56,79] | 13/32, 28/28 | 0/60: 60 UNPARSED = truncated, uninterpretable |
Reading (provisional, one model, 60 items/task, one parse rule): free generation does better than Y3's A/B log-probs on D2-D4 (Y3: D3 61/50%, D4 54/51%) but keeps the Call-heavy error pattern on D3/D4 (Fold recall 21/32 and 13/32, Call recall 28/28). So the A/B letter readout understated native performance, yet the Fold-recall deficit remains. With step-by-step text, D3 reaches 60/60, which fits a no-CoT limit for equity-from-outs, but D2/D4 CoT were cut off by my 400-token cap, so the CoT prediction is only testable on D1/D3. Y10 re-runs D2/D4 CoT at 1000 tokens (post hoc cap change, same prompt). Not a utility finding; says nothing about internal representation.
