# Kuhn EV gate pilot, format variants (same 480 rows as gate_pilot; behavior only)
Script modal_kuhn_pilot2.py, analysis scripts/analyze_kuhn_pilot2.py. 215 s in the successful function run, plus one failed launch (prefill tokenization error, ~load time). Est total ~$0.5 (public rates, bill not read).

Template check: tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False) is identical to the manual string used in the first pilot (ends with the empty think block). So the first pilot's "To" top-1 was not a template mismatch: with the original wording "Return only A or B." the model's unforced next token is "To" (477/480).

Gate accuracy (forced A/B logits, 480 prompts; required 85% per rendering):
- V0 original wording, official template: 0.527 (r0 .685, r1 .607, r2 .492, r3 .370). Top-1 is "To" in 477/480.
- V1 explicit: "Reply with only the single letter A or B and nothing else. Do not reason or explain.": 0.715 (r0 .883, r1 .766, r2 .726, r3 .529). Top-1 is A or B in 480/480. By opponent action: bet 0.856, check 0.586.
- V2 V1 + assistant prefill "The better option is": 0.746 (r0 .748, r1 .813, r2 .726, r3 .710). Top-1 is A/B only 34% of the time ("to" still leads, 309).
Part of the first failure was prompt wording (+19 points from one instruction sentence). No variant reaches 85% on any rendering except r0 at V1 (0.883), and r0 is one rendering of four, so it does not pass the per-rendering gate.

Policy sensitivity (descriptive, V1): within cell (own card, observed opponent action), the model's action barely varies with the opponent policy: Jack after check: check 77/85 while the solver-best is bet in 68/85; King after check: bet 81/88 (best bet 88/88); Jack after bet: fold 80/84. The cell-level lookup (card, history) explains most of its behavior. The pooled majority-best baseline (91.5%) is an oracle-style descriptive number computed over this same pool, not a fit on train and test.

Stop rule applied: Kuhn EV is marked gated for this model and these formats (best 0.746 overall). I stop tuning formats. Not shown: that the model lacks policy-dependent value representations; that a CoT-free probe could not find them; that other formats pass. Novelty relative to the closest prior art (2605.00226, 2609.07478) is still unestablished.
