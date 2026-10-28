# Kuhn poker EV: behavior-only competence pilot (no hidden-state capture)
Qwen3.6-27B (pinned), no CoT, forced choice: compare next-token logits of "A" vs "B" after the empty think block. 480 prompts sampled (seed 1) from x1/kuhn_ev.py dataset(seed=7, n_random=40): 4 fixed + 40 random opponent policies, 4 renderings, 2 label orders. Generator verified earlier by Monte Carlo (conditioned on the observed opponent action, 0 mismatches). 97 s in function, est ~$0.1 (public rates).

Measured:
- Gate accuracy (pick == solver-best action): 52.7% overall. By rendering r0 68.5%, r1 60.7%, r2 49.2%, r3 37.0%. By opponent action: after a bet (call/fold) 77.3%; after a check (check/bet) 30.3%. By advantage: 33-43% in the mid bins, 60.6% when advantage >= 0.7. Required: >= 85% per rendering. Gate failed on every rendering.
- Public-history composition baseline: the majority-best action by (own card, observed opponent action) over the pool gets 91.5% on the same prompts. The model is far below this trivial baseline.
- Model actions: after the opponent checks, the model mostly checks (73% of 251) when betting is solver-best for 79% of those states (King: best is bet in 88/88, model bets 30/88). After a bet it folds Jack (70/84, correct) and calls Queen/King mostly.
- Unforced behavior: the model's top-1 next token is "To" in 477/480 prompts (it wants to start reasoning). So A/B logits here are a forced choice, not its natural no-CoT output; a prefilled-answer format was not tested.
Reading: in this prompt format the model is not competent at solver-best Kuhn decisions without CoT, so value-decoding or patching claims would be gated. It does not show the model lacks value representations. The composition baseline being 91.5% means any EV probe would also have to beat a (card, history) lookup. Pilot only; one prompt format, one scoring rule, one model.
Not established: whether other formats (answer prefill, fewer policy numbers, stated EV hints) change competence.
