# Y1 design (locked before run, 2026-10-29 ~04:55 IST)
Goal: find a game decision where Qwen3.6-27B (thinking disabled, answer = next token, no CoT) is competent, so that later mechanism work has a valid substrate. Behaviour only, no lens.
Tasks (100 each, margins enforced in gen.py, option order randomised, seed 7): pot_odds_stated (equity given), outs_draw (equity must be derived from outs/unseen, then compared to pot odds), gamble (EV of sure vs lottery), blackjack (exact infinite-deck EV, hit vs stand).
Scoring: argmax of next-token logit over A/B vs gold letter. Two renderings: plain answer, and prefill "The better choice is". Gate: >=85% accuracy within a task (gate failure gates interpretation only).
Predictions (written before run): gamble >=90%; pot_odds_stated 70-90% (arithmetic); outs_draw lower than pot_odds_stated by >=10 points; blackjack 60-80% (memorised basic strategy may help on extremes only). Position-letter bias will be checked as well as accuracy.
Estimated cost about $0.2-0.4 (one A100 model load plus 800 forward passes).
