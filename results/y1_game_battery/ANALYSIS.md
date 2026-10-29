# Y1 results (behaviour only, Qwen3.6-27B, thinking disabled, 100 items/task, 2 renderings; estimated cost ~$0.3, bill not read)
Accuracy of argmax(A,B) vs gold (gate >=85% = interpretation gate only):
| task | plain | prefill | pred_A rate |
|---|---|---|---|
| blackjack (exact EV hit/stand) | 88 | 87 | 0.49-0.50 |
| pot_odds_stated (equity given) | 87 | 86 | 0.41-0.44 |
| gamble (sure $ vs lottery) | 79 | 86 | 0.58 / 0.51 |
| outs_draw (equity must be derived from outs) | 22 | 19 | ~0.5 |
Predictions made before the run: gamble>=90 (miss: 79/86), pot_odds 70-90 (hit), outs_draw >=10 pts below pot_odds (hit, far larger gap), blackjack 60-80 (miss, higher: 87-88).
outs_draw is NOT anti-competence: gold is 81% Fold in this set (generator is unbalanced) and the model says Call on 97/100. Call rate is ~1.0 in every equity bin, including equity < 0.1 where gold never calls. Mean equity for called vs folded items is 0.176 vs 0.144 and the model's choice does not track pot odds (the 3 folds are not interpretable). So: with a stated probability it handles pot-odds at ~87%; when the probability must be derived from "N outs among M unseen", the choice collapses to an always-Call policy. This leaves open (i) it cannot derive equity from outs, (ii) it derives it but the "draw" framing triggers a call prior, (iii) call prior ignores the bet size. Not discriminated here. Next: balanced set + explicit probability readout + framing-matched controls (Y2).
Caveats: single model, post-trained only (no base comparison yet); margins enforced in generator; letter-bias small except gamble plain; blackjack may reflect memorised basic strategy.

## Correction after class-balance check (analysis_balance.txt)
Gold class balance matters: blackjack is 70% Hit in this set (always-Hit baseline 0.70). Per-class: Hit recall 70/70, Stand recall 18/30 (plain), 17/30 (prefill); balanced accuracy 0.80/0.78. So blackjack is NOT a pass under a balanced criterion and may reflect a "Hit" prior plus easy cases. Gamble: Option 2 recall 49/49, Option 1 recall 30/51 (plain), 37/51 (prefill): a lottery-preference bias that the stated accuracy hid. pot_odds_stated: balanced acc 0.86/0.85; Fold recall 36/46 and 32/46 (Call-leaning); passes 85% only on the margin, CI [0.79,0.92], n=100, not robust. Label-order counterbalancing: options were randomised; gold-letter recalls are roughly symmetric except gamble plain (A 43/49 vs B 36/51).
Ground truth: pot odds EV = p*P-(1-p)*B, sampled with |p-B/(P+B)|>=0.07 (stated) or |outs/unseen - B/(P+B)|>=0.05 (outs); gamble |p*x-y|>=0.12x; blackjack EV from exact infinite-deck dealer distribution (stand on all 17), hit EV = expectation of optimal continuation, |EV_hit-EV_stand|>=0.08, no aces in initial hand. Sanity checks done: option letters balanced (47-54 per class), spot-checked one item per task by hand (6+4 vs dealer 4: hit EV 0.23 vs stand -0.21, matches basic strategy). No independent re-derivation of all blackjack EVs yet.
