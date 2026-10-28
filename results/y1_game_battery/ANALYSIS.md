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
