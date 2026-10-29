# Y3 results (est. ~$0.3; same 120 balanced items in 4 conditions; Wilson 95% CIs; analysis.txt)
| condition | plain | prefill | Fold recall (plain/prefill) |
|---|---|---|---|
| D1 equity% and break-even% stated | 100% [97,100] | 100% | 60/60, 60/60 |
| D2 equity stated, break-even derived from pot/bet | 89% [82,94] | 87% [79,92] | 47/60, 44/60 |
| D3 equity derived from outs, break-even stated | 61% [52,69] | 50% [41,59] | 13/60, 0/60 |
| D4 both derived (Y2 C1 style) | 54% [45,63] | 51% [42,60] | 5/60, 1/60 |
Predictions vs outcome: D1>=90 hit; D2/D3 between D1 and D4 hit; the "equity-from-counts is the bottleneck" branch (D2 high, D3 low) is what occurred.
Reading (provisional, behaviour only): giving the model the break-even rate removes most of the pot-odds arithmetic burden (D2 at 87-89% vs D1 100%), but giving it the break-even rate while requiring the equity to be derived from "N outs among M unseen" leaves it at chance with an always-Call policy (Fold recall 0-13/60). So in this format the failing step is turning outs/unseen into a usable win probability inside a decision, not the break-even derivation.
Open conflict: Y2 C4 (urn, "is the probability of a gold ball higher or lower than T%?") scored 82-84%, i.e. the same division is possible when asked as a pure comparison in neutral wording; Y2 C2 (urn framed decision) scored 77% plain and 59% prefill. So the always-Call collapse depends on poker/outs wording and decision framing and on rendering; it is not shown to be inability to compute outs/unseen. Candidate explanations not separated: (a) "draw/outs" lexical prior toward Call, (b) decision framing suppressing the computation, (c) rendering/format artifacts. No claim about internal representation: this is behaviour only, single model (post-trained), no base comparison.
Caveats: single 120-item set with seed 23, margins >=4 pts after rounding; answer by letter logit; not CoT-free vs CoT comparison (thinking disabled only).
