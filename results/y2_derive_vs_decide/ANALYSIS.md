# Y2 results (est. ~$0.3, balanced 50/50, 120 items per condition, Wilson 95% CIs; analysis.txt)
| condition | plain | prefill | notes |
|---|---|---|---|
| C1 poker, derive equity from outs | 57% [48,65] | 50% [41,59] | Fold recall 8/60 and 0/60; Call recall 60/60: always-Call |
| C2 urn framing, outs, play/pass | 77% [68,83] | 59% [50,68] | Pass recall 34/60 and 11/60; Play-biased, and very rendering-sensitive |
| C3 poker, equity stated | 78% [69,84] | 88% [80,92] | Call recall 34/60 plain: Fold-leaning plain, balanced with prefill |
| C4 derivation only: outs/unseen above or below threshold % | 84% [77,90] | 82% [75,88] | Lower recall 59/60, Higher 42/60: not at the 85% gate |
Predictions vs outcome: C3>=80 hit only with prefill; C1 near 50% and Call-heavy hit; C2 less call-bias than C1 hit in plain and not in prefill; C4>=85 missed (82-84).
Reading (provisional): a single always-action policy appears whenever the model must both derive the probability and compare it to pot odds (C1). Splitting the pieces: derivation-only is 82-84% (not at gate, 'Lower'-biased), stated-equity decision is 78-88%. So neither piece alone is at floor, and C1 is far below what the pieces predict (a naive independence estimate would be ~0.8*0.85), which points to integration or to the number of unstated steps (C1 also requires deriving B/(P+B)), not derivation of the equity alone. Framing: C2 plain beats C1 by 20 points but the prefill rendering erases it, so the framing effect is not robust across renderings. Not discriminated: whether the failure is a missing internal computation, a prior over actions, or format. Next: Y3 supplies each quantity separately.
