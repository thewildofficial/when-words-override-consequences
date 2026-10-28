# H1 fine alpha grid: any valid-region J-lens edit? (review draft, scoped)

Same model, lens, prompts, edit as results/v2_jlens_klmatch. Single layers 32, 34, 40; alpha 0.5 to 1.0 in steps of 0.05; semantic swap and unrelated-category control; 33 eligible prompts (scenario, target items) per layer. Regions fixed in scripts/analyze_jlens_fine.py before looking: fraction of prompts that reach target top-1 with per-prompt KL < 1 nat at some grid alpha. >= 20% success; <= 5% failure; in between inconclusive.

Measured:
- Semantic: 1/33 prompts (3.0%) at each of the three layers reach top-1 with KL < 1 at any grid alpha. Region: failure, for this grid, model, category. Unrelated control: 0/33 at each layer.
- Ever top-1 at any grid alpha (KL unrestricted): semantic 25, 24, 24 of 33; unrelated 1 of 33 each.
- Median KL at the first flip: semantic 3.7, 4.1, 3.9 nats (smallest 0.14-0.15); unrelated 1.4-1.8.
- Per-alpha counts are in analysis.txt. Example, layer 34: alpha 0.8 gives 12/33 top-1 at median KL 0.72, but only 1 of those prompts is under 1 nat; alpha 1.0 gives 24/33 at median KL 5.1.

Reading (provisional, scoped): within this grid, model and category, the semantic swap does not reach target top-1 inside the 1-nat validity region for more than one prompt per layer. This does not show that no valid editor exists in general (other positions, layer sets, operations, directions, models and categories are untested), and the one-prompt cases are not characterized. Target-specificity relative to the unrelated control persists at high KL (semantic flips 24-25/33 vs unrelated 1/33 ever).

Cost: 358.2 s on A100-80GB, estimated about $0.29 (public Modal rates; provider bill not read).
