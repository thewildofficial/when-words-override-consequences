# Bounded search: causal interventions on utility/preference directions (CPU, 07:40)
Search was a handful of queries, not exhaustive. Depth labels per paper.

1. Probing Persona-Dependent Preferences in Language Models (arXiv 2605.13339; https://arxiv.org/html/2605.13339; read depth PART: abstract, probe/steering sections, controls text, related work; appendices mostly unread).
   - Gemma-3-27B and Qwen-3.5-122B. Ridge probes on residual activations predict revealed pairwise task choices (Thurstonian utilities; 4,000 tasks fit, 1,000 select alpha, 1,000 test; 14 topics). Probe tracks shifts under persona prompts (r~0.95 on targeted tasks).
   - Causal test (Gemma-3-27B): adding the probe direction at layer 23 over each task's token span, contrastive +c/-c, moves P(choose steered task) from ~0.05 to ~0.95 across c=-0.06..+0.06 of mean activation norm (swing ~0.90; per-persona 0.81-0.96); steering one task only recovers a comparable swing. Refusals stay near baseline.
   - Ablation control: projecting out the direction leaves choices unchanged (agreement 0.98-0.99) while a random rank-1 projection shifts them (0.75-0.96); the authors read this as distributed choice computation, and note a weaker reading (rank-1 too small) would need rank-k tests. They frame the vector as an evaluative representation upstream of choice; they do not claim welfare conclusions.
   - Closest prior work they cite: Lu et al. 2025, a "Valence-Assent Axis" (Qwen2.5-14B-Instruct, PCA on stated value judgments) that causally controls choices across value, sentiment and truth tasks (not read by me).
2. Representation Engineering (Zou et al., 2310.01405; https://arxiv.org/html/2310.01405; read depth ABS+: Section 5.1 utility keyword passages only). LAT on high vs low "utility" scenarios gives a dominant first PCA component; steering with the reading vector is evaluated with a sentiment classifier on generations.
3. Utility Engineering Section 7: SFT only (already read; see ASSESSMENT_utility_objective.md).

## Assessment
- A causal steering test of a revealed-preference direction already exists (2605.13339, one model family, task-choice domain) with a random-direction control and a rank-1 ablation result. So "is a probe direction causally used for choice" is NOT open in general; I should not call it open.
- What it does not do in the parts I read: separate the direction from belief/consequence knowledge in a game with known payoffs, or tie it to a Thurstonian expected-utility structure; Utility Engineering's utilities and these steering results have not (in what I read) been joined.
- Relation to WWOC: they steer within a preference (A vs B task) readout; WWOC asks whether action follows consequence knowledge. Different question; their positive steering result does not transfer to WWOC and our J-space null (tested grid only) does not contradict it.
- Honest outcome: no clear new lead from this search. Novelty is not claimed; nothing run.
