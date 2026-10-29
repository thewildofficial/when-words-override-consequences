# State vs report-target readout, round 2 (format selected on a separate pilot; locked before this run)
Adaptation note: round 1 (results/v2_state_report_prosp) failed the competence gate. Format F6 was chosen on a separate behavior-only pilot (seed 777, gatepilot*/). This run uses seed 20261029 and is a new held-out run with the pilot-selected format, not a pristine first prospective test.
Design: state index = clue a (0-3). A per-prompt permutation table maps private index -> report index (all 24 permutations x 4 clues x 4 repeats = 384 prompts). Report index = perm[a]. Word set (6 sets) and label order random per prompt. Moved prompts: perm[a] != a (state and report differ).
Evidence/read: same as round 1 (final prompt token, band layers 36-43, J-lens and logit lens through model.unembed, candidate_evidence). Control column jA uses (a+1) mod 4, an arbitrary unrelated index, not a principled distractor.
Locked contrasts on moved prompts, 95% bootstrap CI clustered by permutation (24 clusters):
- G: report accuracy >= 85% (candidate-logit argmax). Otherwise not interpreted.
- P1: J-lens mean (Q-K). >0 CI excluding 0: report-poised; <0: state; including 0: inconclusive.
- P2: logit-lens mean (Q-K).
Exploratory only: other positions (the digit token position is where the state is written, so K there is input identity).
Not claimed: causal use, retained state. Correlational readout; cross-policy probe failure from Stage 1 (0.475/0.250, chance 0.25) still stands for any state-decode claim. Scope: this model, F6 format, this lens band. Est. $0.3, timeout 1500 s.
