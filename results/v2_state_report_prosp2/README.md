# State vs report-target readout, round 2 (F6 format): gate passed, contrasts inconclusive
Design: DESIGN_LOCKED.md (committed before the run, 6a68c62). Format chosen on a separate pilot (gatepilot*/), so this is a held-out run with an adapted format, not a pristine first prospective test. 384 prompts (24 permutations x 4 clues x 4), 288 moved (state != report). 130 s in function, est ~$0.3 (public rates).

- Gate G passed: report accuracy 96.5% on moved prompts (278/288); 100% when state = report. 4 of 384 prompts have a non-candidate top-1 token.
- P1 J-lens Q-K on moved prompts: -0.050 [-0.285, +0.186]. Inconclusive (CI includes 0).
- P2 logit-lens Q-K: +0.153 [-0.044, +0.358]. Inconclusive.
- Components (moved, J-lens): K +0.132 [-0.009, +0.259], Q +0.081 [-0.093, +0.250]. Logit lens: K +0.052 [-0.066, +0.166], Q +0.205 [+0.041, +0.358]. Unrelated-index control (arbitrary (a+1) mod 4) J: -0.159 [-0.313, +0.013].
- Descriptive, not a contrast: when state = report (s=0) J-lens K=Q = +0.526, while on moved prompts both K and Q are small (+0.13, +0.08). The final-position lens signal is large when the state equals the answer and weak for both when the table moves the answer, although behavior is 96.5% correct.
- Not shown here: that the state is or is not represented; that the lens reads the report; anything causal. This says that in this format, model and band, the final-position J-lens does not separate state from report with these 24 clusters.
- Caveats: the state is the clue digit written in the prompt; K here can reflect the digit. Only the final position is a locked contrast. The unrelated-index control is arbitrary. One model, one format.
