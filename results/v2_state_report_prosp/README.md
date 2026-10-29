# State vs report-target readout (prospective): GATE FAILED, readout not interpreted

Design: DESIGN_LOCKED.md (committed before the run, c72c240). Run: modal_state_report.py, 384 prompts, A100-80GB, 173 s in function, est ~$0.3 (public rates, bill not read). Raw: raw/state_report_raw.json. Analysis: analysis.txt (scripts/analyze_state_report.py).

## Result
- Gate G failed: report accuracy on s != 0 prompts is 25.3% (n=288; chance 25%). Per s: 0.240 / 0.260 / 0.260. At s=0 (state = report) accuracy is 31.2%. Required: >= 85%.
- Post hoc behavior diagnostic (not pre-specified): the picked index equals the stated offset s in 52.6% of all prompts, versus 25% by chance. The picked index distribution is 2:145, 1:131, 3:88, 0:20. So in this prompt format the model mostly does not compute the clue-derived index without CoT; it often picks the label at index s.
- Per the locked rule, P1 and P2 are reported but not interpreted: J-lens Q-K -0.036 [-0.293, +0.211]; logit-lens Q-K +0.378 [+0.097, +0.675] (all s != 0). On the 73 gate-correct prompts the signs change (J-lens +0.351 [-0.179, +0.833], logit-lens -0.006), and these are the model's chance-level hits, so no reading is justified. The distractor-control J evidence is near 0 (-0.012).
- A competence gate failure gates interpretation. It does not show that the model lacks a latent state representation. It shows that this task format is too hard no-CoT for this model, so there is no reliably held state to read.

## Consequence
The next prospective version needs a task the model performs at >= 85% no-CoT while the state is still not written at the read position. Candidates (untested): one-step state (a single stated clue maps to the state via a visible lookup the model already follows, as in Stage 1) with the offset applied by an explicit code table that is varied independently across prompts, or running a gate pilot first on CPU-free logits only (cheap) to choose the format. This run counts toward spend.
