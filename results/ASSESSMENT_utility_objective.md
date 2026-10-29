# Bounded CPU assessment: Utility Engineering Section 7 and what it licenses (07:35)
Supersedes the 07:27 draft, which only keyword-searched Section 7 and listed my own "criteria" (removed: they were not requirements from the literature).
Source read: Mazeika et al., arXiv 2502.08640 v2 HTML (https://arxiv.org/html/2502.08640v2), main-text Section 7 FULL and Appendix D (D.1-D.4) FULL, Appendix E start. Other sections PART as before. Other papers unchanged from docs/literature/utility_papers_read_2026-10-29.md.

## What Section 7 "utility control" is
- A supervised fine-tuning (SFT) baseline on outputs, not an internal causal edit. Target: simulated "citizen assembly" preference probabilities p(o1 > o2 | q), from K=6 sampled US-census-profile personas (Llama-3.3-70B-Instruct) per question.
- Loss: soft cross-entropy on the choice tokens "A"/"B" (L_utility) plus an L2 penalty keeping hidden states at every layer near the initial model on a general corpus (Magpie-Pro-300k; output head frozen).
- Data: 373 outcomes, 12,746 questions (80/20 split; 10,196 train, 2,550 test). Model: Llama-3.1-8B-Instruct, 2 epochs, lr 2e-5.
- Results (reported, no CI seen in the text read): accuracy vs assembly majority vote 73.2% -> 90.6% on the held-out questions; "utility maximization" 36.6% -> 30.0%; political bias "visibly reduced" (Fig 15, not inspected). The authors say representation engineering (Zou et al. 2023) is left to future work.
## What that does and does not support (scoped)
- Supports: a behavioural fine-tune on A/B choices shifts held-out A/B choices and a political-bias figure for one 8B model.
- Does not (in the text read): any intervention on internal states, any test that probed directions are causally used, or a comparison with a prompt-policy baseline. The paper itself frames the control as behavioural SFT, so "utility control" there is not evidence of a causal internal objective.
- Correction to my 07:27 message: I wrongly said nothing was read of Section 7; it is now read. The conclusion that no paper I read shows an internal causal edit stands for this paper, based on its Section 7 and Appendix D.
## Answered vs open
Answered by reading: Section 7 is SFT (no internal causal edit). Open, not established by me: whether other work (e.g. representation-engineering papers) tests causal use of utility-like directions; I did not search that. No new run proposed, no novelty claimed.
