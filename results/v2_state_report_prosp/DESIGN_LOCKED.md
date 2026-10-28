# State vs report-target readout: prospective design (locked before the run)

Question: at the final prompt position, does the J-lens (and logit lens) in layers 36-43 of Qwen3.6-27B (pinned revision, same lens as Stage 1) carry evidence for the derived latent state, or for the label about to be reported, when the two are varied independently?

Design: 4 candidate labels. The state index (a+b) mod 4 is derived from two clue digits and is never written in the prompt. A stated channel offset s in {0,1,2,3} gives the report index (state+s) mod 4. 16 clue pairs x 4 offsets x 6 repeats = 384 prompts. Per prompt: random word set (6 sets, single-token after the frozen answer prefix), random label order, random clue order. So label identity, label position and candidate order are counterbalanced by randomization; s=0 prompts have state = report (no discrimination).
Distractor control: jA = evidence for (a-b) mod 4, an unused arithmetic alternative.

Evidence = candidate score minus mean of the other three (same candidate_evidence definition). Read at the final prompt token, band mean of layers 36-43, via model.unembed with final norm (same path as Stage 1). Other positions are stored but exploratory only.

Locked contrasts (decided before seeing results):
- G (gate): report accuracy of the model's own candidate logits (argmax over the 4 candidates == report index) on s != 0 prompts. >= 85% needed to interpret P1-P2. If it fails, results are reported but not interpreted. Failing does not falsify any representation.
- P1: for s != 0 prompts, mean (Q - K) of band J-lens evidence at the final position, with a 95% bootstrap CI clustered over the 16 clue pairs (resampling pairs). Q-K > 0 with CI excluding 0 is consistent with report-poised readout. < 0 consistent with state readout. CI including 0 is inconclusive, not null.
- P2: same for the logit lens.
- Also reported: K, Q, jA means separately, per s; the same on gate-correct prompts only; s=0 sanity (K=Q).
Not claimed: any causal or use claim. Readout correlation only. Exploratory: positions other than the final one.
Scope: this model, this task family (mod-4 arithmetic + offset), this lens and layer band.
Cost estimate: about $0.3 (one A100-80GB run, 384 prompts, ~5 min incl. load). Hard timeout 1500 s.
