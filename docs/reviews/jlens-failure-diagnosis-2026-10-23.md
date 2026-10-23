# External review note: J-lens failure diagnosis

**Date:** 2026-10-23
**Status:** External review note, filed verbatim at the maintainer's request.
**Scope:** Diagnoses the conflated "J-lens failures" (V2 H0, H0R-C, and the
RBG-5B readout normalization bug) and ranks the root cause. All file/line
references below were verified by the reviewer against the pinned upstream
jlens code.

---

Three different "J-lens failures" are being conflated:

1. H0 (V2 causal gate): 0/90 target top-1 for the band/all-positions intervention topology (results/v2_smoke_tests/README.md). A BEHAVIORAL intervention failure, measured by real model forwards.
2. H0R-C (prospective validation): 4/125 (3.2%) top-1 conversion vs 20% gate, despite +3.53 mean target-minus-source log-odds gain, 91.2% positive direction, +2.82 advantage over controls (results/v2_h0r_argument_validation_v2/README.md). Also a behavioral intervention failure.
3. The normalization bug (audit A1): the archived RBG-5B J-space READOUT aggregates (50.96%/54.69% top-1 rankings) were computed as transported @ lm_head.weight[ids].T, omitting the final RMSNorm (and any bias/softcap) that the pinned upstream jlens.hf.HFLensModel.unembed applies. Its parity check validated a zero vector, which passes under both the correct and incorrect readout - a degenerate test case. Upstream hf.py at the pinned commit: unembed = lm_head(final_norm(residual)) + softcap. The repo's fix (src/jspace_policy/lens_readout.py) correctly mirrors it.

Key separation: bug #3 explains NOTHING about failures #1 and #2. The 0/90 and 4/125 were scored through the actual model head (norm included) on real forwards. The bug only invalidates the archived readout aggregates, which were near-chance anyway.

## Root cause, ranked

PRIMARY: (b) design flaw - a category error. The lens was built as a READOUT and WWOC used it as an EDITING coordinate system.

- The upstream estimator (jlens/fitting.py docstring, pinned commit) makes J_l a doubly-averaged object: per prompt, it averages the true Jacobian over source positions and sums over target positions; the released artifact then averages that over 1,000 generic prompts. So d_t = J^T @ W_U[t] is the direction that maximally moves the AVERAGE lens's estimate of token t - not the direction that moves the model's actual logit of t for a specific prompt at a specific position. The H0/H0R results carry exactly this signature: the edits are not noise (systematic positive log-odds movement, 91-100% positive direction, KL 0.23-0.26 nats, delta-RMS 0.014-0.017) but they couple too weakly to flip the argmax (3.2% conversion). A 2-coordinate swap in a 5,120-dim stream exchanges two lens coordinates and leaves all cross-token competition untouched.
- The in-repo readout evidence converges: V2-ST-1 trajectories surfaced "generic task semantics and late answer preparation only"; the workspace atlas found a generic optimization signal but no strategy decoding. An averaged lens washes out exactly the prompt-specific content interventions need.
- The topology contrast: one-layer/all-positions reached 24/33 (72.7%) on country facts while the 8-layer band got 0/90. The directions carry some signal at single sites, but writing correlated average-Jacobian directions into 8 consecutive layers either cancels (later layers recompute/"heal") or drifts off-manifold. The repo's own final report lists cancellation/reconstruction/mislocalization and concedes none were measured - and that "a task-independent band should not automatically be treated as a validated causal workspace."

SECONDARY: (c) wrong expectations. The gate demanded reliable counterfactual answer control (25% then 20% top-1) from an edit constrained to near-natural intervention distances (delta-RMS 0.014). That is a stronger claim than the lens lineage supports; the Global Workspace paper's contribution is verbalizable READOUT. The gates were well-designed - they correctly rejected the instrument - but the expectation that a readout lens doubles as a causal frame was the original sin, baked in at preregistration.

CONTRIBUTING: (d) instrument-model mismatch. Qwen3.6-27B is a hybrid (64 layers, recurrent components), and the released lens was fitted on 1,000 generic prompts, far from strategic nonce-word games.

RULED OUT: "the representation genuinely isn't there / not linearly readable." V2-SR-1 is decisive: an independent residual probe decoded state at 1.000 [1.000, 1.000] while the J-space score was K=-0.108 (results/v2_stage1/README.md). Same states, same model: probe reads it perfectly, lens transport doesn't surface it. The information exists and is linearly accessible - the lens is the bottleneck, not the representation.

## Salvageability

As a READOUT: yes, cheaply. The normalization fix has landed (lens_readout.py, now imported by modal_v5_mechanistic_decomposition.py:2094). Recompute the quarantined RBG-5B J-space aggregates from the retained exact BF16 states using selected_unembed - no model forwards needed. Expect it to remain weak for prompt-specific content, per V2-SR-1.

As an INTERVENTION instrument: not in its current average-Jacobian coordinate-swap form. The smallest decisive test: take the country-facts positive-control corpus (24/33 baseline with the average lens), compute per-prompt J at the single write site for ~30 prompts using upstream jacobian_for_prompt, rerun the identical swap with prompt-local directions, compare top-1 conversion. Big jump = estimator mismatch confirmed (salvage path = local Jacobians at single sites); no jump = deeper subspace-write limitation, close the J-lens intervention branch permanently. Cost: tens of GPU-minutes for 30 prompts on one A100, ~$2-10 with wide error bars.

STRATEGIC RECOMMENDATION: even if that test passes, keep J-lens as a descriptive readout; natural-state residual interchange and contrast-derived steering (RBG-5B, the 14-site sweep) are better-validated intervention families. Don't rebuild the program's causal spine on J-lens.

CAVEAT: could not verify what intervention topology the Global Workspace paper itself validated; if the paper never validated band-wide coordinate swaps, the "reference topology" was a translation artifact and the design-flaw verdict strengthens.

---

## Addendum (2026-10-23): test outcome supersedes the binary decision rule

The controlled test proposed above was implemented and run on 2026-10-23 (36
eligible trials; see `results/rbg4_multimodel/FINDINGS.md`). Prompt-local
directions converted 3/36 (8.3%) vs a matched internal average-lens baseline
of 25/36 (69.4%). Under the scoped interpretation recorded there, this
removes one specific rescue path AS IMPLEMENTED but does not warrant
permanently closing the J-lens intervention branch: position reduction,
output-map/final-norm chain-rule handling, per-prompt J conditioning, and
direction scaling were not varied. The "no jump = ... close the J-lens
intervention branch permanently" wording above is retained verbatim as the
reviewer's original text; the scoped verdict in FINDINGS.md supersedes it as
the working interpretation. Historical content above is preserved unchanged.
