# Stage 1 locked readouts, reanalysed by report policy (CPU only, no new compute; review draft)

Source: results/v2_stage1/raw/mechanistic_locked.json (480 rows: substage 1A/1B x policy T or M, 120 each; stored per-layer J-lens and logit-lens K/M scores, layers 36-43, final pre-output position). Script: scripts/reanalyze_stage1_readout.py; full table: output.txt. K = true-state evidence and M = report-target evidence as stored by the original analysis (definitions in docs/v2/stage1-analysis-plan.md; not re-derived here). In policy T the report target equals the state, so Q equals K there (band Q 0.387 = K in 1A-T); in policy M the report target is a transformed function of the state.

Measured (band mean over layers 36-43):
| substage, policy | J-lens K | J-lens M | output K |
|---|---|---|---|
| 1A, T (report = state) | +0.387 | -0.121 | +11.4 |
| 1A, M (report = transformed) | -0.108 | +0.551 | -1.66 |
| 1B, T | +0.069 | -0.012 | +10.0 |
| 1B, M | -0.017 | +0.263 | -0.62 |

- Layer-wise (1A, M): K is slightly positive at layers 36-38 (0.03-0.06) and negative at layers 39-43 (-0.13 to -0.26), where M rises to 0.68-0.85. In 1A, T, K is 0.50-0.55 at layers 39-43.
- The logit lens shows the same sign pattern with smaller magnitude (1A, M, layers 39-42: K -0.04 to -0.10, M 0.26-0.29).
- By family, 1A, M band K is negative in all three (-0.08 to -0.15); in 1A, T it is positive in all three (0.33-0.44).

Interpretation (hypothesis, not tested here): the pooled "J-space score is negative while a residual probe decodes state at 1.000" result is what you would see if the lens reads content poised to be reported (the report target) rather than the latent state itself. In policy T the two coincide, so positive K there is not independent evidence of state reading. The J-space paper defines the J-space as what the model is poised to verbalize, so this reading is consistent with the lens working as defined and with the "miscalibration" being a mismatch between what was expected of it and what it reads. Caveats: stored K/M definitions not re-derived; one model, three families per substage; the residual probes were trained within the pooled policies, and the locked cross-policy transfer was weak in one direction (docs: truth-to-transformed 0.475 and transformed-to-truth 0.250 in 1A, chance 0.25), so "state is decodable" is shown mainly within the trained policies. Decodability is not causal use.

Untested follow-up that would discriminate this reading: a policy-matched control where the report target is varied independently of the state and the lens is read at the report-planning position, then an edit along each coordinate; needs a new GPU run.

## Provenance and validity checks (added after review request)
- Definitions (src/jspace_policy/stage1_analysis.py::candidate_evidence): evidence for a candidate is its score minus the mean of the other three candidate scores; K uses the true-state candidate, Q the expected-report candidate, M the transformed-state candidate, D = K - Q. Four candidates, chance structure 0.25.
- Normalization: modal_stage1.py (lines ~550-551) scores J-lens and logit lens through model.unembed(...) of the pinned jlens adapter, i.e. with the final norm. Audit A1 (docs/next-sprint/audit.md) concerns the separate candidate-row path in modal_v5_mechanistic_decomposition.py, which multiplied by lm_head weights directly. The Stage 1 locked run (created 2026-08-17, git 8ecf0e3, lens commit 581d398) is not on that path, so A1 does not apply to these scores by code reading. I have not re-run the readout to confirm numerically.
- Consistency with the original analysis: my policy means reproduce the published paired quantities. Report preparation (M under policy M minus M under policy T): 0.551 - (-0.121) = 0.672 (report: 0.673 in 1A). State change (K under M minus K under T): -0.108 - 0.387 = -0.495 (report: -0.495). So this reanalysis restates the original policy decomposition by policy rather than adding a new estimate; the new element is the layer-wise table and the Q = K coincidence under policy T.
- Position: scores are at the final pre-output prompt position. That this position is where the report is planned is not established here.
- Label mapping and position counterbalancing in the original dataset were not re-audited in this reanalysis. The proposed follow-up must counterbalance candidate order, label mapping and positions itself.
