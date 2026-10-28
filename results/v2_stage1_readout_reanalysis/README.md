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
