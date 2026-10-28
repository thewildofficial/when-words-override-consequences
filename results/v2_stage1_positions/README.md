# Stage 1 per-position J-lens readout (exploratory, post hoc)

Status: exploratory. Run on the already-opened Stage 1 locked rows (480 prompts, layers 36-43), so it is not a prospective test. Script: modal_stage1_positions.py (deployed app, spawn via run_positions_spawn.py), analysis: scripts/analyze_stage1_positions.py, output: analysis.txt, raw/positions_raw.json.

Method: same forward pass, lens, normalization (model.unembed incl. final norm) and K/Q/M definitions as modal_stage1.py, but evidence is computed at every prompt position, not only the final one. Offsets are counted back from the final prompt token (0 = final).

Sanity: the final-position band K reproduces stored band_K, max abs difference 0.057, mean 0.016 (batch/bf16 numeric noise, not exact equality). GPU run 137 s inside the function, estimated cost about $0.25 incl. load (public rates, bill not read).

Measured (band mean J-lens, 120 prompts per cell):
- Final position (0), policy M: report/transformed candidate M=Q is +0.551 (1A) and +0.263 (1B); K is -0.108 and -0.017. Same as the earlier reanalysis.
- Offset 4 (the token(s) just before the final "Answer:" region of the template) has a larger report-target signal: 1A-M Q=M +0.850; 1B-M +0.424. Under policy T the same offset gives K=Q +0.739 (1A), +0.229 (1B). Offsets 6 and 7 in policy M carry positive K (+0.146/+0.187 in 1A, +0.174/+0.214 in 1B) with slightly negative Q.
- Policy M, 1A: the largest K appears at offsets 51-60 (K about +1.1 to +1.3), far from the end.

Cautions (not resolved here):
- I did not map offsets to prompt tokens. The large K at far offsets and at offsets 6-7 may be the input token that states the private label or channel (copying/identity readout), not a retained latent state. Telling the two apart needs token-level alignment and a prompt where the state is never written at the position read.
- Not counterbalanced for position or label mapping, and the lens was trained at final-token-like positions. Positional validity of the lens at other tokens is unestablished.
- Correlational readout only. No causal or use claim. Cross-policy probe transfer in 1A (0.475 / 0.250, chance 0.25) still accompanies any claim that the state is decodable.
- Post hoc, one model, three families per substage.

Consequence for design: the retained-state vs report question needs a prospective run where the state is a hidden/derived quantity that never appears as a token, with report target varied independently and candidate order and label mapping counterbalanced. This result does not replace that run.
