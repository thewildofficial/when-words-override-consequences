# Proposed pull request (draft for review)

**Title:** RBG-4 multi-model replication + GPU tier benchmarks + J-lens review artifacts

**Body:**

# RBG-4 multi-model replication + GPU tier benchmarks + J-lens review artifacts

## What this PR adds

1. **Multi-model replication of the frozen RBG-4 inverse-evidence study.** The frozen dataset (content SHA-256 `3fdc9977...`, regenerated from the committed generator and verified against the manifest) was run unchanged, with the original scoring procedure (chat-template rendering, thinking disabled where supported, conditional last-token logits over single-token A/B and X/Y continuations, no sampling), on:
   - Qwen/Qwen3.6-27B @ `6a9e13bd` - **parity re-run** against committed `behavior_v1.json`
   - mistralai/Mistral-Small-3.2-24B-Instruct-2506 @ `95a6d26c`
   - Qwen/Qwen3-30B-A3B-Instruct-2507 @ `0d7cf239` and Qwen/Qwen3-30B-A3B-Base @ `1b75feb7` (base-vs-instruct arm; no official Qwen3.6 base exists)
   - zai-org/GLM-4.5-Air @ `a24ceef6`

2. **Harness parity verification.** The Qwen3.6-27B re-run reproduces the committed payload: 2/768 near-tie action flips (kernel nondeterminism), 0/1536 report flips, identical overall accuracy, all four frozen gates pass on the fresh payload (H2 gap 22.92 pts, p=.000275).

3. **GPU tier cost/perf benchmarks** across A100-80GB / H100 / H200 for the ~24-30B class and 4xA100-80GB vs 2xH200 for the ~106B class, with explicit cold-load/warm-run boundaries.

4. **J-lens artifacts**: the external failure diagnosis filed verbatim (`docs/reviews/jlens-failure-diagnosis-2026-10-23.md`) and its proposed controlled comparison implemented and run (`modal_jlens_local_test.py`).

## Headline results (full detail in `results/rbg4_multimodel/FINDINGS.md`)

- **Directional replication in every instruction-tuned model.** Opposed-assertion action accuracy drops after four redundant *correct* demonstrations in Qwen3.6-27B (-22.9 pts overall), Mistral-24B (-65.6), Qwen3-30B-A3B-Instruct (-29.2), GLM-4.5-Air (-40.6); all significant by the frozen exact cluster sign-flip test. Prose-mapping harm: Mistral 79.2 pts > GLM 58.3 > Qwen3.6 47.9 > Qwen3-30B-Instruct 27.1.
- **Table protection observed only for Qwen3.6 in these cells.** Qwen3.6 shows no table harm in this sample (83.3% -> 85.4%); under the identical table Mistral loses 52.1 pts, GLM 22.9 pts, and Qwen3-30B-Instruct 31.2 pts. Observed cell behavior in the tested models, not an established invariance of the format.
- **Base vs instruct.** Qwen3-30B-A3B-Base cannot play (opaque 50.8% = chance) and shows no effect (-2.1 pts, p=.89); its instruct sibling plays partially and shows the effect. In this pair, the instruct model shows the directional effect while the near-chance base model does not; the design cannot separate post-training effects from differences in task competence.
- **Gate-level replication does not transfer.** Only Qwen3.6 passes the frozen H1 competence gate (opaque >=95%, reports >=95%). GLM passes reports (96.8%) but fails opaque (86.5%); Mistral fails both. Effect sizes remain large but H2/H3 are formally not promoted for comparators.
- **Semantic interference vs general difficulty (descriptive).** The structurally matched (not token-matched) opaque-token surface *improves* with redundant history in Mistral (+40.6 pts) and GLM (+18.8) while assertions collapse - inconsistent with a simple monotone prompt-load account, not decisive against all difficulty accounts (learning benefit vs interference can trade off at different baselines).
- **J-lens controlled comparison: one rescue path fails as implemented.** Prompt-local Jacobians at the H0 single write site (layer 40, all positions, alpha=1.0) convert 3/36 (8.3%) vs a matched internal average-lens baseline of 25/36 (69.4%), which closely replicates the H0 reference 24/33 (72.7%). Paired: both 3, average-only 22, local-only 0, neither 11 (exact two-sided p=4.8e-07). Scoped read: replacing the averaged lens with naive per-prompt Jacobians does not rescue interventions under this implementation; this is not a general verdict on estimator mismatch or subspace writes (position reduction, output-map/final-norm chain-rule handling, J conditioning, and direction scaling were not varied). Consistent with, but does not by itself settle, the review note's keep-as-readout recommendation.

## Interpretation boundary

Cross-model results are a behavioral lead, not evidence for a shared mechanism: two non-Qwen comparators, 24 shared base games, frozen competence gates failing for all comparators, and consequence reports measured in separate queries (not action-time internal state). Mechanistic readings in the findings doc are calibrated proposals only.

## Cost (elapsed-rate estimates; Modal list rates verified 2026-10-23)

All figures are measured elapsed seconds x Modal list rates (A100-80GB $0.000694/s, H100 $0.001097/s, H200 $0.001261/s), NOT provider-billed amounts; Modal billing was not retrieved and actual charges may differ. GPU, CPU, and memory are metered separately (CPU/memory are not multiplied by GPU count).

| Item | Basis | Cost |
|---|---|---|
| Behavior runs (5 models) | measured | $2.33 (qwen36 parity $0.258, mistral $0.146, qwen3-instruct $0.097, qwen3-base $0.086, glm $1.740 on 4xA100-80GB) |
| GPU tier benchmarks (8 runs) | measured | $2.03 ($0.59 small models, $1.44 GLM) |
| J-lens local test | measured | $0.379 |
| Downloads/preflights (~$0.16) + cancelled attempts (~$0.41) | estimated | ~$0.57 |
| **Total (unbuffered estimate)** | | **~$5.31** |

Coverage: the append-only ledger (`results/rbg4_multimodel/cost_ledger.jsonl`) records 8 of the 14 metered runs (measured $4.14; conventionally buffered 1.2x rows $4.97); the 6 small-model benchmarks ($0.59) are costed from benchmark-payload timed seconds at the same rates, not ledger rows. Incidentals are included once, unbuffered. Applying the repo's 20% uncertainty convention to all metered compute plus incidentals gives a conservative planning envelope of ~$6.3. All figures are elapsed-rate estimates, not billed charges, and sit well within the pre-approved $20 spend cap.

## Provenance

- Raw payloads: `results/rbg4_multimodel/raw/` (run IDs, resolved revisions, GPU, library versions per model)
- Analyses: `results/rbg4_multimodel/analysis/` (unmodified `scripts/analyze_v5_inverse_evidence.py`)
- Benchmarks: `results/rbg4_multimodel/benchmark/`
- Harness: `modal_rbg4_multimodel.py` (new; model as runtime parameter, frozen hash verification retained, per-model preflight, volume-backed persistence)
