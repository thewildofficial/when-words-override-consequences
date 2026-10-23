# RBG-4 multi-model replication findings

**Date:** 2026-10-23
**Status:** Frozen runs archived; code/documentation review corrections applied, Mistral scoring sensitivity and corrected multi-GPU load benchmarks pending (not run)
**Dataset:** frozen V5-RBG-4 inverse-evidence dataset, regenerated from the
committed generator; content SHA-256 verified against
`configs/v5/inverse_evidence/dataset_manifest.json`
(`3fdc9977f03700dbfb2118414ec9602a4e76d9d6e2b35788a81bc6acf4a1f5d1`). 768 rows,
2,304 queries per model: one forced-choice action query (A/B) and two
consequence-report queries (X/Y) per row.

## Method

The original RBG-4 harness (`modal_inverse_evidence.py`) was reused verbatim:
same system prompt, same chat-template rendering with
`add_generation_prompt=True`, thinking disabled where the template supports the
`enable_thinking` kwarg (Qwen and GLM templates; Mistral's template has no such
kwarg), and the same conditional scoring: raw logits at the final position over
the single-token candidate continuations, argmax over candidates, no sampling.
Each model ran a CPU preflight first (tokenizer load, chat-template render,
one-token-continuation gate for every candidate label in all 2,304 queries).
Analysis used `scripts/analyze_v5_inverse_evidence.py` unmodified, including
the frozen preregistered gates, which are retained here as the common
scoreboard even though they were frozen for Qwen3.6-27B specifically.

All runs executed on Modal (single A100-80GB per run, bf16, except GLM-4.5-Air
on 4xA100-80GB). Model weights pinned by revision SHA. Raw payloads in
`results/rbg4_multimodel/raw/`, analysis JSONs in
`results/rbg4_multimodel/analysis/`.

## Harness parity check (prerequisite)

Qwen3.6-27B was re-run at the frozen revision
(`6a9e13bd...`) and compared row-by-row against the committed
`results/v5_inverse_evidence/raw/behavior_v1.json`:

- Action choice mismatches: 2/768 (0.26%), near-tie argmax flips consistent with numerical variation; cause not established; overall action accuracy identical (0.9128 both)
- Report choice mismatches: 0/1,536
- Maximum absolute logit difference: 0.75, confined to the flip rows
- All four frozen gates reproduced on the fresh payload (H1, H2, H3, H4 all
  pass; H2 gap 22.92 pts, p=0.000275)

The parity check validates the Qwen3.6 path, not equivalent answer scoring across tokenizers. Cross-model scores below use the frozen bare-token scorer; Mistral sensitivity remains unresolved.

## Cross-model scoreboard

All percentages are action accuracy unless noted. "Prose +/- demos" is the
opposed-incentive assertion cell with no history vs four redundant correct
demonstrations, prose mapping. "Table" is the same cell with the consequence
table. Gates marked PASS/FAIL are the frozen RBG-4 gates applied verbatim.

| Model | Opaque | Reports | Prose no-demos | Prose +demos | Prose harm | Table no-demos | Table +demos | Table harm | H1 | H2 gap (p) |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.6-27B (parity) | 100.0% | 98.96% | 75.0% | 27.1% | 47.9 pts | 83.3% | 85.4% | -2.1 pts | PASS | 22.9 pts (p=.00027) PASS |
| Mistral-Small-3.2-24B | 80.5% | 89.3% | 85.4% | 6.2% | 79.2 pts | 77.1% | 25.0% | 52.1 pts | FAIL | 65.6 pts (p=2.4e-07) not promoted |
| Qwen3-30B-A3B-Instruct | 88.8% | 80.9% | 64.6% | 37.5% | 27.1 pts | 79.2% | 47.9% | 31.2 pts | FAIL | 29.2 pts (p=.034) not promoted |
| Qwen3-30B-A3B-Base | 50.8% | 55.9% | 66.7% | 64.6% | 2.1 pts | 64.6% | 70.8% | -6.2 pts | FAIL | -2.1 pts (p=.89) not promoted |
| GLM-4.5-Air | 86.5% | 96.8% | 62.5% | 4.2% | 58.3 pts | 41.7% | 18.8% | 22.9 pts | FAIL (opaque) | 40.6 pts (p=.00088) not promoted |

## Findings

### 1. Directional history harm under the frozen bare-token scorer

Every instruction-tuned model tested shows worse opposed-assertion action
accuracy after four redundant *correct* demonstrations than with no history:
Qwen3.6-27B -22.9 pts (overall opposed assertion), Mistral-Small-3.2-24B -65.6
pts, Qwen3-30B-A3B-Instruct -29.2 pts. The effect direction is unanimous and
each is significant by the exact cluster sign-flip test.

### 2. The consequence table is not universally protective

In these cells, Qwen3.6-27B shows no table harm (83.3% -> 85.4%). The other
tested models show no such protection: Mistral loses 52.1 pts under a table,
Qwen3-30B-A3B-Instruct loses 31.2 pts. The table protection observed for
Qwen3.6-27B in RBG-4 is, in this sample, an observed behavior of the tested
model and cells, not an established invariance of the representation format.

### 3. Base vs instruct pair under this chat-format evaluation

Under this chat-format evaluation, Qwen3-30B-A3B-Base performs near chance: opaque-token
actions at 50.8% (coin flip), reports at 55.9%, aligned-goal assertion
accuracy 31.8%. The history contrast is -2.1 pts (p=.89) in this evaluation, not evidence that redundant history has no effect generally. Its top token is `user` on 761/768 action queries; no action or report top tokens are legal candidates. A base-appropriate continuation evaluation was not run. Its instruction-tuned sibling plays partially and shows
the failure. In this pair, the instruct model shows the directional effect
while the near-chance base model does not; the design cannot separate
post-training effects from differences in task competence or evaluation-format fit.

### 4. Gate-level replication is Qwen3.6-specific

Only Qwen3.6-27B passes the H1 competence gate (opaque >=95%, reports >=95%).
Mistral and Qwen3-30B-A3B-Instruct show larger history harm but fail H1, so
H2/H3 are formally "not promoted" under the frozen gates. This is expected:
gates frozen on one model's competence profile do not transfer verbatim. The directional results remain descriptive under the frozen scorer; failed competence gates and unresolved scoring sensitivity prevent promoting them to gate-level replication.



### 5. GLM-4.5-Air: the failure at high report competence

GLM-4.5-Air reports option consequences at 96.8% (passing the 95% report gate)
and plays aligned games at 89.6%, yet collapses to 4.2% (2/48 both frames,
1/24 strategic-only) on the opposed/redundant/prose assertion cell. Its
opaque-token control (86.5%) fails the 95% gate, so the frozen H1 gate does
not transfer; the behavioral pattern nonetheless matches the RBG-4 dissociation
profile. Reports remain below Qwen3.6-27B (98.96%), so this is not a
higher-competence replication.

### 6. Semantic interference vs general difficulty: the sign-flip control

The assertion and opaque-token surfaces use structurally matched demonstrations with the same correctness and placement, but are not token- or character-length matched. The sign contrast tests a simple monotone prompt-load account, not general difficulty accounts as a class. The
history effect (opposed rows, no-history accuracy minus redundant-history):

| Model | Assertion harm | Opaque-token harm |
|---|---|---|
| Qwen3.6-27B | -22.9 pts | 0.0 pts |
| Mistral-Small-3.2-24B | -65.6 pts | **+40.6 pts (improves)** |
| Qwen3-30B-A3B-Instruct | -29.2 pts | -3.1 pts |
| Qwen3-30B-A3B-Base | +2.1 pts | +1.0 pts |
| GLM-4.5-Air | -40.6 pts | **+18.8 pts (improves)** |

For Mistral and GLM, the structurally matched demonstrations help when their
content is meaningless and hurt when it is meaningful, within the same model.
This rules out a *simple monotone* account in which extra prompt load degrades
both surfaces alike; it does not rule out every difficulty-based explanation,
because a learning benefit (demonstrations rehearsing the contrarian mapping)
and semantic interference can coexist and trade off differently at different
baseline competences. Caveats: (a) the surfaces are structurally matched, not
token-matched - redundant-history assertion prompts run ~65 characters longer
than opaque-token prompts (mean 907 vs 842 chars; token counts differ per
tokenizer), so a residual length confound remains possible in principle; the
repo's separate token-matched follow-up (semantic_action_mechanisms) addresses
token matching for a related state effect, not for this history contrast;
(b) Mistral's opaque no-history baseline is low (40.6%), leaving large headroom
for the opaque improvement - the sign asymmetry, not the magnitudes, is the
informative part. The frozen H3 difference-in-differences captures the same
contrast: GLM 59.4 pts p=.00042; Mistral 106.3 pts p=2.4e-07.

### 7. Dissociation signature is design-forced (consistency check only)

In this two-alternative contrarian design, when both option reports are
correct and the action is wrong, the wrong option's assertion content equals
the target response by construction. The N/N counts (Qwen3.6 51/51, Mistral
49/49, Qwen3-30B-Instruct 26/26, GLM 97/97) are therefore a consistency
property of the game, not independent mechanistic evidence. The base model had
only 2 such rows.

## Review correction: answer scoring and reproducibility

The frozen bare-token results are preserved unchanged. Mistral's global top
answer is space-prefixed A/B on 625/768 actions and EOS on 143; all 1,536
report top tokens are space-prefixed X/Y. Stripping the legal global top token
disagrees with the bare scorer on 8 actions and 34 reports. This verifies an
answer-format sensitivity, not a complete alternative scorer. Saved payloads
contain only bare candidate logits and global top-1, so a full comparison of
whitespace candidate scores, particularly the 143 EOS-top action rows, cannot
be recovered offline. A separate whitespace-aware sensitivity analysis remains
pending; no new inference was run. Do not promote Mistral to tokenizer-robust
cross-model replication from these scores alone.

The committed behavior entrypoint multiplied host CPU/memory charges by GPU
count: its GLM timeout estimate was $22.9796352 buffered even with an empty
ledger. Applying GPU count only to GPU charges gives $14.7391488 before
adding existing ledger spend. The archived GLM ledger row already uses the
correct elapsed-rate accounting ($1.7398002 unbuffered); no historical row or
raw payload is rewritten. The GLM payload records the then-main commit, not
a reproducible invocation of this previously blocked entrypoint. Historical
execution records confirm that GLM was launched directly through the deployed
remote behavior function, bypassing the local entrypoint and its admission
check; the correct ledger row was recorded separately. This explains the
payload despite the entrypoint bug, but is a reproducibility and enforcement
gap, not evidence that the local guard worked. The code correction
is not a new GPU validation.

## Cost

All figures are elapsed-time estimates (measured seconds x Modal list rates
verified 2026-10-23: A100-80GB $0.000694/s, H100 $0.001097/s, H200
$0.001261/s), not provider-billed amounts; Modal billing was not retrieved
and actual charges may differ. GPU, CPU, and memory are metered separately
(CPU/memory are not multiplied by GPU count).

| Item | Basis | Cost |
|---|---|---|
| qwen36-27b parity behavior | measured | $0.258 |
| mistral-small-32-24b behavior | measured | $0.146 |
| qwen3-30b-a3b-instruct behavior | measured | $0.097 |
| qwen3-30b-a3b-base behavior | measured | $0.086 |
| glm45-air behavior (4xA100-80GB, 509.9s) | measured | $1.740 |
| 6 GPU tier benchmarks (small models) | measured | $0.593 |
| glm45-air benchmarks (4xA100-80GB, 2xH200) | measured | $1.437 |
| J-lens local test (A100-80GB, 463.4s) | measured | $0.379 |
| Weight downloads + preflights (CPU) | estimated | ~$0.16 |
| Cancelled/failed attempts | estimated | ~$0.41 |
| **Total (unbuffered estimate)** | | **~$5.31** |

Coverage and uncertainty: the append-only ledger
(`results/rbg4_multimodel/cost_ledger.jsonl`) records 8 of the 14 metered
runs (5 behavior runs, 2 GLM benchmarks, the J-lens test): measured
subtotals $4.14, conventionally buffered (1.2x) rows $4.97. The 6
small-model benchmarks ($0.59) were not written to the ledger; their cost is
computed from the benchmark payloads' timed seconds at the same list rates.
The ~$0.57 incidentals are coarse estimates, included once, unbuffered. The
unbuffered total above (~$5.31) is the headline estimate; applying the
repo's 20% uncertainty convention to all metered compute plus incidentals
would give a conservative planning envelope of ~$6.3. Both figures are
elapsed-rate estimates within the pre-approved $20 spend cap, not billed
charges.

## Provenance

- Dataset SHA-256: `3fdc9977f03700dbfb2118414ec9602a4e76d9d6e2b35788a81bc6acf4a1f5d1`
- Run IDs and per-model revisions: see `results/rbg4_multimodel/raw/*/metadata`
- Harness: `modal_rbg4_multimodel.py` (new file, this branch)
- Analysis: `scripts/analyze_v5_inverse_evidence.py` (unmodified)

## GPU tier benchmarks

Method: 96 action queries (uniformly sampled by sequence length from the 768)
timed per tier after weight load; full-run estimate = load_seconds +
query_seconds x (2304/96). Costs are GPU-only at verified Modal list rates
(2026-10-23). `load_seconds` includes tokenizer loading, model deserialization and device transfer. A cached-weight invocation still pays those costs; subtracting them estimates a resident-model run, not a warm-cache invocation. GLM multi-GPU load measurements include an avoidable CPU load followed by a second dispatched load. Preserve these historical measurements but withdraw corrected-load performance claims pending a separately authorized rerun. Benchmark payloads in
`results/rbg4_multimodel/benchmark/`.

| Model (class) | Tier | Load (s) | ms/query | Full-run est. (s) | GPU cost |
|---|---|---|---|---|---|
| Qwen3.6-27B (~27B hybrid) | A100-80GB x1 | 40.4 | 103.9 | 279.8 | $0.194 |
| Qwen3.6-27B | H100 x1 | 56.2 | 89.1 | 261.5 | $0.287 |
| Qwen3.6-27B | H200 x1 | 39.2 | 94.6 | 257.2 | $0.324 |
| Qwen3-30B-A3B (~30B MoE) | A100-80GB x1 | 70.1 | 27.7 | 133.9 | $0.093 |
| Qwen3-30B-A3B | H100 x1 | 149.0 | 36.3 | 232.7 | $0.255 |
| Qwen3-30B-A3B | H200 x1 | 94.0 | 33.4 | 170.8 | $0.215 |
| GLM-4.5-Air (~106B MoE) | A100-80GB x4 | 277.5 | 58.8 | 413.1 | $1.147 |
| GLM-4.5-Air | H200 x2 | 144.1 | 51.9 | 263.7 | $0.665 |

Readout: these are single 96-action-query samples, extrapolated to 2,304
queries including reports, not repeated production measurements. Single-GPU
results are workload-specific: A100 is cheapest in the measured configurations,
but H100/H200 do not consistently improve wall-clock time (the Qwen3-30B
samples are slower). The GLM historical projections favor 2xH200 but their
load terms include the double-load bug; withdraw any claim of corrected-load
dominance or a causal explanation for the difference until remeasured.
Placement uses `device_map="auto"`, not explicit tensor parallelism. The
Qwen3.6 GPU projection $0.194 vs measured $0.206 is about 6% apart; that
single comparison does not validate all projections within 15%.



## Interpretation boundary

Everything above the scoreboard line is descriptive. The cross-model pattern is
a behavioral lead, not evidence for a shared mechanism: two non-Qwen
comparators, 24 shared base games, no comparator besides Qwen3.6-27B passing
the frozen competence gates, and consequence reports measured in separate
queries (which do not establish the action-query internal state). Any
mechanistic reading is a calibrated proposal for future work, not a claim of
this study.

Proposed next contrast (not run): a multi-lexical-family counterbalanced
variant in which demonstration correctness under the *same* causal mapping is
preserved while semantic relatedness between demonstration content and action
surfaces is varied across families (the design pattern of
experiments/semantic_action_mechanisms). Naively shuffling concept labels
within a game would produce demonstrations inconsistent with the same mapping
and cannot isolate semantic interference. Related follow-up: run the RBG-6
self-report-card trajectory arm on the comparators (RBG-6 on Qwen3.6-27B:
46/46 conditional on both reports correct, 46/48 = 95.83% across all primary
trajectories, 0/48 dissociation) to test whether the rescue transfers.

## J-lens controlled comparison: prompt-local Jacobians vs the averaged lens

The external review note (`docs/reviews/jlens-failure-diagnosis-2026-10-23.md`)
proposed a small controlled test of one specific rescue path for J-space
interventions: replacing the averaged-lens directions with per-prompt
Jacobians at the same write site. This is a comparison of two direction
estimators under one fixed implementation and topology, not a general verdict
on subspace-write interventions.

Method (`modal_jlens_local_test.py`): the H0 country category (4 countries x
4 functions x 3 targets = 48 trials, none dropped for tokenization), single
write site layer 40, all positions, alpha=1.0, identical swap code path and
direction formula (normalized W_U[t] @ J) for both estimators. Eligibility
(baseline top-1 equals the source answer) was computed from the baseline
forward BEFORE either intervention: 36 of 48 trials eligible (capital 12/12,
continent 12/12, language 9/12, currency 3/12). The H0 reference run had 33
eligible country trials; the +3 difference is an observed discrepancy whose
cause was not established (no repeated-baseline diagnostic was run here). Every eligible trial ran both
conditions in the same process. Prompt-local Jacobians used upstream
`jacobian_for_prompt` at the pinned jlens commit (dim_batch=64); because the
country prompts are 8-12 tokens (shorter than the fitting default
skip_first=16), only the BOS position was skipped.

Result (36 eligible trials, paired):

| Directions | Target top-1 | Mean target log-odds |
|---|---|---|
| Average lens (matched H0 baseline) | 25/36 (69.4%) | -0.97 |
| Prompt-local Jacobian | 3/36 (8.3%) | -5.31 |

Paired counts: both succeed 3, average-only 22, local-only 0, neither 11
(discordant 22, all favoring the average lens). Targets share each source
prompt's baseline and local Jacobian, so trial-level independence is not
appropriate. Aggregate paired success differences by source prompt: 12
eligible prompt clusters, 9 nonzero, all favoring the average lens; exact
two-sided prompt-cluster sign-flip p=0.00390625. This assumes prompt-cluster
sign exchangeability; shared country/function structure and small scope limit
generalization. The earlier trial-level p=4.8e-07 is superseded.
The internal baseline (25/36, 69.4%) closely replicates the H0 reference
(24/33, 72.7%), which is a baseline consistency check, not validation of all implementation choices. Per-function: capital avg
12/12 local 0/12; continent 9/12 vs 0/12; language 4/9 vs 3/9; currency
0/3 vs 0/3.

Interpretation, scoped: under THIS implementation (position reduction with
skip_first=1, shared normalization with clamp_min(1e-12), single site,
alpha=1.0), prompt-local directions convert far worse than the averaged lens. This removes
one specific rescue path as implemented; it does NOT establish that the
averaged estimator is never a bottleneck, that subspace writes are
fundamentally limited, or that the intervention branch should be closed.
Unmeasured factors that could materially change the outcome include the
position reduction (skip_first=1 here vs 16 in the released lens fitting),
per-prompt J conditioning and direction norm distributions (not recorded in
this run), direction scaling/swap conditioning, and the output-map chain
rule: both direction sets use raw lm_head weights without the final RMSNorm.
That is a shared implementation choice, but it can affect the validity of
local and averaged directions differently and is NOT a ruled-out concern for
the local estimator's validity: a correct per-prompt logit Jacobian at the write
site must compose through the final norm (and any bias/softcap, per the
upstream unembed), and its omission may degrade naive per-prompt directions
disproportionately relative to the doubly-averaged directions. The result is consistent
with the review note's strategic recommendation (keep J-lens as a descriptive
readout; do not rebuild the causal spine on it) but does not by itself settle
the branch's fate.

Cost: $0.379 elapsed-rate estimate (463.4s A100-80GB).
