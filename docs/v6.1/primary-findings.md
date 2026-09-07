# V6.1 primary findings — Qwen3.6-27B

**Study:** `V6.1-ES-2`

**Model:** `Qwen/Qwen3.6-27B`, pinned revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`

**Locked split:** 2,496 rows

**Config:** `c2c272a2fb7f3feeceb3651b85b334bfcd9177b05353fb1ee5e86c9b0ff1b9a5`

**Dataset:** `14aea862bd5f195b11ebe77f97fb6be857717bd10c47375e9c11cd773214d77a`

## Result

The forced-choice stage generated all 7,488 frozen rows across discovery,
validation, and locked splits. Confirmatory analysis used only the
preregistered 2,496-row locked split. It did not pass the preregistered study
gate: all seven family-specific hypothesis gates were falsified. The result is
retained as a family-resolved behavioral negative; pooled accuracy is not used
to rescue any failed family.

| Family | Locked accuracy | Gate |
|---|---:|---|
| `ledger_binding` | 91.25% | Falsified |
| `policy_composition` | 50.00% | Falsified |
| `evidence_update` | 60.76% | Falsified |
| `recursive_strategy` | 45.31% | Falsified |
| `active_information` | 51.39% | Falsified |
| `monitoring_goal` | 66.67% | Falsified |
| `scaffold_order` | 68.75% | Falsified |

### Confirmatory endpoints

| Hypothesis | Frozen endpoint | Result |
|---|---|---:|
| H1 ledger binding | Report accuracy | 100% |
| H1 ledger binding | Modeled-belief action-pair rate | 12.5% |
| H1 ledger binding | Actual-belief invariance | 87.5% |
| H2 policy composition | Literal↔contrarian correct-switch rate | 0% |
| H3 evidence update | Independent↔copied identifying contrast | 1.39% |
| H3 evidence update | Explicit-rule contrast | 0% |
| H3 evidence update | Provenance-only contrast | 2.78% |
| H4 recursive strategy | Depth-0 accuracy | 68.75% |
| H4 recursive strategy | Complete depth-0…3 sequence | 0% |
| H5 active information | Positive↔negative VOI switch | 0% |
| H5 active information | Same-matrix cost switch | 0% |
| H5 active information | Same-matrix reliability switch | 0% |
| H6 monitoring goal | Audit-cue discordance | 0% |
| H6 monitoring goal | Real-penalty correct-switch rate | 33.3% |
| H7 scaffold order | Fixed-correct − fixed-wrong gap | 0 |
| H7 scaffold order | Direct report / action→report accuracy | 100% / 100% |
| H7 scaffold order | Direct/action→report trajectory change | 0% |

The forced-choice measurement was 62.30% overall on the locked split. A
separate free-generation run covered the same 2,496 locked conditions: it
achieved 62.66% accuracy, parsed every response, and generated only legal
`A`/`B` labels. The selected answer agreed with the logits measurement on
99.24% of matched conditions. This is a useful interface-level replication,
not evidence for a shared internal mechanism.

No activations were collected. The preregistered interpretation policy
therefore permits neither mechanistic nor deception claims. The black-box
stage was run separately because the combined workflow's conservative cost
projection reached $2.26 against the frozen $2.00 study limit. This is an
execution deviation: the separate Actions job started with a fresh job-local
ledger and therefore bypassed cumulative admission accounting. It did not
change the frozen dataset, hypotheses, endpoints, or model inputs; black-box
remains a descriptive companion and is not used to rescue the failed
confirmatory gates. The measured black-box subtotal was $0.42, the logits
stage measured $0.84, and the measured total was $1.26 (approximately $1.51
with the recorded 20% buffer). Future multi-job runs must persist and reload
the study ledger before GPU admission.

## Provenance

- [Machine-readable analysis](../../results/v6_1_epistemic_repair/analysis_primary.json)
- [Model-run manifest](../../results/v6_1_epistemic_repair/model_run_manifest_primary.json)
- [Cost ledger](../../results/v6_1_epistemic_repair/cost_ledger_primary.jsonl)
- [Full production analyzer output](../../results/v6_1_epistemic_repair/analysis_primary_production.json)
- [Behavior/full-primary Actions artifact](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34152531767/artifacts/10030218459)
- [Black-box Actions artifact](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34154058341/artifacts/10030538627)
- [Persistent raw-output release archive](https://github.com/thewildofficial/when-words-override-consequences/releases/tag/v6.1-es2-qwen36-primary)

The raw model outputs are also archived in the persistent release above rather
than being added as large git blobs. The analysis files record their content
hashes and the exact pinned protocol inputs.
