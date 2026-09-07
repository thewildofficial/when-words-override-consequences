# V6-ES-1 run record

This directory records the CPU-only protocol freeze and the primary
Qwen3.6-27B execution artifacts. The generated dataset is frozen under
[`configs/v6/epistemic_strategic/`](../../configs/v6/epistemic_strategic/).

The audit verified:

- 1,740 deterministic rows across `discovery`, `validation`, and `locked`;
- complete factorial coverage for each family;
- unique action/report certificates and unique condition IDs;
- modeled-belief action switches and hidden-actual-belief invariance controls;
- higher-order visibility/role cells;
- provenance source/truth/time/surface cells;
- safety-frame identification cells and evaluation-framing invariance.

The CPU audit is immutable. The primary model outputs below are also immutable
and should be interpreted only through the family-specific gates in
`control_audit.json` and the preregistration.

## Primary run outcome

The pinned `Qwen/Qwen3.6-27B` tokenizer preflight, black-box generation,
forced-choice logits, and matched trajectory controls completed successfully.
The behavior artifact is retained under `raw/behavior_primary.json`; its
family-level gate is **false** despite 0.945 overall accuracy:

- `core_tom`: gate false; pair identification `0.625`;
- `higher_order`: gate false; accuracy `0.454`;
- `provenance`: gate true; accuracy and claim-content invariance `1.000`;
- `strategy_pressure`: gate false; risk-pair accuracy `0.429`.

Because the preregistered behavior gate failed, observational activation
discovery and causal activation interchange were not run. The black-box and
trajectory outputs remain separate evidence levels. See
[`model_run_manifest_primary.json`](model_run_manifest_primary.json) for
workflow IDs, hashes, measured cost, and compact summaries.

Qwen3.8 remains unpaid and locked: its preflight matched candidate token IDs
but not Qwen3.6's rendered prompts or prompt-token IDs. That is a replication
confound, not a pooled model result.
