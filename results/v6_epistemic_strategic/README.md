# V6-ES-1 local control audit

This directory records the CPU-only protocol freeze. It contains no model
result. The generated dataset is frozen under
[`configs/v6/epistemic_strategic/`](../../configs/v6/epistemic_strategic/).

The audit verified:

- 1,740 deterministic rows across `discovery`, `validation`, and `locked`;
- complete factorial coverage for each family;
- unique action/report certificates and unique condition IDs;
- modeled-belief action switches and hidden-actual-belief invariance controls;
- higher-order visibility/role cells;
- provenance source/truth/time/surface cells;
- safety-frame identification cells and evaluation-framing invariance.

Model execution is deliberately absent. Run the Modal stages described in
[`../../docs/v6/README.md`](../../docs/v6/README.md) only after reviewing the
family-specific gates in `control_audit.json`.
