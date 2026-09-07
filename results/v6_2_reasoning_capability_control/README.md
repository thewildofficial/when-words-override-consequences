# V6.2 reasoning capability control results

This directory is the V6.2 result namespace. The initial committed artifact is
CPU-only:

> No model forwards have occurred on this branch yet.

The CPU controls record the V6.1 source hashes, selected-row IDs, subset hash,
semantic prompt contract, parser tests, endpoint coverage, choice-mapping
invariance, budget authorization, and zero GPU/model-forward counts. Paid
artifacts are added only by the separately dispatched GitHub Actions stages.

Required result files after paid execution are content-hashed raw artifacts
under `raw/`, immutable per-stage `analysis_*.json` files, derived
`analysis_cumulative.json`, compact model-run manifests, and the persistent
`cost_ledger.jsonl`. GitHub Actions artifacts are retained for 30 days and are
not archival storage.

[`v61_reference_endpoints.json`](v61_reference_endpoints.json) is the immutable
secondary reanalysis that supplies the strict H3 Qwen3.6 reference used by the
cross-model comparison. It is derived from the archived V6.1 locked raw
artifact and is not a modification of V6.1's production analysis or gates.
