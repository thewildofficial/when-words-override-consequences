# V6.2 reasoning capability control results

This directory is the V6.2 result namespace. The initial committed artifact is
CPU-only:

> No model forwards have occurred on this branch yet.

The CPU controls record the V6.1 source hashes, selected-row IDs, subset hash,
semantic prompt contract, parser tests, endpoint coverage, choice-mapping
invariance, budget authorization, and zero GPU/model-forward counts. Paid
artifacts are added only by the separately dispatched GitHub Actions stages.

Required result files after paid execution are content-hashed raw artifacts
under `raw/`, `analysis.json`, compact model-run manifests, and the persistent
`cost_ledger.jsonl`. GitHub Actions artifacts are retained for 30 days and are
not archival storage.
