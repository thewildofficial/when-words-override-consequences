# V6.1 epistemic repair — run record

Study ID: `V6.1-ES-2`
Branch: `research/v6-1-epistemic-repair`

## Current status

The local CPU protocol audit passed. It generated and verified 6,144 rows
across seven family-specific factorials and three splits, including 2,048
locked rows. The generated table is hash-pinned but git-ignored; it is
materialized by the control step in every clean checkout. No model forward
pass has occurred in this checkout yet.

The machine-readable records are:

- [`control_audit.json`](control_audit.json) — prompt completeness, identifying
  cells, trajectory prompt identity, and family structural gates;
- [`run_manifest.json`](run_manifest.json) — content hashes and zero-compute
  provenance.

The expected remote artifacts, once the manual GitHub Actions workflow runs,
are stored under `raw/`:

- `preflight_primary.json`
- `behavior_primary.json`
- `blackbox_primary.json`
- `analysis_primary.json`

These files are content-addressed and refuse non-identical overwrite. A model
failure is retained as a family-specific negative; infrastructure or parity
failures are labeled separately and are not scored as model behavior.

Activation collection is sealed for this study. Even a positive behavioral
run would only authorize a separately frozen mechanistic follow-up.
