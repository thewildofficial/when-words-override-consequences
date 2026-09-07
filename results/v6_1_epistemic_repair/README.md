# V6.1 epistemic repair — run record

Study ID: `V6.1-ES-2`
Branch: `research/v6-1-epistemic-repair`

## Current status

The local CPU protocol audit passed. It generated and verified 7,488 rows
across seven family-specific factorials and three splits, including 2,496
locked rows. The generated table is hash-pinned but git-ignored; it is
materialized by the control step in every clean checkout. No model forward
pass has occurred in this checkout yet.

The machine-readable records are:

- [`control_audit.json`](control_audit.json) — prompt completeness, identifying
  cells, trajectory prompt identity, and family structural gates;
- [`run_manifest.json`](run_manifest.json) — content hashes and zero-compute
  provenance.

Confirmatory analysis aggregates each endpoint to the declared `game_id`
cluster before bootstrapping. Paired sign-flip tests are conditional on a
family gate and never rescue a failed gate. H5 uses matched
positive/negative VOI profiles plus a same-matrix cost/reliability threshold
control; H7 treats fixed scaffolds as content-only and reserves the
“self-generated” label for a real preceding assistant report turn.
All primary matched pairs also reuse one deterministic A/B label permutation;
the mapping hash excludes the treatment and is audited before execution.

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
