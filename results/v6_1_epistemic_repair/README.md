# V6.1 epistemic repair — run record

Study ID: `V6.1-ES-2`
Branch: `research/v6-1-epistemic-repair`

## Current status

The local CPU protocol audit passed. It generated and verified 7,488 rows
across seven family-specific factorials and three splits, including 2,496
locked rows. The generated table is hash-pinned but git-ignored; it is
materialized by the control step in every clean checkout.

The pinned Qwen3.6-27B primary run is complete. The forced-choice and
free-generation stages cover the locked split; all seven preregistered family
gates were falsified. The human-readable interpretation is in
[`docs/v6.1/primary-findings.md`](../../docs/v6.1/primary-findings.md).

The original [`run_manifest.json`](run_manifest.json) remains the immutable
pre-run CPU-control record. Post-run provenance is recorded separately in
[`model_run_manifest_primary.json`](model_run_manifest_primary.json),
[`analysis_primary.json`](analysis_primary.json), and
[`analysis_primary_production.json`](analysis_primary_production.json), which
is the exact full output of the production analyzer,
[`cost_ledger_primary.jsonl`](cost_ledger_primary.jsonl).

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

The raw remote artifacts are retained in the persistent
[V6.1 Qwen3.6 raw-output release](https://github.com/thewildofficial/when-words-override-consequences/releases/tag/v6.1-es2-qwen36-primary)
and mirrored in the original GitHub Actions artifacts. Their content hashes
and links are recorded in [`model_run_manifest_primary.json`](model_run_manifest_primary.json):

- `preflight_primary.json`
- `behavior_primary.json`
- `blackbox_primary.json`
- `analysis_primary.json` (compact overview)
- `analysis_primary_production.json` (full production analyzer output)

These files are content-addressed and refuse non-identical overwrite. A model
failure is retained as a family-specific negative; infrastructure or parity
failures are labeled separately and are not scored as model behavior.

Activation collection is sealed for this study. Even a positive behavioral
run would only authorize a separately frozen mechanistic follow-up.
