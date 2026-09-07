# Experiment configurations

The configuration files are the written instructions for a run. They make
prompts, controls, thresholds, model settings, and frozen intervention choices
inspectable instead of hiding them inside a runner.

| Area | Contents |
|---|---|
| [`experiment.json`](experiment.json) | Earlier experiment settings. |
| [`concepts.json`](concepts.json) | Token and concept families used by the earlier analyses. |
| [`v2/`](v2/README.md) | Protocol V2 preregistered controls, task-independent mapping, diagnostics, and frozen protocol files. |
| [`v3/jspace_interventions/`](v3/jspace_interventions/experiment.json) | V3-JI1 intervention-replication configuration and candidate pools. |
| [`v4/strategic_epistemic_search/`](v4/strategic_epistemic_search/experiment.json) | V4 rationale-interference datasets and frozen analyses. |
| [`v5/`](v5/mechanistic_decomposition/experiment.json) | V5 revealed-belief, inverse-evidence, and mechanistic-decomposition configurations. |
| [`v7/epistemic_agentic_strategy/`](v7/epistemic_agentic_strategy/experiment.json) | V7-EAS-1 frozen behavior-first epistemic strategy screen and dataset manifest. |

## A note about locked files

Some Protocol V2 files are intentionally immutable after a phase opens. In particular,
the H0R candidate protocol records the layers, positions, operation, strength,
validity limits, and prospective thresholds used for the fresh test.

The human explanation of those choices is in [`../docs/v2/README.md`](../docs/v2/README.md).

[Back to the project README](../README.md)
