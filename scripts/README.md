# Analysis scripts

These scripts turn committed raw rows into summaries and figures. They are
analysis code, not new experiment definitions.

| Script | Purpose |
|---|---|
| [`analyze_v2_h0.py`](analyze_v2_h0.py) | Rebuilds the original V2 topology table, topology figure, and workspace-selection figure. |
| [`analyze_h0r.py`](analyze_h0r.py) | Rebuilds the H0R diagnostic summaries, candidate ranking, and diagnostic figures. |
| [`analyze_results.py`](analyze_results.py) | Summarizes the Stage 1A–1D outputs from the exploratory V1 foundation protocol. |
| [`analyze_causal_discovery.py`](analyze_causal_discovery.py) | Rebuilds the V1 causal-discovery summaries. |
| [`analyze_stage1.py`](analyze_stage1.py) | Rebuilds the V2-SR-1 state/report tables and figure. |
| [`analyze_jspace_interventions.py`](analyze_jspace_interventions.py) | Rebuilds the V3-JI1 intervention-characterization summaries. |
| [`analyze_strategic_trajectories.py`](analyze_strategic_trajectories.py) | Rebuilds the V2-ST-1 trajectory checks. |
| [`analyze_v4_*.py`](.) | Rebuilds the V4 rationale-interference analyses. |
| [`analyze_v5_*.py`](.) | Rebuilds the V5 game-study analyses. |
| [`analyze_v5_mechanistic_decomposition.py`](analyze_v5_mechanistic_decomposition.py) | Rebuilds the locked RBG-5 patch gate and probe tables from downloaded immutable payloads. |
| [`analyze_v5_full_action_trajectory.py`](analyze_v5_full_action_trajectory.py) | Rebuilds the V5-RBG-6 endogenous trajectory analysis locally. |
| [`freeze_v5_full_action_trajectory.py`](freeze_v5_full_action_trajectory.py) | Materializes the RBG-6 dataset from the immutable RBG-4 payload. |
| [`prepare_v5_full_action_trajectory.py`](prepare_v5_full_action_trajectory.py) | Tokenizes and validates all RBG-6 branches on CPU before Modal GPU execution. |
| [`generate_dataset.py`](generate_dataset.py) | Creates deterministic experiment data from the configured prompt families. |
| [`modal_v2.py`](../modal_v2.py) | Runs the original V2 GPU workflow, including the H0 gate. |
| [`modal_h0r.py`](../modal_h0r.py) | Runs the H0R diagnostic and prospective GPU workflows. |
| [`run_v7_epistemic_agentic_strategy_controls.py`](run_v7_epistemic_agentic_strategy_controls.py) | Generates, freezes, and semantically audits the V7-EAS-1 CPU dataset before any model call. |
| [`analyze_v7_epistemic_agentic_strategy.py`](analyze_v7_epistemic_agentic_strategy.py) | Analyzes locked V7 direct-logit outputs by family and matched epistemic pair. |

The two most important regeneration commands are shown in the [root README](../README.md#reproduce-the-local-checks).

The scripts should read existing raw output and write derived tables or plots.
They should not silently change a locked protocol or overwrite a prospective
result directory.

[Back to the project README](../README.md)
