# V7-EAS-1 result namespace

This directory is reserved for immutable outputs from the V7-EAS-1 direct
behavioral screen. `control_audit.json` and `run_manifest.json` are the
pre-run CPU record; raw model outputs and analysis are added only after a
manual paid-stage launch.

No model forwards have occurred on this branch yet. The expected order is:

1. `local-controls`: run tests, freeze the dataset, and run the CPU semantic
   audit;
2. `preflight`: validate Qwen3.8-27B tokenizer and one-token A/B/C/D
   continuations, then manually review the artifact; and
3. `behavior`: dispatch with the reviewed preflight run ID, revalidate its
   commit/config/dataset/model/query hashes, run the locked split's
   forced-choice next-token logits, and analyze them locally.

There is no native-reasoning, activation, or remote tool-trajectory stage in
this first screen. Raw behavior and analysis artifacts must carry content
hashes and may not be overwritten non-identically.
