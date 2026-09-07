# V6 decision log

Append-only. New decisions are added at the bottom with a date and a reason.

## 2026-09-07 — V6-ES-1 protocol freeze

- Created branch `research/epistemic-strategic-experiments` from the clean
  pruning commit.
- Chose `V6-ES-1` and the `v6_epistemic_strategic` path namespace so future
  runs do not collide with V4/V5 artifacts.
- Kept Qwen3.6-27B as the primary model because it is the repository's pinned
  baseline. Registered Qwen3.8-27B as a separate pinned, parity-locked
  replication after the earlier rename/parity failure.
- Separated black-box generation, forced-choice logit behavior, observational
  activation probes, and causal activation interchange.
- Froze four family gates instead of a pooled gate: `core_tom`, `higher_order`,
  `provenance`, and `strategy_pressure`.
- Moved all dataset generation, certificates, grouping, and prompt checks to
  CPU. Modal is reserved for tokenizer preflight and model forward passes.
- Sealed the causal activation stage until a new natural counterfactual patch
  protocol and retention controls are frozen.

## 2026-09-07 — local CPU audit

- Generated and verified 1,740 rows.
- Dataset SHA256: `e5192e32126fcdfe5a606708a98a8264753092871bce3f427de8d5254d681995`.
- Config SHA256: `8eed1a0c09ad37f329137e119f6ac7a973a86592a618b1de074d1d7553cb3706`.
- All structural family gates passed.
- No model forward pass or GPU allocation occurred.
