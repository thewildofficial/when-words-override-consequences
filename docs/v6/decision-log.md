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

## 2026-09-07 — Qwen3.8 preflight infrastructure stop

- Attempted the registered Qwen3.8 tokenizer parity preflight through GitHub
  Actions/Modal run `34074486263`.
- The run stopped before any prompt or token parity check because the Modal
  image lacked `torchvision` and Pillow, which the model's multimodal
  processor imports.
- This is recorded as an infrastructure failure, not as a Qwen3.8 parity or
  model finding. The image now declares those dependencies; the preflight must
  be rerun before any Qwen3.8 paid stage can be considered.

## 2026-09-07 — Qwen3.8 exact parity stop

- The corrected preflight completed the processor load and one-token checks,
  then failed the exact cross-model contract in run `34074986246`.
- Qwen3.8 matched Qwen3.6 on candidate labels and candidate token IDs, but did
  not match rendered chat prompts or prompt-token IDs.
- Because that framing/tokenization difference is a replication confound, no
  Qwen3.8 paid stage is authorized. A future Qwen3.8 study needs its own
  preregistered prompt protocol and separate gates rather than a pooled result.

## 2026-09-07 — primary Qwen3.6 execution

- Tokenizer preflight, black-box generation, forced-choice logits, and matched
  trajectory controls completed through GitHub Actions/Modal.
- The forced-choice family gate passed only for `provenance`; `core_tom`,
  `higher_order`, and `strategy_pressure` failed their preregistered gates.
- Activation discovery was therefore not run. The negative and heterogeneous
  result is retained in `results/v6_epistemic_strategic/raw/` with its measured
  cost ledger; no causal claim is made.
