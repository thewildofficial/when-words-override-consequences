# HDC-1 / HDC-2: history composition and payoff ordering (exploratory)

Status: **exploratory, not preregistered, not a WWOC replication.** Run 2026-10-24 on Modal (A100-80GB). HDC-2 was chosen after seeing HDC-1. n = 24 bases per cell, so one item is 4.2 points; family-level bootstrap CIs are roughly +-18 points.

Models (revisions were NOT pinned at run time; the current Hub heads below predate the runs):
- Qwen/Qwen2.5-14B-Instruct @ cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
- microsoft/phi-4 @ 2db69c1c3e91a05d2c64a3185acfbaf36f744e25
Neither passes the repo's competence gate in the aligned/no-history assertion cell (Qwen 62.5%, phi-4 50%).

## Files
- `hdc.py`: game generator (RBG-4 strategic template with rewritten history) and HDC-1 runner. Conditions: assertion/opaque x aligned/opposed x history {none, balanced, homog_correct, homog_wrong, homog_idx0} x K {2,4,8}; 1,248 rows.
- `hdc2.py`: HDC-2 variants on opposed/aligned assertion, balanced K=4: baseline, state sentence removed, state replaced by an unrelated word, payoff restated, payoff placed before the state sentence; 240 rows.
- `analyze.py`, `feat.py`: per-cell accuracy with base-bootstrap CI; choice-feature breakdown (truthful claim, goal-naming claim, repeat of last action).
- `results/`: raw per-row outputs (`res_*.json` HDC-1, `hdc2_*.json` HDC-2).

## Reproduce
Needs a Modal token. `modal run --detach hdc.py` (writes to Modal volume `hdc1-results`; fetch with `modal volume get`), then `python analyze.py qwen25-14b phi4` and `python feat.py <model>`. Scoring is deterministic forced-choice (A/B logit at the final position, chat template, bf16, batch 24, no sampling). Run `hdc2.py` the same way (it imports `hdc`). Note a local client timeout kills non-detached runs.

## Results (see raw files; these are descriptive)
Opposed + assertion, balanced K=4 history, correct/24:
| | baseline | state removed | unrelated word | payoff restated | payoff before state |
|---|---|---|---|---|---|
| Qwen2.5-14B | 8 | 17 | 16 | 6 | 17 |
| phi-4 | 10 | 15 | 18 | 9 | 22 |

HDC-1: opaque actions near ceiling with history (Qwen 100%, phi-4 86-100% opposed); opposed assertion with history Qwen 43% pooled, phi-4 31%; for Qwen, opposed accuracy under all-wrong-action history rose 50 -> 71 -> 92% over K = 2, 4, 8, while all-correct-action history did not help. Interpretations (e.g. that the private state acts as the default goal) are hypotheses.

## Limits
Two models, one prompt skeleton, small n, unpinned revisions, failed competence gate on one cell, post-hoc selection of HDC-2. Not evidence about Qwen3.6-27B.
