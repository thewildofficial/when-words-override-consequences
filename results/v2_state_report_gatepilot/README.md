# Behavior-only format pilot (no hidden-state capture), Qwen3.6-27B, no CoT
Scripts: modal_gate_pilot.py, modal_gate_pilot2.py (spawn: run_gate_pilot*_spawn.py). Raw: gatepilot/raw/pilot.json, gatepilot2/raw/pilot2.json. Seed 777, 96 prompts per format, candidate-logit argmax over the 4 label tokens. Est. cost ~$0.2 for both (A100-80GB, ~98 s + ~78 s in function; public rates).

Scoring checks (done because the first run's pick==offset looked suspicious): in the pilot, the top-1 token is a candidate word on 97-100% of prompts, greedy generated outputs match the candidate-logit pick on the 12 inspected, and the F4 identity format (report = label at clue index) scores 100% (96/96). So tokenization, candidate order and the evaluator are working. The earlier failure is task competence.

Gate accuracy (report index), 96 each:
- F1 original (sum of two clues mod 4, then add offset): 0.396 (by s: 0.42/0.37/0.48/0.32)
- F2 clue index + offset mod 4: 0.479 (s=0 1.00, s=1 0.43, s=2 0.18, s=3 0.21)
- F3 sum only: 0.594
- F4 identity: 1.000
- F5 shift through list: 0.542 (s=0 1.00, then 0.43/0.29/0.32)
- F6 per-prompt report table (index->index), state = clue digit: 0.990
- F7 two lookups (clue->private, private->report): 0.365
Reading: without CoT the model does not do offset or sum arithmetic here but follows one explicit lookup table. F6 passes the 85% gate. The pilot used its own seed and is adaptation data, not a prospective test. Caveat on F6: the state is the clue digit written in the prompt (the label for that index is not written beside it), so "derived" is weak; the report index is only available through the table lookup. K/Q separation holds only on prompts where the table moves the index.
