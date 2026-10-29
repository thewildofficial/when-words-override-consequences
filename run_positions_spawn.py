"""Spawn deployed per-position Stage 1 readout run; collect with: collect <call_id> <timeout>."""
import json, subprocess, sys
from pathlib import Path
import modal
fn = modal.Function.from_name("jspace-v2-stage1-positions", "positions_remote")
if sys.argv[1] == "spawn":
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    call = fn.spawn(json.loads(Path("configs/v2/stage1_dataset.json").read_text()), json.loads(Path("configs/v2/stage1.json").read_text()), commit)
    print(call.object_id)
else:
    call = modal.FunctionCall.from_id(sys.argv[2])
    try:
        out = call.get(timeout=float(sys.argv[3]))
    except TimeoutError:
        print("pending"); sys.exit(0)
    Path("results/v2_stage1_positions/raw").mkdir(parents=True, exist_ok=True)
    Path("results/v2_stage1_positions/raw/positions_raw.json").write_text(out)
    print("saved", len(out))
