import sys
from pathlib import Path
import modal
fn = modal.Function.from_name("jspace-gate-pilot", "pilot_remote")
if sys.argv[1] == "spawn": print(fn.spawn(96, 6).object_id)
else:
    call = modal.FunctionCall.from_id(sys.argv[2])
    try: out = call.get(timeout=float(sys.argv[3]))
    except TimeoutError: print("pending"); sys.exit(0)
    Path("results/v2_state_report_gatepilot/raw").mkdir(parents=True, exist_ok=True)
    Path("results/v2_state_report_gatepilot/raw/pilot.json").write_text(out); print("saved", len(out))
