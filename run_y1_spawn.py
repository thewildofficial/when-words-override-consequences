import sys, json
from pathlib import Path
import modal
fn = modal.Function.from_name("y1-game-battery", "y1_remote")
if sys.argv[1] == "spawn":
    rows = json.load(open("y1/y1_rows.json")); print(fn.spawn(rows).object_id)
else:
    call = modal.FunctionCall.from_id(sys.argv[2])
    try: out = call.get(timeout=float(sys.argv[3]))
    except TimeoutError: print("pending"); sys.exit(0)
    Path("results/y1_game_battery/raw").mkdir(parents=True, exist_ok=True)
    Path("results/y1_game_battery/raw/y1.json").write_text(out); print("saved", len(out))
