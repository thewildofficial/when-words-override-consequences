import sys, json, base64
from pathlib import Path
import modal
fn = modal.Function.from_name("y1-game-battery", "y4_remote")
if sys.argv[1] == "spawn":
    print(fn.spawn(json.load(open(sys.argv[2]))).object_id)
else:
    call = modal.FunctionCall.from_id(sys.argv[2])
    try: out = json.loads(call.get(timeout=float(sys.argv[3])))
    except TimeoutError: print("pending"); sys.exit(0)
    p = Path(sys.argv[4]); p.parent.mkdir(parents=True, exist_ok=True)
    (p).write_bytes(base64.b64decode(out.pop("npz_b64"))); Path(str(p)+".meta.json").write_text(json.dumps(out)); print("saved", p.stat().st_size, out["n_layers"])
