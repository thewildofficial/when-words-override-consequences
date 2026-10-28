import sys, json, random
from pathlib import Path
sys.path.insert(0, "x1")
import modal, kuhn_ev
fn = modal.Function.from_name("kuhn-gate-pilot2", "kuhn_remote")
if sys.argv[1] == "spawn":
    ds = kuhn_ev.dataset(seed=7, n_random=40)
    rg = random.Random(1); rg.shuffle(ds); ds = ds[:480]
    rows = [{"prompt": r["prompt"], "best_label": r["best_label"], "policy": r["policy"], "card": r["card"], "obs": r["obs"], "advantage": r["advantage"], "rendering": r["rendering"], "q": r["q"], "labels": r["labels"]} for r in ds]
    print(fn.spawn(rows).object_id)
else:
    call = modal.FunctionCall.from_id(sys.argv[2])
    try: out = call.get(timeout=float(sys.argv[3]))
    except TimeoutError: print("pending"); sys.exit(0)
    Path("results/x1_kuhn_gate_pilot2/raw").mkdir(parents=True, exist_ok=True)
    Path("results/x1_kuhn_gate_pilot2/raw/pilot2.json").write_text(out); print("saved", len(out))
