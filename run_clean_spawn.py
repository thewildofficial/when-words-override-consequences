"""Spawn the deployed KL-match run and collect (survives client exit when re-run with --collect)."""
import json, subprocess, sys
from pathlib import Path
import modal

country = next(i for i in json.loads(Path("configs/v2/flexible_generalization_smoke.json").read_text())["data"]["categories"] if i["name"] == "countries")
fn = modal.Function.from_name("jspace-h0r-clean-dep", "run_clean_remote")
if sys.argv[1] == "spawn":
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    call = fn.spawn(country, commit)
    print(call.object_id)
else:
    call = modal.FunctionCall.from_id(sys.argv[2])
    try:
        out = call.get(timeout=float(sys.argv[3]))
    except TimeoutError:
        print("pending"); sys.exit(0)
    Path("results/v2_jlens_recon/raw").mkdir(parents=True, exist_ok=True)
    Path("results/v2_jlens_recon/raw/clean_raw.json").write_text(out)
    print("saved", len(out))
