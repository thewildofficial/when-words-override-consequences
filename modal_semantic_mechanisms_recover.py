"""Finish the one censored frozen site, then apply unchanged held-out gates."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

import modal_semantic_action_mechanisms as base
from jspace_policy.semantic_action_mechanisms import LAYERS, paired_rows
from jspace_policy.sprint_runtime import (
    digest,
    load_json_file,
    reserve,
    write_gzip_new,
    write_new,
)

SOURCE_RUN = "semantic-mechanisms-qwen36-gha-38143348343"
SOURCE_ROOT = base.ROOT / SOURCE_RUN
TIMEOUTS = {"finish": 120, "locked": 400}
CEILINGS = {s: (t + 10) * base.GPU_RATE * 1.2 for s, t in TIMEOUTS.items()}
app = modal.App("semantic-mechanisms-final-site-recovery")
image = base.gpu_image.add_local_file(
    "modal_semantic_action_mechanisms.py",
    remote_path="/root/modal_semantic_action_mechanisms.py",
)


def validate_completed(prepared, completed):
    base.verify_prepared(prepared)
    expected_sites = {
        (layer, anchor) for layer in LAYERS for anchor in ("state_end", "policy_end")
    }
    sites = {(r["layer"], r["anchor"]) for r in completed["records"]}
    if expected_sites - sites != {(56, "policy_end")} or sites - expected_sites:
        raise RuntimeError("recovery is restricted to the one censored frozen site")
    if completed["identity_error"] > 1e-5:
        raise RuntimeError("completed sites failed identity gate")
    pairs = [
        (s, r)
        for s, r in paired_rows(prepared["dataset"], "state")
        if s["target_index"] == int(s["family"].split("-")[-1]) % 2 and s["permutation"] == 0
    ]
    expected = {(s["id"], r["id"], "forward") for s, r in pairs} | {
        (r["id"], s["id"], "reverse") for s, r in pairs
    }
    for site in sites:
        rows = [r for r in completed["records"] if (r["layer"], r["anchor"]) == site]
        if (
            len(rows) != 24
            or {(r["source"], r["destination"], r["direction"]) for r in rows} != expected
        ):
            raise RuntimeError("completed checkpoint omitted or duplicated a frozen pair")
        if any(base.score_error(r["identity"], r["baseline"]) > 1e-5 for r in rows):
            raise RuntimeError("stored identity control failed")


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=65536,
    timeout=120,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/cache": base.cache, "/artifacts": base.artifacts},
)
def finish_site(prepared: dict, provenance: dict) -> str:
    return json.dumps(
        base.mechanistic_impl(prepared, "state", [(56, "policy_end")], True, provenance)
    )


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=65536,
    timeout=400,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/cache": base.cache, "/artifacts": base.artifacts},
)
def locked_recovery(prepared: dict, site: dict, provenance: dict) -> str:
    return base.locked_remote.get_raw_f()(prepared, "state", site, provenance)


@app.local_entrypoint()
def run(run_id: str):
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("invalid run ID")
    root = base.ROOT / run_id
    if root.exists():
        raise RuntimeError("refusing overwrite or automatic retry")
    prepared = load_json_file(SOURCE_ROOT / "prepared.json.gz")
    completed = load_json_file(SOURCE_ROOT / "recovered_sites.json.gz")
    protocol = load_json_file(
        "experiments/semantic_action_mechanisms/aligned/recovery_protocol.json"
    )
    if digest(completed) != protocol["completed_checkpoint_digest"]:
        raise RuntimeError("recovered checkpoint changed")
    validate_completed(prepared["discovery"], completed)
    ledger = base.ROOT / "reservations.jsonl"
    spent = sum(json.loads(line)["ceiling_usd"] for line in ledger.read_text().splitlines())
    if spent + sum(CEILINGS.values()) > 5:
        raise RuntimeError("salvage exceeds cumulative $5 reservation limit")
    provenance = {
        "run_id": run_id,
        "source_run": SOURCE_RUN,
        "protocol_sha256": digest(protocol),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    write_new(root / "protocol.json", protocol)
    write_new(root / "provenance.json", provenance)

    def dispatch(stage, function, *args):
        reservation = reserve(
            ledger, run_id + ":" + stage, "mechanistic", CEILINGS[stage], global_ceiling_usd=5
        )
        write_new(root / f"{stage}_reservation.json", reservation)
        try:
            value = json.loads(function.remote(*args))
        except Exception as exc:
            write_new(
                root / f"{stage}_failure.json",
                {"type": type(exc).__name__, "message": str(exc), "reservation_retained": True},
            )
            raise
        write_gzip_new(root / f"{stage}_raw.json.gz", value)
        write_new(
            root / f"{stage}_cost_estimate.json",
            base.estimate_cost(
                "A100-80GB", value["elapsed_seconds"], cpu_cores=8, memory_gib=64
            ).__dict__,
        )
        return value

    final = dispatch("finish", finish_site, prepared["discovery"], provenance)
    if final["status"] != "patched":
        print("Final-site numerical gate failed; no selection or confirmation.")
        return
    if len(final["records"]) != 24 or {(r["layer"], r["anchor"]) for r in final["records"]} != {
        (56, "policy_end")
    }:
        raise RuntimeError("final-site inventory mismatch")
    merged = {
        **final,
        "records": completed["records"] + final["records"],
        "identity_error": max(completed["identity_error"], final["identity_error"]),
        "recovery": {
            "completed_sites": 13,
            "source_run": SOURCE_RUN,
            "original_capture_replay_gate_passed_by_execution_path": True,
        },
    }
    write_gzip_new(root / "complete_discovery_raw.json.gz", merged)
    analysis = base.patch_analysis(merged)
    write_new(root / "discovery_analysis.json", analysis)
    write_new(
        root / "site_freeze.json", {"site": analysis["selected"], "sha256": digest(analysis)}
    )
    print(json.dumps(analysis), flush=True)
    if analysis["selected"] is None:
        print(
            "All fourteen sites complete; no qualifying patch. Held-out data remains unopened."
        )
        return
    locked = dispatch(
        "locked", locked_recovery, prepared["locked"], analysis["selected"], provenance
    )
    locked_analysis = base.patch_analysis(locked)
    write_new(root / "locked_analysis.json", locked_analysis)
    write_new(root / "locked_verdict.json", base.locked_verdict(locked, locked_analysis))
    print(json.dumps(locked_analysis), flush=True)
