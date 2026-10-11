"""Token-aligned state-only follow-up, sharing the validated causal implementation."""

from __future__ import annotations

import json
import subprocess
from itertools import product
from pathlib import Path

import modal
import numpy as np

import modal_semantic_action_mechanisms as base
from jspace_policy.semantic_action_mechanisms import make_row, queries_for, summarize_contrast
from jspace_policy.sprint_runtime import (
    digest,
    load_json_file,
    reserve,
    write_gzip_new,
    write_new,
)

STUDY = "semantic_action_mechanisms_aligned"
DIRECTORY = Path("experiments/semantic_action_mechanisms/aligned")
TIMEOUTS = {"behavior": 400, "discovery": 400, "locked": 500}
CEILINGS = {stage: (seconds + 10) * base.GPU_RATE * 1.2 for stage, seconds in TIMEOUTS.items()}
app = modal.App("semantic-mechanisms-token-aligned")
image = base.gpu_image.add_local_file(
    "modal_semantic_action_mechanisms.py",
    remote_path="/root/modal_semantic_action_mechanisms.py",
)


def dataset_for(split, pairs):
    rows = [
        make_row(
            split,
            family,
            target,
            permutation,
            surface,
            "redundant",
            state,
            "prose",
            False,
            concepts_override=tuple(pairs[family]),
        )
        for family, (target, permutation, surface, state) in product(
            range(len(pairs)),
            product(range(2), range(2), ("assertion", "opaque"), ("matching", "opposed")),
        )
    ]
    body = {"study_id": STUDY, "split": split, "rows": rows}
    return {**body, "sha256": digest(body)}


def aligned_analysis(prepared, result):
    if result["status"] != "scored":
        raise RuntimeError("scoring instrument failed")
    rows, scores = prepared["dataset"]["rows"], result["scores"]
    if set(scores) != {q["id"] for r in rows for q in queries_for(r)}:
        raise RuntimeError("incomplete scores")
    contrasts = {}
    cells = {}
    for surface in ("assertion", "opaque"):
        contrasts[surface] = summarize_contrast(
            rows,
            scores,
            lambda r, surface=surface: r["surface"] == surface and r["state"] == "matching",
            lambda r, surface=surface: r["surface"] == surface and r["state"] == "opposed",
        )
        for state in ("matching", "opposed"):
            cell = [r for r in rows if r["surface"] == surface and r["state"] == state]
            cells[f"{surface}/{state}"] = {
                "rows": len(cell),
                "accuracy": float(
                    np.mean(
                        [
                            scores[r["id"] + ":action"]["choice"] == r["expected_action"]
                            for r in cell
                        ]
                    )
                ),
            }
    reports = [
        scores[q["id"]]["choice"] == q["expected"] for r in rows for q in queries_for(r)[1:]
    ]
    interaction = (
        contrasts["assertion"]["accuracy_difference"]
        - contrasts["opaque"]["accuracy_difference"]
    )
    gate = (
        cells["opaque/matching"]["accuracy"] >= 0.95
        and cells["opaque/opposed"]["accuracy"] >= 0.95
        and np.mean(reports) >= 0.90
        and interaction >= 0.15
        and contrasts["assertion"]["margin_difference"] >= 0.5
    )
    return {
        "status": "exploratory_token_aligned",
        "cells": cells,
        "contrasts": contrasts,
        "state_surface_accuracy_interaction": interaction,
        "report_accuracy": float(np.mean(reports)),
        "causal_gate": bool(gate),
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=65536,
    timeout=400,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/cache": base.cache},
)
def behavior_aligned(prepared: dict, provenance: dict) -> str:
    return base.behavior_remote.get_raw_f()(prepared, provenance)


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
def discovery_aligned(prepared: dict, provenance: dict) -> str:
    return base.discovery_remote.get_raw_f()(prepared, "state", provenance)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=65536,
    timeout=500,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/cache": base.cache, "/artifacts": base.artifacts},
)
def locked_aligned(prepared: dict, site: dict, provenance: dict) -> str:
    return base.locked_remote.get_raw_f()(prepared, "state", site, provenance)


@app.local_entrypoint()
def run(run_id: str):
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("invalid run_id")
    protocol = load_json_file(DIRECTORY / "protocol.json")
    prepared = load_json_file(DIRECTORY / "prepared.json.gz")
    root = base.ROOT / run_id
    if root.exists():
        raise RuntimeError("run directory exists; overwrite/retry refused")
    for split, value in prepared.items():
        base.verify_prepared(value)
        assert value["sha256"] == protocol["prepared_sha256"][split]
    ledger = base.ROOT / "reservations.jsonl"
    spent = sum(json.loads(line)["ceiling_usd"] for line in ledger.read_text().splitlines())
    if spent + sum(CEILINGS.values()) > 5:
        raise RuntimeError("follow-up cannot fit the cumulative $5 reservation ceiling")
    provenance = {
        "run_id": run_id,
        "protocol_sha256": digest(protocol),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    write_new(root / "protocol.json", protocol)
    write_new(root / "provenance.json", provenance)
    write_gzip_new(root / "prepared.json.gz", prepared)

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

    behavior = dispatch("behavior", behavior_aligned, prepared["discovery"], provenance)
    if behavior["status"] != "scored":
        return
    analysis = aligned_analysis(prepared["discovery"], behavior)
    write_new(root / "behavior_analysis.json", analysis)
    print(json.dumps(analysis), flush=True)
    if not analysis["causal_gate"]:
        print("State contrast failed; causal stages withheld.")
        return
    discovery = dispatch("discovery", discovery_aligned, prepared["discovery"], provenance)
    summary = base.patch_analysis(discovery)
    write_new(root / "discovery_analysis.json", summary)
    write_new(
        root / "site_freeze.json", {"selected": summary["selected"], "sha256": digest(summary)}
    )
    if summary["selected"] is None:
        print("No qualifying patch; held-out stage withheld.")
        return
    locked = dispatch(
        "locked", locked_aligned, prepared["locked"], summary["selected"], provenance
    )
    locked_summary = base.patch_analysis(locked)
    write_new(root / "locked_analysis.json", locked_summary)
    write_new(root / "locked_verdict.json", base.locked_verdict(locked, locked_summary))
    print(json.dumps(locked_summary), flush=True)
