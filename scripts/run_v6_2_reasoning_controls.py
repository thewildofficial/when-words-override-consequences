#!/usr/bin/env python3
"""Freeze V6.2 selection and semantic controls without model execution."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from jspace_policy.v6_2_reasoning_capability import (
    DEFAULT_CONFIG,
    build_manifest,
    canonical_sha256,
    verify_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results/v6_2_reasoning_capability_control"
DEFAULT_MANIFEST = ROOT / "configs/v6.2/reasoning_capability_control/subset_manifest.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once(path: Path, value: object, *, replace: bool = False) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if replace:
            path.write_text(serialized, encoding="utf-8")
            return
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(f"refusing to overwrite non-identical artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def _write_text_once(path: Path, value: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _artifact(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "content_sha256": canonical_sha256(body)}


def _git_sha() -> str:
    supplied = os.environ.get("GITHUB_SHA")
    if supplied:
        return supplied
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("could not determine the reviewed protocol commit") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--freeze-subsets",
        action="store_true",
        help="write the deterministic subset manifest if it is absent",
    )
    parser.add_argument(
        "--refresh-generated",
        action="store_true",
        help="replace only generated control artifacts after a code fix",
    )
    args = parser.parse_args()
    protocol_commit_sha = _git_sha()

    config = _read_json(args.config)
    if config.get("status") != "preregistered_before_qwen38_execution":
        raise RuntimeError("V6.2 config is not prospectively frozen")
    manifest = build_manifest(config)
    if args.manifest.exists():
        existing = _read_json(args.manifest)
        if args.refresh_generated:
            _write_once(args.manifest, manifest, replace=True)
        else:
            verify_manifest(existing, config)
            manifest = existing
    elif args.freeze_subsets:
        _write_once(args.manifest, manifest, replace=args.refresh_generated)
    else:
        raise RuntimeError("subset manifest is absent; pass --freeze-subsets to create it")

    results = args.results
    _write_once(results / "config.json", config, replace=args.refresh_generated)
    _write_once(results / "subset_manifest.json", manifest, replace=args.refresh_generated)
    control = _artifact(
        {
            "schema_version": 1,
            "study_id": config["study_id"],
            "status": "cpu_semantic_controls_passed_no_model_execution",
            "interpretation": "No model forwards have occurred on this branch yet.",
            "config_sha256": canonical_sha256(config),
            "subset_manifest_sha256": manifest["content_sha256"],
            "source_dataset_sha256": manifest["source"]["dataset_sha256"],
            "source_locked_rows": manifest["source"]["locked_row_count"],
            "diagnostic_rows": manifest["subsets"]["diagnostic"]["row_count"],
            "selected_families": manifest["diagnostic_families"],
            "parser_passed": manifest["parser_contract"]["passed"],
            "endpoint_coverage_passed": manifest["endpoint_coverage"]["passed"],
            "choice_mapping_pair_failures": manifest["choice_mapping_pair_failures"],
            "constant_label_pair_failures": manifest["constant_label_pair_failures"],
            "model_forwards": 0,
            "gpu_stages_ran": False,
            "base_commit": config["source"]["base_commit"],
            "head_commit": protocol_commit_sha,
            "protocol_commit_sha": protocol_commit_sha,
        }
    )
    _write_once(results / "control_audit.json", control, replace=args.refresh_generated)
    run_manifest = _artifact(
        {
            "schema_version": 1,
            "study_id": config["study_id"],
            "status": "local_cpu_controls_complete_no_model_run",
            "interpretation": "No model forwards have occurred on this branch yet.",
            "config_sha256": canonical_sha256(config),
            "source_dataset_sha256": manifest["source"]["dataset_sha256"],
            "subset_manifest_sha256": manifest["content_sha256"],
            "row_counts": {
                selection: details["row_count"]
                for selection, details in manifest["subsets"].items()
            },
            "compute": {"executor": "local_cpu", "gpu_seconds": 0, "model_forward_passes": 0},
            "gpu_stages_ran": False,
            "branch": "research/v6-2-reasoning-capability-control",
            "base_commit": config["source"]["base_commit"],
            "head_commit": protocol_commit_sha,
            "protocol_commit_sha": protocol_commit_sha,
        }
    )
    _write_once(results / "run_manifest.json", run_manifest, replace=args.refresh_generated)
    _write_text_once(results / "cost_ledger.jsonl", "")
    print(
        json.dumps(
            {
                "study_id": config["study_id"],
                "status": control["status"],
                "config_sha256": control["config_sha256"],
                "source_dataset_sha256": control["source_dataset_sha256"],
                "subset_manifest_sha256": control["subset_manifest_sha256"],
                "direct_rows": manifest["subsets"]["direct"]["row_count"],
                "pilot_rows": manifest["subsets"]["pilot"]["row_count"],
                "diagnostic_rows": manifest["subsets"]["diagnostic"]["row_count"],
                "model_forwards": 0,
                "gpu_stages_ran": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
