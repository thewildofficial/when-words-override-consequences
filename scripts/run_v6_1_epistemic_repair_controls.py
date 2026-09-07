#!/usr/bin/env python3
"""Freeze and audit the V6.1 protocol without loading a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jspace_policy.v6_1_epistemic_repair import (
    STUDY_ID,
    canonical_sha256,
    control_audit,
    dataset_payload,
    verify_dataset_payload,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/v6.1/epistemic_repair/experiment.json"
DEFAULT_DATASET = ROOT / "configs/v6.1/epistemic_repair/dataset.json"
DEFAULT_MANIFEST = ROOT / "configs/v6.1/epistemic_repair/dataset_manifest.json"
DEFAULT_RESULTS = ROOT / "results/v6_1_epistemic_repair"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--freeze-dataset",
        action="store_true",
        help="write the deterministic dataset and manifest if they are absent",
    )
    parser.add_argument(
        "--refresh-generated",
        action="store_true",
        help="replace only the uncommitted generated freeze/audit artifacts after a code fix",
    )
    args = parser.parse_args()

    config = _read_json(args.config)
    generated = dataset_payload(config)
    verify_dataset_payload(generated, config)
    if args.freeze_dataset:
        _write_once(args.dataset, generated, replace=args.refresh_generated)
        _write_once(
            args.manifest,
            {
                "schema_version": 1,
                "study_id": STUDY_ID,
                "status": "frozen_after_cpu_semantic_audit_before_model_execution",
                "config_sha256": canonical_sha256(config),
                "dataset_sha256": generated["content_sha256"],
                "row_count": len(generated["rows"]),
            },
            replace=args.refresh_generated,
        )
        dataset = generated
    elif args.dataset.exists():
        dataset = _read_json(args.dataset)
        verify_dataset_payload(dataset, config)
    else:
        dataset = generated

    if dataset["content_sha256"] != generated["content_sha256"]:
        raise RuntimeError("frozen dataset differs from deterministic local generation")
    audit = control_audit(dataset, config)
    _write_once(args.results / "control_audit.json", audit, replace=args.refresh_generated)
    run_manifest = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "local_cpu_controls_complete_no_model_run",
        "config_sha256": canonical_sha256(config),
        "dataset_sha256": dataset["content_sha256"],
        "audit_sha256": canonical_sha256(audit),
        "row_count": len(dataset["rows"]),
        "compute": {"executor": "local_cpu", "gpu_seconds": 0, "model_forward_passes": 0},
    }
    _write_once(
        args.results / "run_manifest.json", run_manifest, replace=args.refresh_generated
    )
    print(
        json.dumps(
            {
                "study_id": STUDY_ID,
                "config_sha256": run_manifest["config_sha256"],
                "dataset_sha256": run_manifest["dataset_sha256"],
                "rows": len(dataset["rows"]),
                "status": audit["status"],
                "all_family_gates": audit["gates"]["all_family_gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
