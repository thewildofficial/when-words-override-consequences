#!/usr/bin/env python3
"""Materialize and audit the V6 symbolic protocol on CPU.

This command never loads a language model.  It freezes the exact synthetic
dataset, verifies all certificates and factorial controls, and writes an audit
record that can be passed unchanged to the Modal stages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jspace_policy.epistemic_strategic_experiments import (
    STUDY_ID,
    canonical_sha256,
    control_audit,
    dataset_payload,
    verify_dataset_payload,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/v6/epistemic_strategic/experiment.json"
DEFAULT_DATASET = ROOT / "configs/v6/epistemic_strategic/dataset.json"
DEFAULT_MANIFEST = ROOT / "configs/v6/epistemic_strategic/dataset_manifest.json"
DEFAULT_RESULTS = ROOT / "results/v6_epistemic_strategic"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once(path: Path, value: object) -> None:
    if path.exists():
        existing = _read_json(path)
        if existing != value:
            raise RuntimeError(f"refusing to overwrite non-identical artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _freeze_dataset(
    config: dict[str, Any], dataset_path: Path, manifest_path: Path
) -> dict[str, Any]:
    payload = dataset_payload(config)
    verify_dataset_payload(payload, config)
    _write_once(dataset_path, payload)
    manifest = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "frozen_after_cpu_control_audit_before_model_execution",
        "config_sha256": canonical_sha256(config),
        "dataset_sha256": payload["content_sha256"],
        "row_count": len(payload["rows"]),
    }
    _write_once(manifest_path, manifest)
    return payload


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
    args = parser.parse_args()

    config = _read_json(args.config)
    generated = dataset_payload(config)
    verify_dataset_payload(generated, config)
    if args.freeze_dataset:
        dataset = _freeze_dataset(config, args.dataset, args.manifest)
    elif args.dataset.exists():
        dataset = _read_json(args.dataset)
        verify_dataset_payload(dataset, config)
    else:
        dataset = generated
    if dataset["content_sha256"] != generated["content_sha256"]:
        raise RuntimeError("frozen dataset differs from deterministic local generation")

    audit = control_audit(dataset, config)
    audit_path = args.results / "control_audit.json"
    _write_once(audit_path, audit)
    config_sha256 = canonical_sha256(config)
    run_manifest = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "local_cpu_controls_complete_no_model_run",
        "config_sha256": config_sha256,
        "dataset_sha256": dataset["content_sha256"],
        "audit_sha256": canonical_sha256(audit),
        "row_count": len(dataset["rows"]),
        "compute": {"executor": "local_cpu", "gpu_seconds": 0, "model_forward_passes": 0},
    }
    _write_once(args.results / "run_manifest.json", run_manifest)
    print(
        json.dumps(
            {
                "study_id": STUDY_ID,
                "config_sha256": config_sha256,
                "dataset_sha256": dataset["content_sha256"],
                "rows": len(dataset["rows"]),
                "status": audit["status"],
                "results": str(args.results),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
