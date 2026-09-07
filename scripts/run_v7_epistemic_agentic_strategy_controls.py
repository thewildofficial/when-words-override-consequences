#!/usr/bin/env python3
"""Freeze and audit V7-EAS-1 without loading a model or using Modal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jspace_policy.budget import estimate_cost
from jspace_policy.v7_epistemic_agentic_strategy import (
    STUDY_ID,
    canonical_sha256,
    control_audit,
    dataset_payload,
    verify_dataset_payload,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/v7/epistemic_agentic_strategy/experiment.json"
DEFAULT_DATASET = ROOT / "configs/v7/epistemic_agentic_strategy/dataset.json"
DEFAULT_MANIFEST = ROOT / "configs/v7/epistemic_agentic_strategy/dataset_manifest.json"
DEFAULT_RESULTS = ROOT / "results/v7_epistemic_agentic_strategy"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once(path: Path, value: object, *, replace: bool = False) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and not replace:
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
    parser.add_argument("--freeze-dataset", action="store_true")
    parser.add_argument("--refresh-generated", action="store_true")
    args = parser.parse_args()

    config = _read(args.config)
    generated = dataset_payload(config)
    verify_dataset_payload(generated, config)
    if args.freeze_dataset:
        _write_once(args.dataset, generated, replace=args.refresh_generated)
        dataset = generated
    elif args.dataset.exists():
        dataset = _read(args.dataset)
        verify_dataset_payload(dataset, config)
    else:
        dataset = generated
    if dataset["content_sha256"] != generated["content_sha256"]:
        raise RuntimeError("frozen dataset differs from deterministic local generation")

    audit = control_audit(dataset, config)
    if not audit["passed"]:
        raise RuntimeError(json.dumps(audit, indent=2, sort_keys=True))
    config_sha256 = canonical_sha256(config)
    estimate = estimate_cost(
        config["execution"]["gpu"],
        config["execution"]["estimated_ceiling_seconds"],
        cpu_cores=8,
        memory_gib=32,
    )
    _write_once(
        args.manifest,
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "status": "frozen_after_cpu_semantic_audit_before_model_execution",
            "config_sha256": config_sha256,
            "dataset_sha256": dataset["content_sha256"],
            "row_count": len(dataset["rows"]),
            "locked_row_count": sum(row["split"] == "locked" for row in dataset["rows"]),
        },
        replace=args.refresh_generated,
    )
    _write_once(args.results / "control_audit.json", audit, replace=args.refresh_generated)
    _write_once(
        args.results / "run_manifest.json",
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "status": "local_cpu_controls_complete_no_model_run",
            "config_sha256": config_sha256,
            "dataset_sha256": dataset["content_sha256"],
            "audit_sha256": canonical_sha256(audit),
            "row_count": len(dataset["rows"]),
            "model": config["model"],
            "compute": {
                "executor": "local_cpu",
                "gpu_seconds": 0,
                "model_forward_passes": 0,
                "gpu_actions_run": False,
            },
            "budget": {
                "modal_timeout_seconds": config["execution"]["modal_timeout_seconds"],
                "hard_cost_limit_usd": config["execution"]["hard_cost_limit_usd"],
                "buffered_ceiling_usd": estimate.buffered_usd,
            },
        },
        replace=args.refresh_generated,
    )
    print(
        json.dumps(
            {
                "study_id": STUDY_ID,
                "config_sha256": config_sha256,
                "dataset_sha256": dataset["content_sha256"],
                "rows": len(dataset["rows"]),
                "locked_rows": sum(row["split"] == "locked" for row in dataset["rows"]),
                "buffered_ceiling_usd": estimate.buffered_usd,
                "status": audit["status"],
                "gpu_actions_run": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
