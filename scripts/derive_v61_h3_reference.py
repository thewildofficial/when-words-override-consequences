#!/usr/bin/env python3
"""Derive V6.1's strict H3 endpoints from the archived raw artifact.

This is a deterministic secondary reanalysis.  It does not change the V6.1
production analysis or any V6.1 gate.  The output is a small immutable
reference artifact used to compare the same strict H3 endpoint in V6.2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from jspace_policy.v6_2_reasoning_capability import canonical_sha256

DEFAULT_OUTPUT = Path(
    "results/v6_2_reasoning_capability_control/v61_reference_endpoints.json"
)
DEFAULT_RELEASE_URL = (
    "https://github.com/thewildofficial/when-words-override-consequences/releases/"
    "download/v6.1-es2-qwen36-primary/v6.1-es2-qwen36-primary-raw.tar.gz"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _strict_pair_success(pair: list[dict[str, Any]]) -> bool:
    return (
        len(pair) == 2
        and all(row.get("selected_index") is not None for row in pair)
        and all(bool(row.get("correct")) for row in pair)
        and len({row["selected_index"] for row in pair}) == 2
    )


def _derive_endpoints(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["game_id"],
                row["prior"],
                row["message_source"],
                row["evidence_prompt_mode"],
            )
        ].append(row)

    by_mode: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for key, group in sorted(groups.items()):
        evidence = {row["evidence_mode"]: row for row in group}
        if set(evidence) != {"independent", "copied"}:
            continue
        if {row["expected_value"] for row in group} != {0, 1}:
            continue
        by_mode[key[-1]].append(
            (key[0], _strict_pair_success([evidence["independent"], evidence["copied"]]))
        )

    endpoints: dict[str, dict[str, Any]] = {}
    names = {
        "explicit_rule": "H3_explicit_both_correct_and_switch",
        "provenance_only": "H3_provenance_both_correct_and_switch",
    }
    for mode, cells in sorted(by_mode.items()):
        by_game: dict[str, list[float]] = defaultdict(list)
        for game_id, success in cells:
            by_game[game_id].append(float(success))
        game_rates = {
            game_id: sum(values) / len(values)
            for game_id, values in sorted(by_game.items())
        }
        endpoints[names[mode]] = {
            "prompt_mode": mode,
            "definition": (
                "Both independent and copied rows are correct and their selected "
                "semantic indexes differ. Mean is the unweighted mean of game-level "
                "cell rates."
            ),
            "n_pair_cells": len(cells),
            "n_games": len(game_rates),
            "successes": sum(success for _game_id, success in cells),
            "game_rates": game_rates,
            "mean": sum(game_rates.values()) / len(game_rates),
        }
    if set(endpoints) != set(names.values()):
        raise RuntimeError("archived V6.1 artifact lacks both H3 prompt modes")
    return endpoints


def derive(
    raw_path: Path,
    *,
    release_url: str,
    release_asset_sha256: str,
) -> dict[str, Any]:
    payload = _read_json(raw_path)
    if payload.get("status") != "forced_choice_logit_behavior_complete":
        raise RuntimeError("unexpected V6.1 raw artifact status")
    rows = [row for row in payload["records"] if row.get("split") == "locked"]
    h3_rows = [
        row
        for row in rows
        if row.get("experiment_family") == "evidence_update"
        and row.get("task_kind") == "report"
        and row.get("evidence_direction") == "prior"
        and row.get("evidence_count") == 4
    ]
    metadata = payload.get("metadata", {})
    body = {
        "schema_version": 1,
        "study_id": "V6.2-RCC-1",
        "reference_study_id": "V6.1-ES-2",
        "reference_model_id": "Qwen/Qwen3.6-27B",
        "reference_condition": "qwen36_direct",
        "definition": (
            "Secondary reanalysis of the V6.1 locked forced-choice artifact; "
            "this endpoint is not a replacement V6.1 preregistered gate."
        ),
        "source": {
            "release_asset_url": release_url,
            "release_asset_sha256": release_asset_sha256,
            "raw_member": "v6.1-es2-qwen36-primary/raw/behavior_primary.json",
            "raw_file_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "raw_payload_content_sha256": payload.get("content_sha256"),
            "source_config_sha256": metadata.get("config_sha256"),
            "source_dataset_sha256": metadata.get("source_dataset_sha256"),
            "locked_row_count": len(rows),
            "locked_h3_pair_row_count": len(h3_rows),
        },
        "endpoints": _derive_endpoints(h3_rows),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--release-url", default=DEFAULT_RELEASE_URL)
    parser.add_argument("--release-asset-sha256", required=True)
    args = parser.parse_args()
    artifact = derive(
        args.raw_artifact,
        release_url=args.release_url,
        release_asset_sha256=args.release_asset_sha256,
    )
    if args.output.exists():
        existing = _read_json(args.output)
        if existing != artifact:
            raise RuntimeError(f"refusing to overwrite non-identical artifact: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
