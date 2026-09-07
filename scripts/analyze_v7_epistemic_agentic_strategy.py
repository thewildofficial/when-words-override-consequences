#!/usr/bin/env python3
"""Analyze the locked direct-logit V7-EAS-1 artifact.

The analyzer is deliberately family-shaped.  It never pools the study into a
single agent score and it never selects a favorable endpoint after seeing the
model output.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jspace_policy.v7_epistemic_agentic_strategy import (
    FAMILIES,
    STUDY_ID,
    SURFACES,
    canonical_sha256,
    control_audit,
    verify_dataset_payload,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/v7/epistemic_agentic_strategy/experiment.json"
DEFAULT_DATASET = ROOT / "configs/v7/epistemic_agentic_strategy/dataset.json"
DEFAULT_RESULTS = ROOT / "results/v7_epistemic_agentic_strategy"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once(path: Path, value: object) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(f"refusing to overwrite non-identical artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def _hash_valid(payload: dict[str, Any]) -> bool:
    claimed = payload.get("content_sha256")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    return claimed == canonical_sha256(body)


def _group(
    rows: list[dict[str, Any]], *keys: str
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def _rate(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> float:
    return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else 0.0


def _bootstrap(values: list[float], *, seed: int, draws: int) -> dict[str, Any]:
    if not values:
        return {"n_clusters": 0, "mean": None, "ci95": [None, None]}
    rng = random.Random(seed)
    sampled = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(draws)
    ]
    sampled.sort()
    return {
        "n_clusters": len(values),
        "mean": sum(values) / len(values),
        "ci95": [sampled[int(0.025 * (draws - 1))], sampled[int(0.975 * (draws - 1))]],
    }


def _cluster_rate(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    values = [_rate(group, predicate) for group in _group(rows, "game_id").values()]
    return _bootstrap(values, seed=seed, draws=draws)


def _cluster_mean(
    rows: list[dict[str, Any]],
    field: str,
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    values = []
    for group in _group(rows, "game_id").values():
        observed = [float(row[field]) for row in group if row.get(field) is not None]
        if observed:
            values.append(sum(observed) / len(observed))
    return _bootstrap(values, seed=seed, draws=draws)


def _pair_success(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("selected_index") is not None
        and right.get("selected_index") is not None
        and bool(left.get("correct"))
        and bool(right.get("correct"))
        and left["selected_index"] != right["selected_index"]
    )


def _pair_invariant(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("selected_index") is not None
        and right.get("selected_index") is not None
        and bool(left.get("correct"))
        and bool(right.get("correct"))
        and left["selected_index"] == right["selected_index"]
    )


def _matched_pair_rate(
    rows: list[dict[str, Any]],
    factor: str,
    left_value: str,
    right_value: str,
    *,
    seed: int,
    draws: int,
    predicate: Callable[[dict[str, Any], dict[str, Any]], bool] = _pair_success,
) -> dict[str, Any]:
    cells_by_game: dict[str, list[float]] = defaultdict(list)
    for group in _group(rows, "game_id", "surface").values():
        by_value = {row["condition_factors"][factor]: row for row in group}
        if left_value in by_value and right_value in by_value:
            cells_by_game[group[0]["game_id"]].append(
                float(predicate(by_value[left_value], by_value[right_value]))
            )
    clusters = [sum(values) / len(values) for values in cells_by_game.values()]
    return {
        **_bootstrap(clusters, seed=seed, draws=draws),
        "n_cells": sum(map(len, cells_by_game.values())),
    }


def _surface_invariance(rows: list[dict[str, Any]], *, seed: int, draws: int) -> dict[str, Any]:
    cells_by_game: dict[str, list[float]] = defaultdict(list)
    for game_group in _group(rows, "game_id").values():
        condition_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in game_group:
            key = tuple(
                sorted(
                    (name, value)
                    for name, value in row["condition_factors"].items()
                    if name != "surface"
                )
            )
            condition_groups[key].append(row)
        for condition_rows in condition_groups.values():
            by_surface = {row["surface"]: row for row in condition_rows}
            if set(by_surface) == set(SURFACES):
                cells_by_game[game_group[0]["game_id"]].append(
                    float(
                        all(bool(row.get("correct")) for row in by_surface.values())
                        and len({row.get("selected_index") for row in by_surface.values()}) == 1
                    )
                )
    clusters = [sum(values) / len(values) for values in cells_by_game.values()]
    return {
        **_bootstrap(clusters, seed=seed, draws=draws),
        "n_cells": sum(map(len, cells_by_game.values())),
    }


def _passes(metric: dict[str, Any], threshold: float) -> bool:
    return metric.get("mean") is not None and metric["mean"] >= threshold


def _validate_records(
    payload: dict[str, Any], dataset: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    metadata = payload.get("metadata", {})
    if metadata.get("config_sha256") != canonical_sha256(config):
        raise RuntimeError("model artifact was produced from a different config")
    if metadata.get("dataset_sha256") != dataset["content_sha256"]:
        raise RuntimeError("model artifact was produced from a different dataset")
    expected_rows = [row for row in dataset["rows"] if row["split"] == "locked"]
    expected_by_id = {row["condition_id"]: row for row in expected_rows}
    records = payload.get("records", [])
    if {record.get("condition_id") for record in records} != set(expected_by_id):
        raise RuntimeError("model artifact does not contain exactly the locked rows")
    for record in records:
        row = expected_by_id[record["condition_id"]]
        for key in ("expected", "expected_index", "matched_group_id", "choice_mapping"):
            expected = row["expected_choice"] if key == "expected" else row[key]
            if record.get(key) != expected:
                raise RuntimeError(f"{key} changed for {record['condition_id']}")
        selected = record.get("selected")
        selected_index = row["choice_mapping"].get(selected) if selected else None
        if record.get("selected_index") != selected_index:
            raise RuntimeError(f"selected index mismatch for {record['condition_id']}")
        if record.get("correct") != (selected == row["expected_choice"]):
            raise RuntimeError(f"correctness mismatch for {record['condition_id']}")
        if record.get("condition_factors") != row["condition_factors"]:
            raise RuntimeError(f"condition changed for {record['condition_id']}")
    return records


def _report_metrics(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    draws = int(config["statistics"]["bootstrap_draws"])
    report_rows = [row for row in rows if row["task_kind"] == "report"]
    by_target = {
        target: _cluster_rate(
            [row for row in report_rows if row.get("report_target") == target],
            lambda row: row["correct"],
            seed=70100 + index,
            draws=draws,
        )
        for index, target in enumerate(
            sorted({row.get("report_target") for row in report_rows})
        )
    }
    return {
        "all_reports": _cluster_rate(
            report_rows, lambda row: row["correct"], seed=70001, draws=draws
        ),
        "by_target": by_target,
        "predicted_opponent_action": by_target.get("predicted_opponent_action"),
        "final_strategy": by_target.get("final_strategy"),
    }


def _family_common(
    rows: list[dict[str, Any]], config: dict[str, Any], seed: int
) -> dict[str, Any]:
    draws = int(config["statistics"]["bootstrap_draws"])
    actions = [row for row in rows if row["task_kind"] == "action"]
    reports = _report_metrics(rows, config)
    action_accuracy = _cluster_rate(
        actions, lambda row: row["correct"], seed=seed + 2, draws=draws
    )
    surface_invariance = _surface_invariance(actions, seed=seed + 3, draws=draws)
    regrets = _cluster_mean(actions, "regret", seed=seed + 1, draws=draws)
    return {
        "report_competence": reports,
        "predicted_opponent_action_competence": reports["predicted_opponent_action"],
        "final_action_accuracy": action_accuracy,
        "mean_regret": regrets,
        "surface_name_invariance": surface_invariance,
        "trajectory_success": {
            "status": "not_collected_in_direct_screen",
            "rate": None,
        },
        "gates": {
            "report_accuracy": _passes(
                reports["all_reports"], config["gates"]["minimum_locked_report_accuracy"]
            ),
            "action_accuracy": _passes(
                action_accuracy,
                config["gates"]["minimum_locked_action_accuracy"],
            ),
            "surface_invariance": _passes(
                surface_invariance,
                config["gates"]["minimum_surface_invariance"],
            ),
        },
    }


def _publicity_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    result = _family_common(rows, config, 71000)
    actions = [row for row in rows if row["task_kind"] == "action"]
    pairs = {
        "shared_private_to_public_announcement": _matched_pair_rate(
            actions,
            "epistemic_level",
            "shared_private",
            "public_announcement",
            seed=71010,
            draws=5000,
        ),
        "shared_private_to_common_knowledge": _matched_pair_rate(
            actions,
            "epistemic_level",
            "shared_private",
            "common_knowledge",
            seed=71011,
            draws=5000,
        ),
        "public_announcement_to_common_knowledge_invariance": _matched_pair_rate(
            actions,
            "epistemic_level",
            "public_announcement",
            "common_knowledge",
            seed=71012,
            draws=5000,
            predicate=_pair_invariant,
        ),
    }
    result["identifying_pairs"] = pairs
    result["gates"] = {
        **result["gates"],
        "publicity_pair_switches": all(
            (pairs[name]["mean"] or 0.0)
            >= config["gates"]["minimum_publicity_pair_switch_rate"]
            for name in (
                "shared_private_to_public_announcement",
                "shared_private_to_common_knowledge",
            )
        ),
        "public_common_invariance": (
            pairs["public_announcement_to_common_knowledge_invariance"]["mean"] or 0.0
        )
        >= config["gates"]["minimum_publicity_public_common_invariance"],
    }
    result["status"] = "supported" if all(result["gates"].values()) else "falsified"
    return result


def _commitment_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    result = _family_common(rows, config, 72000)
    actions = [row for row in rows if row["task_kind"] == "action"]
    pairs = {
        "public_to_private_switch": _matched_pair_rate(
            actions,
            "commitment_modality",
            "public_irreversible",
            "private_irreversible",
            seed=72010,
            draws=5000,
        ),
        "public_to_promise_switch": _matched_pair_rate(
            actions,
            "commitment_modality",
            "public_irreversible",
            "cheap_promise",
            seed=72011,
            draws=5000,
        ),
        "public_to_tool_invariance": _matched_pair_rate(
            actions,
            "commitment_modality",
            "public_irreversible",
            "public_tool",
            seed=72012,
            draws=5000,
            predicate=_pair_invariant,
        ),
    }
    result["identifying_pairs"] = pairs
    result["gates"] = {
        **result["gates"],
        "public_private_switch": (pairs["public_to_private_switch"]["mean"] or 0.0)
        >= config["gates"]["minimum_commitment_public_private_switch_rate"],
        "public_promise_switch": (pairs["public_to_promise_switch"]["mean"] or 0.0)
        >= config["gates"]["minimum_commitment_public_promise_switch_rate"],
        "public_tool_invariance": (pairs["public_to_tool_invariance"]["mean"] or 0.0)
        >= config["gates"]["minimum_commitment_public_tool_invariance"],
    }
    result["status"] = "supported" if all(result["gates"].values()) else "falsified"
    return result


def _acquisition_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    result = _family_common(rows, config, 73000)
    actions = [row for row in rows if row["task_kind"] == "action"]
    by_uncertainty = {
        uncertainty: _cluster_rate(
            [
                row
                for row in actions
                if row["condition_factors"]["uncertainty_type"] == uncertainty
            ],
            lambda row: row["correct"],
            seed=73010 + index,
            draws=5000,
        )
        for index, uncertainty in enumerate(
            sorted({row["condition_factors"]["uncertainty_type"] for row in actions})
        )
    }
    target_indices = {
        "world_state": 1,
        "world_state_known": 0,
        "opponent_belief": 2,
        "opponent_belief_known": 0,
        "opponent_policy": 3,
        "opponent_policy_known": 0,
    }
    targeted_choice = _cluster_rate(
        actions,
        lambda row: (
            row.get("selected_index")
            == target_indices[row["condition_factors"]["uncertainty_type"]]
        ),
        seed=73020,
        draws=5000,
    )
    pairs = {
        "world_to_world_known_switch": _matched_pair_rate(
            actions,
            "uncertainty_type",
            "world_state",
            "world_state_known",
            seed=73030,
            draws=5000,
        ),
        "opponent_belief_to_belief_known_switch": _matched_pair_rate(
            actions,
            "uncertainty_type",
            "opponent_belief",
            "opponent_belief_known",
            seed=73031,
            draws=5000,
        ),
        "opponent_policy_to_policy_known_switch": _matched_pair_rate(
            actions,
            "uncertainty_type",
            "opponent_policy",
            "opponent_policy_known",
            seed=73032,
            draws=5000,
        ),
    }
    result.update(
        {
            "accuracy_by_uncertainty": by_uncertainty,
            "targeted_choice_rate": targeted_choice,
            "known_target_act_rate": _cluster_rate(
                [
                    row
                    for row in actions
                    if row["condition_factors"]["uncertainty_type"].endswith("_known")
                ],
                lambda row: row.get("selected_index") == 0,
                seed=73021,
                draws=5000,
            ),
            "identifying_pairs": pairs,
        }
    )
    result["gates"] = {
        **result["gates"],
        "targeted_choice": (targeted_choice["mean"] or 0.0)
        >= config["gates"]["minimum_acquisition_targeted_choice_rate"],
        "known_target_act": (result["known_target_act_rate"]["mean"] or 0.0)
        >= config["gates"]["minimum_acquisition_known_target_act_rate"],
        "world_pair_switch": (pairs["world_to_world_known_switch"]["mean"] or 0.0)
        >= config["gates"]["minimum_acquisition_world_pair_switch_rate"],
        "belief_pair_switch": (
            pairs["opponent_belief_to_belief_known_switch"]["mean"] or 0.0
        )
        >= config["gates"]["minimum_acquisition_belief_pair_switch_rate"],
        "policy_pair_switch": (
            pairs["opponent_policy_to_policy_known_switch"]["mean"] or 0.0
        )
        >= config["gates"]["minimum_acquisition_policy_pair_switch_rate"],
    }
    result["status"] = "supported" if all(result["gates"].values()) else "falsified"
    return result


def analyze(
    config_path: Path = DEFAULT_CONFIG,
    dataset_path: Path = DEFAULT_DATASET,
    behavior_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    config = _read(config_path)
    dataset = _read(dataset_path)
    verify_dataset_payload(dataset, config)
    audit = control_audit(dataset, config)
    if not audit["passed"]:
        raise RuntimeError(f"dataset audit failed: {audit['failures'][:3]}")
    behavior_path = behavior_path or DEFAULT_RESULTS / "raw/behavior.json"
    payload = _read(behavior_path)
    if not _hash_valid(payload):
        raise RuntimeError("invalid behavior artifact content hash")
    if payload.get("study_id") != STUDY_ID:
        raise RuntimeError("wrong study ID in behavior artifact")
    records = _validate_records(payload, dataset, config)
    locked = [row for row in records if row["split"] == "locked"]
    family_rows = {
        family: [row for row in locked if row["experiment_family"] == family]
        for family in FAMILIES
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "analysis_complete",
        "created_at": datetime.now(UTC).isoformat(),
        "source_behavior_sha256": payload["content_sha256"],
        "config_sha256": canonical_sha256(config),
        "dataset_sha256": dataset["content_sha256"],
        "locked_record_count": len(locked),
        "families": {
            "publicity_threshold": _publicity_summary(
                family_rows["publicity_threshold"], config
            ),
            "credible_commitment": _commitment_summary(
                family_rows["credible_commitment"], config
            ),
            "information_acquisition": _acquisition_summary(
                family_rows["information_acquisition"], config
            ),
        },
        "trajectory_success": {
            "status": "not_collected_in_direct_screen",
            "rate": None,
        },
    }
    result["content_sha256"] = canonical_sha256(result)
    if output_path is not None:
        _write_once(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("behavior", type=Path, nargs="?", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = analyze(args.config, args.dataset, args.behavior, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
