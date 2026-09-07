#!/usr/bin/env python3
"""Analyze V6.2 direct and native-reasoning behavioral artifacts.

The analyzer keeps three questions separate: Qwen3.6-direct versus
Qwen3.8-direct, Qwen3.8-direct versus Qwen3.8-thinking, and the descriptive
content of the generated reasoning text.  Thinking text is never used as a
primary correctness signal.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jspace_policy.v6_2_reasoning_capability import (
    DEFAULT_CONFIG,
    DEFAULT_SOURCE_CONFIG,
    DEFAULT_SOURCE_MANIFEST,
    STUDY_ID,
    _group,
    build_manifest,
    canonical_sha256,
    source_dataset,
    subset_payload,
    validate_records,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results/v6_2_reasoning_capability_control"
V61_REFERENCE = ROOT / "results/v6_1_epistemic_repair/analysis_primary_production.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _hash_valid(payload: dict[str, Any]) -> bool:
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    return payload.get("content_sha256") == canonical_sha256(body)


def _rate(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> float:
    return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else 0.0


def _bootstrap(values: list[float], *, seed: int, draws: int) -> dict[str, Any]:
    if not values:
        return {"n_clusters": 0, "mean": None, "ci95": [None, None]}
    rng = random.Random(seed)
    means = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(draws)
    ]
    means.sort()
    return {
        "n_clusters": len(values),
        "mean": sum(values) / len(values),
        "ci95": [means[int(0.025 * (draws - 1))], means[int(0.975 * (draws - 1))]],
    }


def _cluster_rate(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    values = [_rate(group, predicate) for _key, group in _group(rows, "game_id").items()]
    return _bootstrap(values, seed=seed, draws=draws)


def _cluster_values(
    values: list[tuple[Any, float]], *, seed: int, draws: int
) -> dict[str, Any]:
    by_game: dict[Any, list[float]] = defaultdict(list)
    for game_id, value in values:
        by_game[game_id].append(float(value))
    return _bootstrap(
        [sum(group) / len(group) for group in by_game.values()],
        seed=seed,
        draws=draws,
    ) | {"n_cells": len(values)}


def _pair_switch(rows: list[dict[str, Any]]) -> bool:
    return (
        len(rows) == 2
        and all(row.get("selected_index") is not None for row in rows)
        and all(bool(row.get("correct")) for row in rows)
        and len({row["selected_index"] for row in rows}) == 2
    )


def _pair_stat(cells: list[tuple[Any, bool]], *, seed: int, draws: int) -> dict[str, Any]:
    return _cluster_values(
        [(game_id, float(success)) for game_id, success in cells],
        seed=seed,
        draws=draws,
    )


def _paired_sign_flip(
    differences: list[tuple[Any, float]], *, seed: int, draws: int
) -> dict[str, Any]:
    """Run the preregistered two-sided sign-flip test on game-level deltas."""

    by_game: dict[Any, list[float]] = defaultdict(list)
    for game_id, difference in differences:
        by_game[game_id].append(float(difference))
    game_differences = [
        sum(values) / len(values) for values in by_game.values() if values
    ]
    if not game_differences:
        return {
            "status": "not_run_no_matched_cells",
            "n_cells": 0,
            "n_clusters": 0,
            "observed_mean": None,
            "p_value": None,
        }
    observed = sum(game_differences) / len(game_differences)
    rng = random.Random(seed)
    null_means = [
        sum(value if rng.randrange(2) else -value for value in game_differences)
        / len(game_differences)
        for _ in range(draws)
    ]
    extreme = sum(abs(value) >= abs(observed) for value in null_means)
    ordered = sorted(null_means)
    return {
        "status": "complete",
        "alternative": "two-sided",
        "n_cells": len(differences),
        "n_clusters": len(game_differences),
        "observed_mean": observed,
        "p_value": (extreme + 1) / (draws + 1),
        "null_ci95": [
            ordered[int(0.025 * (draws - 1))],
            ordered[int(0.975 * (draws - 1))],
        ],
        "draws": draws,
    }


def _cluster_mean(values: list[tuple[Any, float]]) -> float | None:
    """Return the mean after averaging cells within each game cluster."""

    by_game: dict[Any, list[float]] = defaultdict(list)
    for game_id, value in values:
        by_game[game_id].append(float(value))
    means = [sum(game_values) / len(game_values) for game_values in by_game.values()]
    return sum(means) / len(means) if means else None


def _not_run_paired_sign_flip(reason: str) -> dict[str, Any]:
    return {
        "status": "not_run_gate_failed",
        "n_cells": 0,
        "n_clusters": 0,
        "observed_mean": None,
        "p_value": None,
        "reason": reason,
    }


_ENDPOINT_FAMILIES = {
    "H1_report_accuracy": "ledger_binding",
    "H1_action_pair_rate": "ledger_binding",
    "H1_actual_belief_invariance": "ledger_binding",
    "H2_policy_pair_rate": "policy_composition",
    "H3_explicit_contrast": "evidence_update",
    "H3_provenance_contrast": "evidence_update",
    "H5_voi_pair_rate": "active_information",
    "H5_cost_switch_rate": "active_information",
    "H5_reliability_switch_rate": "active_information",
}


def _endpoint_cells(
    rows: list[dict[str, Any]], endpoint: str
) -> dict[tuple[Any, ...], tuple[Any, bool]]:
    """Return stable endpoint cells keyed identically across direct/thinking runs."""

    cells: dict[tuple[Any, ...], tuple[Any, bool]] = {}
    if endpoint == "H1_report_accuracy":
        for row in rows:
            if row["experiment_family"] == "ledger_binding" and row["task_kind"] == "report":
                cells[(row["condition_id"],)] = (row["game_id"], bool(row["correct"]))
        return cells

    actions = [
        row
        for row in rows
        if row["experiment_family"] == "ledger_binding" and row["task_kind"] == "action"
    ]
    if endpoint == "H1_action_pair_rate":
        for key, group in _group(actions, "game_id", "actual_receiver_belief").items():
            by_modeled = {row["modeled_receiver_belief"]: row for row in group}
            if set(by_modeled) == {0, 1}:
                cells[key] = (key[0], _pair_switch(list(by_modeled.values())))
        return cells
    if endpoint == "H1_actual_belief_invariance":
        for key, group in _group(actions, "game_id", "modeled_receiver_belief").items():
            by_actual = {row["actual_receiver_belief"]: row for row in group}
            if set(by_actual) == {0, 1}:
                cells[key] = (
                    key[0],
                    all(row.get("selected_index") is not None for row in by_actual.values())
                    and by_actual[0]["selected_index"] == by_actual[1]["selected_index"],
                )
        return cells

    if endpoint == "H2_policy_pair_rate":
        policy_rows = [
            row for row in rows if row["experiment_family"] == "policy_composition"
        ]
        for key, group in _group(
            policy_rows, "game_id", "modeled_receiver_belief"
        ).items():
            by_policy = {row["receiver_policy"]: row for row in group}
            if set(by_policy) == {"literal", "contrarian"}:
                cells[key] = (key[0], _pair_switch(list(by_policy.values())))
        return cells

    if endpoint.startswith("H3_"):
        prompt_mode = (
            "explicit_rule" if endpoint == "H3_explicit_contrast" else "provenance_only"
        )
        reports = [
            row
            for row in rows
            if row["experiment_family"] == "evidence_update"
            and row["task_kind"] == "report"
            and row.get("evidence_direction") == "prior"
            and row.get("evidence_count") == 4
            and row.get("evidence_prompt_mode") == prompt_mode
        ]
        for key, group in _group(
            reports,
            "game_id",
            "prior",
            "message_source",
            "evidence_prompt_mode",
        ).items():
            evidence = {row["evidence_mode"]: row for row in group}
            if set(evidence) == {"independent", "copied"} and (
                evidence["independent"]["expected_value"]
                != evidence["copied"]["expected_value"]
            ):
                cells[key] = (
                    key[0],
                    _pair_switch([evidence["independent"], evidence["copied"]]),
                )
        return cells

    active = [
        row for row in rows if row["experiment_family"] == "active_information"
    ]
    for key, group in _group(active, "matched_group_id").items():
        by_profile = {row.get("payoff_profile"): row for row in group}
        if endpoint == "H5_voi_pair_rate":
            if set(by_profile) == {"voi_positive", "voi_negative"}:
                cells[(key[0], "voi")] = (
                    group[0]["game_id"],
                    _pair_switch(list(by_profile.values())),
                )
        elif set(by_profile) == {"voi_threshold"}:
            by_cell = {
                (row["inspection_cost"], row["signal_reliability"]): row for row in group
            }
            required = {
                ("low", "perfect"),
                ("high", "perfect"),
                ("low", "noisy"),
                ("high", "noisy"),
            }
            if set(by_cell) == required:
                if endpoint == "H5_cost_switch_rate":
                    cells[(key[0], "cost")] = (
                        group[0]["game_id"],
                        _pair_switch(
                            [by_cell[("low", "perfect")], by_cell[("high", "perfect")]]
                        ),
                    )
                elif endpoint == "H5_reliability_switch_rate":
                    cells[(key[0], "reliability")] = (
                        group[0]["game_id"],
                        _pair_switch(
                            [by_cell[("low", "perfect")], by_cell[("low", "noisy")]]
                        ),
                    )
    return cells


def _family_gate_status(summary: dict[str, Any], family: str) -> str | None:
    key = {
        "ledger_binding": "ledger_binding_endpoints",
        "policy_composition": "policy_composition_endpoints",
        "evidence_update": "evidence_independence_endpoints",
        "active_information": "active_information_endpoints",
    }[family]
    return summary.get(key, {}).get("status")


def _paired_endpoint_comparisons(
    direct_rows: list[dict[str, Any]],
    thinking_rows: list[dict[str, Any]],
    direct_summary: dict[str, Any],
    thinking_summary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    draws = int(config["statistics"]["bootstrap_draws"])
    output: dict[str, Any] = {}
    for index, endpoint in enumerate(_ENDPOINT_FAMILIES):
        direct_cells = _endpoint_cells(direct_rows, endpoint)
        thinking_cells = _endpoint_cells(thinking_rows, endpoint)
        common = sorted(set(direct_cells) & set(thinking_cells), key=str)
        differences = [
            (
                direct_cells[key][0],
                float(thinking_cells[key][1]) - float(direct_cells[key][1]),
            )
            for key in common
            if direct_cells[key][0] == thinking_cells[key][0]
        ]
        family = _ENDPOINT_FAMILIES[endpoint]
        direct_gate = _family_gate_status(direct_summary, family)
        thinking_gate = _family_gate_status(thinking_summary, family)
        eligible = direct_gate == "supported" and thinking_gate == "supported"
        direct_values = [(game_id, float(value)) for game_id, value in direct_cells.values()]
        thinking_values = [
            (game_id, float(value)) for game_id, value in thinking_cells.values()
        ]
        output[endpoint] = {
            "direct": _cluster_mean(direct_values),
            "thinking": _cluster_mean(thinking_values),
            "delta_thinking_minus_direct": _cluster_mean(differences),
            "matched_cells": len(differences),
            "family_gate": {
                "direct": direct_gate,
                "thinking": thinking_gate,
                "eligible": eligible,
            },
            "paired_cluster_sign_flip": (
                _paired_sign_flip(
                    differences,
                    seed=6700 + index,
                    draws=draws,
                )
                if eligible
                else _not_run_paired_sign_flip(
                    "paired test requires the family gate in both conditions"
                )
            ),
        }
    return output


def _h1(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    draws = int(config["statistics"]["bootstrap_draws"])
    reports = [row for row in rows if row["task_kind"] == "report"]
    actions = [row for row in rows if row["task_kind"] == "action"]
    pair_cells: list[tuple[Any, bool]] = []
    for key, group in _group(actions, "game_id", "actual_receiver_belief").items():
        by_modeled = {row["modeled_receiver_belief"]: row for row in group}
        if set(by_modeled) == {0, 1}:
            pair_cells.append((key[0], _pair_switch(list(by_modeled.values()))))
    invariance_cells: list[tuple[Any, bool]] = []
    for key, group in _group(actions, "game_id", "modeled_receiver_belief").items():
        by_actual = {row["actual_receiver_belief"]: row for row in group}
        if set(by_actual) == {0, 1}:
            invariance_cells.append(
                (
                    key[0],
                    all(
                        row.get("selected_index") is not None
                        for row in by_actual.values()
                    )
                    and by_actual[0]["selected_index"] == by_actual[1]["selected_index"],
                )
            )
    report_by_target = {
        target: _cluster_rate(
            [row for row in reports if row.get("report_target") == target],
            lambda row: row["correct"],
            seed=6100 + index,
            draws=draws,
        )
        for index, target in enumerate(sorted({row.get("report_target") for row in reports}))
    }
    report_accuracy = _cluster_rate(reports, lambda row: row["correct"], seed=6101, draws=draws)
    pair_rate = _pair_stat(pair_cells, seed=6102, draws=draws)
    invariance = _pair_stat(invariance_cells, seed=6103, draws=draws)
    gate = (
        (report_accuracy["mean"] or 0.0) >= config["gates"]["minimum_h1_report_accuracy"]
        and (pair_rate["mean"] or 0.0) >= config["gates"]["minimum_h1_action_pair_rate"]
        and (invariance["mean"] or 0.0)
        >= config["gates"]["minimum_h1_actual_belief_invariance"]
    )
    return {
        "status": "supported" if gate else "falsified",
        "report_accuracy": report_accuracy,
        "report_accuracy_by_target": report_by_target,
        "action_pair_rate": pair_rate,
        "actual_belief_invariance": invariance,
        "n_action_pair_cells": len(pair_cells),
        "n_invariance_pair_cells": len(invariance_cells),
    }


def _h2(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    draws = int(config["statistics"]["bootstrap_draws"])
    cells: list[tuple[Any, bool]] = []
    for key, group in _group(rows, "game_id", "modeled_receiver_belief").items():
        by_policy = {row["receiver_policy"]: row for row in group}
        if set(by_policy) == {"literal", "contrarian"}:
            cells.append((key[0], _pair_switch(list(by_policy.values()))))
    pair_rate = _pair_stat(cells, seed=6201, draws=draws)
    accuracy = _cluster_rate(rows, lambda row: row["correct"], seed=6202, draws=draws)
    return {
        "status": "supported"
        if (pair_rate["mean"] or 0.0) >= config["gates"]["minimum_h2_policy_pair_rate"]
        else "falsified",
        "within_game_policy_pair_rate": pair_rate,
        "accuracy": accuracy,
        "n_pair_cells": len(cells),
    }


def _h3(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    draws = int(config["statistics"]["bootstrap_draws"])
    reports = [
        row
        for row in rows
        if row["task_kind"] == "report"
        and row.get("evidence_direction") == "prior"
        and row.get("evidence_count") == 4
    ]
    by_mode: dict[str, list[tuple[Any, bool]]] = defaultdict(list)
    by_mode_correct: dict[str, list[tuple[Any, bool]]] = defaultdict(list)
    for key, group in _group(
        reports,
        "game_id",
        "prior",
        "message_source",
        "evidence_prompt_mode",
    ).items():
        evidence = {row["evidence_mode"]: row for row in group}
        if set(evidence) != {"independent", "copied"}:
            continue
        if evidence["independent"]["expected_value"] == evidence["copied"]["expected_value"]:
            continue
        independent = evidence["independent"]
        copied = evidence["copied"]
        mode = str(key[-1])
        by_mode[mode].append(
            (
                key[0],
                independent.get("selected_index") is not None
                and copied.get("selected_index") is not None
                and independent.get("selected_index") != copied.get("selected_index"),
            )
        )
        by_mode_correct[mode].append((key[0], _pair_switch([independent, copied])))
    accuracy = _cluster_rate(rows, lambda row: row["correct"], seed=6301, draws=draws)
    contrast = {
        mode: _pair_stat(values, seed=6310 + index, draws=draws)
        for index, (mode, values) in enumerate(sorted(by_mode.items()))
    }
    correct_switch = {
        mode: _pair_stat(values, seed=6320 + index, draws=draws)
        for index, (mode, values) in enumerate(sorted(by_mode_correct.items()))
    }
    required_modes = set(config["endpoints"]["H3_evidence_independence"]["prompt_modes"])
    gate = (
        (accuracy["mean"] or 0.0) >= config["gates"]["minimum_accuracy"]
        and set(correct_switch) == required_modes
        and all(
            (correct_switch[mode]["mean"] or 0.0)
            >= config["gates"]["minimum_h3_prompt_mode_pair_rate"]
            for mode in sorted(required_modes)
        )
    )
    return {
        "status": "supported" if gate else "falsified",
        "accuracy": accuracy,
        "independence_contrast_by_prompt_mode": contrast,
        "both_correct_and_switch_by_prompt_mode": correct_switch,
        "primary_gate_endpoint": "both_correct_and_switch_by_prompt_mode",
        "n_identifying_pair_cells": sum(len(values) for values in by_mode.values()),
    }


def _h5(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    draws = int(config["statistics"]["bootstrap_draws"])
    profile_cells: list[tuple[Any, bool]] = []
    cost_cells: list[tuple[Any, bool]] = []
    reliability_cells: list[tuple[Any, bool]] = []
    for _key, group in _group(rows, "matched_group_id").items():
        by_profile = {row.get("payoff_profile"): row for row in group}
        if set(by_profile) == {"voi_positive", "voi_negative"}:
            profile_cells.append((group[0]["game_id"], _pair_switch(list(by_profile.values()))))
        if set(by_profile) != {"voi_threshold"}:
            continue
        by_cell = {(row["inspection_cost"], row["signal_reliability"]): row for row in group}
        if set(by_cell) != {
            ("low", "perfect"),
            ("high", "perfect"),
            ("low", "noisy"),
            ("high", "noisy"),
        }:
            continue
        cost_cells.append(
            (
                group[0]["game_id"],
                _pair_switch([by_cell[("low", "perfect")], by_cell[("high", "perfect")]]),
            )
        )
        reliability_cells.append(
            (
                group[0]["game_id"],
                _pair_switch([by_cell[("low", "perfect")], by_cell[("low", "noisy")]]),
            )
        )
    accuracy = _cluster_rate(rows, lambda row: row["correct"], seed=6501, draws=draws)
    profile_rate = _pair_stat(profile_cells, seed=6502, draws=draws)
    cost_rate = _pair_stat(cost_cells, seed=6503, draws=draws)
    reliability_rate = _pair_stat(reliability_cells, seed=6504, draws=draws)
    gate = (
        (accuracy["mean"] or 0.0) >= config["gates"]["minimum_accuracy"]
        and (profile_rate["mean"] or 0.0) >= config["gates"]["minimum_h5_voi_pair_rate"]
        and (cost_rate["mean"] or 0.0) >= config["gates"]["minimum_h5_cost_threshold_rate"]
        and (reliability_rate["mean"] or 0.0)
        >= config["gates"]["minimum_h5_reliability_threshold_rate"]
    )
    return {
        "status": "supported" if gate else "falsified",
        "accuracy": accuracy,
        "within_pair_voi_switch_rate": profile_rate,
        "same_matrix_cost_switch_rate": cost_rate,
        "same_matrix_reliability_switch_rate": reliability_rate,
        "n_voi_pair_cells": len(profile_cells),
        "n_cost_threshold_cells": len(cost_cells),
        "n_reliability_threshold_cells": len(reliability_cells),
    }


def _family_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    families = {
        family: [row for row in rows if row["experiment_family"] == family]
        for family in (
            "ledger_binding",
            "policy_composition",
            "evidence_update",
            "recursive_strategy",
            "active_information",
            "monitoring_goal",
            "scaffold_order",
        )
    }
    result: dict[str, Any] = {
        family: {
            "records": len(values),
            "accuracy": _rate(values, lambda row: row["correct"]),
            "parse_rate": _rate(values, lambda row: row["parseable"]),
        }
        for family, values in families.items()
        if values
    }
    if families["ledger_binding"]:
        result["ledger_binding_endpoints"] = _h1(families["ledger_binding"], config)
    if families["policy_composition"]:
        result["policy_composition_endpoints"] = _h2(families["policy_composition"], config)
    if families["evidence_update"]:
        result["evidence_independence_endpoints"] = _h3(families["evidence_update"], config)
    if families["active_information"]:
        result["active_information_endpoints"] = _h5(families["active_information"], config)
    return result


def _reference_summary() -> dict[str, Any]:
    reference = _read_json(V61_REFERENCE)
    hypotheses = reference["hypotheses"]
    return {
        "study_id": reference["study_id"],
        "model_key": reference["model_key"],
        "model_id": "Qwen/Qwen3.6-27B",
        "condition": "qwen36_direct",
        "locked_rows": reference["locked_rows"],
        "endpoints": {
            "H1_report_accuracy": hypotheses["ledger_binding"]["report_accuracy"]["mean"],
            "H1_action_pair_rate": hypotheses["ledger_binding"]["action_pair_rate"]["mean"],
            "H1_actual_belief_invariance": hypotheses["ledger_binding"][
                "actual_belief_invariance"
            ]["mean"],
            "H2_policy_pair_rate": hypotheses["policy_composition"][
                "within_game_policy_pair_rate"
            ]["mean"],
            "H3_explicit_contrast": hypotheses["evidence_update"][
                "independence_contrast_by_prompt_mode"
            ]["explicit_rule"]["mean"],
            "H3_provenance_contrast": hypotheses["evidence_update"][
                "independence_contrast_by_prompt_mode"
            ]["provenance_only"]["mean"],
            "H5_voi_pair_rate": hypotheses["active_information"]["within_pair_voi_switch_rate"][
                "mean"
            ],
            "H5_cost_switch_rate": hypotheses["active_information"][
                "same_matrix_cost_switch_rate"
            ]["mean"],
            "H5_reliability_switch_rate": hypotheses["active_information"][
                "same_matrix_reliability_switch_rate"
            ]["mean"],
        },
    }


def _direct_endpoints(summary: dict[str, Any]) -> dict[str, float | None]:
    return {
        "H1_report_accuracy": summary.get("ledger_binding_endpoints", {})
        .get("report_accuracy", {})
        .get("mean"),
        "H1_action_pair_rate": summary.get("ledger_binding_endpoints", {})
        .get("action_pair_rate", {})
        .get("mean"),
        "H1_actual_belief_invariance": summary.get("ledger_binding_endpoints", {})
        .get("actual_belief_invariance", {})
        .get("mean"),
        "H2_policy_pair_rate": summary.get("policy_composition_endpoints", {})
        .get("within_game_policy_pair_rate", {})
        .get("mean"),
        "H3_explicit_contrast": summary.get("evidence_independence_endpoints", {})
        .get("both_correct_and_switch_by_prompt_mode", {})
        .get("explicit_rule", {})
        .get("mean"),
        "H3_provenance_contrast": summary.get("evidence_independence_endpoints", {})
        .get("both_correct_and_switch_by_prompt_mode", {})
        .get("provenance_only", {})
        .get("mean"),
        "H5_voi_pair_rate": summary.get("active_information_endpoints", {})
        .get("within_pair_voi_switch_rate", {})
        .get("mean"),
        "H5_cost_switch_rate": summary.get("active_information_endpoints", {})
        .get("same_matrix_cost_switch_rate", {})
        .get("mean"),
        "H5_reliability_switch_rate": summary.get("active_information_endpoints", {})
        .get("same_matrix_reliability_switch_rate", {})
        .get("mean"),
    }


def _comparison(
    left: dict[str, float | None], right: dict[str, float | None]
) -> dict[str, Any]:
    return {
        key: {
            "left": left.get(key),
            "right": right.get(key),
            "delta_right_minus_left": (
                right[key] - left[key]
                if left.get(key) is not None and right.get(key) is not None
                else None
            ),
        }
        for key in sorted(set(left) | set(right))
    }


def _load_artifact(
    path: Path,
    *,
    subset: dict[str, Any],
    source: dict[str, Any],
    config: dict[str, Any],
    thinking: bool,
    protocol_commit_sha: str,
    preflight_content_sha256: str,
) -> dict[str, Any]:
    payload = _read_json(path)
    expected_status = (
        "qwen38_thinking_generation_complete" if thinking else "qwen38_direct_behavior_complete"
    )
    if not _hash_valid(payload) or payload.get("status") != expected_status:
        raise RuntimeError(f"invalid V6.2 model artifact: {path}")
    validate_records(
        payload,
        subset,
        config,
        thinking=thinking,
        protocol_commit_sha=protocol_commit_sha,
        preflight_content_sha256=preflight_content_sha256,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--stage",
        choices=("qwen38-direct", "qwen38-thinking-pilot", "qwen38-thinking-diagnostic"),
        help="immutable analysis artifact to write for this Actions stage",
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    protocol_commit_sha = _git_sha()

    config = _read_json(args.config)
    source, _source_config, _source_manifest = source_dataset(
        config,
        source_config_path=DEFAULT_SOURCE_CONFIG,
        source_manifest_path=DEFAULT_SOURCE_MANIFEST,
    )
    manifest = build_manifest(config)
    subsets = {
        selection: subset_payload(source, config, selection)
        for selection in ("direct", "pilot", "diagnostic")
    }
    preflight_path = args.results / "raw/preflight_qwen38.json"
    preflight = _read_json(preflight_path) if preflight_path.exists() else None
    if preflight is None or not _hash_valid(preflight):
        raise RuntimeError("model analysis requires a valid Qwen3.8 preflight artifact")
    if preflight.get("status") != "qwen38_tokenizer_preflight_passed":
        raise RuntimeError("model analysis requires a passed Qwen3.8 preflight")
    if preflight.get("protocol_commit_sha") != protocol_commit_sha:
        raise RuntimeError("preflight protocol commit does not match the analyzer commit")
    if preflight.get("config_sha256") != canonical_sha256(config):
        raise RuntimeError("preflight config hash does not match the analyzer config")
    preflight_content_sha256 = preflight["content_sha256"]
    direct_path = args.results / "raw/qwen38_direct.json"
    direct = (
        _load_artifact(
            direct_path,
            subset=subsets["direct"],
            source=source,
            config=config,
            thinking=False,
            protocol_commit_sha=protocol_commit_sha,
            preflight_content_sha256=preflight_content_sha256,
        )
        if direct_path.exists()
        else None
    )
    thinking_payloads: dict[str, dict[str, Any]] = {}
    for stage, selection in (
        ("qwen38_thinking_pilot", "pilot"),
        ("qwen38_thinking_diagnostic", "diagnostic"),
    ):
        path = args.results / f"raw/{stage}.json"
        if path.exists():
            thinking_payloads[stage] = _load_artifact(
                path,
                subset=subsets[selection],
                source=source,
                config=config,
                thinking=True,
                protocol_commit_sha=protocol_commit_sha,
                preflight_content_sha256=preflight_content_sha256,
            )
    if direct is None and not thinking_payloads:
        raise RuntimeError("no V6.2 model artifact is available to analyze")

    result: dict[str, Any] = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "behavioral_analysis_complete",
        "source_config_sha256": canonical_sha256(config),
        "source_dataset_sha256": source["content_sha256"],
        "subset_manifest_sha256": manifest["content_sha256"],
        "protocol_commit_sha": protocol_commit_sha,
        "model_forwards_in_cpu_controls": 0,
        "reasoning_text_primary_signal": False,
        "interpretation_policy": {
            "no_mechanistic_inference_from_behavior": True,
            "no_reasoning_text_as_preregistered_correctness_signal": True,
            "unparseable_thinking_generations_are_failures": True,
        },
        "v61_reference": _reference_summary(),
        "conditions": {},
        "comparisons": {},
    }
    if direct is not None:
        direct_rows = direct["records"]
        direct_summary = {
            "condition": "qwen38_direct",
            "model_id": direct["metadata"].get("model_id"),
            "model_revision": direct["metadata"].get("model_revision_resolved"),
            "thinking": False,
            "records_total": len(direct_rows),
            "locked_records": sum(row["split"] == "locked" for row in direct_rows),
            "records_by_split": {
                split: sum(row["split"] == split for row in direct_rows)
                for split in sorted({row["split"] for row in direct_rows})
            },
            "summary": _family_summary(direct_rows, config),
            "parse_failures": sum(not row["parseable"] for row in direct_rows),
        }
        result["conditions"]["qwen38_direct"] = direct_summary
        result["comparisons"]["qwen36_direct_vs_qwen38_direct"] = _comparison(
            _reference_summary()["endpoints"],
            _direct_endpoints(direct_summary["summary"]),
        )
    for stage, payload in thinking_payloads.items():
        rows = payload["records"]
        condition_key = "qwen38_thinking_" + ("pilot" if "pilot" in stage else "diagnostic")
        thinking_summary = {
            "condition": "qwen38_thinking",
            "stage": stage,
            "model_id": payload["metadata"].get("model_id"),
            "model_revision": payload["metadata"].get("model_revision_resolved"),
            "thinking": True,
            "records_total": len(rows),
            "locked_records": sum(row["split"] == "locked" for row in rows),
            "records_by_split": {
                split: sum(row["split"] == split for row in rows)
                for split in sorted({row["split"] for row in rows})
            },
            "summary": _family_summary(rows, config),
            "parse_failures": sum(not row["parseable"] for row in rows),
            "reasoning_text_retained_in_raw_artifact": True,
        }
        result["conditions"][condition_key] = thinking_summary
        if direct is not None and "diagnostic" in stage:
            diagnostic_ids = {row["condition_id"] for row in subsets["diagnostic"]["rows"]}
            direct_diag = [
                row for row in direct["records"] if row["condition_id"] in diagnostic_ids
            ]
            direct_diag_summary = _family_summary(direct_diag, config)
            result["comparisons"]["qwen38_direct_vs_qwen38_thinking"] = _comparison(
                _direct_endpoints(direct_diag_summary),
                _direct_endpoints(thinking_summary["summary"]),
            )
            result["comparisons"]["qwen38_direct_vs_qwen38_thinking"][
                "paired_endpoints"
            ] = _paired_endpoint_comparisons(
                direct_diag,
                rows,
                direct_diag_summary,
                thinking_summary["summary"],
                config,
            )
    if args.stage is None:
        if "qwen38_thinking_diagnostic" in thinking_payloads:
            args.stage = "qwen38-thinking-diagnostic"
        elif "qwen38_thinking_pilot" in thinking_payloads:
            args.stage = "qwen38-thinking-pilot"
        elif direct is not None:
            args.stage = "qwen38-direct"
        else:
            raise RuntimeError("cannot infer an immutable analysis stage")
    result["analysis_stage"] = args.stage
    result["content_sha256"] = canonical_sha256(result)
    output = args.results / f"analysis_{args.stage.replace('-', '_')}.json"
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if (
        output.exists()
        and output.read_text(encoding="utf-8") != serialized
    ):
        raise RuntimeError(f"refusing to overwrite non-identical analysis: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    cumulative = args.results / "analysis_cumulative.json"
    cumulative.write_text(serialized, encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
