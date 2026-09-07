from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from jspace_policy.epistemic_strategic_experiments import (
    DECISION_FRAMES,
    EVALUATION_CONTEXTS,
    EXPERIMENT_FAMILIES,
    REPORT_TARGETS,
    SOURCE_LEVELS,
    SURFACE_LEVELS,
    TEMPORAL_LEVELS,
    TRUTH_LEVELS,
    VISIBILITY_LEVELS,
    canonical_sha256,
    control_audit,
    dataset_payload,
    expected_row_count,
    trajectory_messages,
    verify_dataset_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v6/epistemic_strategic/experiment.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_dataset_is_deterministic_and_exactly_factorial() -> None:
    config = _config()
    left = dataset_payload(config)
    right = dataset_payload(config)
    assert left == right
    verify_dataset_payload(left, config)
    assert len(left["rows"]) == expected_row_count(config) == 1740
    assert {row["experiment_family"] for row in left["rows"]} == set(
        EXPERIMENT_FAMILIES
    )
    assert {row["split"] for row in left["rows"]} == {
        "discovery",
        "validation",
        "locked",
    }


def test_core_tom_has_independent_model_and_hidden_state_contrasts() -> None:
    rows = dataset_payload(_config())["rows"]
    by_actual: dict[tuple[str, str, int], dict[int, dict]] = defaultdict(dict)
    by_modeled: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for row in rows:
        if row["experiment_family"] != "core_tom":
            continue
        key = (row["split"], row["game_id"], row["actual_other_belief"])
        by_actual[key][row["modeled_other_belief"]] = row
        modeled_key = (row["split"], row["game_id"], row["modeled_other_belief"])
        by_modeled[modeled_key].add(row["expected_action"])
    assert all(
        pair[0]["expected_action"] != pair[1]["expected_action"]
        for pair in by_actual.values()
    )
    assert all(len(actions) == 1 for actions in by_modeled.values())


def test_higher_order_role_and_visibility_cells_are_complete() -> None:
    rows = [
        row
        for row in dataset_payload(_config())["rows"]
        if row["experiment_family"] == "higher_order"
    ]
    groups: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    values: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for row in rows:
        groups[(row["split"], row["game_id"])].add(
            (row["report_target"], row["visibility"])
        )
        values[(row["split"], row["game_id"], row["report_target"])].add(
            row["expected_report_value"]
        )
    expected_cells = {
        (target, visibility)
        for target in REPORT_TARGETS
        for visibility in VISIBILITY_LEVELS
    }
    assert all(cells == expected_cells for cells in groups.values())
    assert all(len(target_values) == 1 for target_values in values.values())


def test_provenance_is_a_report_only_crossed_control() -> None:
    rows = [
        row
        for row in dataset_payload(_config())["rows"]
        if row["experiment_family"] == "provenance"
    ]
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["game_id"], row["statement_value"])].append(row)
    expected_cells = {
        (source, truth, temporal, surface)
        for source in SOURCE_LEVELS
        for truth in TRUTH_LEVELS
        for temporal in TEMPORAL_LEVELS
        for surface in SURFACE_LEVELS
    }
    for group in groups.values():
        observed = {
            (
                row["source"],
                row["truth_status"],
                row["temporal_position"],
                row["surface"],
            )
            for row in group
        }
        assert observed == expected_cells
        assert {row["task_kind"] for row in group} == {"report"}
        assert len({row["expected_report_value"] for row in group}) == 1


def test_strategy_pressure_separates_safety_and_evaluation_controls() -> None:
    rows = [
        row
        for row in dataset_payload(_config())["rows"]
        if row["experiment_family"] == "strategy_pressure"
    ]
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["game_id"], row["modeled_other_belief"])].append(row)
    assert all(
        {(row["decision_frame"], row["evaluation_context"]) for row in group}
        == {(frame, context) for frame in DECISION_FRAMES for context in EVALUATION_CONTEXTS}
        for group in groups.values()
    )
    assert all(
        {
            row["expected_action"]
            for row in group
            if row["decision_frame"] == frame
            and row["evaluation_context"] == "ordinary"
        }
        == {
            row["expected_action"]
            for row in group
            if row["decision_frame"] == frame
            and row["evaluation_context"] == "audited"
        }
        and len(
            {
                row["expected_action"]
                for row in group
                if row["decision_frame"] == frame
            }
        )
        == 1
        for group in groups.values()
        for frame in DECISION_FRAMES
    )
    assert any(
        {
            row["expected_action"]
            for row in group
            if row["decision_frame"] == "utility_only"
        }
        != {
            row["expected_action"]
            for row in group
            if row["decision_frame"] == "risk_sensitive"
        }
        for group in groups.values()
    )


def test_report_label_mapping_is_consistent_with_prompt_certificate() -> None:
    for row in dataset_payload(_config())["rows"]:
        assert row["report_mapping"][row["expected_report"]] == row[
            "game_certificate"
        ]["concepts"][row["expected_report_value"]]
        expected_surface = (
            f"{row['expected_report']}={row['report_mapping'][row['expected_report']]}"
        )
        assert expected_surface in row["report_prompt"]


def test_trajectory_controls_are_matched_and_forced_choice() -> None:
    row = next(
        row
        for row in dataset_payload(_config())["rows"]
        if row["experiment_family"] == "core_tom"
    )
    direct = trajectory_messages(row, condition="direct_action")
    action_then_report = trajectory_messages(
        row, condition="action_then_report", first_answer=row["expected_action"]
    )
    report_then_action = trajectory_messages(
        row, condition="report_then_action", first_answer=row["expected_report"]
    )
    assert [message["role"] for message in direct] == ["system", "user"]
    assert [message["role"] for message in action_then_report] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert [message["role"] for message in report_then_action] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert row["action_prompt"] in action_then_report[1]["content"]
    assert row["report_prompt"] in action_then_report[3]["content"]


def test_cpu_audit_has_separate_family_gates() -> None:
    config = _config()
    payload = dataset_payload(config)
    audit = control_audit(payload, config)
    assert audit["status"] == "cpu_symbolic_controls_passed"
    assert set(audit["family_gates"]) == set(EXPERIMENT_FAMILIES)
    for gates in audit["family_gates"].values():
        assert all(value for name, value in gates.items() if name != "stop_if_failed")
    assert audit["gates"]["hash_and_factorial"] is True
    assert audit["gates"]["all_family_gates"] is True


def test_model_namespaces_are_pinned_and_qwen38_is_locked_separately() -> None:
    config = _config()
    assert config["model"] == {
        "id": "Qwen/Qwen3.6-27B",
        "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "dtype": "bfloat16",
        "thinking": False,
    }
    qwen38 = config["replication_models"]["qwen38_27b"]
    assert qwen38["id"] == "Qwen/Qwen3.8-27B"
    assert qwen38["revision"] == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert qwen38["enabled"] is False
    assert qwen38["separate_gate_namespace"] == "qwen38"


def test_tampering_is_detected_by_content_hash() -> None:
    config = _config()
    payload = dataset_payload(config)
    payload["rows"][0]["expected_action"] = "Z"
    try:
        verify_dataset_payload(payload, config)
    except ValueError as error:
        assert "hash" in str(error)
    else:
        raise AssertionError("tampered dataset was accepted")


def test_config_hash_is_stable() -> None:
    config = _config()
    assert canonical_sha256(config) == canonical_sha256(json.loads(json.dumps(config)))
