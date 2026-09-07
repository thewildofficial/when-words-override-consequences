from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from jspace_policy.v6_1_epistemic_repair import (
    CHOICES,
    FAMILIES,
    REPORT_TARGETS,
    _contains_expected_target,
    _prompt_contract_audit,
    control_audit,
    dataset_payload,
    expected_row_count,
    trajectory_messages,
    verify_dataset_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v6.1/epistemic_repair/experiment.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_v6_1_dataset_is_deterministic_and_semantically_audited() -> None:
    config = _config()
    left = dataset_payload(config)
    right = dataset_payload(config)
    assert left == right
    verify_dataset_payload(left, config)
    assert len(left["rows"]) == expected_row_count(config) == 6144
    assert {row["experiment_family"] for row in left["rows"]} == set(FAMILIES)
    audit = control_audit(left, config)
    assert audit["status"] == "cpu_semantic_controls_passed"
    assert audit["gates"]["semantic_prompt_contract"]
    assert audit["gates"]["all_family_gates"]


def test_complete_ledger_renders_every_role_indexed_value() -> None:
    rows = dataset_payload(_config())["rows"]
    report_rows = [
        row
        for row in rows
        if row["experiment_family"] == "ledger_binding" and row["task_kind"] == "report"
    ]
    assert {row["report_target"] for row in report_rows} == set(REPORT_TARGETS)
    for row in report_rows:
        concept = row["game_certificate"]["concepts"][row["expected_value"]]
        assert concept in row["prompt"]
        assert "Complete epistemic ledger" in row["prompt"]
        assert row["expected_choice"] in CHOICES


def test_the_prompt_audit_catches_a_hidden_expected_value() -> None:
    row = next(
        row
        for row in dataset_payload(_config())["rows"]
        if row["experiment_family"] == "ledger_binding"
        and row["task_kind"] == "report"
        and row["report_target"] == "modeled_receiver_belief"
    )
    concept = row["game_certificate"]["concepts"][row["expected_value"]]
    broken = dict(row)
    broken["prompt"] = broken["prompt"].replace(
        f"- your model of the receiver's belief = {concept}",
        "- your model of the receiver's belief = OMITTED",
    )
    assert not _contains_expected_target(broken)
    audit = _prompt_contract_audit([broken])
    assert row["condition_id"] in audit["missing_expected_target_rows"]
    assert not audit["passed"]


def test_policy_flip_keeps_game_fixed_and_does_not_supply_derived_response() -> None:
    rows = [
        row
        for row in dataset_payload(_config())["rows"]
        if row["experiment_family"] == "policy_composition"
    ]
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["game_id"], row["modeled_receiver_belief"])].append(row)
    assert groups
    for group in groups.values():
        assert {row["receiver_policy"] for row in group} == {"literal", "contrarian"}
        assert len({row["prompt"] for row in group}) == 2
        assert all("modeled receiver response is" not in row["prompt"].lower() for row in group)


def test_evidence_cells_distinguish_independent_from_copied_observations() -> None:
    rows = [
        row
        for row in dataset_payload(_config())["rows"]
        if row["experiment_family"] == "evidence_update"
        and row["task_kind"] == "report"
        and row["evidence_direction"] == "prior"
        and row["evidence_count"] == 4
    ]
    groups: dict[tuple[str, int, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        groups[(row["game_id"], row["prior"], row["message_source"])][row["evidence_mode"]] = (
            row
        )
    identifying = [
        group
        for group in groups.values()
        if group["independent"]["expected_value"] != group["copied"]["expected_value"]
    ]
    assert identifying
    assert all(
        "independent observation" in row["prompt"]
        for group in identifying
        for row in [group["independent"]]
    )
    assert all(
        "copied instance" in row["prompt"] for group in identifying for row in [group["copied"]]
    )


def test_monitoring_has_many_forced_safety_switch_cells_and_audit_invariance() -> None:
    rows = [
        row
        for row in dataset_payload(_config())["rows"]
        if row["experiment_family"] == "monitoring_goal"
    ]
    by_cell: dict[tuple[str, int, str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[
            (
                row["game_id"],
                row["modeled_receiver_belief"],
                row["audit_cue"],
                row["audit_probability"],
                row["surface"],
            )
        ].append(row)
    assert len(by_cell) == 36 * 2 * 2 * 2 * 2
    assert all({row["real_penalty"] for row in group} == {0, 8} for group in by_cell.values())
    assert all(len({row["expected_index"] for row in group}) == 2 for group in by_cell.values())


def test_trajectory_report_prompt_is_identical_before_and_after_action() -> None:
    rows = [
        row
        for row in dataset_payload(_config())["rows"]
        if row["experiment_family"] == "scaffold_order"
    ]
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["game_id"], row["modeled_receiver_belief"])].append(row)
    for group in groups.values():
        direct = next(
            row for row in group if row.get("trajectory_condition") == "direct_report"
        )
        after = next(
            row for row in group if row.get("trajectory_condition") == "action_then_report"
        )
        action = next(
            row
            for row in group
            if row["task_kind"] == "action" and row["scaffold_source"] == "none"
        )
        assert direct["prompt"] == after["prompt"]
        assert after["trajectory_action_row_id"] == action["condition_id"]
        messages = trajectory_messages(
            after, first_answer="A", preceding_prompt=action["prompt"]
        )
        assert messages[1]["content"] == action["prompt"]
        assert messages[-1]["content"] == direct["prompt"]
