from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from jspace_policy.v7_epistemic_agentic_strategy import (
    ACQUISITION_UNCERTAINTIES,
    COMMITMENT_MODALITIES,
    FAMILIES,
    PUBLICITY_LEVELS,
    REPORT_TARGETS,
    SURFACES,
    _contains_expected_target,
    commitment_tool_transition,
    control_audit,
    dataset_payload,
    expected_row_count,
    verify_dataset_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/v7/epistemic_agentic_strategy/experiment.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _rows():
    return dataset_payload(_config())["rows"]


def test_v7_dataset_is_deterministic_and_cpu_audited() -> None:
    config = _config()
    left = dataset_payload(config)
    right = dataset_payload(config)
    assert left == right
    verify_dataset_payload(left, config)
    assert len(left["rows"]) == expected_row_count(config) == 3744
    assert {row["experiment_family"] for row in left["rows"]} == set(FAMILIES)
    audit = control_audit(left, config)
    assert audit["status"] == "cpu_semantic_controls_passed"
    assert audit["passed"]
    assert all(audit["gates"].values())


def test_publicity_crosses_only_at_public_knowledge_threshold() -> None:
    rows = [
        row
        for row in _rows()
        if row["experiment_family"] == "publicity_threshold" and row["task_kind"] == "action"
    ]
    groups: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        groups[(row["game_id"], row["surface"])][
            row["condition_factors"]["epistemic_level"]
        ] = row
    assert groups
    for group in groups.values():
        assert set(group) == set(PUBLICITY_LEVELS)
        assert group["shared_private"]["expected_index"] == 1
        assert group["public_announcement"]["expected_index"] == 0
        assert group["common_knowledge"]["expected_index"] == 0
        assert (
            group["public_announcement"]["choice_mapping"]
            == group["shared_private"]["choice_mapping"]
        )


def test_each_strategic_decision_has_static_report_forks() -> None:
    rows = _rows()
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["shared_prefix_id"]].append(row)
    assert groups
    for group in groups.values():
        assert {row["task_kind"] for row in group} == {"action", "report"}
        assert {
            row.get("report_target") for row in group if row["task_kind"] == "report"
        } == set(REPORT_TARGETS)
        assert all(row.get("trajectory_condition") is None for row in group)
        assert all(
            "Auditor ledger for this independent diagnostic question:" in row["prompt"]
            for row in group
            if row["task_kind"] == "report"
        )


def test_commitment_modalities_change_the_game_tree_not_the_payoff_surface() -> None:
    rows = [
        row
        for row in _rows()
        if row["experiment_family"] == "credible_commitment"
        and row["task_kind"] == "action"
        and row["surface"] == "semantic"
    ]
    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        groups[row["game_id"]][row["condition_factors"]["commitment_modality"]] = row
    assert groups
    for group in groups.values():
        assert set(group) == set(COMMITMENT_MODALITIES)
        assert group["public_irreversible"]["expected_index"] == 0
        assert group["public_tool"]["expected_index"] == 0
        assert group["private_irreversible"]["expected_index"] == 1
        assert group["cheap_promise"]["expected_index"] == 1
    assert commitment_tool_transition("public_tool", True) == {
        "future_options": ["X"],
        "publicly_observed": True,
        "binding": True,
    }
    assert commitment_tool_transition("cheap_promise", True)["future_options"] == ["X", "Y"]


def test_acquisition_cells_are_matched_and_target_specific() -> None:
    rows = [
        row
        for row in _rows()
        if row["experiment_family"] == "information_acquisition"
        and row["task_kind"] == "action"
    ]
    groups: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        groups[(row["game_id"], row["surface"])][
            row["condition_factors"]["uncertainty_type"]
        ] = row
    assert groups
    for group in groups.values():
        assert set(group) == set(ACQUISITION_UNCERTAINTIES)
        assert {row["expected_index"] for row in group.values()} == {0, 1, 2, 3}
        mappings = {json.dumps(row["choice_mapping"], sort_keys=True) for row in group.values()}
        assert len(mappings) == 1
        assert group["world_state"]["expected_index"] == 1
        assert group["opponent_belief"]["expected_index"] == 2
        assert group["opponent_policy"]["expected_index"] == 3
        assert group["none"]["expected_index"] == 0
        assert "LITERAL produces response STATE_ZERO" in group["opponent_policy"]["prompt"]
        assert "B uses the LITERAL policy" in group["opponent_belief"]["prompt"]
        assert "relevant target is" in group["none"]["prompt"]


def test_diagnostic_reports_use_typed_semantic_options() -> None:
    rows = [row for row in _rows() if row["task_kind"] == "report"]
    expected = {
        ("publicity_threshold", "opponent_policy"): (
            "REQUIRE_BOTH_AGENTS_TO_KNOW",
            "REQUIRE_B_KNOWS_ONLY",
        ),
        ("publicity_threshold", "predicted_opponent_action"): ("COMMIT", "SAFE"),
        ("credible_commitment", "opponent_policy"): (
            "STAY_OUT_AFTER_VISIBLE_REAL_LOCK",
            "ENTER_UNLESS_VISIBLE_LOCK",
        ),
        ("credible_commitment", "predicted_opponent_action"): ("STAY_OUT", "ENTER"),
        ("information_acquisition", "opponent_policy"): ("LITERAL", "CONTRARIAN"),
        ("information_acquisition", "predicted_opponent_action"): (
            "STATE_ZERO",
            "STATE_ONE",
        ),
    }
    for (family, target), names in expected.items():
        row = next(
            row
            for row in rows
            if row["experiment_family"] == family and row["report_target"] == target
        )
        assert all(name in row["prompt"] for name in names)


def test_surface_controls_change_names_but_not_semantic_answers() -> None:
    rows = [row for row in _rows() if row["task_kind"] == "action" and row["split"] == "locked"]
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        factor = next(
            value for key, value in row["condition_factors"].items() if key != "surface"
        )
        groups[(row["experiment_family"], row["game_id"], str(factor))].append(row)
    assert groups
    for group in groups.values():
        assert {row["surface"] for row in group} == set(SURFACES)
        assert len({row["choice_mapping"]["A"] for row in group}) == 1
        assert len({row["expected_index"] for row in group}) == 1
        assert len({row["prompt"] for row in group}) == 3


def test_prompt_audit_does_not_accept_a_removed_publicity_policy() -> None:
    row = next(
        row
        for row in _rows()
        if row["experiment_family"] == "publicity_threshold" and row["task_kind"] == "action"
    )
    broken = dict(row)
    broken["prompt"] = broken["prompt"].replace(
        "B follows a fixed coordination rule:", "B follows an unspecified rule:"
    )
    assert not _contains_expected_target(broken)
