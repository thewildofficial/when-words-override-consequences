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
    SEMANTIC_ACTION_NAMES,
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
    assert len(left["rows"]) == expected_row_count(config) == 4320
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
        assert all("Execution mapping:" in row["prompt"] for row in group)
        assert all(
            "Auditor ledger for this independent diagnostic question:" not in row["prompt"]
            for row in group
            if row["task_kind"] == "report"
        )


def test_every_action_surface_declares_its_execution_mapping() -> None:
    for row in _rows():
        if row["task_kind"] != "action":
            continue
        display = row["epistemic_certificate"]["display_action_names"]
        semantic = SEMANTIC_ACTION_NAMES[row["experiment_family"]]
        assert all(
            f"{alias} executes {meaning}" in row["prompt"]
            for alias, meaning in zip(display, semantic, strict=True)
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
        assert "It costs 1 point" in group["cheap_promise"]["prompt"]
    assert commitment_tool_transition("public_tool", True) == {
        "future_options": ["X"],
        "publicly_observed": True,
        "binding": True,
    }
    assert commitment_tool_transition("cheap_promise", True)["future_options"] == ["X", "Y"]


def test_commitment_reports_distinguish_known_nonbinding_from_unknown() -> None:
    rows = [
        row
        for row in _rows()
        if row["experiment_family"] == "credible_commitment" and row["task_kind"] == "report"
    ]

    def _row(modality: str, target: str) -> dict:
        return next(
            row
            for row in rows
            if row["condition_factors"]["commitment_modality"] == modality
            and row["report_target"] == target
        )

    cheap_own = _row("cheap_promise", "own_information")
    assert cheap_own["expected_semantic"] == "CHANNEL_NONBINDING"
    assert "what A knows about the decision-relevant state" not in cheap_own["prompt"]
    assert "binding irreversible lock" in cheap_own["prompt"]

    cheap_obs = _row("cheap_promise", "opponent_information")
    assert cheap_obs["expected_semantic"] == "B_DOES_NOT_OBSERVE_LOCK"
    assert "what B knows about the decision-relevant state" not in cheap_obs["prompt"]

    cheap_belief = _row("cheap_promise", "opponent_belief")
    assert cheap_belief["expected_semantic"] == "B_BELIEVES_A_NOT_LOCKED"
    assert "B's belief about A's information" not in cheap_belief["prompt"]
    assert "B believes A is irreversibly locked" in cheap_belief["prompt"]

    public_own = _row("public_irreversible", "own_information")
    assert public_own["expected_semantic"] == "CHANNEL_BINDING"
    public_obs = _row("public_irreversible", "opponent_information")
    assert public_obs["expected_semantic"] == "B_OBSERVES_LOCK"
    public_belief = _row("public_irreversible", "opponent_belief")
    assert public_belief["expected_semantic"] == "B_BELIEVES_A_LOCKED"

    private_obs = _row("private_irreversible", "opponent_information")
    assert private_obs["expected_semantic"] == "B_DOES_NOT_OBSERVE_LOCK"
    private_belief = _row("private_irreversible", "opponent_belief")
    assert private_belief["expected_semantic"] == "B_BELIEVES_A_NOT_LOCKED"



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
        assert group["world_state_known"]["expected_index"] == 0
        assert group["opponent_belief_known"]["expected_index"] == 0
        assert group["opponent_policy_known"]["expected_index"] == 0
        assert "LITERAL produces response STATE_ZERO" in group["opponent_policy"]["prompt"]
        assert "B uses the LITERAL policy" in group["opponent_belief"]["prompt"]
        assert "known to A" in group["world_state_known"]["prompt"]
        target_indices = {
            "world_state": 1,
            "opponent_belief": 2,
            "opponent_policy": 3,
        }
        for uncertainty, row in group.items():
            values = {
                int(index): value
                for index, value in row["epistemic_certificate"]["query_values"].items()
            }
            if uncertainty.endswith("_known"):
                assert all(values[index] < values[0] for index in (1, 2, 3))
            else:
                assert values[target_indices[uncertainty]] > values[0]


def test_acquisition_action_rows_have_finite_regrets() -> None:
    rows = [
        row
        for row in _rows()
        if row["experiment_family"] == "information_acquisition"
        and row["task_kind"] == "action"
    ]
    assert rows
    for row in rows:
        regrets = row["regret_by_index"]
        assert regrets is not None
        assert len(regrets) == 4
        assert all(isinstance(value, (int, float)) for value in regrets)
        assert all(value >= 0 for value in regrets)
        assert regrets[row["expected_index"]] == 0
        assert row["epistemic_certificate"]["regret_by_index"] == regrets



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
        ("credible_commitment", "own_information"): (
            "CHANNEL_NONBINDING",
            "CHANNEL_BINDING",
        ),
        ("credible_commitment", "opponent_information"): (
            "B_DOES_NOT_OBSERVE_LOCK",
            "B_OBSERVES_LOCK",
        ),
        ("credible_commitment", "opponent_belief"): (
            "B_BELIEVES_A_NOT_LOCKED",
            "B_BELIEVES_A_LOCKED",
        ),
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


def test_hidden_acquisition_reports_do_not_ask_for_unshown_values() -> None:
    rows = _rows()
    hidden_world = next(
        row
        for row in rows
        if row["experiment_family"] == "information_acquisition"
        and row["condition_factors"]["uncertainty_type"] == "world_state"
        and row["task_kind"] == "report"
        and row["report_target"] == "world_state"
    )
    assert hidden_world["expected_semantic"] == "UNKNOWN_STATE"
    assert "whether A can identify the physical world state" in hidden_world["prompt"]
    assert "UNKNOWN_STATE" in hidden_world["prompt"]
    assert "Auditor ledger" not in hidden_world["prompt"]


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
        "B's policy is REQUIRE_BOTH_AGENTS_TO_KNOW:", "B's policy is unspecified:"
    )
    assert not _contains_expected_target(broken)
