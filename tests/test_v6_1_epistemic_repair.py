from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from jspace_policy.v6_1_epistemic_repair import (
    ACTIVE_PAYOFF_PROFILES,
    AUDIT_CUES,
    CHOICES,
    EVIDENCE_PROMPT_MODES,
    FAMILIES,
    REPORT_TARGETS,
    SCAFFOLD_SOURCES,
    SURFACES,
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
    assert len(left["rows"]) == expected_row_count(config) == 7392
    assert {row["experiment_family"] for row in left["rows"]} == set(FAMILIES)
    audit = control_audit(left, config)
    assert audit["status"] == "cpu_semantic_controls_passed"
    assert audit["gates"]["semantic_prompt_contract"]
    assert audit["gates"]["all_family_gates"]
    assert all(value["passed"] for value in audit["prompt_audit"]["family_audit"].values())


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
    groups: dict[tuple[str, int, str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        groups[
            (row["game_id"], row["prior"], row["message_source"], row["evidence_prompt_mode"])
        ][row["evidence_mode"]] = row
    identifying = [
        group
        for group in groups.values()
        if group["independent"]["expected_value"] != group["copied"]["expected_value"]
    ]
    assert identifying
    assert {group["independent"]["evidence_prompt_mode"] for group in identifying} == set(
        EVIDENCE_PROMPT_MODES
    )
    assert all(
        "provenance_id=independent-" in group["independent"]["prompt"]
        for group in identifying
    )
    assert all(
        "provenance_id=shared-1" in group["copied"]["prompt"] for group in identifying
    )
    for group in identifying:
        if group["independent"]["evidence_prompt_mode"] == "explicit_rule":
            assert "each independent observation has weight 2" in group["independent"]["prompt"]
        else:
            assert (
                "each independent observation has weight 2"
                not in group["independent"]["prompt"]
            )
            assert (
                "distinct provenance identifiers are distinct reports"
                not in group["independent"]["prompt"]
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
    config = _config()
    n_games = len(config["dataset"]["splits"]) * int(
        config["dataset"]["monitoring_games_per_split"]
    )
    assert len(by_cell) == n_games * 2 * len(AUDIT_CUES) * 2 * len(SURFACES)
    assert all({row["real_penalty"] for row in group} == {0, 8} for group in by_cell.values())
    assert all(len({row["expected_index"] for row in group}) == 2 for group in by_cell.values())
    lexical = [row for row in rows if row["surface"] == "lexical_conflict"]
    assert lexical
    assert all(
        "protective route" in row["prompt"] and "hazardous route" in row["prompt"]
        for row in lexical
    )


def test_active_information_profiles_are_matched_and_have_opposite_certified_voi() -> None:
    rows = [
        row
        for row in dataset_payload(_config())["rows"]
        if row["experiment_family"] == "active_information"
    ]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["matched_group_id"]].append(row)
    assert groups
    superficial = ("uncertainty", "stake", "inspection_cost", "signal_reliability")
    assert all(
        {row["payoff_profile"] for row in group} == set(ACTIVE_PAYOFF_PROFILES)
        for group in groups.values()
    )
    assert all(
        {row["expected_index"] for row in group} == {0, 1}
        for group in groups.values()
    )
    assert all(
        len({tuple(row[key] for key in superficial) for row in group}) == 1
        and row_by_profile["voi_positive"]["vo_i"] > 0
        and row_by_profile["voi_negative"]["vo_i"] < 0
        for group in groups.values()
        for row_by_profile in [{row["payoff_profile"]: row for row in group}]
    )


def test_scaffold_sources_do_not_claim_hidden_provenance_and_self_generated_is_a_real_turn(
) -> None:
    rows = [
        row
        for row in dataset_payload(_config())["rows"]
        if row["experiment_family"] == "scaffold_order"
    ]
    assert {row["scaffold_source"] for row in rows if row["task_kind"] == "action"} == set(
        SCAFFOLD_SOURCES
    )
    self_action = next(
        row
        for row in rows
        if row["task_kind"] == "action" and row["scaffold_source"] == "self_generated"
    )
    source_report = next(
        row for row in rows if row["condition_id"] == self_action["scaffold_report_id"]
    )
    assert "previous report label" not in self_action["prompt"]
    messages = trajectory_messages(
        self_action,
        first_answer="A",
        preceding_prompt=source_report["prompt"],
    )
    assert messages[1] == {"role": "user", "content": source_report["prompt"]}
    assert messages[2] == {"role": "assistant", "content": "A"}
    assert messages[3] == {"role": "user", "content": self_action["prompt"]}


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
