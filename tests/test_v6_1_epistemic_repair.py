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
    _active_payoff_matrix,
    _contains_expected_target,
    _expected_query,
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
    assert audit["prompt_audit"]["choice_mapping_pair_failures"] == []
    assert audit["prompt_audit"]["choice_mapping_balance_failures"] == []
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
            assert (
                "each unique provenance_id identifies one conditionally independent "
                "sensor event"
            ) in group["independent"]["prompt"].lower()
            assert "binary log-odds model" in group["independent"]["prompt"]
            assert "copies of one observation count only once" not in group[
                "independent"
            ]["prompt"]


def test_primary_matched_pairs_hold_surface_label_mappings_fixed() -> None:
    rows = dataset_payload(_config())["rows"]
    specifications = (
        (
            "ledger action modeled-belief",
            [
                row
                for row in rows
                if row["experiment_family"] == "ledger_binding"
                and row["task_kind"] == "action"
            ],
            ("game_id", "actual_receiver_belief"),
        ),
        (
            "policy receiver-policy",
            [row for row in rows if row["experiment_family"] == "policy_composition"],
            ("game_id", "modeled_receiver_belief"),
        ),
        (
            "evidence independent-copied",
            [
                row
                for row in rows
                if row["experiment_family"] == "evidence_update"
                and row["task_kind"] == "report"
                and row["evidence_count"] == 4
            ],
            (
                "game_id",
                "prior",
                "evidence_direction",
                "message_source",
                "evidence_prompt_mode",
            ),
        ),
        (
            "recursive depth",
            [row for row in rows if row["experiment_family"] == "recursive_strategy"],
            ("game_id", "task_kind"),
        ),
        (
            "active positive-negative VOI",
            [row for row in rows if row["experiment_family"] == "active_information"],
            (
                "game_id",
                "uncertainty",
                "stake",
                "inspection_cost",
                "signal_reliability",
            ),
        ),
        (
            "monitoring penalty",
            [row for row in rows if row["experiment_family"] == "monitoring_goal"],
            (
                "game_id",
                "modeled_receiver_belief",
                "audit_cue",
                "audit_probability",
                "surface",
            ),
        ),
        (
            "scaffold action source",
            [
                row
                for row in rows
                if row["experiment_family"] == "scaffold_order"
                and row["task_kind"] == "action"
            ],
            ("game_id", "modeled_receiver_belief"),
        ),
        (
            "scaffold report trajectory",
            [
                row
                for row in rows
                if row["experiment_family"] == "scaffold_order"
                and row["task_kind"] == "report"
            ],
            ("game_id", "modeled_receiver_belief"),
        ),
    )
    for name, selected, keys in specifications:
        groups: dict[tuple[object, ...], list[dict]] = defaultdict(list)
        for row in selected:
            groups[tuple(row[key] for key in keys)].append(row)
        assert groups, name
        assert all(
            len({json.dumps(row["choice_mapping"], sort_keys=True) for row in group}) == 1
            for group in groups.values()
        ), name


def test_constant_surface_label_cannot_pass_identifying_switch_pairs() -> None:
    rows = dataset_payload(_config())["rows"]
    specifications = (
        (
            [
                row
                for row in rows
                if row["experiment_family"] == "ledger_binding"
                and row["task_kind"] == "action"
            ],
            ("game_id", "actual_receiver_belief"),
        ),
        (
            [row for row in rows if row["experiment_family"] == "policy_composition"],
            ("game_id", "modeled_receiver_belief"),
        ),
        (
            [
                row
                for row in rows
                if row["experiment_family"] == "evidence_update"
                and row["task_kind"] == "report"
                and row["evidence_count"] == 4
            ],
            (
                "game_id",
                "prior",
                "evidence_direction",
                "message_source",
                "evidence_prompt_mode",
            ),
        ),
        (
            [row for row in rows if row["experiment_family"] == "active_information"],
            (
                "game_id",
                "uncertainty",
                "stake",
                "inspection_cost",
                "signal_reliability",
            ),
        ),
        (
            [row for row in rows if row["experiment_family"] == "monitoring_goal"],
            (
                "game_id",
                "modeled_receiver_belief",
                "audit_cue",
                "audit_probability",
                "surface",
            ),
        ),
    )
    for selected, keys in specifications:
        groups: dict[tuple[object, ...], list[dict]] = defaultdict(list)
        for row in selected:
            groups[tuple(row[key] for key in keys)].append(row)
        identifying = [
            group
            for group in groups.values()
            if {row["expected_index"] for row in group} == {0, 1}
        ]
        assert identifying
        for group in identifying:
            constant_semantic_indices = {
                row["choice_mapping"]["A"] for row in group
            }
            assert len(constant_semantic_indices) == 1
            assert not (
                len(constant_semantic_indices) == 2
                and all(
                    row["choice_mapping"]["A"] == row["expected_index"] for row in group
                )
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


def test_negative_active_profile_is_state_dependent_but_below_inspection_cost() -> None:
    utility = _active_payoff_matrix(
        p_state_one=0.5,
        cost=1.0,
        reliability=1.0,
        stake="low",
        profile="voi_negative",
    )
    assert utility[0][0] > utility[1][0]
    assert utility[0][1] < utility[1][1]
    for prior in (0.5, 0.8):
        for cost in (1.0, 5.0):
            for reliability in (1.0, 0.7):
                inspect, _ev_inspect, _ev_act = _expected_query(
                    prior, utility, cost, reliability
                )
                assert not inspect


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
