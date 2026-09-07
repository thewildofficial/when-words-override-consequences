from __future__ import annotations

import json
from pathlib import Path

import pytest

from jspace_policy.budget import admit_run, estimate_cost, ledger_total
from jspace_policy.v6_2_reasoning_capability import (
    CHOICES,
    DEFAULT_CONFIG,
    DIAGNOSTIC_FAMILIES,
    build_manifest,
    canonical_sha256,
    model_record,
    normalize_task_prompt,
    parse_thinking_final,
    select_rows,
    source_dataset,
    subset_payload,
    thinking_prompt,
    validate_records,
    verify_manifest,
    verify_subset_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_source_hashes_and_selection_sizes_are_frozen() -> None:
    source, source_config, source_manifest = source_dataset(CONFIG)
    assert canonical_sha256(source_config) == CONFIG["source"]["config_sha256"]
    assert source_manifest["dataset_sha256"] == CONFIG["source"]["dataset_sha256"]
    assert len(source["rows"]) == 7488
    assert len(select_rows(source, "direct")) == 2496
    assert len(select_rows(source, "pilot")) == 34
    assert len(select_rows(source, "diagnostic")) == 640


def test_manifest_is_deterministic_and_explicitly_cpu_only() -> None:
    manifest = build_manifest(CONFIG)
    assert manifest["status"] == "cpu_controls_complete_no_model_execution"
    assert manifest["interpretation"] == "No model forwards have occurred on this branch yet."
    assert manifest["model_forwards"] == 0
    assert manifest["gpu_stages_ran"] is False
    assert manifest["source"]["v61_cpu_audit_status"] == "cpu_semantic_controls_passed"
    assert manifest["subsets"]["direct"]["row_count"] == 2496
    assert manifest["subsets"]["pilot"]["row_count"] == 34
    assert manifest["subsets"]["diagnostic"]["row_count"] == 640
    assert manifest["semantic_parity"]["direct_thinking_match_on_diagnostic"] is True
    assert manifest["endpoint_coverage"]["passed"] is True
    assert manifest["parser_contract"]["passed"] is True
    assert manifest["choice_mapping_pair_failures"] == []
    assert all(value == 0 for value in manifest["constant_label_pair_failures"].values())
    verify_manifest(manifest, CONFIG)


def test_thinking_interface_changes_only_the_answer_interface() -> None:
    source, _source_config, _source_manifest = source_dataset(CONFIG)
    for row in select_rows(source, "diagnostic"):
        assert normalize_task_prompt(row["prompt"]) == normalize_task_prompt(
            thinking_prompt(row["prompt"])
        )
        assert row["candidate_labels"] == list(CHOICES)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("reasoning\nFINAL: A", "A"), ("<think>x</think>\nFINAL: B\n", "B")],
)
def test_thinking_parser_accepts_one_legal_final_line(text: str, expected: str) -> None:
    assert parse_thinking_final(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "reasoning only",
        "FINAL: A\nFINAL: B",
        "FINAL: C",
        "The answer is FINAL: A",
        "FINAL:A and more text",
    ],
)
def test_thinking_parser_rejects_missing_ambiguous_or_inline_answers(text: str) -> None:
    assert parse_thinking_final(text) is None


def test_unparseable_generation_is_retained_and_scores_as_failure() -> None:
    source, _source_config, _source_manifest = source_dataset(CONFIG)
    row = select_rows(source, "pilot")[0]
    record = model_record(
        row,
        {
            "legal_choice": None,
            "parseable": False,
            "generated_text": "I refuse to choose.",
        },
        generated=True,
    )
    assert record["generated_text"] == "I refuse to choose."
    assert record["parseable"] is False
    assert record["correct"] is False


def test_diagnostic_subset_contains_all_declared_identifying_families() -> None:
    source, _source_config, _source_manifest = source_dataset(CONFIG)
    rows = select_rows(source, "diagnostic")
    assert {row["experiment_family"] for row in rows} == set(DIAGNOSTIC_FAMILIES)
    assert {
        row["report_target"]
        for row in rows
        if row["task_kind"] == "report" and row["experiment_family"] == "ledger_binding"
    } == {
        "world_state",
        "self_belief",
        "modeled_receiver_belief",
        "other_model_of_self",
    }
    assert {
        row["receiver_policy"]
        for row in rows
        if row["experiment_family"] == "policy_composition"
    } == {
        "literal",
        "contrarian",
    }
    evidence = [row for row in rows if row["experiment_family"] == "evidence_update"]
    assert {row["evidence_prompt_mode"] for row in evidence} == {
        "explicit_rule",
        "provenance_only",
    }
    active = [row for row in rows if row["experiment_family"] == "active_information"]
    assert {row["payoff_profile"] for row in active} == {
        "voi_positive",
        "voi_negative",
        "voi_threshold",
    }


def test_subset_hash_fails_closed_on_row_or_config_change() -> None:
    source, _source_config, _source_manifest = source_dataset(CONFIG)
    subset = subset_payload(source, CONFIG, "diagnostic")
    verify_subset_payload(subset, source, CONFIG)
    broken = dict(subset)
    broken["rows"] = list(subset["rows"])
    broken["rows"][0] = dict(broken["rows"][0])
    broken["rows"][0]["prompt"] += " changed"
    with pytest.raises(RuntimeError, match="hash mismatch|deterministic selection"):
        verify_subset_payload(broken, source, CONFIG)


def test_budget_plan_is_below_branch_authorization_and_ledger_is_cumulative(
    tmp_path: Path,
) -> None:
    authorization = float(CONFIG["execution"]["authorization_usd"])
    total = 0.0
    for spec in CONFIG["execution"]["stage_limits"].values():
        total += estimate_cost(
            spec["gpu"],
            spec["timeout_seconds"],
            cpu_cores=spec["cpu_cores"],
            memory_gib=spec["memory_gib"],
            uncertainty_fraction=CONFIG["execution"]["uncertainty_fraction"],
        ).buffered_usd
    assert total <= authorization
    ledger = tmp_path / "cost_ledger.jsonl"
    first = estimate_cost("A100-80GB", 1.0, cpu_cores=8, memory_gib=32)
    admit_run(ledger, first, study_limit_usd=authorization)
    from jspace_policy.budget import append_ledger

    append_ledger(ledger, first, run_id="one", stage="V6.2:test")
    assert ledger_total(ledger) == pytest.approx(first.buffered_usd)


def test_thinking_artifact_validation_rejects_missing_final_as_failed_record() -> None:
    source, _source_config, _source_manifest = source_dataset(CONFIG)
    subset = subset_payload(source, CONFIG, "pilot")
    row = subset["rows"][0]
    record = model_record(
        row,
        {"legal_choice": None, "parseable": False, "generated_text": "missing"},
        generated=True,
    )
    payload_body = {
        "schema_version": 1,
        "study_id": CONFIG["study_id"],
        "status": "qwen38_thinking_generation_complete",
        "thinking": True,
        "config_sha256": canonical_sha256(CONFIG),
        "source_dataset_sha256": source["content_sha256"],
        "subset_sha256": subset["content_sha256"],
        "records": [record],
    }
    payload = {**payload_body, "content_sha256": canonical_sha256(payload_body)}
    with pytest.raises(RuntimeError, match="rows do not match"):
        validate_records(payload, subset, CONFIG, thinking=True)
