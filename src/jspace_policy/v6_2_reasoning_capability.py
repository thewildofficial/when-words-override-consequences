"""CPU-frozen controls for the V6.2 reasoning-capability study.

V6.2 is a prospective capability control for V6.1-ES-2.  It never regenerates
or edits the V6.1 protocol.  Instead, it loads the V6.1 generator, verifies its
content hash, and constructs three immutable views of the locked split:

* ``direct``: every locked V6.1 row, for Qwen3.8 forced-choice logits;
* ``pilot``: a small deterministic reasoning-mode smoke set from the V6.1
  validation split; and
* ``diagnostic``: all H1/H2/H3/H5 identifying rows needed to distinguish
  generic reasoning failure from information-to-action failure.

This module is deliberately model-free.  The control command can therefore be
run in CI before any tokenizer, model, or Modal function is called.  The
thinking pilot is drawn from V6.1's validation split; only the direct and
diagnostic views touch the locked split.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from jspace_policy import v6_1_epistemic_repair as v61

STUDY_ID = "V6.2-RCC-1"
SCHEMA_VERSION = 1
SOURCE_STUDY_ID = v61.STUDY_ID
CHOICES = v61.CHOICES
LOCKED_SPLIT = "locked"
VALIDATION_SPLIT = "validation"
SELECTION_SPLITS = {
    "direct": LOCKED_SPLIT,
    "diagnostic": LOCKED_SPLIT,
    "pilot": VALIDATION_SPLIT,
}
DIAGNOSTIC_FAMILIES = (
    "ledger_binding",
    "policy_composition",
    "evidence_update",
    "active_information",
)
THINKING_MARKERS = ("<think>", "<|think|>", "<|assistant_thinking|>")
FINAL_LINE_RE = re.compile(r"(?im)^\s*FINAL\s*:\s*([AB])\s*$")
ANSWER_INTERFACE = "Return only A or B.\nAnswer:"
THINKING_INTERFACE = (
    "Reason using only the visible task information. You may use native reasoning. "
    "End your response with exactly one line containing FINAL: A or FINAL: B.\nAnswer:"
)

# These keys define the identifying comparisons that must survive selection.
IDENTIFYING_GROUP_KEYS = {
    "ledger_action": ("game_id", "actual_receiver_belief"),
    "policy": ("game_id", "modeled_receiver_belief"),
    "evidence": (
        "game_id",
        "prior",
        "message_source",
        "evidence_prompt_mode",
    ),
    "active_profile": ("matched_group_id",),
    "active_threshold": ("matched_group_id",),
}

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/v6.2/reasoning_capability_control/experiment.json"
DEFAULT_SOURCE_CONFIG = ROOT / "configs/v6.1/epistemic_repair/experiment.json"
DEFAULT_SOURCE_MANIFEST = ROOT / "configs/v6.1/epistemic_repair/dataset_manifest.json"


def canonical_sha256(value: object) -> str:
    """Hash JSON independent of whitespace and dictionary insertion order."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _group(
    rows: list[dict[str, Any]], *keys: str
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def _row_ids(rows: list[dict[str, Any]]) -> list[str]:
    return sorted(str(row["condition_id"]) for row in rows)


def _without_hash(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "content_sha256"}


def _hash_valid(payload: dict[str, Any]) -> bool:
    return payload.get("content_sha256") == canonical_sha256(_without_hash(payload))


def source_dataset(
    config: dict[str, Any],
    *,
    source_config_path: Path = DEFAULT_SOURCE_CONFIG,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return the generated V6.1 dataset plus its verified source metadata."""

    source_config = _read_json(source_config_path)
    source_manifest = _read_json(source_manifest_path)
    expected = config["source"]
    actual_config_hash = canonical_sha256(source_config)
    if actual_config_hash != expected["config_sha256"]:
        raise RuntimeError("V6.1 source config hash changed")
    if source_manifest.get("config_sha256") != expected["config_sha256"]:
        raise RuntimeError("V6.1 dataset manifest points at another config")
    if source_manifest.get("dataset_sha256") != expected["dataset_sha256"]:
        raise RuntimeError("V6.1 dataset manifest hash changed")
    generated = v61.dataset_payload(source_config)
    v61.verify_dataset_payload(generated, source_config)
    if generated["study_id"] != SOURCE_STUDY_ID:
        raise RuntimeError("unexpected V6.1 source study ID")
    if generated["content_sha256"] != expected["dataset_sha256"]:
        raise RuntimeError("deterministic V6.1 dataset hash changed")
    if len(generated["rows"]) != int(expected["row_count"]):
        raise RuntimeError("V6.1 source row count changed")
    locked_count = sum(row["split"] == LOCKED_SPLIT for row in generated["rows"])
    if locked_count != int(expected["locked_row_count"]):
        raise RuntimeError("V6.1 locked row count changed")
    return generated, source_config, source_manifest


def normalize_task_prompt(prompt: str) -> str:
    """Remove only the direct or thinking answer interface from a task prompt."""

    direct_count = prompt.count(ANSWER_INTERFACE)
    thinking_count = prompt.count(THINKING_INTERFACE)
    if direct_count == 1 and thinking_count == 0:
        return prompt.replace(ANSWER_INTERFACE, "[[CHOICE_INTERFACE]]")
    if direct_count == 0 and thinking_count == 1:
        return prompt.replace(THINKING_INTERFACE, "[[CHOICE_INTERFACE]]")
    raise ValueError("task prompt does not have one recognized answer interface")


def thinking_prompt(prompt: str) -> str:
    """Replace the direct label interface with the preregistered reasoning interface."""

    if prompt.count(ANSWER_INTERFACE) != 1:
        raise ValueError("source prompt does not have the unique V6.1 answer interface")
    return prompt.replace(ANSWER_INTERFACE, THINKING_INTERFACE)


def trajectory_messages(
    row: dict[str, Any],
    *,
    thinking: bool,
    first_answer: str | None = None,
    materialized_prompt: str | None = None,
    preceding_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Build the V6.1 trajectory with only its final-answer interface changed."""

    source_prompt = materialized_prompt or row["prompt"]
    prompt = thinking_prompt(source_prompt) if thinking else source_prompt
    system = (
        "Treat the synthetic environment as exact. Do not infer information that "
        "is absent from the user message. Keep agent-indexed beliefs distinct. "
    )
    system += (
        "Use native reasoning if enabled and end with one machine-readable FINAL line."
        if thinking
        else "Return only the requested legal label."
    )
    previous = preceding_prompt or row["prompt"]
    previous = thinking_prompt(previous) if thinking else previous
    if row.get("trajectory_condition") == "action_then_report" and first_answer is not None:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": previous},
            {"role": "assistant", "content": first_answer},
            {"role": "user", "content": prompt},
        ]
    if row["task_kind"] == "report":
        return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    if first_answer is None:
        return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": previous},
        {"role": "assistant", "content": first_answer},
        {"role": "user", "content": prompt},
    ]


def semantic_prompt_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Hash task semantics while ignoring tokenizer/template and answer-interface details."""

    ordered = sorted(rows, key=lambda row: row["condition_id"])
    body = [
        (
            row["condition_id"],
            normalize_task_prompt(row["prompt"]),
            list(row["candidate_labels"]),
        )
        for row in ordered
    ]
    return {
        "row_count": len(body),
        "query_ids_sha256": canonical_sha256([item[0] for item in body]),
        "task_prompts_sha256": canonical_sha256([(item[0], item[1]) for item in body]),
        "candidate_labels_sha256": canonical_sha256([(item[0], item[2]) for item in body]),
        "contract_sha256": canonical_sha256(body),
    }


def _locked(source: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in source["rows"] if row["split"] == LOCKED_SPLIT]


def _validation(source: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in source["rows"] if row["split"] == VALIDATION_SPLIT]


def _h3_identifying_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    candidates = [
        row
        for row in rows
        if row["experiment_family"] == "evidence_update"
        and row["task_kind"] == "report"
        and row["evidence_direction"] == "prior"
        and row["evidence_count"] == 4
    ]
    groups = _group(candidates, *IDENTIFYING_GROUP_KEYS["evidence"])
    return [
        group
        for _key, group in sorted(groups.items())
        if {row["evidence_mode"] for row in group} == set(v61.EVIDENCE_MODES)
        and {row["expected_value"] for row in group} == {0, 1}
    ]


def _h5_groups(
    rows: list[dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    active = [row for row in rows if row["experiment_family"] == "active_information"]
    groups = _group(active, *IDENTIFYING_GROUP_KEYS["active_profile"])
    profile_groups = [
        group
        for _key, group in sorted(groups.items())
        if {row["payoff_profile"] for row in group} == set(v61.ACTIVE_PAYOFF_PROFILES)
    ]
    threshold_groups = [
        group
        for _key, group in sorted(groups.items())
        if {row["payoff_profile"] for row in group} == {v61.ACTIVE_THRESHOLD_PROFILE}
    ]
    return profile_groups, threshold_groups


def select_rows(source: dict[str, Any], selection: str) -> list[dict[str, Any]]:
    """Apply the frozen, selection-only-before-output rule."""

    locked = _locked(source)
    if selection == "direct":
        return sorted(locked, key=lambda row: row["condition_id"])
    if selection == "diagnostic":
        selected: list[dict[str, Any]] = []
        selected.extend(
            row for row in locked if row["experiment_family"] in DIAGNOSTIC_FAMILIES[:2]
        )
        selected.extend(row for group in _h3_identifying_groups(locked) for row in group)
        profile_groups, threshold_groups = _h5_groups(locked)
        selected.extend(row for group in profile_groups + threshold_groups for row in group)
        result = sorted(
            {row["condition_id"]: row for row in selected}.values(),
            key=lambda row: row["condition_id"],
        )
        if {row["experiment_family"] for row in result} != set(DIAGNOSTIC_FAMILIES):
            raise RuntimeError("diagnostic selection did not cover all declared families")
        return result
    if selection == "pilot":
        # One complete ledger game, one policy game, one independent/copy pair
        # in each prompt mode, one positive/negative VOI pair, and one full
        # same-matrix threshold cell, drawn from validation rather than locked.
        # The sort order is part of the protocol.
        selected = []
        validation = _validation(source)
        ledger = [
            row for row in validation if row["experiment_family"] == "ledger_binding"
        ]
        first_ledger_game = min(row["game_id"] for row in ledger)
        selected.extend(row for row in ledger if row["game_id"] == first_ledger_game)
        policy = [
            row for row in validation if row["experiment_family"] == "policy_composition"
        ]
        first_policy_game = min(row["game_id"] for row in policy)
        selected.extend(row for row in policy if row["game_id"] == first_policy_game)
        h3_groups = _h3_identifying_groups(validation)
        for prompt_mode in v61.EVIDENCE_PROMPT_MODES:
            group = next(
                group for group in h3_groups if group[0]["evidence_prompt_mode"] == prompt_mode
            )
            selected.extend(group)
        profile_groups, threshold_groups = _h5_groups(validation)
        selected.extend(profile_groups[0])
        selected.extend(threshold_groups[0])
        return sorted(
            {row["condition_id"]: row for row in selected}.values(),
            key=lambda row: row["condition_id"],
        )
    raise ValueError(f"unknown selection: {selection}")


def subset_payload(
    source: dict[str, Any], config: dict[str, Any], selection: str
) -> dict[str, Any]:
    rows = select_rows(source, selection)
    expected_split = SELECTION_SPLITS.get(selection)
    if expected_split is None:
        raise ValueError(f"unknown selection: {selection}")
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "status": "frozen_before_model_execution",
        "selection": selection,
        "split": expected_split,
        "source_study_id": SOURCE_STUDY_ID,
        "source_dataset_sha256": source["content_sha256"],
        "config_sha256": canonical_sha256(config),
        "rows": rows,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def verify_subset_payload(
    payload: dict[str, Any], source: dict[str, Any], config: dict[str, Any]
) -> None:
    if not _hash_valid(payload):
        raise RuntimeError("subset content hash mismatch")
    if payload.get("study_id") != STUDY_ID:
        raise RuntimeError("subset has the wrong study ID")
    if payload.get("source_study_id") != SOURCE_STUDY_ID:
        raise RuntimeError("subset has the wrong source study ID")
    if payload.get("source_dataset_sha256") != source["content_sha256"]:
        raise RuntimeError("subset source dataset hash mismatch")
    if payload.get("config_sha256") != canonical_sha256(config):
        raise RuntimeError("subset config hash mismatch")
    expected = subset_payload(source, config, str(payload.get("selection")))
    if payload.get("content_sha256") != expected["content_sha256"]:
        raise RuntimeError("subset does not match the deterministic selection rule")
    selection = str(payload.get("selection"))
    expected_split = SELECTION_SPLITS.get(selection)
    if expected_split is None:
        raise RuntimeError("subset has an unknown selection")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("subset rows are missing")
    if any(row.get("split") != expected_split for row in rows):
        raise RuntimeError(f"{selection} subset contains a row outside {expected_split}")
    if payload.get("split") != expected_split:
        raise RuntimeError(f"{selection} subset split metadata is inconsistent")
    if len({row["condition_id"] for row in rows}) != len(rows):
        raise RuntimeError("subset contains duplicate condition IDs")


def _choice_mapping_pair_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    specifications = (
        (
            "ledger_action",
            [
                r
                for r in rows
                if r["experiment_family"] == "ledger_binding" and r["task_kind"] == "action"
            ],
            IDENTIFYING_GROUP_KEYS["ledger_action"],
        ),
        (
            "policy",
            [r for r in rows if r["experiment_family"] == "policy_composition"],
            IDENTIFYING_GROUP_KEYS["policy"],
        ),
        (
            "evidence",
            [
                r
                for r in rows
                if r["experiment_family"] == "evidence_update"
                and r["task_kind"] == "report"
                and r["evidence_count"] == 4
            ],
            IDENTIFYING_GROUP_KEYS["evidence"],
        ),
        (
            "active",
            [r for r in rows if r["experiment_family"] == "active_information"],
            IDENTIFYING_GROUP_KEYS["active_profile"],
        ),
    )
    for name, selected, keys in specifications:
        for key, group in _group(selected, *keys).items():
            signatures = {json.dumps(row["choice_mapping"], sort_keys=True) for row in group}
            if len(signatures) > 1:
                failures.append({"comparison": name, "key": list(key)})
    return failures


def _pair_switch_success(records: list[dict[str, Any]]) -> bool:
    """The exact paired endpoint used by the analyzer and constant-label control."""

    return (
        len(records) == 2
        and all(record.get("selected_index") is not None for record in records)
        and all(bool(record.get("correct")) for record in records)
        and len({record["selected_index"] for record in records}) == 2
    )


def _constant_label_failures(rows: list[dict[str, Any]]) -> dict[str, int]:
    checks: dict[str, int] = {}
    specifications = (
        (
            "ledger",
            [
                r
                for r in rows
                if r["experiment_family"] == "ledger_binding" and r["task_kind"] == "action"
            ],
            IDENTIFYING_GROUP_KEYS["ledger_action"],
        ),
        (
            "policy",
            [r for r in rows if r["experiment_family"] == "policy_composition"],
            IDENTIFYING_GROUP_KEYS["policy"],
        ),
        (
            "evidence",
            [
                r
                for r in rows
                if r["experiment_family"] == "evidence_update"
                and r["task_kind"] == "report"
                and r["evidence_count"] == 4
            ],
            IDENTIFYING_GROUP_KEYS["evidence"],
        ),
        (
            "active",
            [r for r in rows if r["experiment_family"] == "active_information"],
            IDENTIFYING_GROUP_KEYS["active_profile"],
        ),
    )
    for name, selected, keys in specifications:
        identifying = []
        for _key, group in _group(selected, *keys).items():
            if {row["expected_index"] for row in group} == {0, 1} and len(group) == 2:
                identifying.append(group)
        failures = 0
        for group in identifying:
            for label in CHOICES:
                synthetic = [
                    {
                        "selected_index": row["choice_mapping"][label],
                        "correct": row["choice_mapping"][label] == row["expected_index"],
                    }
                    for row in group
                ]
                failures += int(_pair_switch_success(synthetic))
        checks[name] = failures
    return checks


def parse_thinking_final(text: str, candidates: tuple[str, ...] = CHOICES) -> str | None:
    """Parse exactly one machine-readable final line; ambiguity is a failure."""

    matches = [match.group(1) for match in FINAL_LINE_RE.finditer(str(text))]
    if len(matches) != 1 or matches[0] not in candidates:
        return None
    return matches[0]


def parser_contract() -> dict[str, Any]:
    accepted = {
        "FINAL: A": parse_thinking_final("some reasoning\nFINAL: A"),
        "FINAL: B": parse_thinking_final("<think>x</think>\nFINAL: B\n"),
    }
    rejected = {
        "missing": parse_thinking_final("some reasoning"),
        "ambiguous": parse_thinking_final("FINAL: A\nFINAL: B"),
        "wrong_label": parse_thinking_final("FINAL: C"),
        "inline": parse_thinking_final("The answer is FINAL: A"),
    }
    passed = accepted == {"FINAL: A": "A", "FINAL: B": "B"} and all(
        value is None for value in rejected.values()
    )
    return {"accepted": accepted, "rejected": rejected, "passed": passed}


def _endpoint_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    h1 = [row for row in rows if row["experiment_family"] == "ledger_binding"]
    h2 = [row for row in rows if row["experiment_family"] == "policy_composition"]
    h3_groups = _h3_identifying_groups(rows)
    h5_profiles, h5_thresholds = _h5_groups(rows)
    h1_action_groups = _group(
        [row for row in h1 if row["task_kind"] == "action"],
        *IDENTIFYING_GROUP_KEYS["ledger_action"],
    )
    h2_groups = _group(h2, *IDENTIFYING_GROUP_KEYS["policy"])
    return {
        "h1_all_report_targets": set(
            row.get("report_target") for row in h1 if row["task_kind"] == "report"
        )
        == set(v61.REPORT_TARGETS),
        "h1_modeled_belief_pairs": sum(
            len(group) == 2 and {row["modeled_receiver_belief"] for row in group} == {0, 1}
            for group in h1_action_groups.values()
        ),
        "h2_literal_contrarian_pairs": sum(
            {row["receiver_policy"] for row in group} == {"literal", "contrarian"}
            for group in h2_groups.values()
        ),
        "h3_identifying_pairs": len(h3_groups),
        "h3_prompt_modes": sorted({group[0]["evidence_prompt_mode"] for group in h3_groups}),
        "h5_positive_negative_pairs": len(h5_profiles),
        "h5_threshold_cells": len(h5_thresholds),
        "passed": (
            set(row.get("report_target") for row in h1 if row["task_kind"] == "report")
            == set(v61.REPORT_TARGETS)
            and bool(h1_action_groups)
            and bool(h2_groups)
            and len(h3_groups) > 0
            and set(group[0]["evidence_prompt_mode"] for group in h3_groups)
            == set(v61.EVIDENCE_PROMPT_MODES)
            and len(h5_profiles) > 0
            and len(h5_thresholds) > 0
        ),
    }


def build_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """Build the complete deterministic pre-run manifest."""

    source, source_config, source_manifest = source_dataset(config)
    v61_audit = v61.control_audit(source, source_config)
    if not v61_audit["gates"]["all_family_gates"]:
        raise RuntimeError("V6.1 source CPU audit no longer passes")
    subsets = {
        selection: subset_payload(source, config, selection)
        for selection in ("direct", "pilot", "diagnostic")
    }
    for payload in subsets.values():
        verify_subset_payload(payload, source, config)
    diagnostic_rows = subsets["diagnostic"]["rows"]
    direct_rows = subsets["direct"]["rows"]
    semantic_direct = semantic_prompt_contract(direct_rows)
    semantic_diagnostic = semantic_prompt_contract(diagnostic_rows)
    diagnostic_ids = {row["condition_id"] for row in diagnostic_rows}
    semantic_direct_projection = semantic_prompt_contract(
        [row for row in direct_rows if row["condition_id"] in diagnostic_ids]
    )
    endpoint_coverage = _endpoint_coverage(diagnostic_rows)
    parser = parser_contract()
    mapping_failures = _choice_mapping_pair_failures(diagnostic_rows)
    constant_failures = _constant_label_failures(diagnostic_rows)
    source_summary = {
        "study_id": SOURCE_STUDY_ID,
        "config_sha256": source_manifest["config_sha256"],
        "dataset_sha256": source_manifest["dataset_sha256"],
        "row_count": source_manifest["row_count"],
        "locked_row_count": config["source"]["locked_row_count"],
        "v61_cpu_audit_status": v61_audit["status"],
    }
    subset_summary = {}
    for selection, payload in subsets.items():
        rows = payload["rows"]
        subset_summary[selection] = {
            "content_sha256": payload["content_sha256"],
            "row_count": len(rows),
            "condition_ids": _row_ids(rows),
            "condition_ids_sha256": canonical_sha256(_row_ids(rows)),
            "semantic_prompt_contract": semantic_prompt_contract(rows),
            "split_counts": {
                split: sum(row["split"] == split for row in rows)
                for split in sorted({row["split"] for row in rows})
            },
        }
    body = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "status": "cpu_controls_complete_no_model_execution",
        "interpretation": "No model forwards have occurred on this branch yet.",
        "source": source_summary,
        "model": config["model"],
        "subsets": subset_summary,
        "diagnostic_families": list(DIAGNOSTIC_FAMILIES),
        "endpoint_coverage": endpoint_coverage,
        "parser_contract": parser,
        "choice_mapping_pair_failures": mapping_failures,
        "constant_label_pair_failures": constant_failures,
        "semantic_parity": {
            "direct_locked": semantic_direct,
            "direct_diagnostic_projection": semantic_direct_projection,
            "thinking_diagnostic": semantic_diagnostic,
            "direct_thinking_match_on_diagnostic": semantic_direct_projection["contract_sha256"]
            == semantic_diagnostic["contract_sha256"],
        },
        "endpoint_definitions": config["endpoints"],
        "budget": config["execution"],
        "source_v61_cpu_audit": v61_audit,
        "model_forwards": 0,
        "gpu_stages_ran": False,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def verify_manifest(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    if not _hash_valid(manifest):
        raise RuntimeError("V6.2 subset manifest hash mismatch")
    if manifest.get("study_id") != STUDY_ID:
        raise RuntimeError("V6.2 subset manifest study ID mismatch")
    expected = build_manifest(config)
    if manifest != expected:
        raise RuntimeError("committed V6.2 subset manifest is not deterministic")


def model_record(
    row: dict[str, Any],
    result: dict[str, Any],
    *,
    materialization: dict[str, Any] | None = None,
    generated: bool = False,
) -> dict[str, Any]:
    selected = result.get("legal_choice")
    selected_index = row["choice_mapping"].get(selected) if selected else None
    record = {
        "condition_id": row["condition_id"],
        "source_study_id": row["study_id"],
        "experiment_family": row["experiment_family"],
        "task_kind": row["task_kind"],
        "split": row["split"],
        "game_id": row["game_id"],
        "matched_group_id": row["matched_group_id"],
        "expected": row["expected_choice"],
        "selected": selected,
        "expected_index": row["expected_index"],
        "expected_value": row["expected_value"],
        "expected_semantic": row["expected_semantic"],
        "selected_index": selected_index,
        "choice_mapping": row["choice_mapping"],
        "vo_i": row["vo_i"],
        "swapped_labels": row["swapped_labels"],
        "correct": selected == row["expected_choice"],
        "parseable": selected in CHOICES,
        "formatting_compliant": result.get("formatting_compliant", True),
        "result": result,
        "materialization": materialization or {},
    }
    record.update(row["condition_factors"])
    if generated:
        record["generated_text"] = str(result.get("generated_text", ""))
    return record


def validate_records(
    payload: dict[str, Any],
    subset: dict[str, Any],
    config: dict[str, Any],
    *,
    thinking: bool,
    protocol_commit_sha: str | None = None,
    preflight_content_sha256: str | None = None,
) -> None:
    if not _hash_valid(payload):
        raise RuntimeError("model artifact content hash mismatch")
    expected_rows = {row["condition_id"]: row for row in subset["rows"]}
    if payload.get("source_dataset_sha256") != subset["source_dataset_sha256"]:
        raise RuntimeError("model artifact source dataset hash mismatch")
    if payload.get("subset_sha256") != subset["content_sha256"]:
        raise RuntimeError("model artifact subset hash mismatch")
    if payload.get("config_sha256") != canonical_sha256(config):
        raise RuntimeError("model artifact config hash mismatch")
    if (
        protocol_commit_sha is not None
        and payload.get("protocol_commit_sha") != protocol_commit_sha
    ):
        raise RuntimeError("model artifact protocol commit mismatch")
    if (
        preflight_content_sha256 is not None
        and payload.get("preflight_content_sha256") != preflight_content_sha256
    ):
        raise RuntimeError("model artifact preflight binding mismatch")
    if payload.get("thinking") is not thinking:
        raise RuntimeError("model artifact thinking flag mismatch")
    records = payload.get("records", [])
    if {record.get("condition_id") for record in records} != set(expected_rows):
        raise RuntimeError("model artifact rows do not match the frozen subset")
    if not isinstance(payload.get("preflight_content_sha256"), str):
        raise RuntimeError("model artifact is missing its preflight binding")
    if thinking:
        metadata = payload.get("metadata", {})
        if metadata.get("reasoning_effort") != config["conditions"]["qwen38_thinking"][
            "reasoning_effort"
        ]:
            raise RuntimeError("model artifact reasoning effort mismatch")
        generation = metadata.get("generation_settings", {})
        expected_generation = {
            "do_sample": config["behavior"]["thinking_do_sample"],
            "temperature": config["behavior"]["thinking_temperature"],
            "top_p": config["behavior"]["thinking_top_p"],
            "top_k": config["behavior"]["thinking_top_k"],
            "repetition_penalty": config["behavior"]["thinking_repetition_penalty"],
            "seed": config["behavior"]["thinking_seed"],
        }
        if any(generation.get(key) != value for key, value in expected_generation.items()):
            raise RuntimeError("model artifact generation settings mismatch")
    for record in records:
        row = expected_rows[record["condition_id"]]
        if record.get("expected") != row["expected_choice"]:
            raise RuntimeError("expected label changed in model artifact")
        if record.get("expected_index") != row["expected_index"]:
            raise RuntimeError("expected index changed in model artifact")
        if record.get("choice_mapping") != row["choice_mapping"]:
            raise RuntimeError("choice mapping changed in model artifact")
        selected = record.get("selected")
        expected_index = row["choice_mapping"].get(selected) if selected else None
        if record.get("selected_index") != expected_index:
            raise RuntimeError("selected index mismatch in model artifact")
        if record.get("correct") != (selected == row["expected_choice"]):
            raise RuntimeError("correctness mismatch in model artifact")
        if thinking and record.get("parseable") is not (selected in CHOICES):
            raise RuntimeError("thinking parser status mismatch in model artifact")
