from __future__ import annotations

import json
from pathlib import Path
from runpy import run_path

from jspace_policy.v7_epistemic_agentic_strategy import canonical_sha256, dataset_payload

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/v7/epistemic_agentic_strategy/experiment.json"
analyze = run_path(ROOT / "scripts/analyze_v7_epistemic_agentic_strategy.py")["analyze"]


def _perfect_records(dataset: dict) -> list[dict]:
    records = []
    for row in dataset["rows"]:
        if row["split"] != "locked":
            continue
        selected = row["expected_choice"]
        records.append(
            {
                "condition_id": row["condition_id"],
                "experiment_family": row["experiment_family"],
                "task_kind": row["task_kind"],
                "split": row["split"],
                "game_id": row["game_id"],
                "matched_group_id": row["matched_group_id"],
                "shared_prefix_id": row["shared_prefix_id"],
                "condition_factors": row["condition_factors"],
                "surface": row["surface"],
                "report_target": row.get("report_target"),
                "expected": selected,
                "selected": selected,
                "expected_index": row["expected_index"],
                "selected_index": row["expected_index"],
                "choice_mapping": row["choice_mapping"],
                "expected_semantic": row["expected_semantic"],
                "regret": 0.0 if row["task_kind"] == "action" else None,
                "correct": True,
                "parseable": True,
                "formatting_compliant": True,
                "result": {"legal_choice": selected, "formatting_compliant": True},
            }
        )
    return records


def _write_behavior(
    tmp_path: Path, config: dict, dataset: dict, records: list[dict]
) -> tuple[Path, Path]:
    payload = {
        "schema_version": 1,
        "study_id": "V7-EAS-1",
        "status": "forced_choice_logit_behavior_complete",
        "metadata": {
            "config_sha256": canonical_sha256(config),
            "dataset_sha256": dataset["content_sha256"],
        },
        "records": records,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    behavior_path = tmp_path / "behavior.json"
    behavior_path.write_text(json.dumps(payload), encoding="utf-8")
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    return behavior_path, dataset_path


def test_analyzer_accepts_a_perfect_locked_artifact(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    dataset = dataset_payload(config)
    records = _perfect_records(dataset)
    behavior_path, dataset_path = _write_behavior(tmp_path, config, dataset, records)

    result = analyze(CONFIG_PATH, dataset_path, behavior_path)

    expected = [row for row in dataset["rows"] if row["split"] == "locked"]
    assert result["locked_record_count"] == len(expected)
    assert all(family["status"] == "supported" for family in result["families"].values())
    publicity = result["families"]["publicity_threshold"]
    assert (
        publicity["identifying_pairs"]["shared_private_to_public_announcement"]["mean"] == 1.0
    )
    assert (
        publicity["identifying_pairs"]["shared_private_to_public_announcement"]["n_clusters"]
        == 4
    )
    commitment = result["families"]["credible_commitment"]
    assert commitment["identifying_pairs"]["public_to_tool_invariance"]["mean"] == 1.0
    acquisition = result["families"]["information_acquisition"]
    assert acquisition["targeted_choice_rate"]["mean"] == 1.0
    assert acquisition["mean_regret"]["mean"] == 0.0
    assert (
        acquisition["identifying_pairs"]["world_to_world_known_switch"]["n_clusters"] == 4
    )
    assert set(acquisition["gates"]) == {
        "report_accuracy",
        "action_accuracy",
        "surface_invariance",
        "targeted_choice",
        "known_target_act",
        "world_pair_switch",
        "belief_pair_switch",
        "policy_pair_switch",
    }
    assert all(acquisition["gates"].values())


def test_analyzer_does_not_let_aggregate_h5_metrics_rescue_a_failed_pair(
    tmp_path: Path,
) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    dataset = dataset_payload(config)
    records = _perfect_records(dataset)
    for record in records:
        if (
            record["experiment_family"] != "information_acquisition"
            or record["task_kind"] != "action"
            or record["condition_factors"]["uncertainty_type"] != "opponent_policy"
        ):
            continue
        mapping = record["choice_mapping"]
        selected = next(label for label, index in mapping.items() if index == 0)
        record["selected"] = selected
        record["selected_index"] = 0
        record["correct"] = selected == record["expected"]
        record["regret"] = 1.0
        record["result"] = {"legal_choice": selected, "formatting_compliant": True}

    behavior_path, dataset_path = _write_behavior(tmp_path, config, dataset, records)
    result = analyze(CONFIG_PATH, dataset_path, behavior_path)
    acquisition = result["families"]["information_acquisition"]
    assert acquisition["targeted_choice_rate"]["mean"] > 0.75
    assert acquisition["known_target_act_rate"]["mean"] >= 0.75
    assert acquisition["gates"]["world_pair_switch"]
    assert acquisition["gates"]["belief_pair_switch"]
    assert not acquisition["gates"]["policy_pair_switch"]
    assert acquisition["status"] == "falsified"

