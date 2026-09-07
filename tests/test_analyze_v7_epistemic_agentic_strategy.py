from __future__ import annotations

import json
from pathlib import Path
from runpy import run_path

from jspace_policy.v7_epistemic_agentic_strategy import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/v7/epistemic_agentic_strategy/experiment.json"
DATASET_PATH = ROOT / "configs/v7/epistemic_agentic_strategy/dataset.json"
analyze = run_path(ROOT / "scripts/analyze_v7_epistemic_agentic_strategy.py")["analyze"]


def test_analyzer_accepts_a_perfect_locked_artifact(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    expected = [row for row in dataset["rows"] if row["split"] == "locked"]
    records = []
    for row in expected:
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
                "regret": 0.0,
                "correct": True,
                "parseable": True,
                "formatting_compliant": True,
                "result": {"legal_choice": selected, "formatting_compliant": True},
            }
        )
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

    result = analyze(CONFIG_PATH, DATASET_PATH, behavior_path)

    assert result["locked_record_count"] == len(expected)
    assert all(family["status"] == "supported" for family in result["families"].values())
    publicity = result["families"]["publicity_threshold"]
    assert (
        publicity["identifying_pairs"]["shared_private_to_public_announcement"]["mean"] == 1.0
    )
    commitment = result["families"]["credible_commitment"]
    assert commitment["identifying_pairs"]["public_to_tool_invariance"]["mean"] == 1.0
    acquisition = result["families"]["information_acquisition"]
    assert acquisition["targeted_choice_rate"]["mean"] == 1.0
