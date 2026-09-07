from __future__ import annotations

import copy
import json
from pathlib import Path
from runpy import run_path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = run_path(ROOT / "scripts/analyze_v6_2_reasoning_capability.py")
CONFIG = json.loads(
    (ROOT / "configs/v6.2/reasoning_capability_control/experiment.json").read_text(
        encoding="utf-8"
    )
)


def _h3_row(
    game_id: str,
    prompt_mode: str,
    evidence_mode: str,
    *,
    selected_index: int,
    correct: bool,
) -> dict[str, object]:
    expected_value = 0 if evidence_mode == "independent" else 1
    return {
        "task_kind": "report",
        "experiment_family": "evidence_update",
        "evidence_direction": "prior",
        "evidence_count": 4,
        "game_id": game_id,
        "prior": 0,
        "message_source": "same_message",
        "evidence_prompt_mode": prompt_mode,
        "evidence_mode": evidence_mode,
        "expected_value": expected_value,
        "selected_index": selected_index,
        "correct": correct,
    }


def test_h3_gate_rejects_raw_wrong_wrong_contrasts() -> None:
    rows = [
        _h3_row(
            f"game-{mode}",
            mode,
            evidence_mode,
            selected_index=index,
            correct=False,
        )
        for mode in ("explicit_rule", "provenance_only")
        for evidence_mode, index in (("independent", 0), ("copied", 1))
    ]
    config = copy.deepcopy(CONFIG)
    config["gates"]["minimum_accuracy"] = 0.0
    result = ANALYSIS["_h3"](rows, config)
    assert result["independence_contrast_by_prompt_mode"]["explicit_rule"]["mean"] == 1.0
    assert result["both_correct_and_switch_by_prompt_mode"]["explicit_rule"]["mean"] == 0.0
    assert result["status"] == "falsified"


def test_h1_unparseable_invariance_pair_is_a_failure_cell() -> None:
    rows = [
        {
            "task_kind": "action",
            "experiment_family": "ledger_binding",
            "game_id": "game-1",
            "actual_receiver_belief": actual,
            "modeled_receiver_belief": 0,
            "selected_index": selected,
            "correct": selected is not None,
        }
        for actual, selected in ((0, None), (1, 1))
    ]
    result = ANALYSIS["_h1"](rows, CONFIG)
    assert result["n_invariance_pair_cells"] == 1
    assert result["actual_belief_invariance"]["mean"] == 0.0


def test_paired_sign_flip_aggregates_cells_by_game() -> None:
    result = ANALYSIS["_paired_sign_flip"](
        [("game-a", 1.0), ("game-a", -1.0), ("game-b", 1.0)],
        seed=2,
        draws=200,
    )
    assert result["status"] == "complete"
    assert result["n_cells"] == 3
    assert result["n_clusters"] == 2
    assert result["observed_mean"] == 0.5


def test_v61_reference_uses_strict_h3_reanalysis() -> None:
    reference = ANALYSIS["_reference_summary"]()
    derived = json.loads(
        (
            ROOT / "results/v6_2_reasoning_capability_control/"
            "v61_reference_endpoints.json"
        ).read_text(encoding="utf-8")
    )
    assert reference["endpoints"]["H3_explicit_contrast"] == derived["endpoints"][
        "H3_explicit_both_correct_and_switch"
    ]["mean"]
    assert reference["endpoints"]["H3_provenance_contrast"] == derived["endpoints"][
        "H3_provenance_both_correct_and_switch"
    ]["mean"]
    assert reference["strict_h3_reference"]["content_sha256"] == derived[
        "content_sha256"
    ]
    assert (
        reference["strict_h3_reference"]["source_raw_payload_content_sha256"]
        == derived["source"]["raw_payload_content_sha256"]
    )


def test_paired_endpoint_test_runs_when_one_family_gate_fails() -> None:
    def policy_row(policy: str, selected_index: int) -> dict[str, object]:
        return {
            "experiment_family": "policy_composition",
            "game_id": "game-1",
            "modeled_receiver_belief": 0,
            "receiver_policy": policy,
            "selected_index": selected_index,
            "correct": True,
        }

    direct = [policy_row("literal", 0), policy_row("contrarian", 0)]
    thinking = [policy_row("literal", 0), policy_row("contrarian", 1)]
    result = ANALYSIS["_paired_endpoint_comparisons"](
        direct,
        thinking,
        {"policy_composition_endpoints": {"status": "falsified"}},
        {"policy_composition_endpoints": {"status": "supported"}},
        CONFIG,
    )["H2_policy_pair_rate"]
    assert result["family_gate"] == {
        "direct": "falsified",
        "thinking": "supported",
        "both_supported": False,
    }
    assert result["paired_cluster_sign_flip"]["status"] == "complete"
    assert result["paired_cluster_sign_flip"]["observed_mean"] == 1.0
