import ast
import importlib.util
import json
from pathlib import Path

import pytest

from jspace_policy.semantic_action_mechanisms import MODEL_ID, MODEL_REVISION, make_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "semantic_modal", ROOT / "modal_semantic_action_mechanisms.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_executable_reservation_matches_frozen_protocol(runner):
    protocol = json.loads(
        (ROOT / "experiments/semantic_action_mechanisms/protocol.json").read_text()
    )
    assert runner.TIMEOUTS == protocol["timeouts_seconds"]
    assert runner.MODEL_ID == protocol["model_id"] == MODEL_ID
    assert runner.MODEL_REVISION == protocol["revision"] == MODEL_REVISION
    assert sum(runner.CEILINGS.values()) < protocol["hard_ceiling_usd"] == 5.0
    tree = ast.parse((ROOT / "modal_semantic_action_mechanisms.py").read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for stage in ("behavior", "discovery", "locked"):
        decorator = functions[stage + "_remote"].decorator_list[0]
        kwargs = {
            kw.arg: ast.literal_eval(kw.value)
            for kw in decorator.keywords
            if kw.arg in ("timeout", "retries", "max_containers", "memory", "cpu")
        }
        assert kwargs == {
            "timeout": runner.TIMEOUTS[stage],
            "retries": 0,
            "max_containers": 1,
            "memory": 65536,
            "cpu": 8,
        }


def test_queries_inventory_and_goal_position():
    dataset = make_dataset("discovery")
    from jspace_policy.semantic_action_mechanisms import queries_for

    assert sum(len(queries_for(r)) for r in dataset["rows"]) == 3360
    for row in dataset["rows"]:
        prompt = row["prompt"]
        for name in ("policy_end", "state_end"):
            text = row["anchors"][name]
            if text:
                assert prompt.index(text) + len(text) < prompt.index(
                    row["anchors"]["payoff_end"]
                )


def test_identity_failure_prevents_site_selection(runner):
    result = {"status": "identity_gate_failed", "records": [], "identity_error": 0.5}
    assert runner.patch_analysis(result)["selected"] is None
    assert not runner.locked_verdict(result, {"sites": []})["supported"]


def test_wrong_game_and_reports_are_required_for_locked_claim(runner):
    raw = {
        "status": "patched",
        "identity_error": 0,
        "records": [{"reports": [{"expected": "X", "patched": {"choice": "Y"}}]}],
    }
    summary = {
        "sites": [
            {
                "families": 12,
                "failures": 10,
                "repair_fraction": 0.8,
                "directional_margin_change": 2,
                "sign_flip_p": 0.001,
                "forward_minus_wrong_game": 0,
            }
        ]
    }
    verdict = runner.locked_verdict(raw, summary)
    assert not verdict["supported"]
    assert not verdict["gates"]["report_preservation"]
    assert not verdict["gates"]["wrong_game_specificity"]


def test_workflow_always_archives_and_does_not_reset_ledger():
    workflow = (ROOT / ".github/workflows/semantic-action-mechanisms.yml").read_text()
    assert "always()" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "retention-days: 90" in workflow
    assert "rm " not in workflow
