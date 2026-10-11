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
    source = (ROOT / "modal_semantic_action_mechanisms.py").read_text()
    cpu_definition = source.split("cpu_image =", 1)[1].split("gpu_image =", 1)[0]
    assert '"jinja2==3.1.6"' in cpu_definition


def test_aligned_followup_preserves_parent_dataset_and_every_anchor(runner):
    from jspace_policy.semantic_action_mechanisms import paired_rows
    from jspace_policy.sprint_runtime import load_json_file

    directory = ROOT / "experiments/semantic_action_mechanisms/aligned"
    protocol = load_json_file(directory / "protocol.json")
    prepared = load_json_file(directory / "prepared.json.gz")
    assert protocol["maximum_cumulative_reserved_usd"] < 5
    assert set(w for p in protocol["concept_pairs"]["discovery"] for w in p).isdisjoint(
        w for p in protocol["concept_pairs"]["locked"] for w in p
    )
    for split, value in prepared.items():
        runner.verify_prepared(value)
        assert value["sha256"] == protocol["prepared_sha256"][split]
        assert len(value["queries"]) == 576
        pairs = paired_rows(value["dataset"], "state")
        assert len(pairs) == 48
        for source, recipient in pairs:
            sq = value["queries"][source["id"] + ":action"]
            rq = value["queries"][recipient["id"] + ":action"]
            assert len(sq["input_ids"]) == len(rq["input_ids"])
            assert sq["anchors"] == rq["anchors"]
    frozen = json.loads(
        (ROOT / "experiments/semantic_action_mechanisms/protocol.json").read_text()
    )
    assert {s: make_dataset(s)["sha256"] for s in ("discovery", "locked")} == frozen[
        "dataset_sha256"
    ]


def test_aligned_decorators_enforce_reduced_timeout_reservations():
    tree = ast.parse((ROOT / "modal_semantic_mechanisms_aligned.py").read_text())
    expected = {"behavior_aligned": 400, "discovery_aligned": 400, "locked_aligned": 500}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in expected:
            kwargs = {
                kw.arg: ast.literal_eval(kw.value)
                for kw in node.decorator_list[0].keywords
                if kw.arg in ("timeout", "retries")
            }
            assert kwargs == {"timeout": expected[node.name], "retries": 0}


def test_recovery_preserves_all_completed_sites_and_rejects_omission():
    from copy import deepcopy

    from jspace_policy.sprint_runtime import load_json_file

    spec = importlib.util.spec_from_file_location(
        "mechanism_recovery", ROOT / "modal_semantic_mechanisms_recover.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded_prepared, loaded_completed, loaded_protocol = module.load_recovery_inputs()
    assert len(loaded_completed["records"]) == 312
    assert (
        loaded_prepared["discovery"]["sha256"]
        == loaded_protocol["prepared_sha256"]["discovery"]
    )
    directory = (
        ROOT / "results/semantic_action_mechanisms/semantic-mechanisms-qwen36-gha-38143348343"
    )
    prepared = load_json_file(directory / "prepared.json.gz")["discovery"]
    completed = load_json_file(directory / "recovered_sites.json.gz")
    module.validate_completed(prepared, completed)
    bad = deepcopy(completed)
    bad["records"].pop()
    with pytest.raises(RuntimeError, match="omitted"):
        module.validate_completed(prepared, bad)
    protocol = load_json_file(
        ROOT / "experiments/semantic_action_mechanisms/aligned/recovery_protocol.json"
    )
    assert protocol["maximum_cumulative_reserved_usd"] < 5
    assert protocol["scientific_changes"] is False


def test_limited_heldout_sample_is_balanced_and_uses_every_family(runner):
    from collections import Counter

    from jspace_policy.semantic_action_mechanisms import paired_rows
    from jspace_policy.sprint_runtime import load_json_file

    prepared = load_json_file(
        ROOT / "experiments/semantic_action_mechanisms/aligned/prepared.json.gz"
    )
    pairs = runner.balanced_locked_pairs(paired_rows(prepared["locked"]["dataset"], "state"))
    assert len(pairs) == len({s["family"] for s, _ in pairs}) == 12
    assert Counter((s["target_index"], s["permutation"]) for s, _ in pairs) == {
        (0, 0): 3,
        (0, 1): 3,
        (1, 0): 3,
        (1, 1): 3,
    }
    assert sum(s["expected_action"] == "A" for s, _ in pairs) == 6
