from copy import deepcopy

import pytest

from jspace_policy.semantic_action_mechanisms import (
    CONCEPT_PAIRS,
    LOCKED_PAIRS,
    analyze,
    make_dataset,
    paired_rows,
    queries_for,
    select_mechanistic_contrast,
    verify_dataset,
)


def oracle_scores(dataset):
    scores = {}
    for row in dataset["rows"]:
        for query in queries_for(row):
            scores[query["id"]] = {
                "choice": query["expected"],
                "logits": [
                    4.0 if label == query["expected"] else 0.0 for label in query["labels"]
                ],
            }
    return {"status": "scored", "dataset_sha256": dataset["sha256"], "scores": scores}


def test_state_intervention_preserves_oracle_and_other_facts():
    dataset = make_dataset("discovery")
    verify_dataset(dataset)
    for source, recipient in paired_rows(dataset, "state"):
        for key in ("target", "responses", "surfaces", "expected_action", "permutation"):
            assert source[key] == recipient[key]
        assert source["anchors"]["policy_end"] == recipient["anchors"]["policy_end"]
        assert source["prompt"].replace(source["anchors"]["state_end"], "STATE") == (
            recipient["prompt"].replace(recipient["anchors"]["state_end"], "STATE")
        )


def test_binding_intervention_preserves_semantics_and_goal_after_patch_anchor():
    for source, recipient in paired_rows(make_dataset("locked"), "binding"):
        assert source["responses"] == recipient["responses"]
        assert source["expected_action"] == recipient["expected_action"]
        assert "Option A (" in source["anchors"]["policy_end"]
        assert "Option A (" not in recipient["anchors"]["policy_end"]
        for row in (source, recipient):
            assert row["prompt"].index(row["anchors"]["policy_end"]) < (
                row["prompt"].index(row["anchors"]["payoff_end"])
            )


def test_lexical_holdout_and_permutation_balance():
    assert {word for pair in CONCEPT_PAIRS for word in pair}.isdisjoint(
        {word for pair in LOCKED_PAIRS for word in pair}
    )
    dataset = make_dataset("locked")
    verify_dataset(dataset)
    assert len(paired_rows(dataset, "state")) == 48
    assert len(paired_rows(dataset, "binding")) == 48
    assert sum(r["expected_action"] == "A" for r in dataset["rows"]) == 560
    assert make_dataset("locked")["sha256"] == dataset["sha256"]


def test_analyzer_finds_truth_sensitive_error_and_uses_family_clusters():
    dataset = make_dataset("discovery")
    scores = oracle_scores(dataset)
    for row in dataset["rows"]:
        if (
            row["surface"] == "assertion"
            and row["history"] == "redundant"
            and row["state"] == "opposed"
            and row["layout"] == "prose"
            and not row["binding"]
        ):
            correct = row["expected_action"]
            scores["scores"][row["id"] + ":action"] = {
                "choice": "B" if correct == "A" else "A",
                "logits": [0.0 if label == correct else 4.0 for label in ("A", "B")],
            }
    result = analyze(dataset, scores)
    assert result["contrasts"]["assertion:matching-opposed"]["families"] == 12
    assert result["state_surface_interaction"]["accuracy_difference"] == 1.0
    assert select_mechanistic_contrast(result) == "state"
    assert result["report_accuracy"] == 1.0


def test_analyzer_rejects_missing_scores_and_gate_failure():
    dataset = make_dataset("discovery")
    result = oracle_scores(dataset)
    result["scores"].pop(next(iter(result["scores"])))
    with pytest.raises(ValueError, match="missing"):
        analyze(dataset, result)
    result = oracle_scores(dataset)
    result["status"] = "instrument_gate_failed"
    with pytest.raises(ValueError, match="unscored"):
        analyze(dataset, result)
    changed = deepcopy(dataset)
    changed["rows"][0]["target"] = "INVALID"
    with pytest.raises(AssertionError):
        verify_dataset(changed)


def test_ceiling_does_not_select_mechanistic_stage():
    dataset = make_dataset("discovery")
    assert select_mechanistic_contrast(analyze(dataset, oracle_scores(dataset))) is None
