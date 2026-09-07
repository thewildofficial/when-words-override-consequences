from __future__ import annotations

import json
from pathlib import Path
from runpy import run_path

from jspace_policy.v6_1_epistemic_repair import dataset_payload

_MODAL = run_path(Path(__file__).resolve().parents[1] / "modal_v6_1_epistemic_repair.py")
_preceding_assistant_content = _MODAL["_preceding_assistant_content"]
_report_materialization = _MODAL["_report_materialization"]
_query_rows = _MODAL["_query_rows"]
_semantic_preflight_contract = _MODAL["_semantic_preflight_contract"]
_CONFIG = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "configs/v6.1/epistemic_repair/experiment.json"
    ).read_text(
        encoding="utf-8"
    )
)


def _query(*, content: str, rendered: str, token_ids: list[int]) -> dict:
    return {
        "query_id": "q-1",
        "messages": [{"role": "user", "content": content}],
        "rendered": rendered,
        "prompt_token_ids": token_ids,
        "candidate_labels": ["A", "B"],
        "candidate_token_ids": [token_ids[-1] + 1, token_ids[-1] + 2],
    }


def test_replication_contract_ignores_valid_tokenizer_and_template_differences() -> None:
    primary = _semantic_preflight_contract(
        [_query(content="same semantic prompt", rendered="template 36", token_ids=[1, 2])]
    )
    replication = _semantic_preflight_contract(
        [_query(content="same semantic prompt", rendered="template 38", token_ids=[101, 202])]
    )
    assert primary == replication


def test_replication_contract_rejects_changed_message_content() -> None:
    primary = _semantic_preflight_contract(
        [_query(content="prompt A", rendered="template", token_ids=[1, 2])]
    )
    changed = _semantic_preflight_contract(
        [_query(content="prompt B", rendered="template", token_ids=[1, 2])]
    )
    assert primary != changed


def test_blackbox_trajectory_preserves_raw_unparseable_assistant_text() -> None:
    result = {"legal_choice": None, "generated_text": "I cannot select A or B."}
    assert _preceding_assistant_content(result, mode="blackbox") == result[
        "generated_text"
    ]
    assert _preceding_assistant_content(
        {"legal_choice": "A"}, mode="behavior"
    ) == "A"


def test_unparseable_self_generated_report_is_retained_as_a_failed_source() -> None:
    row = {
        "scaffold_report_id": "report-1",
        "prompt": (
            "Choose the next action. Prior label: {SELF_REPORT_LABEL}; "
            "prior concept: {SELF_REPORT_CONCEPT}."
        ),
    }
    report_row = {
        "prompt": "Report the receiver's choice.",
        "choice_mapping": {"A": 0, "B": 1},
        "game_certificate": {"concepts": ["KITE", "MOSS"]},
        "expected_choice": "A",
    }
    prompt, materialization = _report_materialization(
        row,
        {"report-1": {"legal_choice": None, "generated_text": "free text"}},
        {"report-1": report_row},
        mode="blackbox",
    )
    assert "{SELF_REPORT_LABEL}" not in prompt
    assert "{SELF_REPORT_CONCEPT}" not in prompt
    assert "[unparseable prior report]" in prompt
    assert materialization["source_report_parseable"] is False
    assert materialization["source_report_assistant_content"] == "free text"


class _ContractTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **_kwargs):
        return json.dumps(messages, sort_keys=True)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text.endswith("A"):
            return self.encode(text[:-1]) + [101]
        if text.endswith("B"):
            return self.encode(text[:-1]) + [102]
        return [ord(character) for character in text]

    def decode(self, token_ids, **_kwargs):
        return {101: "A", 102: "B"}[token_ids[0]]


def test_blackbox_query_rows_keep_raw_text_for_both_dynamic_trajectories() -> None:
    rows = dataset_payload(_CONFIG)["rows"]
    self_action = next(
        row
        for row in rows
        if row.get("scaffold_source") == "self_generated" and row["split"] == "locked"
    )
    source_report = next(
        row for row in rows if row["condition_id"] == self_action["scaffold_report_id"]
    )
    after_report = next(
        row
        for row in rows
        if row.get("trajectory_condition") == "action_then_report" and row["split"] == "locked"
    )
    action = next(
        row for row in rows if row["condition_id"] == after_report["trajectory_action_row_id"]
    )
    subset = {
        row["condition_id"]: row
        for row in (source_report, self_action, action, after_report)
    }
    queries, _metadata = _query_rows(
        _ContractTokenizer(),
        {"rows": list(subset.values())},
        static_outputs={
            source_report["condition_id"]: {
                "legal_choice": None,
                "generated_text": "unparseable report text",
            },
            action["condition_id"]: {
                "legal_choice": None,
                "generated_text": "unparseable action text",
            },
        },
        mode="blackbox",
    )
    by_id = {query["query_id"]: query for query in queries}
    self_messages = by_id[self_action["condition_id"]]["messages"]
    after_messages = by_id[after_report["condition_id"]]["messages"]
    assert self_messages[2] == {
        "role": "assistant",
        "content": "unparseable report text",
    }
    assert "previous report interaction is shown above" in self_messages[3]["content"]
    assert after_messages[2] == {
        "role": "assistant",
        "content": "unparseable action text",
    }
