from __future__ import annotations

from pathlib import Path
from runpy import run_path

_MODAL = run_path(Path(__file__).resolve().parents[1] / "modal_v6_1_epistemic_repair.py")
_semantic_preflight_contract = _MODAL["_semantic_preflight_contract"]


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
