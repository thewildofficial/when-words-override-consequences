from __future__ import annotations

import json
from pathlib import Path
from runpy import run_path

from jspace_policy.v6_2_reasoning_capability import (
    select_rows,
    source_dataset,
    subset_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "configs/v6.2/reasoning_capability_control/experiment.json").read_text(
        encoding="utf-8"
    )
)
MODAL = run_path(ROOT / "modal_v6_2_reasoning_capability.py")


class _ContractTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, *, enable_thinking=False, **_kwargs):
        prefix = "<think>\n" if enable_thinking else ""
        return prefix + json.dumps(messages, sort_keys=True)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text.endswith("A"):
            return self.encode(text[:-1]) + [101]
        if text.endswith("B"):
            return self.encode(text[:-1]) + [102]
        return [ord(character) for character in text]

    def decode(self, token_ids, **_kwargs):
        return {101: "A", 102: "B"}[token_ids[0]]


def test_preflight_detects_native_thinking_render_and_direct_label_tokens() -> None:
    source, _source_config, _source_manifest = source_dataset(CONFIG)
    row = select_rows(source, "pilot")[0]
    tokenizer = _ContractTokenizer()
    sample = MODAL["_preflight_sample"](tokenizer, row)
    assert sample["passed"] is True
    assert sample["off"]["thinking_markers"] == []
    assert "<think>" in sample["on"]["thinking_markers"]


def test_direct_and_thinking_query_contracts_share_task_semantics() -> None:
    source, _source_config, _source_manifest = source_dataset(CONFIG)
    subset = subset_payload(source, CONFIG, "pilot")
    tokenizer = _ContractTokenizer()
    direct, _ = MODAL["_query_rows"](tokenizer, subset, thinking=False)
    thinking, _ = MODAL["_query_rows"](tokenizer, subset, thinking=True)
    direct_contract = MODAL["_query_contract"](direct, semantic=True)
    thinking_contract = MODAL["_query_contract"](thinking, semantic=True)
    assert direct_contract["query_ids_sha256"] == thinking_contract["query_ids_sha256"]
    assert direct_contract["messages_sha256"] == thinking_contract["messages_sha256"]
    assert direct_contract["candidate_labels_sha256"] == thinking_contract[
        "candidate_labels_sha256"
    ]


def test_thinking_generation_parser_is_not_used_for_direct_logits() -> None:
    assert MODAL["parse_thinking_final"]("FINAL: A") == "A"
    assert MODAL["parse_thinking_final"]("FINAL: A\nFINAL: B") is None
    assert MODAL["_validate_config"](CONFIG) is None
