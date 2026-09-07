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

    def apply_chat_template(
        self,
        messages,
        *,
        enable_thinking=False,
        chat_template_kwargs=None,
        **_kwargs,
    ):
        if chat_template_kwargs:
            enable_thinking = chat_template_kwargs.get("enable_thinking", enable_thinking)
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
    assert direct_contract == thinking_contract
    assert set(direct_contract) == {
        "query_count",
        "query_ids_sha256",
        "messages_sha256",
        "candidate_labels_sha256",
    }

    direct_full = MODAL["_query_contract"](direct, semantic=False)
    thinking_full = MODAL["_query_contract"](thinking, semantic=False)
    assert direct_full["rendered_sha256"] != thinking_full["rendered_sha256"]
    assert direct_full["prompt_token_ids_sha256"] != thinking_full["prompt_token_ids_sha256"]

    preflight_contract = MODAL["_preflight_contract"](direct, thinking)
    assert preflight_contract["semantic_match"] is True


def test_thinking_generation_parser_is_not_used_for_direct_logits() -> None:
    assert MODAL["parse_thinking_final"]("FINAL: A") == "A"
    assert MODAL["parse_thinking_final"]("FINAL: A\nFINAL: B") is None
    assert MODAL["_validate_config"](CONFIG) is None


def test_modal_hard_timeouts_equal_frozen_stage_limits() -> None:
    assert MODAL["MODAL_HARD_TIMEOUTS"] == {
        stage: spec["timeout_seconds"]
        for stage, spec in CONFIG["execution"]["stage_limits"].items()
    }


def test_thinking_render_passes_frozen_reasoning_effort() -> None:
    class _RecordingTokenizer(_ContractTokenizer):
        def __init__(self):
            self.template_kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            nested = kwargs.get("chat_template_kwargs", {})
            self.template_kwargs = {
                key: kwargs.get(key, nested.get(key))
                for key in ("enable_thinking", "preserve_thinking", "reasoning_effort")
            }
            return super().apply_chat_template(messages, **kwargs)

    tokenizer = _RecordingTokenizer()
    row = select_rows(source_dataset(CONFIG)[0], "pilot")[0]
    MODAL["_prepare_query"](
        tokenizer,
        row,
        thinking=True,
        reasoning_effort=CONFIG["conditions"]["qwen38_thinking"]["reasoning_effort"],
    )
    assert tokenizer.template_kwargs == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_effort": "xhigh",
    }


def test_thinking_generation_uses_the_frozen_sampling_regime() -> None:
    kwargs = MODAL["_thinking_generation_kwargs"](
        CONFIG, max_new_tokens=1536, pad_token_id=0
    )
    assert kwargs == {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "max_new_tokens": 1536,
        "pad_token_id": 0,
    }


def test_direct_query_contract_uses_the_flat_preflight_contract() -> None:
    source, _source_config, _source_manifest = source_dataset(CONFIG)
    subset = subset_payload(source, CONFIG, "pilot")
    tokenizer = _ContractTokenizer()
    queries, _ = MODAL["_query_rows"](tokenizer, subset, thinking=False)
    preflight = {
        "preflight_contract": {
            "direct": MODAL["_query_contract"](queries, semantic=False)
        }
    }
    validated = MODAL["_validate_query_contract"](
        tokenizer,
        subset,
        preflight,
        selection="direct",
        thinking=False,
        reasoning_effort="xhigh",
    )
    assert len(validated) == len(queries)
