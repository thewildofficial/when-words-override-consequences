"""Manual Modal entrypoints for the V6.2 reasoning-capability control.

There is intentionally no local entrypoint that chains paid stages.  The
workflow dispatches preflight, direct, pilot, and diagnostic stages separately
so each later launch is an explicit decision made after inspecting the prior
artifact and restoring the persistent cost ledger.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

from jspace_policy.budget import admit_run, append_ledger, estimate_cost
from jspace_policy.v6_2_reasoning_capability import (
    CHOICES,
    DEFAULT_SOURCE_CONFIG,
    DEFAULT_SOURCE_MANIFEST,
    DIAGNOSTIC_FAMILIES,
    STUDY_ID,
    THINKING_MARKERS,
    build_manifest,
    canonical_sha256,
    model_record,
    normalize_task_prompt,
    parse_thinking_final,
    source_dataset,
    subset_payload,
    trajectory_messages,
    verify_subset_payload,
)

RESULT_ROOT = Path("results/v6_2_reasoning_capability_control")
LEDGER_PATH = RESULT_ROOT / "cost_ledger.jsonl"
CONFIG_PATH = Path("configs/v6.2/reasoning_capability_control/experiment.json")
SOURCE_CONFIG_PATH = DEFAULT_SOURCE_CONFIG
SOURCE_MANIFEST_PATH = DEFAULT_SOURCE_MANIFEST

app = modal.App("jspace-v6-2-reasoning-capability-control")
cache = modal.Volume.from_name("jspace-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install(
        "numpy>=2.0",
        "pillow>=11",
        "scikit-learn>=1.6",
        "torch>=2.8",
        "torchvision>=0.23",
        "transformers>=5.5",
        "huggingface_hub>=0.34",
    )
    .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src/jspace_policy", remote_path="/root/jspace_policy")
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, value: object) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite prospective artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config() -> dict[str, Any]:
    config = _read_json(CONFIG_PATH)
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("study_id") != STUDY_ID:
        raise RuntimeError("V6.2 study ID changed")
    if config.get("status") != "preregistered_before_qwen38_execution":
        raise RuntimeError("V6.2 protocol is not prospectively frozen")
    if config["model"]["id"] != "Qwen/Qwen3.8-27B":
        raise RuntimeError("V6.2 model ID changed")
    if config["conditions"]["qwen38_direct"]["thinking"] is not False:
        raise RuntimeError("direct condition must disable thinking")
    if config["conditions"]["qwen38_thinking"]["thinking"] is not True:
        raise RuntimeError("thinking condition must enable thinking")
    if config["behavior"]["unparseable_is_failure"] is not True:
        raise RuntimeError("unparseable generations must remain failures")
    limits = config["execution"]["stage_limits"]
    expected_stages = {
        "preflight-qwen38",
        "qwen38-direct",
        "qwen38-thinking-pilot",
        "qwen38-thinking-diagnostic",
    }
    if set(limits) != expected_stages:
        raise RuntimeError("stage authorization set changed")
    ceiling = 0.0
    for spec in limits.values():
        estimate = estimate_cost(
            spec["gpu"],
            float(spec["timeout_seconds"]),
            cpu_cores=float(spec["cpu_cores"]),
            memory_gib=float(spec["memory_gib"]),
            uncertainty_fraction=float(config["execution"]["uncertainty_fraction"]),
        )
        ceiling += estimate.buffered_usd
    if ceiling > float(config["execution"]["authorization_usd"]):
        raise RuntimeError("worst-case V6.2 stage plan exceeds its authorization")


def _inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _config()
    source, _source_config, _source_manifest = source_dataset(
        config,
        source_config_path=SOURCE_CONFIG_PATH,
        source_manifest_path=SOURCE_MANIFEST_PATH,
    )
    subsets = {
        selection: subset_payload(source, config, selection)
        for selection in ("direct", "pilot", "diagnostic")
    }
    for subset in subsets.values():
        verify_subset_payload(subset, source, config)
    return config, source, subsets


def _render(tokenizer: Any, messages: list[dict[str, str]], *, thinking: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=thinking, **kwargs)
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                chat_template_kwargs={"enable_thinking": thinking},
                **kwargs,
            )
        except TypeError:
            if thinking:
                raise RuntimeError(
                    "Qwen3.8 tokenizer does not expose an explicit thinking switch"
                ) from None
            return tokenizer.apply_chat_template(messages, **kwargs)


def _load_tokenizer(spec: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    import transformers
    from huggingface_hub import model_info

    if "3.8" in str(spec["id"]):
        processor = transformers.AutoProcessor.from_pretrained(
            spec["id"], revision=spec["revision"]
        )
        tokenizer = getattr(processor, "tokenizer", processor)
        loader = "AutoProcessor"
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            spec["id"], revision=spec["revision"]
        )
        loader = "AutoTokenizer"
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        raise RuntimeError("tokenizer has no padding or EOS token")
    resolved = model_info(spec["id"], revision=spec["revision"]).sha
    return tokenizer, {
        "tokenizer_loader": loader,
        "tokenizer_revision_requested": spec["revision"],
        "tokenizer_revision_resolved": resolved,
        "transformers_version": transformers.__version__,
    }


def _continuation_id(tokenizer: Any, rendered: str, answer: str) -> int:
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    full = tokenizer.encode(rendered + answer, add_special_tokens=False)
    if full[: len(prefix)] == prefix and len(full) == len(prefix) + 1:
        return int(full[-1])
    raise ValueError(f"{answer!r} is not one token after the frozen prompt")


def _prepare_query(tokenizer: Any, row: dict[str, Any], *, thinking: bool) -> dict[str, Any]:
    messages = trajectory_messages(row, thinking=thinking)
    rendered = _render(tokenizer, messages, thinking=thinking)
    prompt_token_ids = list(map(int, tokenizer.encode(rendered, add_special_tokens=False)))
    query = {
        "query_id": row["condition_id"],
        "messages": messages,
        "rendered": rendered,
        "prompt_token_ids": prompt_token_ids,
        "candidate_labels": list(row["candidate_labels"]),
        "sequence_length": len(prompt_token_ids),
        "thinking": thinking,
    }
    if not thinking:
        candidate_ids = [
            _continuation_id(tokenizer, rendered, label) for label in row["candidate_labels"]
        ]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(f"candidate token collision: {row['condition_id']}")
        for label, token_id in zip(row["candidate_labels"], candidate_ids, strict=True):
            decoded = tokenizer.decode(
                [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            if decoded != label:
                raise ValueError(f"candidate does not decode exactly to {label!r}")
        query["candidate_token_ids"] = candidate_ids
    return query


def _dynamic_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {
        row["condition_id"]
        for row in rows
        if row.get("scaffold_source") == "self_generated"
        or row.get("trajectory_condition") == "action_then_report"
    }


def _preceding_content(result: dict[str, Any], *, thinking: bool) -> str:
    if thinking:
        return str(result.get("generated_text") or "")
    label = result.get("legal_choice")
    return str(label) if label in CHOICES else ""


def _report_materialization(
    row: dict[str, Any],
    report_outputs: dict[str, dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    *,
    thinking: bool,
) -> tuple[str, dict[str, Any]]:
    report_id = str(row["scaffold_report_id"])
    report_row = row_by_id[report_id]
    report_result = report_outputs[report_id]
    label = report_result.get("legal_choice")
    value = report_row["choice_mapping"].get(label) if label in CHOICES else None
    concepts = report_row["game_certificate"]["concepts"]
    concept = concepts[value] if value is not None else "[unparseable prior report]"
    label_text = label if label in CHOICES else "[unparseable prior report]"
    prompt = (
        row["prompt"]
        .replace("{SELF_REPORT_LABEL}", label_text)
        .replace("{SELF_REPORT_CONCEPT}", concept)
    )
    return prompt, {
        "source_report_id": report_id,
        "source_report_label": label,
        "source_report_value": value,
        "source_report_correct": label == report_row["expected_choice"],
        "source_report_parseable": label in CHOICES,
        "source_report_assistant_content": _preceding_content(report_result, thinking=thinking),
    }


def _query_rows(
    tokenizer: Any,
    subset: dict[str, Any],
    *,
    thinking: bool,
    static_outputs: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = subset["rows"]
    row_by_id = {row["condition_id"]: row for row in rows}
    dynamic = _dynamic_ids(rows)
    queries: list[dict[str, Any]] = []
    materialization: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["condition_id"] in dynamic:
            continue
        queries.append(_prepare_query(tokenizer, row, thinking=thinking))
        materialization[row["condition_id"]] = {"materialization": "static"}
    if static_outputs is None:
        placeholder = "FINAL: A" if thinking else "A"
        static_outputs = {
            row["condition_id"]: {"legal_choice": "A", "generated_text": placeholder}
            for row in rows
            if row["condition_id"] not in dynamic
        }
    for row in rows:
        condition_id = row["condition_id"]
        if condition_id not in dynamic:
            continue
        if row.get("scaffold_source") == "self_generated":
            prompt, metadata = _report_materialization(
                row, static_outputs, row_by_id, thinking=thinking
            )
            report_row = row_by_id[str(row["scaffold_report_id"])]
            messages = trajectory_messages(
                row,
                thinking=thinking,
                first_answer=metadata["source_report_assistant_content"],
                materialized_prompt=prompt,
                preceding_prompt=report_row["prompt"],
            )
        else:
            action_row = row_by_id[str(row["trajectory_action_row_id"])]
            action_output = static_outputs[action_row["condition_id"]]
            action_content = _preceding_content(action_output, thinking=thinking)
            metadata = {
                "preceding_action_id": action_row["condition_id"],
                "preceding_action_label": action_output.get("legal_choice"),
                "preceding_action_parseable": action_output.get("legal_choice") in CHOICES,
                "preceding_action_assistant_content": action_content,
            }
            messages = trajectory_messages(
                row,
                thinking=thinking,
                first_answer=action_content,
                preceding_prompt=action_row["prompt"],
            )
        query = _prepare_query(tokenizer, row, thinking=thinking)
        query["messages"] = messages
        query["rendered"] = _render(tokenizer, messages, thinking=thinking)
        query["prompt_token_ids"] = list(
            map(int, tokenizer.encode(query["rendered"], add_special_tokens=False))
        )
        query["sequence_length"] = len(query["prompt_token_ids"])
        if not thinking:
            query["candidate_token_ids"] = [
                _continuation_id(tokenizer, query["rendered"], label)
                for label in row["candidate_labels"]
            ]
        queries.append(query)
        materialization[condition_id] = metadata
    return queries, materialization


def _left_padded(rows: list[dict[str, Any]], pad_token_id: int) -> tuple[Any, Any]:
    import torch

    width = max(row["sequence_length"] for row in rows)
    input_ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long, device="cuda")
    attention = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        tokens = torch.tensor(row["prompt_token_ids"], dtype=torch.long, device="cuda")
        input_ids[index, width - len(tokens) :] = tokens
        attention[index, width - len(tokens) :] = 1
    return input_ids, attention


def _logit_query(
    model: Any, tokenizer: Any, rows: list[dict[str, Any]], batch_size: int
) -> tuple[dict[str, dict[str, Any]], int]:
    import torch

    output: dict[str, dict[str, Any]] = {}
    ordered = sorted(rows, key=lambda row: (row["sequence_length"], row["query_id"]))
    batches = 0
    for start in range(0, len(ordered), batch_size):
        part = ordered[start : start + batch_size]
        input_ids, attention = _left_padded(part, tokenizer.pad_token_id)
        with torch.inference_mode():
            try:
                model_output = model(
                    input_ids=input_ids,
                    attention_mask=attention,
                    logits_to_keep=1,
                )
            except TypeError:
                model_output = model(input_ids=input_ids, attention_mask=attention)
            logits = model_output.logits[:, -1].float()
        for index, row in enumerate(part):
            final = logits[index]
            candidate_ids = list(map(int, row["candidate_token_ids"]))
            candidate_logits = final[candidate_ids]
            choice_index = int(candidate_logits.argmax().cpu())
            top1_id = int(final.argmax().cpu())
            output[row["query_id"]] = {
                "legal_choice": row["candidate_labels"][choice_index],
                "choice_index": choice_index,
                "legal_logits": {
                    label: float(value)
                    for label, value in zip(
                        row["candidate_labels"], candidate_logits.detach().cpu(), strict=True
                    )
                },
                "top1_token_id": top1_id,
                "top1_text": tokenizer.decode([top1_id]),
                "formatting_compliant": top1_id in candidate_ids,
            }
        batches += 1
        del input_ids, attention, model_output, logits
    return output, batches


def _generation_batch(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    import torch

    input_ids, attention = _left_padded(rows, tokenizer.pad_token_id)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    output: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        continuation = generated[index, input_ids.shape[1] :]
        text = tokenizer.decode(continuation, skip_special_tokens=True)
        selected = parse_thinking_final(text)
        output[row["query_id"]] = {
            "legal_choice": selected,
            "parseable": selected is not None,
            "generated_text": text,
        }
    del input_ids, attention, generated
    return output, 1


def _load_model(spec: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    import transformers
    from huggingface_hub import model_info

    tokenizer, tokenizer_metadata = _load_tokenizer(spec)
    dtype = torch.bfloat16 if spec.get("dtype") == "bfloat16" else torch.float16
    model = None
    loader_name = None
    for candidate in (
        "AutoModelForImageTextToText",
        "AutoModelForMultimodalLM",
        "AutoModelForCausalLM",
    ):
        loader = getattr(transformers, candidate, None)
        if loader is None:
            continue
        try:
            model = loader.from_pretrained(
                spec["id"],
                revision=spec["revision"],
                dtype=dtype,
                low_cpu_mem_usage=True,
            ).cuda()
            loader_name = candidate
            break
        except (OSError, TypeError, ValueError):
            continue
    if model is None or loader_name is None:
        raise RuntimeError(f"could not load model class for {spec['id']}")
    model.eval()
    return (
        model,
        tokenizer,
        {
            "model_id": spec["id"],
            "model_revision_requested": spec["revision"],
            "model_revision_resolved": model_info(spec["id"], revision=spec["revision"]).sha,
            "dtype": spec.get("dtype"),
            "model_loader": loader_name,
            "gpu_actual": torch.cuda.get_device_name(0),
            "torch_version": str(torch.__version__),
            **tokenizer_metadata,
        },
    )


def _query_contract(queries: list[dict[str, Any]], *, semantic: bool) -> dict[str, Any]:
    ordered = sorted(queries, key=lambda query: query["query_id"])
    if semantic:

        def semantic_message(message: dict[str, str]) -> tuple[str, str]:
            if message["role"] == "user":
                return message["role"], normalize_task_prompt(message["content"])
            if message["role"] == "system":
                return message["role"], (
                    "Treat the synthetic environment as exact. Do not infer information "
                    "that is absent from the user message. Keep agent-indexed beliefs distinct."
                )
            return message["role"], "[[PRECEDING_ASSISTANT_OUTPUT]]"

        messages = [
            (
                query["query_id"],
                [semantic_message(message) for message in query["messages"]],
                query["candidate_labels"],
            )
            for query in ordered
        ]
    else:
        messages = [
            (query["query_id"], query["messages"], query["candidate_labels"])
            for query in ordered
        ]
    return {
        "query_count": len(ordered),
        "query_ids_sha256": canonical_sha256([query["query_id"] for query in ordered]),
        "messages_sha256": canonical_sha256(messages),
        "candidate_labels_sha256": canonical_sha256(
            [(query["query_id"], query["candidate_labels"]) for query in ordered]
        ),
        "rendered_sha256": canonical_sha256(
            [(query["query_id"], query["rendered"]) for query in ordered]
        ),
        "prompt_token_ids_sha256": canonical_sha256(
            [(query["query_id"], query["prompt_token_ids"]) for query in ordered]
        ),
    }


def _preflight_contract(
    direct_queries: list[dict[str, Any]], thinking_queries: list[dict[str, Any]]
) -> dict[str, Any]:
    direct_ids = {query["query_id"] for query in direct_queries}
    thinking_ids = {query["query_id"] for query in thinking_queries}
    if not thinking_ids.issubset(direct_ids):
        raise RuntimeError("thinking preflight rows are not a subset of direct rows")
    direct_projection = [query for query in direct_queries if query["query_id"] in thinking_ids]
    return {
        "direct": _query_contract(direct_queries, semantic=False),
        "thinking": _query_contract(thinking_queries, semantic=False),
        "direct_projection": _query_contract(direct_projection, semantic=True),
        "thinking_semantic": _query_contract(thinking_queries, semantic=True),
        "semantic_match": _query_contract(direct_projection, semantic=True)
        == _query_contract(thinking_queries, semantic=True),
    }


def _preflight_sample(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    off = _prepare_query(tokenizer, row, thinking=False)
    on = _prepare_query(tokenizer, row, thinking=True)
    off_markers = [marker for marker in THINKING_MARKERS if marker in off["rendered"]]
    on_markers = [marker for marker in THINKING_MARKERS if marker in on["rendered"]]
    if off_markers or not on_markers:
        raise RuntimeError(
            f"native thinking render contract failed: off={off_markers}, on={on_markers}"
        )
    return {
        "off": {
            "rendered_sha256": canonical_sha256(off["rendered"]),
            "thinking_markers": off_markers,
        },
        "on": {
            "rendered_sha256": canonical_sha256(on["rendered"]),
            "thinking_markers": on_markers,
        },
        "passed": True,
    }


def _compact_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_family.setdefault(record["experiment_family"], []).append(record)
    return {
        "records_total": len(records),
        "locked_records": sum(record["split"] == "locked" for record in records),
        "parse_rate": sum(record["parseable"] for record in records) / len(records)
        if records
        else 0.0,
        "accuracy": sum(record["correct"] for record in records) / len(records)
        if records
        else 0.0,
        "accuracy_by_family": {
            family: sum(record["correct"] for record in values) / len(values)
            for family, values in sorted(by_family.items())
        },
    }


def _artifact(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "content_sha256": canonical_sha256(body)}


def _remote_metadata(
    *,
    config: dict[str, Any],
    subset: dict[str, Any],
    condition: str,
    thinking: bool,
    model_metadata: dict[str, Any],
    started: float,
    forward_calls: int,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    import torch

    elapsed = time.perf_counter() - started
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "run_id": uuid.uuid4().hex,
        "created_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed,
        "condition": condition,
        "thinking": thinking,
        "reasoning_effort": config["conditions"]["qwen38_thinking"].get("reasoning_effort"),
        "config_sha256": canonical_sha256(config),
        "source_dataset_sha256": subset["source_dataset_sha256"],
        "subset_sha256": subset["content_sha256"],
        "records_total": len(records),
        "locked_records": sum(record["split"] == "locked" for record in records),
        "model_forward_calls": forward_calls,
        **model_metadata,
    }


@app.function(image=image, volumes={"/cache": cache}, cpu=2, memory=8192, timeout=900)
def preflight_remote(
    source: dict[str, Any],
    direct_subset: dict[str, Any],
    diagnostic_subset: dict[str, Any],
    config: dict[str, Any],
) -> str:
    _validate_config(config)
    verify_subset_payload(direct_subset, source, config)
    verify_subset_payload(diagnostic_subset, source, config)
    started = time.perf_counter()
    tokenizer, tokenizer_metadata = _load_tokenizer(config["model"])
    direct_queries, _ = _query_rows(tokenizer, direct_subset, thinking=False)
    thinking_queries, _ = _query_rows(tokenizer, diagnostic_subset, thinking=True)
    contract = _preflight_contract(direct_queries, thinking_queries)
    if not contract["semantic_match"]:
        raise RuntimeError("Qwen3.8 thinking/direct semantic prompt contract failed")
    sample = _preflight_sample(tokenizer, diagnostic_subset["rows"][0])
    elapsed = time.perf_counter() - started
    payload = _artifact(
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "status": "qwen38_tokenizer_preflight_passed",
            "model_id": config["model"]["id"],
            "model_revision_requested": config["model"]["revision"],
            "source_dataset_sha256": source["content_sha256"],
            "direct_subset_sha256": direct_subset["content_sha256"],
            "diagnostic_subset_sha256": diagnostic_subset["content_sha256"],
            "config_sha256": canonical_sha256(config),
            "model_forwards": 0,
            "metadata": {
                "run_id": uuid.uuid4().hex,
                "created_at": datetime.now(UTC).isoformat(),
                "elapsed_seconds": elapsed,
                "model_forward_calls": 0,
                "condition": "qwen38_preflight",
                **tokenizer_metadata,
            },
            "preflight_contract": contract,
            "thinking_render_contract": sample,
            **tokenizer_metadata,
        }
    )
    cache.commit()
    return json.dumps(payload, allow_nan=False, sort_keys=True)


def _run_direct(
    subset: dict[str, Any], config: dict[str, Any], *, condition: str
) -> dict[str, Any]:
    started = time.perf_counter()
    model, tokenizer, model_metadata = _load_model(config["model"])
    rows = subset["rows"]
    dynamic = _dynamic_ids(rows)
    static_queries, _ = _query_rows(tokenizer, subset, thinking=False)
    static_queries = [query for query in static_queries if query["query_id"] not in dynamic]
    outputs, calls = _logit_query(
        model, tokenizer, static_queries, int(config["behavior"]["batch_size"])
    )
    dynamic_queries, materialization = _query_rows(
        tokenizer, subset, thinking=False, static_outputs=outputs
    )
    dynamic_queries = [query for query in dynamic_queries if query["query_id"] in dynamic]
    dynamic_outputs, dynamic_calls = _logit_query(
        model, tokenizer, dynamic_queries, int(config["behavior"]["batch_size"])
    )
    outputs.update(dynamic_outputs)
    records = [
        model_record(
            row,
            outputs[row["condition_id"]],
            materialization=materialization.get(row["condition_id"]),
        )
        for row in rows
    ]
    metadata = _remote_metadata(
        config=config,
        subset=subset,
        condition=condition,
        thinking=False,
        model_metadata=model_metadata,
        started=started,
        forward_calls=calls + dynamic_calls,
        records=records,
    )
    return _artifact(
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "status": "qwen38_direct_behavior_complete",
            "condition": condition,
            "thinking": False,
            "config_sha256": canonical_sha256(config),
            "source_dataset_sha256": subset["source_dataset_sha256"],
            "subset_sha256": subset["content_sha256"],
            "records_total": len(records),
            "locked_records": len(records),
            "diagnostic_records": sum(
                row["experiment_family"] in set(DIAGNOSTIC_FAMILIES) for row in rows
            ),
            "metadata": metadata,
            "summary": _compact_summary(records),
            "records": records,
        }
    )


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=32768,
    volumes={"/cache": cache},
    timeout=2700,
)
def direct_remote(
    source: dict[str, Any], subset: dict[str, Any], config: dict[str, Any]
) -> str:
    _validate_config(config)
    verify_subset_payload(subset, source, config)
    payload = _run_direct(subset, config, condition="qwen38_direct")
    cache.commit()
    return json.dumps(payload, allow_nan=False, sort_keys=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=32768,
    volumes={"/cache": cache},
    timeout=2700,
)
def thinking_remote(
    source: dict[str, Any], subset: dict[str, Any], config: dict[str, Any], stage: str
) -> str:
    import torch

    _validate_config(config)
    verify_subset_payload(subset, source, config)
    if subset["selection"] not in {"pilot", "diagnostic"}:
        raise RuntimeError("thinking stage received a non-thinking subset")
    started = time.perf_counter()
    model, tokenizer, model_metadata = _load_model(config["model"])
    rows = subset["rows"]
    dynamic = _dynamic_ids(rows)
    static_queries, _ = _query_rows(tokenizer, subset, thinking=True)
    static_queries = [query for query in static_queries if query["query_id"] not in dynamic]
    outputs: dict[str, dict[str, Any]] = {}
    calls = 0
    batch_size = int(config["behavior"]["batch_size"])
    for start in range(0, len(static_queries), batch_size):
        batch_outputs, batch_calls = _generation_batch(
            model,
            tokenizer,
            static_queries[start : start + batch_size],
            max_new_tokens=int(config["behavior"]["thinking_max_new_tokens"]),
        )
        outputs.update(batch_outputs)
        calls += batch_calls
    dynamic_queries, materialization = _query_rows(
        tokenizer, subset, thinking=True, static_outputs=outputs
    )
    dynamic_queries = [query for query in dynamic_queries if query["query_id"] in dynamic]
    for start in range(0, len(dynamic_queries), batch_size):
        batch_outputs, batch_calls = _generation_batch(
            model,
            tokenizer,
            dynamic_queries[start : start + batch_size],
            max_new_tokens=int(config["behavior"]["thinking_max_new_tokens"]),
        )
        outputs.update(batch_outputs)
        calls += batch_calls
    records = [
        model_record(
            row,
            outputs[row["condition_id"]],
            materialization=materialization.get(row["condition_id"]),
            generated=True,
        )
        for row in rows
    ]
    torch.cuda.synchronize()
    metadata = _remote_metadata(
        config=config,
        subset=subset,
        condition="qwen38_thinking",
        thinking=True,
        model_metadata=model_metadata,
        started=started,
        forward_calls=calls,
        records=records,
    )
    payload = _artifact(
        {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "status": "qwen38_thinking_generation_complete",
            "condition": "qwen38_thinking",
            "stage": stage,
            "thinking": True,
            "config_sha256": canonical_sha256(config),
            "source_dataset_sha256": subset["source_dataset_sha256"],
            "subset_sha256": subset["content_sha256"],
            "records_total": len(records),
            "locked_records": len(records),
            "diagnostic_records": len(records),
            "metadata": metadata,
            "summary": _compact_summary(records),
            "records": records,
        }
    )
    cache.commit()
    return json.dumps(payload, allow_nan=False, sort_keys=True)


def _hash_valid(payload: dict[str, Any]) -> bool:
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    return payload.get("content_sha256") == canonical_sha256(body)


def _write_remote_result(stage: str, payload_text: str) -> tuple[dict[str, Any], Path]:
    payload = json.loads(payload_text)
    if not _hash_valid(payload):
        raise RuntimeError(f"{stage} returned an invalid content hash")
    if payload.get("study_id") != STUDY_ID:
        raise RuntimeError(f"{stage} returned the wrong study ID")
    path = RESULT_ROOT / "raw" / f"{stage}.json"
    _write_new(path, payload)
    return payload, path


def _stage_estimate(config: dict[str, Any], stage: str, *, seconds: float) -> Any:
    spec = config["execution"]["stage_limits"][stage]
    return estimate_cost(
        spec["gpu"],
        seconds,
        cpu_cores=float(spec["cpu_cores"]),
        memory_gib=float(spec["memory_gib"]),
        uncertainty_fraction=float(config["execution"]["uncertainty_fraction"]),
    )


def _admit_stage(config: dict[str, Any], stage: str) -> None:
    spec = config["execution"]["stage_limits"][stage]
    admit_run(
        LEDGER_PATH,
        _stage_estimate(config, stage, seconds=float(spec["timeout_seconds"])),
        study_limit_usd=float(config["execution"]["authorization_usd"]),
    )


def _record_cost(config: dict[str, Any], stage: str, payload: dict[str, Any]) -> Any:
    estimate = _stage_estimate(
        config, stage, seconds=float(payload["metadata"]["elapsed_seconds"])
    )
    append_ledger(
        LEDGER_PATH,
        estimate,
        run_id=str(payload["metadata"]["run_id"]),
        stage=f"{STUDY_ID}:{stage}",
    )
    return estimate


def _git_sha() -> str:
    supplied = os.environ.get("GITHUB_SHA")
    if supplied:
        return supplied
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_model_manifest(stage: str, payload: dict[str, Any], estimate: Any | None) -> None:
    body = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "model_run_manifest_complete",
        "stage": stage,
        "execution_commit_sha": _git_sha(),
        "actions_run_id": os.environ.get("GITHUB_RUN_ID"),
        "modal_run_id": payload.get("metadata", {}).get("run_id"),
        "model_id": payload.get("metadata", {}).get("model_id"),
        "model_revision_requested": payload.get("metadata", {}).get("model_revision_requested"),
        "model_revision_resolved": payload.get("metadata", {}).get("model_revision_resolved"),
        "tokenizer_revision_resolved": payload.get("metadata", {}).get(
            "tokenizer_revision_resolved"
        ),
        "thinking": payload.get("thinking"),
        "reasoning_effort": payload.get("metadata", {}).get("reasoning_effort"),
        "source_dataset_sha256": payload.get("source_dataset_sha256"),
        "subset_sha256": payload.get("subset_sha256"),
        "config_sha256": payload.get("config_sha256"),
        "records_total": payload.get("records_total"),
        "locked_records": payload.get("locked_records"),
        "diagnostic_records": payload.get("diagnostic_records"),
        "raw_artifact_sha256": payload.get("content_sha256"),
        "elapsed_seconds": payload.get("metadata", {}).get("elapsed_seconds"),
        "model_forward_calls": payload.get("metadata", {}).get("model_forward_calls"),
        "cost": asdict(estimate) if estimate is not None else None,
    }
    _write_new(RESULT_ROOT / f"model_run_manifest_{stage}.json", _artifact(body))


def _require_preflight() -> dict[str, Any]:
    path = RESULT_ROOT / "raw" / "preflight_qwen38.json"
    if not path.exists():
        raise RuntimeError("run preflight-qwen38 first")
    payload = _read_json(path)
    if not _hash_valid(payload) or payload.get("status") != "qwen38_tokenizer_preflight_passed":
        raise RuntimeError("invalid Qwen3.8 preflight artifact")
    return payload


@app.local_entrypoint(name="preflight_qwen38")
def preflight_qwen38() -> None:
    config, source, subsets = _inputs()
    _admit_stage(config, "preflight-qwen38")
    payload_text = preflight_remote.remote(
        source, subsets["direct"], subsets["diagnostic"], config
    )
    payload, _path = _write_remote_result("preflight_qwen38", payload_text)
    estimate = _record_cost(config, "preflight-qwen38", payload)
    _write_model_manifest("preflight_qwen38", payload, estimate)
    print(json.dumps(payload, indent=2, sort_keys=True))


@app.local_entrypoint(name="qwen38_direct")
def qwen38_direct() -> None:
    config, source, subsets = _inputs()
    _require_preflight()
    _admit_stage(config, "qwen38-direct")
    payload_text = direct_remote.remote(source, subsets["direct"], config)
    payload, _path = _write_remote_result("qwen38_direct", payload_text)
    estimate = _record_cost(config, "qwen38-direct", payload)
    _write_model_manifest("qwen38_direct", payload, estimate)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


def _thinking_stage(stage: str, selection: str) -> None:
    config, source, subsets = _inputs()
    _require_preflight()
    _admit_stage(config, stage)
    payload_text = thinking_remote.remote(source, subsets[selection], config, stage)
    payload, _path = _write_remote_result(stage.replace("-", "_"), payload_text)
    estimate = _record_cost(config, stage, payload)
    _write_model_manifest(stage.replace("-", "_"), payload, estimate)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


@app.local_entrypoint(name="qwen38_thinking_pilot")
def qwen38_thinking_pilot() -> None:
    _thinking_stage("qwen38-thinking-pilot", "pilot")


@app.local_entrypoint(name="qwen38_thinking_diagnostic")
def qwen38_thinking_diagnostic() -> None:
    _thinking_stage("qwen38-thinking-diagnostic", "diagnostic")


@app.local_entrypoint(name="freeze_controls")
def freeze_controls() -> None:
    config = _config()
    manifest = build_manifest(config)
    print(
        json.dumps(
            {
                "study_id": STUDY_ID,
                "manifest_sha256": manifest["content_sha256"],
                "model_forwards": 0,
                "gpu_stages_ran": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
