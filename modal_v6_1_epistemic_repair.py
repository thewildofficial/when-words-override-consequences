"""Modal entrypoints for the corrected V6.1 epistemic repair study.

The remote stages are deliberately behavioral only.  They collect one-token
candidate logits and a separate locked-split free-generation measurement.  No
activations are captured and no causal/mechanistic stage exists in this study.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

from jspace_policy.budget import admit_run, append_ledger, estimate_cost
from jspace_policy.v6_1_epistemic_repair import (
    CHOICES,
    STUDY_ID,
    canonical_sha256,
    dataset_payload,
    result_summary,
    trajectory_messages,
    verify_dataset_payload,
)

CONFIG_PATH = Path("configs/v6.1/epistemic_repair/experiment.json")
DATASET_PATH = Path("configs/v6.1/epistemic_repair/dataset.json")
RESULT_ROOT = Path("results/v6_1_epistemic_repair")
LEDGER_PATH = RESULT_ROOT / "cost_ledger.jsonl"

app = modal.App("jspace-v6-1-epistemic-repair")
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, value: object) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite prospective artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config() -> dict[str, Any]:
    config = _load_json(CONFIG_PATH)
    if config["study_id"] != STUDY_ID:
        raise RuntimeError("V6.1 study ID changed")
    return config


def _dataset(config: dict[str, Any]) -> dict[str, Any]:
    dataset = _load_json(DATASET_PATH)
    verify_dataset_payload(dataset, config)
    return dataset


def _validate_config(config: dict[str, Any]) -> None:
    if config["status"] != "preregistered_before_dataset_freeze_or_model_execution":
        raise RuntimeError("protocol is not prospectively frozen")
    if config["behavior"]["query_mode"] != "forced_choice_next_token_logits":
        raise RuntimeError("behavior query mode changed")
    if config["blackbox"]["parser"] != "first_legal_label_in_generation_order":
        raise RuntimeError("black-box parser contract changed")
    if float(config["execution"]["hard_cost_limit_usd"]) > 2.0:
        raise RuntimeError("V6.1 hard cost ceiling may not exceed USD 2")


def _model_spec_for_preflight(config: dict[str, Any], model_key: str) -> dict[str, Any]:
    if model_key == "primary":
        return config["model"]
    try:
        return config["replication_models"][model_key]
    except KeyError as error:
        raise RuntimeError(f"unknown model key: {model_key}") from error


def _model_spec(config: dict[str, Any], model_key: str) -> dict[str, Any]:
    spec = _model_spec_for_preflight(config, model_key)
    if model_key != "primary" and not spec.get("enabled", False):
        raise RuntimeError(f"{model_key} is separately gated and currently disabled")
    return spec


def _render(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                chat_template_kwargs={"enable_thinking": False},
                **kwargs,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)


def _load_tokenizer(spec: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    import transformers

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
    return tokenizer, {"tokenizer_loader": loader}


def _continuation_id(tokenizer: Any, rendered: str, answer: str) -> int:
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    full = tokenizer.encode(rendered + answer, add_special_tokens=False)
    if full[: len(prefix)] == prefix and len(full) == len(prefix) + 1:
        return int(full[-1])
    raise ValueError(f"{answer!r} is not one token after the frozen prompt")


def _prepare_query(
    tokenizer: Any,
    query_id: str,
    messages: list[dict[str, str]],
    candidates: tuple[str, ...],
) -> dict[str, Any]:
    rendered = _render(tokenizer, messages)
    prompt_token_ids = list(map(int, tokenizer.encode(rendered, add_special_tokens=False)))
    candidate_ids = [_continuation_id(tokenizer, rendered, label) for label in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError(f"candidate token collision: {query_id}")
    for label, token_id in zip(candidates, candidate_ids, strict=True):
        decoded = tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if decoded != label:
            raise ValueError(f"candidate does not decode exactly to {label!r}: {query_id}")
    return {
        "query_id": query_id,
        "messages": messages,
        "rendered": rendered,
        "prompt_token_ids": prompt_token_ids,
        "candidate_labels": list(candidates),
        "candidate_token_ids": candidate_ids,
        "sequence_length": len(prompt_token_ids),
    }


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
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    import torch

    output: dict[str, dict[str, Any]] = {}
    ordered = sorted(rows, key=lambda row: (row["sequence_length"], row["query_id"]))
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
        del input_ids, attention, model_output, logits
    return output


def _load_model(spec: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    import transformers
    from huggingface_hub import model_info

    tokenizer, tokenizer_meta = _load_tokenizer(spec)
    dtype = torch.bfloat16 if spec.get("dtype") == "bfloat16" else torch.float16
    model = None
    loader_name = None
    for loader_name_candidate in (
        "AutoModelForImageTextToText",
        "AutoModelForMultimodalLM",
        "AutoModelForCausalLM",
    ):
        loader = getattr(transformers, loader_name_candidate, None)
        if loader is None:
            continue
        try:
            model = loader.from_pretrained(
                spec["id"],
                revision=spec["revision"],
                dtype=dtype,
                low_cpu_mem_usage=True,
            ).cuda()
            loader_name = loader_name_candidate
            break
        except (OSError, TypeError, ValueError):
            continue
    if model is None or loader_name is None:
        raise RuntimeError(f"could not load model class for {spec['id']}")
    model.eval()
    metadata = {
        "model_id": spec["id"],
        "model_revision_requested": spec["revision"],
        "model_revision_resolved": model_info(spec["id"], revision=spec["revision"]).sha,
        "dtype": spec.get("dtype"),
        "thinking": spec.get("thinking"),
        "model_loader": loader_name,
        **tokenizer_meta,
        "gpu_actual": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
    }
    return model, tokenizer, metadata


def _preceding_assistant_content(result: dict[str, Any], *, mode: str) -> str:
    """Return the assistant turn that a trajectory should actually contain."""

    if mode == "blackbox" and "generated_text" in result:
        # Preserve the complete sampled response, including explanations or an
        # unparseable answer.  Black-box trajectories must not collapse into a
        # parsed-label intervention.
        generated_text = result["generated_text"]
        return "" if generated_text is None else str(generated_text)
    label = result.get("legal_choice")
    return str(label) if label in CHOICES else ""


def _report_materialization(
    row: dict[str, Any],
    report_outputs: dict[str, dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    *,
    mode: str,
) -> tuple[str, dict[str, Any]]:
    report_id = str(row["scaffold_report_id"])
    report_row = row_by_id[report_id]
    report_result = report_outputs[report_id]
    label = report_result.get("legal_choice")
    value = report_row["choice_mapping"].get(label) if label in CHOICES else None
    concept = (
        report_row["game_certificate"]["concepts"][value]
        if value is not None
        else "[unparseable prior report]"
    )
    label_text = label if label in CHOICES else "[unparseable prior report]"
    prompt = (
        row["prompt"]
        .replace("{SELF_REPORT_LABEL}", label_text)
        .replace("{SELF_REPORT_CONCEPT}", concept)
    )
    assistant_content = _preceding_assistant_content(report_result, mode=mode)
    return prompt, {
        "source_report_id": report_id,
        "source_report_prompt": report_row["prompt"],
        "source_report_label": label,
        "source_report_value": value,
        "source_report_correct": label == report_row["expected_choice"],
        "source_report_parseable": label in CHOICES,
        "source_report_assistant_content": assistant_content,
    }


def _query_rows(
    tokenizer: Any,
    dataset: dict[str, Any],
    *,
    locked_only: bool = False,
    static_outputs: dict[str, dict[str, Any]] | None = None,
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Prepare static rows, then bind endogenous scaffold/trajectory rows."""

    rows = [row for row in dataset["rows"] if not locked_only or row["split"] == "locked"]
    row_by_id = {row["condition_id"]: row for row in dataset["rows"]}
    dynamic_ids = {
        row["condition_id"]
        for row in rows
        if row.get("scaffold_source") == "self_generated"
        or row.get("trajectory_condition") == "action_then_report"
    }
    queries: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["condition_id"] in dynamic_ids:
            continue
        messages = trajectory_messages(row)
        query = _prepare_query(
            tokenizer, row["condition_id"], messages, tuple(row["candidate_labels"])
        )
        queries.append(query)
        metadata[row["condition_id"]] = {"materialization": "static"}

    if static_outputs is None:
        # This branch is used by tokenizer preflight.  A legal placeholder
        # answer makes every dynamic prompt concrete without using model data.
        static_outputs = {
            row["condition_id"]: {"legal_choice": "A"}
            for row in rows
            if row["condition_id"] not in dynamic_ids
        }
    for row in rows:
        condition_id = row["condition_id"]
        if condition_id not in dynamic_ids:
            continue
        if row.get("scaffold_source") == "self_generated":
            prompt, materialization = _report_materialization(
                row, static_outputs, row_by_id, mode=mode
            )
            report_row = row_by_id[str(row["scaffold_report_id"])]
            messages = trajectory_messages(
                row,
                first_answer=materialization["source_report_assistant_content"],
                materialized_prompt=prompt,
                preceding_prompt=report_row["prompt"],
            )
        else:
            action_row = row_by_id[str(row["trajectory_action_row_id"])]
            action_output = static_outputs[action_row["condition_id"]]
            action_content = _preceding_assistant_content(action_output, mode=mode)
            materialization = {
                "preceding_action_id": action_row["condition_id"],
                "preceding_action_label": action_output.get("legal_choice"),
                "preceding_action_parseable": action_output.get("legal_choice") in CHOICES,
                "preceding_action_assistant_content": action_content,
            }
            messages = trajectory_messages(
                row,
                first_answer=action_content,
                preceding_prompt=action_row["prompt"],
            )
        query = _prepare_query(
            tokenizer, condition_id, messages, tuple(row["candidate_labels"])
        )
        queries.append(query)
        metadata[condition_id] = materialization
    return queries, metadata


def _records(
    dataset: dict[str, Any],
    outputs: dict[str, dict[str, Any]],
    materialization: dict[str, dict[str, Any]],
    *,
    locked_only: bool = False,
    generated: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    rows = [row for row in dataset["rows"] if not locked_only or row["split"] == "locked"]
    for row in rows:
        result = outputs.get(row["condition_id"])
        if result is None:
            raise RuntimeError(f"missing output for {row['condition_id']}")
        selected = result.get("legal_choice")
        expected = row["expected_choice"]
        record = {
            "condition_id": row["condition_id"],
            "experiment_family": row["experiment_family"],
            "task_kind": row["task_kind"],
            "split": row["split"],
            "game_id": row["game_id"],
            "matched_group_id": row["matched_group_id"],
            "expected": expected,
            "selected": selected,
            "expected_index": row["expected_index"],
            "expected_value": row["expected_value"],
            "expected_semantic": row["expected_semantic"],
            "selected_index": row["choice_mapping"].get(selected) if selected else None,
            "choice_mapping": row["choice_mapping"],
            "vo_i": row["vo_i"],
            "scaffold_source": row["scaffold_source"],
            "trajectory_condition": row.get("trajectory_condition"),
            "swapped_labels": row["swapped_labels"],
            "correct": selected == expected,
            "parseable": selected in CHOICES,
            "formatting_compliant": result.get("formatting_compliant", True),
            "result": result,
            "materialization": materialization.get(row["condition_id"], {}),
        }
        record.update(row["condition_factors"])
        if generated:
            record["generated_text"] = result.get("generated_text", "")
        records.append(record)
    return records


def _parse_generation(text: str, candidates: tuple[str, ...]) -> str | None:
    for match in re.finditer(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", text.upper()):
        if match.group(1) in candidates:
            return match.group(1)
    return None


def _generation_batch(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int,
) -> dict[str, dict[str, Any]]:
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
        selected = _parse_generation(text, tuple(row["candidate_labels"]))
        output[row["query_id"]] = {
            "legal_choice": selected,
            "parseable": selected is not None,
            "generated_text": text,
        }
    del input_ids, attention, generated
    return output


def _preflight_queries(tokenizer: Any, dataset: dict[str, Any]) -> list[dict[str, Any]]:
    queries, _ = _query_rows(tokenizer, dataset, mode="preflight")
    return queries


def _preflight_contract(queries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "query_count": len(queries),
        "query_ids_sha256": canonical_sha256([q["query_id"] for q in queries]),
        "rendered_prompts_sha256": canonical_sha256(
            [(q["query_id"], q["rendered"]) for q in queries]
        ),
        "prompt_token_ids_sha256": canonical_sha256(
            [(q["query_id"], q["prompt_token_ids"]) for q in queries]
        ),
        "candidate_labels_sha256": canonical_sha256(
            [(q["query_id"], q["candidate_labels"]) for q in queries]
        ),
        "candidate_token_ids_sha256": canonical_sha256(
            [(q["query_id"], q["candidate_token_ids"]) for q in queries]
        ),
    }


def _semantic_preflight_contract(queries: list[dict[str, Any]]) -> dict[str, Any]:
    """Hash the model-independent prompt/label contract.

    Token IDs, rendered chat templates, and tokenizer-specific special-token
    behavior are recorded by ``_preflight_contract`` but are not replication
    invariants.  A valid checkpoint can encode the same semantic stimulus with
    different IDs or a different chat wrapper.  Cross-model parity therefore
    compares query identity, message content, and legal candidate labels only.
    """

    return {
        "query_count": len(queries),
        "query_ids_sha256": canonical_sha256([q["query_id"] for q in queries]),
        "messages_sha256": canonical_sha256(
            [(q["query_id"], q["messages"]) for q in queries]
        ),
        "candidate_labels_sha256": canonical_sha256(
            [(q["query_id"], q["candidate_labels"]) for q in queries]
        ),
    }


def _hash_valid(payload: dict[str, Any]) -> bool:
    claimed = payload.get("content_sha256")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    return claimed == canonical_sha256(body)


def _record_cost(config: dict[str, Any], stage: str, payload: dict[str, Any]) -> None:
    seconds = float(payload["metadata"]["elapsed_seconds"])
    estimate = estimate_cost(
        str(config["execution"]["gpu"]), seconds, cpu_cores=8, memory_gib=32
    )
    append_ledger(
        LEDGER_PATH,
        estimate,
        run_id=str(payload["metadata"]["run_id"]),
        stage=f"{STUDY_ID}:{stage}",
    )


def _admit_gpu(config: dict[str, Any], stage: str) -> None:
    estimate = estimate_cost(
        str(config["execution"]["gpu"]),
        float(config["execution"]["estimated_ceiling_seconds"]),
        cpu_cores=8,
        memory_gib=32,
    )
    admit_run(
        LEDGER_PATH,
        estimate,
        study_limit_usd=float(config["execution"]["hard_cost_limit_usd"]),
    )
    if stage not in {"behavior", "blackbox"}:
        raise RuntimeError(f"unknown paid stage: {stage}")


@app.function(image=image, volumes={"/cache": cache}, cpu=2, memory=8192, timeout=900)
def preflight_remote(dataset: dict[str, Any], config: dict[str, Any], model_key: str) -> str:
    _validate_config(config)
    verify_dataset_payload(dataset, config)
    spec = _model_spec_for_preflight(config, model_key)
    tokenizer, tokenizer_metadata = _load_tokenizer(spec)
    queries = _preflight_queries(tokenizer, dataset)
    contract = _preflight_contract(queries)
    semantic_contract = _semantic_preflight_contract(queries)
    cross_model_parity: dict[str, Any] = {
        "status": "not_applicable_primary_model",
        "passed": True,
    }
    if model_key != "primary":
        primary_tokenizer, _ = _load_tokenizer(config["model"])
        primary_queries = _preflight_queries(primary_tokenizer, dataset)
        primary_contract = _preflight_contract(primary_queries)
        primary_semantic_contract = _semantic_preflight_contract(primary_queries)
        cross_model_parity = {
            "status": "semantic_prompt_contract_match",
            "passed": semantic_contract == primary_semantic_contract,
            "primary": primary_contract,
            "replication": contract,
            "primary_semantic": primary_semantic_contract,
            "replication_semantic": semantic_contract,
        }
        if not cross_model_parity["passed"]:
            raise RuntimeError("replication semantic prompt/label parity failed")
    payload = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "tokenizer_preflight_passed",
        "created_at": datetime.now(UTC).isoformat(),
        "model_key": model_key,
        "model_id": spec["id"],
        "model_revision_requested": spec["revision"],
        "source_dataset_sha256": dataset["content_sha256"],
        "preflight_contract": contract,
        "semantic_preflight_contract": semantic_contract,
        "cross_model_parity": cross_model_parity,
        **tokenizer_metadata,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    cache.commit()
    return json.dumps(payload, allow_nan=False, sort_keys=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=32768,
    volumes={"/cache": cache},
    timeout=1500,
)
def behavior_remote(dataset: dict[str, Any], config: dict[str, Any], model_key: str) -> str:
    _validate_config(config)
    spec = _model_spec(config, model_key)
    started = time.perf_counter()
    model, tokenizer, metadata = _load_model(spec)
    static_queries, _ = _query_rows(tokenizer, dataset, mode="behavior")
    # _query_rows returns all static and dynamic queries only after binding.  To
    # obtain endogenous answers, run the static subset first and then bind the
    # dynamic queries from those outputs.
    dynamic_rows = [
        row
        for row in dataset["rows"]
        if row.get("scaffold_source") == "self_generated"
        or row.get("trajectory_condition") == "action_then_report"
    ]
    static_queries = [
        q
        for q in static_queries
        if q["query_id"] not in {r["condition_id"] for r in dynamic_rows}
    ]
    outputs = _logit_query(
        model, tokenizer, static_queries, int(config["behavior"]["batch_size"])
    )
    dynamic_queries, materialization = _query_rows(
        tokenizer, dataset, mode="behavior", static_outputs=outputs
    )
    dynamic_queries = [
        q for q in dynamic_queries if q["query_id"] in {r["condition_id"] for r in dynamic_rows}
    ]
    outputs.update(
        _logit_query(model, tokenizer, dynamic_queries, int(config["behavior"]["batch_size"]))
    )
    records = _records(dataset, outputs, materialization)
    import torch

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "forced_choice_logit_behavior_complete",
        "metadata": {
            "run_id": uuid.uuid4().hex,
            "created_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed,
            "model_key": model_key,
            "source_dataset_sha256": dataset["content_sha256"],
            "config_sha256": canonical_sha256(config),
            "measurement_mode": "candidate_token_logits_no_activations",
            **metadata,
        },
        "summary": result_summary(records, config),
        "records": records,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    cache.commit()
    return json.dumps(payload, allow_nan=False, sort_keys=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=32768,
    volumes={"/cache": cache},
    timeout=1500,
)
def blackbox_remote(dataset: dict[str, Any], config: dict[str, Any], model_key: str) -> str:
    _validate_config(config)
    spec = _model_spec(config, model_key)
    started = time.perf_counter()
    model, tokenizer, metadata = _load_model(spec)
    locked_rows = [row for row in dataset["rows"] if row["split"] == "locked"]
    dynamic_rows = [
        row
        for row in locked_rows
        if row.get("scaffold_source") == "self_generated"
        or row.get("trajectory_condition") == "action_then_report"
    ]
    static_queries, _ = _query_rows(tokenizer, dataset, locked_only=True, mode="blackbox")
    dynamic_ids = {row["condition_id"] for row in dynamic_rows}
    static_queries = [q for q in static_queries if q["query_id"] not in dynamic_ids]
    outputs = {}
    for start in range(0, len(static_queries), int(config["behavior"]["batch_size"])):
        outputs.update(
            _generation_batch(
                model,
                tokenizer,
                static_queries[start : start + int(config["behavior"]["batch_size"])],
                max_new_tokens=int(config["blackbox"]["max_new_tokens"]),
            )
        )
    dynamic_queries, materialization = _query_rows(
        tokenizer, dataset, locked_only=True, static_outputs=outputs, mode="blackbox"
    )
    dynamic_queries = [q for q in dynamic_queries if q["query_id"] in dynamic_ids]
    for start in range(0, len(dynamic_queries), int(config["behavior"]["batch_size"])):
        outputs.update(
            _generation_batch(
                model,
                tokenizer,
                dynamic_queries[start : start + int(config["behavior"]["batch_size"])],
                max_new_tokens=int(config["blackbox"]["max_new_tokens"]),
            )
        )
    records = _records(dataset, outputs, materialization, locked_only=True, generated=True)
    import torch

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "blackbox_free_generation_complete",
        "metadata": {
            "run_id": uuid.uuid4().hex,
            "created_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed,
            "model_key": model_key,
            "source_dataset_sha256": dataset["content_sha256"],
            "config_sha256": canonical_sha256(config),
            "measurement_mode": "free_generation_text_only",
            **metadata,
        },
        "summary": result_summary(records, config),
        "records": records,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    cache.commit()
    return json.dumps(payload, allow_nan=False, sort_keys=True)


def _write_remote_result(stage: str, model_key: str, payload_text: str) -> Path:
    payload = json.loads(payload_text)
    if not _hash_valid(payload):
        raise RuntimeError(f"{stage} returned an invalid content hash")
    if payload.get("study_id") != STUDY_ID:
        raise RuntimeError(f"{stage} returned the wrong study ID")
    path = RESULT_ROOT / "raw" / f"{stage}_{model_key}.json"
    _write_new(path, payload)
    return path


def _require_preflight(model_key: str) -> dict[str, Any]:
    path = RESULT_ROOT / "raw" / f"preflight_{model_key}.json"
    if not path.exists():
        raise RuntimeError(f"run tokenizer preflight first: {path}")
    payload = _load_json(path)
    if not _hash_valid(payload) or payload.get("status") != "tokenizer_preflight_passed":
        raise RuntimeError("invalid tokenizer preflight artifact")
    return payload


def _ensure_preflight(
    config: dict[str, Any], dataset: dict[str, Any], model_key: str
) -> dict[str, Any]:
    path = RESULT_ROOT / "raw" / f"preflight_{model_key}.json"
    if path.exists():
        return _require_preflight(model_key)
    payload = json.loads(preflight_remote.remote(dataset, config, model_key))
    _write_remote_result("preflight", model_key, json.dumps(payload))
    return payload


def _ensure_behavior(
    config: dict[str, Any], dataset: dict[str, Any], model_key: str
) -> dict[str, Any]:
    path = RESULT_ROOT / "raw" / f"behavior_{model_key}.json"
    if path.exists():
        payload = _load_json(path)
        if (
            not _hash_valid(payload)
            or payload.get("status") != "forced_choice_logit_behavior_complete"
        ):
            raise RuntimeError("invalid behavior artifact")
        return payload
    _ensure_preflight(config, dataset, model_key)
    _admit_gpu(config, "behavior")
    payload = json.loads(behavior_remote.remote(dataset, config, model_key))
    _write_remote_result("behavior", model_key, json.dumps(payload))
    _record_cost(config, "behavior", payload)
    return payload


@app.local_entrypoint(name="preflight")
def preflight(model_key: str = "primary") -> None:
    config = _config()
    dataset = _dataset(config)
    payload = json.loads(preflight_remote.remote(dataset, config, model_key))
    _write_remote_result("preflight", model_key, json.dumps(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))


@app.local_entrypoint(name="behavior")
def behavior(model_key: str = "primary") -> None:
    config = _config()
    dataset = _dataset(config)
    payload = _ensure_behavior(config, dataset, model_key)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


@app.local_entrypoint(name="blackbox")
def blackbox(model_key: str = "primary") -> None:
    config = _config()
    dataset = _dataset(config)
    _ensure_preflight(config, dataset, model_key)
    _admit_gpu(config, "blackbox")
    payload = json.loads(blackbox_remote.remote(dataset, config, model_key))
    _write_remote_result("blackbox", model_key, json.dumps(payload))
    _record_cost(config, "blackbox", payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


@app.local_entrypoint(name="full_primary")
def full_primary(model_key: str = "primary") -> None:
    config = _config()
    dataset = _dataset(config)
    _ensure_preflight(config, dataset, model_key)
    behavior_payload = _ensure_behavior(config, dataset, model_key)
    _admit_gpu(config, "blackbox")
    blackbox_payload = json.loads(blackbox_remote.remote(dataset, config, model_key))
    _write_remote_result("blackbox", model_key, json.dumps(blackbox_payload))
    _record_cost(config, "blackbox", blackbox_payload)
    print(
        json.dumps(
            {"behavior": behavior_payload["summary"], "blackbox": blackbox_payload["summary"]},
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint(name="freeze_dataset")
def freeze_dataset() -> None:
    config = _config()
    payload = dataset_payload(config)
    verify_dataset_payload(payload, config)
    _write_new(DATASET_PATH, payload)
    print(json.dumps({"rows": len(payload["rows"]), "sha256": payload["content_sha256"]}))
