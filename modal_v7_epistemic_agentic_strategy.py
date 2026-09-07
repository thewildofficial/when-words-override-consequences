"""Modal entrypoints for the direct-only V7-EAS-1 behavioral screen.

The CPU dataset and exact solver live in ``jspace_policy``.  This file only
renders the frozen prompts, validates candidate-token contracts, and performs
the Qwen3.8-27B forward passes when explicitly launched through GitHub
Actions.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

from jspace_policy.budget import admit_run, append_ledger, estimate_cost
from jspace_policy.v7_epistemic_agentic_strategy import (
    ACTION_LABELS,
    STUDY_ID,
    canonical_sha256,
    dataset_payload,
    result_summary,
    trajectory_messages,
    verify_dataset_payload,
)

CONFIG_PATH = Path("configs/v7/epistemic_agentic_strategy/experiment.json")
DATASET_PATH = Path("configs/v7/epistemic_agentic_strategy/dataset.json")
RESULT_ROOT = Path("results/v7_epistemic_agentic_strategy")
LEDGER_PATH = RESULT_ROOT / "cost_ledger.jsonl"

app = modal.App("wwoc-v7-eas-1")
cache = modal.Volume.from_name("wwoc-v7-eas-1-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install(
        "numpy>=2.0",
        "torch>=2.8",
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
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise RuntimeError(f"refusing to overwrite non-identical artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config() -> dict[str, Any]:
    config = _load_json(CONFIG_PATH)
    if config["study_id"] != STUDY_ID:
        raise RuntimeError("V7 study ID changed")
    if config["model"]["id"] != "Qwen/Qwen3.8-27B":
        raise RuntimeError("V7 primary model changed")
    if config["model"].get("thinking", False) or config["behavior"].get("thinking", False):
        raise RuntimeError("V7 direct screen cannot enable native reasoning")
    return config


def _dataset(config: dict[str, Any]) -> dict[str, Any]:
    dataset = _load_json(DATASET_PATH)
    verify_dataset_payload(dataset, config)
    return dataset


def _validate_config(config: dict[str, Any]) -> None:
    if config["status"] != "preregistered_before_dataset_freeze_or_model_execution":
        raise RuntimeError("V7 protocol is not prospectively frozen")
    behavior = config["behavior"]
    if behavior["query_mode"] != "forced_choice_next_token_logits":
        raise RuntimeError("V7 behavior query mode changed")
    if config["model"].get("thinking", False):
        raise RuntimeError("V7 model spec cannot enable native reasoning")
    if not behavior["direct_only"] or behavior["native_reasoning_enabled"]:
        raise RuntimeError("V7 direct-only screen contract changed")
    if float(config["execution"]["hard_cost_limit_usd"]) > 2.50:
        raise RuntimeError("V7 hard cost limit exceeds the registered ceiling")
    if int(config["execution"]["modal_timeout_seconds"]) != int(
        config["execution"]["estimated_ceiling_seconds"]
    ):
        raise RuntimeError("config budget and Modal timeout disagree")


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

    processor = transformers.AutoProcessor.from_pretrained(
        spec["id"], revision=spec["revision"]
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        raise RuntimeError("tokenizer has no padding or EOS token")
    return tokenizer, {
        "tokenizer_loader": "AutoProcessor",
        "tokenizer_class": type(tokenizer).__name__,
        "transformers_version": transformers.__version__,
    }


def _continuation_id(tokenizer: Any, rendered: str, answer: str) -> int:
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    full = tokenizer.encode(rendered + answer, add_special_tokens=False)
    if full[: len(prefix)] == prefix and len(full) == len(prefix) + 1:
        return int(full[-1])
    raise ValueError(f"{answer!r} is not one token after the frozen prompt")


def _prepare_query(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    messages = trajectory_messages(row)
    rendered = _render(tokenizer, messages)
    prompt_token_ids = list(map(int, tokenizer.encode(rendered, add_special_tokens=False)))
    labels = tuple(row["candidate_labels"])
    if labels != ACTION_LABELS[: len(labels)]:
        raise ValueError(f"candidate-label contract changed: {row['condition_id']}")
    candidate_ids = [_continuation_id(tokenizer, rendered, label) for label in labels]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError(f"candidate token collision: {row['condition_id']}")
    for label, token_id in zip(labels, candidate_ids, strict=True):
        decoded = tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if decoded != label:
            raise ValueError(f"candidate does not decode exactly to {label!r}")
    return {
        "query_id": row["condition_id"],
        "messages": messages,
        "rendered": rendered,
        "prompt_token_ids": prompt_token_ids,
        "candidate_labels": list(labels),
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
    model: Any, tokenizer: Any, rows: list[dict[str, Any]], batch_size: int
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

    tokenizer, tokenizer_metadata = _load_tokenizer(spec)
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
        "model_loader": loader_name,
        **tokenizer_metadata,
        "gpu_actual": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
    }
    return model, tokenizer, metadata


def _records(
    rows: list[dict[str, Any]],
    outputs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        result = outputs[row["condition_id"]]
        selected = result.get("legal_choice")
        mapping = row["choice_mapping"]
        selected_index = mapping.get(selected) if selected else None
        regret = None
        regrets = row.get("regret_by_index")
        if selected_index is not None and regrets is not None:
            regret = float(regrets[selected_index])
        records.append(
            {
                "condition_id": row["condition_id"],
                "experiment_family": row["experiment_family"],
                "task_kind": row["task_kind"],
                "split": row["split"],
                "game_id": row["game_id"],
                "matched_group_id": row["matched_group_id"],
                "shared_prefix_id": row["shared_prefix_id"],
                "condition_factors": row["condition_factors"],
                "surface": row["surface"],
                "report_target": row.get("report_target"),
                "expected": row["expected_choice"],
                "selected": selected,
                "expected_index": row["expected_index"],
                "selected_index": selected_index,
                "choice_mapping": mapping,
                "expected_semantic": row["expected_semantic"],
                "regret": regret,
                "correct": selected == row["expected_choice"],
                "parseable": selected in row["candidate_labels"],
                "formatting_compliant": result.get("formatting_compliant", True),
                "result": result,
            }
        )
    return records


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


def _semantic_contract(queries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "query_count": len(queries),
        "query_ids_sha256": canonical_sha256([q["query_id"] for q in queries]),
        "messages_sha256": canonical_sha256([(q["query_id"], q["messages"]) for q in queries]),
        "candidate_labels_sha256": canonical_sha256(
            [(q["query_id"], q["candidate_labels"]) for q in queries]
        ),
    }


def _hash_valid(payload: dict[str, Any]) -> bool:
    claimed = payload.get("content_sha256")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    return claimed == canonical_sha256(body)


def _record_cost(config: dict[str, Any], payload: dict[str, Any]) -> None:
    seconds = float(payload["metadata"]["elapsed_seconds"])
    estimate = estimate_cost(
        str(config["execution"]["gpu"]), seconds, cpu_cores=8, memory_gib=32
    )
    append_ledger(
        LEDGER_PATH,
        estimate,
        run_id=str(payload["metadata"]["run_id"]),
        stage=f"{STUDY_ID}:behavior",
    )


def _admit_gpu(config: dict[str, Any]) -> None:
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


@app.function(image=image, volumes={"/cache": cache}, cpu=2, memory=8192, timeout=600)
def preflight_remote(dataset: dict[str, Any], config: dict[str, Any]) -> str:
    _validate_config(config)
    verify_dataset_payload(dataset, config)
    tokenizer, tokenizer_metadata = _load_tokenizer(config["model"])
    queries = [_prepare_query(tokenizer, row) for row in dataset["rows"]]
    payload = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "tokenizer_preflight_passed",
        "created_at": datetime.now(UTC).isoformat(),
        "model_id": config["model"]["id"],
        "model_revision_requested": config["model"]["revision"],
        "source_dataset_sha256": dataset["content_sha256"],
        "preflight_contract": _preflight_contract(queries),
        "semantic_preflight_contract": _semantic_contract(queries),
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
def behavior_remote(dataset: dict[str, Any], config: dict[str, Any]) -> str:
    _validate_config(config)
    verify_dataset_payload(dataset, config)
    rows = [
        row for row in dataset["rows"] if row["split"] == config["behavior"]["primary_split"]
    ]
    started = time.perf_counter()
    model, tokenizer, metadata = _load_model(config["model"])
    queries = [_prepare_query(tokenizer, row) for row in rows]
    outputs = _logit_query(model, tokenizer, queries, int(config["behavior"]["batch_size"]))
    records = _records(rows, outputs)
    metadata.update(
        {
            "run_id": str(uuid.uuid4()),
            "elapsed_seconds": time.perf_counter() - started,
            "config_sha256": canonical_sha256(config),
            "dataset_sha256": dataset["content_sha256"],
            "split": config["behavior"]["primary_split"],
            "query_count": len(queries),
        }
    )
    payload = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "forced_choice_logit_behavior_complete",
        "created_at": datetime.now(UTC).isoformat(),
        "metadata": metadata,
        "records": records,
        "summary": result_summary(records, config),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    cache.commit()
    return json.dumps(payload, allow_nan=False, sort_keys=True)


def _write_remote_result(stage: str, payload_text: str) -> Path:
    payload = json.loads(payload_text)
    if not _hash_valid(payload):
        raise RuntimeError(f"{stage} returned an invalid content hash")
    if payload.get("study_id") != STUDY_ID:
        raise RuntimeError(f"{stage} returned the wrong study ID")
    path = RESULT_ROOT / "raw" / f"{stage}.json"
    _write_new(path, payload)
    return path


def _require_preflight() -> dict[str, Any]:
    path = RESULT_ROOT / "raw" / "preflight.json"
    if not path.exists():
        raise RuntimeError("run V7 tokenizer preflight first")
    payload = _load_json(path)
    if not _hash_valid(payload) or payload.get("status") != "tokenizer_preflight_passed":
        raise RuntimeError("invalid V7 tokenizer preflight artifact")
    return payload


@app.local_entrypoint(name="preflight")
def preflight() -> None:
    config = _config()
    dataset = _dataset(config)
    payload = json.loads(preflight_remote.remote(dataset, config))
    _write_remote_result("preflight", json.dumps(payload))
    print(json.dumps(payload["preflight_contract"], indent=2, sort_keys=True))


@app.local_entrypoint(name="behavior")
def behavior() -> None:
    config = _config()
    dataset = _dataset(config)
    _require_preflight()
    _admit_gpu(config)
    payload = json.loads(behavior_remote.remote(dataset, config))
    _write_remote_result("behavior", json.dumps(payload))
    _record_cost(config, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


@app.local_entrypoint(name="freeze_dataset")
def freeze_dataset() -> None:
    config = _config()
    payload = dataset_payload(config)
    verify_dataset_payload(payload, config)
    _write_new(DATASET_PATH, payload)
    print(json.dumps({"rows": len(payload["rows"]), "sha256": payload["content_sha256"]}))
