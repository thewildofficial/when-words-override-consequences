"""V5-RBG-4 multi-model replication screen.

Runs the frozen RBG-4 inverse-evidence dataset (content SHA-256
3fdc9977f03700dbfb2118414ec9602a4e76d9d6e2b35788a81bc6acf4a1f5d1) unchanged
against additional open-weight model families, reusing the exact scoring
procedure of modal_inverse_evidence.py: chat-template rendering with thinking
disabled where the template supports it, forced-choice conditional scoring
over single-token continuations at the final position, no sampling.

The model is a runtime parameter; the frozen dataset/config hashes are still
verified against configs/v5/inverse_evidence/* before any GPU work.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

from jspace_policy.budget import admit_run, append_ledger, estimate_cost
from jspace_policy.inverse_evidence_games import verify_dataset_payload

CONFIG_PATH = Path("configs/v5/inverse_evidence/experiment.json")
DATASET_PATH = Path("configs/v5/inverse_evidence/dataset.json")
RESULT_ROOT = Path("results/rbg4_multimodel")
SYSTEM_PROMPT = (
    "Treat the described game and prior episodes as exact formal information. "
    "Follow the requested forced-choice format exactly."
)

# Revision SHAs pinned from the Hugging Face Hub on 2026-10-23.
MODELS: dict[str, dict[str, Any]] = {
    # Parity re-run of the original RBG-4 model on the same revision.
    "qwen36-27b": {
        "model_id": "Qwen/Qwen3.6-27B",
        "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "gpu": "A100-80GB",
        "gpu_count": 1,
        "cpu": 8,
        "memory": 32768,
        "timeout": 2400,
        "batch_size": 24,
        "enable_thinking": False,
        "family": "qwen",
    },
    # Mistral-family comparator.
    "mistral-small-32-24b": {
        "model_id": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "revision": "95a6d26c4bfb886c58daf9d3f7332c857cb27b43",
        "gpu": "A100-80GB",
        "gpu_count": 1,
        "cpu": 8,
        "memory": 32768,
        "timeout": 2400,
        "batch_size": 24,
        "enable_thinking": None,
        "family": "mistral",
    },
    # Qwen-family base-vs-instruct contrast (same generation, same size class;
    # Qwen3.6 has no official base release as of 2026-10-23).
    "qwen3-30b-a3b-instruct": {
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        "gpu": "A100-80GB",
        "gpu_count": 1,
        "cpu": 8,
        "memory": 32768,
        "timeout": 2400,
        "batch_size": 24,
        "enable_thinking": False,
        "family": "qwen",
    },
    "qwen3-30b-a3b-base": {
        "model_id": "Qwen/Qwen3-30B-A3B-Base",
        "revision": "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9",
        "gpu": "A100-80GB",
        "gpu_count": 1,
        "cpu": 8,
        "memory": 32768,
        "timeout": 2400,
        "batch_size": 24,
        "enable_thinking": False,
        "family": "qwen",
    },
    # GLM-family comparator (open weights, ~106B MoE, thinking disabled).
    "glm45-air": {
        "model_id": "zai-org/GLM-4.5-Air",
        "revision": "a24ceef6ce4f3536971efe9b778bdaa1bab18daa",
        "gpu": "A100-80GB",
        "gpu_count": 4,
        "cpu": 16,
        "memory": 196608,
        "timeout": 3600,
        "batch_size": 24,
        "enable_thinking": False,
        "family": "glm",
    },
}

app = modal.App("jspace-v5-rbg4-multimodel")
cache = modal.Volume.from_name("jspace-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install(
        "torch>=2.8",
        "transformers>=5.15",
        "huggingface_hub>=0.34",
        "mistral-common>=1.8",
        "accelerate>=1.10",
        "hf_transfer>=0.1",
    )
    .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir("src/jspace_policy", remote_path="/root/jspace_policy")
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_new(path: Path, value: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite prospective artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# --- scoring helpers, copied verbatim from modal_inverse_evidence.py ---


def _continuation_id(tokenizer: Any, rendered: str, answer: str) -> int:
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    full = tokenizer.encode(rendered + answer, add_special_tokens=False)
    if full[: len(prefix)] == prefix and len(full) == len(prefix) + 1:
        return int(full[-1])
    raise ValueError(f"{answer!r} is not one token after the frozen Answer: prefix")


def _prepare_query(
    tokenizer: Any,
    query_id: str,
    messages: list[dict[str, str]],
    candidates: list[str],
    enable_thinking: bool | None,
) -> dict[str, Any]:
    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    rendered = tokenizer.apply_chat_template(messages, **template_kwargs)
    token_ids = list(map(int, tokenizer.encode(rendered, add_special_tokens=False)))
    return {
        "query_id": query_id,
        "prompt_token_ids": token_ids,
        "candidate_labels": candidates,
        "candidate_token_ids": [
            _continuation_id(tokenizer, rendered, candidate) for candidate in candidates
        ],
        "sequence_length": len(token_ids),
    }


def _left_padded(rows: list[dict[str, Any]], pad_token_id: int) -> tuple[Any, Any]:
    import torch

    width = max(row["sequence_length"] for row in rows)
    input_ids = torch.full(
        (len(rows), width), pad_token_id, dtype=torch.long, device="cuda"
    )
    attention = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        tokens = torch.tensor(row["prompt_token_ids"], dtype=torch.long, device="cuda")
        input_ids[index, width - len(tokens) :] = tokens
        attention[index, width - len(tokens) :] = 1
    return input_ids, attention


def _query(
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
            logits = model(
                input_ids=input_ids,
                attention_mask=attention,
                logits_to_keep=1,
            ).logits[:, -1].float()
        for index, row in enumerate(part):
            final = logits[index]
            candidate_ids = list(map(int, row["candidate_token_ids"]))
            candidate_logits = final[candidate_ids]
            choice_index = int(candidate_logits.argmax().cpu())
            top1_id = int(final.argmax().cpu())
            output[row["query_id"]] = {
                "legal_choice": row["candidate_labels"][choice_index],
                "legal_logits": {
                    label: float(value)
                    for label, value in zip(
                        row["candidate_labels"],
                        candidate_logits.detach().cpu(),
                        strict=True,
                    )
                },
                "top1_token_id": top1_id,
                "top1_text": tokenizer.decode([top1_id]),
                "formatting_compliant": top1_id in candidate_ids,
            }
        del input_ids, attention, logits
    return output


# --- prompt construction, same as modal_inverse_evidence.py ---


def _action_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["prompt"]},
    ]


def _report_messages(row: dict[str, Any], action: str) -> tuple[list[dict[str, str]], str]:
    from jspace_policy.inverse_evidence_games import report_question

    question, expected = report_question(row, action)
    return (
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{row['scenario']}\n{question}"},
        ],
        expected,
    )


def _load_tokenizer(spec: dict[str, Any]) -> Any:
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        spec["model_id"], revision=spec["revision"]
    )
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    return tokenizer


def _load_model(spec: dict[str, Any]) -> Any:
    import torch
    import transformers

    kwargs = {
        "revision": spec["revision"],
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if spec["gpu_count"] > 1:
        kwargs["device_map"] = "auto"
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            spec["model_id"], **kwargs
        )
    except ValueError:
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            spec["model_id"], **kwargs
        )
    if spec["gpu_count"] == 1:
        model = model.cuda()
    model.eval()
    return model


@app.function(image=image, cpu=2, memory=8192, volumes={"/cache": cache}, timeout=900)
def rbg4_preflight_remote(
    dataset: dict[str, Any], config: dict[str, Any], model_key: str
) -> str:
    spec = MODELS[model_key]
    verify_dataset_payload(dataset, config)
    tokenizer = _load_tokenizer(spec)
    lengths = []
    queries = 0
    for row in dataset["rows"]:
        prepared = _prepare_query(
            tokenizer,
            f"{row['condition_id']}:action",
            _action_messages(row),
            ["A", "B"],
            spec["enable_thinking"],
        )
        lengths.append(prepared["sequence_length"])
        queries += 1
        for action in ("A", "B"):
            messages, _ = _report_messages(row, action)
            prepared = _prepare_query(
                tokenizer,
                f"{row['condition_id']}:{action}",
                messages,
                ["X", "Y"],
                spec["enable_thinking"],
            )
            lengths.append(prepared["sequence_length"])
            queries += 1
    return json.dumps(
        {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "model_key": model_key,
            "model_id": spec["model_id"],
            "model_revision": spec["revision"],
            "source_dataset_sha256": dataset["content_sha256"],
            "queries_checked": queries,
            "minimum_tokens": min(lengths),
            "maximum_tokens": max(lengths),
            "enable_thinking": spec["enable_thinking"],
            "status": "all_prompt_and_one_token_continuation_checks_passed",
        },
        sort_keys=True,
    )


@app.function(
    image=image,
    gpu="A100-80GB:4",
    cpu=16,
    memory=196608,
    volumes={"/cache": cache},
    timeout=3600,
)
def rbg4_behavior_4xa100(
    dataset: dict[str, Any], config: dict[str, Any], model_key: str
) -> str:
    return _behavior(dataset, config, model_key)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=32768,
    volumes={"/cache": cache},
    timeout=2400,
)
def rbg4_behavior_1xa100(
    dataset: dict[str, Any], config: dict[str, Any], model_key: str
) -> str:
    return _behavior(dataset, config, model_key)


def _behavior(dataset: dict[str, Any], config: dict[str, Any], model_key: str) -> str:
    import torch
    import transformers
    from huggingface_hub import model_info

    spec = MODELS[model_key]
    verify_dataset_payload(dataset, config)
    started = time.perf_counter()
    torch.manual_seed(82429)
    tokenizer = _load_tokenizer(spec)
    tokenizer.padding_side = "left"
    model = _load_model(spec)
    action_queries = [
        _prepare_query(
            tokenizer,
            row["condition_id"],
            _action_messages(row),
            ["A", "B"],
            spec["enable_thinking"],
        )
        for row in dataset["rows"]
    ]
    actions = _query(model, tokenizer, action_queries, spec["batch_size"])
    report_queries = []
    expected_reports: dict[str, str] = {}
    for row in dataset["rows"]:
        for action in ("A", "B"):
            query_id = f"{row['condition_id']}:{action}"
            messages, expected = _report_messages(row, action)
            expected_reports[query_id] = expected
            report_queries.append(
                _prepare_query(
                    tokenizer, query_id, messages, ["X", "Y"], spec["enable_thinking"]
                )
            )
    reports = _query(model, tokenizer, report_queries, spec["batch_size"])
    output = []
    for row in dataset["rows"]:
        condition_id = row["condition_id"]
        selected = actions[condition_id]["legal_choice"]
        option_reports = {}
        for action in ("A", "B"):
            query_id = f"{condition_id}:{action}"
            result = reports[query_id]
            option_reports[action] = {
                "choice": result["legal_choice"],
                "expected": expected_reports[query_id],
                "correct": result["legal_choice"] == expected_reports[query_id],
                "result": result,
            }
        output.append(
            {
                **row,
                "selected_action": selected,
                "action_correct": selected == row["expected_action"],
                "action_result": actions[condition_id],
                "option_reports": option_reports,
            }
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    cache.commit()
    volume_backup = Path("/cache/rbg4_multimodel")
    volume_backup.mkdir(parents=True, exist_ok=True)
    (volume_backup / f"behavior_{model_key}_{started:.0f}.json").write_text(
        "pending", encoding="utf-8"
    )
    payload = {
        "schema_version": 1,
        "metadata": {
            "run_id": uuid.uuid4().hex,
            "created_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed,
            "source_dataset_sha256": dataset["content_sha256"],
            "config_sha256": _canonical_sha256(config),
            "model_key": model_key,
            "model_id": spec["model_id"],
            "model_revision_requested": spec["revision"],
            "model_revision_resolved": model_info(
                spec["model_id"], revision=spec["revision"]
            ).sha,
            "model_family": spec["family"],
            "dtype": "bfloat16",
            "thinking": False,
            "enable_thinking_kwarg": spec["enable_thinking"],
            "gpu_requested": f"{spec['gpu']}x{spec['gpu_count']}",
            "gpu_actual": torch.cuda.get_device_name(0),
            "torch_version": str(torch.__version__),
            "transformers_version": transformers.__version__,
        },
        "rows": sorted(output, key=lambda row: row["condition_id"]),
    }
    rendered = json.dumps(payload, allow_nan=False, sort_keys=True)
    (volume_backup / f"behavior_{model_key}_{started:.0f}.json").write_text(
        rendered, encoding="utf-8"
    )
    cache.commit()
    return rendered


@app.local_entrypoint()
def preflight(model_key: str) -> None:
    config = _load_json(CONFIG_PATH)
    dataset = _load_json(DATASET_PATH)
    verify_dataset_payload(dataset, config)
    payload = json.loads(rbg4_preflight_remote.remote(dataset, config, model_key))
    target = RESULT_ROOT / f"raw/preflight_{model_key}.json"
    _write_new(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


@app.local_entrypoint()
def behavior(model_key: str) -> None:
    config = _load_json(CONFIG_PATH)
    dataset = _load_json(DATASET_PATH)
    verify_dataset_payload(dataset, config)
    spec = MODELS[model_key]
    ledger = RESULT_ROOT / "cost_ledger.jsonl"
    estimate = estimate_cost(
        spec["gpu"],
        float(spec["timeout"]),
        gpu_count=spec["gpu_count"],
        cpu_cores=spec["cpu"],
        memory_gib=spec["memory"] / 1024,
    )
    admit_run(ledger, estimate, study_limit_usd=20.0)
    if spec["gpu_count"] == 4:
        raw = rbg4_behavior_4xa100.remote(dataset, config, model_key)
    elif spec["gpu_count"] == 1:
        raw = rbg4_behavior_1xa100.remote(dataset, config, model_key)
    else:
        raise ValueError(f"no behavior function for gpu_count={spec['gpu_count']}")
    payload = json.loads(raw)
    payload["metadata"]["git_commit"] = _git_head()
    target = RESULT_ROOT / f"raw/behavior_{model_key}.json"
    _write_new(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    measured = estimate_cost(
        spec["gpu"],
        float(payload["metadata"]["elapsed_seconds"]),
        gpu_count=spec["gpu_count"],
        cpu_cores=spec["cpu"],
        memory_gib=spec["memory"] / 1024,
    )
    append_ledger(
        ledger,
        measured,
        run_id=str(payload["metadata"]["run_id"]),
        stage=f"rbg4_multimodel_behavior_{model_key}",
    )
    print(
        json.dumps(
            {
                "run_id": payload["metadata"]["run_id"],
                "model_key": model_key,
                "rows": len(payload["rows"]),
                "elapsed_seconds": payload["metadata"]["elapsed_seconds"],
                "measured_cost_usd": measured.subtotal_usd,
                "buffered_cost_usd": measured.buffered_usd,
                "output": str(target),
            },
            indent=2,
            sort_keys=True,
        )
    )


# --- weight pre-warm (CPU-billed download into the shared volume) ---


@app.function(image=image, cpu=4, memory=16384, volumes={"/cache": cache}, timeout=3600)
def rbg4_download(model_key: str) -> str:
    from huggingface_hub import snapshot_download

    spec = MODELS[model_key]
    started = time.perf_counter()
    path = snapshot_download(spec["model_id"], revision=spec["revision"])
    cache.commit()
    return json.dumps(
        {
            "model_key": model_key,
            "path": path,
            "elapsed_seconds": time.perf_counter() - started,
        },
        sort_keys=True,
    )


@app.local_entrypoint()
def download(model_key: str) -> None:
    print(rbg4_download.remote(model_key))


# --- GPU tier benchmark: fixed query slice, wall-clock + cost per tier ---

# Modal public rates (USD/second) checked 2026-10-23 against modal.com/pricing.
TIER_GPU_USD_PER_SECOND = {
    "A100-80GB": 0.000694,
    "H100": 0.001097,
    "H200": 0.001261,
}
BENCH_QUERIES = 96
BENCH_TOTAL_QUERIES = 2304


def _benchmark(dataset: dict[str, Any], config: dict[str, Any], model_key: str, tier: str) -> str:
    import torch

    spec = MODELS[model_key]
    verify_dataset_payload(dataset, config)
    torch.manual_seed(82429)
    load_started = time.perf_counter()
    tokenizer = _load_tokenizer(spec)
    tokenizer.padding_side = "left"
    model = _load_model(spec)
    load_seconds = time.perf_counter() - load_started
    action_queries = [
        _prepare_query(
            tokenizer,
            row["condition_id"],
            _action_messages(row),
            ["A", "B"],
            spec["enable_thinking"],
        )
        for row in dataset["rows"]
    ]
    ordered = sorted(
        action_queries, key=lambda row: (row["sequence_length"], row["query_id"])
    )
    step = max(1, len(ordered) // BENCH_QUERIES)
    sample = ordered[::step][:BENCH_QUERIES]
    query_started = time.perf_counter()
    _query(model, tokenizer, sample, spec["batch_size"])
    torch.cuda.synchronize()
    query_seconds = time.perf_counter() - query_started
    gpu_rate = TIER_GPU_USD_PER_SECOND[tier.split("x")[0]] * int(tier.split("x")[1])
    full_run_seconds = query_seconds * (BENCH_TOTAL_QUERIES / len(sample)) + load_seconds
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model_key": model_key,
        "model_id": spec["model_id"],
        "tier": tier,
        "gpu_actual": torch.cuda.get_device_name(0),
        "n_queries": len(sample),
        "load_seconds": load_seconds,
        "query_seconds": query_seconds,
        "seconds_per_query": query_seconds / len(sample),
        "extrapolated_full_run_seconds": full_run_seconds,
        "gpu_usd_per_second": gpu_rate,
        "estimated_full_run_gpu_cost_usd": gpu_rate * full_run_seconds,
        "estimated_full_run_gpu_cost_buffered_usd": gpu_rate * full_run_seconds * 1.2,
        "source_dataset_sha256": dataset["content_sha256"],
    }
    rendered = json.dumps(payload, allow_nan=False, sort_keys=True)
    bench_backup = Path("/cache/rbg4_multimodel")
    bench_backup.mkdir(parents=True, exist_ok=True)
    (bench_backup / f"benchmark_{model_key}_{tier}.json").write_text(rendered, encoding="utf-8")
    cache.commit()
    return rendered


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=32768, volumes={"/cache": cache}, timeout=2400)
def bench_a100x1(dataset: dict[str, Any], config: dict[str, Any], model_key: str) -> str:
    return _benchmark(dataset, config, model_key, "A100-80GBx1")


@app.function(image=image, gpu="H100", cpu=8, memory=32768, volumes={"/cache": cache}, timeout=2400)
def bench_h100x1(dataset: dict[str, Any], config: dict[str, Any], model_key: str) -> str:
    return _benchmark(dataset, config, model_key, "H100x1")


@app.function(image=image, gpu="H200", cpu=8, memory=32768, volumes={"/cache": cache}, timeout=2400)
def bench_h200x1(dataset: dict[str, Any], config: dict[str, Any], model_key: str) -> str:
    return _benchmark(dataset, config, model_key, "H200x1")


@app.function(image=image, gpu="A100-80GB:4", cpu=16, memory=196608, volumes={"/cache": cache}, timeout=3600)
def bench_a100x4(dataset: dict[str, Any], config: dict[str, Any], model_key: str) -> str:
    return _benchmark(dataset, config, model_key, "A100-80GBx4")


@app.function(image=image, gpu="H200:2", cpu=16, memory=196608, volumes={"/cache": cache}, timeout=3600)
def bench_h200x2(dataset: dict[str, Any], config: dict[str, Any], model_key: str) -> str:
    return _benchmark(dataset, config, model_key, "H200x2")


BENCH_FUNCTIONS = {
    "A100-80GBx1": bench_a100x1,
    "H100x1": bench_h100x1,
    "H200x1": bench_h200x1,
    "A100-80GBx4": bench_a100x4,
    "H200x2": bench_h200x2,
}

# Benchmark matrix: which tiers each model size class is measured on.
BENCH_MATRIX = {
    "qwen36-27b": ["A100-80GBx1", "H100x1", "H200x1"],
    "qwen3-30b-a3b-instruct": ["A100-80GBx1", "H100x1", "H200x1"],
    "glm45-air": ["A100-80GBx4", "H200x2"],
}


@app.local_entrypoint()
def benchmark(model_key: str) -> None:
    config = _load_json(CONFIG_PATH)
    dataset = _load_json(DATASET_PATH)
    verify_dataset_payload(dataset, config)
    for tier in BENCH_MATRIX[model_key]:
        payload = json.loads(BENCH_FUNCTIONS[tier].remote(dataset, config, model_key))
        target = RESULT_ROOT / f"benchmark/{model_key}_{tier}.json"
        _write_new(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
