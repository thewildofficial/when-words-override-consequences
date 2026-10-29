"""Prospective Modal execution for V2 Stage 1 latent state-report dissociation.

Commands are intentionally separated:

    modal run modal_stage1.py::freeze_dataset
    modal run modal_stage1.py::behavior --phase dev
    modal run modal_stage1.py::mechanistic --phase dev
    modal run modal_stage1.py::behavior --phase locked
    modal run modal_stage1.py::mechanistic --phase locked
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

MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILENAME = "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"

app = modal.App("kuhn-gate-pilot2")
cache = modal.Volume.from_name("jspace-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install(
        "numpy>=2.0",
        "scikit-learn>=1.6",
        "torch>=2.8",
        "transformers>=5.5",
        "huggingface_hub>=0.34",
        f"git+https://github.com/anthropics/jacobian-lens.git@{JLENS_COMMIT}",
    )
    .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src/jspace_policy", remote_path="/root/jspace_policy")
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _content_hash_valid(payload: dict[str, Any]) -> bool:
    claimed = payload.get("content_sha256")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    return claimed == _canonical_sha256(body)


def _continuation_id(tokenizer: Any, prompt: str, answer: str) -> int:
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    full = tokenizer.encode(prompt + answer, add_special_tokens=False)
    if full[: len(prefix)] == prefix and len(full) == len(prefix) + 1:
        return int(full[-1])
    raise ValueError(f"{answer!r} is not one token after frozen answer prefix")


def _tokenize_payload(config: dict[str, Any]) -> dict[str, Any]:
    import transformers

    from jspace_policy.stage1 import dataset_payload

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    payload = dataset_payload(config)
    for row in payload["rows"]:
        task_text = str(row["prompt"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": task_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        row["task_text"] = task_text
        row["task_sha256"] = row["prompt_sha256"]
        row["prompt"] = prompt
        row["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        row["prompt_token_ids"] = tokenizer.encode(prompt, add_special_tokens=False)
        row["candidate_token_ids"] = [
            _continuation_id(tokenizer, prompt, candidate) for candidate in row["candidates"]
        ]
        if len(set(row["candidate_token_ids"])) != 4:
            raise RuntimeError(f"candidate token collision in {row['condition_id']}")
        report_index = row["candidates"].index(row["expected_report"])
        row["expected_report_token_id"] = row["candidate_token_ids"][report_index]
        row["true_state_token_id"] = row["candidate_token_ids"][row["world_state_id"]]
        transformed_index = row["candidates"].index(row["transformed_state"])
        row["transformed_state_token_id"] = row["candidate_token_ids"][transformed_index]
        row["sequence_length"] = len(row["prompt_token_ids"])
    payload["tokenizer_revision"] = MODEL_REVISION
    payload["created_at"] = datetime.now(UTC).isoformat()
    payload["content_sha256"] = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    cache.commit()
    return payload


@app.function(image=image, volumes={"/cache": cache}, cpu=2.0, memory=4096, timeout=900)
def tokenize_dataset_remote(config: dict[str, Any]) -> str:
    return json.dumps(_tokenize_payload(config), sort_keys=True)


def _load_model(*, with_lens: bool) -> tuple[Any, Any | None, dict[str, Any]]:
    import torch
    import transformers
    from huggingface_hub import model_info
    from jlens import JacobianLens, from_hf

    seed = 981723
    torch.manual_seed(seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda()
    hf_model.eval()
    model = from_hf(hf_model, tokenizer)
    lens = None
    if with_lens:
        lens = JacobianLens.from_pretrained(
            LENS_REPO, filename=LENS_FILENAME, revision=LENS_REVISION
        )
        if lens.d_model != model.d_model:
            raise RuntimeError(f"lens width {lens.d_model} != model width {model.d_model}")
    resolved = model_info(MODEL_ID, revision=MODEL_REVISION).sha
    metadata = {
        "model_id": MODEL_ID,
        "model_revision_requested": MODEL_REVISION,
        "model_revision_resolved": resolved,
        "tokenizer_revision": resolved,
        "lens_repo": LENS_REPO if with_lens else None,
        "lens_revision": LENS_REVISION if with_lens else None,
        "lens_filename": LENS_FILENAME if with_lens else None,
        "lens_code_commit": JLENS_COMMIT if with_lens else None,
        "lens_source_layers": [int(layer) for layer in lens.source_layers] if lens else [],
        "seed": seed,
        "dtype": "bfloat16",
        "gpu_requested": "A100-80GB",
        "gpu_actual": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
    }
    return model, lens, metadata


def _select_rows(dataset: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    allowed = {"discovery", "validation"} if phase == "dev" else {"locked_test"}
    if phase not in {"dev", "locked"}:
        raise ValueError("phase must be dev or locked")
    return [row for row in dataset["rows"] if row["split"] in allowed]


def _left_padded_batch(rows: list[dict[str, Any]]) -> tuple[Any, Any, list[int]]:
    import torch

    lengths = [len(row["prompt_token_ids"]) for row in rows]
    width = max(lengths)
    input_ids = torch.zeros((len(rows), width), dtype=torch.long, device="cuda")
    attention = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        tokens = torch.tensor(row["prompt_token_ids"], dtype=torch.long, device="cuda")
        input_ids[index, width - len(tokens) :] = tokens
        attention[index, width - len(tokens) :] = 1
    return input_ids, attention, lengths


def _behavior_rows(
    model: Any, rows: list[dict[str, Any]], batch_size: int
) -> list[dict[str, Any]]:
    import torch

    output: list[dict[str, Any]] = []
    ordered = sorted(rows, key=lambda row: (row["sequence_length"], row["condition_id"]))
    for start in range(0, len(ordered), batch_size):
        part = ordered[start : start + batch_size]
        input_ids, attention, _ = _left_padded_batch(part)
        with torch.inference_mode():
            logits = model._hf_model(
                input_ids,
                attention_mask=attention,
                logits_to_keep=1,
            ).logits
        for index, row in enumerate(part):
            final = logits[index, -1].float()
            log_probs = final.log_softmax(-1)
            candidate_ids = [int(token) for token in row["candidate_token_ids"]]
            top1 = int(final.argmax().detach().cpu())
            output.append(
                {
                    "condition_id": row["condition_id"],
                    "substage": row["substage"],
                    "split": row["split"],
                    "family": row["family"],
                    "base_scenario_id": row["base_scenario_id"],
                    "world_state_id": row["world_state_id"],
                    "policy_id": row["policy_id"],
                    "expected_report_token_id": row["expected_report_token_id"],
                    "top1_token_id": top1,
                    "top1_text": model.tokenizer.decode([top1]),
                    "correct": top1 == int(row["expected_report_token_id"]),
                    "formatting_compliant": top1 in candidate_ids,
                    "candidate_log_probs": [
                        float(log_probs[token].detach().cpu()) for token in candidate_ids
                    ],
                }
            )
        del input_ids, attention, logits
    return sorted(output, key=lambda row: row["condition_id"])


def _behavior_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    gates = config["behavior"]
    substages: dict[str, Any] = {}
    for substage in ("1A", "1B"):
        part = [row for row in rows if row["substage"] == substage]
        overall = sum(row["correct"] for row in part) / len(part)
        cell_accuracy = {}
        for state in range(4):
            for policy in ("T", "M"):
                cell = [
                    row
                    for row in part
                    if row["world_state_id"] == state and row["policy_id"] == policy
                ]
                cell_accuracy[f"x{state}_{policy}"] = sum(row["correct"] for row in cell) / len(
                    cell
                )
        family_accuracy = {}
        for family in sorted({str(row["family"]) for row in part}):
            family_rows = [row for row in part if row["family"] == family]
            family_accuracy[family] = sum(row["correct"] for row in family_rows) / len(
                family_rows
            )
        passed = (
            overall >= float(gates["minimum_overall_accuracy"])
            and min(cell_accuracy.values())
            >= float(gates["minimum_state_policy_cell_accuracy"])
            and min(family_accuracy.values()) >= float(gates["minimum_family_accuracy"])
        )
        substages[substage] = {
            "n_rows": len(part),
            "overall_accuracy": overall,
            "cell_accuracy": cell_accuracy,
            "family_accuracy": family_accuracy,
            "gate_pass": passed,
        }
    return {
        "substages": substages,
        "gate_pass": all(summary["gate_pass"] for summary in substages.values()),
    }


@app.function(
    image=image,
    volumes={"/cache": cache},
    cpu=4.0,
    memory=32768,
    gpu="A100-80GB",
    timeout=3600,
    max_containers=1,
    retries=0,
)
def behavior_remote(
    phase: str,
    dataset: dict[str, Any],
    config: dict[str, Any],
    git_commit: str,
) -> str:
    import torch

    if not _content_hash_valid(dataset):
        raise RuntimeError("frozen Stage 1 dataset hash mismatch")
    started = time.perf_counter()
    model, _, metadata = _load_model(with_lens=False)
    selected = _select_rows(dataset, phase)
    rows = _behavior_rows(model, selected, int(config["behavior"]["batch_size"]))
    summary = _behavior_summary(rows, config)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    cache.commit()
    result = {
        "metadata": {
            "schema_version": 1,
            "run_id": uuid.uuid4().hex,
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": git_commit,
            "phase": phase,
            "dataset_sha256": dataset["content_sha256"],
            "config_sha256": _canonical_sha256(config),
            "activations_saved": False,
            "jlens_opened": False,
            "elapsed_seconds": elapsed,
            **metadata,
        },
        "summary": summary,
        "rows": rows,
    }
    return json.dumps(result, allow_nan=False, sort_keys=True)


def _evidence(scores: list[float], index: int) -> float:
    from jspace_policy.stage1_analysis import candidate_evidence

    return candidate_evidence(scores, index)


def _serialize_estimator(
    scaler: Any, classifier: Any, layer: int, c_value: float
) -> dict[str, Any]:
    return {
        "layer": layer,
        "C": c_value,
        "classes": [int(value) for value in classifier.classes_],
        "scaler_mean": scaler.mean_.astype("float32").tolist(),
        "scaler_scale": scaler.scale_.astype("float32").tolist(),
        "coef": classifier.coef_.astype("float32").tolist(),
        "intercept": classifier.intercept_.astype("float32").tolist(),
    }


def _fit_estimator(
    features: Any, labels: Any, layer: int, c_value: float, max_iter: int
) -> tuple[Any, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(
        C=c_value,
        max_iter=max_iter,
        random_state=981723,
        solver="lbfgs",
    ).fit(scaler.transform(features), labels)
    return scaler, classifier


def _choose_probe(
    feature_by_layer: dict[int, Any],
    metadata: list[dict[str, Any]],
    target: str,
    probe_config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np

    labels = np.asarray([row[target] for row in metadata], dtype="int64")
    discovery = np.asarray([row["split"] == "discovery" for row in metadata])
    validation = np.asarray([row["split"] == "validation" for row in metadata])
    candidates: list[tuple[float, int, float, Any, Any]] = []
    table: list[dict[str, Any]] = []
    for layer in probe_config["candidate_layers"]:
        layer = int(layer)
        for c_value in probe_config["regularization_c"]:
            c_value = float(c_value)
            scaler, classifier = _fit_estimator(
                feature_by_layer[layer][discovery],
                labels[discovery],
                layer,
                c_value,
                int(probe_config["maximum_iterations"]),
            )
            accuracy = float(
                classifier.score(
                    scaler.transform(feature_by_layer[layer][validation]), labels[validation]
                )
            )
            table.append({"layer": layer, "C": c_value, "validation_accuracy": accuracy})
            candidates.append((accuracy, layer, c_value, scaler, classifier))
    winner = sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))[0]
    return _serialize_estimator(winner[3], winner[4], winner[1], winner[2]), table


def _choose_cross_policy_probe(
    feature_by_layer: dict[int, Any],
    metadata: list[dict[str, Any]],
    probe_config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np

    labels = np.asarray([row["state_target"] for row in metadata], dtype="int64")
    split = np.asarray([row["split"] for row in metadata])
    policy = np.asarray([row["policy_id"] for row in metadata])
    candidates: list[tuple[float, int, float, Any, Any, Any, Any]] = []
    table: list[dict[str, Any]] = []
    for layer in probe_config["candidate_layers"]:
        layer = int(layer)
        for c_value in probe_config["regularization_c"]:
            c_value = float(c_value)
            train_t = (split == "discovery") & (policy == "T")
            train_m = (split == "discovery") & (policy == "M")
            val_t = (split == "validation") & (policy == "T")
            val_m = (split == "validation") & (policy == "M")
            scaler_t, classifier_t = _fit_estimator(
                feature_by_layer[layer][train_t],
                labels[train_t],
                layer,
                c_value,
                int(probe_config["maximum_iterations"]),
            )
            scaler_m, classifier_m = _fit_estimator(
                feature_by_layer[layer][train_m],
                labels[train_m],
                layer,
                c_value,
                int(probe_config["maximum_iterations"]),
            )
            t_to_m = float(
                classifier_t.score(
                    scaler_t.transform(feature_by_layer[layer][val_m]), labels[val_m]
                )
            )
            m_to_t = float(
                classifier_m.score(
                    scaler_m.transform(feature_by_layer[layer][val_t]), labels[val_t]
                )
            )
            mean_accuracy = (t_to_m + m_to_t) / 2
            table.append(
                {
                    "layer": layer,
                    "C": c_value,
                    "truth_to_transformed_validation_accuracy": t_to_m,
                    "transformed_to_truth_validation_accuracy": m_to_t,
                    "mean_validation_accuracy": mean_accuracy,
                }
            )
            candidates.append(
                (mean_accuracy, layer, c_value, scaler_t, classifier_t, scaler_m, classifier_m)
            )
    winner = sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))[0]
    return {
        "layer": winner[1],
        "C": winner[2],
        "truth_trained": _serialize_estimator(winner[3], winner[4], winner[1], winner[2]),
        "transformed_trained": _serialize_estimator(winner[5], winner[6], winner[1], winner[2]),
    }, table


def _predict_serialized(
    model: dict[str, Any], features: Any
) -> tuple[list[int], list[list[float]]]:
    import numpy as np

    mean = np.asarray(model["scaler_mean"], dtype="float32")
    scale = np.asarray(model["scaler_scale"], dtype="float32")
    coef = np.asarray(model["coef"], dtype="float32")
    intercept = np.asarray(model["intercept"], dtype="float32")
    standardized = (features - mean) / scale
    logits = standardized @ coef.T + intercept
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    classes = np.asarray(model["classes"], dtype="int64")
    predictions = classes[probabilities.argmax(axis=1)]
    return predictions.tolist(), probabilities.tolist()


def _attach_probes(
    rows: list[dict[str, Any]],
    feature_by_substage: dict[str, dict[int, Any]],
    metadata_by_substage: dict[str, list[dict[str, Any]]],
    artifact: dict[str, Any],
) -> None:
    rows_by_condition = {row["condition_id"]: row for row in rows}
    for substage in ("1A", "1B"):
        metadata = metadata_by_substage[substage]
        features = feature_by_substage[substage]
        models = artifact["substages"][substage]["models"]
        for name in ("state", "report"):
            model = models[name]
            predictions, probabilities = _predict_serialized(
                model, features[int(model["layer"])]
            )
            for index, item in enumerate(metadata):
                row = rows_by_condition[item["condition_id"]]
                row[f"{name}_probe_prediction"] = predictions[index]
                row[f"{name}_probe_probabilities"] = probabilities[index]
        cross = models["cross_policy_state"]
        for prefix, model_name in (
            ("truth_trained", "truth_trained"),
            ("transformed_trained", "transformed_trained"),
        ):
            model = cross[model_name]
            predictions, probabilities = _predict_serialized(
                model, features[int(model["layer"])]
            )
            for index, item in enumerate(metadata):
                row = rows_by_condition[item["condition_id"]]
                row[f"cross_{prefix}_state_prediction"] = predictions[index]
                row[f"cross_{prefix}_state_probabilities"] = probabilities[index]


def _mechanistic_rows(
    model: Any,
    lens: Any,
    selected: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[int, Any]], dict[str, list[dict[str, Any]]]]:
    import numpy as np
    import torch
    from jlens import ActivationRecorder

    layers = [int(layer) for layer in config["primary_layers"]]
    by_substage_features: dict[str, dict[int, list[Any]]] = {
        substage: {layer: [] for layer in layers} for substage in ("1A", "1B")
    }
    by_substage_metadata: dict[str, list[dict[str, Any]]] = {"1A": [], "1B": []}
    output: list[dict[str, Any]] = []
    ordered = sorted(selected, key=lambda row: (row["sequence_length"], row["condition_id"]))
    batch_size = int(config["behavior"]["batch_size"])
    for start in range(0, len(ordered), batch_size):
        part = ordered[start : start + batch_size]
        input_ids, attention, _ = _left_padded_batch(part)
        with torch.inference_mode(), ActivationRecorder(model.layers, at=layers) as recorder:
            model_output = model._hf_model(
                input_ids,
                attention_mask=attention,
                logits_to_keep=1,
            )
        for index, row in enumerate(part):
            final_position = int(input_ids.shape[1]) - 1
            candidate_ids = [int(token) for token in row["candidate_token_ids"]]
            output_logits = model_output.logits[index, -1].float()
            output_log_probs = output_logits.log_softmax(-1)
            output_scores = [
                float(output_logits[token].detach().cpu()) for token in candidate_ids
            ]
            true_index = int(row["world_state_id"])
            transformed_index = row["candidates"].index(row["transformed_state"])
            report_index = row["candidates"].index(row["expected_report"])
            layer_rows = []
            for layer in layers:
                residual = recorder.activations[layer][index, final_position].detach().float()
                transported = residual @ lens.jacobians[layer].to(residual.device).T
                jlens_full = model.unembed(transported).float()
                logit_lens_full = model.unembed(residual).float()
                jlens_scores = [
                    float(jlens_full[token].detach().cpu()) for token in candidate_ids
                ]
                logit_lens_scores = [
                    float(logit_lens_full[token].detach().cpu()) for token in candidate_ids
                ]
                layer_rows.append(
                    {
                        "layer": layer,
                        "jlens_candidate_scores": jlens_scores,
                        "logit_lens_candidate_scores": logit_lens_scores,
                        "K": _evidence(jlens_scores, true_index),
                        "Q": _evidence(jlens_scores, report_index),
                        "M": _evidence(jlens_scores, transformed_index),
                        "D": _evidence(jlens_scores, true_index)
                        - _evidence(jlens_scores, report_index),
                        "logit_lens_K": _evidence(logit_lens_scores, true_index),
                        "logit_lens_M": _evidence(logit_lens_scores, transformed_index),
                    }
                )
                by_substage_features[row["substage"]][layer].append(residual.cpu().numpy())
            band_k = sum(item["K"] for item in layer_rows) / len(layer_rows)
            band_q = sum(item["Q"] for item in layer_rows) / len(layer_rows)
            band_m = sum(item["M"] for item in layer_rows) / len(layer_rows)
            output.append(
                {
                    "condition_id": row["condition_id"],
                    "substage": row["substage"],
                    "split": row["split"],
                    "family": row["family"],
                    "base_scenario_id": row["base_scenario_id"],
                    "world_state_id": true_index,
                    "policy_id": row["policy_id"],
                    "report_target": report_index,
                    "prompt_sha256": row["prompt_sha256"],
                    "candidate_token_ids": candidate_ids,
                    "output_top1_token_id": int(output_logits.argmax().detach().cpu()),
                    "output_candidate_log_probs": [
                        float(output_log_probs[token].detach().cpu()) for token in candidate_ids
                    ],
                    "output_K": _evidence(output_scores, true_index),
                    "output_M": _evidence(output_scores, transformed_index),
                    "layer_scores": layer_rows,
                    "band_K": band_k,
                    "band_Q": band_q,
                    "band_M": band_m,
                    "band_D": band_k - band_q,
                }
            )
            by_substage_metadata[row["substage"]].append(
                {
                    "condition_id": row["condition_id"],
                    "split": row["split"],
                    "policy_id": row["policy_id"],
                    "state_target": true_index,
                    "report_target": report_index,
                }
            )
        del input_ids, attention, model_output
    arrays = {
        substage: {
            layer: np.stack(values).astype("float32") for layer, values in layer_map.items()
        }
        for substage, layer_map in by_substage_features.items()
    }
    return sorted(output, key=lambda row: row["condition_id"]), arrays, by_substage_metadata


def _fit_probe_artifact(
    features: dict[str, dict[int, Any]],
    metadata: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    dataset_sha256: str,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "status": "frozen_after_dev_before_locked_test",
        "dataset_sha256": dataset_sha256,
        "config_sha256": _canonical_sha256(config),
        "created_at": datetime.now(UTC).isoformat(),
        "substages": {},
    }
    for substage in ("1A", "1B"):
        state, state_table = _choose_probe(
            features[substage], metadata[substage], "state_target", config["probe"]
        )
        report, report_table = _choose_probe(
            features[substage], metadata[substage], "report_target", config["probe"]
        )
        cross, cross_table = _choose_cross_policy_probe(
            features[substage], metadata[substage], config["probe"]
        )
        artifact["substages"][substage] = {
            "models": {"state": state, "report": report, "cross_policy_state": cross},
            "selection_tables": {
                "state": state_table,
                "report": report_table,
                "cross_policy_state": cross_table,
            },
        }
    artifact["content_sha256"] = _canonical_sha256(artifact)
    return artifact






@app.function(image=image, volumes={"/cache": cache}, cpu=8.0, memory=65536, gpu="A100-80GB", timeout=1200, max_containers=1, retries=0)
def kuhn_remote(rows: list) -> str:
    import torch, transformers
    started = time.perf_counter()
    model, _, metadata = _load_model(with_lens=False)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    def build(text, prefill):
        p = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return p + prefill
    manual = "<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    official = build("x", "")
    info = {"template_matches_manual": official == manual, "official_tail": official[-80:]}
    variants = {"V0_orig_official_template": ("orig", ""), "V1_explicit_noexplain": ("explicit", ""), "V2_explicit_prefill": ("explicit", "The better option is")}
    for vname, (kind, prefill) in variants.items():
        ctx = build("x", prefill)
        sp = " " if prefill else ""
        A = _continuation_id(tok, ctx, sp + "A"); B = _continuation_id(tok, ctx, sp + "B")
        for start in range(0, len(rows), 8):
            part = rows[start:start + 8]
            texts = [r["prompt"] if kind == "orig" else r["prompt"].replace("Return only A or B.", "Reply with only the single letter A or B and nothing else. Do not reason or explain.") for r in part]
            ids = [tok.encode(build(t, prefill), add_special_tokens=False) for t in texts]
            w = max(map(len, ids))
            x = torch.zeros((len(part), w), dtype=torch.long, device="cuda"); m = torch.zeros_like(x)
            for i, t in enumerate(ids):
                x[i, w - len(t):] = torch.tensor(t, device="cuda"); m[i, w - len(t):] = 1
            with torch.inference_mode():
                lg = model._hf_model(x, attention_mask=m, logits_to_keep=1).logits[:, -1].float()
            for i, r in enumerate(part):
                r.setdefault("v", {})[vname] = {"A": float(lg[i, A]), "B": float(lg[i, B]), "top1": tok.decode([int(lg[i].argmax())])}
    return json.dumps({"elapsed": time.perf_counter() - started, "rows": rows, "info": info})
