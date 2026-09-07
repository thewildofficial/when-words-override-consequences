"""V6 execution entrypoints for epistemic and strategic model tests.

The stages are intentionally separate:

* ``preflight`` uses CPU-only tokenizer and prompt checks;
* ``blackbox`` uses free generation and only the returned text;
* ``behavior`` uses one-step candidate-token logits without activations;
* ``trajectory`` runs matched report/action order controls;
* ``activation_discovery`` reads residual states for held-out probes;
* ``activation_locked`` remains sealed until a fresh causal patch protocol is
  explicitly frozen.

The primary model is Qwen3.6-27B.  Qwen3.8-27B is available only through an
explicit model key and is blocked from paid stages until its parity preflight is
promoted by hand.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

from jspace_policy.budget import admit_run, append_ledger, estimate_cost
from jspace_policy.epistemic_strategic_experiments import (
    ACTION_LABELS,
    REPORT_LABELS,
    STUDY_ID,
    canonical_sha256,
    dataset_payload,
    trajectory_messages,
    verify_dataset_payload,
)

CONFIG_PATH = Path("configs/v6/epistemic_strategic/experiment.json")
DATASET_PATH = Path("configs/v6/epistemic_strategic/dataset.json")
RESULT_ROOT = Path("results/v6_epistemic_strategic")
LEDGER_PATH = RESULT_ROOT / "cost_ledger.jsonl"

app = modal.App("jspace-v6-epistemic-strategic")
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


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _config() -> dict[str, Any]:
    config = _load_json(CONFIG_PATH)
    if config["study_id"] != STUDY_ID:
        raise RuntimeError("V6 study ID changed")
    return config


def _dataset(config: dict[str, Any]) -> dict[str, Any]:
    dataset = _load_json(DATASET_PATH)
    verify_dataset_payload(dataset, config)
    return dataset


def _model_spec(config: dict[str, Any], model_key: str) -> dict[str, Any]:
    if model_key == "primary":
        return config["model"]
    try:
        spec = config["replication_models"][model_key]
    except KeyError as error:
        raise RuntimeError(f"unknown V6 model key: {model_key}") from error
    if not spec.get("enabled", False):
        raise RuntimeError(
            f"{model_key} is parity-locked; run preflight first and promote it explicitly"
        )
    return spec


def _model_spec_for_preflight(config: dict[str, Any], model_key: str) -> dict[str, Any]:
    if model_key == "primary":
        return config["model"]
    try:
        return config["replication_models"][model_key]
    except KeyError as error:
        raise RuntimeError(f"unknown V6 model key: {model_key}") from error


def _validate_config(config: dict[str, Any]) -> None:
    if config["status"] != "preregistered_before_dataset_freeze_or_model_execution":
        raise RuntimeError("protocol is not prospectively frozen")
    dataset = config["dataset"]
    if dataset["action_labels"] != list(ACTION_LABELS):
        raise RuntimeError("action labels changed")
    if dataset["report_labels"] != list(REPORT_LABELS):
        raise RuntimeError("report labels changed")
    if config["behavior"]["query_mode"] != "forced_choice_next_token_logits":
        raise RuntimeError("logit behavior measurement mode changed")
    if config["blackbox"]["parser"] != "first_legal_label_after_optional_reasoning":
        raise RuntimeError("black-box parser changed")
    if float(config["execution"]["hard_cost_limit_usd"]) > 4.0:
        raise RuntimeError("V6 hard cost ceiling may not exceed USD 4")


def _config_sha256(config: dict[str, Any]) -> str:
    return canonical_sha256(config)


def _hash_valid(payload: dict[str, Any]) -> bool:
    claimed = payload.get("content_sha256")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    return claimed == canonical_sha256(body)


def _continuation_id(tokenizer: Any, rendered: str, answer: str) -> int:
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    full = tokenizer.encode(rendered + answer, add_special_tokens=False)
    if full[: len(prefix)] == prefix and len(full) == len(prefix) + 1:
        return int(full[-1])
    raise ValueError(f"{answer!r} is not one token after the frozen prompt")


def _render(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
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

    model_id = str(spec["id"])
    revision = str(spec["revision"])
    if "3.8" in model_id:
        processor = transformers.AutoProcessor.from_pretrained(
            model_id, revision=revision
        )
        tokenizer = getattr(processor, "tokenizer", processor)
        loader = "AutoProcessor"
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, revision=revision
        )
        loader = "AutoTokenizer"
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        raise RuntimeError("tokenizer has no padding or EOS token")
    return tokenizer, {"tokenizer_loader": loader}


def _prepare_query(
    tokenizer: Any,
    query_id: str,
    messages: list[dict[str, str]],
    candidates: tuple[str, ...],
) -> dict[str, Any]:
    rendered = _render(tokenizer, messages)
    token_ids = list(map(int, tokenizer.encode(rendered, add_special_tokens=False)))
    candidate_ids = [
        _continuation_id(tokenizer, rendered, candidate) for candidate in candidates
    ]
    return {
        "query_id": query_id,
        "messages": messages,
        "rendered": rendered,
        "prompt_token_ids": token_ids,
        "candidate_labels": list(candidates),
        "candidate_token_ids": candidate_ids,
        "sequence_length": len(token_ids),
    }


def _candidates(row: dict[str, Any]) -> tuple[str, ...]:
    return ACTION_LABELS if row["task_kind"] == "action" else REPORT_LABELS


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
        except (OSError, ValueError, TypeError):
            continue
    if model is None or loader_name is None:
        raise RuntimeError(f"could not load model class for {spec['id']}")
    model.eval()
    resolved = model_info(spec["id"], revision=spec["revision"]).sha
    metadata = {
        "model_id": spec["id"],
        "model_revision_requested": spec["revision"],
        "model_revision_resolved": resolved,
        "dtype": spec.get("dtype"),
        "thinking": spec.get("thinking"),
        "model_loader": loader_name,
        **tokenizer_meta,
        "gpu_actual": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
    }
    return model, tokenizer, metadata


def _all_preflight_queries(tokenizer: Any, dataset: dict[str, Any]) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for row in dataset["rows"]:
        queries.append(
            _prepare_query(
                tokenizer,
                f"{row['condition_id']}:action",
                trajectory_messages(row, condition="direct_action"),
                ACTION_LABELS,
            )
        )
        queries.append(
            _prepare_query(
                tokenizer,
                f"{row['condition_id']}:report",
                trajectory_messages(row, condition="direct_report"),
                REPORT_LABELS,
            )
        )
    core_locked = [
        row
        for row in dataset["rows"]
        if row["experiment_family"] == "core_tom" and row["split"] == "locked"
    ]
    for row in core_locked:
        queries.append(
            _prepare_query(
                tokenizer,
                f"{row['condition_id']}:action_then_report",
                trajectory_messages(
                    row,
                    condition="action_then_report",
                    first_answer=row["expected_action"],
                ),
                REPORT_LABELS,
            )
        )
        queries.append(
            _prepare_query(
                tokenizer,
                f"{row['condition_id']}:report_then_action",
                trajectory_messages(
                    row,
                    condition="report_then_action",
                    first_answer=row["expected_report"],
                ),
                ACTION_LABELS,
            )
        )
    return queries


def _family_records(
    dataset: dict[str, Any],
    outputs: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        result = outputs[row["condition_id"]]
        expected = (
            row["expected_action"]
            if row["task_kind"] == "action"
            else row["expected_report"]
        )
        records.append(
            {
                "condition_id": row["condition_id"],
                "experiment_family": row["experiment_family"],
                "task_kind": row["task_kind"],
                "split": row["split"],
                "game_id": row["game_id"],
                "actual_other_belief": row["actual_other_belief"],
                "modeled_other_belief": row["modeled_other_belief"],
                "report_target": row["report_target"],
                "visibility": row["visibility"],
                "source": row["source"],
                "truth_status": row["truth_status"],
                "temporal_position": row["temporal_position"],
                "surface": row["surface"],
                "decision_frame": row["decision_frame"],
                "evaluation_context": row["evaluation_context"],
                "expected": expected,
                "selected": result["legal_choice"],
                "correct": result["legal_choice"] == expected,
                "predicted_value": (
                    row["report_mapping"].get(result["legal_choice"])
                    if row["task_kind"] == "report"
                    else None
                ),
                "result": result,
            }
        )
    return records


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _behavior_summary(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    gates = config["gates"]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_family[record["experiment_family"]].append(record)

    core_pairs: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    core_actual_pairs: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in by_family["core_tom"]:
        core_pairs[
            (record["split"], record["game_id"], record["actual_other_belief"])
        ].append(record)
        core_actual_pairs[
            (record["split"], record["game_id"], record["modeled_other_belief"])
        ].append(record)
    core_pair_rows = []
    for key, pair in sorted(core_pairs.items()):
        by_modeled = {row["modeled_other_belief"]: row for row in pair}
        if set(by_modeled) != {0, 1}:
            continue
        core_pair_rows.append(
            {
                "group": list(key),
                "both_correct": all(row["correct"] for row in by_modeled.values()),
                "selected_action_switch": by_modeled[0]["selected"]
                != by_modeled[1]["selected"],
                "expected_action_switch": by_modeled[0]["expected"]
                != by_modeled[1]["expected"],
            }
        )
    actual_invariance = [
        len({row["selected"] for row in pair}) == 1
        for pair in core_actual_pairs.values()
        if len(pair) == 2
    ]
    core_rows = by_family["core_tom"]
    core_pair_accuracy = _safe_rate(
        sum(row["both_correct"] and row["selected_action_switch"] for row in core_pair_rows),
        len(core_pair_rows),
    )
    core_gate = {
        "n_rows": len(core_rows),
        "accuracy": _safe_rate(sum(row["correct"] for row in core_rows), len(core_rows)),
        "pair_identification_accuracy": core_pair_accuracy,
        "actual_hidden_state_invariance": _safe_rate(
            sum(actual_invariance), len(actual_invariance)
        ),
        "gate_pass": (
            _safe_rate(sum(row["correct"] for row in core_rows), len(core_rows))
            >= float(gates["minimum_core_action_accuracy"])
            and core_pair_accuracy >= float(gates["minimum_core_tom_pair_accuracy"])
            and _safe_rate(sum(actual_invariance), len(actual_invariance))
            >= float(gates["minimum_core_actual_invariance"])
        ),
        "stop_if_failed": "do not promote strategic ToM or open activation patching",
    }

    higher_rows = by_family["higher_order"]
    higher_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    higher_visibility: dict[str, list[dict[str, Any]]] = defaultdict(list)
    higher_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in higher_rows:
        higher_by_target[row["report_target"]].append(row)
        higher_visibility[row["visibility"]].append(row)
        higher_by_split[row["split"]].append(row)
    higher_target_accuracy = {
        target: _safe_rate(sum(row["correct"] for row in target_rows), len(target_rows))
        for target, target_rows in sorted(higher_by_target.items())
    }
    higher_gate = {
        "n_rows": len(higher_rows),
        "accuracy": _safe_rate(sum(row["correct"] for row in higher_rows), len(higher_rows)),
        "accuracy_by_target": higher_target_accuracy,
        "accuracy_by_visibility": {
            visibility: _safe_rate(
                sum(row["correct"] for row in visibility_rows), len(visibility_rows)
            )
            for visibility, visibility_rows in sorted(higher_visibility.items())
        },
        "accuracy_by_split": {
            split: _safe_rate(
                sum(row["correct"] for row in split_rows), len(split_rows)
            )
            for split, split_rows in sorted(higher_by_split.items())
        },
        "gate_pass": (
            _safe_rate(sum(row["correct"] for row in higher_rows), len(higher_rows))
            >= float(gates["minimum_report_accuracy"])
            and all(
                accuracy >= float(gates["minimum_report_accuracy"])
                for accuracy in higher_target_accuracy.values()
            )
            and len(higher_by_split["discovery"]) > 0
            and _safe_rate(
                sum(row["correct"] for row in higher_by_split["discovery"]),
                len(higher_by_split["discovery"]),
            )
            >= float(gates["minimum_discovery_report_accuracy"])
            and all(
                _safe_rate(
                    sum(row["correct"] for row in higher_by_split[split]),
                    len(higher_by_split[split]),
                )
                >= float(gates["minimum_report_accuracy"])
                for split in ("validation", "locked")
            )
        ),
        "stop_if_failed": "do not claim role-indexed higher-order belief use",
    }

    provenance_rows = by_family["provenance"]
    provenance_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in provenance_rows:
        provenance_groups[
            (row["split"], row["game_id"], row["modeled_other_belief"])
        ].append(row)
    provenance_invariance = [
        len({row["predicted_value"] for row in group}) == 1
        for group in provenance_groups.values()
        if group
    ]
    provenance_gate = {
        "n_rows": len(provenance_rows),
        "accuracy": _safe_rate(
            sum(row["correct"] for row in provenance_rows), len(provenance_rows)
        ),
        "claim_content_invariance": _safe_rate(
            sum(provenance_invariance), len(provenance_invariance)
        ),
        "accuracy_by_source": {
            source: _safe_rate(
                sum(row["correct"] for row in source_rows), len(source_rows)
            )
            for source, source_rows in sorted(
                _group_by(provenance_rows, "source").items()
            )
        },
        "accuracy_by_surface": {
            surface: _safe_rate(
                sum(row["correct"] for row in surface_rows), len(surface_rows)
            )
            for surface, surface_rows in sorted(
                _group_by(provenance_rows, "surface").items()
            )
        },
        "gate_pass": (
            _safe_rate(
                sum(row["correct"] for row in provenance_rows), len(provenance_rows)
            )
            >= float(gates["minimum_report_accuracy"])
            and _safe_rate(sum(provenance_invariance), len(provenance_invariance))
            >= float(gates["minimum_provenance_invariance"])
        ),
        "stop_if_failed": (
            "do not pool provenance, temporal, and surface effects into truth claims"
        ),
    }

    pressure_rows = by_family["strategy_pressure"]
    evaluation_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    risk_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in pressure_rows:
        evaluation_groups[
            (row["split"], row["game_id"], row["decision_frame"])
        ].append(row)
        risk_groups[
            (row["split"], row["game_id"], row["modeled_other_belief"])
        ].append(row)
    evaluation_discordance = []
    for group in evaluation_groups.values():
        by_context = {row["evaluation_context"]: row for row in group}
        if set(by_context) == {"ordinary", "audited"}:
            evaluation_discordance.append(
                by_context["ordinary"]["selected"] != by_context["audited"]["selected"]
            )
    risk_identification = []
    for group in risk_groups.values():
        by_frame = {row["decision_frame"]: row for row in group}
        if set(by_frame) != {"utility_only", "risk_sensitive"}:
            continue
        if by_frame["utility_only"]["expected"] != by_frame["risk_sensitive"]["expected"]:
            risk_identification.append(
                {
                    "both_correct": all(row["correct"] for row in by_frame.values()),
                    "selected_switch": by_frame["utility_only"]["selected"]
                    != by_frame["risk_sensitive"]["selected"],
                }
            )
    pressure_gate = {
        "n_rows": len(pressure_rows),
        "accuracy": _safe_rate(
            sum(row["correct"] for row in pressure_rows), len(pressure_rows)
        ),
        "evaluation_framing_discordance": _safe_rate(
            sum(evaluation_discordance), len(evaluation_discordance)
        ),
        "risk_reweighting_cells": len(risk_identification),
        "risk_reweighting_pair_accuracy": _safe_rate(
            sum(row["both_correct"] and row["selected_switch"] for row in risk_identification),
            len(risk_identification),
        ),
        "gate_pass": (
            _safe_rate(sum(row["correct"] for row in pressure_rows), len(pressure_rows))
            >= float(gates["minimum_strategy_pressure_accuracy"])
            and _safe_rate(sum(evaluation_discordance), len(evaluation_discordance))
            <= float(gates["maximum_evaluation_framing_discordance"])
            and _safe_rate(
                sum(
                    row["both_correct"] and row["selected_switch"]
                    for row in risk_identification
                ),
                len(risk_identification),
            )
            >= float(gates["minimum_risk_pair_accuracy"])
        ),
        "stop_if_failed": (
            "do not call audit-framing effects evaluation awareness without the risk control"
        ),
    }
    family_gates = {
        "core_tom": core_gate,
        "higher_order": higher_gate,
        "provenance": provenance_gate,
        "strategy_pressure": pressure_gate,
    }
    return {
        "n_records": len(records),
        "accuracy": _safe_rate(sum(row["correct"] for row in records), len(records)),
        "by_family": {
            family: {
                "n": len(family_rows),
                "accuracy": _safe_rate(
                    sum(row["correct"] for row in family_rows), len(family_rows)
                ),
            }
            for family, family_rows in sorted(by_family.items())
        },
        "core_tom_pair_rows": core_pair_rows,
        "family_gates": family_gates,
        "gate_pass": all(gate["gate_pass"] for gate in family_gates.values()),
    }


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return groups


def _parse_blackbox(text: str, candidates: tuple[str, ...]) -> str | None:
    final_matches = re.findall(r"FINAL\s*:?\s*\b([A-Z])\b", text.upper())
    for match in reversed(final_matches):
        if match in candidates:
            return match
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    for match in reversed(matches):
        if match in candidates:
            return match
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
        candidates = tuple(row["candidate_labels"])
        selected = _parse_blackbox(text, candidates)
        output[row["query_id"]] = {
            "generated_text": text,
            "legal_choice": selected,
            "parseable": selected is not None,
        }
    del input_ids, attention, generated
    return output


def _generation_query_rows(tokenizer: Any, dataset: dict[str, Any]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in dataset["rows"]
        if row["split"] == "locked"
    ]
    queries = []
    for row in selected:
        messages = trajectory_messages(
            row,
            condition="direct_action" if row["task_kind"] == "action" else "direct_report",
        )
        query = _prepare_query(
            tokenizer,
            row["condition_id"],
            messages,
            _candidates(row),
        )
        queries.append(query)
    return queries


def _record_cost(config: dict[str, Any], stage: str, payload: dict[str, Any]) -> None:
    seconds = float(payload["metadata"]["elapsed_seconds"])
    estimate = estimate_cost(
        str(config["execution"]["gpu"]),
        seconds,
        cpu_cores=8,
        memory_gib=32,
    )
    append_ledger(
        LEDGER_PATH,
        estimate,
        run_id=str(payload["metadata"]["run_id"]),
        stage=f"{STUDY_ID}:{stage}",
    )


def _admit_gpu(config: dict[str, Any], stage: str) -> None:
    seconds = float(config["execution"]["estimated_ceiling_seconds"])
    estimate = estimate_cost(
        str(config["execution"]["gpu"]), seconds, cpu_cores=8, memory_gib=32
    )
    admit_run(
        LEDGER_PATH,
        estimate,
        study_limit_usd=float(config["execution"]["hard_cost_limit_usd"]),
    )
    if stage not in {"blackbox", "behavior", "trajectory", "activation_discovery"}:
        raise RuntimeError(f"unknown paid stage: {stage}")


@app.function(image=image, volumes={"/cache": cache}, cpu=2, memory=8192, timeout=900)
def preflight_remote(
    dataset: dict[str, Any], config: dict[str, Any], model_key: str
) -> str:
    _validate_config(config)
    verify_dataset_payload(dataset, config)
    spec = _model_spec_for_preflight(config, model_key)
    tokenizer, tokenizer_metadata = _load_tokenizer(spec)
    queries = _all_preflight_queries(tokenizer, dataset)
    lengths = [query["sequence_length"] for query in queries]
    payload = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "tokenizer_preflight_passed",
        "created_at": datetime.now(UTC).isoformat(),
        "model_key": model_key,
        "model_id": spec["id"],
        "model_revision_requested": spec["revision"],
        "source_dataset_sha256": dataset["content_sha256"],
        "queries_checked": len(queries),
        "minimum_tokens": min(lengths),
        "maximum_tokens": max(lengths),
        "candidate_token_sets": {
            "actions": list(ACTION_LABELS),
            "reports": list(REPORT_LABELS),
        },
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
    timeout=1200,
)
def behavior_remote(
    dataset: dict[str, Any], config: dict[str, Any], model_key: str
) -> str:
    _validate_config(config)
    spec = _model_spec(config, model_key)
    started = time.perf_counter()
    model, tokenizer, model_metadata = _load_model(spec)
    queries = [
        _prepare_query(
            tokenizer,
            row["condition_id"],
            trajectory_messages(
                row,
                condition="direct_action"
                if row["task_kind"] == "action"
                else "direct_report",
            ),
            _candidates(row),
        )
        for row in dataset["rows"]
    ]
    outputs = _logit_query(
        model,
        tokenizer,
        queries,
        int(config["behavior"]["batch_size"]),
    )
    records = _family_records(dataset, outputs, dataset["rows"])
    torch = __import__("torch")
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
            "config_sha256": _config_sha256(config),
            **model_metadata,
        },
        "summary": _behavior_summary(records, config),
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
    timeout=1200,
)
def blackbox_remote(
    dataset: dict[str, Any], config: dict[str, Any], model_key: str
) -> str:
    _validate_config(config)
    spec = _model_spec(config, model_key)
    started = time.perf_counter()
    model, tokenizer, model_metadata = _load_model(spec)
    queries = _generation_query_rows(tokenizer, dataset)
    outputs: dict[str, dict[str, Any]] = {}
    batch_size = int(config["behavior"]["batch_size"])
    for start in range(0, len(queries), batch_size):
        outputs.update(
            _generation_batch(
                model,
                tokenizer,
                queries[start : start + batch_size],
                max_new_tokens=int(config["blackbox"]["max_new_tokens"]),
            )
        )
    selected_rows = [
        row for row in dataset["rows"] if row["split"] == config["blackbox"]["primary_split"]
    ]
    records = []
    for row in selected_rows:
        result = outputs[row["condition_id"]]
        expected = (
            row["expected_action"]
            if row["task_kind"] == "action"
            else row["expected_report"]
        )
        records.append(
            {
                "condition_id": row["condition_id"],
                "experiment_family": row["experiment_family"],
                "task_kind": row["task_kind"],
                "split": row["split"],
                "expected": expected,
                "selected": result["legal_choice"],
                "parseable": result["parseable"],
                "correct": result["legal_choice"] == expected,
                "generated_text": result["generated_text"],
            }
        )
    torch = __import__("torch")
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
            "config_sha256": _config_sha256(config),
            "measurement_mode": "free_generation_no_logits_or_activations_exposed",
            **model_metadata,
        },
        "summary": {
            "n_records": len(records),
            "parse_rate": sum(row["parseable"] for row in records) / len(records),
            "accuracy": sum(row["correct"] for row in records) / len(records),
            "by_family": {
                family: {
                    "n": len(family_rows),
                    "parse_rate": sum(row["parseable"] for row in family_rows)
                    / len(family_rows),
                    "accuracy": sum(row["correct"] for row in family_rows)
                    / len(family_rows),
                }
                for family, family_rows in sorted(
                    _group_by(records, "experiment_family").items()
                )
            },
        },
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
    timeout=1200,
)
def trajectory_remote(
    dataset: dict[str, Any], config: dict[str, Any], model_key: str
) -> str:
    _validate_config(config)
    spec = _model_spec(config, model_key)
    started = time.perf_counter()
    model, tokenizer, model_metadata = _load_model(spec)
    rows = [
        row
        for row in dataset["rows"]
        if row["experiment_family"] == "core_tom" and row["split"] == "locked"
    ]
    direct_action_queries = [
        _prepare_query(
            tokenizer,
            f"{row['condition_id']}:direct_action",
            trajectory_messages(row, condition="direct_action"),
            ACTION_LABELS,
        )
        for row in rows
    ]
    direct_report_queries = [
        _prepare_query(
            tokenizer,
            f"{row['condition_id']}:direct_report",
            trajectory_messages(row, condition="direct_report"),
            REPORT_LABELS,
        )
        for row in rows
    ]
    batch_size = int(config["behavior"]["batch_size"])
    direct_actions = _logit_query(model, tokenizer, direct_action_queries, batch_size)
    direct_reports = _logit_query(model, tokenizer, direct_report_queries, batch_size)
    action_then_report_queries = []
    report_then_action_queries = []
    for row in rows:
        condition_id = row["condition_id"]
        action_answer = direct_actions[f"{condition_id}:direct_action"]["legal_choice"]
        report_answer = direct_reports[f"{condition_id}:direct_report"]["legal_choice"]
        action_then_report_queries.append(
            _prepare_query(
                tokenizer,
                f"{condition_id}:action_then_report",
                trajectory_messages(
                    row,
                    condition="action_then_report",
                    first_answer=action_answer,
                ),
                REPORT_LABELS,
            )
        )
        report_then_action_queries.append(
            _prepare_query(
                tokenizer,
                f"{condition_id}:report_then_action",
                trajectory_messages(
                    row,
                    condition="report_then_action",
                    first_answer=report_answer,
                ),
                ACTION_LABELS,
            )
        )
    action_then_report = _logit_query(
        model, tokenizer, action_then_report_queries, batch_size
    )
    report_then_action = _logit_query(
        model, tokenizer, report_then_action_queries, batch_size
    )
    records = []
    for row in rows:
        condition_id = row["condition_id"]
        records.append(
            {
                "condition_id": condition_id,
                "expected_action": row["expected_action"],
                "expected_report": row["expected_report"],
                "direct_action": direct_actions[f"{condition_id}:direct_action"],
                "direct_report": direct_reports[f"{condition_id}:direct_report"],
                "action_then_report": action_then_report[
                    f"{condition_id}:action_then_report"
                ],
                "report_then_action": report_then_action[
                    f"{condition_id}:report_then_action"
                ],
            }
        )
    torch = __import__("torch")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "matched_trajectory_controls_complete",
        "metadata": {
            "run_id": uuid.uuid4().hex,
            "created_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed,
            "model_key": model_key,
            "source_dataset_sha256": dataset["content_sha256"],
            "config_sha256": _config_sha256(config),
            **model_metadata,
        },
        "records": records,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    cache.commit()
    return json.dumps(payload, allow_nan=False, sort_keys=True)


def _hidden_batch(model: Any, rows: list[dict[str, Any]], pad_token_id: int) -> Any:
    import torch

    input_ids, attention = _left_padded(rows, pad_token_id)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden = output.hidden_states[-1][:, -1].float().cpu()
    del input_ids, attention, output
    return hidden


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=32768,
    volumes={"/cache": cache},
    timeout=1200,
)
def activation_discovery_remote(
    dataset: dict[str, Any],
    config: dict[str, Any],
    behavior_payload: dict[str, Any],
    model_key: str,
) -> str:
    _validate_config(config)
    if not behavior_payload.get("summary", {}).get("gate_pass", False):
        raise RuntimeError("behavioral family gates failed; activation discovery is sealed")
    spec = _model_spec(config, model_key)
    started = time.perf_counter()
    model, tokenizer, model_metadata = _load_model(spec)
    rows = [
        row
        for row in dataset["rows"]
        if row["experiment_family"] == "core_tom"
    ]
    queries = [
        _prepare_query(
            tokenizer,
            row["condition_id"],
            trajectory_messages(row, condition="direct_action"),
            ACTION_LABELS,
        )
        for row in rows
    ]
    hidden_parts = []
    batch_size = int(config["behavior"]["batch_size"])
    for start in range(0, len(queries), batch_size):
        hidden_parts.append(
            _hidden_batch(
                model,
                queries[start : start + batch_size],
                tokenizer.pad_token_id,
            )
        )
    features = __import__("torch").cat(hidden_parts).numpy()
    from sklearn.linear_model import LogisticRegression

    row_by_id = {row["condition_id"]: row for row in rows}
    ordered_rows = [row_by_id[query["query_id"]] for query in queries]
    split_indices = {
        split: [index for index, row in enumerate(ordered_rows) if row["split"] == split]
        for split in ("discovery", "validation", "locked")
    }
    probe_results = {}
    for target in config["mechanistic"]["representation_targets"]:
        labels = __import__("numpy").array([row[target] for row in ordered_rows])
        train = split_indices["discovery"]
        probe = LogisticRegression(max_iter=1000, solver="liblinear", random_state=260907)
        probe.fit(features[train], labels[train])
        probe_results[target] = {
            "discovery_accuracy": float(probe.score(features[train], labels[train])),
            "validation_accuracy": float(
                probe.score(
                    features[split_indices["validation"]],
                    labels[split_indices["validation"]],
                )
            ),
            "locked_accuracy": float(
                probe.score(features[split_indices["locked"]], labels[split_indices["locked"]])
            ),
            "method": "sklearn.LogisticRegression_on_final_hidden_state",
        }
    torch = __import__("torch")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "observational_activation_discovery_complete",
        "metadata": {
            "run_id": uuid.uuid4().hex,
            "created_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed,
            "model_key": model_key,
            "source_dataset_sha256": dataset["content_sha256"],
            "config_sha256": _config_sha256(config),
            "behavior_content_sha256": behavior_payload["content_sha256"],
            "causal_claim_status": "not_causal",
            **model_metadata,
        },
        "probe_results": probe_results,
        "status_note": (
            "Probe readability is not evidence that the model uses the state for action."
        ),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    cache.commit()
    return json.dumps(payload, allow_nan=False, sort_keys=True)


def _require_preflight(model_key: str) -> dict[str, Any]:
    path = RESULT_ROOT / "raw" / f"preflight_{model_key}.json"
    if not path.exists():
        raise RuntimeError(f"run tokenizer preflight first: {path}")
    payload = _load_json(path)
    if not _hash_valid(payload) or payload.get("status") != "tokenizer_preflight_passed":
        raise RuntimeError("tokenizer preflight artifact is invalid")
    if payload.get("model_key") != model_key:
        raise RuntimeError("tokenizer preflight artifact has the wrong model key")
    return payload


def _write_remote_result(stage: str, model_key: str, payload_text: str) -> Path:
    payload = json.loads(payload_text)
    if not _hash_valid(payload):
        raise RuntimeError(f"{stage} returned an invalid content hash")
    if payload.get("study_id") != STUDY_ID:
        raise RuntimeError(f"{stage} returned the wrong study ID")
    path = RESULT_ROOT / "raw" / f"{stage}_{model_key}.json"
    _write_new(path, payload)
    return path


def _ensure_preflight(
    config: dict[str, Any], dataset: dict[str, Any], model_key: str
) -> dict[str, Any]:
    """Materialize the model-specific CPU preflight exactly once."""
    path = RESULT_ROOT / "raw" / f"preflight_{model_key}.json"
    if path.exists():
        return _require_preflight(model_key)
    payload = json.loads(preflight_remote.remote(dataset, config, model_key))
    _write_remote_result("preflight", model_key, json.dumps(payload))
    return payload


def _ensure_behavior(
    config: dict[str, Any], dataset: dict[str, Any], model_key: str
) -> dict[str, Any]:
    """Materialize the model-specific behavior run exactly once."""
    path = RESULT_ROOT / "raw" / f"behavior_{model_key}.json"
    if path.exists():
        payload = _load_json(path)
        if (
            not _hash_valid(payload)
            or payload.get("status") != "forced_choice_logit_behavior_complete"
            or payload.get("metadata", {}).get("model_key") != model_key
        ):
            raise RuntimeError("behavior artifact is invalid")
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
    _validate_config(config)
    dataset = _dataset(config)
    payload = json.loads(preflight_remote.remote(dataset, config, model_key))
    _write_remote_result("preflight", model_key, json.dumps(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))


@app.local_entrypoint(name="behavior")
def behavior(model_key: str = "primary") -> None:
    config = _config()
    _validate_config(config)
    dataset = _dataset(config)
    payload = _ensure_behavior(config, dataset, model_key)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


@app.local_entrypoint(name="blackbox")
def blackbox(model_key: str = "primary") -> None:
    config = _config()
    _validate_config(config)
    if not config["blackbox"]["enabled"]:
        raise RuntimeError("black-box generation is disabled in the frozen config")
    dataset = _dataset(config)
    _ensure_preflight(config, dataset, model_key)
    _admit_gpu(config, "blackbox")
    payload = json.loads(blackbox_remote.remote(dataset, config, model_key))
    _write_remote_result("blackbox", model_key, json.dumps(payload))
    _record_cost(config, "blackbox", payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


@app.local_entrypoint(name="trajectory")
def trajectory(model_key: str = "primary") -> None:
    config = _config()
    _validate_config(config)
    dataset = _dataset(config)
    _ensure_preflight(config, dataset, model_key)
    _admit_gpu(config, "trajectory")
    payload = json.loads(trajectory_remote.remote(dataset, config, model_key))
    _write_remote_result("trajectory", model_key, json.dumps(payload))
    _record_cost(config, "trajectory", payload)
    print(json.dumps({"records": len(payload["records"])}, indent=2, sort_keys=True))


@app.local_entrypoint(name="activation_discovery")
def activation_discovery(model_key: str = "primary") -> None:
    config = _config()
    _validate_config(config)
    dataset = _dataset(config)
    _ensure_preflight(config, dataset, model_key)
    behavior_payload = _ensure_behavior(config, dataset, model_key)
    if not behavior_payload.get("summary", {}).get("gate_pass", False):
        raise RuntimeError(
            "family behavioral gates failed; activation discovery remains sealed"
        )
    _admit_gpu(config, "activation_discovery")
    payload = json.loads(
        activation_discovery_remote.remote(dataset, config, behavior_payload, model_key)
    )
    _write_remote_result("activation_discovery", model_key, json.dumps(payload))
    _record_cost(config, "activation_discovery", payload)
    print(json.dumps(payload["probe_results"], indent=2, sort_keys=True))


@app.local_entrypoint(name="activation_locked")
def activation_locked(model_key: str = "primary") -> None:
    config = _config()
    _validate_config(config)
    dataset = _dataset(config)
    _ensure_preflight(config, dataset, model_key)
    raise RuntimeError(
        "V6 locked causal activation stage is sealed: freeze a fresh natural "
        "counterfactual patch protocol and its random/unrelated controls first"
    )


@app.local_entrypoint(name="freeze_dataset")
def freeze_dataset() -> None:
    config = _config()
    payload = dataset_payload(config)
    verify_dataset_payload(payload, config)
    _write_new(DATASET_PATH, payload)
    print(json.dumps({"rows": len(payload["rows"]), "sha256": payload["content_sha256"]}))
