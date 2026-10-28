"""Modal entry points for H0R causal-instrument recovery."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILENAME = (
    "qwen3.6-27b/jlens/Salesforce-wikitext/"
    "Qwen3.6-27B_jacobian_lens_n1000.pt"
)
SEED = 1729

app = modal.App("jspace-h0r-cleanlow-dep")
cache = modal.Volume.from_name("jspace-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install(
        "torch>=2.8",
        "transformers>=5.5",
        "huggingface_hub>=0.34",
        f"git+https://github.com/anthropics/jacobian-lens.git@{JLENS_COMMIT}",
    )
    .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src/jspace_policy", remote_path="/root/jspace_policy")
)


@app.function(image=image, volumes={"/cache": cache}, cpu=2.0, memory=4096, timeout=900)
def tokenize_locked_controls() -> str:
    import transformers

    from jspace_policy.h0r import build_locked_controls

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    controls = build_locked_controls(tokenizer)
    controls["created_at"] = datetime.now(UTC).isoformat()
    controls["model_id"] = MODEL_ID
    controls["tokenizer_revision"] = MODEL_REVISION
    payload = json.dumps(controls, sort_keys=True, separators=(",", ":"))
    controls["content_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    cache.commit()
    return json.dumps(controls, indent=2, sort_keys=True)


@app.function(image=image, volumes={"/cache": cache}, cpu=2.0, memory=4096, timeout=900)
def tokenize_locked_argument_v2() -> str:
    import transformers

    from jspace_policy.h0r import build_locked_argument_control_v2

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    controls = build_locked_argument_control_v2(tokenizer)
    controls["created_at"] = datetime.now(UTC).isoformat()
    controls["model_id"] = MODEL_ID
    controls["tokenizer_revision"] = MODEL_REVISION
    payload = json.dumps(controls, sort_keys=True, separators=(",", ":"))
    controls["content_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    cache.commit()
    return json.dumps(controls, indent=2, sort_keys=True)


@app.local_entrypoint()
def freeze_controls() -> None:
    output = Path("configs/v2/h0r_locked_controls.json")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite locked control: {output}")
    payload = tokenize_locked_controls.remote()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    parsed = json.loads(payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "parent_commit": commit,
                "content_sha256": parsed["content_sha256"],
                "argument_trials": len(parsed["argument_control"]["trials"]),
                "intermediate_trials": len(parsed["intermediate_control"]["trials"]),
            },
            indent=2,
        )
    )


@app.local_entrypoint()
def freeze_argument_v2() -> None:
    output = Path("configs/v2/h0r_locked_controls_v2.json")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite locked control: {output}")
    payload = tokenize_locked_argument_v2.remote()
    output.write_text(payload + "\n")
    parsed = json.loads(payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "content_sha256": parsed["content_sha256"],
                "argument_trials": len(parsed["argument_control"]["trials"]),
                "status": parsed["status"],
            },
            indent=2,
        )
    )


def _precompute_directions(
    model: Any, lens: Any, layers: list[int], token_ids: list[int]
) -> dict[int, dict[int, Any]]:
    weights = model._hf_model.lm_head.weight.detach()[token_ids].float()
    result: dict[int, dict[int, Any]] = {}
    for layer in layers:
        jacobian = lens.jacobians[layer].to(weights.device)
        vectors = weights @ jacobian
        vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        result[layer] = {
            token_id: vectors[index].detach()
            for index, token_id in enumerate(token_ids)
        }
        del jacobian
    return result


def _argument_positions(tokenizer: Any, prompt: str, argument: str) -> list[int]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    for surface in (argument, f" {argument}"):
        piece = tokenizer.encode(surface, add_special_tokens=False)
        for start in range(len(prompt_ids) - len(piece) + 1):
            if prompt_ids[start : start + len(piece)] == piece:
                return list(range(start, start + len(piece)))
    raise RuntimeError(f"argument span not found: {argument!r} in {prompt!r}")


def _token_id(tokenizer: Any, value: str) -> int:
    for surface in (f" {value}", value):
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    raise RuntimeError(f"not a single token: {value}")


def _capture_layers(model: Any, input_ids: Any, layers: list[int]) -> dict[int, Any]:
    import torch

    captured: dict[int, Any] = {}
    handles = []
    for layer in layers:
        def hook(module: Any, inputs: object, output: object, layer: int = layer) -> None:
            tensor = output if torch.is_tensor(output) else output[0]
            captured[layer] = tensor.detach()[0].clone()

        handles.append(model.layers[layer].register_forward_hook(hook))
    try:
        with torch.inference_mode():
            model._hf_model(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def _output_metrics(
    baseline_log_probs: Any,
    intervened_logits: Any,
    source_answer_id: int,
    target_answer_id: int,
    candidate_ids: list[int],
) -> dict[str, object]:
    log_probs = intervened_logits.float().log_softmax(-1)
    probabilities = log_probs.exp()
    baseline_odds = baseline_log_probs[target_answer_id] - baseline_log_probs[
        source_answer_id
    ]
    intervened_odds = log_probs[target_answer_id] - log_probs[source_answer_id]
    return {
        "baseline_source_log_probability": float(
            baseline_log_probs[source_answer_id].detach().cpu()
        ),
        "baseline_target_log_probability": float(
            baseline_log_probs[target_answer_id].detach().cpu()
        ),
        "intervened_source_log_probability": float(
            log_probs[source_answer_id].detach().cpu()
        ),
        "intervened_target_log_probability": float(
            log_probs[target_answer_id].detach().cpu()
        ),
        "baseline_target_vs_source_logodds": float(baseline_odds.detach().cpu()),
        "intervened_target_vs_source_logodds": float(intervened_odds.detach().cpu()),
        "target_logodds_gain": float((intervened_odds - baseline_odds).detach().cpu()),
        "intervened_top1_token_id": int(log_probs.argmax().detach().cpu()),
        "target_top1": int(log_probs.argmax().detach().cpu()) == target_answer_id,
        "output_kl_nats": float(
            (probabilities * (log_probs - baseline_log_probs)).sum().detach().cpu()
        ),
        "output_entropy_nats": float(
            -(probabilities * log_probs).sum().detach().cpu()
        ),
        "candidate_log_probabilities": {
            str(token_id): float(log_probs[token_id].detach().cpu())
            for token_id in candidate_ids
        },
    }


def _summarize(rows: list[dict[str, object]], keys: list[str]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        values = tuple(
            tuple(row[key]) if isinstance(row[key], list) else row[key]
            for key in keys
        )
        groups[values].append(row)
    summaries: list[dict[str, object]] = []
    for values, part in groups.items():
        eligible = [row for row in part if row["baseline_correct"]]
        denominator = max(len(eligible), 1)
        summaries.append(
            {
                **dict(zip(keys, values, strict=True)),
                "n_trials": len(part),
                "n_eligible": len(eligible),
                "mean_target_logodds_gain": sum(
                    float(row["target_logodds_gain"]) for row in eligible
                )
                / denominator,
                "positive_gain_fraction": sum(
                    float(row["target_logodds_gain"]) > 0 for row in eligible
                )
                / denominator,
                "target_top1_fraction": sum(
                    bool(row["target_top1"]) for row in eligible
                )
                / denominator,
                "mean_output_kl_nats": sum(
                    float(row["output_kl_nats"]) for row in eligible
                )
                / denominator,
                "mean_delta_rms_ratio": sum(
                    float(row["mean_delta_rms_ratio"]) for row in eligible
                )
                / denominator,
                "median_condition_number": sorted(
                    float(row["median_condition_number"]) for row in eligible
                )[len(eligible) // 2]
                if eligible
                else float("nan"),
            }
        )
    return summaries


def _evaluate_country_configs(
    model: Any,
    lens: Any,
    country: dict[str, object],
    configurations: list[dict[str, object]],
) -> list[dict[str, object]]:
    import torch

    from jspace_policy.h0r_diagnostics import (
        directional_coordinate_write,
        donor_coordinate_patch,
        resolve_position_mask,
        symmetric_coordinate_swap,
    )
    from jspace_policy.interventions import MultiLayerResidualIntervention

    arguments = [str(value) for value in country["args"]]
    argument_ids = {arg: _token_id(model.tokenizer, arg) for arg in arguments}
    answer_ids = {
        str(answer): _token_id(model.tokenizer, str(answer))
        for function in country["funcs"]
        for answer in function["answers"].values()
    }
    all_layers = sorted(
        {int(layer) for config in configurations for layer in config["layers"]}
    )
    direction_ids = sorted({*argument_ids.values(), *answer_ids.values()})
    directions = _precompute_directions(model, lens, all_layers, direction_ids)
    candidate_answer_ids = sorted(set(answer_ids.values()))
    rows: list[dict[str, object]] = []

    for function in country["funcs"]:
        answers = {str(key): str(value) for key, value in function["answers"].items()}
        for source in arguments:
            prompt = str(function["template"]).format(arg=source)
            input_ids = model.encode(prompt, max_length=512)
            argument_positions = _argument_positions(model.tokenizer, prompt, source)
            source_answer_id = answer_ids[answers[source]]
            with torch.inference_mode():
                baseline_logits = model._hf_model(input_ids).logits[0, -1].float()
            baseline_log_probs = baseline_logits.log_softmax(-1)
            baseline_top1 = int(baseline_logits.argmax().detach().cpu())

            variants = [
                (target, config)
                for target in arguments
                if target != source
                for config in configurations
            ]
            needs_donor = any(
                config["operation"] == "donor_coordinate_patch"
                for _, config in variants
            )
            donor_states: dict[tuple[str, int], Any] = {}
            if needs_donor:
                for target in arguments:
                    if target == source:
                        continue
                    donor_prompt = str(function["template"]).format(arg=target)
                    donor_ids = model.encode(donor_prompt, max_length=512)
                    captured = _capture_layers(model, donor_ids, all_layers)
                    for layer, value in captured.items():
                        if value.shape[0] == input_ids.shape[1]:
                            donor_states[(target, layer)] = value

            clean_states: dict[int, Any] = {}
            if any(config.get("clean_delta") for _, config in variants):
                for layer, value in _capture_layers(model, input_ids, all_layers).items():
                    clean_states[layer] = value
            batch_ids = input_ids.repeat(len(variants), 1)
            diagnostics: dict[int, dict[int, dict[str, object]]] = defaultdict(dict)

            def make_transform(
                layer: int,
                current_variants: list[tuple[str, dict[str, object]]],
                current_argument_positions: list[int],
                current_source: str,
                current_source_answer_id: int,
                current_answers: dict[str, str],
                current_donor_states: dict[tuple[str, int], Any],
                current_diagnostics: dict[int, dict[int, dict[str, object]]],
            ):
                def transform(tensor: Any) -> Any:
                    for index, (target, config) in enumerate(current_variants):
                        if layer not in config["layers"]:
                            continue
                        positions = resolve_position_mask(
                            str(config["position_mask"]),
                            sequence_length=int(tensor.shape[1]),
                            argument_positions=current_argument_positions,
                        )
                        before = tensor[index, positions, :].clone()
                        control = str(config.get("control", "semantic"))
                        direction_target = target
                        source_id = argument_ids[current_source]
                        target_id = argument_ids[target]
                        if control == "unrelated_same_category":
                            direction_target = next(
                                item
                                for item in arguments
                                if item not in {current_source, target}
                            )
                            target_id = argument_ids[direction_target]
                        elif control == "direct_answer_direction":
                            source_id = current_source_answer_id
                            target_id = answer_ids[current_answers[target]]
                        source_vector = directions[layer][source_id]
                        target_vector = directions[layer][target_id]
                        if control == "identity_source_source":
                            after = before
                            record = {
                                "basis": {
                                    "cosine": 1.0,
                                    "absolute_cosine": 1.0,
                                    "condition_number": 1.0,
                                },
                                "coordinates": {
                                    "source_pre": [],
                                    "target_pre": [],
                                    "source_post": [],
                                    "target_post": [],
                                    "fp32_exchange_max_abs_error": 0.0,
                                    "hidden_l2": float(
                                        before.float().norm(dim=-1).mean().cpu()
                                    ),
                                    "delta_l2": 0.0,
                                    "delta_rms_ratio": 0.0,
                                },
                            }
                        else:
                            if control == "norm_matched_random_basis":
                                generator = torch.Generator(device="cpu").manual_seed(
                                    SEED + layer * 1009 + int(config.get("random_seed", index))
                                )
                                source_vector = torch.randn(
                                    source_vector.shape, generator=generator
                                ).to(source_vector.device)
                                target_vector = torch.randn(
                                    target_vector.shape, generator=generator
                                ).to(target_vector.device)
                                source_vector = source_vector / source_vector.norm()
                                target_vector = target_vector - (
                                    target_vector @ source_vector
                                ) * source_vector
                                target_vector = target_vector / target_vector.norm()
                            operation = str(config["operation"])
                            if operation == "pseudoinverse_coordinate_swap" and config.get("clean_delta"):
                                clean_before = clean_states[layer][positions, :].to(before.dtype)
                                clean_after, record = symmetric_coordinate_swap(
                                    clean_before,
                                    source_vector,
                                    target_vector,
                                    alpha=float(config["alpha"]),
                                )
                                after = before + (clean_after - clean_before)
                            elif operation == "pseudoinverse_coordinate_swap":
                                after, record = symmetric_coordinate_swap(
                                    before,
                                    source_vector,
                                    target_vector,
                                    alpha=float(config["alpha"]),
                                )
                            elif operation == "source_suppression_target_installation":
                                matrix = torch.stack(
                                    [source_vector.float(), target_vector.float()], dim=1
                                )
                                coordinate = before.float() @ torch.linalg.pinv(matrix).T
                                scale = float(
                                    coordinate.std(dim=0, unbiased=False).mean().clamp_min(0.1)
                                )
                                after, record = directional_coordinate_write(
                                    before,
                                    source_vector,
                                    target_vector,
                                    source_shift=scale,
                                    target_shift=scale,
                                    alpha=float(config["alpha"]),
                                )
                                record["natural_coordinate_scale"] = scale
                            elif operation == "donor_coordinate_patch":
                                donor = current_donor_states.get(
                                    (direction_target, layer)
                                )
                                if donor is None:
                                    raise RuntimeError(
                                        "donor/source sequence-length mismatch"
                                    )
                                after, record = donor_coordinate_patch(
                                    before,
                                    donor[positions, :],
                                    source_vector,
                                    target_vector,
                                    alpha=float(config["alpha"]),
                                )
                            else:
                                raise ValueError(f"unknown operation: {operation}")
                        tensor[index, positions, :] = after
                        current_diagnostics[index][layer] = record
                    return tensor

                return transform

            transforms = {
                layer: make_transform(
                    layer,
                    variants,
                    argument_positions,
                    source,
                    source_answer_id,
                    answers,
                    donor_states,
                    diagnostics,
                )
                for layer in all_layers
            }
            with torch.inference_mode(), MultiLayerResidualIntervention(
                model.layers, transforms
            ):
                intervened_logits = model._hf_model(batch_ids).logits[:, -1].float()

            for index, (target, config) in enumerate(variants):
                layer_records = diagnostics[index]
                coordinate_records = [
                    value["coordinates"] for value in layer_records.values()
                ]
                bases = [value["basis"] for value in layer_records.values()]
                output = _output_metrics(
                    baseline_log_probs,
                    intervened_logits[index],
                    source_answer_id,
                    answer_ids[answers[target]],
                    candidate_answer_ids,
                )
                rows.append(
                    {
                        "trial_id": uuid.uuid4().hex,
                        "scenario_id": hashlib.sha256(
                            f"countries:{function['name']}:{source}".encode()
                        ).hexdigest()[:16],
                        "category": "countries",
                        "function": function["name"],
                        "prompt": prompt,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "prompt_token_ids": [int(token) for token in input_ids[0].cpu()],
                        "argument_positions": argument_positions,
                        "source_argument": source,
                        "target_argument": target,
                        "source_answer": answers[source],
                        "source_answer_token_id": source_answer_id,
                        "target_answer": answers[target],
                        "target_answer_token_id": answer_ids[answers[target]],
                        "baseline_top1_token_id": baseline_top1,
                        "baseline_correct": baseline_top1 == source_answer_id,
                        "configuration_id": config["configuration_id"],
                        "layers": config["layers"],
                        "position_mask": config["position_mask"],
                        "operation": config["operation"],
                        "control": config.get("control", "semantic"),
                        "alpha": config["alpha"],
                        "layer_diagnostics": {
                            str(layer): value
                            for layer, value in sorted(layer_records.items())
                        },
                        "mean_delta_rms_ratio": sum(
                            float(value["delta_rms_ratio"])
                            for value in coordinate_records
                        )
                        / max(len(coordinate_records), 1),
                        "median_condition_number": sorted(
                            float(value["condition_number"]) for value in bases
                        )[len(bases) // 2],
                        "maximum_absolute_cosine": max(
                            (float(value["absolute_cosine"]) for value in bases),
                            default=1.0,
                        ),
                        **output,
                    }
                )
    return rows


def _reconstruction_runs(
    model: Any,
    lens: Any,
    country: dict[str, object],
    candidate: dict[str, object],
) -> list[dict[str, object]]:
    import torch

    from jspace_policy.h0r_diagnostics import (
        projected_coordinates,
        reconstruction_fraction,
        resolve_position_mask,
        symmetric_coordinate_swap,
    )

    arguments = [str(value) for value in country["args"]]
    argument_ids = {arg: _token_id(model.tokenizer, arg) for arg in arguments}
    write_layer = int(candidate["layers"][0])
    downstream = [
        int(layer) for layer in lens.source_layers if int(layer) >= write_layer
    ]
    directions = _precompute_directions(
        model, lens, downstream, sorted(argument_ids.values())
    )
    rows: list[dict[str, object]] = []
    for function in country["funcs"]:
        answers = {str(key): str(value) for key, value in function["answers"].items()}
        answer_ids = {
            arg: _token_id(model.tokenizer, answer) for arg, answer in answers.items()
        }
        for source in arguments:
            prompt = str(function["template"]).format(arg=source)
            input_ids = model.encode(prompt, max_length=512)
            argument_positions = _argument_positions(model.tokenizer, prompt, source)
            clean = _capture_layers(model, input_ids, downstream)
            with torch.inference_mode():
                baseline_logits = model._hf_model(input_ids).logits[0, -1].float()
            for target in arguments:
                if target == source:
                    continue
                captured: dict[int, Any] = {}
                installed: dict[str, object] = {}
                handles = []

                def make_hook(
                    layer: int,
                    current_source: str,
                    current_target: str,
                    current_argument_positions: list[int],
                    current_captured: dict[int, Any],
                    current_installed: dict[str, object],
                ):
                    def hook(module: Any, inputs: object, output: object) -> object:
                        tensor = output if torch.is_tensor(output) else output[0]
                        updated = tensor
                        if layer == write_layer:
                            positions = resolve_position_mask(
                                str(candidate["position_mask"]),
                                sequence_length=int(tensor.shape[1]),
                                argument_positions=current_argument_positions,
                            )
                            updated = tensor.clone()
                            patched, record = symmetric_coordinate_swap(
                                updated[0, positions, :],
                                directions[layer][argument_ids[current_source]],
                                directions[layer][argument_ids[current_target]],
                                alpha=float(candidate["alpha"]),
                            )
                            updated[0, positions, :] = patched
                            current_installed.update(record)
                        current_captured[layer] = updated.detach()[0].clone()
                        if torch.is_tensor(output):
                            return updated
                        return (updated, *output[1:])

                    return hook

                for layer in downstream:
                    handles.append(
                        model.layers[layer].register_forward_hook(
                            make_hook(
                                layer,
                                source,
                                target,
                                argument_positions,
                                captured,
                                installed,
                            )
                        )
                    )
                try:
                    with torch.inference_mode():
                        logits = model._hf_model(input_ids).logits[0, -1].float()
                finally:
                    for handle in handles:
                        handle.remove()
                install_coordinates = installed["coordinates"]
                installed_oriented = sum(install_coordinates["source_post"]) / len(
                    install_coordinates["source_post"]
                ) - sum(install_coordinates["target_post"]) / len(
                    install_coordinates["target_post"]
                )
                for layer in downstream:
                    matrix = torch.stack(
                        [
                            directions[layer][argument_ids[source]].float(),
                            directions[layer][argument_ids[target]].float(),
                        ],
                        dim=1,
                    )
                    clean_coordinate = projected_coordinates(clean[layer].float(), matrix)
                    current_coordinate = projected_coordinates(
                        captured[layer].float(), matrix
                    )
                    clean_oriented = float(
                        (clean_coordinate[:, 0] - clean_coordinate[:, 1]).mean().cpu()
                    )
                    current_oriented = float(
                        (current_coordinate[:, 0] - current_coordinate[:, 1]).mean().cpu()
                    )
                    rows.append(
                        {
                            "trial_id": uuid.uuid4().hex,
                            "scenario_id": hashlib.sha256(
                                f"countries:{function['name']}:{source}".encode()
                            ).hexdigest()[:16],
                            "function": function["name"],
                            "source_argument": source,
                            "target_argument": target,
                            "write_layer": write_layer,
                            "measurement_layer": layer,
                            "clean_oriented_coordinate": clean_oriented,
                            "installed_oriented_coordinate": installed_oriented,
                            "current_oriented_coordinate": current_oriented,
                            "reconstruction_fraction": reconstruction_fraction(
                                current_oriented,
                                clean_oriented,
                                installed_oriented,
                            ),
                            "baseline_correct": int(
                                baseline_logits.argmax().cpu()
                            )
                            == answer_ids[source],
                            "target_top1": int(logits.argmax().cpu())
                            == answer_ids[target],
                        }
                    )
    return rows


def _load_model() -> tuple[Any, Any, dict[str, object]]:
    import torch
    import transformers
    from huggingface_hub import model_info
    from jlens import JacobianLens, from_hf

    torch.manual_seed(SEED)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda()
    hf_model.eval()
    model = from_hf(hf_model, tokenizer)
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
        "lens_repo": LENS_REPO,
        "lens_filename": LENS_FILENAME,
        "lens_revision": LENS_REVISION,
        "lens_code_commit": JLENS_COMMIT,
        "lens_source_layers": [int(layer) for layer in lens.source_layers],
        "seed": SEED,
        "dtype": "bfloat16",
        "gpu_requested": "A100-80GB",
        "gpu_actual": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
    }
    return model, lens, metadata


@app.function(
    image=image,
    volumes={"/cache": cache},
    cpu=4.0,
    memory=32768,
    gpu="A100-80GB",
    timeout=900,
    max_containers=1,
    retries=0,
)
def run_diagnostic_remote(
    phase: str,
    country: dict[str, object],
    diagnostic: dict[str, object],
    context: dict[str, object],
    git_commit: str,
    dirty_tree: bool,
) -> str:
    import torch

    started = time.perf_counter()
    model, lens, metadata = _load_model()
    artifacts: dict[str, list[dict[str, object]]] = {}
    if phase == "layer":
        configurations = [
            {
                "configuration_id": f"layer_{layer}",
                "layers": [layer],
                "position_mask": diagnostic["layer_sweep"]["position_mask"],
                "operation": diagnostic["layer_sweep"]["operation"],
                "alpha": diagnostic["layer_sweep"]["alpha"],
            }
            for layer in diagnostic["layer_sweep"]["layers"]
        ]
        artifacts["layer_sweep"] = _evaluate_country_configs(
            model, lens, country, configurations
        )
    elif phase == "topology":
        top_layers = [int(layer) for layer in context["top_layers"]]
        position_configs = [
            {
                "configuration_id": f"layer_{layer}__{mask}",
                "layers": [layer],
                "position_mask": mask,
                "operation": "pseudoinverse_coordinate_swap",
                "alpha": 1.0,
            }
            for layer in top_layers
            for mask in diagnostic["position_masks"]
        ]
        position_rows = _evaluate_country_configs(
            model, lens, country, position_configs
        )
        position_summary = _summarize(position_rows, ["configuration_id"])
        winners = sorted(
            position_summary,
            key=lambda row: (
                float(row["mean_target_logodds_gain"]),
                float(row["positive_gain_fraction"]),
            ),
            reverse=True,
        )[: int(diagnostic["top_layer_count"])]
        by_id = {config["configuration_id"]: config for config in position_configs}
        strength_configs = [
            {
                **by_id[str(winner["configuration_id"])],
                "configuration_id": (
                    f"{winner['configuration_id']}__alpha_{float(alpha):g}"
                ),
                "alpha": float(alpha),
            }
            for winner in winners
            for alpha in diagnostic["strengths"]
        ]
        artifacts["position_sweep"] = position_rows
        artifacts["strength_sweep"] = _evaluate_country_configs(
            model, lens, country, strength_configs
        )
    elif phase == "cancellation":
        candidate = context["exploratory_candidate"]
        cumulative_configs = [
            {
                "configuration_id": "layers_" + "_".join(map(str, layers)),
                "layers": layers,
                "position_mask": candidate["position_mask"],
                "operation": "pseudoinverse_coordinate_swap",
                "alpha": candidate["alpha"],
            }
            for layers in diagnostic["cumulative_layer_sets"]
        ]
        operation_configs = [
            {
                "configuration_id": f"operation_{operation}",
                "layers": candidate["layers"],
                "position_mask": candidate["position_mask"],
                "operation": operation,
                "alpha": candidate["alpha"],
            }
            for operation in diagnostic["operations"]
        ]
        control_configs = [
            {
                "configuration_id": f"control_{control}",
                "layers": [20]
                if control == "out_of_window_layer"
                else candidate["layers"],
                "position_mask": candidate["position_mask"],
                "operation": "pseudoinverse_coordinate_swap",
                "control": control,
                "alpha": candidate["alpha"],
            }
            for control in diagnostic["controls"]
        ]
        artifacts["cumulative_layers"] = _evaluate_country_configs(
            model, lens, country, cumulative_configs
        )
        operation_rows = _evaluate_country_configs(
            model, lens, country, operation_configs
        )
        artifacts["coordinate_trajectories"] = operation_rows
        artifacts["controls"] = _evaluate_country_configs(
            model, lens, country, control_configs
        )
        artifacts["reconstruction"] = _reconstruction_runs(
            model, lens, country, candidate
        )
    else:
        raise ValueError(f"unknown diagnostic phase: {phase}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    cache.commit()
    result = {
        "metadata": {
            "schema_version": 1,
            "run_id": uuid.uuid4().hex,
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": git_commit,
            "dirty_tree": dirty_tree,
            "phase": phase,
            "diagnostic_config_sha256": hashlib.sha256(
                json.dumps(diagnostic, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "elapsed_seconds": elapsed,
            **metadata,
        },
        "artifacts": artifacts,
    }
    return json.dumps(result, allow_nan=False)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _top_layer_context(path: Path, count: int) -> dict[str, object]:
    rows = _read_jsonl(path)
    summary = _summarize(rows, ["configuration_id", "layers"])
    ordered = sorted(
        summary,
        key=lambda row: (
            float(row["mean_target_logodds_gain"]),
            float(row["positive_gain_fraction"]),
            float(row["target_top1_fraction"]),
        ),
        reverse=True,
    )
    return {"top_layers": [int(row["layers"][0]) for row in ordered[:count]]}


def _candidate_context(path: Path) -> dict[str, object]:
    rows = _read_jsonl(path)
    summary = _summarize(
        rows,
        ["configuration_id", "layers", "position_mask", "operation", "alpha"],
    )
    valid = [
        row
        for row in summary
        if float(row["median_condition_number"]) <= 20
        and float(row["mean_output_kl_nats"]) <= 1
        and float(row["mean_delta_rms_ratio"]) <= 0.25
    ]
    if not valid:
        raise RuntimeError("no strength-sweep configuration is inside validity limits")
    winner = max(
        valid,
        key=lambda row: (
            float(row["mean_target_logodds_gain"]),
            float(row["positive_gain_fraction"]),
            float(row["target_top1_fraction"]),
        ),
    )
    return {
        "exploratory_candidate": {
            "layers": winner["layers"],
            "position_mask": winner["position_mask"],
            "operation": winner["operation"],
            "alpha": winner["alpha"],
        }
    }


def _content_hash_matches(payload: dict[str, object]) -> bool:
    expected = str(payload["content_sha256"])
    without_hash = {key: value for key, value in payload.items() if key != "content_sha256"}
    actual = hashlib.sha256(
        json.dumps(without_hash, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return expected == actual


def _computed_argument_positions(trial: dict[str, object]) -> list[int]:
    prompt_ids = [int(value) for value in trial["prompt_token_ids"]]
    # The rendered arithmetic subsequence is immutable in the locked prompt. Its
    # token IDs are recovered by slicing from the final occurrence of the first
    # operand through the second operand, excluding the terminal arrow token.
    a = int(trial["a"])
    b = int(trial["b"])
    prompt = str(trial["prompt"])
    rendered = f"{a} + {b}"
    character_start = prompt.rfind(rendered)
    if character_start < 0:
        raise RuntimeError(f"locked arithmetic span absent: {rendered}")
    # Qwen's tokenizer offsets are resolved remotely; this marker is replaced by
    # exact subsequence matching in `_evaluate_locked_trials`.
    return prompt_ids


def _baseline_locked_trials(
    model: Any, trials: list[dict[str, object]], control_type: str
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    import torch

    source_answer_key = "source_answer_token_id"
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for trial in trials:
        grouped[str(trial["scenario_id"])].append(trial)
    rows: list[dict[str, object]] = []
    logits_by_scenario: dict[str, Any] = {}
    for scenario_id, part in grouped.items():
        first = part[0]
        input_ids = torch.tensor(
            [[int(token) for token in first["prompt_token_ids"]]],
            device="cuda",
            dtype=torch.long,
        )
        with torch.inference_mode():
            logits = model._hf_model(input_ids).logits[0, -1].float()
        logits_by_scenario[scenario_id] = logits
        top1 = int(logits.argmax().detach().cpu())
        for trial in part:
            family = (
                str(trial["category"])
                if control_type == "argument"
                else str(trial["family"])
            )
            rows.append(
                {
                    "record_type": "baseline",
                    "trial_id": trial["trial_id"],
                    "scenario_id": scenario_id,
                    "family": family,
                    "function": trial.get("function", trial.get("family")),
                    "prompt_sha256": trial["prompt_sha256"],
                    "source_answer_token_id": trial[source_answer_key],
                    "target_answer_token_id": trial["target_answer_token_id"],
                    "baseline_top1_token_id": top1,
                    "baseline_correct": top1 == int(trial[source_answer_key]),
                }
            )
    return rows, logits_by_scenario


def _find_locked_span(tokenizer: Any, trial: dict[str, object]) -> list[int]:
    prompt_ids = [int(token) for token in trial["prompt_token_ids"]]
    if "argument_positions" in trial:
        return [int(position) for position in trial["argument_positions"]]
    rendered = f"{trial['a']} + {trial['b']}"
    for surface in (rendered, f" {rendered}"):
        piece = tokenizer.encode(surface, add_special_tokens=False)
        for start in range(len(prompt_ids) - len(piece) + 1):
            if prompt_ids[start : start + len(piece)] == piece:
                return list(range(start, start + len(piece)))
    raise RuntimeError(f"computed argument span not found for {trial['trial_id']}")


def _evaluate_locked_trials(
    model: Any,
    lens: Any,
    trials: list[dict[str, object]],
    protocol: dict[str, object],
    validation: dict[str, object],
    control_type: str,
    baseline_rows: list[dict[str, object]],
    baseline_logits_by_scenario: dict[str, Any],
) -> list[dict[str, object]]:
    import torch

    from jspace_policy.h0r_diagnostics import (
        resolve_position_mask,
        symmetric_coordinate_swap,
    )
    from jspace_policy.interventions import MultiLayerResidualIntervention

    layers = [int(layer) for layer in protocol["layers"]]
    state_prefix = "argument" if control_type == "argument" else "intermediate"
    source_state_key = f"source_{state_prefix}_token_id"
    target_state_key = f"target_{state_prefix}_token_id"
    all_token_ids = {
        int(trial[key])
        for trial in trials
        for key in (
            source_state_key,
            target_state_key,
            "source_answer_token_id",
            "target_answer_token_id",
        )
    }
    unrelated_ids: tuple[int, int] | None = None
    if control_type == "intermediate":
        pair = validation["intermediate_unrelated_pair"]
        unrelated_ids = (
            _token_id(model.tokenizer, str(pair["source"])),
            _token_id(model.tokenizer, str(pair["target"])),
        )
        all_token_ids.update(unrelated_ids)
    directions = _precompute_directions(model, lens, layers, sorted(all_token_ids))
    baseline_by_trial = {str(row["trial_id"]): row for row in baseline_rows}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for trial in trials:
        grouped[str(trial["scenario_id"])].append(trial)
    control_names = validation[f"{control_type}_controls"]
    rows: list[dict[str, object]] = []
    for scenario_id, scenario_trials in grouped.items():
        first = scenario_trials[0]
        input_ids = torch.tensor(
            [[int(token) for token in first["prompt_token_ids"]]],
            device="cuda",
            dtype=torch.long,
        )
        argument_positions = _find_locked_span(model.tokenizer, first)
        variants = [
            (trial, str(control))
            for trial in scenario_trials
            for control in control_names
        ]
        batch_ids = input_ids.repeat(len(variants), 1)
        diagnostics: dict[int, dict[int, dict[str, object]]] = defaultdict(dict)

        def make_transform(
            layer: int,
            current_variants: list[tuple[dict[str, object], str]],
            current_positions: list[int],
            current_diagnostics: dict[int, dict[int, dict[str, object]]],
        ):
            def transform(tensor: Any) -> Any:
                for index, (trial, control) in enumerate(current_variants):
                    positions = resolve_position_mask(
                        str(protocol["position_mask"]),
                        sequence_length=int(tensor.shape[1]),
                        argument_positions=current_positions,
                    )
                    source_id = int(trial[source_state_key])
                    target_id = int(trial[target_state_key])
                    if control == "direct_answer_direction":
                        source_id = int(trial["source_answer_token_id"])
                        target_id = int(trial["target_answer_token_id"])
                    elif control == "unrelated_fixed_pair":
                        if unrelated_ids is None:
                            raise RuntimeError("missing fixed unrelated token pair")
                        source_id, target_id = unrelated_ids
                    elif control == "unrelated_same_category":
                        candidates = sorted(
                            {
                                int(item[target_state_key])
                                for item in trials
                                if item["category"] == trial["category"]
                                and int(item[target_state_key])
                                not in {
                                    int(trial[source_state_key]),
                                    int(trial[target_state_key]),
                                }
                            }
                        )
                        if not candidates:
                            raise RuntimeError("no unrelated within-category state")
                        target_id = candidates[0]
                    source_vector = directions[layer][source_id]
                    target_vector = directions[layer][target_id]
                    if control == "norm_matched_random_basis":
                        generator = torch.Generator(device="cpu").manual_seed(
                            SEED + layer * 1009 + int(config.get("random_seed", index))
                        )
                        source_vector = torch.randn(
                            source_vector.shape, generator=generator
                        ).to(source_vector.device)
                        target_vector = torch.randn(
                            target_vector.shape, generator=generator
                        ).to(target_vector.device)
                        source_vector = source_vector / source_vector.norm()
                        target_vector = target_vector - (
                            target_vector @ source_vector
                        ) * source_vector
                        target_vector = target_vector / target_vector.norm()
                    before = tensor[index, positions, :].clone()
                    after, record = symmetric_coordinate_swap(
                        before,
                        source_vector,
                        target_vector,
                        alpha=float(protocol["alpha"]),
                    )
                    tensor[index, positions, :] = after
                    current_diagnostics[index][layer] = record
                return tensor

            return transform

        transforms = {
            layer: make_transform(layer, variants, argument_positions, diagnostics)
            for layer in layers
        }
        with torch.inference_mode(), MultiLayerResidualIntervention(
            model.layers, transforms
        ):
            intervened_logits = model._hf_model(batch_ids).logits[:, -1].float()
        baseline_log_probs = baseline_logits_by_scenario[scenario_id].log_softmax(-1)
        for index, (trial, control) in enumerate(variants):
            records = diagnostics[index]
            coordinates = [record["coordinates"] for record in records.values()]
            bases = [record["basis"] for record in records.values()]
            output = _output_metrics(
                baseline_log_probs,
                intervened_logits[index],
                int(trial["source_answer_token_id"]),
                int(trial["target_answer_token_id"]),
                [
                    int(trial["source_answer_token_id"]),
                    int(trial["target_answer_token_id"]),
                ],
            )
            baseline = baseline_by_trial[str(trial["trial_id"])]
            rows.append(
                {
                    "record_type": "intervention",
                    "trial_id": trial["trial_id"],
                    "scenario_id": scenario_id,
                    "family": baseline["family"],
                    "function": baseline["function"],
                    "prompt_sha256": trial["prompt_sha256"],
                    "control": control,
                    "baseline_correct": baseline["baseline_correct"],
                    "layers": layers,
                    "position_mask": protocol["position_mask"],
                    "operation": protocol["operation"],
                    "alpha": protocol["alpha"],
                    "source_state_token_id": trial[source_state_key],
                    "target_state_token_id": trial[target_state_key],
                    "source_answer_token_id": trial["source_answer_token_id"],
                    "target_answer_token_id": trial["target_answer_token_id"],
                    "layer_diagnostics": {
                        str(layer): record for layer, record in sorted(records.items())
                    },
                    "mean_delta_rms_ratio": sum(
                        float(record["delta_rms_ratio"]) for record in coordinates
                    )
                    / len(coordinates),
                    "median_condition_number": sorted(
                        float(record["condition_number"]) for record in bases
                    )[len(bases) // 2],
                    "high_cosine_fraction": sum(
                        float(record["absolute_cosine"]) > 0.95 for record in bases
                    )
                    / len(bases),
                    **output,
                }
            )
    return rows


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
def run_validation_remote(
    control_type: str,
    locked: dict[str, object],
    protocol: dict[str, object],
    validation: dict[str, object],
    git_commit: str,
) -> str:
    import torch

    if not _content_hash_matches(locked) or not _content_hash_matches(protocol):
        raise RuntimeError("locked control or candidate protocol hash mismatch")
    started = time.perf_counter()
    model, lens, metadata = _load_model()
    key = "argument_control" if control_type == "argument" else "intermediate_control"
    trials = locked[key]["trials"]
    baseline_rows, baseline_logits = _baseline_locked_trials(
        model, trials, control_type
    )
    baseline_accuracy = sum(row["baseline_correct"] for row in baseline_rows) / len(
        baseline_rows
    )
    baseline_gate_pass = (
        baseline_accuracy
        >= float(validation["argument_baseline_gate"]["minimum_overall_accuracy"])
        if control_type == "argument"
        else True
    )
    intervention_rows: list[dict[str, object]] = []
    if baseline_gate_pass:
        intervention_rows = _evaluate_locked_trials(
            model,
            lens,
            trials,
            protocol,
            validation,
            control_type,
            baseline_rows,
            baseline_logits,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    cache.commit()
    return json.dumps(
        {
            "metadata": {
                "schema_version": 1,
                "run_id": uuid.uuid4().hex,
                "created_at": datetime.now(UTC).isoformat(),
                "git_commit": git_commit,
                "dirty_tree": False,
                "control_type": control_type,
                "locked_control_sha256": locked["content_sha256"],
                "candidate_protocol_sha256": protocol["content_sha256"],
                "validation_config_sha256": hashlib.sha256(
                    json.dumps(validation, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "baseline_accuracy": baseline_accuracy,
                "baseline_gate_pass": baseline_gate_pass,
                "intervention_opened": bool(intervention_rows),
                "elapsed_seconds": elapsed,
                **metadata,
            },
            "baseline_rows": baseline_rows,
            "intervention_rows": intervention_rows,
        },
        allow_nan=False,
    )


def _bootstrap_validation(rows: list[dict[str, object]], draws: int) -> list[float]:
    import random

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scenario_id"])].append(row)
    scenarios = sorted(grouped)
    rng = random.Random(SEED)
    values = []
    for _ in range(draws):
        picked = [scenarios[rng.randrange(len(scenarios))] for _ in scenarios]
        sample = [row for scenario in picked for row in grouped[scenario]]
        values.append(
            sum(float(row["target_logodds_gain"]) for row in sample) / len(sample)
        )
    return sorted(values)


def _validation_summary(
    result: dict[str, object],
    protocol: dict[str, object],
    validation: dict[str, object],
) -> dict[str, object]:
    baseline = result["baseline_rows"]
    metadata = result["metadata"]
    family_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in baseline:
        family_rows[str(row["family"])].append(row)
    summary: dict[str, object] = {
        "run_id": metadata["run_id"],
        "control_type": metadata["control_type"],
        "baseline_accuracy": metadata["baseline_accuracy"],
        "baseline_accuracy_by_family": {
            family: sum(row["baseline_correct"] for row in rows) / len(rows)
            for family, rows in sorted(family_rows.items())
        },
        "baseline_gate_pass": metadata["baseline_gate_pass"],
        "intervention_opened": metadata["intervention_opened"],
    }
    if not metadata["intervention_opened"]:
        summary["gate_pass"] = False
        summary["failure_reason"] = "argument baseline accuracy below 0.80"
        return summary
    eligible = [
        row for row in result["intervention_rows"] if row["baseline_correct"]
    ]
    by_control: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in eligible:
        by_control[str(row["control"])].append(row)
    control_summary = {}
    for control, rows in sorted(by_control.items()):
        control_summary[control] = {
            "n_eligible": len(rows),
            "mean_target_logodds_gain": sum(
                float(row["target_logodds_gain"]) for row in rows
            )
            / len(rows),
            "positive_gain_fraction": sum(
                float(row["target_logodds_gain"]) > 0 for row in rows
            )
            / len(rows),
            "target_top1_fraction": sum(bool(row["target_top1"]) for row in rows)
            / len(rows),
            "mean_output_kl_nats": sum(float(row["output_kl_nats"]) for row in rows)
            / len(rows),
            "mean_delta_rms_ratio": sum(
                float(row["mean_delta_rms_ratio"]) for row in rows
            )
            / len(rows),
            "median_condition_number": sorted(
                float(row["median_condition_number"]) for row in rows
            )[len(rows) // 2],
            "high_cosine_fraction": sum(
                float(row["high_cosine_fraction"]) > 0.05 for row in rows
            )
            / len(rows),
        }
    semantic = by_control["semantic"]
    boot = _bootstrap_validation(semantic, int(validation["bootstrap_draws"]))
    low = boot[int(0.025 * len(boot))]
    high = boot[int(0.975 * len(boot)) - 1]
    semantic_summary = control_summary["semantic"]
    semantic_summary["gain_ci_95"] = [low, high]
    validity = protocol["validity_domain"]
    validity_pass = (
        semantic_summary["median_condition_number"]
        <= validity["maximum_median_condition_number"]
        and semantic_summary["high_cosine_fraction"]
        <= validity["maximum_high_cosine_fraction"]
        and semantic_summary["mean_output_kl_nats"]
        <= validity["maximum_mean_output_kl_nats"]
        and semantic_summary["mean_delta_rms_ratio"]
        <= validity["maximum_mean_delta_rms_ratio"]
    )
    if metadata["control_type"] == "argument":
        thresholds = protocol["prospective_thresholds"]["argument_control"]
        control_gain = max(
            control_summary["norm_matched_random_basis"]["mean_target_logodds_gain"],
            control_summary["unrelated_same_category"]["mean_target_logodds_gain"],
        )
        contrast = semantic_summary["mean_target_logodds_gain"] - control_gain
        gate_pass = (
            low > 0
            and semantic_summary["mean_target_logodds_gain"]
            >= thresholds["minimum_mean_gain"]
            and semantic_summary["positive_gain_fraction"]
            >= thresholds["minimum_positive_gain_fraction"]
            and semantic_summary["target_top1_fraction"]
            >= thresholds["minimum_target_top1_fraction"]
            and contrast >= thresholds["minimum_semantic_minus_control_gain"]
            and validity_pass
        )
        summary["semantic_minus_max_random_unrelated_gain"] = contrast
    else:
        thresholds = protocol["prospective_thresholds"]["intermediate_control"]
        random_gain = control_summary["norm_matched_random_basis"][
            "mean_target_logodds_gain"
        ]
        unrelated_gain = control_summary["unrelated_fixed_pair"][
            "mean_target_logodds_gain"
        ]
        direct_gain = control_summary["direct_answer_direction"][
            "mean_target_logodds_gain"
        ]
        gate_pass = (
            low > 0
            and semantic_summary["mean_target_logodds_gain"]
            >= thresholds["minimum_mean_gain"]
            and semantic_summary["positive_gain_fraction"]
            >= thresholds["minimum_positive_gain_fraction"]
            and semantic_summary["mean_target_logodds_gain"] > random_gain
            and semantic_summary["mean_target_logodds_gain"] > unrelated_gain
            and semantic_summary["mean_target_logodds_gain"] > direct_gain
            and validity_pass
        )
    summary["controls"] = control_summary
    summary["validity_pass"] = validity_pass
    summary["gate_pass"] = gate_pass
    return summary


@app.local_entrypoint()
def validate(control_type: str = "argument") -> None:
    """Run a frozen H0R-C or H0R-D prospective validation exactly once."""
    from jspace_policy.budget import admit_run, append_ledger, estimate_cost

    if control_type not in {"argument", "intermediate"}:
        raise ValueError("control_type must be argument or intermediate")
    locked_path = Path(
        "configs/v2/h0r_locked_controls_v2.json"
        if control_type == "argument"
        else "configs/v2/h0r_locked_controls.json"
    )
    locked = json.loads(locked_path.read_text())
    protocol = json.loads(Path("configs/v2/h0r_candidate_protocol.json").read_text())
    validation = json.loads(Path("configs/v2/h0r_validation.json").read_text())
    if control_type == "intermediate":
        argument_summary = Path("results/v2_h0r_argument_validation/summary.json")
        if not argument_summary.exists() or not json.loads(
            argument_summary.read_text()
        )["gate_pass"]:
            raise RuntimeError("H0R-D is forbidden unless H0R-C passed")
    root = Path(
        "results/v2_h0r_argument_validation_v2"
        if control_type == "argument"
        else "results/v2_h0r_intermediate_validation"
    )
    if root.exists():
        raise RuntimeError(f"refusing to overwrite prospective result: {root}")
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    relevant_status = [
        line
        for line in subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).splitlines()
        if ".prismor/" not in line
    ]
    if relevant_status:
        raise RuntimeError(f"prospective validation requires a clean tree: {relevant_status}")
    estimate = estimate_cost("A100-80GB", 900, memory_gib=32.0)
    ledger = Path("artifacts/raw/cost_ledger.jsonl")
    admit_run(ledger, estimate)
    result = json.loads(
        run_validation_remote.remote(
            control_type, locked, protocol, validation, git_commit
        )
    )
    root.joinpath("raw").mkdir(parents=True)
    metadata = result["metadata"]
    with root.joinpath("raw/baseline.jsonl").open("w") as handle:
        for row in result["baseline_rows"]:
            handle.write(json.dumps(metadata | row, sort_keys=True) + "\n")
    if result["intervention_rows"]:
        with root.joinpath("raw/interventions.jsonl").open("w") as handle:
            for row in result["intervention_rows"]:
                handle.write(json.dumps(metadata | row, sort_keys=True) + "\n")
    summary = _validation_summary(result, protocol, validation)
    root.joinpath("summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    root.joinpath("run_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    measured = estimate_cost(
        "A100-80GB", float(metadata["elapsed_seconds"]), memory_gib=32.0
    )
    append_ledger(
        ledger,
        measured,
        run_id=str(metadata["run_id"]),
        stage=f"v2-h0r-{control_type}-validation",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


@app.local_entrypoint()
def diagnostic(phase: str = "layer") -> None:
    """Run one preregistered H0R-B diagnostic GPU phase and download raw rows."""
    from jspace_policy.budget import admit_run, append_ledger, estimate_cost

    if phase not in {"layer", "topology", "cancellation"}:
        raise ValueError("phase must be layer, topology, or cancellation")
    diagnostic_config = json.loads(Path("configs/v2/h0r_diagnostic.json").read_text())
    flexible = json.loads(
        Path("configs/v2/flexible_generalization_smoke.json").read_text()
    )
    country = next(
        item for item in flexible["data"]["categories"] if item["name"] == "countries"
    )
    raw = Path("results/v2_h0r_diagnostic/raw")
    context: dict[str, object] = {}
    if phase == "topology":
        context = _top_layer_context(
            raw / "layer_sweep.jsonl", int(diagnostic_config["top_layer_count"])
        )
    elif phase == "cancellation":
        context = _candidate_context(raw / "strength_sweep.jsonl")
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(["git", "status", "--short"], text=True)
    generated_prefixes = (
        "?? .prismor/",
        "?? results/v2_h0r_diagnostic/",
        " M artifacts/raw/cost_ledger.jsonl",
    )
    dirty_tree = bool(
        "\n".join(
            line
            for line in status.splitlines()
            if not line.startswith(generated_prefixes)
        )
    )
    estimate = estimate_cost("A100-80GB", 3600, memory_gib=32.0)
    ledger = Path("artifacts/raw/cost_ledger.jsonl")
    admit_run(ledger, estimate)
    result = json.loads(
        run_diagnostic_remote.remote(
            phase,
            country,
            diagnostic_config,
            context,
            git_commit,
            dirty_tree,
        )
    )
    raw.mkdir(parents=True, exist_ok=True)
    metadata = result["metadata"]
    for name, rows in result["artifacts"].items():
        with (raw / f"{name}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(metadata | row, sort_keys=True) + "\n")
    manifest = Path("results/v2_h0r_diagnostic/run_manifest.json")
    existing = json.loads(manifest.read_text()) if manifest.exists() else {"runs": []}
    existing["runs"].append(metadata)
    manifest.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    measured = estimate_cost(
        "A100-80GB", float(metadata["elapsed_seconds"]), memory_gib=32.0
    )
    append_ledger(
        ledger,
        measured,
        run_id=str(metadata["run_id"]),
        stage=f"v2-h0r-diagnostic-{phase}",
    )
    print(
        json.dumps(
            {
                "phase": phase,
                "run_id": metadata["run_id"],
                "elapsed_seconds": metadata["elapsed_seconds"],
                "recorded_cost_usd": measured.buffered_usd,
                "rows": {
                    name: len(rows) for name, rows in result["artifacts"].items()
                },
                "context": context,
            },
            indent=2,
        )
    )


BAND_ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.5]
BAND_LAYER_SETS = [[32], [34], [40], [32, 33, 34], [36, 37, 38, 39, 40, 41, 42]]


def _klmatch_configs() -> list[dict[str, object]]:
    base = {"position_mask": "argument_through_end", "operation": "pseudoinverse_coordinate_swap"}
    out: list[dict[str, object]] = []
    for layers in BAND_LAYER_SETS:
        tag = "L" + "_".join(map(str, layers))
        for a in BAND_ALPHAS:
            out.append({**base, "configuration_id": f"sem_{tag}_a{a:g}", "layers": layers, "alpha": a, "control": "semantic"})
    for layers in ([40], [36, 37, 38, 39, 40, 41, 42]):
        tag = "L" + "_".join(map(str, layers))
        for a in BAND_ALPHAS:
            out.append({**base, "configuration_id": f"unrel_{tag}_a{a:g}", "layers": layers, "alpha": a, "control": "unrelated_same_category"})
    return out


def _clean_configs() -> list[dict[str, object]]:
    out = []
    op = "pseudoinverse_coordinate_swap"
    for tag, layers in (("B8_36_43", list(range(36, 44))), ("B7_36_42", list(range(36, 43)))):
        for a in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4):
            out.append({"configuration_id": f"clean_{tag}_a{a:g}", "layers": layers, "position_mask": "argument_through_end", "operation": op, "alpha": a, "control": "semantic", "clean_delta": True})
    for a in (0.1, 0.2, 0.3):
        out.append({"configuration_id": f"cleanunrel_B8_36_43_a{a:g}", "layers": list(range(36, 44)), "position_mask": "argument_through_end", "operation": op, "alpha": a, "control": "unrelated_same_category", "clean_delta": True})
    return out


@app.function(
    image=image,
    volumes={"/cache": cache},
    cpu=4.0,
    memory=32768,
    gpu="A100-80GB",
    timeout=1200,
    max_containers=1,
    retries=0,
)
def run_clean_remote(country: dict[str, object], git_commit: str) -> str:
    import torch

    started = time.perf_counter()
    model, lens, metadata = _load_model()
    rows = []
    cfgs = _clean_configs()
    for i in range(0, len(cfgs), 2):
        rows += _evaluate_country_configs(model, lens, country, cfgs[i : i + 2])
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {"metadata": {"run_id": uuid.uuid4().hex, "created_at": datetime.now(UTC).isoformat(), "git_commit": git_commit, "elapsed_seconds": elapsed, **metadata}, "rows": rows}
    return json.dumps(result, allow_nan=False)


@app.local_entrypoint()
def klmatch() -> None:
    from jspace_policy.budget import admit_run, append_ledger, estimate_cost

    flexible = json.loads(Path("configs/v2/flexible_generalization_smoke.json").read_text())
    country = next(i for i in flexible["data"]["categories"] if i["name"] == "countries")
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    ledger = Path("results/v2_jlens_band/cost_ledger.jsonl")
    admit_run(ledger, estimate_cost("A100-80GB", 900, memory_gib=32.0), study_limit_usd=2.0)
    result = json.loads(run_klmatch_remote.remote(country, git_commit))
    out = Path("results/v2_jlens_band/raw")
    out.mkdir(parents=True, exist_ok=True)
    md = result["metadata"]
    with (out / "klmatch.jsonl").open("w") as h:
        for r in result["rows"]:
            h.write(json.dumps(md | r, sort_keys=True) + "\n")
    measured = estimate_cost("A100-80GB", float(md["elapsed_seconds"]), memory_gib=32.0)
    append_ledger(ledger, measured, run_id=str(md["run_id"]), stage="v2-jlens-klmatch")
    print(json.dumps({"run_id": md["run_id"], "elapsed_seconds": md["elapsed_seconds"], "recorded_cost_usd": measured.buffered_usd, "rows": len(result["rows"])}))


@app.function(image=image, timeout=60)
def _noop() -> int:
    return 1
