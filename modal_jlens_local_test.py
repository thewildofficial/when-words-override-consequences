"""Decisive test: prompt-local Jacobians vs the averaged J-lens at one write site.

Context: the V2 H0 smoke test reached 24/33 target top-1 conversions on the
country-facts category with the *average* J-lens directions at the
single_layer_all_positions topology (layer 40, alpha=1.0), while the 8-layer
band topology got 0/90. The external review (docs/reviews/jlens-failure-diagnosis-2026-10-23.md)
proposes the smallest decisive test: rerun the identical swap with per-prompt
Jacobians (upstream jlens jacobian_for_prompt) at the same single site. A big
conversion jump confirms estimator mismatch (average-J is the bottleneck);
no jump indicates a deeper subspace-write limitation.

Design: matched internal baseline. Every eligible trial is run twice in the
same process, once with average-lens directions (the H0 condition, replicating
24/33) and once with prompt-local directions. Same prompts, same tokenizer
checks, same swap code path (_apply_basis_swap semantics), same alpha=1.0,
same layer 40, all positions.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
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
WRITE_LAYER = 40  # H0 center layer of the frozen band [36..43]
ALPHA = 1.0

COUNTRIES = {
    "name": "countries",
    "args": ["France", "Canada", "China", "Egypt"],
    "funcs": [
        {"name": "capital", "template": "The capital of {arg} is the city of",
         "answers": {"France": "Paris", "Canada": "Ottawa", "China": "Beijing", "Egypt": "Cairo"}},
        {"name": "language", "template": "Most people in {arg} speak",
         "answers": {"France": "French", "Canada": "English", "China": "Chinese", "Egypt": "Arabic"}},
        {"name": "continent", "template": "{arg} is a country on the continent of",
         "answers": {"France": "Europe", "Canada": "North", "China": "Asia", "Egypt": "Africa"}},
        {"name": "currency", "template": "The currency used in {arg} is the",
         "answers": {"France": "Euro", "Canada": "Dollar", "China": "Yuan", "Egypt": "Pound"}},
    ],
}

app = modal.App("jspace-jlens-local-test")
cache = modal.Volume.from_name("jspace-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install(
        "torch>=2.8",
        "transformers>=5.5",
        "huggingface_hub>=0.34",
        f"git+https://github.com/anthropics/jacobian-lens.git@{JLENS_COMMIT}",
        "hf_transfer>=0.1",
    )
    .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false",
          "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def _one_token_id(tokenizer: Any, word: str) -> int:
    for variant in (f" {word}", word):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    raise ValueError(f"{word!r} has no accepted one-token form")


def _continuation_id(tokenizer: Any, prompt: str, answer: str) -> int:
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    for suffix in (f" {answer}", answer):
        full = tokenizer.encode(prompt + suffix, add_special_tokens=False)
        if full[: len(prefix)] == prefix and len(full) == len(prefix) + 1:
            return int(full[-1])
    raise ValueError(f"{answer!r} is not one token after prompt {prompt!r}")


def _apply_basis_swap(hidden: Any, source: Any, target: Any, alpha: float) -> Any:
    """Identical to modal_v2._apply_basis_swap (returns updated hidden)."""
    import torch

    original_dtype = hidden.dtype
    value = hidden.float()
    matrix = torch.stack([source.float(), target.float()], dim=1)
    coords = value @ torch.linalg.pinv(matrix).T
    delta = (alpha * (coords.flip(-1) - coords)) @ matrix.T
    return (value + delta).to(original_dtype)


@app.function(image=image, volumes={"/cache": cache}, cpu=4.0, memory=32768,
              gpu="A100-80GB", timeout=3600, max_containers=1, retries=0)
def jlens_local_test() -> str:
    import torch
    import transformers
    from huggingface_hub import model_info
    from jlens import JacobianLens, from_hf
    from jlens.fitting import jacobian_for_prompt

    started = time.perf_counter()
    torch.manual_seed(SEED)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).cuda()
    hf_model.eval()
    model = from_hf(hf_model, tokenizer)
    lens = JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILENAME, revision=LENS_REVISION)
    if lens.d_model != model.d_model:
        raise RuntimeError("lens width mismatch")

    unembed = hf_model.lm_head.weight.detach().float()  # [vocab, d_model]
    args = COUNTRIES["args"]
    arg_ids = {a: _one_token_id(tokenizer, a) for a in args}
    avg_jacobian = lens.jacobians[WRITE_LAYER].to(unembed.device)  # [d_model, d_model]

    def directions_from(jacobian: Any) -> dict[str, Any]:
        vectors = unembed[[arg_ids[a] for a in args]] @ jacobian
        vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return {a: vectors[i].detach() for i, a in enumerate(args)}

    avg_dirs = directions_from(avg_jacobian)

    rows: list[dict[str, Any]] = []
    prompts_seen = 0
    for func in COUNTRIES["funcs"]:
        for source in args:
            prompt = str(func["template"]).format(arg=source)
            input_ids = model.encode(prompt, max_length=512)
            try:
                source_answer_id = _continuation_id(tokenizer, prompt, func["answers"][source])
            except ValueError:
                continue
            with torch.inference_mode():
                baseline_logits = hf_model(input_ids).logits[0, -1].float()
            baseline_top1 = int(baseline_logits.argmax().cpu())
            baseline_correct = baseline_top1 == source_answer_id

            # Prompt-local Jacobian at the single write site. Prompts here are
            # ~8-12 tokens, shorter than the fitting default skip_first=16, so
            # skip only the BOS position; the position-reduction (mean over
            # valid source positions, summed cotangents) follows upstream.
            local_jacobians, seq_len, n_valid = jacobian_for_prompt(
                model, prompt, [WRITE_LAYER], dim_batch=64, max_seq_len=128, skip_first=1
            )
            prompts_seen += 1
            local_dirs = directions_from(local_jacobians[WRITE_LAYER].to(unembed.device))
            del local_jacobians
            torch.cuda.empty_cache()

            for target in args:
                if target == source:
                    continue
                try:
                    target_answer_id = _continuation_id(tokenizer, prompt, func["answers"][target])
                except ValueError:
                    continue
                row = {
                    "function": func["name"], "source": source, "target": target,
                    "prompt": prompt, "seq_len": seq_len, "n_valid_positions": n_valid,
                    "baseline_correct": baseline_correct,
                    "baseline_top1": baseline_top1,
                    "source_answer_id": source_answer_id,
                    "target_answer_id": target_answer_id,
                }
                for label, dirs in (("average_lens", avg_dirs), ("prompt_local", local_dirs)):
                    source_vec = dirs[source]
                    target_vec = dirs[target]

                    def hook(module: Any, inputs: Any, output: Any) -> Any:
                        tensor = output if torch.is_tensor(output) else output[0]
                        updated = _apply_basis_swap(tensor, source_vec, target_vec, ALPHA)
                        if torch.is_tensor(output):
                            return updated
                        return (updated,) + tuple(output[1:])

                    handle = model.layers[WRITE_LAYER].register_forward_hook(hook)
                    with torch.inference_mode():
                        intervened_logits = hf_model(input_ids).logits[0, -1].float()
                    handle.remove()
                    row[f"{label}_top1_is_target"] = int(intervened_logits.argmax().cpu()) == target_answer_id
                    row[f"{label}_target_logodds"] = float(
                        intervened_logits.log_softmax(-1)[target_answer_id].cpu()
                    )
                rows.append(row)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    eligible = [r for r in rows if r["baseline_correct"]]
    summary = {
        "n_rows": len(rows),
        "n_eligible": len(eligible),
        "prompts_with_local_jacobian": prompts_seen,
        "average_lens_top1": sum(r["average_lens_top1_is_target"] for r in eligible),
        "prompt_local_top1": sum(r["prompt_local_top1_is_target"] for r in eligible),
        "average_lens_rate": (
            sum(r["average_lens_top1_is_target"] for r in eligible) / len(eligible)
            if eligible else None
        ),
        "prompt_local_rate": (
            sum(r["prompt_local_top1_is_target"] for r in eligible) / len(eligible)
            if eligible else None
        ),
        "average_lens_mean_target_logodds": (
            sum(r["average_lens_target_logodds"] for r in eligible) / len(eligible)
            if eligible else None
        ),
        "prompt_local_mean_target_logodds": (
            sum(r["prompt_local_target_logodds"] for r in eligible) / len(eligible)
            if eligible else None
        ),
    }
    result = {
        "metadata": {
            "schema_version": 1,
            "run_id": uuid.uuid4().hex,
            "created_at": datetime.now(UTC).isoformat(),
            "model_id": MODEL_ID,
            "model_revision_requested": MODEL_REVISION,
            "model_revision_resolved": model_info(MODEL_ID, revision=MODEL_REVISION).sha,
            "lens_repo": LENS_REPO,
            "lens_filename": LENS_FILENAME,
            "lens_revision": LENS_REVISION,
            "lens_code_commit": JLENS_COMMIT,
            "write_layer": WRITE_LAYER,
            "alpha": ALPHA,
            "topology": "single_layer_all_positions",
            "local_jacobian_skip_first": 1,
            "local_jacobian_dim_batch": 64,
            "seed": SEED,
            "dtype": "bfloat16",
            "gpu_actual": torch.cuda.get_device_name(0),
            "torch_version": str(torch.__version__),
            "transformers_version": transformers.__version__,
            "elapsed_seconds": elapsed,
            "h0_reference": "24/33 eligible country trials, average lens, layer 40 all positions",
        },
        "summary": summary,
        "rows": rows,
    }
    rendered = json.dumps(result, allow_nan=False)
    backup = __import__("pathlib").Path("/cache/rbg4_multimodel")
    backup.mkdir(parents=True, exist_ok=True)
    (backup / "jlens_local_test.json").write_text(rendered, encoding="utf-8")
    cache.commit()
    return rendered


@app.local_entrypoint()
def main() -> None:
    print(jlens_local_test.remote())
