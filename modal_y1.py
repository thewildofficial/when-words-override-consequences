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

app = modal.App("y1-game-battery")
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


def _continuation_id(tokenizer: Any, prompt: str, answer: str) -> int:
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    full = tokenizer.encode(prompt + answer, add_special_tokens=False)
    if full[: len(prefix)] == prefix and len(full) == len(prefix) + 1:
        return int(full[-1])
    raise ValueError(f"{answer!r} is not one token after frozen answer prefix")


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




@app.function(image=image, volumes={"/cache": cache}, cpu=8.0, memory=65536, gpu="A100-80GB", timeout=1200, max_containers=1, retries=0)
def y1_remote(rows: list) -> str:
    import torch, transformers
    started = time.perf_counter()
    model, _, metadata = _load_model(with_lens=False)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    def build(text, prefill):
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False) + prefill
    variants = {"V1_plain": ("", ""), "V2_prefill": ("", "The better choice is")}
    for vname, (_, prefill) in variants.items():
        ctx = build("x", prefill); sp = " " if prefill else ""
        A = _continuation_id(tok, ctx, sp + "A"); B = _continuation_id(tok, ctx, sp + "B")
        for start in range(0, len(rows), 8):
            part = rows[start:start + 8]
            ids = [tok.encode(build(r["prompt"], prefill), add_special_tokens=False) for r in part]
            w = max(map(len, ids))
            x = torch.zeros((len(part), w), dtype=torch.long, device="cuda"); m = torch.zeros_like(x)
            for i, t in enumerate(ids):
                x[i, w - len(t):] = torch.tensor(t, device="cuda"); m[i, w - len(t):] = 1
            with torch.inference_mode():
                lg = model._hf_model(x, attention_mask=m, logits_to_keep=1).logits[:, -1].float()
            for i, r in enumerate(part):
                r.setdefault("v", {})[vname] = {"A": float(lg[i, A]), "B": float(lg[i, B]), "top1": tok.decode([int(lg[i].argmax())])}
    return json.dumps({"elapsed": time.perf_counter() - started, "rows": rows, "meta": metadata})
