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


@app.function(image=image, volumes={"/cache": cache}, cpu=8.0, memory=65536, gpu="A100-80GB", timeout=1200, max_containers=1, retries=0)
def y4_remote(rows: list) -> str:
    """Final-prompt-position residual stream at every 4th layer (plain rendering), as base64 float16 npz, plus A/B logits."""
    import base64, io, numpy as np, torch, transformers
    started = time.perf_counter()
    model, _, metadata = _load_model(with_lens=False)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    ctx = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    A = _continuation_id(tok, ctx, "A"); B = _continuation_id(tok, ctx, "B")
    n_layers = None; feats = {}; ab = []
    for start in range(0, len(rows), 8):
        part = rows[start:start + 8]
        ids = [tok.encode(tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False) for r in part]
        w = max(map(len, ids))
        x = torch.zeros((len(part), w), dtype=torch.long, device="cuda"); m = torch.zeros_like(x)
        for i, t in enumerate(ids):
            x[i, w - len(t):] = torch.tensor(t, device="cuda"); m[i, w - len(t):] = 1
        with torch.inference_mode():
            out = model._hf_model(x, attention_mask=m, output_hidden_states=True, logits_to_keep=1)
        hs = out.hidden_states
        n_layers = len(hs)
        sel = list(range(0, n_layers, 4)) + ([n_layers - 1] if (n_layers - 1) % 4 else [])
        for L in sel:
            feats.setdefault(L, []).append(hs[L][:, -1].float().cpu().numpy().astype("float16"))
        lg = out.logits[:, -1].float()
        for i in range(len(part)):
            ab.append([float(lg[i, A]), float(lg[i, B])])
    buf = io.BytesIO()
    np.savez_compressed(buf, ab=np.array(ab, dtype="float32"), **{f"L{L}": np.concatenate(v) for L, v in feats.items()})
    return json.dumps({"elapsed": time.perf_counter() - started, "n_layers": n_layers, "npz_b64": base64.b64encode(buf.getvalue()).decode(), "meta": metadata})


@app.function(image=image, volumes={"/cache": cache}, cpu=8.0, memory=65536, gpu="A100-80GB", timeout=1500, max_containers=1, retries=0)
def y5_remote(rows: list, model_id: str, mode: str) -> str:
    """Base-vs-instruct comparison on the same items. mode='chat' (thinking off) or 'raw' (plain text + 'Answer:')."""
    import torch, transformers
    from huggingface_hub import model_info
    started = time.perf_counter()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda().eval()
    sha = model_info(model_id).sha
    def build(text):
        if mode == "chat":
            return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return text.replace("Answer with a single letter.", "") .rstrip() + "\nAnswer:"
    sp = "" if mode == "chat" else " "
    ctx = build("x")
    def tid(s):
        pre = tok.encode(ctx, add_special_tokens=False); full = tok.encode(ctx + s, add_special_tokens=False)
        assert full[:len(pre)] == pre and len(full) == len(pre) + 1, s
        return full[-1]
    A = tid(sp + "A"); B = tid(sp + "B")
    tok.padding_side = "left"
    for start in range(0, len(rows), 8):
        part = rows[start:start + 8]
        ids = [tok.encode(build(r["prompt"]), add_special_tokens=False) for r in part]
        w = max(map(len, ids))
        x = torch.zeros((len(part), w), dtype=torch.long, device="cuda"); m = torch.zeros_like(x)
        for i, t in enumerate(ids):
            x[i, w - len(t):] = torch.tensor(t, device="cuda"); m[i, w - len(t):] = 1
        with torch.inference_mode():
            lg = hf(x, attention_mask=m, logits_to_keep=1).logits[:, -1].float()
        for i, r in enumerate(part):
            r.setdefault("v", {})[mode] = {"A": float(lg[i, A]), "B": float(lg[i, B]), "top1": tok.decode([int(lg[i].argmax())])}
    return json.dumps({"elapsed": time.perf_counter() - started, "rows": rows, "model_id": model_id, "sha": sha, "mode": mode})


@app.function(image=image, volumes={"/cache": cache}, cpu=8.0, memory=65536, gpu="A100-80GB", timeout=2400, max_containers=1, retries=0)
def y6_remote(rows: list) -> str:
    """Native generation (no A/B letters) for the pinned 27B: modes free_word (greedy, <=8 new tokens) and cot (greedy, <=400 new tokens, thinking template off but explicit step-by-step instruction)."""
    import torch, transformers
    started = time.perf_counter()
    model, _, metadata = _load_model(with_lens=False)
    hf = model._hf_model
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION); tok.padding_side = "left"
    def prompt(r, mode):
        q = r["question"]; a, b = r["labels"]
        if mode == "free_word": t = f"{q}\n\nReply with exactly one word: {a} or {b}."
        else: t = f"{q}\n\nThink step by step briefly, then finish with the line 'Final answer: {a}' or 'Final answer: {b}'."
        return tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    for mode, mx in (("free_word", 8), ("cot", 400)):
        for start in range(0, len(rows), 8 if mode == "free_word" else 6):
            part = rows[start:start + (8 if mode == "free_word" else 6)]
            enc = tok([prompt(r, mode) for r in part], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                out = hf.generate(**enc, max_new_tokens=mx, do_sample=False)
            for i, r in enumerate(part):
                r.setdefault("gen", {})[mode] = tok.decode(out[i, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return json.dumps({"elapsed": time.perf_counter() - started, "rows": rows, "meta": metadata})


@app.function(image=image, volumes={"/cache": cache}, cpu=8.0, memory=65536, gpu="A100-80GB", timeout=2400, max_containers=1, retries=0)
def y7_remote(rows: list) -> str:
    """WWOC tool-use: A/B letter logits for 'ab' rows; for tool rows, summed log-prob of each tool name after the prefill '<tool_call>\\n<function=' (separately tokenised continuation), plus greedy free generation for a syntax gate on the first 64 tool rows."""
    import torch, transformers
    started = time.perf_counter()
    model, _, metadata = _load_model(with_lens=False)
    hf = model._hf_model
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    ctx0 = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    A = _continuation_id(tok, ctx0, "A"); B = _continuation_id(tok, ctx0, "B")
    def ctx_for(r):
        return tok.apply_chat_template([{"role": "user", "content": r["user"]}], tools=r["tools"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prefill = "<tool_call>\n<function="
    ntool = 0
    for r in rows:
        if r["fmt"] == "ab":
            text = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = torch.tensor([tok.encode(text, add_special_tokens=False)], device="cuda")
            with torch.inference_mode():
                lg = hf(ids, logits_to_keep=1).logits[0, -1].float()
            r["score"] = {"A": float(lg[A]), "B": float(lg[B])}
        else:
            base = tok.encode(ctx_for(r) + prefill, add_special_tokens=False)
            sc = {}
            for label, nm in r["names"].items():
                cont = tok.encode(nm + ">", add_special_tokens=False)
                ids = torch.tensor([base + cont], device="cuda")
                with torch.inference_mode():
                    logits = hf(ids).logits[0].float().log_softmax(-1)
                lp = 0.0
                for k, t in enumerate(cont):
                    lp += float(logits[len(base) - 1 + k, t])
                sc[label] = lp
            r["score"] = sc
            if ntool < 64:
                ntool += 1
                enc = tok(ctx_for(r), return_tensors="pt", add_special_tokens=False).to("cuda")
                with torch.inference_mode():
                    o = hf.generate(**enc, max_new_tokens=60, do_sample=False)
                r["gen"] = tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=False)
    return json.dumps({"elapsed": time.perf_counter() - started, "rows": rows, "meta": metadata})


@app.function(image=image, volumes={"/cache": cache}, cpu=8.0, memory=65536, gpu="A100-80GB", timeout=3000, max_containers=1, retries=0)
def y9_remote(rows: list) -> str:
    """Native tool-call generation for tool rows (greedy, <=80 new tokens, no prefill, thinking off). Raw text returned; parsing/mock execution done offline."""
    import torch, transformers
    started = time.perf_counter()
    model, _, metadata = _load_model(with_lens=False)
    hf = model._hf_model
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION); tok.padding_side = "left"
    def ctx(r):
        return tok.apply_chat_template([{"role": "user", "content": r["user"]}], tools=r["tools"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    for s in range(0, len(rows), 6):
        part = rows[s:s + 6]
        enc = tok([ctx(r) for r in part], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            out = hf.generate(**enc, max_new_tokens=400, do_sample=False)
        for i, r in enumerate(part):
            r["gen"] = tok.decode(out[i, enc["input_ids"].shape[1]:], skip_special_tokens=False)
    return json.dumps({"elapsed": time.perf_counter() - started, "rows": rows, "meta": metadata})


@app.function(image=image, volumes={"/cache": cache}, cpu=8.0, memory=65536, gpu="A100-80GB", timeout=3000, max_containers=1, retries=0)
def y10_remote(rows: list) -> str:
    """Y6 cot mode only, max_new_tokens=1000 (same prompt/template as y6 cot)."""
    import torch, transformers
    started = time.perf_counter()
    model, _, metadata = _load_model(with_lens=False)
    hf = model._hf_model
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION); tok.padding_side = "left"
    def prompt(r):
        q = r["question"]; a, b = r["labels"]
        t = f"{q}\n\nThink step by step briefly, then finish with the line 'Final answer: {a}' or 'Final answer: {b}'."
        return tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    for s in range(0, len(rows), 8):
        part = rows[s:s + 8]
        enc = tok([prompt(r) for r in part], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            out = hf.generate(**enc, max_new_tokens=1000, do_sample=False)
        for i, r in enumerate(part):
            r["gen"] = {"cot1000": tok.decode(out[i, enc["input_ids"].shape[1]:], skip_special_tokens=True)}
    return json.dumps({"elapsed": time.perf_counter() - started, "rows": rows, "meta": metadata})


@app.function(image=image, volumes={"/cache": cache}, cpu=8.0, memory=65536, gpu="A100-80GB", timeout=3300, max_containers=1, retries=0)
def y11_remote(rows: list) -> str:
    """Uniform cap: every row greedy, max_new_tokens=400, tools in the chat-template format, thinking off."""
    import torch, transformers
    started = time.perf_counter()
    model, _, metadata = _load_model(with_lens=False)
    hf = model._hf_model
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION); tok.padding_side = "left"
    CAP = 400
    def ctx(r):
        return tok.apply_chat_template([{"role": "user", "content": r["user"]}], tools=r["tools"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    for s in range(0, len(rows), 12):
        part = rows[s:s + 12]
        enc = tok([ctx(r) for r in part], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            out = hf.generate(**enc, max_new_tokens=CAP, do_sample=False)
        for i, r in enumerate(part):
            r["gen"] = tok.decode(out[i, enc["input_ids"].shape[1]:], skip_special_tokens=False)
            r["n_new_tokens"] = int((out[i, enc["input_ids"].shape[1]:] != tok.pad_token_id).sum())
    return json.dumps({"elapsed": time.perf_counter() - started, "cap": CAP, "rows": rows, "meta": metadata})


@app.function(image=image, volumes={"/cache": cache}, cpu=8.0, memory=65536, gpu="A100-80GB", timeout=3300, max_containers=1, retries=0)
def y12_remote(rows: list) -> str:
    """Same as y11_remote (CAP=400, greedy, batch 12), distinct function name for a fresh container."""
    import torch, transformers
    started = time.perf_counter()
    model, _, metadata = _load_model(with_lens=False)
    hf = model._hf_model
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION); tok.padding_side = "left"
    CAP = 400
    def ctx(r):
        return tok.apply_chat_template([{"role": "user", "content": r["user"]}], tools=r["tools"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    for s in range(0, len(rows), 16):
        part = rows[s:s + 16]
        enc = tok([ctx(r) for r in part], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            out = hf.generate(**enc, max_new_tokens=CAP, do_sample=False)
        for i, r in enumerate(part):
            r["gen"] = tok.decode(out[i, enc["input_ids"].shape[1]:], skip_special_tokens=False)
            r["n_new_tokens"] = int((out[i, enc["input_ids"].shape[1]:] != tok.pad_token_id).sum())
    return json.dumps({"elapsed": time.perf_counter() - started, "cap": CAP, "rows": rows, "meta": metadata})


@app.function(image=image, volumes={"/cache": cache}, cpu=8.0, memory=65536, gpu="A100-80GB", timeout=3300, max_containers=1, retries=0)
def y13_remote(rows: list) -> str:
    """Same as y11_remote (CAP=400, greedy, batch 12), distinct function name for a fresh container."""
    import torch, transformers
    started = time.perf_counter()
    model, _, metadata = _load_model(with_lens=False)
    hf = model._hf_model
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION); tok.padding_side = "left"
    CAP = 400
    def ctx(r):
        return tok.apply_chat_template([{"role": "user", "content": r["user"]}], tools=r["tools"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    for s in range(0, len(rows), 16):
        part = rows[s:s + 16]
        enc = tok([ctx(r) for r in part], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            out = hf.generate(**enc, max_new_tokens=CAP, do_sample=False)
        for i, r in enumerate(part):
            r["gen"] = tok.decode(out[i, enc["input_ids"].shape[1]:], skip_special_tokens=False)
            r["n_new_tokens"] = int((out[i, enc["input_ids"].shape[1]:] != tok.pad_token_id).sum())
    return json.dumps({"elapsed": time.perf_counter() - started, "cap": CAP, "rows": rows, "meta": metadata})
