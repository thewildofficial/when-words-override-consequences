"""Bounded, sequential Modal experiment; run locally or from GitHub Actions."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

from jspace_policy.budget import estimate_cost
from jspace_policy.semantic_action_mechanisms import (
    LAYERS,
    MODEL_ID,
    MODEL_REVISION,
    STUDY_ID,
    SYSTEM,
    analyze,
    make_dataset,
    margin,
    paired_rows,
    queries_for,
    select_mechanistic_contrast,
    verify_dataset,
)
from jspace_policy.sprint_runtime import (
    digest,
    load_json_file,
    reserve,
    write_gzip_new,
    write_new,
)

ROOT = Path("results/semantic_action_mechanisms")
PROTOCOL = Path("experiments/semantic_action_mechanisms/protocol.json")
TIMEOUTS = {"prepare": 300, "behavior": 1500, "discovery": 1500, "locked": 900}
GPU_RATE = 0.000694 + 8 * 0.0000131 + 64 * 0.00000222
CPU_RATE = 2 * 0.0000131 + 8 * 0.00000222
CEILINGS = {
    k: (v + 10) * (CPU_RATE if k == "prepare" else GPU_RATE) * 1.2 for k, v in TIMEOUTS.items()
}
CEILING = 5.0
app = modal.App("semantic-action-mechanisms")
cache = modal.Volume.from_name("jspace-hf-cache", create_if_missing=True)
artifacts = modal.Volume.from_name(
    "semantic-action-mechanisms-artifacts", create_if_missing=True
)
cpu_image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(
        "transformers==5.15.1", "huggingface_hub==1.30.0", "numpy>=2", "jinja2==3.1.6"
    )
    .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src/jspace_policy", remote_path="/root/jspace_policy")
)
gpu_image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("torch==2.13.0", "transformers==5.15.1", "huggingface_hub==1.30.0")
    .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src/jspace_policy", remote_path="/root/jspace_policy")
)


def prepare_dataset(tokenizer, dataset):
    verify_dataset(dataset)
    queries = {}
    for row in dataset["rows"]:
        for spec in queries_for(row):
            rendered = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": spec["text"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
            ids = encoded["input_ids"]
            candidate_ids = []
            for answer in spec["labels"]:
                full = tokenizer.encode(rendered + answer, add_special_tokens=False)
                if full[:-1] != ids or len(full) != len(ids) + 1:
                    raise RuntimeError(f"non-single-token candidate: {spec['id']}:{answer}")
                candidate_ids.append(full[-1])
            anchors = {}
            for name, text in row["anchors"].items():
                if not text:
                    continue
                end = rendered.index(text) + len(text)
                positions = [
                    i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < end and b > 0
                ]
                anchors[name] = max(positions)
            queries[spec["id"]] = {
                "input_ids": ids,
                "candidate_ids": candidate_ids,
                "labels": spec["labels"],
                "expected": spec["expected"],
                "anchors": anchors,
                "rendered_sha256": digest(rendered),
            }
    body = {
        "study_id": STUDY_ID,
        "dataset": dataset,
        "queries": queries,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
    }
    return {**body, "sha256": digest(body)}


@app.function(
    image=cpu_image,
    cpu=2,
    memory=8192,
    timeout=300,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/cache": cache},
)
def prepare_remote(discovery: dict, locked: dict) -> str:
    from transformers import AutoTokenizer

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    result = {
        "discovery": prepare_dataset(tokenizer, discovery),
        "locked": prepare_dataset(tokenizer, locked),
        "elapsed_seconds": time.perf_counter() - started,
    }
    cache.commit()
    return json.dumps(result)


def verify_prepared(prepared):
    if prepared["sha256"] != digest({k: v for k, v in prepared.items() if k != "sha256"}):
        raise RuntimeError("prepared payload changed")
    verify_dataset(prepared["dataset"])
    if (prepared["model_id"], prepared["revision"]) != (MODEL_ID, MODEL_REVISION):
        raise RuntimeError("checkpoint changed")
    if set(prepared["queries"]) != {
        q["id"] for row in prepared["dataset"]["rows"] for q in queries_for(row)
    }:
        raise RuntimeError("prepared query inventory changed")


def load_model():
    import torch
    import transformers

    torch.manual_seed(101126)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .cuda()
        .eval()
    )
    layers = getattr(model.model, "layers", None)
    if layers is None:
        layers = model.model.language_model.layers
    if len(layers) != 64 or model.config.hidden_size != 5120:
        raise RuntimeError("checkpoint architecture changed")
    metadata = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "use_cache": False,
        "dtype": "bfloat16",
    }
    return model, layers, metadata


def forward(model, queries, pad, layers=None, capture=None, patch=None):
    """Same execution path for baseline, capture, identity and donor patches."""
    import torch

    width = max(len(q["input_ids"]) for q in queries)
    ids = torch.full((len(queries), width), pad, device="cuda", dtype=torch.long)
    attention = torch.zeros_like(ids)
    for i, query in enumerate(queries):
        ids[i, -len(query["input_ids"]) :] = torch.tensor(query["input_ids"], device="cuda")
        attention[i, -len(query["input_ids"]) :] = 1
    if (capture or patch) and len(queries) != 1:
        raise RuntimeError("causal forwards must be singleton")
    vectors, handles = {}, []
    if capture:
        for layer_index, anchor in capture:

            def hook(_module, _inputs, output, layer_index=layer_index, anchor=anchor):
                hidden = output[0] if isinstance(output, tuple) else output
                vectors[(layer_index, anchor)] = (
                    hidden[0, queries[0]["anchors"][anchor]].detach().cpu().clone()
                )

            handles.append(layers[layer_index].register_forward_hook(hook))
    if patch:
        layer_index, anchor, vector = patch

        def patch_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            modified = hidden.clone()
            modified[0, queries[0]["anchors"][anchor]] = vector.to("cuda")
            return (modified, *output[1:]) if isinstance(output, tuple) else modified

        handles.append(layers[layer_index].register_forward_hook(patch_hook))
    try:
        with torch.inference_mode():
            logits = (
                model(
                    input_ids=ids, attention_mask=attention, use_cache=False, logits_to_keep=1
                )
                .logits[:, -1]
                .float()
            )
        output = []
        for i, query in enumerate(queries):
            legal = logits[i, query["candidate_ids"]]
            output.append(
                {
                    "choice": query["labels"][int(legal.argmax().item())],
                    "logits": legal.cpu().tolist(),
                    "format_valid": int(logits[i].argmax().item()) in query["candidate_ids"],
                }
            )
        return output, vectors
    finally:
        for handle in handles:
            handle.remove()


def score_error(left, right):
    return max(abs(a - b) for a, b in zip(left["logits"], right["logits"], strict=True))


@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    cpu=8,
    memory=65536,
    timeout=1500,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/cache": cache},
)
def behavior_remote(prepared: dict, provenance: dict) -> str:
    verify_prepared(prepared)
    started = time.perf_counter()
    model, _, metadata = load_model()
    items = sorted(prepared["queries"].items(), key=lambda x: (len(x[1]["input_ids"]), x[0]))
    pad = prepared["pad_token_id"]
    # Probe the failing assertion cells, controls and reports across lengths.
    indices = sorted(
        set([0, len(items) - 1] + [int(i * (len(items) - 1) / 15) for i in range(16)])
    )
    singleton_error = 0.0
    for i in indices:
        first = forward(model, [items[i][1]], pad)[0][0]
        replay = forward(model, [items[i][1]], pad)[0][0]
        singleton_error = max(singleton_error, score_error(first, replay))
    if singleton_error > 1e-5:
        return json.dumps(
            {
                "status": "instrument_gate_failed",
                "scores": {},
                "replay_error": singleton_error,
                "metadata": metadata,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    batch_error, batch_choices_agree = 0.0, True
    for start in range(4):
        probe = [items[i][1] for i in indices[start::4]]
        batched = forward(model, probe, pad)[0]
        singles = [forward(model, [q], pad)[0][0] for q in probe]
        batch_error = max(
            batch_error, max(score_error(a, b) for a, b in zip(batched, singles, strict=True))
        )
        batch_choices_agree &= all(
            a["choice"] == b["choice"] for a, b in zip(batched, singles, strict=True)
        )
    # Predeclared singleton fallback; no tolerance widening or new paid dispatch.
    batch_size = 4 if batch_error <= 0.125 and batch_choices_agree else 1
    scores = {}
    for start in range(0, len(items), batch_size):
        part = items[start : start + batch_size]
        values = forward(model, [q for _, q in part], pad)[0]
        scores.update({key: value for (key, _), value in zip(part, values, strict=True)})
        if start % 240 == 0:
            print(f"behavior: {start}/{len(items)}, batch={batch_size}", flush=True)
    return json.dumps(
        {
            "status": "scored",
            "scores": scores,
            "dataset_sha256": prepared["dataset"]["sha256"],
            "prepared_sha256": prepared["sha256"],
            "metadata": metadata,
            "provenance": provenance,
            "batch_size": batch_size,
            "instrument": {
                "replay_error": singleton_error,
                "batch_error": batch_error,
                "batch_choices_agree": batch_choices_agree,
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
    )


def mechanistic_impl(prepared, contrast, sites, discovery, provenance):
    import torch

    verify_prepared(prepared)
    started = time.perf_counter()
    model, layers, metadata = load_model()
    pad, query_map = prepared["pad_token_id"], prepared["queries"]
    pairs = paired_rows(prepared["dataset"], contrast)
    if discovery:
        # One predeclared target/permutation per family, never selected on errors.
        pairs = [
            (s, r)
            for s, r in pairs
            if s["target_index"] == int(s["family"].split("-")[-1]) % 2
            and s["permutation"] == 0
        ]
    eligible, excluded = [], []
    for source, recipient in pairs:
        sq, rq = query_map[source["id"] + ":action"], query_map[recipient["id"] + ":action"]
        if contrast == "state" and (
            len(sq["input_ids"]) != len(rq["input_ids"])
            or any(sq["anchors"][a] != rq["anchors"][a] for _, a in sites)
        ):
            excluded.append(source["family"])
        else:
            eligible.append((source, recipient))
    if len({s["family"] for s, _ in eligible}) < 8:
        return {
            "status": "alignment_gate_failed",
            "excluded": excluded,
            "elapsed_seconds": time.perf_counter() - started,
        }
    captures, baselines = {}, {}
    maximum_replay_error = 0.0
    for source, recipient in eligible:
        for row in (source, recipient):
            query = query_map[row["id"] + ":action"]
            value, vectors = forward(model, [query], pad, layers, capture=sites)
            replay = forward(model, [query], pad)[0][0]
            maximum_replay_error = max(maximum_replay_error, score_error(value[0], replay))
            captures[row["id"]], baselines[row["id"]] = vectors, replay
    if maximum_replay_error > 1e-5:
        return {
            "status": "capture_replay_gate_failed",
            "replay_error": maximum_replay_error,
            "elapsed_seconds": time.perf_counter() - started,
        }
    # Persist capture before patching so an instrument failure or timeout cannot
    # silently discard the states and contemporaneous unpatched baselines.
    artifact_path = (
        Path("/artifacts")
        / provenance["run_id"]
        / ("discovery_states.pt" if discovery else "locked_states.pt")
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("xb") as stream:
        torch.save({"captures": captures, "baselines": baselines, "sites": sites}, stream)
    artifacts.commit()
    records = []
    identity_error = 0.0
    for layer, anchor in sites:
        for source, recipient in eligible:
            for donor, destination, direction in (
                (source, recipient, "forward"),
                (recipient, source, "reverse"),
            ):
                query = query_map[destination["id"] + ":action"]
                own = captures[destination["id"]][layer, anchor]
                identity = forward(model, [query], pad, layers, patch=(layer, anchor, own))[0][
                    0
                ]
                identity_error = max(
                    identity_error, score_error(identity, baselines[destination["id"]])
                )
                if identity_error > 1e-5:
                    return {
                        "status": "identity_gate_failed",
                        "records": records,
                        "identity_error": identity_error,
                        "artifact_path": str(artifact_path),
                        "provenance": provenance,
                        "metadata": metadata,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                patched = forward(
                    model,
                    [query],
                    pad,
                    layers,
                    patch=(layer, anchor, captures[donor["id"]][layer, anchor]),
                )[0][0]
                record = {
                    "family": destination["family"],
                    "source": donor["id"],
                    "destination": destination["id"],
                    "direction": direction,
                    "layer": layer,
                    "anchor": anchor,
                    "expected_action": destination["expected_action"],
                    "baseline": baselines[destination["id"]],
                    "patched": patched,
                    "identity": identity,
                    "margin_change": margin(destination, patched)
                    - margin(destination, baselines[destination["id"]]),
                }
                if not discovery:
                    # A different lexical game with the same destination state/binding
                    # and same output label tests generic donor-state transport.
                    others = [
                        candidate
                        for pair in eligible
                        for candidate in pair
                        if candidate["family"] != destination["family"]
                        and candidate["expected_action"] == destination["expected_action"]
                        and candidate["state"] == destination["state"]
                        and candidate["binding"] == destination["binding"]
                    ]
                    wrong = sorted(others, key=lambda r: r["id"])[0]
                    wrong_score = forward(
                        model,
                        [query],
                        pad,
                        layers,
                        patch=(layer, anchor, captures[wrong["id"]][layer, anchor]),
                    )[0][0]
                    record["wrong_game"] = wrong_score
                    record["wrong_game_margin_change"] = margin(
                        destination, wrong_score
                    ) - margin(destination, baselines[destination["id"]])
                    if direction == "forward":
                        report_results = []
                        for action in ("A", "B"):
                            rq = query_map[destination["id"] + ":report_" + action]
                            report_base = forward(model, [rq], pad)[0][0]
                            report_patch = forward(
                                model,
                                [rq],
                                pad,
                                layers,
                                patch=(layer, anchor, captures[donor["id"]][layer, anchor]),
                            )[0][0]
                            report_results.append(
                                {
                                    "expected": rq["expected"],
                                    "baseline": report_base,
                                    "patched": report_patch,
                                }
                            )
                        record["reports"] = report_results
                records.append(record)
        print(f"patch site complete: layer={layer}, anchor={anchor}", flush=True)
        checkpoint = artifact_path.parent / f"{artifact_path.stem}-layer{layer}-{anchor}.json"
        with checkpoint.open("x") as stream:
            json.dump({"records": records, "identity_error": identity_error}, stream)
        artifacts.commit()
    return {
        "status": "patched" if identity_error <= 1e-5 else "identity_gate_failed",
        "records": records,
        "identity_error": identity_error,
        "capture_replay_error": maximum_replay_error,
        "excluded_families": sorted(set(excluded)),
        "metadata": metadata,
        "provenance": provenance,
        "contrast": contrast,
        "dataset_sha256": prepared["dataset"]["sha256"],
        "prepared_sha256": prepared["sha256"],
        "artifact_path": str(artifact_path),
        "elapsed_seconds": time.perf_counter() - started,
    }


@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    cpu=8,
    memory=65536,
    timeout=1500,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/cache": cache, "/artifacts": artifacts},
)
def discovery_remote(prepared: dict, contrast: str, provenance: dict) -> str:
    anchors = ("state_end", "policy_end") if contrast == "state" else ("policy_end",)
    return json.dumps(
        mechanistic_impl(
            prepared,
            contrast,
            [(layer, anchor) for layer in LAYERS for anchor in anchors],
            True,
            provenance,
        )
    )


@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    cpu=8,
    memory=65536,
    timeout=900,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/cache": cache, "/artifacts": artifacts},
)
def locked_remote(prepared: dict, contrast: str, site: dict, provenance: dict) -> str:
    return json.dumps(
        mechanistic_impl(
            prepared, contrast, [(site["layer"], site["anchor"])], False, provenance
        )
    )


def patch_analysis(result):
    import numpy as np

    if result["status"] != "patched":
        return {"status": result["status"], "selected": None}
    groups = {}
    for row in result["records"]:
        key = (row["layer"], row["anchor"])
        groups.setdefault(key, {}).setdefault(row["family"], []).append(row)
    summaries = []
    for (layer, anchor), families in sorted(groups.items()):
        directional = [
            np.mean(
                [r["margin_change"] * (1 if r["direction"] == "forward" else -1) for r in rows]
            )
            for rows in families.values()
        ]
        values = np.array(directional)
        signs = np.random.default_rng(101126).choice([-1, 1], (10000, len(values)))
        p = float(
            (np.sum(np.abs((signs * values).mean(axis=1)) >= abs(values.mean())) + 1) / 10001
        )
        forward = [r for rows in families.values() for r in rows if r["direction"] == "forward"]
        failures = [r for r in forward if r["baseline"]["choice"] != r["expected_action"]]
        repaired = [
            r
            for r in failures
            if r["patched"]["choice"] == r["expected_action"]
            and margin({"expected_action": r["expected_action"]}, r["patched"]) > 0
        ]
        summary = {
            "layer": layer,
            "anchor": anchor,
            "families": len(values),
            "directional_margin_change": float(values.mean()),
            "family_directional_changes": values.tolist(),
            "sign_flip_p": p,
            "failures": len(failures),
            "repairs": len(repaired),
            "repair_fraction": len(repaired) / len(failures) if failures else 0.0,
        }
        if any("wrong_game_margin_change" in r for r in forward):
            summary["forward_minus_wrong_game"] = float(
                np.mean(
                    [
                        np.mean(
                            [
                                r["margin_change"] - r["wrong_game_margin_change"]
                                for r in rows
                                if r["direction"] == "forward"
                            ]
                        )
                        for rows in families.values()
                    ]
                )
            )
        summaries.append(summary)
    # Discovery selection only; held-out site was frozen before locked inference.
    qualified = [
        s
        for s in summaries
        if s["directional_margin_change"] >= 0.25
        and s["repair_fraction"] >= 0.20
        and s["families"] >= 8
    ]
    selected = max(qualified, key=lambda x: x["directional_margin_change"], default=None)
    return {"status": "analyzed", "sites": summaries, "selected": selected}


def locked_verdict(raw, summary):
    if raw["status"] != "patched" or not summary.get("sites"):
        return {"supported": False, "reason": raw["status"]}
    site = summary["sites"][0]
    reports = [r for row in raw["records"] for r in row.get("reports", [])]
    report_accuracy = sum(r["patched"]["choice"] == r["expected"] for r in reports) / len(
        reports
    )
    gates = {
        "identity": raw["identity_error"] <= 1e-5,
        "families": site["families"] >= 8,
        "repair": site["failures"] >= 8 and site["repair_fraction"] >= 0.20,
        "directional_margin": site["directional_margin_change"] >= 0.25,
        "paired_test": site["sign_flip_p"] < 0.05,
        "wrong_game_specificity": site["forward_minus_wrong_game"] >= 0.10,
        "report_preservation": report_accuracy >= 0.95,
    }
    return {
        "supported": all(gates.values()),
        "gates": gates,
        "patched_report_accuracy": report_accuracy,
        "interpretation": "Transport at the frozen site/contrast; not a complete circuit.",
    }


@app.local_entrypoint()
def run(run_id: str = "semantic-mechanisms-qwen36-20261011-v1", prepare_only: bool = False):
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("invalid run_id")
    protocol = load_json_file(PROTOCOL)
    if protocol["hard_ceiling_usd"] != CEILING or sum(CEILINGS.values()) > CEILING:
        raise RuntimeError("execution plan exceeds ceiling")
    root = ROOT / run_id
    if root.exists():
        raise RuntimeError("run directory exists; retry/overwrite refused")
    provenance = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "protocol_sha256": digest(protocol),
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path("src/jspace_policy/semantic_action_mechanisms.py"))
        },
    }
    write_new(root / "provenance.json", provenance)
    write_new(root / "protocol.json", protocol)
    print(json.dumps({"run_id": run_id, "maximum_reserved_usd": sum(CEILINGS.values())}))
    ledger = ROOT / "reservations.jsonl"

    def dispatch(stage, function, *args):
        reservation = reserve(
            ledger,
            run_id + ":" + stage,
            "preflight" if stage == "prepare" else "mechanistic",
            CEILINGS[stage],
            global_ceiling_usd=CEILING,
        )
        write_new(root / f"{stage}_reservation.json", reservation)
        try:
            value = json.loads(function.remote(*args))
        except Exception as exc:
            write_new(
                root / f"{stage}_failure.json",
                {"type": type(exc).__name__, "message": str(exc), "reservation_retained": True},
            )
            raise
        write_gzip_new(root / f"{stage}_raw.json.gz", value)
        if stage != "prepare":
            write_new(
                root / f"{stage}_cost_estimate.json",
                estimate_cost(
                    "A100-80GB", value["elapsed_seconds"], cpu_cores=8, memory_gib=64
                ).__dict__,
            )
        return value

    datasets = {split: make_dataset(split) for split in ("discovery", "locked")}
    if {split: value["sha256"] for split, value in datasets.items()} != protocol[
        "dataset_sha256"
    ]:
        raise RuntimeError("dataset differs from committed protocol freeze")
    prepared = dispatch("prepare", prepare_remote, datasets["discovery"], datasets["locked"])
    write_new(
        root / "prepared_manifest.json",
        {s: prepared[s]["sha256"] for s in ("discovery", "locked")},
    )
    if prepare_only:
        print("CPU prepare completed; no GPU inference.")
        return
    behavior = dispatch("behavior", behavior_remote, prepared["discovery"], provenance)
    if behavior["status"] != "scored":
        print("Instrument gate failed; later stages withheld.")
        return
    behavior_summary = analyze(prepared["discovery"]["dataset"], behavior)
    write_new(root / "behavior_analysis.json", behavior_summary)
    contrast = select_mechanistic_contrast(behavior_summary)
    write_new(
        root / "contrast_freeze.json",
        {"contrast": contrast, "analysis_sha256": digest(behavior_summary)},
    )
    print(json.dumps({"contrast": contrast, "contrasts": behavior_summary["contrasts"]}))
    if contrast is None:
        print("No eligible behavioral contrast; causal stages withheld.")
        return
    discovery = dispatch(
        "discovery", discovery_remote, prepared["discovery"], contrast, provenance
    )
    discovery_summary = patch_analysis(discovery)
    write_new(root / "discovery_analysis.json", discovery_summary)
    site = discovery_summary["selected"]
    write_new(
        root / "site_freeze.json", {"site": site, "analysis_sha256": digest(discovery_summary)}
    )
    if site is None:
        print("No eligible discovery patch; locked stage withheld.")
        return
    locked = dispatch("locked", locked_remote, prepared["locked"], contrast, site, provenance)
    locked_summary = patch_analysis(locked)
    write_new(root / "locked_analysis.json", locked_summary)
    write_new(root / "locked_verdict.json", locked_verdict(locked, locked_summary))
    print(json.dumps(locked_summary, indent=2))
