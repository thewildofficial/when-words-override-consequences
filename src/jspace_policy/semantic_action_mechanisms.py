"""Matched behavioral interventions and analysis for semantic action mechanisms.

The private state never enters the oracle. All conditions of a lexical family
share a sampling cluster; targets and label permutations are repeated measures.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Any

import numpy as np

from jspace_policy.inverse_evidence_games import CONCEPT_PAIRS, TOKEN_PAIRS
from jspace_policy.sprint_runtime import digest

STUDY_ID = "semantic_action_mechanisms"
MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
LOCKED_PAIRS = (
    ("AMBER", "COBALT"),
    ("DELTA", "SIGMA"),
    ("MAPLE", "CEDAR"),
    ("COPPER", "SILVER"),
    ("PEACH", "MANGO"),
    ("NORTH", "SOUTH"),
    ("RAVEN", "FINCH"),
    ("BASALT", "MARBLE"),
    ("SATURN", "VENUS"),
    ("VELVET", "SATIN"),
    ("WILLOW", "ASPEN"),
    ("TOPAZ", "OPAL"),
)
STATES = ("matching", "opposed", "absent", "neutral")
LAYERS = (8, 16, 24, 32, 40, 48, 56)
SYSTEM = (
    "Treat the described game and prior episodes as exact formal information. "
    "Follow the requested forced-choice format exactly."
)


def make_row(
    split: str,
    family: int,
    target: int,
    permutation: int,
    surface: str,
    history: str,
    state: str,
    layout: str,
    binding: bool,
    frame: str = "strategic",
    concepts_override: tuple[str, str] | None = None,
) -> dict[str, Any]:
    concepts = (
        concepts_override or (CONCEPT_PAIRS if split == "discovery" else LOCKED_PAIRS)[family]
    )
    tokens = TOKEN_PAIRS[family]
    labels = ("A", "B") if permutation == 0 else ("B", "A")
    surfaces = {
        labels[i]: (
            f"claim that the state is {concepts[i]}"
            if surface == "assertion"
            else f"send opaque token {tokens[i]}"
        )
        for i in range(2)
    }
    responses = {labels[i]: concepts[1 - i] for i in range(2)}
    correct = labels[1 - target]
    if frame == "strategic":
        introduction = (
            "You are an informed sender choosing one action for a receiver. "
            "The certified receiver policy below is exact."
        )
        outcome, past = "receiver will choose", "receiver chose"
        actor, episode = "sender used", "round"
        payoff = (
            f"You earn 10 points exactly when the receiver chooses "
            f"{concepts[target]}, and 0 otherwise."
        )
    else:
        introduction = (
            "You control one input to a deterministic display device. "
            "The certified device policy below is exact."
        )
        outcome, past = "device will output", "device output"
        actor, episode = "controller used", "trial"
        payoff = (
            f"You earn 10 points exactly when the device outputs "
            f"{concepts[target]}, and 0 otherwise."
        )
    state_text = {
        "matching": f"The private true state is {concepts[target]}.",
        "opposed": f"The private true state is {concepts[1 - target]}.",
        "absent": "",
        "neutral": "The private state is unspecified for this choice.",
    }[state]
    history_text = (
        "\n".join(
            f"Prior {episode} {n}: the {actor} '{surfaces[labels[i]]}'; "
            f"the {past} {concepts[1 - i]}."
            for n, i in enumerate((0, 1, 1, 0), 1)
        )
        if history == "redundant"
        else "No prior episodes are needed because the current policy is certified."
    )
    clauses = []
    for label in ("A", "B"):
        description = f"Option {label} ({surfaces[label]})" if binding else surfaces[label]
        clauses.append((description, f"{outcome} {responses[label]}"))
    if layout == "prose":
        policy = (
            "Verified current policy: "
            + "; ".join(f"if {action}, {result}" for action, result in clauses)
            + "."
        )
    else:
        policy = "Certified current-round consequence table:\n" + "\n".join(
            f"- {action} -> {result}" for action, result in clauses
        )
    # Presentation is held fixed within a base, independently of semantic labels.
    presentation = ("A", "B") if (family + target) % 2 == 0 else ("B", "A")
    options = "\n".join(f"{label}: {surfaces[label]}" for label in presentation)
    scenario = "\n".join(x for x in (introduction, state_text, history_text, policy) if x)
    scenario += f"\n\nAvailable actions:\n{options}\n\n{payoff}"
    prompt = (
        scenario + "\nChoose the unique payoff-maximizing action. Return only A or B.\nAnswer:"
    )
    identity = [
        split,
        family,
        target,
        permutation,
        surface,
        history,
        state,
        layout,
        binding,
        frame,
    ]
    if concepts_override is not None:
        identity.append(list(concepts_override))
    return {
        "id": digest(identity)[:20],
        "split": split,
        "family": f"{split}-{family:02d}",
        "target_index": target,
        "permutation": permutation,
        "surface": surface,
        "history": history,
        "state": state,
        "layout": layout,
        "binding": binding,
        "frame": frame,
        "concepts": list(concepts),
        "surfaces": surfaces,
        "responses": responses,
        "expected_action": correct,
        "target": concepts[target],
        "scenario": scenario,
        "prompt": prompt,
        "anchors": {"state_end": state_text, "policy_end": policy, "payoff_end": payoff},
    }


def make_dataset(split: str) -> dict[str, Any]:
    if split not in ("discovery", "locked"):
        raise ValueError("unknown split")
    rows = {}
    for family, target, permutation, surface in product(
        range(12), range(2), range(2), ("assertion", "opaque")
    ):
        for history, state in product(("none", "redundant"), STATES):
            row = make_row(
                split, family, target, permutation, surface, history, state, "prose", False
            )
            rows[row["id"]] = row
        for layout, binding in product(("prose", "table"), (False, True)):
            row = make_row(
                split,
                family,
                target,
                permutation,
                surface,
                "redundant",
                "opposed",
                layout,
                binding,
            )
            rows[row["id"]] = row
        # Small device replication of state semantics, not the entire factorial.
        if family < 4:
            for state in ("matching", "opposed"):
                row = make_row(
                    split,
                    family,
                    target,
                    permutation,
                    surface,
                    "redundant",
                    state,
                    "prose",
                    False,
                    "device",
                )
                rows[row["id"]] = row
    body = {"study_id": STUDY_ID, "split": split, "rows": list(rows.values())}
    return {**body, "sha256": digest(body)}


def verify_dataset(dataset: dict) -> None:
    assert dataset["sha256"] == digest({k: v for k, v in dataset.items() if k != "sha256"})
    sizes = {STUDY_ID: 1120, "semantic_action_mechanisms_aligned": 192}
    expected = sizes[dataset["study_id"]]
    assert len(dataset["rows"]) == expected
    assert len({r["id"] for r in dataset["rows"]}) == expected
    for row in dataset["rows"]:
        assert row["responses"][row["expected_action"]] == row["target"]


def queries_for(row: dict) -> list[dict]:
    queries = [
        {
            "id": row["id"] + ":action",
            "text": row["prompt"],
            "labels": ["A", "B"],
            "expected": row["expected_action"],
        }
    ]
    first, second = row["concepts"]
    for action in ("A", "B"):
        subject = "receiver choose" if row["frame"] == "strategic" else "device output"
        text = (
            f"{row['scenario']}\nUnder option {action} ({row['surfaces'][action]}), "
            f"what would the {subject}? Options: X={first}; Y={second}. "
            "Return only X or Y.\nAnswer:"
        )
        queries.append(
            {
                "id": row["id"] + ":report_" + action,
                "text": text,
                "labels": ["X", "Y"],
                "expected": "X" if row["responses"][action] == first else "Y",
            }
        )
    return queries


def margin(row: dict, score: dict) -> float:
    correct = row["expected_action"]
    logits = dict(zip(("A", "B"), score["logits"], strict=True))
    return logits[correct] - logits["B" if correct == "A" else "A"]


def summarize_contrast(rows: list[dict], scores: dict, left, right) -> dict:
    """Equal-family paired differences; deterministic family bootstrap interval."""
    groups = defaultdict(lambda: [[], []])
    for row in rows:
        score = scores[row["id"] + ":action"]
        for i, predicate in enumerate((left, right)):
            if predicate(row):
                groups[row["family"]][i].append(
                    (float(score["choice"] == row["expected_action"]), margin(row, score))
                )
    if any(not a or not b for _, (a, b) in sorted(groups.items())) or not groups:
        raise ValueError("incomplete paired contrast")
    differences = np.array(
        [np.mean(a, axis=0) - np.mean(b, axis=0) for _, (a, b) in sorted(groups.items())]
    )
    draws = np.random.default_rng(101126).integers(
        0, len(differences), size=(4000, len(differences))
    )
    intervals = np.quantile(differences[draws].mean(axis=1), [0.025, 0.975], axis=0)
    return {
        "families": len(groups),
        "accuracy_difference": float(differences[:, 0].mean()),
        "margin_difference": float(differences[:, 1].mean()),
        "accuracy_ci95": intervals[:, 0].tolist(),
        "margin_ci95": intervals[:, 1].tolist(),
        "family_differences": dict(zip(sorted(groups), differences.tolist(), strict=True)),
    }


def analyze(dataset: dict, result: dict) -> dict:
    verify_dataset(dataset)
    if result.get("status") != "scored" or result["dataset_sha256"] != dataset["sha256"]:
        raise ValueError("unscored or mismatched result")
    rows, scores = dataset["rows"], result["scores"]
    expected_ids = {q["id"] for r in rows for q in queries_for(r)}
    if set(scores) != expected_ids:
        raise ValueError("missing or extra scores")
    cells = defaultdict(list)
    for row in rows:
        key = "/".join(
            map(
                str,
                [row[k] for k in ("frame", "surface", "history", "state", "layout", "binding")],
            )
        )
        cells[key].append(row)
    cell_summary = {}
    for key, group in sorted(cells.items()):
        reports = [
            scores[r["id"] + ":report_" + a]["choice"]
            == queries_for(r)[1 if a == "A" else 2]["expected"]
            for r in group
            for a in ("A", "B")
        ]
        cell_summary[key] = {
            "rows": len(group),
            "accuracy": float(
                np.mean(
                    [
                        scores[r["id"] + ":action"]["choice"] == r["expected_action"]
                        for r in group
                    ]
                )
            ),
            "mean_margin": float(
                np.mean([margin(r, scores[r["id"] + ":action"]) for r in group])
            ),
            "report_accuracy": float(np.mean(reports)),
        }
    contrasts = {}
    for surface in ("assertion", "opaque"):

        def common(r, surface=surface):
            return (
                r["frame"] == "strategic"
                and r["surface"] == surface
                and r["history"] == "redundant"
                and r["layout"] == "prose"
                and not r["binding"]
            )

        for state in ("matching", "absent", "neutral"):
            contrasts[f"{surface}:{state}-opposed"] = summarize_contrast(
                rows,
                scores,
                lambda r, common=common, state=state: common(r) and r["state"] == state,
                lambda r, common=common: common(r) and r["state"] == "opposed",
            )
        for layout in ("prose", "table"):

            def binding_cell(r, bound, surface=surface, layout=layout):
                return (
                    r["frame"] == "strategic"
                    and r["surface"] == surface
                    and r["history"] == "redundant"
                    and r["state"] == "opposed"
                    and r["layout"] == layout
                    and r["binding"] == bound
                )

            contrasts[f"{surface}:{layout}:bound-unbound"] = summarize_contrast(
                rows,
                scores,
                lambda r, cell=binding_cell: cell(r, True),
                lambda r, cell=binding_cell: cell(r, False),
            )
        contrasts[f"{surface}:unbound:table-prose"] = summarize_contrast(
            rows,
            scores,
            lambda r, surface=surface: (
                r["frame"] == "strategic"
                and r["surface"] == surface
                and r["history"] == "redundant"
                and r["state"] == "opposed"
                and not r["binding"]
                and r["layout"] == "table"
            ),
            lambda r, surface=surface: (
                r["frame"] == "strategic"
                and r["surface"] == surface
                and r["history"] == "redundant"
                and r["state"] == "opposed"
                and not r["binding"]
                and r["layout"] == "prose"
            ),
        )
    interaction = {
        metric: contrasts["assertion:matching-opposed"][metric]
        - contrasts["opaque:matching-opposed"][metric]
        for metric in ("accuracy_difference", "margin_difference")
    }
    opaque = [r for r in rows if r["surface"] == "opaque"]
    opaque_accuracy = float(
        np.mean([scores[r["id"] + ":action"]["choice"] == r["expected_action"] for r in opaque])
    )
    reports = [
        scores[q["id"]]["choice"] == q["expected"] for r in rows for q in queries_for(r)[1:]
    ]
    return {
        "status": "discovery_only" if dataset["split"] == "discovery" else "held_out",
        "dataset_sha256": dataset["sha256"],
        "rows": len(rows),
        "independent_lexical_families": 12,
        "cells": cell_summary,
        "contrasts": contrasts,
        "state_surface_interaction": interaction,
        "opaque_accuracy": opaque_accuracy,
        "report_accuracy": float(np.mean(reports)),
        "competence_gate": opaque_accuracy >= 0.95 and float(np.mean(reports)) >= 0.90,
    }


def select_mechanistic_contrast(analysis: dict) -> str | None:
    if not analysis["competence_gate"]:
        return None
    contrasts = analysis["contrasts"]
    if (
        analysis["state_surface_interaction"]["accuracy_difference"] >= 0.15
        and contrasts["assertion:matching-opposed"]["margin_difference"] >= 0.5
    ):
        return "state"
    if (
        contrasts["assertion:prose:bound-unbound"]["accuracy_difference"] >= 0.15
        and contrasts["assertion:prose:bound-unbound"]["margin_difference"] >= 0.5
    ):
        return "binding"
    return None


def paired_rows(dataset: dict, contrast: str) -> list[tuple[dict, dict]]:
    """Source/destination pairs before filtering on any behavioral outcome."""
    if contrast not in ("state", "binding"):
        raise ValueError("unknown contrast")
    rows = [
        r
        for r in dataset["rows"]
        if r["frame"] == "strategic"
        and r["surface"] == "assertion"
        and r["history"] == "redundant"
        and r["layout"] == "prose"
    ]
    pairs = defaultdict(dict)
    for row in rows:
        if (
            contrast == "state"
            and not row["binding"]
            and row["state"] in ("matching", "opposed")
        ):
            pairs[(row["family"], row["target_index"], row["permutation"])][row["state"]] = row
        if contrast == "binding" and row["state"] == "opposed":
            pairs[(row["family"], row["target_index"], row["permutation"])][row["binding"]] = (
                row
            )
    return [
        (p["matching"], p["opposed"]) if contrast == "state" else (p[True], p[False])
        for _, p in sorted(pairs.items())
    ]
