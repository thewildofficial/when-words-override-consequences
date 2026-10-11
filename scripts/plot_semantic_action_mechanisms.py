"""Plot the completed matched behavioral screen with lexical-family intervals."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root
    with gzip.open(root / "prepare_raw.json.gz", "rt") as stream:
        prepared = json.load(stream)["discovery"]
    with gzip.open(root / "behavior_raw.json.gz", "rt") as stream:
        result = json.load(stream)
    from jspace_policy.semantic_action_mechanisms import analyze

    analyze(prepared["dataset"], result)  # Require complete, provenance-matched scores.
    rows, scores = prepared["dataset"]["rows"], result["scores"]
    output = root / "behavior_mechanisms.png"
    if output.exists():
        raise RuntimeError("refusing to overwrite figure")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), sharey=True)
    for panel, axis in enumerate(axes):
        conditions = [(s, "prose", False) for s in ("matching", "opposed", "absent", "neutral")]
        if panel:
            conditions = [
                ("opposed", layout, bound)
                for layout in ("prose", "table")
                for bound in (False, True)
            ]
        for offset, surface, color, label in (
            (-0.19, "assertion", "#BF643A", "Assertion actions"),
            (0.19, "opaque", "#356984", "Opaque actions"),
        ):
            means, low, high = [], [], []
            for state, layout, binding in conditions:
                cell = [
                    r
                    for r in rows
                    if r["frame"] == "strategic"
                    and r["surface"] == surface
                    and r["history"] == "redundant"
                    and (r["state"], r["layout"], r["binding"]) == (state, layout, binding)
                ]
                values = np.array(
                    [
                        np.mean(
                            [
                                scores[r["id"] + ":action"]["choice"] == r["expected_action"]
                                for r in cell
                                if r["family"] == family
                            ]
                        )
                        for family in sorted({r["family"] for r in cell})
                    ]
                )
                mean = 100 * values.mean()
                draws = (
                    np.random.default_rng(101126).choice(values, (4000, len(values))).mean(1)
                )
                lower, upper = 100 * np.quantile(draws, [0.025, 0.975])
                means.append(mean)
                low.append(mean - lower)
                high.append(upper - mean)
            bars = axis.bar(
                np.arange(4) + offset,
                means,
                0.34,
                color=color,
                label=label,
                yerr=[low, high],
                capsize=3,
                error_kw={"linewidth": 1},
            )
            axis.bar_label(bars, labels=[f"{v:.1f}%" for v in means], padding=5, fontsize=8)
        axis.set_ylim(0, 113)
        axis.set_yticks(range(0, 101, 20))
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xticks(range(4))
        axis.set_xticklabels(
            ["State matches\ngoal", "State opposes\ngoal", "State omitted", "Neutral\nsentence"]
            if not panel
            else [
                "Prose\nwithout A/B",
                "Prose\nwith A/B",
                "Table\nwithout A/B",
                "Table\nwith A/B",
            ],
            fontsize=9,
        )
        axis.set_title(
            "Changing an irrelevant state" if not panel else "Layout and label binding", pad=18
        )
    axes[0].set_ylabel("Correct actions (%)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, frameon=False)
    fig.suptitle(
        "Semantic action errors depend on state cues and representation", y=0.98, fontsize=14
    )
    fig.text(
        0.5,
        0.035,
        "Discovery: 12 lexical families, 48 conditions per cell. "
        "Whiskers: family-bootstrap 95% intervals.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(output, dpi=180, facecolor="white")
    print(output.resolve())


if __name__ == "__main__":
    main()
