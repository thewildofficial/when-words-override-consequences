from __future__ import annotations

from pathlib import Path
from runpy import run_path

_ANALYSIS = run_path(
    Path(__file__).resolve().parents[1] / "scripts/analyze_v6_1_epistemic_repair.py"
)
_clustered_bootstrap = _ANALYSIS["_clustered_bootstrap"]
_paired_sign_flip = _ANALYSIS["_paired_sign_flip"]


def test_clustered_bootstrap_aggregates_cells_before_resampling() -> None:
    result = _clustered_bootstrap(
        [("game-a", 1.0), ("game-a", 0.0), ("game-b", 0.0)],
        seed=1,
        draws=200,
    )
    assert result["n_cells"] == 3
    assert result["n_clusters"] == 2
    assert result["mean"] == 0.25


def test_sign_flip_aggregates_paired_contrasts_by_game() -> None:
    result = _paired_sign_flip(
        [("game-a", 1.0), ("game-a", -1.0), ("game-b", 1.0)],
        seed=2,
        draws=200,
    )
    assert result["n_clusters"] == 2
    assert result["observed_mean"] == 0.5
