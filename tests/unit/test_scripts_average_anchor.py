"""End-to-end wiring of the average-strategy anchor (AGENTS.md §5, D16).

The tracker itself is tested in ``test_eval_average_policy.py``. What is at
stake here is that a run configured to produce the anchor actually produces it,
that the curve is monotone-ish in the way the O(T^-1/2) bound implies, and that
the feature stays off — bit for bit off — unless asked for. The last point
matters because every committed result artifact in ``docs/`` was produced
without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mjai.scripts.experiment import ExperimentConfig, run_experiment


def _run(tmp_path: Path, **overrides: object) -> list[dict[str, object]]:
    cfg = ExperimentConfig(
        game="kuhn",
        algo="ach",
        self_play_mode="mirror",
        policy_kind="tabular",
        n_steps=20,
        episodes_per_round=20,
        eval_during_training=True,
        eval_every_steps=5,
        save_every_steps=1000,
        seed=0,
        verbose=False,
        out_dir=str(tmp_path / "run"),
        **overrides,  # type: ignore[arg-type]
    )
    out = run_experiment(cfg)
    return json.loads((out / "train_curve.json").read_text())


def test_anchor_columns_appear_when_enabled(tmp_path):
    rows = _run(tmp_path, track_average_policy=True)
    assert rows
    for row in rows:
        assert "eval/avg_nash_conv" in row
        assert "eval/avg_exploitability" in row  # 2 players, so the identity holds
    # One iterate folded in per eval point, in order.
    assert [row["eval/avg_iterates"] for row in rows] == [float(i + 1) for i in range(len(rows))]


def test_anchor_is_absent_by_default(tmp_path):
    """Off unless asked: every existing result artifact was made without it."""
    rows = _run(tmp_path)
    assert rows
    assert all("eval/avg_nash_conv" not in row for row in rows)


def test_anchor_curve_decreases(tmp_path):
    """The point of the anchor: a known-shape curve to sanity-check against.

    Tabular ACH is a CFR+ wrapper (D4/D5), so its average strategy is the CFR
    average and its NashConv must fall. Asserting the endpoints rather than
    monotonicity at every point keeps this robust to a short run's noise while
    still failing if the tracker is accumulating garbage.
    """
    rows = _run(tmp_path, track_average_policy=True)
    first = float(rows[0]["eval/avg_nash_conv"])  # type: ignore[arg-type]
    last = float(rows[-1]["eval/avg_nash_conv"])  # type: ignore[arg-type]
    assert last < first


def test_linear_and_uniform_weighting_give_different_curves(tmp_path):
    """They are different objects, and the config must actually select between
    them — a knob that silently does nothing is worse than no knob."""
    uniform = _run(tmp_path / "u", track_average_policy=True)
    linear = _run(tmp_path / "l", track_average_policy=True, average_policy_weighting="linear")
    assert float(uniform[-1]["eval/avg_nash_conv"]) != pytest.approx(  # type: ignore[arg-type]
        float(linear[-1]["eval/avg_nash_conv"]),
        abs=1e-12,  # type: ignore[arg-type]
    )


def test_bad_weighting_fails_at_config_time(tmp_path):
    with pytest.raises(ValueError, match="average_policy_weighting"):
        ExperimentConfig(
            game="kuhn",
            algo="ach",
            self_play_mode="mirror",
            average_policy_weighting="geometric",
            out_dir=str(tmp_path),
        )


def test_unsupported_game_fails_at_startup_not_mid_run(tmp_path):
    """A run that asked for the anchor and silently did not get it would be
    worse than one that stopped (AGENTS.md §11). BRPS is simultaneous, so it
    has no sequence form at all."""
    cfg = ExperimentConfig(
        game="brps",
        algo="ach",
        self_play_mode="mirror",
        policy_kind="tabular",
        n_steps=2,
        episodes_per_round=2,
        seed=0,
        verbose=False,
        out_dir=str(tmp_path / "run"),
        track_average_policy=True,
    )
    with pytest.raises(ValueError, match="simultaneous"):
        run_experiment(cfg)
