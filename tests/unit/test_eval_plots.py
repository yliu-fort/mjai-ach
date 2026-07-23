"""Unit tests for the plot helpers (AGENTS.md §5). Matplotlib uses Agg backend."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mjai.eval.plots import (
    cell_label,
    load_train_curve,
    plot_brps_trajectory,
    plot_crossplay_heatmap,
    plot_equilibrium_curves,
    plot_final_metric_bars,
    plot_forgetting_curve,
    safe_float,
)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _brps_curve(n=10):
    return [
        {
            "step": i * 10,
            "brps/P_R": 0.33 + 0.01 * i,
            "brps/P_P": 0.33,
            "brps/P_S": 0.34 - 0.01 * i,
            "brps/nash_distance": 0.5 - 0.04 * i,
            "eval/exact_nash_distance": 0.5 - 0.04 * i,
        }
        for i in range(n)
    ]


def test_brps_trajectory_returns_figure(tmp_path):
    fig = plot_brps_trajectory(_brps_curve(), save_path=tmp_path / "brps.png")
    assert isinstance(fig, plt.Figure)
    assert (tmp_path / "brps.png").is_file()
    plt.close(fig)


def test_brps_trajectory_handles_empty_curve(tmp_path):
    # Empty curve => no labeled artists; matplotlib emits a UserWarning that
    # our strict filterwarnings would otherwise turn into an error. Catch it.
    with pytest.warns(UserWarning, match="No artists"):
        fig = plot_brps_trajectory([], save_path=tmp_path / "empty.png")
    # Still produces a valid figure with the Nash lines (no crash).
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_equilibrium_curves_overlays_multiple_cells(tmp_path):
    curves = {"ACH/mirror": _brps_curve(5), "PPO/mirror": _brps_curve(5)[::-1]}
    fig = plot_equilibrium_curves(curves, save_path=tmp_path / "eq.png")
    assert isinstance(fig, plt.Figure)
    # Legend should have 2 entries (one per cell).
    assert len(fig.axes[0].get_legend_handles_labels()[1]) == 2
    plt.close(fig)


def test_equilibrium_curves_fallback_metric():
    """When the preferred metric is missing, falls back through the list."""
    rows = [{"step": 0, "eval/nash_conv": 5.0}, {"step": 1, "eval/nash_conv": 4.0}]
    fig = plot_equilibrium_curves({"x": rows}, metric_key="eval/exploitability")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_helpers_draw_into_a_caller_supplied_axes():
    """``ax=`` composes into the caller's grid instead of making a new figure.

    The notebook's side-by-side panels depend on this: a helper that always
    calls plt.subplots forces the notebook to flush the empty grid as its own
    image, which is exactly the bug this parameter removes.
    """
    fig, axes = plt.subplots(1, 2)
    before = set(plt.get_fignums())
    out_a = plot_brps_trajectory(_brps_curve(4), ax=axes[0])
    out_b = plot_equilibrium_curves({"ACH/mirror": _brps_curve(4)}, ax=axes[1])
    assert out_a is fig and out_b is fig  # no new figure was created
    assert set(plt.get_fignums()) == before
    assert axes[0].get_lines() and axes[1].get_lines()  # both panels got content
    plt.close(fig)


def test_final_metric_bars_returns_figure(tmp_path):
    results = {
        ("brps", "ppo", "mirror"): 0.12,
        ("brps", "ach", "mirror"): 0.08,
        ("brps", "ppo", "league"): 0.11,
        ("brps", "ach", "league"): 0.07,
        ("kuhn", "ppo", "mirror"): 0.04,
    }
    fig = plot_final_metric_bars(results, games=["brps", "kuhn"], save_path=tmp_path / "bars.png")
    assert isinstance(fig, plt.Figure)
    assert (tmp_path / "bars.png").is_file()
    plt.close(fig)


def test_crossplay_heatmap_saves_and_returns_path(tmp_path):
    payoff = np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])
    names = ["s0", "s1", "s2"]
    out = plot_crossplay_heatmap(payoff, names, title="t", save_path=tmp_path / "hm.png")
    assert out == str(tmp_path / "hm.png")
    assert (tmp_path / "hm.png").is_file()


def test_forgetting_curve_saves(tmp_path):
    out = plot_forgetting_curve(
        [0.8, 0.6, 0.4, 0.3], ["s0", "s1", "s2", "s3"], title="t", save_path=tmp_path / "fg.png"
    )
    assert (tmp_path / "fg.png").is_file()
    assert out == str(tmp_path / "fg.png")


def test_load_train_curve_returns_empty_for_missing(tmp_path):
    assert load_train_curve(tmp_path) == []


def test_load_train_curve_roundtrips(tmp_path):
    rows = [{"step": 1, "eval/exploitability": 0.5}]
    (tmp_path / "train_curve.json").write_text(json.dumps(rows))
    assert load_train_curve(tmp_path) == rows


def test_cell_label():
    assert cell_label("ach", "mirror") == "ACH / mirror"
    assert cell_label("ppo", "league") == "PPO / league"


def test_safe_float_handles_garbage():
    assert safe_float(0.5) == 0.5
    assert math.isnan(safe_float(None))
    assert math.isnan(safe_float("nope"))
    assert safe_float(float("inf")) != float("inf")  # replaced with nan


# Used in the last assertion; imported at end to keep top clean.
import math  # noqa: E402
