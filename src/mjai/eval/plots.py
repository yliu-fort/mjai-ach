"""Plot helpers for the one-click notebook (AGENTS.md Fig 1 + Fig 2 + diagnostics).

Layered output per AGENTS.md §6:
  - "Core" figures (论文 Fig 1 BRPS trajectory, Fig 2 exploitability curves, the
    final 2x2 comparison table) are returned as matplotlib Figure objects so the
    notebook can inline them.
  - "Detail" figures (per-game cross-play heatmaps, per-game training curves,
    forgetting curves) are written to ``runs/<cell>/plots/`` and the path is
    returned; the notebook shows the path as a link.

All figures are also saved to disk; nothing is lost. matplotlib is the backend
(static PNG, no JS, no new heavy deps).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _f(row: dict[str, object], key: str) -> float:
    """Typed extractor: pull a float field from a JSON-loaded curve row."""
    return float(row[key])  # type: ignore


def _i(row: dict[str, object], key: str) -> int:
    """Typed extractor: pull an int field from a JSON-loaded curve row."""
    return int(row[key])  # type: ignore


def _f_typed(row: dict[str, object], key: str) -> float:
    """Like :func:`_f` but for a runtime-computed key (e.g. chosen metric)."""
    return float(row[key])  # type: ignore


def _i_typed(row: dict[str, object], key: str) -> int:
    """Like :func:`_i` but for a runtime-computed key."""
    return int(row[key])  # type: ignore


def _save_fig(fig: Any, save_path: str | Path | None) -> None:
    """Save a figure, creating parent dirs first. No-op when save_path is None."""
    if save_path is None:
        return
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=110, bbox_inches="tight")


def _axes(ax: Any, figsize: tuple[float, float]) -> tuple[Any, Any, bool]:
    """Resolve the (figure, axes, owned) triple for a plot helper.

    ``ax=None`` means "make your own figure" (the standalone call the notebook
    saves to disk). Passing an axes means the CALLER owns the figure and is
    composing a grid, so the helper must not lay out or close it — a helper
    that always calls ``plt.subplots`` cannot be composed, and a notebook that
    tries anyway ends up flushing an empty grid alongside the real plots.
    """
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax.figure, ax, False
    fig, own_ax = plt.subplots(figsize=figsize)
    return fig, own_ax, True


# matplotlib imported lazily inside each plot function so importing this module
# is cheap and tests don't need a display backend configured.


# -----------------------------------------------------------------------------
# Core figure 1: BRPS policy trajectory (论文 Fig 1 reproduction).
# -----------------------------------------------------------------------------


def plot_brps_trajectory(
    curve_rows: list[dict[str, object]],
    *,
    title: str = "Biased-RPS policy trajectory",
    save_path: str | Path | None = None,
    ax: Any = None,
) -> Any:
    """Plot P(R)/P(P)/P(S) over training steps with the Nash levels dashed.

    Pass ``ax`` to draw one cell into a caller-owned grid (the notebook's
    PPO-vs-ACH side-by-side panel); omit it for a standalone figure to save.
    """
    steps = [_i(r, "step") for r in curve_rows if "brps/P_R" in r]
    p_r = [_f(r, "brps/P_R") for r in curve_rows if "brps/P_R" in r]
    p_p = [_f(r, "brps/P_P") for r in curve_rows if "brps/P_P" in r]
    p_s = [_f(r, "brps/P_S") for r in curve_rows if "brps/P_S" in r]

    fig, ax, owned = _axes(ax, (7, 4.5))
    if steps:
        ax.plot(steps, p_r, label="P(Rock)", color="tab:red")
        ax.plot(steps, p_p, label="P(Paper)", color="tab:blue")
        ax.plot(steps, p_s, label="P(Scissors)", color="tab:green")
    # Nash equilibrium horizontal lines: (1/16, 10/16, 5/16).
    for val, color, _name in [
        (1 / 16, "tab:red", "Nash R = 1/16"),
        (10 / 16, "tab:blue", "Nash P = 10/16"),
        (5 / 16, "tab:green", "Nash S = 5/16"),
    ]:
        ax.axhline(val, color=color, linestyle="--", alpha=0.4, lw=1)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Action probability")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    if owned:
        fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


# -----------------------------------------------------------------------------
# Core figure 2: exploitability / NashConv training curves (论文 Fig 2).
# -----------------------------------------------------------------------------


def plot_equilibrium_curves(
    curves: dict[str, list[dict[str, object]]],
    *,
    metric_key: str = "eval/exploitability",
    fallback_keys: tuple[str, ...] = ("eval/nash_conv", "eval/exact_nash_distance"),
    title: str = "Equilibrium distance over training",
    log_y: bool = True,
    save_path: str | Path | None = None,
    ax: Any = None,
) -> Any:
    """Plot the equilibrium metric vs step, one line per (algo, mode) cell.

    Args:
        curves: ``{cell_label: curve_rows}`` — one entry per experiment cell.
            The label becomes the legend entry (e.g. ``"ACH / mirror"``).
        metric_key: preferred metric; falls back through ``fallback_keys`` per
            cell if absent.
        ax: draw into this axes (one panel of a caller-owned per-game grid)
            instead of making a standalone figure.
    """
    fig, ax, owned = _axes(ax, (7, 4.5))
    for label, rows in curves.items():
        # Pick the first available metric.
        keys = (metric_key, *fallback_keys)
        used_key = next((k for k in keys if rows and k in rows[-1]), None)
        if used_key is None:
            continue
        xs = [_i(r, "step") for r in rows if used_key in r]
        ys = [_f_typed(r, used_key) for r in rows if used_key in r]
        ax.plot(xs, ys, label=f"{label} ({used_key.removeprefix('eval/')})", lw=1.8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Equilibrium distance (lower = closer to Nash)")
    if log_y:
        ax.set_yscale("symlog")
        ax.set_ylabel("Equilibrium distance (symlog)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    if owned:
        fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


# -----------------------------------------------------------------------------
# Core figure 3: final 2x2 comparison (per game, bar chart).
# -----------------------------------------------------------------------------


def plot_final_metric_bars(
    results: dict[tuple[str, str, str], float],
    *,
    games: list[str],
    metric_name: str = "exploitability",
    save_path: str | Path | None = None,
) -> Any:
    """Grouped bar chart: per game, 4 bars = {PPO, ACH} x {mirror, league}.

    Args:
        results: ``{(game, algo, mode): metric_value}`` for the final step.
    """
    import matplotlib.pyplot as plt

    cell_keys = [("ppo", "mirror"), ("ach", "mirror"), ("ppo", "league"), ("ach", "league")]
    cell_labels = ["PPO/mirror", "ACH/mirror", "PPO/league", "ACH/league"]
    n_games = len(games)
    n_cells = len(cell_keys)
    width = 0.8 / n_cells
    x = np.arange(n_games)

    fig, ax = plt.subplots(figsize=(max(7, 1.5 * n_games), 4.5))
    for i, (algo, mode) in enumerate(cell_keys):
        vals = [results.get((g, algo, mode), float("nan")) for g in games]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=cell_labels[i])
    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=20, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Final {metric_name} per cell (lower = closer to Nash)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


# -----------------------------------------------------------------------------
# Detail figures: per-game cross-play heatmap + forgetting curve.
# Written to runs/<cell>/plots/, path returned.
# -----------------------------------------------------------------------------


def plot_crossplay_heatmap(
    payoff: np.ndarray,
    policy_names: list[str],
    *,
    title: str,
    save_path: str | Path,
) -> str:
    """Render a cross-play payoff matrix as a heatmap; save and return path."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(max(4, 0.5 * len(policy_names)), max(3, 0.5 * len(policy_names)))
    )
    im = ax.imshow(
        payoff, cmap="RdBu", vmin=float(-abs(payoff).max()), vmax=float(abs(payoff).max())
    )
    ax.set_xticks(range(len(policy_names)))
    ax.set_yticks(range(len(policy_names)))
    ax.set_xticklabels(policy_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(policy_names, fontsize=7)
    ax.set_xlabel("opponent (seat 1)")
    ax.set_ylabel("policy (seat 0)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save_fig(fig, save_path)
    plt.close(fig)
    return str(save_path)


def plot_forgetting_curve(
    win_rate_row: list[float],
    checkpoint_names: list[str],
    *,
    title: str,
    save_path: str | Path,
) -> str:
    """Final policy's win rate vs each checkpoint; save and return path."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(win_rate_row)), win_rate_row, color="tab:purple")
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.5, label="50% (no edge)")
    ax.set_xticks(range(len(checkpoint_names)))
    ax.set_xticklabels(checkpoint_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Final policy win-rate")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, save_path)
    plt.close(fig)
    return str(save_path)


def plot_nontransitivity_over_training(
    curves: dict[str, list[dict[str, object]]],
    *,
    save_path: str | Path | None = None,
    ax: Any = None,
) -> Any:
    """Nontransitivity per cell over training (detail). Returns a Figure."""
    fig, ax, owned = _axes(ax, (7, 4))
    for label, rows in curves.items():
        xs = [_i(r, "step") for r in rows if "nontransitivity" in r]
        ys = [_f(r, "nontransitivity") for r in rows if "nontransitivity" in r]
        if xs:
            ax.plot(xs, ys, label=label, lw=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Nontransitivity (spectral)")
    ax.set_title("Non-transitivity over training (higher = more RPS-like cycling)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    if owned:
        fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------


def load_train_curve(run_dir: str | Path) -> list[dict[str, object]]:
    """Load a run's ``train_curve.json`` (empty list if absent)."""
    p = Path(run_dir) / "train_curve.json"
    if not p.is_file():
        return []
    data: list[dict[str, object]] = json.loads(p.read_text(encoding="utf-8"))
    return data


def cell_label(algo: str, mode: str) -> str:
    """Legend label for a cell."""
    return f"{algo.upper()} / {mode}"


def safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default
