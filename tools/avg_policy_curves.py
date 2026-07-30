"""Plot ACH average-policy NC vs current-policy NC (AGENTS.md D16).

Reads the ``eval/avg_*`` curves a ``track_average_policy: true`` run emits and
plots, per game, the current-policy NashConv against the running-average
strategy's NashConv in both weightings:

  * ``eval/nash_conv``           — current policy ``pi = softmax(y)`` (the object
    ``docs/reproduce_report.md`` plots; the paper's figure choice).
  * ``eval/avg_nash_conv``       — uniform average (the object Theorem 1 bounds).
  * ``eval/avg_nash_conv_lin``   — linear / CFR+ average (weight = t).

The question this answers: how much smaller is the average-policy NashConv than
the current-policy one, and is it the 1-2 orders of magnitude the CFR intuition
predicts? ACH's average is NOT covered by Theorem 1 (biased ``y``, entropy
regularization), so the gap is empirical — this tool reports it, it does not
prove anything.

Usage::

    uv run python tools/avg_policy_curves.py                 # all 3 games
    uv run python tools/avg_policy_curves.py --root runs/avg_anchor
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAMES = ["kuhn", "leduc", "liars_dice1"]
LABELS = {
    "kuhn": "Kuhn poker",
    "leduc": "Leduc poker",
    "liars_dice1": "Liar's Dice (1 die)",
}
# Exploitability = NashConv / |P| (2 players here). This is the unit the paper
# (Fig 10 y-axis), docs/reproduce_report.md and tools/compare_with_paper.py use;
# eval/nash_conv is 2x this and must not be plotted against paper numbers.
CURRENT = "eval/exploitability"
UNIFORM = "eval/avg_exploitability"
LINEAR = "eval/avg_exploitability_lin"
FLOOR = 1e-12  # log-scale guard; ACH never hits exact 0 but the average can get small


def _series(rows: list[dict], key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for r in rows:
        if key not in r:
            continue
        v = float(r[key])
        xs.append(float(r["env_steps"]))
        ys.append(max(v, FLOOR))
    return xs, ys


def _tail_mean(rows: list[dict], key: str, frac: float = 0.1) -> float:
    vals = [float(r[key]) for r in rows if key in r]
    if not vals:
        return float("nan")
    k = max(1, int(len(vals) * frac))
    return sum(vals[-k:]) / k


def _orders(a: float, b: float) -> float:
    """How many orders of magnitude a is below b (negative if a is larger)."""
    if a <= 0 or b <= 0:
        return float("nan")
    return math.log10(b / a)


def load_game(root: Path, game: str) -> list[dict] | None:
    f = root / f"{game}_seed0" / "train_curve.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="runs/avg_anchor", help="Run directory root.")
    p.add_argument("--out", default="docs/figs/avg_vs_current.png", help="Output figure.")
    args = p.parse_args(argv)
    root = Path(args.root)
    out = Path(args.out)

    fig, axes = plt.subplots(1, len(GAMES), figsize=(5 * len(GAMES), 4), sharey=False)
    print(
        f"\n{'game':<14}{'current':>10}{'uniform-avg':>13}{'linear-avg':>12}"
        f"{'cur/uniform':>13}{'cur/linear':>12}"
    )
    print("-" * 74)
    summary: list[dict[str, object]] = []
    for ax, game in zip(axes, GAMES, strict=True):
        rows = load_game(root, game)
        if rows is None:
            ax.set_title(f"{LABELS[game]}\n(not found)")
            ax.text(0.5, 0.5, "no run", ha="center", va="center", transform=ax.transAxes)
            continue
        for key, lbl, style in [
            (CURRENT, "current policy", "-"),
            (UNIFORM, "uniform average (Thm 1)", "--"),
            (LINEAR, "linear average (CFR+)", ":"),
        ]:
            xs, ys = _series(rows, key)
            if xs:
                ax.plot(xs, ys, style, lw=1.6, label=lbl)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("env-steps")
        ax.set_ylabel("exploitability")
        ax.set_title(LABELS[game])
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        cur = _tail_mean(rows, CURRENT)
        uni = _tail_mean(rows, UNIFORM)
        lin = _tail_mean(rows, LINEAR)
        print(
            f"{game:<14}{cur:>10.4g}{uni:>13.4g}{lin:>12.4g}"
            f"{_orders(uni, cur):>12.2f}O{_orders(lin, cur):>11.2f}O"
        )
        summary.append(
            {
                "game": game,
                "current": cur,
                "uniform": uni,
                "linear": lin,
                "orders_uniform": _orders(uni, cur),
                "orders_linear": _orders(lin, cur),
                "n_eval_points": len(rows),
            }
        )

    fig.suptitle(
        "ACH: average-policy exploitability vs current-policy exploitability "
        "(seed 0, 1e7 env-steps)",
        fontsize=12,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nsaved: {out}")
    print("(columns: tail-10% mean EXPLOITABILITY; 'O' = orders of magnitude avg is below current)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
