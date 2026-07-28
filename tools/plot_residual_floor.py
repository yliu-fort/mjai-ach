"""Two figures for ``docs/liars_residual_floor.md``.

Left: the same ACH update rule at matched update counts, under the on-policy
reach weighting versus a flat one, against the real RL run. Right: how much of
the exploitable regret sits in the information sets training barely visits,
per game.

Curves come from the exact-dynamics logs (the reach arms were stopped once they
had delivered their comparison points, so parsing the log rather than the
per-arm JSON is deliberate).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path("runs/exact_ach")
FIGS = Path("docs/figs")

_LINE = re.compile(r"iter\s+(\d+)\s+expl\s+([0-9.]+)")


def curve_from_log(path: Path) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _LINE.search(line)
        if m:
            xs.append(int(m.group(1)))
            ys.append(float(m.group(2)))
    return xs, ys


def rl_curve() -> tuple[list[int], list[float]]:
    data = json.loads(
        Path("runs/ab_fix/liars_fix_lth2_iw20_cap256_seed0/train_curve.json").read_text()
    )
    return [r["step"] for r in data], [r["eval/exploitability"] for r in data]


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    arms = [
        ("liars_uniform.log", "exact ACH, flat weighting (tabular)", "tab:green", "-"),
        ("liars_paper.log", "exact ACH, on-policy reach weighting (tabular)", "tab:red", "-"),
        ("liars_paper_lr0.1.log", "same, learning rate x100", "tab:orange", "--"),
    ]
    for name, label, color, style in arms:
        path = RUNS / name
        if not path.is_file():
            continue
        xs, ys = curve_from_log(path)
        ax1.plot(xs, ys, style, color=color, label=label, lw=2, marker="o", ms=3)
    xs, ys = rl_curve()
    ax1.plot(xs, ys, "-", color="tab:blue", label="real RL run (MLP, sampled)", lw=2, alpha=0.8)
    ax1.axhline(0.1464, color="tab:blue", ls=":", lw=1)
    ax1.text(6e4, 0.153, "RL floor 0.146", color="tab:blue", fontsize=9)

    ax1.set_xlabel("policy updates")
    ax1.set_ylabel("exploitability")
    ax1.set_title("Liar's Dice: only the per-information-set weight differs")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8, loc="upper right")

    shares = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50]
    for game, color in (("kuhn", "tab:blue"), ("leduc", "tab:orange"), ("liars_best", "tab:red")):
        path = RUNS / f"reach_mismatch_{game}.json"
        if not path.is_file():
            continue
        d = json.loads(path.read_text())
        ys = [d["concentration"][f"regret_in_bottom_{s:g}_of_visits"] for s in shares]
        label = f"{game.replace('_best', '')} (mismatch {d['mismatch_ratio']:.2f}x)"
        ax2.plot([s * 100 for s in shares], [y * 100 for y in ys], "-o", color=color, label=label)

    ax2.set_xscale("log")
    ax2.set_xlabel("share of training visits (%), least-visited information sets first")
    ax2.set_ylabel("share of exploitable regret (%)")
    ax2.set_title("Where the exploitability hides")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=9)

    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "liars_residual_floor.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
