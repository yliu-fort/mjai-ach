"""Compare ACH reproduction runs against the digitized paper Fig 10 curves.

Reads ``eval/exploitability`` curves from completed (DONE-marked) reproduction
run dirs, aggregates seeds (mean / min / max on a common env-step grid),
overlays them with the digitized paper curves
(``docs/figs/fig10_ach_digitized.json``), and emits a per-game pass/fail
verdict under the pre-declared D5 criterion:

  pass  <=>  our final mean falls inside the paper's 8-run range at the final
             x, OR |our mean - paper mean| <= half the paper's range width.

"Final" values are averaged over the last 10% of the x-axis to reduce noise.
Read-only analysis tool (AGENTS.md D9: training metrics live in TensorBoard).

Usage (repo venv)::

    python tools/compare_with_paper.py \
        --root runs/reproduce --paper docs/figs/fig10_ach_digitized.json \
        --out-dir docs/figs --json docs/reproduce_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tb_eval import read_many

GAMES = ["kuhn", "liars_dice1", "leduc"]  # display order: smallest y-range first
GAME_TITLES = {
    "kuhn": "Kuhn poker",
    "leduc": "Leduc poker",
    "liars_dice1": "Liar's Dice",
}
# Digitized paper JSON keys (docs/figs/fig10_ach_digitized.json) per game dir name.
PAPER_KEYS = {"kuhn": "kuhn", "leduc": "leduc", "liars_dice1": "liars"}
GRID_POINTS = 200


def _interp(curve: list[tuple[int, float]], grid: np.ndarray) -> np.ndarray:
    x = np.array([p[0] for p in curve], dtype=float)
    y = np.array([p[1] for p in curve], dtype=float)
    order = np.argsort(x)
    return np.interp(grid, x[order], y[order], left=np.nan, right=np.nan)


def _band_stats(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.vstack(curves)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack, axis=0)
        lo = np.nanmin(stack, axis=0)
        hi = np.nanmax(stack, axis=0)
    return mean, lo, hi


def _final_stats(grid: np.ndarray, values: np.ndarray) -> float:
    tail = grid >= grid[-1] * 0.9
    tail_vals = values[tail & ~np.isnan(values)]
    return float(np.mean(tail_vals)) if tail_vals.size else float("nan")


def _verdict(
    grid: np.ndarray,
    om: np.ndarray,
    pm_g: np.ndarray,
    plo_g: np.ndarray,
    phi_g: np.ndarray,
) -> tuple[str, dict[str, Any]]:
    ours_final = _final_stats(grid, om)
    paper_final = _final_stats(grid, pm_g)
    paper_lo = _final_stats(grid, plo_g)
    paper_hi = _final_stats(grid, phi_g)
    half_width = (paper_hi - paper_lo) / 2.0
    inside = paper_lo <= ours_final <= paper_hi
    close = abs(ours_final - paper_final) <= half_width
    verdict = "pass" if (inside or close) else "fail"
    return verdict, {
        "ours_final": round(ours_final, 4),
        "paper_final_mean": round(paper_final, 4),
        "paper_final_range": [round(paper_lo, 4), round(paper_hi, 4)],
    }


def _plot_overlay(
    fig_path: Path,
    game: str,
    grid: np.ndarray,
    om: np.ndarray,
    olo: np.ndarray,
    ohi: np.ndarray,
    pm_g: np.ndarray,
    plo_g: np.ndarray,
    phi_g: np.ndarray,
    n: int,
    verdict: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.fill_between(
        grid / 1e7, plo_g, phi_g, color="tab:red", alpha=0.15, label="paper 8-run range"
    )
    ax.plot(grid / 1e7, pm_g, color="tab:red", ls="--", lw=1.5, label="paper ACH mean (Fig 10)")
    ax.fill_between(grid / 1e7, olo, ohi, color="tab:blue", alpha=0.15, label=f"ours range (n={n})")
    ax.plot(grid / 1e7, om, color="tab:blue", lw=1.8, label="ours ACH mean")
    ax.set_xlabel("Training Steps (x1e7)")
    ax.set_ylabel("Exploitability")
    ax.set_title(f"{GAME_TITLES[game]} — ACH reproduction vs paper  [{verdict}]")
    ax.set_xlim(0, grid[-1] / 1e7)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--root", default="runs/reproduce")
    parser.add_argument("--paper", default="docs/figs/fig10_ach_digitized.json")
    parser.add_argument("--out-dir", default="docs/figs")
    parser.add_argument("--json", default="docs/reproduce_comparison.json")
    args = parser.parse_args()

    root = Path(args.root)
    paper = json.loads(Path(args.paper).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {}
    for game in GAMES:
        seed_dirs = sorted(root.glob(f"{game}_ach_mlp_mirror/seed_*"))
        done_dirs = [sd for sd in seed_dirs if (sd / "DONE").exists()]
        curves_map = read_many([sd / "tb" for sd in done_dirs])
        curves_raw = [(sd.name, c) for sd in done_dirs if (c := curves_map.get(str(sd / "tb"), []))]
        if not curves_raw or PAPER_KEYS[game] not in paper:
            report[game] = {"status": "no data", "n_seeds": len(curves_raw)}
            continue

        x_max = max(c[-1][0] for _, c in curves_raw)
        grid = np.linspace(0.0, min(x_max, 1e7), GRID_POINTS)
        ours = [_interp(c, grid) for _, c in curves_raw]
        om, olo, ohi = _band_stats(ours)

        pkey = PAPER_KEYS[game]
        px = np.array(paper[pkey]["x"], dtype=float)
        pm = np.array(paper[pkey]["mean"], dtype=float)
        plo = np.array(paper[pkey]["lo"], dtype=float)
        phi = np.array(paper[pkey]["hi"], dtype=float)
        pm_g = np.interp(grid, px, pm, left=np.nan, right=np.nan)
        plo_g = np.interp(grid, px, plo, left=np.nan, right=np.nan)
        phi_g = np.interp(grid, px, phi, left=np.nan, right=np.nan)

        # --- verdict (D5): final = mean over last 10% of x ---
        verdict, verdict_stats = _verdict(grid, om, pm_g, plo_g, phi_g)

        # --- overlay plot ---
        fig_path = out_dir / f"compare_{game}.png"
        _plot_overlay(fig_path, game, grid, om, olo, ohi, pm_g, plo_g, phi_g, len(ours), verdict)

        report[game] = {
            "status": "ok",
            "n_seeds": len(ours),
            "x_max_env_steps": x_max,
            **verdict_stats,
            "verdict": verdict,
            "figure": str(fig_path),
        }

    out_json = Path(args.json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
