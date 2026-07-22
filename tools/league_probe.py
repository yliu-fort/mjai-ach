"""Mirror-vs-league A/B probe (F2): exploitability convergence comparison.

Arms: {kuhn, brps} x {mirror, league}, N seeds each, paper-faithful ACH
protocol loaded from configs/exp/<game>_ach_mlp_<mode>.yaml with the step
budget overridden. Each arm writes runs/league_probe/<game>_<mode>/seed_S
(+ a DONE marker); --summarize aggregates the TB eval curves into per-arm
mean/min/max bands (summary.json) and a two-panel figure.

Usage::

    python tools/league_probe.py --game kuhn --mode league --seed 0 \
        --total-env-steps 60000 --eval-every 5000
    python tools/league_probe.py --summarize
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tb_eval import read_many  # noqa: E402

from mjai.scripts.experiment import ExperimentConfig, run_experiment  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PROBE_ROOT = REPO / "runs" / "league_probe"
GAMES = ("kuhn", "brps")
MODES = ("mirror", "league")


def parse_seeds(spec: str) -> list[int]:
    """Parse "0-3", "0,2,5" or "3" into a sorted unique seed list."""
    seeds: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.update(range(int(lo), int(hi) + 1))
        else:
            seeds.add(int(part))
    if not seeds:
        raise ValueError(f"no seeds parsed from {spec!r}")
    return sorted(seeds)


def load_arm_config(game: str, mode: str) -> ExperimentConfig:
    """Load configs/exp/<game>_ach_mlp_<mode>.yaml (unknown keys fail loudly)."""
    path = REPO / "configs" / "exp" / f"{game}_ach_mlp_{mode}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no experiment config for arm {game}/{mode}: {path}")
    return ExperimentConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))


def arm_dir(root: Path, game: str, mode: str, seed: int) -> Path:
    return root / f"{game}_{mode}" / f"seed_{seed}"


def run_arm(
    game: str,
    mode: str,
    seed: int,
    *,
    total_env_steps: int,
    eval_every_env_steps: int,
    root: Path = PROBE_ROOT,
) -> Path:
    out = arm_dir(root, game, mode, seed)
    cfg = dataclasses.replace(
        load_arm_config(game, mode),
        seed=seed,
        out_dir=str(out),
        total_env_steps=total_env_steps,
        eval_every_env_steps=eval_every_env_steps,
        verbose=False,
    )
    run_experiment(cfg)
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    return out


def interp_forward(curve: list[tuple[int, float]], grid: list[int]) -> list[float | None]:
    """Last-value-carried-forward lookup; None before the curve's first point."""
    out: list[float | None] = []
    i = -1
    for g in grid:
        while i + 1 < len(curve) and curve[i + 1][0] <= g:
            i += 1
        out.append(curve[i][1] if i >= 0 else None)
    return out


def band(curves: list[list[tuple[int, float]]], grid: list[int]) -> dict[str, list]:
    """Per-grid-point mean/min/max across seeds (over available seeds only)."""
    cols = [interp_forward(c, grid) for c in curves]
    mean: list[float | None] = []
    lo: list[float | None] = []
    hi: list[float | None] = []
    n: list[int] = []
    for j in range(len(grid)):
        vals = [c[j] for c in cols if c[j] is not None]
        n.append(len(vals))
        if vals:
            mean.append(sum(vals) / len(vals))
            lo.append(min(vals))
            hi.append(max(vals))
        else:
            mean.append(None)
            lo.append(None)
            hi.append(None)
    return {"grid": grid, "mean": mean, "min": lo, "max": hi, "n_seeds": n}


def summarize(root: Path = PROBE_ROOT, tag: str = "eval/exploitability") -> dict[str, object]:
    """Aggregate all finished arms under root into bands + write summary.json."""
    tb_dirs = sorted(root.glob("*_*/seed_*/tb"))
    curves = read_many(tb_dirs, tag=tag)
    by_arm: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for d, curve in curves.items():
        p = Path(d)
        arm, seed = p.parent.parent.name, p.parent.name
        by_arm.setdefault(arm, {})[seed] = curve
    result: dict[str, object] = {}
    for arm, seeds in sorted(by_arm.items()):
        seed_curves = [c for _, c in sorted(seeds.items()) if c]
        grid = sorted({x for c in seed_curves for x, _ in c})
        result[arm] = {
            "seeds": sorted(seeds),
            "done": sorted(
                s for s in seeds if (root / arm / s / "DONE").exists()
            ),
            "final_per_seed": {s: c[-1][1] for s, c in sorted(seeds.items()) if c},
            "band": band(seed_curves, grid) if grid else None,
        }
    (root / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def render_figure(summary: dict[str, object], root: Path = PROBE_ROOT) -> Path | None:
    """Two-panel mean+min/max-band figure, one panel per game; None if no data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(GAMES), figsize=(11, 4.2), sharey=False)
    drew = False
    for ax, game in zip(axes, GAMES, strict=True):
        for mode, style in (("mirror", "-"), ("league", "--")):
            arm = summary.get(f"{game}_{mode}")
            if not arm or not arm.get("band"):
                continue
            b = arm["band"]
            grid, mean = b["grid"], b["mean"]
            xs = [g for g, m in zip(grid, mean, strict=True) if m is not None]
            if not xs:
                continue
            ys = [m for m in mean if m is not None]
            lo = [v for v in b["min"] if v is not None]
            hi = [v for v in b["max"] if v is not None]
            ax.plot(xs, ys, style, label=f"{mode} (n={max(b['n_seeds'])})")
            ax.fill_between(xs, lo, hi, alpha=0.2)
            drew = True
        ax.set_title(f"{game}: exploitability vs env-steps")
        ax.set_xlabel("env steps (league counts collector-seat decisions)")
        ax.legend()
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("exploitability")
    if not drew:
        return None
    out = root / "figs" / "ab_exploitability.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--game", choices=GAMES)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-env-steps", type=int, default=60_000)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summary = summarize()
        fig = render_figure(summary)
        print(json.dumps({a: v.get("final_per_seed") for a, v in summary.items()}, indent=2))
        if fig:
            print(f"figure: {fig}")
        return 0
    if args.game is None or args.mode is None or args.seed is None:
        parser.error("--game, --mode and --seed are required unless --summarize")
    out = run_arm(
        args.game,
        args.mode,
        args.seed,
        total_env_steps=args.total_env_steps,
        eval_every_env_steps=args.eval_every,
    )
    print(f"DONE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
