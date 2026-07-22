"""Mirror-vs-league A/B probe (F2): equilibrium-metric convergence comparison.

Arms: all 7 Phase-1 games (AGENTS.md D8) x {mirror, league}, N seeds each,
paper-faithful ACH protocol loaded from configs/exp/<game>_ach_mlp_<mode>.yaml
with the step budget overridden. Each arm writes
runs/league_probe/<game>_<mode>/seed_S (+ a DONE marker); --summarize
aggregates the TB eval curves into per-arm mean/min/max bands (summary.json)
and a per-game panel-grid figure.

Equilibrium metric per run follows the fallback chain of
src/mjai/eval/plots.py (exploitability for sequential games, nash_conv for
simultaneous ones): eval/exploitability -> eval/nash_conv ->
eval/exact_nash_distance; the first tag with any points wins.

Usage::

    python tools/league_probe.py --game kuhn --mode league --seed 0 \
        --total-env-steps 60000 --eval-every 5000
    python tools/league_probe.py --summarize
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tb_eval import read_many

from mjai.scripts.experiment import ExperimentConfig, run_experiment

REPO = Path(__file__).resolve().parents[1]
PROBE_ROOT = REPO / "runs" / "league_probe"
# All 7 Phase-1 games (AGENTS.md D8); sequential-metric games first so each
# figure row shares one equilibrium metric.
GAMES = ("kuhn", "leduc", "liars_dice1", "ttt", "brps", "goofspiel5_ii", "oshi_zumo")
MODES = ("mirror", "league")
# Equilibrium-metric fallback chain, mirroring mjai.eval.plots
# (metric_key + fallback_keys): the first tag with any points wins per run.
EVAL_TAG_CHAIN = ("eval/exploitability", "eval/nash_conv", "eval/exact_nash_distance")


def parse_seeds(spec: str) -> list[int]:
    """Parse "0-3", "0,2,5" or "3" into a sorted unique seed list."""
    seeds: set[int] = set()
    for raw in spec.split(","):
        part = raw.strip()
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


def read_curves_fallback(
    tb_dirs: list[str | Path], tags: tuple[str, ...] = EVAL_TAG_CHAIN
) -> tuple[dict[str, list[tuple[int, float]]], dict[str, str]]:
    """Read each run's eval curve, falling back through ``tags`` per run.

    The first tag yielding any points wins (sequential games log
    exploitability; simultaneous ones only nash_conv). Returns the curves
    keyed by tb_dir string plus the winning tag per tb_dir.
    """
    curves: dict[str, list[tuple[int, float]]] = {}
    used_tag: dict[str, str] = {}
    pending = [str(d) for d in tb_dirs]
    for tag in tags:
        if not pending:
            break
        batch = read_many(pending, tag=tag)
        retry = []
        for d, curve in batch.items():
            if curve:
                curves[d] = curve
                used_tag[d] = tag
            else:
                retry.append(d)
        pending = retry
    for d in pending:  # no tag produced points — keep as empty (band=None)
        curves[d] = []
    return curves, used_tag


def summarize(root: Path = PROBE_ROOT, tags: tuple[str, ...] = EVAL_TAG_CHAIN) -> dict[str, object]:
    """Aggregate all finished arms under root into bands + write summary.json."""
    tb_dirs = sorted(root.glob("*_*/seed_*/tb"))
    curves, used_tag = read_curves_fallback(tb_dirs, tags)
    by_arm: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for d, curve in curves.items():
        p = Path(d)
        arm, seed = p.parent.parent.name, p.parent.name
        by_arm.setdefault(arm, {})[seed] = curve
    result: dict[str, object] = {}
    for arm, seeds in sorted(by_arm.items()):
        seed_curves = [c for _, c in sorted(seeds.items()) if c]
        grid = sorted({x for c in seed_curves for x, _ in c})
        arm_tags = sorted({used_tag.get(str(root / arm / s / "tb"), "") for s in seeds} - {""})
        result[arm] = {
            "seeds": sorted(seeds),
            "done": sorted(s for s in seeds if (root / arm / s / "DONE").exists()),
            "tag": arm_tags[0] if len(arm_tags) == 1 else arm_tags,
            "final_per_seed": {s: c[-1][1] for s, c in sorted(seeds.items()) if c},
            "band": band(seed_curves, grid) if grid else None,
        }
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _arm_metric(arm: object) -> str:
    """Short equilibrium-metric name recorded for an arm ("" if unknown)."""
    if not isinstance(arm, dict):
        return ""
    tag = arm.get("tag")
    if isinstance(tag, list):
        tag = tag[0] if tag else ""
    return tag.removeprefix("eval/") if isinstance(tag, str) else ""


def render_figure(summary: dict[str, object], root: Path = PROBE_ROOT) -> Path | None:
    """Per-game panel grid of mean+min/max bands; empty panels hidden.

    One panel per game in ``GAMES`` (2 rows x 4 cols for 7 games); each
    panel's title names the equilibrium metric its curves came from.
    Returns None if no arm has data.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 4
    nrows = math.ceil(len(GAMES) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 4.6, nrows * 4.2), sharey=False, squeeze=False
    )
    drew = False
    for idx, game in enumerate(GAMES):
        ax = axes[idx // ncols][idx % ncols]
        metric = ""
        panel_drew = False
        for mode, style in (("mirror", "-"), ("league", "--")):
            arm = summary.get(f"{game}_{mode}")
            if not arm or not isinstance(arm, dict) or not arm.get("band"):
                continue
            metric = metric or _arm_metric(arm)
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
            drew = panel_drew = True
        ax.set_title(f"{game}: {metric or 'equilibrium metric'} vs env-steps")
        ax.set_xlabel("env steps (league counts collector-seat decisions)")
        if panel_drew:  # legend() with no labeled artists warns (tests: -W error)
            ax.legend()
        ax.grid(alpha=0.3)
        if idx % ncols == 0:
            ax.set_ylabel(metric or "equilibrium metric")
    for idx in range(len(GAMES), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
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
