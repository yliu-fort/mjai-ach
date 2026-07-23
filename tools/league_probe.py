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
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm_cache
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


def theta_tag(theta: float) -> str:
    """Filesystem-safe arm tag: 0.25 -> ``0p25``. (Shared with theta_probe.)"""
    return f"{theta:g}".replace(".", "p").replace("-", "m")


def arm_name(game: str, mode: str, theta: float | None = None) -> str:
    """Arm directory name; the theta suffix appears only when theta is set.

    ``theta=None`` means "whatever the YAML says" (``algo: ach``) and keeps the
    historical ``<game>_<mode>`` name, so arms trained before the knob existed
    still resolve. An explicit theta MUST be in the path — otherwise a PPO arm
    and an ACH arm of the same mode/seed would overwrite each other, which is
    exactly the comparison the knob exists to make.
    """
    return f"{game}_{mode}" if theta is None else f"{game}_{mode}_t{theta_tag(theta)}"


def parse_arm(name: str) -> tuple[str, str, float | None] | None:
    """Split an arm directory name back into ``(game, mode, theta)``.

    Matched against the known games rather than split on "_": three of the
    seven game names contain underscores (``goofspiel5_ii``, ``liars_dice1``,
    ``oshi_zumo``), so splitting is ambiguous.
    """
    for game in GAMES:
        for mode in MODES:
            prefix = f"{game}_{mode}"
            if name == prefix:
                return game, mode, None
            if name.startswith(f"{prefix}_t"):
                tag = name[len(prefix) + 2 :]
                return game, mode, float(tag.replace("p", ".").replace("m", "-"))
    return None


def arm_dir(root: Path, game: str, mode: str, seed: int, theta: float | None = None) -> Path:
    return root / arm_name(game, mode, theta) / f"seed_{seed}"


def arm_config(
    game: str,
    mode: str,
    seed: int,
    *,
    total_env_steps: int,
    eval_every_env_steps: int,
    root: Path = PROBE_ROOT,
    progress_bar: bool = False,
    device: str | None = None,
    theta: float | None = None,
    probe_term_grad_norms: bool = False,
) -> ExperimentConfig:
    """The resolved config for one arm, without running it.

    Split out of :func:`run_arm` so the cache can fingerprint exactly what
    would be run (tools/arm_cache.py) before deciding to run it.

    ``theta`` overrides the YAML's pinned ``algo: ach`` with the interpolated
    rule (D11): 0 = PPO clipped surrogate, 1 = paper-faithful ACH, in between
    the convex blend of the two POLICY terms. Note what this is not — the
    scaffolding still comes from the ACH config (constant-LR SGD, raw
    advantages, one epoch, no grad clipping), so ``theta=0`` here is "PPO's
    policy term on ACH's protocol", NOT ``configs/exp/<game>_ppo_mlp_*.yaml``.
    That one-factor reading is the whole point of the knob.
    """
    overrides: dict[str, object] = {} if device is None else {"device": device}
    if theta is not None:
        overrides.update(algo="theta", theta=theta)
    return dataclasses.replace(
        load_arm_config(game, mode),
        seed=seed,
        out_dir=str(arm_dir(root, game, mode, seed, theta)),
        probe_term_grad_norms=probe_term_grad_norms,
        total_env_steps=total_env_steps,
        eval_every_env_steps=eval_every_env_steps,
        # Probe-scale pool cadence: at 6e4 env-steps the default 200 main
        # rounds would yield ~1 snapshot; 25 fills the 16-member pool with
        # real main history inside the probe budget (B3).
        league_main_save_every_rounds=25,
        verbose=False,
        progress_bar=progress_bar,
        **overrides,  # type: ignore[arg-type]
    )


def arm_status(game: str, mode: str, seed: int, **kwargs: object) -> arm_cache.ArmStatus:
    """Cache verdict for one arm: hit / stale / missing / legacy.

    Takes the same keyword arguments as :func:`run_arm`, so a caller asks
    "would this exact run be a cache hit?" without duplicating config logic.
    """
    cfg = arm_config(game, mode, seed, **kwargs)  # type: ignore[arg-type]
    return arm_cache.status(Path(cfg.out_dir), cfg)


def run_arm(
    game: str,
    mode: str,
    seed: int,
    *,
    total_env_steps: int,
    eval_every_env_steps: int,
    root: Path = PROBE_ROOT,
    progress_bar: bool = False,
    device: str | None = None,
    theta: float | None = None,
    probe_term_grad_norms: bool = False,
) -> Path:
    """Train one arm; ``device`` overrides the config's (None keeps it).

    "cpu" is the right answer for these games: the rollout asks the policy for
    ONE decision at a time, so a small MLP forward is pure launch-and-sync
    overhead on a GPU (measured 2809 env-steps/s on CPU vs 441 on CUDA for
    Liar's Dice).
    """
    cfg = arm_config(
        game,
        mode,
        seed,
        total_env_steps=total_env_steps,
        eval_every_env_steps=eval_every_env_steps,
        root=root,
        progress_bar=progress_bar,
        device=device,
        theta=theta,
        probe_term_grad_norms=probe_term_grad_norms,
    )
    out = Path(cfg.out_dir)
    run_experiment(cfg)
    arm_cache.write_done(out, cfg)
    mark_best_checkpoint(out)
    return out


def mark_best_checkpoint(
    out_dir: Path, tags: tuple[str, ...] = EVAL_TAG_CHAIN
) -> dict[str, object] | None:
    """Copy the checkpoint with the best (lowest) eval metric to ``checkpoints/best``.

    Reads ``train_curve.json`` (one row per eval point), picks the row with the
    lowest value on the first available tag of the fallback chain, copies that
    ``step_N`` checkpoint dir, and writes ``best.json`` next to it — so the
    best snapshot is directly loadable (e.g. by the play CLI). Returns the
    best-row info, or None loudly when no curve/checkpoint exists.
    """
    curve_file = out_dir / "train_curve.json"
    if not curve_file.is_file():
        return None
    rows = json.loads(curve_file.read_text(encoding="utf-8"))
    for tag in tags:
        scored = [(float(r[tag]), r) for r in rows if tag in r]
        if scored:
            break
    else:
        return None
    value, best_row = min(scored, key=lambda t: t[0])
    src = out_dir / "checkpoints" / f"step_{best_row['step']}"
    if not src.is_dir():
        return None
    dst = out_dir / "checkpoints" / "best"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    info: dict[str, object] = {
        "tag": tag,
        "value": value,
        "step": best_row["step"],
        "env_steps": best_row.get("env_steps"),
    }
    (dst / "best.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


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


def _arm_sort_key(item: tuple[str, object]) -> tuple[int, float, str]:
    """Order a game's arms mirror-before-league, then by theta."""
    parsed = parse_arm(item[0])
    if parsed is None:
        return (2, 0.0, item[0])
    _, mode, theta = parsed
    return (MODES.index(mode) if mode in MODES else 2, -1.0 if theta is None else theta, item[0])


def _arm_metric(arm: object) -> str:
    """Short equilibrium-metric name recorded for an arm ("" if unknown)."""
    if not isinstance(arm, dict):
        return ""
    tag = arm.get("tag")
    if isinstance(tag, list):
        tag = tag[0] if tag else ""
    return tag.removeprefix("eval/") if isinstance(tag, str) else ""


def build_figure(
    summary: dict[str, object], games: Sequence[str] | None = None
) -> tuple[object, bool]:
    """Build the per-game panel grid; returns ``(figure, drew_anything)``.

    ``games`` selects the panel set: None keeps all of :data:`GAMES` (the
    whole-probe view, where an empty panel means "that arm has not run yet"),
    while a single-game list is what a per-game ``ab_<game>.ipynb`` wants —
    otherwise its figure is 7 panels of which 6 are permanently blank.

    Built through the pyplot-free Figure API on purpose. ``matplotlib.use()``
    mutates GLOBAL state: calling it from a helper a notebook imports switches
    the kernel off the inline backend for good, and every later plt.show() in
    that notebook silently renders nothing (verified: the A/B notebook's league
    telemetry panel). A bare Figure also never enters pyplot's registry, so
    nothing leaks.
    """
    from matplotlib import colormaps
    from matplotlib.figure import Figure

    panels = list(GAMES if games is None else games)
    ncols = min(4, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig = Figure(figsize=(ncols * 4.6, nrows * 4.2))
    axes = fig.subplots(nrows, ncols, sharey=False, squeeze=False)
    cmap = colormaps["viridis"]
    styles = {"mirror": "-", "league": "--"}
    drew = False
    for idx, game in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        metric = ""
        panel_drew = False
        # Line STYLE encodes the self-play mode, COLOR encodes theta (same
        # viridis ramp the theta-scan notebooks use), so a mode x theta sweep
        # stays readable in one panel and matches the other notebook family.
        for name, arm in sorted(summary.items(), key=_arm_sort_key):
            parsed = parse_arm(name)
            if parsed is None or parsed[0] != game:
                continue
            _, mode, theta = parsed
            if not isinstance(arm, dict) or not arm.get("band"):
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
            color = "tab:blue" if theta is None else cmap(theta)
            label = mode if theta is None else f"{mode} theta={theta:g}"
            ax.plot(
                xs, ys, styles.get(mode, "-"), color=color, label=f"{label} (n={max(b['n_seeds'])})"
            )
            ax.fill_between(xs, lo, hi, color=color, alpha=0.18)
            drew = panel_drew = True
        ax.set_title(f"{game}: {metric or 'equilibrium metric'} vs env-steps")
        ax.set_xlabel("env steps (league counts collector-seat decisions)")
        if panel_drew:  # legend() with no labeled artists warns (tests: -W error)
            ax.legend()
        ax.grid(alpha=0.3)
        if idx % ncols == 0:
            ax.set_ylabel(metric or "equilibrium metric")
    for idx in range(len(panels), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.tight_layout()
    return fig, drew


def render_figure(
    summary: dict[str, object], root: Path = PROBE_ROOT, games: Sequence[str] | None = None
) -> Path | None:
    """Save the panel grid under ``root/figs``; None when no arm has data."""
    fig, drew = build_figure(summary, games)
    if not drew:
        return None
    stem = "ab_exploitability" if games is None else f"ab_{'_'.join(games)}"
    out = root / "figs" / f"{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)  # type: ignore[attr-defined]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--game", choices=GAMES)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-env-steps", type=int, default=60_000)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--root",
        default=str(PROBE_ROOT),
        help="Probe output root (default: runs/league_probe).",
    )
    args = parser.parse_args()
    root = Path(args.root)
    if args.summarize:
        summary = summarize(root)
        fig = render_figure(summary, root)
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
        root=root,
    )
    print(f"DONE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
