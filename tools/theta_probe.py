"""PPO<->ACH theta scan: equilibrium-metric convergence as a function of theta.

Arms: one game x N theta values x M seeds, mirror self-play, the paper-faithful
ACH protocol loaded from ``configs/exp/<game>_ach_mlp_mirror.yaml`` with
``algo`` switched to ``theta`` and the step budget overridden. Because the
scaffolding (SGD constant LR, raw advantages, one epoch per batch, LayerNorm
torso) is shared at every theta, an arm differs from its neighbours in the
**policy term alone** — that is what makes the sweep a one-factor scan rather
than an algorithm comparison confounded by its optimizer.

Each arm writes ``<root>/<game>/theta_<tag>/seed_S`` plus a ``DONE`` marker, so
re-running skips finished arms. ``--summarize`` aggregates the TensorBoard eval
curves into per-theta mean/min-max bands (``summary.json``) and renders the two
figures the notebooks show:

  1. every theta's metric-vs-env-steps curve, overlaid;
  2. final metric vs theta.

"Final" follows the repo's D5 convention (docs/reproduce_report.md): the mean
over the last ``--final-frac`` of the x axis, per seed, then aggregated across
seeds — not the single last point, which is too noisy to rank thetas by.

Equilibrium metric per run follows the same fallback chain as
tools/league_probe.py and src/mjai/eval/plots.py: exploitability for
turn-based games, nash_conv for simultaneous ones (BRPS has no exploitability —
mjai.eval.nash only computes it for non-simultaneous games — so its panels are
labelled nash_conv and must not be read as exploitability).

Usage::

    python tools/theta_probe.py --game kuhn --theta 0.5 --seed 0 \
        --total-env-steps 60000 --eval-every 5000
    python tools/theta_probe.py --summarize --game kuhn
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from league_probe import EVAL_TAG_CHAIN, band, mark_best_checkpoint, read_curves_fallback

from mjai.scripts.experiment import ExperimentConfig, run_experiment

REPO = Path(__file__).resolve().parents[1]
PROBE_ROOT = REPO / "runs" / "theta_probe"
# The three games the theta-scan notebooks cover: one simultaneous cycling game
# and two sequential imperfect-information games with exact exploitability.
GAMES = ("brps", "kuhn", "liars_dice1")
DEFAULT_THETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
# Per-update telemetry worth reading per theta: gradient scale (the ACH term's
# unbounded 1/pi_old vs the PPO term's O(1) surrogate), gate activity, and the
# PPO clip rate.
TELEMETRY_TAGS = ("train/grad_norm", "train/gate_off_frac", "train/clip_frac")


def theta_tag(theta: float) -> str:
    """Filesystem-safe arm tag: 0.25 -> ``0p25``."""
    return f"{theta:g}".replace(".", "p").replace("-", "m")


def parse_thetas(spec: str) -> list[float]:
    """Parse "0,0.25,0.5" into a sorted unique theta list, validating the range."""
    out: set[float] = set()
    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            continue
        value = float(part)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"theta must lie in [0, 1], got {value}")
        out.add(value)
    if not out:
        raise ValueError(f"no thetas parsed from {spec!r}")
    return sorted(out)


def load_base_config(game: str) -> ExperimentConfig:
    """Load the game's ACH mirror arm — the scan's shared scaffolding."""
    path = REPO / "configs" / "exp" / f"{game}_ach_mlp_mirror.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no base config for game {game!r}: {path}")
    return ExperimentConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))


def arm_dir(root: Path, game: str, theta: float, seed: int) -> Path:
    return root / game / f"theta_{theta_tag(theta)}" / f"seed_{seed}"


def run_arm(
    game: str,
    theta: float,
    seed: int,
    *,
    total_env_steps: int,
    eval_every_env_steps: int,
    root: Path = PROBE_ROOT,
    progress_bar: bool = False,
) -> Path:
    """Train one (game, theta, seed) arm to completion and mark it DONE."""
    out = arm_dir(root, game, theta, seed)
    cfg = dataclasses.replace(
        load_base_config(game),
        algo="theta",
        theta=theta,
        seed=seed,
        out_dir=str(out),
        total_env_steps=total_env_steps,
        eval_every_env_steps=eval_every_env_steps,
        verbose=False,
        progress_bar=progress_bar,
    )
    run_experiment(cfg)
    (out / "DONE").write_text("ok\n", encoding="utf-8")
    mark_best_checkpoint(out)
    return out


def final_value(curve: list[tuple[int, float]], final_frac: float) -> float | None:
    """Mean of the curve's last ``final_frac`` of the x axis (D5 convention)."""
    if not curve:
        return None
    x_max = curve[-1][0]
    cutoff = x_max - final_frac * (x_max - curve[0][0])
    tail = [v for x, v in curve if x >= cutoff] or [curve[-1][1]]
    return sum(tail) / len(tail)


def summarize(
    root: Path = PROBE_ROOT,
    game: str | None = None,
    *,
    final_frac: float = 0.1,
    tags: tuple[str, ...] = EVAL_TAG_CHAIN,
) -> dict[str, object]:
    """Aggregate finished arms into per-theta bands + finals; write summary.json.

    Returns ``{game: {"thetas": {tag: {...}}, "metric": str}}``.
    """
    games = [game] if game else sorted({p.name for p in root.glob("*") if p.is_dir()})
    result: dict[str, object] = {}
    for g in games:
        tb_dirs = sorted((root / g).glob("theta_*/seed_*/tb"))
        if not tb_dirs:
            continue
        curves, used_tag = read_curves_fallback(tb_dirs, tags)
        by_theta: dict[str, dict[str, list[tuple[int, float]]]] = {}
        for d, curve in curves.items():
            p = Path(d)
            by_theta.setdefault(p.parent.parent.name, {})[p.parent.name] = curve
        metrics = sorted({t for t in used_tag.values() if t})
        entry: dict[str, object] = {}
        for tag, seeds in sorted(by_theta.items()):
            seed_curves = [c for _, c in sorted(seeds.items()) if c]
            grid = sorted({x for c in seed_curves for x, _ in c})
            finals = {s: final_value(c, final_frac) for s, c in sorted(seeds.items()) if c}
            entry[tag] = {
                "theta": float(tag.removeprefix("theta_").replace("p", ".")),
                "seeds": sorted(seeds),
                "done": sorted(s for s in seeds if (root / g / tag / s / "DONE").exists()),
                "final_per_seed": finals,
                "band": band(seed_curves, grid) if grid else None,
            }
        result[g] = {
            "metric": metrics[0] if len(metrics) == 1 else metrics,
            "final_frac": final_frac,
            "thetas": entry,
        }
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _game_entry(summary: dict[str, object], game: str) -> tuple[dict, str]:
    """Pull one game's theta map + short metric name out of a summary."""
    entry = summary.get(game)
    if not isinstance(entry, dict):
        return {}, ""
    metric = entry.get("metric", "")
    if isinstance(metric, list):
        metric = metric[0] if metric else ""
    thetas = entry.get("thetas")
    return (thetas if isinstance(thetas, dict) else {}), str(metric).removeprefix("eval/")


def _all_positive(ordered_arms: list[tuple[str, dict]]) -> bool:
    """True when every band value across the plotted arms is strictly positive."""
    for _tag, arm in ordered_arms:
        b = arm.get("band")
        if not b:
            continue
        for key in ("mean", "min", "max"):
            if any(v is not None and v <= 0.0 for v in b[key]):
                return False
    return True


def render_curves(summary: dict[str, object], game: str, root: Path = PROBE_ROOT) -> Path | None:
    """Figure 1: every theta's metric-vs-env-steps curve, overlaid with bands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thetas, metric = _game_entry(summary, game)
    if not thetas:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    ordered = sorted(thetas.items(), key=lambda kv: kv[1]["theta"])
    drew = False
    for _tag, arm in ordered:
        b = arm.get("band")
        if not b:
            continue
        xs = [g for g, m in zip(b["grid"], b["mean"], strict=True) if m is not None]
        if not xs:
            continue
        ys = [m for m in b["mean"] if m is not None]
        lo = [v for v in b["min"] if v is not None]
        hi = [v for v in b["max"] if v is not None]
        color = cmap(arm["theta"])
        label = f"theta={arm['theta']:g} (n={max(b['n_seeds'])})"
        ax.plot(xs, ys, color=color, lw=1.8, label=label)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
        drew = True
    if not drew:
        return None
    ax.set_xlabel("env steps")
    ax.set_ylabel(f"{metric or 'equilibrium metric'} (lower = closer to Nash)")
    ax.set_title(f"{game}: {metric or 'equilibrium metric'} per theta (0 = PPO, 1 = ACH)")
    # Plain log while every plotted value is positive (equilibrium metrics are
    # non-negative, and log keeps the tick labels readable on a narrow range);
    # symlog only when something hits zero, which log cannot show.
    ax.set_yscale("log" if _all_positive(ordered) else "symlog")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    out = root / "figs" / f"theta_curves_{game}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_theta_final(
    summary: dict[str, object], game: str, root: Path = PROBE_ROOT
) -> Path | None:
    """Figure 2: final metric vs theta, with a min-max error bar across seeds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thetas, metric = _game_entry(summary, game)
    if not thetas:
        return None
    xs: list[float] = []
    means: list[float] = []
    lo_err: list[float] = []
    hi_err: list[float] = []
    n_seeds: list[int] = []
    for _tag, arm in sorted(thetas.items(), key=lambda kv: kv[1]["theta"]):
        vals = [v for v in arm.get("final_per_seed", {}).values() if v is not None]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        xs.append(arm["theta"])
        means.append(mean)
        lo_err.append(mean - min(vals))
        hi_err.append(max(vals) - mean)
        n_seeds.append(len(vals))
    if not xs:
        return None
    frac = summary.get(game, {}).get("final_frac", 0.1)  # type: ignore[union-attr]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(xs, means, yerr=[lo_err, hi_err], marker="o", capsize=4, lw=1.6, color="tab:purple")
    for x, m, n in zip(xs, means, n_seeds, strict=True):
        ax.annotate(f"n={n}", (x, m), textcoords="offset points", xytext=(0, 8), fontsize=7)
    ax.set_xlabel("theta   (0 = PPO clipped surrogate,  1 = paper-faithful ACH)")
    ax.set_ylabel(f"final {metric or 'equilibrium metric'}")
    ax.set_title(
        f"{game}: final {metric or 'equilibrium metric'} vs theta\n"
        f"(mean of last {float(frac):.0%} of x; error bar = min-max over seeds)",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    out = root / "figs" / f"theta_final_{game}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_telemetry(game: str, root: Path = PROBE_ROOT) -> Path | None:
    """Diagnostic panel: per-update telemetry vs training step, one line per theta.

    The theta blend is a convex combination of two policy losses whose gradient
    magnitudes differ by orders of magnitude (the ACH term carries an unbounded
    ``1/pi_old``), so the effective learning rate varies with theta. This panel
    makes that confounder visible instead of leaving it to be inferred from the
    curves.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tb_eval import read_many

    arm_dirs = sorted((root / game).glob("theta_*/seed_0/tb"))
    if not arm_dirs:
        return None
    fig, axes = plt.subplots(1, len(TELEMETRY_TAGS), figsize=(5 * len(TELEMETRY_TAGS), 3.8))
    cmap = plt.get_cmap("viridis")
    drew = False
    for ax, tag in zip(axes, TELEMETRY_TAGS, strict=True):
        for d, curve in sorted(read_many(list(arm_dirs), tag=tag).items()):
            if not curve:
                continue
            theta = float(Path(d).parent.parent.name.removeprefix("theta_").replace("p", "."))
            ax.plot(
                [s for s, _ in curve],
                [v for _, v in curve],
                color=cmap(theta),
                lw=1.0,
                label=f"theta={theta:g}",
            )
            drew = True
        ax.set_title(tag)
        ax.set_xlabel("update")
        ax.grid(alpha=0.3)
    if not drew:
        plt.close(fig)
        return None
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7)
    out = root / "figs" / f"theta_telemetry_{game}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--game", choices=GAMES)
    parser.add_argument("--theta", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-env-steps", type=int, default=60_000)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--final-frac", type=float, default=0.1)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--root", default=str(PROBE_ROOT), help="Probe output root.")
    args = parser.parse_args()
    root = Path(args.root)
    if args.summarize:
        summary = summarize(root, args.game, final_frac=args.final_frac)
        for g in summary:
            for fig in (render_curves(summary, g, root), render_theta_final(summary, g, root)):
                if fig:
                    print(f"figure: {fig}")
        print(json.dumps({g: v.get("metric") for g, v in summary.items()}, indent=2))  # type: ignore[union-attr]
        return 0
    if args.game is None or args.theta is None or args.seed is None:
        parser.error("--game, --theta and --seed are required unless --summarize")
    out = run_arm(
        args.game,
        args.theta,
        args.seed,
        total_env_steps=args.total_env_steps,
        eval_every_env_steps=args.eval_every,
        root=root,
    )
    print(f"DONE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
