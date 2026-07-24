"""Single-factor A/B probe: compare NAMED arms that differ by one config axis.

Both A/B notebooks this backs run on ONE game (liar's dice), mirror self-play,
and differ from each other only in what the arms vary:

  * ppo_vs_ach_<game>.ipynb  — the POLICY / ALGORITHM axis: best-practice PPO
    (its own Adam + advantage-norm + clip + multi-epoch protocol), single-factor
    PPO (theta=0 on the ACH scaffold), and paper-faithful ACH.
  * sgd_vs_adam_<game>.ipynb — the OPTIMIZER axis, holding the algorithm fixed
    at ACH: constant-LR SGD vs Adam at two learning rates.

An "arm" is a base ``configs/exp/<name>.yaml`` plus an override dict applied on
top (optimizer, learning_rate, ...). That is all the two notebooks need to be
the SAME machinery with different arm lists, so nothing here is game- or
axis-specific — the arm lists live in the generated notebooks' parameter cell.

Everything downstream of "run an arm" is reused from tools/league_probe.py
(band aggregation, the eval-tag fallback chain, the SOTA best-checkpoint copy)
and tools/tb_eval.py (the downsampled telemetry reader). Nothing is
reimplemented (AGENTS.md §7). Arms are cached by a fingerprint of their resolved
ExperimentConfig (tools/arm_cache.py), so changing a budget or an override is
detected rather than silently reused.

Usage::

    python tools/ab_factor_probe.py --config liars_dice1_ach_mlp_mirror \
        --label ach_sgd --seed 0 --root runs/nb_sgd_adam/liars_dice1 \
        --total-env-steps 10000 --eval-every 2500
    python tools/ab_factor_probe.py --summarize --root runs/nb_sgd_adam/liars_dice1
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm_cache
from league_probe import EVAL_TAG_CHAIN, band, mark_best_checkpoint, read_curves_fallback
from tb_eval import read_many_tags

from mjai.scripts.experiment import ExperimentConfig, run_experiment

REPO = Path(__file__).resolve().parents[1]
# Per-update telemetry these A/B notebooks read. The arms sit at theta
# endpoints (0 or 1), so the PER-TERM split and the cosine are empty by
# construction — the informative panels are the TOTAL gradient scale (SGD vs
# Adam differ by orders of magnitude), the ACH gate-off rate (theta=1 arms),
# and the PPO clip rate (theta=0 arms). grad_norm is spike-preserved when
# downsampled; the two frac tags live in [0, 1] and are strided.
TELEMETRY_TAGS = ("train/grad_norm", "train/gate_off_frac", "train/clip_frac")
PEAK_TAGS = ("train/grad_norm",)


def load_config(config_name: str) -> ExperimentConfig:
    """Load ``configs/exp/<config_name>.yaml`` (unknown keys fail loudly, §9)."""
    path = REPO / "configs" / "exp" / f"{config_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no experiment config {config_name!r}: {path}")
    return ExperimentConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))


def arm_dir(root: Path, label: str, seed: int) -> Path:
    """One arm's output dir: ``<root>/<label>/seed_<seed>`` (root carries the game)."""
    return root / label / f"seed_{seed}"


def arm_config(
    label: str,
    config_name: str,
    seed: int,
    *,
    overrides: Mapping[str, object] | None = None,
    total_env_steps: int,
    eval_every_env_steps: int,
    root: Path,
    progress_bar: bool = False,
    device: str | None = None,
    probe_term_grad_norms: bool = False,
) -> ExperimentConfig:
    """Resolve one arm's config WITHOUT running it (so arm_cache can fingerprint it).

    The base YAML is the arm's algorithm/scaffold; ``overrides`` is the single
    axis the A/B varies (e.g. ``{"optimizer": "adam", "learning_rate": 3e-4}``).
    An override naming a field ExperimentConfig does not have raises loudly via
    ``dataclasses.replace`` — the config is never silently mis-set (§9).
    """
    over: dict[str, object] = dict(overrides or {})
    if device is not None:
        over["device"] = device
    return dataclasses.replace(
        load_config(config_name),
        seed=seed,
        out_dir=str(arm_dir(root, label, seed)),
        total_env_steps=total_env_steps,
        eval_every_env_steps=eval_every_env_steps,
        probe_term_grad_norms=probe_term_grad_norms,
        verbose=False,
        progress_bar=progress_bar,
        **over,  # type: ignore[arg-type]
    )


def arm_status(label: str, config_name: str, seed: int, **kwargs: object) -> arm_cache.ArmStatus:
    """Cache verdict for one arm: hit / stale / missing / legacy."""
    cfg = arm_config(label, config_name, seed, **kwargs)  # type: ignore[arg-type]
    return arm_cache.status(Path(cfg.out_dir), cfg)


def run_arm(
    label: str,
    config_name: str,
    seed: int,
    *,
    overrides: Mapping[str, object] | None = None,
    total_env_steps: int,
    eval_every_env_steps: int,
    root: Path,
    progress_bar: bool = False,
    device: str | None = None,
    probe_term_grad_norms: bool = False,
) -> Path:
    """Train one arm to completion, mark it DONE, and copy its SOTA checkpoint."""
    cfg = arm_config(
        label,
        config_name,
        seed,
        overrides=overrides,
        total_env_steps=total_env_steps,
        eval_every_env_steps=eval_every_env_steps,
        root=root,
        progress_bar=progress_bar,
        device=device,
        probe_term_grad_norms=probe_term_grad_norms,
    )
    out = Path(cfg.out_dir)
    run_experiment(cfg)
    arm_cache.write_done(out, cfg)
    mark_best_checkpoint(out)
    return out


def summarize(root: Path, tags: tuple[str, ...] = EVAL_TAG_CHAIN) -> dict[str, object]:
    """Aggregate every finished arm under ``root`` into mean/min-max bands.

    Arms live at ``<root>/<label>/seed_*/tb``. Reuses league_probe's per-run
    eval-tag fallback chain and seed-band builder, keyed by arm label. Writes
    ``summary.json`` and returns ``{label: {...}}``.
    """
    tb_dirs = sorted(root.glob("*/seed_*/tb"))
    curves, used_tag = read_curves_fallback(tb_dirs, tags)
    by_arm: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for d, curve in curves.items():
        p = Path(d)
        by_arm.setdefault(p.parent.parent.name, {})[p.parent.name] = curve
    result: dict[str, object] = {}
    for label, seeds in sorted(by_arm.items()):
        seed_curves = [c for _, c in sorted(seeds.items()) if c]
        grid = sorted({x for c in seed_curves for x, _ in c})
        arm_tags = sorted({used_tag.get(str(root / label / s / "tb"), "") for s in seeds} - {""})
        result[label] = {
            "seeds": sorted(seeds),
            "done": sorted(s for s in seeds if (root / label / s / "DONE").exists()),
            "tag": arm_tags[0] if len(arm_tags) == 1 else arm_tags,
            "final_per_seed": {s: c[-1][1] for s, c in sorted(seeds.items()) if c},
            "band": band(seed_curves, grid) if grid else None,
        }
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _metric_name(summary: Mapping[str, object]) -> str:
    """Short equilibrium-metric label shared by the arms ("" if unknown/mixed)."""
    names: set[str] = set()
    for arm in summary.values():
        tag = arm.get("tag") if isinstance(arm, dict) else None
        if isinstance(tag, str) and tag:
            names.add(tag.removeprefix("eval/"))
    return next(iter(names)) if len(names) == 1 else ""


def _ordered(summary: Mapping[str, object], order: Sequence[str] | None) -> list[str]:
    """Arm labels in plotting order: caller's ``order`` first, then any extras."""
    labels = list(order or [])
    labels += [k for k in sorted(summary) if k not in labels]
    return [k for k in labels if k in summary]


def build_curves_figure(
    summary: Mapping[str, object],
    *,
    title: str,
    order: Sequence[str] | None = None,
    blurbs: Mapping[str, str] | None = None,
) -> tuple[object, bool]:
    """Overlay every arm's metric-vs-env-steps curve (mean + min-max band).

    Pyplot-free on purpose (see :func:`league_probe.build_figure`):
    ``matplotlib.use()`` from a notebook-imported helper kills the kernel's
    inline backend for the rest of the session. One categorical colour per arm.
    """
    from matplotlib import colormaps
    from matplotlib.figure import Figure

    metric = _metric_name(summary)
    labels = _ordered(summary, order)
    cmap = colormaps["tab10"]
    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    drew = False
    all_positive = True
    for i, label in enumerate(labels):
        arm = summary[label]
        b = arm.get("band") if isinstance(arm, dict) else None
        if not b:
            continue
        xs = [g for g, m in zip(b["grid"], b["mean"], strict=True) if m is not None]
        if not xs:
            continue
        ys = [m for m in b["mean"] if m is not None]
        lo = [v for v in b["min"] if v is not None]
        hi = [v for v in b["max"] if v is not None]
        all_positive &= all(v > 0 for v in lo)
        legend = f"{label} (n={max(b['n_seeds'])})"
        if blurbs and blurbs.get(label):
            legend = f"{label}: {blurbs[label]} (n={max(b['n_seeds'])})"
        ax.plot(xs, ys, color=cmap(i % 10), lw=1.8, label=legend)
        ax.fill_between(xs, lo, hi, color=cmap(i % 10), alpha=0.15)
        drew = True
    if not drew:
        return fig, False
    ax.set_xlabel("env steps")
    ax.set_ylabel(f"{metric or 'equilibrium metric'} (lower = closer to Nash)")
    ax.set_title(title)
    ax.set_yscale("log" if all_positive else "symlog")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig, True


def render_curves(
    summary: Mapping[str, object],
    root: Path,
    *,
    title: str,
    order: Sequence[str] | None = None,
    blurbs: Mapping[str, str] | None = None,
) -> Path | None:
    """Save the overlay under ``root/figs``; None when no arm has data."""
    fig, drew = build_curves_figure(summary, title=title, order=order, blurbs=blurbs)
    if not drew:
        return None
    out = root / "figs" / "ab_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)  # type: ignore[attr-defined]
    return out


def build_telemetry_figure(
    root: Path,
    *,
    tags: Sequence[str] = TELEMETRY_TAGS,
    order: Sequence[str] | None = None,
    max_points: int = 2000,
) -> tuple[object, bool]:
    """Per-update telemetry, one panel per tag, one colour per arm (seed 0).

    Reads through :func:`tb_eval.read_many_tags` — one downsampled pass per file
    for all tags — so this panel does not re-introduce the full-resolution read
    the eval curves were fixed to avoid. ``train/grad_norm`` spans orders of
    magnitude (log axis) and keeps its spikes; the frac tags stay linear.
    """
    from matplotlib import colormaps
    from matplotlib.figure import Figure

    arm_dirs = sorted(root.glob("*/seed_0/tb"))
    if not arm_dirs:
        return Figure(), False
    series = read_many_tags(arm_dirs, tags, max_points=max_points, peak_tags=PEAK_TAGS)
    labels = order or sorted({Path(d).parent.parent.name for d in arm_dirs})
    color = {label: colormaps["tab10"](i % 10) for i, label in enumerate(labels)}
    ncols = len(tags)
    fig = Figure(figsize=(5 * ncols, 3.8))
    axes = fig.subplots(1, ncols, squeeze=False)[0]
    drew = False
    for ax, tag in zip(axes, tags, strict=True):
        for d in sorted(series):
            label = Path(d).parent.parent.name
            curve = series[d].get(tag)
            if not curve:
                continue
            ax.plot(
                [s for s, _ in curve],
                [v for _, v in curve],
                color=color.get(label, "0.5"),
                lw=1.0,
                label=label,
            )
            drew = True
        ax.set_title(tag)
        ax.set_xlabel("update")
        ax.grid(alpha=0.3)
        if tag in PEAK_TAGS and ax.get_lines():
            ax.set_yscale("log")
        if not ax.get_lines():
            ax.text(
                0.5,
                0.5,
                "no points",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color="0.5",
            )
    if drew:
        axes[0].legend(fontsize=7)
        fig.tight_layout()
    return fig, drew


def render_telemetry(
    root: Path, *, order: Sequence[str] | None = None, max_points: int = 2000
) -> Path | None:
    """Save the telemetry grid under ``root/figs``; None when there is nothing."""
    fig, drew = build_telemetry_figure(root, order=order, max_points=max_points)
    if not drew:
        return None
    out = root / "figs" / "ab_telemetry.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)  # type: ignore[attr-defined]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", help="base config stem under configs/exp/")
    parser.add_argument("--label", help="arm label (output subdir)")
    parser.add_argument("--overrides", default="{}", help="Python dict literal of overrides")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--root", required=True, help="probe output root (carries the game)")
    parser.add_argument("--total-env-steps", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=2_500)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    if args.summarize:
        summary = summarize(root)
        print(json.dumps({a: v.get("final_per_seed") for a, v in summary.items()}, indent=2))
        return 0
    if not (args.config and args.label and args.seed is not None):
        parser.error("--config, --label and --seed are required unless --summarize")
    out = run_arm(
        args.label,
        args.config,
        args.seed,
        overrides=ast.literal_eval(args.overrides),
        total_env_steps=args.total_env_steps,
        eval_every_env_steps=args.eval_every,
        root=root,
    )
    print(f"DONE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
