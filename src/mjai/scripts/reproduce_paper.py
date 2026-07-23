"""Reproduce the ACH paper's Appendix G OpenSpiel experiments (p25-26).

Runs the paper-faithful :class:`NNACHUpdate` (mirror self-play) on Kuhn poker,
Leduc poker, and Liar's Dice with N independent seeds (paper: 8 runs, p26),
training to 1e7 env-steps with exact-exploitability evaluation every 1e5
env-steps. Serial and resumable: a run directory containing a ``DONE`` marker
is skipped, so re-running the command continues an interrupted sweep.

This script only wires configs into :func:`run_experiment` — it contains no
training logic and no metric logging of its own (D9: TensorBoard only).

Usage (from the repo root)::

    uv run python -m mjai.scripts.reproduce_paper \
        --games kuhn,leduc,liars_dice1 --seeds 0-7 --out runs/reproduce
    uv run python -m mjai.scripts.reproduce_paper --games kuhn --seeds 0 --cpu
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import yaml

from mjai.scripts.experiment import ExperimentConfig, run_experiment

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GAMES = ["kuhn", "leduc", "liars_dice1"]


def _parse_seeds(spec: str) -> list[int]:
    """Parse ``"0-7"`` or ``"0,1,3"`` into a list of ints."""
    out: list[int] = []
    for chunk in spec.split(","):
        part = chunk.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    if not out:
        raise ValueError(f"No seeds parsed from {spec!r}")
    return out


def _load_exp_config(game: str) -> ExperimentConfig:
    path = REPO_ROOT / "configs" / "exp" / f"{game}_ach_mlp_mirror.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ExperimentConfig(**data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--games", default=",".join(DEFAULT_GAMES), help="Comma list.")
    parser.add_argument("--seeds", default="0-7", help="e.g. '0-7' or '0,1,3' (paper: 8 runs).")
    parser.add_argument("--out", default="runs/reproduce", help="Output root directory.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (AGENTS.md §1 D6).")
    parser.add_argument(
        "--legal-mean",
        action="store_true",
        help="ACH gate/loss centered-mean over legal actions only "
        "(centered_mean_legal_only=True; A5 probe for the Liar's Dice gap).",
    )
    args = parser.parse_args(argv)

    if args.cpu:
        from mjai.utils import gpu_assert

        gpu_assert.require_cpu()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    seeds = _parse_seeds(args.seeds)
    out_root = Path(args.out)

    planned = [(g, s) for g in games for s in seeds]
    print(f"reproduce_paper: {len(planned)} runs planned ({len(games)} games x {len(seeds)} seeds)")
    n_done = 0
    for game, seed in planned:
        out_dir = out_root / f"{game}_ach_mlp_mirror" / f"seed_{seed}"
        done_marker = out_dir / "DONE"
        if done_marker.exists():
            print(f"  skip {game} seed={seed} (DONE exists at {out_dir})")
            continue
        cfg = dataclasses.replace(
            _load_exp_config(game),
            seed=seed,
            out_dir=str(out_dir),
            verbose=True,
            centered_mean_legal_only=args.legal_mean,
        )
        print(f"  run  {game} seed={seed} -> {out_dir}")
        # On failure no DONE marker is written, so the next invocation resumes
        # at exactly this run. Interrupts (KeyboardInterrupt) propagate.
        run_experiment(cfg)
        done_marker.write_text("ok\n", encoding="utf-8")
        n_done += 1
    print(f"reproduce_paper: finished {n_done} new runs ({len(planned) - n_done} skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
