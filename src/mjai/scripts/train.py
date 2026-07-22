"""``mjai-train`` entry point: train one experiment from a YAML config.

Usage::

    uv run mjai-train --config configs/exp/kuhn_ach_mirror.yaml
    uv run mjai-train --game kuhn --algo ach --mode mirror --steps 500 --out runs/kuhn_ach_mirror
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from mjai.scripts.experiment import ExperimentConfig, run_experiment


def _load_config(path: str) -> ExperimentConfig:
    """Load a YAML file into an :class:`ExperimentConfig`."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig(**data)


def _bare_defaults_warning() -> str:
    """Loud notice for the no-``--config`` path (audit Q1, docs/audit_4q_2026-07-22.md).

    The dataclass defaults are the NN-tuned 2x2-matrix defaults, not the ACH
    paper reproduction settings, and the two silently differ. Warn instead of
    switching silently (AGENTS.md §11) — the matrix experiments rely on them.
    """
    return (
        "*** WARNING: no --config given; using ExperimentConfig built-in defaults.\n"
        "*** These are NN-tuned matrix defaults, NOT the ACH paper reproduction\n"
        "*** settings — they differ in policy_kind, learning_rate, value_coef,\n"
        "*** entropy_coef, max_grad_norm, and batch/run-length accounting.\n"
        "*** For paper reproduction use:\n"
        "***   uv run mjai-train --config configs/exp/<game>_ach_mlp_mirror.yaml\n"
        "***   uv run python -m mjai.scripts.reproduce_paper   # full 24-run protocol"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one mjai experiment.")
    parser.add_argument("--config", help="Path to a YAML experiment config.")
    parser.add_argument("--game", help="Override game short name.")
    parser.add_argument("--algo", choices=["ppo", "ach"], help="Override algo.")
    parser.add_argument("--mode", choices=["mirror", "league"], help="Override self-play mode.")
    parser.add_argument("--steps", type=int, help="Override n_steps.")
    parser.add_argument("--out", help="Override output directory.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (AGENTS.md §1 D6).")
    args = parser.parse_args(argv)

    if args.cpu:
        from mjai.utils import gpu_assert

        gpu_assert.require_cpu()

    if args.config:
        cfg = _load_config(args.config)
    else:
        if not (args.game and args.algo and args.mode):
            parser.error("Either --config or all of --game/--algo/--mode must be given.")
        print(_bare_defaults_warning(), file=sys.stderr)
        cfg = ExperimentConfig(
            game=args.game,
            algo=args.algo,
            self_play_mode=args.mode,
            out_dir=args.out or f"runs/{args.game}_{args.algo}_{args.mode}",
        )

    # Apply overrides.
    overrides = {
        "game": args.game,
        "algo": args.algo,
        "self_play_mode": args.mode,
        "n_steps": args.steps,
        "out_dir": args.out,
    }
    import dataclasses

    cfg = dataclasses.replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
    # CLI always shows live progress.
    cfg = dataclasses.replace(cfg, verbose=True)

    print(
        f"mjai-train: {cfg.game}/{cfg.algo}/{cfg.self_play_mode} for {cfg.n_steps} steps -> {cfg.out_dir}"
    )
    out = run_experiment(cfg)
    print(f"mjai-train: done. Checkpoints under {out}/checkpoints/, TensorBoard under {out}/tb/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
