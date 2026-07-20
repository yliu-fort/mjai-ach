"""``mjai-eval`` entry point: evaluate a trained run's checkpoints.

Loads the latest (or a specified) checkpoint from a run directory, reconstructs
the policy, and runs the full eval toolkit (exploitability/NashConv/exact-Nash
as applicable) + a cross-play matrix across the run's checkpoints.

Usage::

    uv run mjai-eval --run runs/kuhn_ach_mirror
    uv run mjai-eval --run runs/kuhn_ach_mirror --checkpoint step_500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mjai.agents.base import Policy
from mjai.agents.ckpt_io import CheckpointManifest, discover_checkpoints, read_manifest
from mjai.eval.crossplay import (
    cross_play_matrix,
    forgetting_metric,
    nontransitivity_score,
    worst_case_win_rate,
)
from mjai.eval.nash import evaluate_equilibrium
from mjai.games.loader import load_game
from mjai.pipeline.rollout import RolloutConfig, RolloutWorkerCore


def _load_policy(ckpt_dir: Path) -> tuple[Policy, CheckpointManifest]:
    """Reconstruct the policy stored at ``ckpt_dir`` from its manifest."""
    manifest = read_manifest(ckpt_dir)
    p: Policy
    if manifest.policy_kind == "tabular":
        from mjai.agents.tabular import TabularPolicy

        p = TabularPolicy(num_actions=manifest.num_actions, seed=0)
    elif manifest.policy_kind == "mlp":
        from mjai.utils import gpu_assert

        gpu_assert.require_cpu()
        from mjai.agents.mlp import MLPSharedActorCritic

        p = MLPSharedActorCritic(
            obs_size=manifest.obs_size, num_actions=manifest.num_actions, seed=0
        )
    else:
        raise ValueError(f"Unknown policy_kind in manifest: {manifest.policy_kind}")
    p.load(str(ckpt_dir / manifest.weight_filename()))
    return p, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained mjai run.")
    parser.add_argument("--run", required=True, help="Run directory (contains checkpoints/).")
    parser.add_argument("--checkpoint", help="Specific checkpoint name (default: latest).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    args = parser.parse_args(argv)

    if args.cpu:
        from mjai.utils import gpu_assert

        gpu_assert.require_cpu()

    run_dir = Path(args.run)
    ckpts = discover_checkpoints(run_dir / "checkpoints")
    if not ckpts:
        print(f"mjai-eval: no checkpoints found under {run_dir}/checkpoints/", file=sys.stderr)
        return 1

    # Pick the requested or the latest checkpoint.
    if args.checkpoint:
        target = run_dir / "checkpoints" / args.checkpoint
        ckpt_dir, manifest = target, read_manifest(target)
    else:
        ckpt_dir, manifest = ckpts[-1]

    print(
        f"mjai-eval: evaluating {ckpt_dir.name} ({manifest.game}/{manifest.algo}/{manifest.self_play_mode})"
    )
    policy, _ = _load_policy(ckpt_dir)

    spec = load_game(manifest.game)
    metrics = evaluate_equilibrium(spec, policy)
    print("Equilibrium metrics:")
    for k, v in metrics.items():
        print(f"  {k:25s} {v:.6g}")

    # Cross-play across all checkpoints of this run.
    policies = []
    names = []
    for cdir, _cman in ckpts:
        try:
            p, _ = _load_policy(cdir)
            policies.append(p)
            names.append(cdir.name)
        except Exception as e:
            print(f"  (skipped {cdir.name}: {e})", file=sys.stderr)
    if len(policies) >= 2:
        runner = RolloutWorkerCore(
            spec, learner_player=0, config=RolloutConfig(n_episodes=30, seed=0)
        )
        cpr = cross_play_matrix(spec, policies, runner, n_episodes=30, policy_names=names)
        print("\nCross-play summary:")
        print(f"  worst_case_win_rate (final vs pool): {worst_case_win_rate(cpr):.3f}")
        print(f"  nontransitivity_score:               {nontransitivity_score(cpr):.3f}")
        early = list(range(1, len(policies)))  # all but the final (index 0)
        print(
            f"  forgetting_metric:                   {forgetting_metric(cpr, early_indices=early):.3f}"
        )

    # Save the metrics to disk for the notebook to pick up.
    out = run_dir / "eval_results.json"
    out.write_text(json.dumps({"checkpoint": str(ckpt_dir.name), **metrics}, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
