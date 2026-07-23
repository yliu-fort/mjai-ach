"""Notebook-facing composition of the final-policy view.

:mod:`mjai.eval.policy_table` builds the view but cannot sample self-play
visits itself: ``mjai.eval`` sits BELOW ``mjai.pipeline`` in the layering
(pyproject import-linter contract), so it may not reach for a rollout worker.
This module is where the two meet — the same place ``tools/arm_cache.py``
lives, one layer out from the package, imported by the generated notebooks.

Two entry points:

  - :func:`view_for_arm` — one arm directory in, a ready
    :class:`~mjai.eval.policy_table.PolicyView` out, visits included.
  - :func:`render_arms` — one panel per arm on a shared figure, which is what
    the A/B and theta notebooks actually show.

Run: ``python tools/policy_view.py --run runs/nb_ab/kuhn_mirror/seed_0``
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path

from mjai.eval.policy_table import (
    PolicyView,
    PolicyViewError,
    load_run_policy,
    plot_policy,
    policy_view,
    root_policy,
    visit_counts_from_batch,
    write_csv,
)

# Self-play episodes used to rank info states by how often they are reached.
# 400 keeps the notebook cell at well under a second per arm on these games
# while still separating the opening states from the deep tail.
DEFAULT_VISIT_EPISODES = 400


def slug(label: str) -> str:
    """Arm label -> filename fragment: "mirror theta=0.5  seed=0" -> "mirror_theta0p5_seed0"."""
    text = label.replace(".", "p").replace("=", "")
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def sample_visits(run_dir: str | Path, *, checkpoint: str, episodes: int, seed: int = 0):
    """Self-play ``episodes`` games with the run's own policy; count decisions.

    Plays through :class:`~mjai.pipeline.rollout.RolloutWorkerCore` rather than
    walking the tree here: chance sampling and simultaneous joint actions are
    easy to get subtly wrong and there is exactly one implementation of them.
    """
    from mjai.pipeline.rollout import RolloutConfig, RolloutWorkerCore

    policy, spec, _ = load_run_policy(run_dir, checkpoint)
    runner = RolloutWorkerCore(
        spec,
        learner_player=0,
        config=RolloutConfig(n_episodes=episodes, seed=seed, target_samples=None),
    )
    return visit_counts_from_batch(runner.run_episode(policy, policy))


def view_for_arm(
    run_dir: str | Path,
    *,
    checkpoint: str = "best",
    episodes: int = DEFAULT_VISIT_EPISODES,
    seed: int = 0,
) -> PolicyView:
    """The finished arm's policy, ranked by how often each info state is reached."""
    counts = (
        sample_visits(run_dir, checkpoint=checkpoint, episodes=episodes, seed=seed)
        if (episodes > 0)
        else None
    )
    return policy_view(run_dir, checkpoint=checkpoint, visit_counts=counts, episodes=episodes)


def render_arms(
    arms: Sequence[tuple[str, Path]],
    out_path: str | Path,
    *,
    checkpoint: str = "best",
    episodes: int = DEFAULT_VISIT_EPISODES,
    player: int = 0,
    seed: int = 0,
) -> tuple[Path | None, dict[str, PolicyView], dict[str, str]]:
    """One panel per (label, run_dir); returns (figure path, views, skip reasons).

    Arms that cannot produce a view (a game whose tree will not enumerate, an
    arm with no checkpoint yet) are reported in the third return value instead
    of being dropped silently.
    """
    from matplotlib.figure import Figure

    views: dict[str, PolicyView] = {}
    skipped: dict[str, str] = {}
    for label, run in arms:
        try:
            views[label] = view_for_arm(run, checkpoint=checkpoint, episodes=episodes, seed=seed)
        except (PolicyViewError, FileNotFoundError) as e:
            skipped[label] = str(e)
    if not views:
        return None, views, skipped
    ncols = min(3, len(views))
    nrows = (len(views) + ncols - 1) // ncols
    fig = Figure(figsize=(6.0 * ncols, 4.4 * nrows))
    grid = fig.subplots(nrows, ncols, squeeze=False)
    axes = [grid[i // ncols][i % ncols] for i in range(nrows * ncols)]
    for ax, (label, view) in zip(axes, views.items(), strict=False):
        plot_policy(view, ax=ax, player=player)
        ax.set_title(f"{label}\n({view.game}, seat {player}, {view.mode})", fontsize=9)
    for ax in axes[len(views) :]:
        ax.set_visible(False)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    return out, views, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--run", required=True, help="Arm directory (contains checkpoints/).")
    parser.add_argument("--checkpoint", default="best", help="best | last | step_N")
    parser.add_argument("--episodes", type=int, default=DEFAULT_VISIT_EPISODES)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--csv", help="Also dump the full table here.")
    args = parser.parse_args()

    try:
        view = view_for_arm(args.run, checkpoint=args.checkpoint, episodes=args.episodes)
    except PolicyViewError as e:
        # Two different failures land here: an un-enumerable game (fall back to
        # the opening distribution, which always works) and a run with no
        # checkpoint at all (nothing to fall back to -- say so and exit 1).
        print(f"no full policy table: {e}")
        try:
            print(json.dumps(root_policy(args.run, checkpoint=args.checkpoint), indent=2))
        except PolicyViewError as inner:
            print(f"and no policy to load either: {inner}")
            return 1
        return 0
    print(f"{view.game}: {len(view.labels)} info states x {len(view.action_names)} actions")
    print(f"checkpoint: {view.checkpoint}")
    for i in view.top_rows(args.top_k, player=0):
        probs = " ".join(
            f"{n}={view.probs[i][a]:.3f}"
            for a, n in enumerate(view.action_names)
            if view.legal[i][a]
        )
        visits = "" if view.visits is None else f"[{int(view.visits[i])}x] "
        print(f"  {visits}{view.labels[i][:60]:60s} {probs}")
    if args.csv:
        print(f"wrote {write_csv(view, args.csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
