"""``mjai-play`` entry point: the interactive match CLI (AGENTS.md §1 D10).

Menu-driven: select env -> assign seats (human | load policy) -> mode
(interactive | auto_fast | auto_step) -> run match -> show returns ->
replay/quit. Launched via the ``mjai-play`` console-script or
``python -m mjai.cli``.

The per-game renderer/parser is resolved from the game short name; if a game
lacks either, the CLI refuses to start it (AGENTS.md §4 — adding a game
requires both). Simultaneous-move games use blind human entry.
"""

from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path

from mjai.agents.base import Policy
from mjai.cli.game_registry import list_games
from mjai.cli.interfaces import GameRenderer, HumanInputParser
from mjai.cli.match_runner import MatchRunner, Seat, random_policy_for
from mjai.cli.policy_registry import list_policies
from mjai.games.loader import load_game


def _load_renderer(game_name: str) -> GameRenderer:
    try:
        mod = importlib.import_module(f"mjai.cli.renderers.{game_name}")
    except ImportError as e:
        raise SystemExit(f"No renderer for game {game_name!r} (AGENTS.md §4). {e}") from e
    renderer: GameRenderer = mod.create()
    return renderer


def _load_parser(game_name: str) -> HumanInputParser:
    try:
        mod = importlib.import_module(f"mjai.cli.input_parsers.{game_name}")
    except ImportError as e:
        raise SystemExit(f"No input parser for game {game_name!r} (AGENTS.md §4). {e}") from e
    parser: HumanInputParser = mod.create()
    return parser


def _load_policy_from_manifest(ckpt_dir: Path) -> Policy:
    """Reconstruct a policy from a checkpoint dir."""
    from mjai.agents.ckpt_io import read_manifest

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
        raise SystemExit(f"Unknown policy_kind: {manifest.policy_kind}")
    p.load(str(ckpt_dir / manifest.weight_filename()))
    return p


def main(argv: list[str] | None = None) -> int:
    games = list_games()
    if not games:
        print("No games found. Add YAML under configs/games/ (AGENTS.md §4).")
        return 1

    print("\n=== mjai-play ===")
    print("Available games:")
    for i, g in enumerate(games):
        print(f"  [{i}] {g.name:16s}  {g.notes}")
    game_idx = _read_int("Select game: ", 0, len(games) - 1)
    chosen = games[game_idx]
    print(f"\nSelected: {chosen.name}  ({chosen.game_string})")

    spec = load_game(chosen.name)
    renderer = _load_renderer(chosen.name)
    parser = _load_parser(chosen.name)

    # Seat assignment.
    seats: list[Seat] = []
    for seat_idx in range(2):
        print(f"\n-- Seat {seat_idx} --")
        print("  [0] Human")
        print("  [1] Load saved policy")
        print("  [2] Random policy")
        kind = _read_int("Choose: ", 0, 2)
        if kind == 0:
            seats.append("human")
        elif kind == 1:
            pols = list_policies()
            if not pols:
                print("No saved policies found under runs/. Using random.")
                seats.append(random_policy_for(spec))
                continue
            for i, pe in enumerate(pols):
                print(f"  [{i}] {pe.label}")
            pi = _read_int("Select policy: ", 0, len(pols) - 1)
            seats.append(_load_policy_from_manifest(pols[pi].path))
        else:
            seats.append(random_policy_for(spec))

    print("\n-- Mode --")
    print("  [0] interactive (step through, render each turn)")
    print("  [1] auto fast (both policies, show only final)")
    print("  [2] auto step (both policies, render + pause each turn)")
    mode_idx = _read_int("Choose: ", 0, 2)
    mode = ["interactive", "auto_fast", "auto_step"][mode_idx]
    # auto_fast/auto_step require both seats to be policies.
    if mode != "interactive" and any(s == "human" for s in seats):
        print("Auto modes require both seats to be policies. Falling back to interactive.")
        mode = "interactive"

    runner = MatchRunner(spec, renderer, parser, seats, rng=random.Random(0))
    print()
    result = runner.run(mode=mode)
    print(
        f"\nReturns: seat 0 = {result.returns[0]:+.3f}, seat 1 = {result.returns[1]:+.3f}  ({result.n_steps} steps)"
    )
    return 0


def _read_int(prompt: str, lo: int, hi: int) -> int:
    while True:
        try:
            v = int(input(prompt))
        except (ValueError, EOFError):
            print(f"  enter a number in [{lo}, {hi}]")
            continue
        if lo <= v <= hi:
            return v
        print(f"  must be in [{lo}, {hi}]")


if __name__ == "__main__":
    sys.exit(main())
