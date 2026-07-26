"""``mjai-play`` entry point: the interactive match CLI (AGENTS.md §1 D10).

Menu-driven: select env -> assign seats (human | load policy) -> mode
(interactive | auto_fast | auto_step) -> run match -> show returns ->
replay/quit. Launched via the ``mjai-play`` console-script or
``python -m mjai.cli``.

The per-game renderer/parser is resolved from the game short name; if a game
lacks either, the CLI refuses to start it (AGENTS.md §4 — adding a game
requires both). Simultaneous-move games use blind human entry.

Checkpoint loading is metadata-driven via
:func:`mjai.agents.policy_factory.load_policy_from_checkpoint` (F1); the CLI
never reconstructs architectures itself and never downcasts the returned
Policy (§3.3). Load failures surface as one-line errors with a retry/quit
prompt — no tracebacks.
"""

from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path
from typing import NoReturn

from mjai.agents.base import Policy
from mjai.agents.policy_factory import CheckpointLoadError, load_policy_from_checkpoint
from mjai.cli.game_registry import list_games
from mjai.cli.interfaces import GameRenderer, HumanInputParser
from mjai.cli.match_runner import MatchRunner, Seat, random_policy_for
from mjai.cli.policy_registry import (
    compatible_with,
    filter_labels,
    list_policies,
    page,
)
from mjai.games.loader import GameSpec, load_game
from mjai.utils import gpu_assert

RUNS_ROOT = Path("runs")


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


def _pick_policy(spec: GameSpec) -> Policy | None:
    """Interactive checkpoint picker for one seat (F5); None = back to seat menu.

    Shows only checkpoints trained on ``spec`` with a matching obs/action
    space, newest ~20 first; any non-numeric input substring-filters the list.
    'q' (or EOF) quits the CLI cleanly. A failed load prints a one-line error
    and re-prompts instead of raising (F1).
    """
    entries = list_policies(RUNS_ROOT, game=spec.name)
    compatible, incompatible = compatible_with(
        entries, obs_size=spec.obs_size, num_actions=spec.num_actions
    )
    if not compatible:
        print(
            f"No usable checkpoints for {spec.name!r} under {RUNS_ROOT}/ "
            f"({len(entries)} found, {len(incompatible)} with a mismatched obs/action space)."
        )
        print("Pick a different seat type instead (human or random).")
        return None
    if incompatible:
        print(
            f"Note: skipped {len(incompatible)} {spec.name} checkpoint(s) whose "
            f"obs/action space does not match this game."
        )
    # Playing a neural policy is inference-only; the CLI deliberately uses CPU
    # (explicit opt-in per AGENTS.md §1 D6, not a silent fallback).
    gpu_assert.require_cpu()

    filtered = compatible
    while True:
        shown, remaining = page(filtered)
        for i, pe in enumerate(shown):
            print(f"  [{i}] {pe.label}")
        if remaining:
            print(f"  … and {remaining} more — type any text to filter the full list.")
        raw = _read_line(f"Select policy [0-{len(shown) - 1}], filter text, or 'q' to quit: ")
        if raw.lower() == "q":
            _quit()
        if raw.isdigit():
            idx = int(raw)
            if not 0 <= idx < len(shown):
                print(f"  must be in [0, {len(shown) - 1}]")
                continue
            chosen = shown[idx]
            try:
                policy = load_policy_from_checkpoint(chosen.path)
            except CheckpointLoadError as e:
                print(f"  Could not load that checkpoint: {e}")
                print("  Pick another one, or 'q' to quit.")
                continue
            print(f"Loaded: {chosen.label}")
            return policy
        # Non-numeric input narrows the list; empty result restores it.
        narrowed = filter_labels(compatible, raw)
        if not narrowed:
            print(f"  No checkpoints match {raw!r}; showing the full list again.")
            filtered = compatible
        else:
            filtered = narrowed


def _assign_seat(spec: GameSpec, seat_idx: int) -> Seat:
    """Prompt for one seat; loops until a usable seat is chosen."""
    while True:
        print(f"\n-- Seat {seat_idx} --")
        print("  [0] Human")
        print("  [1] Load saved policy")
        print("  [2] Random policy")
        kind = _read_int("Choose: ", 0, 2)
        if kind == 0:
            return "human"
        if kind == 2:
            return random_policy_for(spec)
        policy = _pick_policy(spec)
        if policy is not None:
            return policy
        # No usable checkpoint for this game: re-offer the seat menu.


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

    # Seat assignment — one prompt per seat the game actually has (D13).
    seats: list[Seat] = [_assign_seat(spec, seat_idx) for seat_idx in range(spec.num_players)]

    print("\n-- Mode --")
    print("  [0] interactive (step through, render each turn)")
    print("  [1] auto fast (all policies, show only final)")
    print("  [2] auto step (all policies, render + pause each turn)")
    mode_idx = _read_int("Choose: ", 0, 2)
    mode = ["interactive", "auto_fast", "auto_step"][mode_idx]
    # auto_fast/auto_step require every seat to be a policy.
    if mode != "interactive" and any(s == "human" for s in seats):
        print("Auto modes require every seat to be a policy. Falling back to interactive.")
        mode = "interactive"

    runner = MatchRunner(spec, renderer, parser, seats, rng=random.Random(0))
    print()
    result = runner.run(mode=mode)
    tally = ", ".join(f"seat {i} = {r:+.3f}" for i, r in enumerate(result.returns))
    print(f"\nReturns: {tally}  ({result.n_steps} steps)")
    return 0


def _read_int(prompt: str, lo: int, hi: int) -> int:
    while True:
        raw = _read_line(prompt)
        try:
            v = int(raw)
        except ValueError:
            print(f"  enter a number in [{lo}, {hi}]")
            continue
        if lo <= v <= hi:
            return v
        print(f"  must be in [{lo}, {hi}]")


def _read_line(prompt: str) -> str:
    """input() that treats EOF (scripted stdin exhausted) as a clean exit."""
    try:
        return input(prompt).strip()
    except EOFError:
        print("\nEnd of input; exiting.")
        _quit()


def _quit() -> NoReturn:
    print("Bye.")
    raise SystemExit(0)


if __name__ == "__main__":
    sys.exit(main())
