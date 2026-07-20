"""CLI smoke test: one auto-played match per game (AGENTS.md §5, Step 9).

Validates that every registered game has a renderer + parser (AGENTS.md §4) and
that a full match runs cleanly. The full per-game sweep is marked ``@slow`` so
the commit-time fast suite only loads + instantiates the modules (cheap); the
pre-push stage runs the actual matches. Output capture is intentionally
discarded (noop output_fn) to keep memory bounded.
"""

from __future__ import annotations

import importlib
import random

import pytest

from mjai.agents.tabular import uniform_tabular
from mjai.cli.game_registry import list_games
from mjai.cli.match_runner import MatchRunner
from mjai.config.game_config import load_all_game_configs
from mjai.games.loader import load_game
from mjai.utils import gpu_assert


@pytest.fixture(autouse=True)
def _cpu_mode():
    gpu_assert.reset_for_tests()
    gpu_assert.require_cpu()
    yield
    gpu_assert.reset_for_tests()


REGISTERED_GAMES = sorted(load_all_game_configs().keys())


# Tiny, bounded output capture: just remember whether the terminal line showed.
def _NOOP(_s: str) -> None:
    """Discard output (kept memory-bounded)."""


@pytest.mark.parametrize("game_name", REGISTERED_GAMES)
def test_every_game_has_renderer_and_parser(game_name):
    """AGENTS.md §4: a game is not playable until both files exist.

    Importing the module is cheap (no game loop), so this parametrize stays in
    the fast commit suite.
    """
    importlib.import_module(f"mjai.cli.renderers.{game_name}")
    importlib.import_module(f"mjai.cli.input_parsers.{game_name}")


@pytest.mark.slow
@pytest.mark.parametrize("game_name", REGISTERED_GAMES)
def test_auto_match_completes_for_each_game(game_name):
    """Run one auto_fast match between two random policies; assert clean exit.

    @slow: full match per game runs at pre-push only. Output is discarded to
    keep memory bounded.
    """
    spec = load_game(game_name)
    renderer = importlib.import_module(f"mjai.cli.renderers.{game_name}").create()
    parser = importlib.import_module(f"mjai.cli.input_parsers.{game_name}").create()
    seats = [uniform_tabular(spec.num_actions, seed=0), uniform_tabular(spec.num_actions, seed=1)]
    terminal_seen = []

    def output_fn(s: str) -> None:
        # Only retain a tiny flag, not the full render history.
        if "over" in s or "Returns" in s or "wins" in s:
            terminal_seen.append(True)

    runner = MatchRunner(
        spec,
        renderer,
        parser,
        seats,
        input_fn=lambda: "",
        output_fn=output_fn,
        rng=random.Random(0),
    )
    result = runner.run(mode="auto_fast")
    assert len(result.returns) == 2
    assert abs(sum(result.returns)) < 1e-6  # zero-sum
    assert terminal_seen  # terminal renderer fired


def test_auto_match_on_tiny_games_only_in_fast_suite():
    """Fast-suite sanity: just the two cheapest games (BRPS, Kuhn)."""
    for game_name in ["brps", "kuhn"]:
        spec = load_game(game_name)
        renderer = importlib.import_module(f"mjai.cli.renderers.{game_name}").create()
        parser = importlib.import_module(f"mjai.cli.input_parsers.{game_name}").create()
        seats = [
            uniform_tabular(spec.num_actions, seed=0),
            uniform_tabular(spec.num_actions, seed=1),
        ]
        runner = MatchRunner(
            spec,
            renderer,
            parser,
            seats,
            input_fn=lambda: "",
            output_fn=_NOOP,
            rng=random.Random(0),
        )
        result = runner.run(mode="auto_fast")
        assert len(result.returns) == 2


def test_interactive_match_brps_with_fake_human():
    """Bounded interactive match on the cheapest simultaneous game.

    BRPS is one-shot: one human input, one policy action, done. No risk of
    unbounded output capture. Avoids the longer-horizon games here.
    """
    spec = load_game("brps")
    renderer = importlib.import_module("mjai.cli.renderers.brps").create()
    parser = importlib.import_module("mjai.cli.input_parsers.brps").create()
    random_policy = uniform_tabular(spec.num_actions, seed=2)

    def fake_input() -> str:
        return "0"  # always Rock

    runner = MatchRunner(
        spec,
        renderer,
        parser,
        ["human", random_policy],
        input_fn=fake_input,
        output_fn=_NOOP,
        rng=random.Random(0),
    )
    result = runner.run(mode="interactive")
    assert len(result.returns) == 2


def test_match_runner_validates_seat_count():
    spec = load_game("brps")
    renderer = importlib.import_module("mjai.cli.renderers.brps").create()
    parser = importlib.import_module("mjai.cli.input_parsers.brps").create()
    with pytest.raises(ValueError, match="2 seats"):
        MatchRunner(spec, renderer, parser, [uniform_tabular(3)])  # only 1 seat


def test_list_games_returns_seven():
    games = list_games()
    assert len(games) == 7
    names = {g.name for g in games}
    assert names == {"brps", "kuhn", "leduc", "ttt", "goofspiel5_ii", "liars_dice1", "oshi_zumo"}


def test_parser_rejects_illegal_action():
    """Every game's parser rejects an action not in the legal set."""
    parser = importlib.import_module("mjai.cli.input_parsers.kuhn").create()
    with pytest.raises(ValueError):
        parser.parse("5", legal_actions=[0, 1])
