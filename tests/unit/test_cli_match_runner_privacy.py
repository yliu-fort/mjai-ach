"""Privacy regression tests for the match runner (bug family B, §5).

Guards INV-1: in an imperfect-information game, a human spectator must NEVER see
any player's private information (hole card, die, hidden simultaneous choice)
during a robot's turn. Before the fix, ``_one_step`` rendered the robot's turn
with ``observer_player=player`` (the robot's own view), which leaks the
opponent's private info to the spectating human. The fix renders the
**public** view (``render_public``) during robot turns and announces the move.

These tests run full interactive matches (human vs robot) on the imperfect-info
games and assert that no private information appears in any output emitted
during a robot's turn.
"""

from __future__ import annotations

import importlib
import random
import re

import pytest

from mjai.agents.tabular import uniform_tabular
from mjai.cli.match_runner import MatchRunner
from mjai.games.loader import load_game
from mjai.utils import gpu_assert


@pytest.fixture(autouse=True)
def _cpu_mode():
    gpu_assert.reset_for_tests()
    gpu_assert.require_cpu()
    yield
    gpu_assert.reset_for_tests()


def _run_human_vs_robot(game_name: str, human_seat: int) -> list[str]:
    """Run one interactive match; return all output lines.

    The human always feeds the first legal action as a numeric string (read
    from the parser's prompt, which lists the legal ids); the robot plays a
    fixed uniform policy. Determinism is pinned by the rng.
    """
    spec = load_game(game_name)
    renderer = importlib.import_module(f"mjai.cli.renderers.{game_name}").create()
    parser = importlib.import_module(f"mjai.cli.input_parsers.{game_name}").create()
    robot = uniform_tabular(spec.num_actions, seed=1)
    seats: list = [None, None]
    seats[human_seat] = "human"
    seats[1 - human_seat] = robot

    outputs: list[str] = []

    def input_fn() -> str:
        # Extract the first legal action id from the most recent prompt line.
        # Two prompt shapes occur: "id=name, ..." (kuhn/leduc) and
        # "[id, id, ...]" (liars/goof). Handle both.
        for line in reversed(outputs):
            # Shape 1: "0=pass..." -> first id before '='.
            m = re.search(r"(\d+)=", line)
            if m:
                return m.group(1)
            # Shape 2: "[0, 1, 2, ...]" -> first id in the bracketed list.
            m = re.search(r"\[(\d+)", line)
            if m:
                return m.group(1)
        return "0"

    def _capture(s: str) -> None:
        outputs.append(s)

    runner = MatchRunner(
        spec,
        renderer,
        parser,
        seats,  # type: ignore[arg-type]
        input_fn=input_fn,
        output_fn=_capture,
        rng=random.Random(0),
    )
    runner.run(mode="interactive")
    return outputs


# --------------------------------------------------------------------------- #
# Kuhn: robot's hole card must never appear during the robot's turn
# --------------------------------------------------------------------------- #


def test_kuhn_robot_turn_does_not_leak_robot_card():
    """Human=P0, robot=P1. The robot's card is private; the human must never
    see it. Before the fix the robot's turn rendered P1's view, leaking P1's
    card to the human spectator (MR1, INV-1)."""
    outputs = _run_human_vs_robot("kuhn", human_seat=0)
    # The robot's turn renders the public view; no "your card" line may appear
    # in a robot-turn block.
    robot_turn_blocks = [o for o in outputs if "public view" in o or "(robot)" in o]
    assert robot_turn_blocks, "expected at least one robot-turn output"
    for block in robot_turn_blocks:
        assert "your card" not in block.lower(), f"robot-turn output leaked private info:\n{block}"
    # Sanity: the human's own turn DID show their card (proves the test runs).
    assert any("your card" in o for o in outputs)


def test_kuhn_robot_move_is_announced():
    """MR2: after a robot moves, a one-line announcement is printed so the
    spectating human knows what happened."""
    outputs = _run_human_vs_robot("kuhn", human_seat=0)
    announces = [o for o in outputs if "(robot)" in o]
    assert announces, "no robot announcement emitted"
    assert all("Player 1 (robot):" in a for a in announces)


# --------------------------------------------------------------------------- #
# Leduc: neither player's hole card leaks during robot turns
# --------------------------------------------------------------------------- #


def test_leduc_robot_turn_does_not_leak_hole_cards():
    outputs = _run_human_vs_robot("leduc", human_seat=0)
    for o in outputs:
        if "public view" in o or "(robot)" in o:
            # Public view shows only the public board card + history, never a
            # private "Private card:" line.
            assert "Private card:" not in o, f"robot-turn leaked hole card:\n{o}"


# --------------------------------------------------------------------------- #
# Liar's Dice: no player's die leaks during robot turns
# --------------------------------------------------------------------------- #


def test_liars_dice_robot_turn_does_not_leak_dice():
    outputs = _run_human_vs_robot("liars_dice1", human_seat=0)
    for o in outputs:
        if "public view" in o or "(robot)" in o:
            assert "your die" not in o.lower(), f"robot-turn leaked die:\n{o}"


# --------------------------------------------------------------------------- #
# Goofspiel: blind entry never reveals opponent's simultaneous bid
# --------------------------------------------------------------------------- #


def test_goofspiel_robot_turn_does_not_reveal_bid():
    """Goofspiel is simultaneous; the human enters blind and the robot plays
    blind. The robot's pending bid must never be shown to the human."""
    outputs = _run_human_vs_robot("goofspiel5_ii", human_seat=0)
    for o in outputs:
        # The human's own render may say 'blind'; the point is no specific
        # opponent bid value is announced before resolution.
        assert "opponent played" not in o.lower()
        assert "opponent bid" not in o.lower()


# --------------------------------------------------------------------------- #
# render_public exists on every renderer (Protocol contract)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "game_name",
    ["brps", "goofspiel5_ii", "kuhn", "leduc", "liars_dice1", "oshi_zumo", "ttt"],
)
def test_every_renderer_implements_render_public(game_name):
    """AGENTS.md §4 / INV-1: every game must expose render_public so a human
    can safely spectate a robot's turn."""
    from mjai.games.loader import GAME_STRINGS

    spec = load_game(game_name)
    renderer = importlib.import_module(f"mjai.cli.renderers.{game_name}").create()
    state = spec.new_state()
    # Resolve any leading chance so the state is non-terminal for most games.
    while state.is_chance_node() and not state.is_terminal():
        outcomes = state.chance_outcomes()
        state.apply_action(outcomes[0][0])
    if state.is_terminal():
        return  # one-shot games may be terminal immediately; skip
    out = renderer.render_public(state)
    assert isinstance(out, str) and len(out) > 0
    # The public view must never contain the words 'your card'/'your die' (those
    # are private-info markers from the private render path).
    assert "your card" not in out.lower()
    assert "your die" not in out.lower()
    # Sanity: load_game recognized the game (suppress unused-import noise).
    assert game_name in GAME_STRINGS
