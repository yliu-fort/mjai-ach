"""Unit tests for the Oshi-Zumo renderer + parser (§5).

Regression coverage for the wrestler-position bug (bug family A), verified
empirically against pyspiel 2.0.1:

  - The renderer queried the wrestler cell from observation slots ``[12:15]``,
    which are empty in this build. The board one-hot is at ``[15:18]`` (cell =
    hot index; 0/1/2 for size=3). So the old renderer always printed
    "Wrestler at cell -1", even after the wrestler moved.

Coin reads were already correct: P0 coins one-hot at ``[0:6]`` (hot index ==
coins), P1 coins one-hot at ``[6:12]`` (coins = hot index - 6). These are
covered here too so a future regression is caught.
"""

from __future__ import annotations

import pyspiel
import pytest

from mjai.cli.input_parsers.oshi_zumo import create as create_parser
from mjai.cli.renderers.oshi_zumo import create as create_renderer


def _game() -> pyspiel.Game:
    return pyspiel.load_game("oshi_zumo(coins=5,size=3,horizon=20)")


def _summary_line(rendered: str) -> str:
    for line in rendered.splitlines():
        if "Your coins:" in line:
            return line
    raise AssertionError(f"no summary line in:\n{rendered}")


# --------------------------------------------------------------------------- #
# Renderer: wrestler position
# --------------------------------------------------------------------------- #


def test_wrestler_starts_at_center():
    """Regression: old code read the wrong slot region and always showed -1.

    The wrestler starts at the center cell of the size=3 ring (cell index 1).
    """
    s = _game().new_initial_state()
    out = create_renderer().render(s, observer_player=0)
    assert "Wrestler at cell 1." in _summary_line(out)
    assert "cell -1" not in out


def test_wrestler_moves_toward_opponent_when_p0_outbids():
    """P0 bids 5, P1 bids 0: wrestler pushed toward P1 (cell 2)."""
    s = _game().new_initial_state()
    s.apply_actions([5, 0])
    out = create_renderer().render(s, observer_player=0)
    assert "Wrestler at cell 2." in _summary_line(out)


def test_wrestler_moves_toward_self_when_p1_outbids():
    """P0 bids 0, P1 bids 5: wrestler pushed toward P0 (cell 0)."""
    s = _game().new_initial_state()
    s.apply_actions([0, 5])
    out = create_renderer().render(s, observer_player=0)
    assert "Wrestler at cell 0." in _summary_line(out)


# --------------------------------------------------------------------------- #
# Renderer: coins
# --------------------------------------------------------------------------- #


def test_coins_start_full():
    s = _game().new_initial_state()
    out = create_renderer().render(s, observer_player=0)
    line = _summary_line(out)
    assert "Your coins: 5" in line and "Opponent coins: 5" in line


def test_coins_deplete_after_bids():
    """P0 bids 3, P1 bids 1: P0 has 2 left, P1 has 4 left."""
    s = _game().new_initial_state()
    s.apply_actions([3, 1])
    out = create_renderer().render(s, observer_player=0)
    line = _summary_line(out)
    assert "Your coins: 2" in line and "Opponent coins: 4" in line


def test_identical_view_for_both_observers():
    """Oshi-Zumo is perfect-info; both observers see the same board."""
    s = _game().new_initial_state()
    s.apply_actions([2, 1])
    out0 = create_renderer().render(s, observer_player=0)
    out1 = create_renderer().render(s, observer_player=1)
    assert _summary_line(out0) == _summary_line(out1)


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def test_parser_accepts_bid():
    p = create_parser()
    assert p.parse("3", [0, 1, 2, 3, 4, 5]) == 3


def test_parser_rejects_bid_above_coins():
    """After spending all coins, only bid 0 is legal."""
    p = create_parser()
    with pytest.raises(ValueError):
        p.parse("5", [0])


def test_parser_rejects_non_numeric():
    p = create_parser()
    with pytest.raises(ValueError):
        p.parse("all-in", [0, 1, 2, 3, 4, 5])
