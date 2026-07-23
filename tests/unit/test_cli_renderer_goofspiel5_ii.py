"""Unit tests for the Goofspiel-5 renderer + parser (§5).

Regression coverage for the round-count bug (bug family A), verified
empirically against pyspiel 2.0.1:

  - The renderer computed rounds-played as ``len(state.history()) // 3``, as if
    each round pushed a (point-card, P0 bid, P1 bid) triple into history. In
    fact each round appends exactly two entries (P0 bid, P1 bid); the point
    card is tracked internally and does NOT appear in ``state.history()``. So
    after 2 rounds (history length 4) the old renderer printed "1 rounds
    played". The correct count is ``len(history) // 2``.

Also guards: legal bids come from the per-player hand (cards are consumed), and
blind entry never reveals the opponent's simultaneous bid.
"""

from __future__ import annotations

import pyspiel
import pytest

from mjai.cli.input_parsers.goofspiel5_ii import create as create_parser
from mjai.cli.renderers.goofspiel5_ii import create as create_renderer


def _game() -> pyspiel.Game:
    return pyspiel.load_game("goofspiel(imp_info=True,num_cards=5,points_order=descending)")


def _rounds(n: int) -> pyspiel.State:
    """A state with ``n`` rounds played using distinct bids (cards can't repeat)."""
    pairs = [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
    s = _game().new_initial_state()
    for r in range(n):
        b0, b1 = pairs[r % len(pairs)]
        s.apply_actions([b0, b1])
    return s


def _rounds_line(rendered: str) -> str:
    for line in rendered.splitlines():
        if "Rounds played:" in line:
            return line
    raise AssertionError(f"no rounds line in:\n{rendered}")


# --------------------------------------------------------------------------- #
# Renderer: round count
# --------------------------------------------------------------------------- #


def test_zero_rounds_at_start():
    s = _game().new_initial_state()
    out = create_renderer().render(s, observer_player=0)
    assert "Rounds played: 0" in _rounds_line(out)


def test_round_count_after_two_rounds():
    """Regression: //3 gave 1 after 2 rounds; //2 gives 2."""
    s = _rounds(2)
    assert len(s.history()) == 4  # sanity: 2 entries per round
    out = create_renderer().render(s, observer_player=0)
    assert "Rounds played: 2" in _rounds_line(out)


def test_round_count_after_three_rounds():
    s = _rounds(3)
    out = create_renderer().render(s, observer_player=0)
    assert "Rounds played: 3" in _rounds_line(out)


# --------------------------------------------------------------------------- #
# Renderer: blind entry / consumed cards
# --------------------------------------------------------------------------- #


def test_played_card_removed_from_legal():
    """Playing a card consumes it; legal_actions shrinks the next round."""
    s = _game().new_initial_state()
    s.apply_actions([0, 4])  # P0 played card 0
    out = create_renderer().render(s, observer_player=0)
    legal_line = next(ln for ln in out.splitlines() if "legal bids" in ln)
    # Card 0 is gone from P0's hand; remaining are 1..4.
    assert "0" not in legal_line.split("values):")[1]


def test_blind_does_not_reveal_opponent_bid():
    """AGENTS.md §4: simultaneous-move human input is blind. The observer's
    render must not reveal the opponent's just-played card value.

    P1 played card 4 in round 0; the render for P0 may say "blind" but must not
    announce the specific value 4 as the opponent's past bid.
    """
    s = _rounds(1)  # P1 played card 4 in round 0
    out = create_renderer().render(s, observer_player=0)
    lower = out.lower()
    # Generic "opponent" wording (e.g. "without seeing opponent's choice") is
    # fine; a statement naming the opponent's played value is not.
    assert "opponent played" not in lower
    assert "opponent bid 4" not in lower
    assert "opponent's card was 4" not in lower


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def test_parser_accepts_card_value():
    p = create_parser()
    assert p.parse("3", [0, 1, 2, 3, 4]) == 3


def test_parser_rejects_card_not_in_hand():
    """Card already played must be rejected."""
    p = create_parser()
    # After playing card 0, legal = [1,2,3,4]; '0' is illegal.
    with pytest.raises(ValueError):
        p.parse("0", [1, 2, 3, 4])


def test_parser_rejects_non_numeric():
    p = create_parser()
    with pytest.raises(ValueError):
        p.parse("rock", [0, 1, 2, 3, 4])
