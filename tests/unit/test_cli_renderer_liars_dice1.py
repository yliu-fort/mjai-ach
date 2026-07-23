"""Unit tests for the Liar's-Dice renderer + parser (F3, §5).

Regression coverage for three bugs that were all present together:

  1. The renderer treated the two opening die rolls (chance outcomes in
     ``state.history()``) as bids, so a fresh table with no bids yet showed a
     bogus "bid 1x5 bid 1x4" history (those were the roll outcome ids).
  2. Bid decode was off by one — ``(a-1)//6+1`` instead of ``a//6+1`` — which
     made faces wrap 6 -> 1 and bids look non-monotonic.
  3. Challenge was mapped to action id 0, but 0 is a valid opening bid; the
     real challenge id is ``num_distinct_actions() - 1`` (= 12 here).

Plus the own-die read: the old code read the player-id one-hot slot and so
always printed "your die: 1".

All encodings below were verified empirically against pyspiel 2.0.1.
"""

from __future__ import annotations

import pyspiel
import pytest

from mjai.cli.input_parsers.liars_dice1 import create as create_parser
from mjai.cli.renderers.liars_dice1 import create as create_renderer


def _game() -> pyspiel.Game:
    return pyspiel.load_game("liars_dice(numdice=1,dice_sides=6)")


def _rolled_state(d0: int, d1: int) -> pyspiel.State:
    """Initial state with both dice rolled (chance resolved). No bids yet."""
    s = _game().new_initial_state()
    s.apply_action(d0)
    s.apply_action(d1)
    return s


def _bidding_history_line(rendered: str) -> str:
    for line in rendered.splitlines():
        if line.startswith("Bidding history:"):
            return line
    raise AssertionError(f"no Bidding history line in:\n{rendered}")


# --------------------------------------------------------------------------- #
# Renderer: bidding history
# --------------------------------------------------------------------------- #


def test_no_bids_history_is_empty_after_rolls():
    """Regression: rolls must NOT render as bids. Fresh table shows no bids."""
    state = _rolled_state(5, 4)  # the exact roll ids from the user's transcript
    out = create_renderer().render(state, observer_player=0)
    assert _bidding_history_line(out) == "Bidding history: (无)"


def test_bid_decode_quantity_and_face_not_off_by_one():
    """Regression: action 4 -> (1,5), action 6 -> (2,1); old code gave (1,4)/(1,6)."""
    state = _rolled_state(0, 0)
    state.apply_action(4)  # bid: one 5
    assert _bidding_history_line(create_renderer().render(state, observer_player=1)) == (
        "Bidding history: 叫牌 1x5"
    )

    state2 = _rolled_state(0, 0)
    state2.apply_action(6)  # bid: two 1
    assert _bidding_history_line(create_renderer().render(state2, observer_player=1)) == (
        "Bidding history: 叫牌 2x1"
    )


def test_bids_remain_monotonic_in_display():
    """A legal ascending bid sequence must render as ascending, not wrapping."""
    state = _rolled_state(0, 0)
    state.apply_action(4)  # (1,5)
    state.apply_action(5)  # (1,6)
    state.apply_action(6)  # (2,1)
    out = _bidding_history_line(create_renderer().render(state, observer_player=0))
    assert out == "Bidding history: 叫牌 1x5 叫牌 1x6 叫牌 2x1"


def test_action_zero_is_decoded_as_bid_not_challenge():
    """Regression: action 0 is a valid opening bid (one 1), not a challenge."""
    # P0 opens with action 0; the history line must show a bid, not "挑战".
    state = _rolled_state(0, 0)
    state.apply_action(0)
    out = _bidding_history_line(create_renderer().render(state, observer_player=1))
    assert out == "Bidding history: 叫牌 1x1"
    assert "挑战" not in out


def test_challenge_id_is_last_action_not_zero():
    """The challenge action id is num_distinct_actions()-1 (=12), and only it ends
    the round. Action 0 does NOT terminate (it's a bid). Guards the old bug that
    mapped challenge -> 0."""
    game = _game()
    challenge_id = game.num_distinct_actions() - 1
    assert challenge_id == 12
    pre = _rolled_state(0, 0)
    pre.apply_action(4)  # a bid exists, so challenge is now legal for P1
    assert challenge_id in pre.legal_actions(1)
    # Only the challenge id terminates; a higher bid does not.
    assert pre.child(challenge_id).is_terminal()
    assert not pre.child(challenge_id - 1).is_terminal()


def test_terminal_renders_winner_and_returns():
    state = _rolled_state(0, 0)
    state.apply_action(4)
    state.apply_action(12)  # P1 challenges P0's "one 5"; P0's die is face 1 -> P1 wins
    assert state.is_terminal()
    out = create_renderer().render(state, observer_player=None)
    assert "Round over" in out and "player 1 wins" in out


# --------------------------------------------------------------------------- #
# Renderer: own die + privacy
# --------------------------------------------------------------------------- #


def test_own_die_displayed_correctly():
    """Regression: old code read slot 0 and always printed die: 1."""
    # Roll outcome 3 -> own die face 4 for player 0.
    state = _rolled_state(3, 0)
    out = create_renderer().render(state, observer_player=0)
    assert "your die: 4" in out


def test_opponent_die_not_leaked():
    """AGENTS.md §4: never reveal the opponent's private die to the observer."""
    state = _rolled_state(3, 5)  # P0 die face 4, P1 die face 6
    out_p0 = create_renderer().render(state, observer_player=0)
    out_p1 = create_renderer().render(state, observer_player=1)
    # Each sees only their own die.
    assert "your die: 4" in out_p0 and "your die: 6" not in out_p0
    assert "your die: 6" in out_p1 and "your die: 4" not in out_p1


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def test_parser_qf_not_off_by_one():
    p = create_parser()
    # "1 5" -> action 4 (one 5); old code returned 5 (one 6).
    assert p.parse("1 5", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) == 4
    # "2 1" -> action 6 (two 1).
    assert p.parse("2 1", [6, 7, 8, 9, 10, 11, 12]) == 6


def test_parser_challenge_maps_to_correct_id():
    p = create_parser()
    legal_with_challenge = [5, 6, 7, 8, 9, 10, 11, 12]
    for word in ("challenge", "c", "call", "liar"):
        assert p.parse(word, legal_with_challenge) == 12


def test_parser_action_zero_is_opening_bid_not_challenge():
    """Regression: '0' must remain a legal opening bid, not silently a challenge."""
    p = create_parser()
    assert p.parse("0", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) == 0


def test_parser_rejects_challenge_at_opening():
    """Challenge (id 12) is illegal before any bid; parser must reject it."""
    p = create_parser()
    opening_legal = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # no 12
    with pytest.raises(ValueError):
        p.parse("challenge", opening_legal)


def test_parser_rejects_illegal_action():
    p = create_parser()
    with pytest.raises(ValueError):
        p.parse("99", [0, 1, 2, 3, 4, 5])
