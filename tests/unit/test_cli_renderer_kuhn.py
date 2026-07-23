"""Unit tests for the Kuhn Poker renderer + parser (§5).

Regression coverage for four bugs that were all present together:

  1. The renderer read ``info[0:3]`` for the private card, but slot 0 is the
     player-id bit (hot for P0 regardless of the card), so it always printed
     "your card: J". The card is at ``info[2:5]``.
  2. ``_public_history`` walked all of ``state.history()``, which includes the
     two opening chance deals, so a freshly dealt hand (history ``[1, 2]``)
     rendered as the bogus sequence "bet check".
  3. The legal-action label called action 1 "bet/raise (fold)" — action 1 is
     Bet; Kuhn has no fold action.
  4. The parser mapped ``call`` -> action 0, which is the *fold*. Typing
     ``call`` while facing a bet folded and lost the pot (returns ``[+1,-1]``).

All encodings below were verified empirically against pyspiel 2.0.1.
"""

from __future__ import annotations

import pyspiel
import pytest

from mjai.cli.input_parsers.kuhn import create as create_parser
from mjai.cli.renderers.kuhn import create as create_renderer


def _game() -> pyspiel.Game:
    return pyspiel.load_game("kuhn_poker")


def _dealt(p0_card: int, p1_card: int) -> pyspiel.State:
    """Initial state with both hole cards dealt (chance resolved), no bets."""
    s = _game().new_initial_state()
    s.apply_action(p0_card)  # first chance outcome = P0's card id
    s.apply_action(p1_card)  # second chance outcome = P1's card id
    return s


def _card_line(rendered: str) -> str:
    for line in rendered.splitlines():
        if line.startswith("Kuhn Poker —"):
            return line
    raise AssertionError(f"no card line in:\n{rendered}")


def _actions_line(rendered: str) -> str:
    for line in rendered.splitlines():
        if "Actions so far:" in line:
            return line
    raise AssertionError(f"no actions line in:\n{rendered}")


# --------------------------------------------------------------------------- #
# Renderer: private card (privacy)
# --------------------------------------------------------------------------- #


def test_private_card_displayed_correctly():
    """Regression: old code read slot 0 (player-id) and always printed J."""
    # P0 holds Q (card id 1) -> must show Q, not J.
    state = _dealt(1, 2)
    out = create_renderer().render(state, observer_player=0)
    assert "your card: Q" in _card_line(out)
    # P0 holds K (card id 2) -> K.
    state = _dealt(2, 0)
    out = create_renderer().render(state, observer_player=0)
    assert "your card: K" in _card_line(out)


def test_opponent_card_not_leaked():
    """AGENTS.md §4 / INV-1: never reveal the opponent's private card."""
    state = _dealt(1, 2)  # P0=Q, P1=K
    out_p0 = create_renderer().render(state, observer_player=0)
    out_p1 = create_renderer().render(state, observer_player=1)
    assert "your card: Q" in out_p0 and "your card: K" not in out_p0
    assert "your card: K" in out_p1 and "your card: Q" not in out_p1


# --------------------------------------------------------------------------- #
# Renderer: action history (chance must not appear)
# --------------------------------------------------------------------------- #


def test_empty_history_after_deal():
    """Regression: the two chance deals must NOT render as 'bet check'."""
    state = _dealt(1, 2)
    out = create_renderer().render(state, observer_player=0)
    assert "Actions so far: (none)" in _actions_line(out)


def test_bet_renders_as_bet_not_chance():
    state = _dealt(0, 0)
    state.apply_action(1)  # P0 bets
    out = _actions_line(create_renderer().render(state, observer_player=1))
    assert "Actions so far: bet" in out


def test_legal_label_does_not_call_bet_a_fold():
    """Regression: action 1 was labelled 'bet/raise (fold)'; it is just Bet.

    The old label embedded "(fold)" on action 1, but Kuhn's fold is action 0
    (a Pass when facing a bet). Assert the new label is a bet/call, not a fold.
    """
    state = _dealt(0, 0)
    out = create_renderer().render(state, observer_player=0)
    legal_line = next(ln for ln in out.splitlines() if ln.startswith("Legal:"))
    assert "1=bet (call)" in legal_line
    # And the "(fold)" suffix that used to sit on action 1 must be gone.
    assert "bet/raise (fold)" not in legal_line


# --------------------------------------------------------------------------- #
# Parser: call must not fold
# --------------------------------------------------------------------------- #


def test_parser_call_is_not_fold_regression():
    """Regression: 'call' -> action 0 folded; it must map to action 1 (the call).

    OpenSpiel: facing a bet, action 0 = Pass (fold, returns [+1,-1]), action 1
    = Bet (call, returns [+2,-2]). 'call' must yield action 1.
    """
    state = _dealt(2, 0)
    state.apply_action(1)  # P0 bets
    legal = list(state.legal_actions(1))
    p = create_parser()
    assert p.parse("call", legal) == 1
    # And it produces the call outcome, not the fold outcome.
    assert state.child(1).returns() == [2.0, -2.0]


def test_parser_fold_maps_to_pass():
    """Facing a bet, 'fold' must map to action 0 (Pass = fold)."""
    state = _dealt(2, 0)
    state.apply_action(1)  # P0 bets
    legal = list(state.legal_actions(1))
    assert create_parser().parse("fold", legal) == 0
    assert state.child(0).returns() == [1.0, -1.0]


def test_parser_check_and_bet_on_opening():
    """Opening move: 'check'/'pass' -> 0, 'bet' -> 1."""
    state = _dealt(0, 0)
    legal = list(state.legal_actions(0))
    p = create_parser()
    for word in ("check", "pass", "p"):
        assert p.parse(word, legal) == 0
    for word in ("bet", "raise", "b"):
        assert p.parse(word, legal) == 1


def test_parser_accepts_numeric_action():
    p = create_parser()
    assert p.parse("0", [0, 1]) == 0
    assert p.parse("1", [0, 1]) == 1


def test_parser_rejects_illegal_and_unparseable():
    p = create_parser()
    with pytest.raises(ValueError):
        p.parse("99", [0, 1])
    with pytest.raises(ValueError):
        p.parse("banana", [0, 1])
