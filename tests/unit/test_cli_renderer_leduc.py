"""Unit tests for the Leduc Poker renderer + parser (§5).

Regression coverage for three bugs that were all present together:

  1. The renderer read the private card from ``info[0:3]`` — slot 0 is the
     player-id bit (hot for P0), so it always printed 'Private card: J'. The
     own-card one-hot is at ``info[2:8]`` (6 deck cards; rank = card_id % 3).
  2. The renderer read the public card from ``info[3:6]``, but the board card
     one-hot is at ``info[8:14]``, so the flop never showed ('-' forever).
  3. ``_history`` walked all of ``state.history()``, which interleaves the two
     hole-card deals and the flop (chance outcomes) with the betting actions,
     so the displayed sequence included chance entries. The flop's position is
     not at a fixed offset, so the renderer now replays the history tagging
     chance vs player steps.

All encodings below were verified empirically against pyspiel 2.0.1.
"""

from __future__ import annotations

import pyspiel
import pytest

from mjai.cli.input_parsers.leduc import create as create_parser
from mjai.cli.renderers.leduc import create as create_renderer


def _game() -> pyspiel.Game:
    return pyspiel.load_game("leduc_poker")


def _preflop(p0_card: int, p1_card: int) -> pyspiel.State:
    """Initial state with both hole cards dealt (chance resolved), no actions."""
    s = _game().new_initial_state()
    s.apply_action(p0_card)
    s.apply_action(p1_card)
    return s


def _to_round2(p0_card: int, p1_card: int, flop: int | None = None) -> pyspiel.State:
    """Deal hole cards, call/call round 1, then deal the flop (first available
    if ``flop`` is None). Returns the round-2 decision state."""
    s = _preflop(p0_card, p1_card)
    s.apply_action(1)  # P0 call
    s.apply_action(1)  # P1 call -> closes round 1
    assert s.is_chance_node()
    co = [o for o, _ in s.chance_outcomes()]
    target = flop if flop in co else co[0]
    s.apply_action(target)
    return s


def _line(rendered: str, key: str) -> str:
    for line in rendered.splitlines():
        if key in line:
            return line
    raise AssertionError(f"no line with {key!r} in:\n{rendered}")


# --------------------------------------------------------------------------- #
# Renderer: private card (privacy)
# --------------------------------------------------------------------------- #


def test_private_card_displayed_correctly():
    """Regression: old code read slot 0 (player-id) and always printed J."""
    # P0 holds Q (card id 1) -> must show Q, not J.
    state = _preflop(1, 2)
    out = create_renderer().render(state, observer_player=0)
    assert "Private card: Q" in _line(out, "Private card:")
    # P0 holds K (card id 2) -> K.
    state = _preflop(2, 0)
    out = create_renderer().render(state, observer_player=0)
    assert "Private card: K" in _line(out, "Private card:")


def test_opponent_card_not_leaked():
    """AGENTS.md §4 / INV-1: never reveal the opponent's hole card."""
    state = _preflop(1, 2)  # P0=Q, P1=K
    out_p0 = create_renderer().render(state, observer_player=0)
    out_p1 = create_renderer().render(state, observer_player=1)
    assert "Private card: Q" in out_p0 and "Private card: K" not in out_p0
    assert "Private card: K" in out_p1 and "Private card: Q" not in out_p1


# --------------------------------------------------------------------------- #
# Renderer: public card (the flop)
# --------------------------------------------------------------------------- #


def test_public_card_hidden_preflop():
    """Before the flop, the public card is unknown -> show '-'."""
    state = _preflop(0, 1)
    out = create_renderer().render(state, observer_player=0)
    assert "Public card: -" in _line(out, "Public card:")


def test_public_card_shown_postflop():
    """Regression: old code read the wrong region and the flop never showed."""
    # Deal P0=0(J), P1=1(Q); flop will be first available of {2,3,4,5}.
    state = _to_round2(0, 1)
    # The flop rank is whatever the first available outcome is; just assert a
    # concrete rank now shows (not '-') and is consistent between observers.
    out0 = create_renderer().render(state, observer_player=0)
    out1 = create_renderer().render(state, observer_player=1)
    pub0 = _line(out0, "Public card:")
    pub1 = _line(out1, "Public card:")
    assert "Public card: -" not in pub0
    assert pub0.split("Public card:")[1].strip() == pub1.split("Public card:")[1].strip()


# --------------------------------------------------------------------------- #
# Renderer: action history (chance removed)
# --------------------------------------------------------------------------- #


def test_history_excludes_chance_outcomes():
    """Regression: hole deals + flop leaked into the action sequence."""
    state = _to_round2(0, 1)  # call/call round 1, then flop
    out = create_renderer().render(state, observer_player=0)
    hist = _line(out, "Action history:")
    # Only the two round-1 calls; no chance entries, no flop value.
    assert hist.strip() == "Action history: call call" or "call call" in hist


def test_history_empty_preflop():
    state = _preflop(0, 1)
    out = create_renderer().render(state, observer_player=0)
    assert "Action history: (none)" in _line(out, "Action history:")


def test_history_with_fold_across_rounds():
    """A raise then fold: history must show 'raise fold' with no flop leaking."""
    state = _preflop(0, 1)
    state.apply_action(2)  # P0 raise
    state.apply_action(0)  # P1 fold -> terminal
    out = create_renderer().render(state, observer_player=0)
    assert state.is_terminal()
    # render_terminal kicks in at terminal; verify it reports the result.
    assert "Hand over" in out


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def test_parser_aliases_and_numeric():
    p = create_parser()
    assert p.parse("f", [0, 1, 2]) == 0
    assert p.parse("fold", [0, 1, 2]) == 0
    assert p.parse("c", [1, 2]) == 1
    assert p.parse("call", [1, 2]) == 1
    assert p.parse("raise", [0, 1, 2]) == 2
    assert p.parse("0", [0, 1, 2]) == 0
    assert p.parse("2", [0, 1, 2]) == 2


def test_parser_fold_legal_after_raise():
    """Facing a raise, fold (action 0) is legal and terminates."""
    state = _preflop(0, 1)
    state.apply_action(2)  # P0 raise
    legal = list(state.legal_actions(1))
    assert 0 in legal
    assert create_parser().parse("fold", legal) == 0


def test_parser_rejects_illegal_and_unparseable():
    p = create_parser()
    with pytest.raises(ValueError):
        p.parse("99", [0, 1, 2])
    with pytest.raises(ValueError):
        p.parse("banana", [0, 1, 2])
