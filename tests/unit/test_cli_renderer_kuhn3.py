"""Unit tests for the 3-player Kuhn renderer + parser (AGENTS.md §5, D13).

The 2p renderer's own docstring records that reading the private card from a
hard-coded slot range was a real bug. Going to three players moves that range
again — the card block starts at slot ``num_players`` and is ``num_players + 1``
wide, so 3p Kuhn reads slots 3..7 where 2p reads 2..5. A renderer that kept the
2p constants would silently print the wrong card at every seat, which is a
privacy bug as much as a display bug. These tests pin the layout at 3p.

All encodings below were verified empirically against pyspiel 2.0.1 on
2026-07-26: info-state tensor length 17 = 3 player-id bits + 4 card slots + 10
history slots; three opening chance deals; card ids 0..3 = J/Q/K/A.
"""

from __future__ import annotations

import pyspiel

from mjai.cli.input_parsers.kuhn3 import create as create_parser
from mjai.cli.renderers.kuhn3 import create as create_renderer


def _game() -> pyspiel.Game:
    return pyspiel.load_game("kuhn_poker(players=3)")


def _dealt(c0: int, c1: int, c2: int) -> pyspiel.State:
    """Initial state with all three hole cards dealt, no bets."""
    s = _game().new_initial_state()
    for card in (c0, c1, c2):
        s.apply_action(card)
    return s


def _card_line(rendered: str) -> str:
    for line in rendered.splitlines():
        if line.startswith("Kuhn Poker (3p) —"):
            return line
    raise AssertionError(f"no card line in:\n{rendered}")


def test_four_card_deck_labels_every_rank():
    """Card ids 0..3 render as J/Q/K/A — a 3-card table would drop the ace."""
    labels = ["J", "Q", "K", "A"]
    for card_id, want in enumerate(labels):
        others = [c for c in range(4) if c != card_id]
        state = _dealt(card_id, others[0], others[1])
        out = create_renderer().render(state, observer_player=0)
        assert f"your card: {want}" in _card_line(out)


def test_private_card_read_from_the_three_player_slot_range():
    """Regression: the card block starts at slot 3 here, not slot 2.

    P0 holds Q (id 1). With the 2p constants the renderer would read slot 2,
    which at 3p is P2's player-id bit, and print the wrong rank.
    """
    state = _dealt(1, 3, 0)
    out = create_renderer().render(state, observer_player=0)
    assert "your card: Q" in _card_line(out)
    assert "your card: ?" not in out


def test_no_seat_leaks_another_seats_card():
    """AGENTS.md §4 / INV-1, now across three seats rather than two."""
    state = _dealt(1, 3, 0)  # P0=Q, P1=A, P2=J
    renderer = create_renderer()
    views = {p: renderer.render(state, observer_player=p) for p in range(3)}
    assert "your card: Q" in views[0]
    assert "your card: A" in views[1]
    assert "your card: J" in views[2]
    # Each view names exactly one card, its own.
    for p, own in ((0, "Q"), (1, "A"), (2, "J")):
        for other in set("QAJ") - {own}:
            assert f"your card: {other}" not in views[p]


def test_public_view_hides_every_card():
    state = _dealt(1, 3, 0)
    out = create_renderer().render_public(state)
    for rank in ("J", "Q", "K", "A"):
        assert f"card: {rank}" not in out


def test_history_skips_all_three_chance_deals():
    """Regression: three deals, not two, precede the first betting action."""
    state = _dealt(1, 3, 0)
    out = create_renderer().render(state, observer_player=0)
    assert "Actions so far: (none)" in out
    state.apply_action(1)  # P0 bets
    out = create_renderer().render(state, observer_player=1)
    assert "Actions so far: bet" in out


def test_pot_counts_three_antes():
    state = _dealt(1, 3, 0)
    assert "Pot: 3" in create_renderer().render(state, observer_player=0)
    state.apply_action(1)  # P0 bets -> one more chip in
    assert "Pot: 4" in create_renderer().render(state, observer_player=1)


def test_terminal_reports_every_seat_not_a_winner_loser_pair():
    """Three-way results have no "the other player lost" shortcut (D13)."""
    state = _dealt(1, 3, 0)
    for action in (1, 1, 0):  # P0 bets, P1 calls, P2 folds
        state.apply_action(action)
    assert state.is_terminal()
    assert state.returns() == [-2.0, 3.0, -1.0]
    out = create_renderer().render_terminal(state)
    assert "player 1 wins" in out
    for seat in ("p0", "p1", "p2"):
        assert seat in out


def test_parser_matches_the_two_player_semantics():
    """Same action ids, so the same intent mapping — notably call -> 1."""
    state = _dealt(1, 3, 0)
    state.apply_action(1)  # P0 bets
    legal = list(state.legal_actions(1))
    parser = create_parser()
    assert parser.parse("call", legal) == 1
    assert parser.parse("fold", legal) == 0
