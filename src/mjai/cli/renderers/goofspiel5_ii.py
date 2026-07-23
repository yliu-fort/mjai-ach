"""Goofspiel-5 (imperfect-info) renderer: simultaneous bidding, blind entry.

5-card deck (point values 5..1, descending). Each turn both players secretly
play a card to contest the current point card; the winner of the point card
takes it. ``imp_info=True`` means the opponent's played card is hidden.

History layout (verified empirically against pyspiel 2.0.1): each round appends
exactly two entries to ``state.history()`` — P0's bid then P1's bid. The point
card dealt each round is tracked internally and does **not** appear in
``state.history()``. The old renderer computed rounds-played as
``len(history) // 3`` (as if each round pushed a point-card/bid/bid triple),
which undercounted: after 2 rounds (history length 4) it printed "1 rounds
played". The correct count is ``len(history) // 2``.

Card (action) ids are the bid values 1..5; playing a card removes it from the
hand, so ``legal_actions(p)`` shrinks each round. The renderer must call
``legal_actions`` per player (action id 0..4), never ``legal_actions(-2)`` which
returns the flat joint action space (misleading).
"""

from __future__ import annotations

import pyspiel

from mjai.cli.interfaces import GameRenderer


def create() -> GameRenderer:
    return _GoofspielRenderer()


class _GoofspielRenderer:
    def render(self, state: pyspiel.State, observer_player: int | None) -> str:
        if state.is_terminal():
            return self.render_terminal(state)
        p = observer_player if observer_player is not None else state.current_player()
        history = self._history(state)
        # Per-player legal actions (the observer's remaining hand). NB: never
        # legal_actions(-2) — that returns the flat joint space and misleads.
        legal = state.legal_actions(p) if p >= 0 else list(state.legal_actions(0))
        lines = [
            f"Goofspiel-5 — you are player {p}.",
            f"Rounds played: {history}",
            f"Your legal bids (card values): {legal}",
            "  (blind: pick one without seeing opponent's choice)",
        ]
        return "\n".join(lines)

    def render_public(self, state: pyspiel.State) -> str:
        """Public-only view: rounds played, no player's hand or pending bid.

        Used when a human is spectating a robot's turn (INV-1). With
        ``imp_info=True`` the opponent's played card is hidden, and each
        player's remaining hand is private; only the round count is public.
        """
        if state.is_terminal():
            return self.render_terminal(state)
        return f"Goofspiel-5 — public view.\nRounds played: {self._history(state)}"

    def render_terminal(self, state: pyspiel.State) -> str:
        ret = state.returns()
        if abs(ret[0]) < 1e-9:
            return "Game over: tie."
        winner = 0 if ret[0] > 0 else 1
        return (
            f"Game over: player {winner} wins by {abs(ret[0]):+.0f} points. "
            f"Returns: {[int(r) for r in ret]}"
        )

    def _history(self, state: pyspiel.State) -> int:
        """Rounds played so far. Each round appends 2 entries (P0 bid, P1 bid);
        the point card is not in history."""
        return len(state.history()) // 2
