"""Goofspiel-5 (imperfect-info) renderer: simultaneous bidding, blind entry.

5-card deck (point values 5..1, descending). Each turn both players secretly
play a card to contest the current point card; winner of the point card takes
it equal to its value. imp_info=True means the opponent's played card is hidden.
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
        # Goofspiel-5 info-state includes the point cards revealed so far + the
        # player's own remaining hand. We summarize via history.
        history = self._history(state)
        legal = state.legal_actions(p)
        lines = [
            f"Goofspiel-5 — you are player {p}.",
            f"Revealed point cards so far: {history}",
            f"Your legal bids (card values): {legal}",
            "  (blind: pick one without seeing opponent's choice)",
        ]
        return "\n".join(lines)

    def render_terminal(self, state: pyspiel.State) -> str:
        ret = state.returns()
        if abs(ret[0]) < 1e-9:
            return "Game over: tie."
        winner = 0 if ret[0] > 0 else 1
        return f"Game over: player {winner} wins by {abs(ret[0]):+.0f} points. Returns: {[int(r) for r in ret]}"

    def _history(self, state: pyspiel.State) -> str:
        # History is the sequence of (point card revealed, p0 bid, p1 bid)
        # triples flattened; we just show length / count for the simple renderer.
        return f"{len(state.history()) // 3} rounds played"
