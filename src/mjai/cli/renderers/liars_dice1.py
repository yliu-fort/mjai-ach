"""Liar's Dice renderer: 1 die each, 6 sides. Player sees only their own die.

Action encoding (OpenSpiel): 0 = challenge (liar), 1..6 = bid one 1..6, ...,
actually Liar's-Dice-1's 13 actions encode (quantity-1)*6 + face-1 for the
non-challenge bids; action 0 is "call bluff / challenge". The parser accepts
the raw action id for simplicity, plus "q f" two-token form.
"""

from __future__ import annotations

import pyspiel

from mjai.cli.interfaces import GameRenderer


def create() -> GameRenderer:
    return _LiarsDiceRenderer()


class _LiarsDiceRenderer:
    def render(self, state: pyspiel.State, observer_player: int | None) -> str:
        if state.is_terminal():
            return self.render_terminal(state)
        p = observer_player if observer_player is not None else state.current_player()
        info = state.information_state_tensor(p)
        # First 6 slots = player's own die one-hot.
        die = next((i + 1 for i in range(6) if info[i] > 0.5), None)
        # Bidding history (public).
        history = self._history(state)
        legal = state.legal_actions(p)
        lines = [
            f"Liar's Dice — you are player {p}, your die: {die}",
            f"Bidding history: {history}",
            f"Legal actions: {legal}",
            "  (0 = challenge; k>0 = bid quantity=(k-1)//6 + 1 of face (k-1)%6 + 1)",
        ]
        return "\n".join(lines)

    def render_terminal(self, state: pyspiel.State) -> str:
        ret = state.returns()
        winner = 0 if ret[0] > 0 else 1
        return f"Round over: player {winner} wins. Returns: {[int(r) for r in ret]}"

    def _history(self, state: pyspiel.State) -> str:
        out = []
        for a in state.history():
            if a == 0:
                out.append("CHALLENGE")
            else:
                q = (a - 1) // 6 + 1
                f = (a - 1) % 6 + 1
                out.append(f"bid {q}x{f}")
        return " ".join(out) or "(none)"
