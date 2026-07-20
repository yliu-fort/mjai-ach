"""Kuhn Poker renderer: 3-card deck (J/Q/K = 0/1/2), per-player private card.

Information-state tensor layout (Kuhn, 11-dim per OpenSpiel):
  [0:3]   private card one-hot (player's own card)
  [1:4]   ... actually OpenSpiel orders it as: card one-hot, then sequence
This renderer reads the public action history from the state's history instead,
which is simpler and avoids depending on the exact tensor bit-layout.
"""

from __future__ import annotations

import pyspiel

from mjai.cli.interfaces import GameRenderer

_CARD = {0: "J", 1: "Q", 2: "K"}


def create() -> GameRenderer:
    return _KuhnRenderer()


class _KuhnRenderer:
    def render(self, state: pyspiel.State, observer_player: int | None) -> str:
        if state.is_terminal():
            return self.render_terminal(state)
        p = observer_player if observer_player is not None else state.current_player()
        # Private card from the info-state tensor (first 3 slots).
        info = state.information_state_tensor(p)
        card_idx = next((i for i in range(3) if info[i] > 0.5), None)
        card = _CARD.get(card_idx if card_idx is not None else -1, "?")
        # Reconstruct public action history (bet/call sequence).
        history = self._public_history(state)
        lines = [
            f"Kuhn Poker — you are player {p}, your card: {card}",
            f"Pot: {self._pot(state)}    Actions so far: {' '.join(history) or '(none)'}",
        ]
        legal = state.legal_actions(p)
        names = {0: "check/call", 1: "bet/raise (fold)"}
        lines.append("Legal: " + ", ".join(f"{a}={names.get(a, str(a))}" for a in legal))
        return "\n".join(lines)

    def render_terminal(self, state: pyspiel.State) -> str:
        ret = state.returns()
        winner = 0 if ret[0] > 0 else 1
        if abs(ret[0]) < 1e-9:
            return "Hand over: tie."
        return (
            f"Hand over: player {winner} wins {abs(ret[0]):+.0f}. Returns: {[int(r) for r in ret]}"
        )

    def _public_history(self, state: pyspiel.State) -> list[str]:
        out = []
        for a in state.history():
            out.append("bet" if a == 1 else "check")
        return out

    def _pot(self, state: pyspiel.State) -> int:
        # Each player antes 1; a bet adds 1. Approximate from history length.
        bets = sum(1 for a in state.history() if a == 1)
        return 2 + bets
