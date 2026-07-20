"""Leduc Poker renderer: 2-round Hold'em with a 6-card deck (AGENTS.md §1 D8)."""

from __future__ import annotations

import pyspiel

from mjai.cli.interfaces import GameRenderer

_RANKS = {0: "J", 1: "Q", 2: "K"}


def create() -> GameRenderer:
    return _LeducRenderer()


class _LeducRenderer:
    def render(self, state: pyspiel.State, observer_player: int | None) -> str:
        if state.is_terminal():
            return self.render_terminal(state)
        p = observer_player if observer_player is not None else state.current_player()
        info = state.information_state_tensor(p)
        # Leduc info-state: private card (one-hot 3) + public card (one-hot 3) +
        # round + per-round betting history. Read private + public from the
        # leading slots.
        private = next((i for i in range(3) if info[i] > 0.5), None)
        public = next((i for i in range(3, 6) if info[i] > 0.5), None)
        lines = [
            f"Leduc Poker — you are player {p}.",
            f"Private card: {_RANKS.get(private if private is not None else -1, '?')}    "
            f"Public card: {_RANKS.get(public if public is not None else -1, '-')}",
            f"Action history: {self._history(state)}",
        ]
        legal = state.legal_actions(p)
        names = {0: "fold", 1: "check/call", 2: "bet/raise"}
        lines.append("Legal: " + ", ".join(f"{a}={names.get(a, str(a))}" for a in legal))
        return "\n".join(lines)

    def render_terminal(self, state: pyspiel.State) -> str:
        ret = state.returns()
        if abs(ret[0]) < 1e-9:
            return "Hand over: tie."
        winner = 0 if ret[0] > 0 else 1
        return f"Hand over: player {winner} wins. Returns: {[int(r) for r in ret]}"

    def _history(self, state: pyspiel.State) -> str:
        names = {0: "F", 1: "c", 2: "b"}
        return " ".join(names.get(a, str(a)) for a in state.history()) or "(none)"
