"""Oshi-Zumo renderer: simultaneous secret bidding, wrestler on a 3-cell ring.

Config: coins=5, size=3, horizon=20. Action = bid (0..coins); higher bid pushes
the wrestler one step toward the opponent. Spent coins are gone. Simultaneous
move => blind entry (AGENTS.md §4).
"""

from __future__ import annotations

import pyspiel

from mjai.cli.interfaces import GameRenderer


def create() -> GameRenderer:
    return _OshiZumoRenderer()


class _OshiZumoRenderer:
    def render(self, state: pyspiel.State, observer_player: int | None) -> str:
        if state.is_terminal():
            return self.render_terminal(state)
        p = observer_player if observer_player is not None else state.current_player()
        # Observation tensor encodes each player's coins + wrestler position.
        # We extract coins and position heuristically from the public state.
        info = state.observation_tensor(p)
        # OpenSpiel oshi_zumo observation layout: roughly [p0_coins_one_hot,
        # p1_coins_one_hot, board_one_hot]. Parse defensively.
        coins_self = self._count_leading_ones(info, 0, 6)
        coins_opp = self._count_leading_ones(info, 6, 12)
        pos = self._wrestler_pos(info, 12, 15)
        legal = state.legal_actions(p)
        lines = [
            f"Oshi-Zumo — you are player {p}.",
            f"Your coins: {coins_self}    Opponent coins: {coins_opp}    Wrestler at cell {pos}.",
            f"Legal bids: {legal}  (0..coins)",
            "  (blind: choose without seeing opponent's bid)",
        ]
        return "\n".join(lines)

    def render_terminal(self, state: pyspiel.State) -> str:
        ret = state.returns()
        if abs(ret[0]) < 1e-9:
            return "Game over: draw."
        winner = 0 if ret[0] > 0 else 1
        return f"Game over: player {winner} wins. Returns: {[int(r) for r in ret]}"

    def _count_leading_ones(self, info: list[float], lo: int, hi: int) -> int:
        # Coins are encoded as a one-hot over (0..max); count is the index of the
        # set bit. Approximate by summing the slice length bounded by data.
        try:
            return next((i for i in range(hi - lo) if info[lo + i] > 0.5), hi - lo)
        except (IndexError, ValueError):
            return -1

    def _wrestler_pos(self, info: list[float], lo: int, hi: int) -> int:
        try:
            return next((i for i in range(hi - lo) if info[lo + i] > 0.5), -1)
        except (IndexError, ValueError):
            return -1
