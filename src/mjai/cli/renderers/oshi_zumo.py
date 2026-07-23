"""Oshi-Zumo renderer: simultaneous secret bidding, wrestler on a 3-cell ring.

Config: ``coins=5, size=3, horizon=20``. Action = bid (0..coins); the higher
bid pushes the wrestler one step toward the opponent. Spent coins are gone.
Simultaneous move => blind entry (AGENTS.md §4).

Observation tensor layout (verified empirically against pyspiel 2.0.1, length
21) — the game is perfect-information so both observers see identical tensors:

  [0:6]   P0 remaining coins one-hot (hot index == coins remaining; 5 -> slot 5)
  [6:12]  P1 remaining coins one-hot (coins = hot index - 6)
  [12:15] (unused gap in this build)
  [15:18] wrestler cell one-hot (cell = hot index - 15; 0/1/2)
  [18:21] (unused gap)

The old renderer queried the wrestler cell from ``[12:15]`` (always empty here),
so it always printed "Wrestler at cell -1". The coin reads were already correct
(hot index within the per-player region), but their helper names and comments
("count leading ones") were misleading — they actually return the one-hot index.
"""

from __future__ import annotations

import pyspiel

from mjai.cli.interfaces import GameRenderer

# Observation-tensor slot ranges (see module docstring).
_P0_COINS_LO, _P0_COINS_HI = 0, 6
_P1_COINS_LO, _P1_COINS_HI = 6, 12
_BOARD_LO, _BOARD_HI = 15, 18


def create() -> GameRenderer:
    return _OshiZumoRenderer()


class _OshiZumoRenderer:
    def render(self, state: pyspiel.State, observer_player: int | None) -> str:
        if state.is_terminal():
            return self.render_terminal(state)
        p = observer_player if observer_player is not None else state.current_player()
        info = state.observation_tensor(p)
        coins_self = self._one_hot_index(info, _P0_COINS_LO, _P0_COINS_HI)
        coins_opp = self._one_hot_index(info, _P1_COINS_LO, _P1_COINS_HI)
        pos = self._one_hot_index(info, _BOARD_LO, _BOARD_HI)
        legal = state.legal_actions(p)
        lines = [
            f"Oshi-Zumo — you are player {p}.",
            f"Your coins: {coins_self}    Opponent coins: {coins_opp}    "
            f"Wrestler at cell {pos}.",
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

    def _one_hot_index(self, info: list[float], lo: int, hi: int) -> int:
        """Index of the hot slot in ``info[lo:hi]`` (relative to ``lo``).

        For the coin regions this is the coins remaining; for the board region
        it is the wrestler cell. Returns ``-1`` if no slot is hot (should not
        happen on a valid state, but degrades cleanly).
        """
        for i in range(hi - lo):
            if info[lo + i] > 0.5:
                return i
        return -1
