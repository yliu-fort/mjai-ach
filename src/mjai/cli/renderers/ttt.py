"""Tic-Tac-Toe renderer: 3x3 grid, perfect info (observer_player unused).

OpenSpiel's tic_tac_toe observation tensor is 3 planes of 9, ordered by
``CellState`` — plane 0 = empty cells, plane 1 = noughts (O, player 1),
plane 2 = crosses (X, player 0). The tensor is board-absolute (verified
empirically: identical for both observers), so we always read player 0's.
"""

from __future__ import annotations

import pyspiel

from mjai.cli.interfaces import GameRenderer

_MARKS = {0: "X", 1: "O", -1: "."}

# Plane offsets within the 27-element observation tensor (see module docstring).
_O_PLANE = 9
_X_PLANE = 18


def create() -> GameRenderer:
    return _TTTRenderer()


class _TTTRenderer:
    def render(self, state: pyspiel.State, observer_player: int | None) -> str:
        if state.is_terminal():
            return self.render_terminal(state)
        board = state.observation_tensor(0)
        cells = [["."] * 3 for _ in range(3)]
        for r in range(3):
            for c in range(3):
                idx = r * 3 + c
                if board[idx + _X_PLANE] == 1:
                    cells[r][c] = "X"
                elif board[idx + _O_PLANE] == 1:
                    cells[r][c] = "O"
                # else plane0 (empty) is set -> "."
        rows = ["  0 1 2", "  -----"]
        for r in range(3):
            rows.append(f"{r} " + " ".join(cells[r]))
        player = state.current_player()
        rows.append(f"\nPlayer {player} ({_MARKS[player]}) to move. Action = row*3 + col.")
        return "\n".join(rows)

    def render_public(self, state: pyspiel.State) -> str:
        """Public-only view. Tic-Tac-Toe is perfect-info, so this is the full
        board with no private state to filter.
        """
        return self.render(state, observer_player=0)

    def render_terminal(self, state: pyspiel.State) -> str:
        ret = state.returns()
        if abs(ret[0]) < 1e-9:
            return "Game over: draw."
        winner = 0 if ret[0] > 0 else 1
        return (
            f"Game over: player {winner} ({_MARKS[winner]}) wins. Returns: {[int(r) for r in ret]}"
        )
