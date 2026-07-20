"""Tic-Tac-Toe renderer: 3x3 grid, perfect info (observer_player unused)."""

from __future__ import annotations

import pyspiel

from mjai.cli.interfaces import GameRenderer

_MARKS = {0: "X", 1: "O", -1: "."}


def create() -> GameRenderer:
    return _TTTRenderer()


class _TTTRenderer:
    def render(self, state: pyspiel.State, observer_player: int | None) -> str:
        if state.is_terminal():
            return self.render_terminal(state)
        board = state.observation_tensor(0)
        # TTT observation is 3 planes of 9 (one per player's marks); flatten to marks.
        cells = [["."] * 3 for _ in range(3)]
        for r in range(3):
            for c in range(3):
                idx = r * 3 + c
                if board[idx] == 1:
                    cells[r][c] = "X"
                elif board[idx + 9] == 1:
                    cells[r][c] = "O"
        rows = ["  0 1 2", "  -----"]
        for r in range(3):
            rows.append(f"{r} " + " ".join(cells[r]))
        player = state.current_player()
        rows.append(f"\nPlayer {player} ({_MARKS[player]}) to move. Action = row*3 + col.")
        return "\n".join(rows)

    def render_terminal(self, state: pyspiel.State) -> str:
        ret = state.returns()
        if abs(ret[0]) < 1e-9:
            return "Game over: draw."
        winner = 0 if ret[0] > 0 else 1
        return (
            f"Game over: player {winner} ({_MARKS[winner]}) wins. Returns: {[int(r) for r in ret]}"
        )
