"""Unit tests for the Tic-Tac-Toe renderer (F3, §5).

OpenSpiel's tic_tac_toe observation planes are [empty, O (player 1), X
(player 0)] — verified empirically against pyspiel. The renderer must show an
empty board as all dots and place X/O marks at their true cells.
"""

from __future__ import annotations

import pyspiel

from mjai.cli.renderers.ttt import create


def _new_game() -> pyspiel.Game:
    return pyspiel.load_game("tic_tac_toe")


def _grid_lines(rendered: str) -> list[str]:
    """The three board rows of the render output (after the 2 header lines)."""
    return rendered.splitlines()[2:5]


def test_empty_board_renders_all_dots():
    state = _new_game().new_initial_state()
    out = create().render(state, observer_player=0)
    assert _grid_lines(out) == ["0 . . .", "1 . . .", "2 . . ."]
    # No spurious marks anywhere on an empty board.
    assert "X" not in "\n".join(_grid_lines(out))
    assert "O" not in "\n".join(_grid_lines(out))


def test_x_mark_appears_at_played_cell():
    state = _new_game().new_initial_state()
    state.apply_action(0)  # X at (row 0, col 0)
    out = create().render(state, observer_player=1)
    assert _grid_lines(out) == ["0 X . .", "1 . . .", "2 . . ."]


def test_o_mark_appears_at_played_cell():
    state = _new_game().new_initial_state()
    state.apply_action(0)  # X at (0,0)
    state.apply_action(4)  # O at (1,1)
    out = create().render(state, observer_player=0)
    assert _grid_lines(out) == ["0 X . .", "1 . O .", "2 . . ."]


def test_marks_are_board_absolute_not_observer_relative():
    """The tensor is identical for both observers; both see the same board."""
    game = _new_game()
    state = game.new_initial_state()
    state.apply_action(8)  # X at (2,2)
    state.apply_action(3)  # O at (1,0)
    for observer in (0, 1, None):
        out = create().render(state, observer_player=observer)
        assert _grid_lines(out) == ["0 . . .", "1 O . .", "2 . . X"]


def test_to_move_line_names_current_player():
    state = _new_game().new_initial_state()
    assert "Player 0 (X) to move" in create().render(state, observer_player=None)
    state.apply_action(4)
    assert "Player 1 (O) to move" in create().render(state, observer_player=None)


def test_terminal_render_still_works():
    state = _new_game().new_initial_state()
    for a in [0, 3, 1, 4, 2]:  # X completes the top row
        state.apply_action(a)
    out = create().render(state, observer_player=None)
    assert "player 0 (X) wins" in out
