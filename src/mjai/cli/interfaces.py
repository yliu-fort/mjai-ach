"""CLI render/parser protocols (AGENTS.md §1 D10, §4).

Each game ships two small files under cli/renderers/<game>.py and
cli/input_parsers/<game>.py. Adding a game = one of each (AGENTS.md §4). The
CLI smoke test and import checker reject a registered game missing either.

  - :class:`GameRenderer` turns a pyspiel state into a human-readable string,
    from the perspective of the acting/observing player.
  - :class:`HumanInputParser` reads raw text from input() and validates it
    against the legal-action set.

For simultaneous-move games (BRPS, Goofspiel, Oshi-Zumo) the renderer must
**not** reveal the opponent's simultaneous choice — the human enters their own
blind, then the joint outcome is shown on the next render. This preserves the
game's information structure (AGENTS.md §4, §1 D10).

Every renderer also exposes :meth:`render_public` — a public-information-only
view used when a human is spectating a robot's turn (the robot's private info
must never leak to the human, INV-1).
"""

from __future__ import annotations

from typing import Protocol

import pyspiel


class GameRenderer(Protocol):
    """Contract for per-game state rendering."""

    def render(self, state: pyspiel.State, observer_player: int | None) -> str:
        """Human-readable view of ``state`` for ``observer_player``.

        For perfect-info games observer_player can be None (full board). For
        imperfect-info games it MUST be set so private info is filtered.
        """
        ...

    def render_public(self, state: pyspiel.State) -> str:
        """Public-information-only view of ``state`` (no player's private info).

        Used when a human is spectating a robot's turn: shows only public state
        (pot, public board, bidding history, action count) and never any
        player's private card / die / hand. For perfect-info games this equals
        ``render(state, None)``; for simultaneous-move games it shows the
        public board but not either player's pending choice. Required to honour
        INV-1 when a human shares a table with a robot (AGENTS.md §4).
        """
        ...

    def render_terminal(self, state: pyspiel.State) -> str:
        """Final-result view: show returns + a short summary."""
        ...


class HumanInputParser(Protocol):
    """Contract for per-game human input parsing."""

    def prompt(self, legal_actions: list[int], observer_player: int) -> str:
        """The text to display before reading input()."""
        ...

    def parse(self, raw: str, legal_actions: list[int]) -> int:
        """Convert raw input text to a legal action id.

        Raises ValueError on anything not parseable / not legal. The caller
        loops on this until a valid action comes in.
        """
        ...
