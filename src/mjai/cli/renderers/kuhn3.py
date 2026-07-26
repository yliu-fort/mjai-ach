"""Three-player Kuhn Poker renderer (AGENTS.md D13).

Four-card deck (J/Q/K/A), three seats, three opening chance deals. The whole
implementation is :class:`mjai.cli.renderers.kuhn.KuhnRenderer` at
``num_players=3`` — see that module for the info-state layout, which is shared
and must stay in one place.
"""

from __future__ import annotations

from mjai.cli.interfaces import GameRenderer
from mjai.cli.renderers.kuhn import KuhnRenderer


def create() -> GameRenderer:
    return KuhnRenderer(num_players=3)
