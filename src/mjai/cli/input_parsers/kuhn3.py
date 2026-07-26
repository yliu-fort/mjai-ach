"""Three-player Kuhn input parser (AGENTS.md D13).

The action set is identical to two-player Kuhn — 0 = Pass, 1 = Bet, with the
same intent-based aliases — so this delegates to the 2p parser rather than
restating the alias table. See :mod:`mjai.cli.input_parsers.kuhn` for why
``call`` must map to 1 and not 0.
"""

from __future__ import annotations

from mjai.cli.input_parsers.kuhn import create as create_kuhn_parser
from mjai.cli.interfaces import HumanInputParser


def create() -> HumanInputParser:
    return create_kuhn_parser()
