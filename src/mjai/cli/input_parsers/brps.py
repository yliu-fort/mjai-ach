"""BRPS input parser: 0/1/2 or R/P/S."""

from __future__ import annotations

from mjai.cli.interfaces import HumanInputParser

_ALIASES = {"r": 0, "rock": 0, "p": 1, "paper": 1, "s": 2, "scissors": 2}


def create() -> HumanInputParser:
    return _BRPSParser()


class _BRPSParser:
    def prompt(self, legal_actions: list[int], observer_player: int) -> str:
        return "Your move [R/P/S or 0/1/2]: "

    def parse(self, raw: str, legal_actions: list[int]) -> int:
        s = raw.strip().lower()
        if s in _ALIASES:
            a = _ALIASES[s]
        elif s.isdigit():
            a = int(s)
        else:
            raise ValueError(f"could not parse {raw!r}; use R/P/S or 0/1/2")
        if a not in legal_actions:
            raise ValueError(f"action {a} not legal; legal = {legal_actions}")
        return a
