"""Leduc input parser: action id 0/1/2 (fold/check/bet) or f/c/b."""

from __future__ import annotations

from mjai.cli.interfaces import HumanInputParser

_ALIASES = {"f": 0, "fold": 0, "c": 1, "check": 1, "call": 1, "b": 2, "bet": 2, "raise": 2}


def create() -> HumanInputParser:
    return _LeducParser()


class _LeducParser:
    def prompt(self, legal_actions: list[int], observer_player: int) -> str:
        names = {0: "fold", 1: "check/call", 2: "bet/raise"}
        opts = ", ".join(f"{a}={names.get(a, str(a))}" for a in legal_actions)
        return f"Your action ({opts}): "

    def parse(self, raw: str, legal_actions: list[int]) -> int:
        s = raw.strip().lower()
        if s in _ALIASES:
            a = _ALIASES[s]
        elif s.isdigit():
            a = int(s)
        else:
            raise ValueError(f"could not parse {raw!r}; use fold/check/bet or 0/1/2")
        if a not in legal_actions:
            raise ValueError(f"action {a} not legal; legal = {legal_actions}")
        return a
