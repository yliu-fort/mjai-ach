"""Kuhn input parser: action id (0=check/call, 1=bet/raise) or 'c'/'b'."""

from __future__ import annotations

from mjai.cli.interfaces import HumanInputParser

_ALIASES = {"c": 0, "check": 0, "call": 0, "b": 1, "bet": 1, "raise": 1}


def create() -> HumanInputParser:
    return _KuhnParser()


class _KuhnParser:
    def prompt(self, legal_actions: list[int], observer_player: int) -> str:
        names = {0: "check/call", 1: "bet/raise"}
        opts = ", ".join(f"{a}={names.get(a, str(a))}" for a in legal_actions)
        return f"Your action ({opts}): "

    def parse(self, raw: str, legal_actions: list[int]) -> int:
        s = raw.strip().lower()
        if s in _ALIASES:
            a = _ALIASES[s]
        elif s.isdigit():
            a = int(s)
        else:
            raise ValueError(f"could not parse {raw!r}; use check/bet or 0/1")
        if a not in legal_actions:
            raise ValueError(f"action {a} not legal; legal = {legal_actions}")
        return a
