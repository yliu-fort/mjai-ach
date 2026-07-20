"""Oshi-Zumo input parser: bid amount 0..coins (the action id)."""

from __future__ import annotations

from mjai.cli.interfaces import HumanInputParser


def create() -> HumanInputParser:
    return _OshiZumoParser()


class _OshiZumoParser:
    def prompt(self, legal_actions: list[int], observer_player: int) -> str:
        return f"Your bid (0..coins; legal = {legal_actions}): "

    def parse(self, raw: str, legal_actions: list[int]) -> int:
        s = raw.strip()
        if s.isdigit():
            a = int(s)
        else:
            raise ValueError(f"could not parse {raw!r}; use a bid 0..coins")
        if a not in legal_actions:
            raise ValueError(f"bid {a} not legal; legal = {legal_actions}")
        return a
