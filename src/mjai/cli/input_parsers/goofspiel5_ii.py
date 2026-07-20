"""Goofspiel-5 input parser: card value 1..5 (the action id)."""

from __future__ import annotations

from mjai.cli.interfaces import HumanInputParser


def create() -> HumanInputParser:
    return _GoofspielParser()


class _GoofspielParser:
    def prompt(self, legal_actions: list[int], observer_player: int) -> str:
        return f"Play a card (value in {legal_actions}, blind): "

    def parse(self, raw: str, legal_actions: list[int]) -> int:
        s = raw.strip()
        if s.isdigit():
            a = int(s)
        else:
            raise ValueError(f"could not parse {raw!r}; use a card value {legal_actions}")
        if a not in legal_actions:
            raise ValueError(f"card {a} not in your hand; legal = {legal_actions}")
        return a
