"""Liar's-Dice input parser: action id, or 'q f' (quantity face), or 'challenge'."""

from __future__ import annotations

from mjai.cli.interfaces import HumanInputParser


def create() -> HumanInputParser:
    return _LiarsDiceParser()


class _LiarsDiceParser:
    def prompt(self, legal_actions: list[int], observer_player: int) -> str:
        return f"Your action (id in {legal_actions}, or 'q f', or 'challenge'): "

    def parse(self, raw: str, legal_actions: list[int]) -> int:
        s = raw.strip().lower()
        if s in ("challenge", "c", "call", "liar", "0"):
            a = 0
        elif " " in s:
            try:
                q_str, f_str = s.split()
                q, f = int(q_str), int(f_str)
            except ValueError as e:
                raise ValueError(f"could not parse {raw!r}; use 'q f'") from e
            # OpenSpiel encoding: action = (q-1)*6 + (f-1) + 1.
            a = (q - 1) * 6 + (f - 1) + 1
        elif s.isdigit():
            a = int(s)
        else:
            raise ValueError(f"could not parse {raw!r}")
        if a not in legal_actions:
            raise ValueError(f"action {a} not legal; legal = {legal_actions}")
        return a
