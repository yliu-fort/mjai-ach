"""Tic-Tac-Toe input parser: accept "rc" (e.g. "01" = row 0 col 1) or action int."""

from __future__ import annotations

from mjai.cli.interfaces import HumanInputParser


def create() -> HumanInputParser:
    return _TTTParser()


class _TTTParser:
    def prompt(self, legal_actions: list[int], observer_player: int) -> str:
        return f"Your move (rc like '01', or action id {legal_actions}): "

    def parse(self, raw: str, legal_actions: list[int]) -> int:
        s = raw.strip()
        # Two-char "rc" form.
        if len(s) == 2 and s.isdigit():
            r, c = int(s[0]), int(s[1])
            a = r * 3 + c
            if a not in legal_actions:
                raise ValueError(f"cell ({r},{c}) is not legal; legal = {legal_actions}")
            return a
        # Plain integer action.
        try:
            a = int(s)
        except ValueError as e:
            raise ValueError(f"could not parse {raw!r}; use 'rc' or an action id") from e
        if a not in legal_actions:
            raise ValueError(f"action {a} not legal; legal = {legal_actions}")
        return a
