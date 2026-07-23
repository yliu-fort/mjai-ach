"""Leduc input parser: action id 0/1/2 (fold/call/raise) or f/c/b.

Leduc Poker action ids (verified via ``action_to_string``): 0 = Fold, 1 = Call,
2 = Raise. There is no check action in Leduc (the ante is mandatory; the first
move is either Call = match the ante or Raise). The old prompt labelled action 1
"check/call" and action 2 "bet/raise", which implied a check/bet that does not
exist in this game; aliases are kept (they map to the right ids via the legality
check) but the prompt now names the real actions.
"""

from __future__ import annotations

from mjai.cli.interfaces import HumanInputParser

_ALIASES = {"f": 0, "fold": 0, "c": 1, "check": 1, "call": 1, "b": 2, "bet": 2, "raise": 2}


def create() -> HumanInputParser:
    return _LeducParser()


class _LeducParser:
    def prompt(self, legal_actions: list[int], observer_player: int) -> str:
        names = {0: "fold", 1: "call", 2: "raise"}
        opts = ", ".join(f"{a}={names.get(a, str(a))}" for a in legal_actions)
        return f"Your action ({opts}): "

    def parse(self, raw: str, legal_actions: list[int]) -> int:
        s = raw.strip().lower()
        if s in _ALIASES:
            a = _ALIASES[s]
        elif s.isdigit():
            a = int(s)
        else:
            raise ValueError(f"could not parse {raw!r}; use fold/call/raise or 0/1/2")
        if a not in legal_actions:
            raise ValueError(f"action {a} not legal; legal = {legal_actions}")
        return a
