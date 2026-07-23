"""Kuhn input parser: action id (0=pass, 1=bet) with poker-style aliases.

Kuhn Poker has only two action ids (verified via ``action_to_string``):

  - action 0 = "Pass": a *check* on the first move, a *fold* when facing a bet.
  - action 1 = "Bet": an opening *bet*, a *call* when facing a bet.

There is no separate fold/call id, so poker aliases must map to the *intent*:
``check`` and ``fold`` -> pass (0); ``bet``, ``raise``, and ``call`` -> bet (1).
The trailing legality check rejects an alias that is illegal in context (e.g.
``fold`` is meaningless on the opening move where only a pass/bet is possible —
but action 0 is legal there too, so ``fold`` and ``check`` are interchangeable
on the first move; they only diverge in *meaning*, not in the resulting id).

Regression: the old parser mapped ``call`` -> 0, which is the *fold*. A human
typing ``call`` while facing a bet would fold and lose the pot (returns
``[+1, -1]`` from their perspective). It now maps to action 1 (the call/bet),
matching OpenSpiel's semantics.
"""

from __future__ import annotations

from mjai.cli.interfaces import HumanInputParser

# Intent-based aliases. "pass" is unambiguous; "check"/"fold" both mean "no
# money in"; "bet"/"raise"/"call" all mean "put a bet in" (Kuhn has one bet size).
_ALIASES = {
    "p": 0,
    "pass": 0,
    "check": 0,
    "fold": 0,
    "b": 1,
    "bet": 1,
    "raise": 1,
    "call": 1,
}


def create() -> HumanInputParser:
    return _KuhnParser()


class _KuhnParser:
    def prompt(self, legal_actions: list[int], observer_player: int) -> str:
        names = {0: "pass (check/fold)", 1: "bet (call)"}
        opts = ", ".join(f"{a}={names.get(a, str(a))}" for a in legal_actions)
        return f"Your action ({opts}): "

    def parse(self, raw: str, legal_actions: list[int]) -> int:
        s = raw.strip().lower()
        if s in _ALIASES:
            a = _ALIASES[s]
        elif s.isdigit():
            a = int(s)
        else:
            raise ValueError(f"could not parse {raw!r}; use pass/bet or 0/1")
        if a not in legal_actions:
            raise ValueError(f"action {a} not legal; legal = {legal_actions}")
        return a
