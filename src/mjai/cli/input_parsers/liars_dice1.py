"""Liar's-Dice input parser: action id, 'q f' (quantity face), or 'challenge'.

OpenSpiel encoding for ``liars_dice(numdice=1,dice_sides=6)`` (the only config
this game is registered with — see ``games/loader.py``):

  - Bid action id = ``(quantity - 1) * 6 + (face - 1)``  (range 0..11).
    So "1 5" -> 4, "2 1" -> 6. (The old parser had a spurious ``+1``.)
  - Challenge ("liar") = action id ``12`` = ``num_distinct_actions() - 1``.
    It is only legal once a bid is on the table; at the opening move entering
    "challenge" yields id 12, which the final legality check rejects.

Action id ``0`` is a legal opening bid ("one 1"), NOT a challenge.
"""

from __future__ import annotations

from mjai.cli.interfaces import HumanInputParser

# Fixed for this game's config (numdice=1, dice_sides=6). Equals
# total_dice * dice_sides = 2 * 6 = num_distinct_actions() - 1.
_CHALLENGE_ID = 12
_DICE_SIDES = 6

# Word aliases for the challenge action. "0" is intentionally NOT here: it is a
# valid opening bid (one 1), so mapping it to challenge would be silently wrong.
_ALIASES = {
    "challenge": _CHALLENGE_ID,
    "c": _CHALLENGE_ID,
    "call": _CHALLENGE_ID,
    "liar": _CHALLENGE_ID,
}


def create() -> HumanInputParser:
    return _LiarsDiceParser()


class _LiarsDiceParser:
    def prompt(self, legal_actions: list[int], observer_player: int) -> str:
        return f"Your action (id in {legal_actions}, or 'q f', or 'challenge'): "

    def parse(self, raw: str, legal_actions: list[int]) -> int:
        s = raw.strip().lower()
        if s in _ALIASES:
            a = _ALIASES[s]
        elif " " in s:
            try:
                q_str, f_str = s.split()
                q, f = int(q_str), int(f_str)
            except ValueError as e:
                raise ValueError(f"could not parse {raw!r}; use 'q f'") from e
            # OpenSpiel encoding: action = (quantity-1)*6 + (face-1).
            a = (q - 1) * _DICE_SIDES + (f - 1)
        elif s.isdigit():
            a = int(s)
        else:
            raise ValueError(f"could not parse {raw!r}")
        if a not in legal_actions:
            raise ValueError(f"action {a} not legal; legal = {legal_actions}")
        return a
