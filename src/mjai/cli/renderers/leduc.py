"""Leduc Poker renderer: 2-round Hold'em with a 6-card deck (AGENTS.md §1 D8).

Information-state tensor layout for ``leduc_poker`` (verified empirically
against pyspiel 2.0.1, length 30):

  [0]      player-id bit 0 (P0 = hot here)
  [1]      player-id bit 1 (P1 = hot here)
  [2:8]    OWN private (hole) card one-hot over the 6 deck cards
           (card_id = slot - 2; rank = card_id % 3 -> 0=J, 1=Q, 2=K)
  [8:14]   PUBLIC board card one-hot over the 6 deck cards (0 until the flop)
  [14:16]  round one-hot (0, 1)
  [16:]    betting sequence

The old renderer read the private card from ``info[0:3]`` (the player-id bits +
first card slot) and the public card from ``info[3:6]``. Slot 0 is the player-id
bit, so the private card was always misread as 'J', and the public card region
is actually ``[8:14]`` (a 6-card one-hot, not a 3-rank one-hot), so the flop
never showed.

``state.history()`` interleaves chance outcomes (the two hole-card deals at the
start, plus the flop between rounds) with player betting actions. The chance
positions are not at fixed offsets — the flop lands at ``2 + n_round1_actions``
— so we rebuild the player-only action sequence by replaying from the initial
state and tagging each step as chance or player move.

Action ids (verified via ``action_to_string``): 0 = Fold, 1 = Call, 2 = Raise.
"""

from __future__ import annotations

import pyspiel

from mjai.cli.interfaces import GameRenderer

_RANK = {0: "J", 1: "Q", 2: "K"}

# Info-state slot ranges (see module docstring).
_OWN_CARD_SLOT_START = 2
_OWN_CARD_SLOT_END = 8  # exclusive; 6 deck cards
_PUBLIC_CARD_SLOT_START = 8
_PUBLIC_CARD_SLOT_END = 14  # exclusive

_ACTION_NAMES = {0: "fold", 1: "call", 2: "raise"}


def create() -> GameRenderer:
    return _LeducRenderer()


class _LeducRenderer:
    def render(self, state: pyspiel.State, observer_player: int | None) -> str:
        if state.is_terminal():
            return self.render_terminal(state)
        p = observer_player if observer_player is not None else state.current_player()
        info = state.information_state_tensor(p)
        private = self._card_rank(info, _OWN_CARD_SLOT_START, _OWN_CARD_SLOT_END)
        public = self._card_rank(info, _PUBLIC_CARD_SLOT_START, _PUBLIC_CARD_SLOT_END)
        private_str = _RANK.get(private, "?") if private is not None else "?"
        public_str = _RANK.get(public, "-") if public is not None else "-"
        lines = [
            f"Leduc Poker — you are player {p}.",
            f"Private card: {private_str}    Public card: {public_str}",
            f"Action history: {self._history(state)}",
        ]
        legal = state.legal_actions(p)
        lines.append("Legal: " + ", ".join(f"{a}={_ACTION_NAMES.get(a, str(a))}" for a in legal))
        return "\n".join(lines)

    def render_public(self, state: pyspiel.State) -> str:
        """Public-only view: board card + betting history, no hole cards.

        Used when a human is spectating a robot's turn (INV-1). The public
        board card and the betting sequence are public; each player's hole card
        is private. The board card is read from P0's info-state, but only the
        *public* region, so no private card leaks.
        """
        if state.is_terminal():
            return self.render_terminal(state)
        info = state.information_state_tensor(0)
        public = self._card_rank(info, _PUBLIC_CARD_SLOT_START, _PUBLIC_CARD_SLOT_END)
        public_str = _RANK.get(public, "-") if public is not None else "-"
        return (
            "Leduc Poker — public view.\n"
            f"Public card: {public_str}\n"
            f"Action history: {self._history(state)}"
        )

    def render_terminal(self, state: pyspiel.State) -> str:
        ret = state.returns()
        if abs(ret[0]) < 1e-9:
            return "Hand over: tie."
        winner = 0 if ret[0] > 0 else 1
        return f"Hand over: player {winner} wins. Returns: {[int(r) for r in ret]}"

    def _card_rank(self, info: list[float], lo: int, hi: int) -> int | None:
        """Rank (0/1/2) of the one-hot card in slots [lo:hi], or None."""
        for slot in range(lo, hi):
            if info[slot] > 0.5:
                return (slot - lo) % 3
        return None

    def _history(self, state: pyspiel.State) -> str:
        """Player betting sequence only, with chance outcomes (deals/flop) removed.

        The flop's position in ``state.history()`` depends on how many round-1
        actions preceded it, so we cannot skip chance by a fixed offset. Instead
        we replay from the initial state, tagging each applied action as chance
        or player move, and keep only the player moves.
        """
        actions = list(state.history())
        game = state.get_game()
        cursor = game.new_initial_state()
        out: list[str] = []
        for a in actions:
            is_chance = cursor.is_chance_node()
            cursor.apply_action(a)
            if not is_chance:
                out.append(_ACTION_NAMES.get(a, str(a)))
        return " ".join(out) or "(none)"
