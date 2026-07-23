"""Liar's Dice renderer: 1 die each, 6 sides. Player sees only their own die.

OpenSpiel action encoding for ``liars_dice(numdice=1,dice_sides=6)`` (verified
empirically against pyspiel 2.0.1):

  - Bids are action ids ``0..11``. Decoding (the *correct* formula — the old
    renderer's ``(a-1)//6+1`` was off by one and made faces wrap 6->1):
    ``quantity = a // 6 + 1``,  ``face = a % 6 + 1``.
    So 0->(1,1), 4->(1,5), 5->(1,6), 6->(2,1), ..., 11->(2,6).
  - Challenge ("liar") is action id ``num_distinct_actions() - 1`` = ``12`` here
    (= ``total_dice * dice_sides``). It is the only action that ends the round;
    action 0 is a valid opening bid ("one 1"), NOT a challenge.

``state.history()`` includes the two opening die rolls (chance outcomes) before
the bids. Those are private/roll info, not bids, and must not appear in the
bidding history. All chance nodes come first in Liar's Dice (rolls happen once
at the start), so we skip the leading ``num_players`` history entries.

Info-state tensor layout (length 21): slots ``[0,1]`` = player-id one-hot,
``[2..7]`` = the observer's OWN die one-hot (face = slot - 1), ``[8..11]`` =
the opponent's die (MUST NOT be shown to the observer), ``[12..20]`` = bid
history. The old renderer read slot 0 (the player-id bit, always hot) and so
always printed "your die: 1".
"""

from __future__ import annotations

import pyspiel

from mjai.cli.interfaces import GameRenderer

# Own-die one-hot occupies info-state slots 2..7 (face = slot - 1).
_OWN_DIE_SLOT_START = 2
_OWN_DIE_SLOT_END = 8  # exclusive
_DICE_SIDES = 6


def create() -> GameRenderer:
    return _LiarsDiceRenderer()


class _LiarsDiceRenderer:
    def render(self, state: pyspiel.State, observer_player: int | None) -> str:
        if state.is_terminal():
            return self.render_terminal(state)
        p = observer_player if observer_player is not None else state.current_player()
        game = state.get_game()
        challenge_id = game.num_distinct_actions() - 1
        die = self._own_die(state, p)
        history = self._history(state, game.num_players(), challenge_id)
        legal = state.legal_actions(p)
        last_bid = challenge_id - 1
        lines = [
            f"Liar's Dice — you are player {p}, your die: {die}",
            f"Bidding history: {history}",
            f"Legal actions: {legal}",
            (
                f"  (0..{last_bid} = 叫牌 quantity=k//6+1 of face=k%6+1; "
                f"{challenge_id} = 挑战/challenge)"
            ),
        ]
        return "\n".join(lines)

    def render_public(self, state: pyspiel.State) -> str:
        """Public-only view: bidding history, no player's die.

        Used when a human is spectating a robot's turn (INV-1). The dice are
        private until a challenge resolves; only the bid/challenge sequence is
        public.
        """
        if state.is_terminal():
            return self.render_terminal(state)
        game = state.get_game()
        challenge_id = game.num_distinct_actions() - 1
        history = self._history(state, game.num_players(), challenge_id)
        return f"Liar's Dice — public view.\nBidding history: {history}"

    def render_terminal(self, state: pyspiel.State) -> str:
        ret = state.returns()
        winner = 0 if ret[0] > 0 else 1
        return f"Round over: player {winner} wins. Returns: {[int(r) for r in ret]}"

    def _own_die(self, state: pyspiel.State, player: int) -> int | None:
        """The observer's own die face, read ONLY from the own-die slots."""
        info = state.information_state_tensor(player)
        for slot in range(_OWN_DIE_SLOT_START, _OWN_DIE_SLOT_END):
            if info[slot] > 0.5:
                return slot - _OWN_DIE_SLOT_START + 1
        return None

    def _history(self, state: pyspiel.State, num_players: int, challenge_id: int) -> str:
        """Bidding history as 叫牌/挑战 tokens, excluding the opening die rolls.

        The first ``num_players`` entries of ``state.history()`` are chance
        (die-roll) outcomes; bids and the optional challenge follow. We skip the
        rolls and decode only the bid/challenge actions.
        """
        out = []
        for a in state.history()[num_players:]:
            if a == challenge_id:
                out.append("挑战")
            else:
                q = a // _DICE_SIDES + 1
                f = a % _DICE_SIDES + 1
                out.append(f"叫牌 {q}x{f}")
        return " ".join(out) or "(无)"
