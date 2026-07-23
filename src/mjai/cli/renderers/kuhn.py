"""Kuhn Poker renderer: 3-card deck (J/Q/K = 0/1/2), per-player private card.

Information-state tensor layout for ``kuhn_poker`` (verified empirically against
pyspiel 2.0.1, length 11):

  [0]      player-id bit 0 (P0 = hot here)
  [1]      player-id bit 1 (P1 = hot here)
  [2:5]    OWN private card one-hot (J/Q/K = slots 2/3/4; card_id = slot - 2)
  [5:11]   public action history encoding (bet/call sequence)

The old renderer read ``info[0:3]`` for the card. Slot 0 is the player-id bit,
which is hot for P0 regardless of the dealt card, so the renderer always printed
"your card: J". The card is actually at ``info[2:5]``.

``state.history()`` **includes the two opening chance deals** (P0's card, then
P1's card) before any betting action. The old ``_public_history`` walked the
entire history and so rendered a freshly dealt hand (history ``[1, 2]``) as the
bogus sequence "bet check". We skip the leading two chance outcomes.

Action semantics (verified via ``action_to_string``):
  - action 0 = "Pass": a check on the first move, a *fold* when facing a bet.
  - action 1 = "Bet": an opening bet, a *call* when facing a bet.
There is no separate fold/call action id in Kuhn.
"""

from __future__ import annotations

import pyspiel

from mjai.cli.interfaces import GameRenderer

_CARD = {0: "J", 1: "Q", 2: "K"}

# Own private card occupies info-state slots 2..4 (card_id = slot - 2).
_CARD_SLOT_START = 2
_CARD_SLOT_END = 5  # exclusive
# The first two entries of state.history() are the chance deals.
_N_CHANCE_DEALS = 2


def create() -> GameRenderer:
    return _KuhnRenderer()


class _KuhnRenderer:
    def render(self, state: pyspiel.State, observer_player: int | None) -> str:
        if state.is_terminal():
            return self.render_terminal(state)
        p = observer_player if observer_player is not None else state.current_player()
        info = state.information_state_tensor(p)
        # Private card from the OWN-card one-hot slots (NOT [0:3] — that's player-id).
        card_idx = next(
            (
                i - _CARD_SLOT_START
                for i in range(_CARD_SLOT_START, _CARD_SLOT_END)
                if info[i] > 0.5
            ),
            None,
        )
        card = _CARD.get(card_idx if card_idx is not None else -1, "?")
        history = self._public_history(state)
        lines = [
            f"Kuhn Poker — you are player {p}, your card: {card}",
            f"Pot: {self._pot(state)}    Actions so far: {' '.join(history) or '(none)'}",
        ]
        legal = state.legal_actions(p)
        names = {0: "pass (check/fold)", 1: "bet (call)"}
        lines.append("Legal: " + ", ".join(f"{a}={names.get(a, str(a))}" for a in legal))
        return "\n".join(lines)

    def render_terminal(self, state: pyspiel.State) -> str:
        ret = state.returns()
        if abs(ret[0]) < 1e-9:
            return "Hand over: tie."
        winner = 0 if ret[0] > 0 else 1
        return (
            f"Hand over: player {winner} wins {abs(ret[0]):+.0f}. "
            f"Returns: {[int(r) for r in ret]}"
        )

    def _public_history(self, state: pyspiel.State) -> list[str]:
        """Betting sequence only, skipping the two opening chance deals.

        Action 0 = Pass (check or fold), action 1 = Bet (bet or call). Both
        are the same ids in Kuhn — only the round context differs.
        """
        out = []
        for a in state.history()[_N_CHANCE_DEALS:]:
            out.append("bet" if a == 1 else "pass")
        return out

    def _pot(self, state: pyspiel.State) -> int:
        # Each player antes 1; a bet adds 1 per player who bets/calls.
        bets = sum(1 for a in state.history()[_N_CHANCE_DEALS:] if a == 1)
        return 2 + bets
