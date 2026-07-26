"""Kuhn Poker renderer, parameterized by player count (2p and 3p).

Information-state tensor layout for ``kuhn_poker``, verified empirically against
pyspiel 2.0.1 at both player counts (2p length 11, 3p length 17):

  [0 : N]            player-id one-hot (N = number of players)
  [N : N + N + 1]    OWN private card one-hot (the deck holds N + 1 cards)
  [N + N + 1 : ]     public action history encoding (bet/call sequence)

So the card block starts at slot ``N``, not at slot 0, and its width tracks the
deck rather than being fixed at 3. The old 2p renderer read ``info[0:3]`` for
the card; slot 0 is the player-id bit, hot for P0 regardless of the dealt card,
so it always printed "your card: J". Keeping ONE parameterized implementation
here (rather than copying it into a kuhn3 module) means that layout knowledge —
which has already been a bug source once — exists in exactly one place.

``state.history()`` **includes the opening chance deals** — one per player,
so two in 2p and three in 3p — before any betting action. The old
``_public_history`` walked the entire history and rendered a freshly dealt hand
(history ``[1, 2]``) as the bogus sequence "bet check". We skip the leading N
chance outcomes.

Action semantics (verified via ``action_to_string``, identical at both counts):
  - action 0 = "Pass": a check on the first move, a *fold* when facing a bet.
  - action 1 = "Bet": an opening bet, a *call* when facing a bet.
There is no separate fold/call action id in Kuhn.
"""

from __future__ import annotations

import pyspiel

from mjai.cli.interfaces import GameRenderer

# Card labels by rank; an N-player deck uses the first N + 1 of them.
_CARD_LABELS = ("J", "Q", "K", "A", "2", "3")


def create() -> GameRenderer:
    """The canonical 2-player Kuhn renderer."""
    return KuhnRenderer(num_players=2)


class KuhnRenderer:
    """Renders Kuhn Poker for ``num_players`` seats (deck = num_players + 1).

    Public because ``renderers/kuhn3.py`` constructs it with ``num_players=3``;
    the CLI itself only ever calls the module-level ``create()``.
    """

    def __init__(self, *, num_players: int) -> None:
        if num_players < 2:
            raise ValueError(f"Kuhn needs at least 2 players, got {num_players}")
        self.num_players = num_players
        self.num_cards = num_players + 1
        # Own private card occupies slots [N, N + num_cards); the leading N
        # slots are the player-id one-hot.
        self._card_slot_start = num_players
        self._card_slot_end = num_players + self.num_cards
        # One chance deal per player precedes the first betting action.
        self._n_chance_deals = num_players

    def render(self, state: pyspiel.State, observer_player: int | None) -> str:
        if state.is_terminal():
            return self.render_terminal(state)
        p = observer_player if observer_player is not None else state.current_player()
        info = state.information_state_tensor(p)
        # Private card from the OWN-card one-hot slots (NOT [0:3] — that's player-id).
        card_idx = next(
            (
                i - self._card_slot_start
                for i in range(self._card_slot_start, self._card_slot_end)
                if info[i] > 0.5
            ),
            None,
        )
        card = self._label(card_idx)
        history = self._public_history(state)
        lines = [
            f"{self._title()} — you are player {p}, your card: {card}",
            f"Pot: {self._pot(state)}    Actions so far: {' '.join(history) or '(none)'}",
        ]
        legal = state.legal_actions(p)
        names = {0: "pass (check/fold)", 1: "bet (call)"}
        lines.append("Legal: " + ", ".join(f"{a}={names.get(a, str(a))}" for a in legal))
        return "\n".join(lines)

    def render_public(self, state: pyspiel.State) -> str:
        """Public-only view: pot + betting history, no player's hole card.

        Used when a human is spectating a robot's turn (INV-1). The cards are
        private; only the pot and the pass/bet sequence are public.
        """
        if state.is_terminal():
            return self.render_terminal(state)
        history = self._public_history(state)
        return (
            f"{self._title()} — public view.\n"
            f"Pot: {self._pot(state)}    Actions so far: "
            f"{' '.join(history) or '(none)'}"
        )

    def render_terminal(self, state: pyspiel.State) -> str:
        """Final-result view.

        Reports every seat's return and names the top scorer(s). Three-player
        Kuhn can split the pot, and "the other player lost" is not a thing once
        N > 2, so there is no winner/loser pair to shortcut to.
        """
        ret = list(state.returns())
        if all(abs(r) < 1e-9 for r in ret):
            return "Hand over: tie."
        best = max(ret)
        winners = [i for i, r in enumerate(ret) if r >= best - 1e-9]
        who = ", ".join(f"player {i}" for i in winners)
        tally = ", ".join(f"p{i} {r:+.0f}" for i, r in enumerate(ret))
        return f"Hand over: {who} wins {best:+.0f}. Returns: {tally}"

    def _title(self) -> str:
        return "Kuhn Poker" if self.num_players == 2 else f"Kuhn Poker ({self.num_players}p)"

    def _label(self, card_idx: int | None) -> str:
        if card_idx is None or not 0 <= card_idx < self.num_cards:
            return "?"
        return _CARD_LABELS[card_idx]

    def _public_history(self, state: pyspiel.State) -> list[str]:
        """Betting sequence only, skipping the opening chance deals.

        Action 0 = Pass (check or fold), action 1 = Bet (bet or call). Both
        are the same ids in Kuhn — only the round context differs.
        """
        out = []
        for a in state.history()[self._n_chance_deals :]:
            out.append("bet" if a == 1 else "pass")
        return out

    def _pot(self, state: pyspiel.State) -> int:
        # Each player antes 1; a bet adds 1 per player who bets/calls.
        bets = sum(1 for a in state.history()[self._n_chance_deals :] if a == 1)
        return self.num_players + bets
