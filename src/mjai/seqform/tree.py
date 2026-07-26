"""Build the sequence-form index of an extensive game by exact tree traversal.

The sequence form (Koller-Megiddo-von Stengel) replaces the exponentially large
space of pure strategies with one *realization probability* per sequence, where
a sequence is "the empty sequence" or "(information set, action)". Under perfect
recall each information set has a unique parent sequence, so a behaviour
strategy induces realization probabilities by a single pass down that tree, and
the expected payoff becomes **multilinear** in the players' realization plans:

    u_s(x_1, ..., x_n) = sum over terminals z of
        chance(z) * util_s(z) * prod over players q of x_q(seq_q(z))

That multilinearity is the whole reason the pACH programme lives in these
coordinates: it is what makes the population's fitness exactly linear in one
actor's realization plan (Generative-ach.md §2.1, Lemma 1), hence what makes a
*linear* critic realizable rather than merely convenient.

**Parameter-count reconciliation.** The research plan quotes 2p Kuhn as "12
behaviour parameters" and 3p Kuhn as "48" — one degree of freedom per binary
information set (12 and 48 information sets respectively). This module carries
the *redundant* softmax parameterization instead: ``max_actions`` logits per
information set, so 24 and 96 slots. The two induce identical behaviour
strategies. They do **not** induce the same reference measure: pushing N(0, I)
through a 2-logit softmax gives a logit gap of variance 2, versus variance 1 for
a single sigmoid logit. Anything that reports coverage against a Gibbs reference
(研究计划 §2.3, §5.1 A2) must state which parameterization it used.

Scope: turn-based games only. Simultaneous-move games have no sequence form in
this sense and raise :class:`UnsupportedGameError` rather than being silently
converted (AGENTS.md §11).
"""

from __future__ import annotations

from dataclasses import dataclass

import pyspiel
import torch

from mjai.games.loader import GameSpec

# Index 0 of every player's sequence vector is the empty sequence, whose
# realization probability is 1 by definition.
EMPTY_SEQUENCE = 0

# Sentinel in ``sequence_of`` for an (information set, action) pair that does not
# exist because the action is illegal there.
NO_SEQUENCE = -1


class UnsupportedGameError(ValueError):
    """The game has no sequence form we are willing to build (e.g. simultaneous)."""


class PerfectRecallError(ValueError):
    """An information set was reached via two different own-action histories.

    Sequence form requires perfect recall: every information set must have ONE
    parent sequence. Games that violate this cannot be represented here, and we
    refuse rather than silently picking a parent.
    """


@dataclass(frozen=True)
class SequenceForm:
    """Static sequence-form index of one game.

    All tensors are on CPU; float ones are float64 (AGENTS.md D19). Built once
    per game and reused — it describes the tree, not any strategy.

    Attributes:
        game_name: short game name it was built from.
        num_players: seat count.
        max_actions: width of the padded per-information-set action axis.
        infoset_keys: information-set string per row, in discovery order.
        infoset_player: owning player per row, int64 ``[I]``.
        infoset_level: depth in the owner's own sequence tree, int64 ``[I]``;
            level 0 means the parent is the empty sequence.
        legal_mask: bool ``[I, max_actions]``.
        parent_sequence: int64 ``[I]``, index into the OWNER's sequence vector.
        sequence_of: int64 ``[I, max_actions]``, index into the owner's sequence
            vector, or :data:`NO_SEQUENCE` where the action is illegal.
        num_sequences: per-player sequence count, including the empty sequence.
        level_rows: information-set rows grouped by level, ascending. Realization
            plans are built by walking this forwards, best responses by walking
            it backwards.
        terminal_chance: float64 ``[T]``, product of chance probabilities.
        terminal_utility: float64 ``[T, num_players]``.
        terminal_sequence: int64 ``[T, num_players]``.
    """

    game_name: str
    num_players: int
    max_actions: int
    infoset_keys: tuple[str, ...]
    infoset_player: torch.Tensor
    infoset_level: torch.Tensor
    legal_mask: torch.Tensor
    parent_sequence: torch.Tensor
    sequence_of: torch.Tensor
    num_sequences: tuple[int, ...]
    level_rows: tuple[torch.Tensor, ...]
    terminal_chance: torch.Tensor
    terminal_utility: torch.Tensor
    terminal_sequence: torch.Tensor

    @property
    def num_infosets(self) -> int:
        return len(self.infoset_keys)

    @property
    def num_terminals(self) -> int:
        return int(self.terminal_chance.shape[0])

    def rows_of(self, player: int) -> torch.Tensor:
        """Information-set row indices owned by ``player``, ascending."""
        return torch.nonzero(self.infoset_player == player, as_tuple=False).flatten()

    def row_of_key(self, player: int, key: str) -> int:
        """Row index for one information-set string; raises if unknown.

        Callers that need to line these rows up with another implementation's
        ordering (the D14 parity check does) should permute through this rather
        than assume the orders agree.
        """
        for row, (owner, name) in enumerate(
            zip(self.infoset_player.tolist(), self.infoset_keys, strict=True)
        ):
            if owner == player and name == key:
                return row
        raise KeyError(f"{self.game_name}: no information set {key!r} for player {player}")

    def __repr__(self) -> str:
        return (
            f"SequenceForm({self.game_name!r}, players={self.num_players}, "
            f"infosets={self.num_infosets}, sequences={self.num_sequences}, "
            f"terminals={self.num_terminals})"
        )


class _Scratch:
    """Mutable traversal state, discarded once the frozen SequenceForm is built."""

    def __init__(self, spec: GameSpec) -> None:
        self.spec = spec
        self.n = spec.num_players
        self.key_to_row: dict[tuple[int, str], int] = {}
        self.keys: list[str] = []
        self.owner: list[int] = []
        self.level: list[int] = []
        self.parent: list[int] = []
        self.legal: list[list[int]] = []
        self.seq_of: list[dict[int, int]] = []
        # Per player, the level of each sequence; index 0 is the empty sequence.
        self.sequence_level: list[list[int]] = [[0] for _ in range(self.n)]
        self.term_chance: list[float] = []
        self.term_util: list[list[float]] = []
        self.term_seq: list[list[int]] = []

    def infoset_string(self, state: pyspiel.State, player: int) -> str:
        """The key identifying this decision point for ``player``.

        Mirrors the loader's encoding choice: information state when the game
        provides one, observation string otherwise. A game where the fallback
        breaks perfect recall trips :class:`PerfectRecallError` rather than
        producing a quietly wrong index.
        """
        if self.spec.obs_kind == "information_state":
            return str(state.information_state_string(player))
        return str(state.observation_string(player))

    def register(self, state: pyspiel.State, player: int, parent_seq: int) -> int:
        """Find or create the row for the information set ``state`` is in."""
        key = self.infoset_string(state, player)
        legal = list(state.legal_actions(player))
        row = self.key_to_row.get((player, key))
        if row is not None:
            self._check_consistent(row, key, player, parent_seq, legal)
            return row
        row = len(self.keys)
        level = self.sequence_level[player][parent_seq]
        self.key_to_row[(player, key)] = row
        self.keys.append(key)
        self.owner.append(player)
        self.parent.append(parent_seq)
        self.legal.append(legal)
        self.level.append(level)
        assigned: dict[int, int] = {}
        for action in legal:
            assigned[action] = len(self.sequence_level[player])
            self.sequence_level[player].append(level + 1)
        self.seq_of.append(assigned)
        return row

    def _check_consistent(
        self, row: int, key: str, player: int, parent_seq: int, legal: list[int]
    ) -> None:
        if self.parent[row] != parent_seq:
            raise PerfectRecallError(
                f"{self.spec.name}: information set {key!r} of player {player} is reachable "
                f"via two different own-action histories (parent sequences "
                f"{self.parent[row]} and {parent_seq}). Sequence form requires perfect recall."
            )
        if self.legal[row] != legal:
            raise PerfectRecallError(
                f"{self.spec.name}: information set {key!r} of player {player} offers "
                f"different legal actions on different visits ({self.legal[row]} vs {legal}); "
                f"the information partition is inconsistent."
            )


def _traverse(scratch: _Scratch, state: pyspiel.State, chance: float, seqs: list[int]) -> None:
    """Depth-first walk recording information sets and terminals."""
    if state.is_terminal():
        scratch.term_chance.append(chance)
        scratch.term_util.append([float(u) for u in state.returns()])
        scratch.term_seq.append(list(seqs))
        return
    if state.is_chance_node():
        for action, prob in state.chance_outcomes():
            _traverse(scratch, state.child(action), chance * float(prob), seqs)
        return
    player = state.current_player()
    row = scratch.register(state, player, seqs[player])
    for action in scratch.legal[row]:
        child_seqs = list(seqs)
        child_seqs[player] = scratch.seq_of[row][action]
        _traverse(scratch, state.child(action), chance, child_seqs)


def build_sequence_form(spec: GameSpec) -> SequenceForm:
    """Enumerate ``spec``'s tree and return its :class:`SequenceForm`.

    Cost is one full traversal — 30 terminals on 2p Kuhn, 312 on 3p Kuhn, and
    still trivial on Leduc. Build once per game and hold the result; nothing
    here depends on a strategy.

    Raises:
        UnsupportedGameError: for simultaneous-move games.
        PerfectRecallError: if the information partition is not perfect-recall.
    """
    if spec.is_simultaneous:
        raise UnsupportedGameError(
            f"{spec.name} is a simultaneous-move game; it has no turn-based sequence "
            f"form. Convert it deliberately (pyspiel.convert_to_turn_based) and pass "
            f"the converted game if that is what you mean."
        )
    scratch = _Scratch(spec)
    _traverse(scratch, spec.new_state(), 1.0, [EMPTY_SEQUENCE] * spec.num_players)
    if not scratch.keys:
        raise UnsupportedGameError(f"{spec.name}: no decision nodes found")

    num_infosets = len(scratch.keys)
    max_actions = spec.num_actions
    legal_mask = torch.zeros((num_infosets, max_actions), dtype=torch.bool)
    sequence_of = torch.full((num_infosets, max_actions), NO_SEQUENCE, dtype=torch.int64)
    for row, actions in enumerate(scratch.legal):
        for action in actions:
            legal_mask[row, action] = True
            sequence_of[row, action] = scratch.seq_of[row][action]

    level = torch.tensor(scratch.level, dtype=torch.int64)
    level_rows = tuple(
        torch.nonzero(level == value, as_tuple=False).flatten()
        for value in range(int(level.max().item()) + 1)
    )
    return SequenceForm(
        game_name=spec.name,
        num_players=spec.num_players,
        max_actions=max_actions,
        infoset_keys=tuple(scratch.keys),
        infoset_player=torch.tensor(scratch.owner, dtype=torch.int64),
        infoset_level=level,
        legal_mask=legal_mask,
        parent_sequence=torch.tensor(scratch.parent, dtype=torch.int64),
        sequence_of=sequence_of,
        num_sequences=tuple(len(levels) for levels in scratch.sequence_level),
        level_rows=level_rows,
        terminal_chance=torch.tensor(scratch.term_chance, dtype=torch.float64),
        terminal_utility=torch.tensor(scratch.term_util, dtype=torch.float64),
        terminal_sequence=torch.tensor(scratch.term_seq, dtype=torch.int64),
    )
