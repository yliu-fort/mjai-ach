"""Unit tests for the sequence-form tree index (AGENTS.md §5, D12).

The numbers pinned here are the ones Generative-ach.md §1 quotes for the games
the pACH programme runs on. Every Step-0 ground-truth artifact is computed in
these coordinates, so if OpenSpiel's tree ever shifts underneath us, this is
where it must surface — not three phases later in a "predicted vs measured"
table that quietly stops matching.
"""

from __future__ import annotations

import pytest
import torch

from mjai.games.loader import load_game
from mjai.seqform.plan import behavior_from_logits, realization_plans
from mjai.seqform.tree import (
    EMPTY_SEQUENCE,
    NO_SEQUENCE,
    PerfectRecallError,
    UnsupportedGameError,
    build_sequence_form,
)

# game -> (information sets, sequences per player, terminal histories).
# Measured against open-spiel 2.0.1 on 2026-07-26; the Kuhn rows match the
# research plan §1 exactly (12 / 48 information sets, 30 / 312 terminals).
EXPECTED_SHAPE = {
    "kuhn": (12, 13, 30),
    "kuhn3": (48, 33, 312),
    "leduc": (936, 1093, 5520),
}


@pytest.fixture(scope="module")
def kuhn_sf():
    return build_sequence_form(load_game("kuhn"))


@pytest.mark.parametrize("name", sorted(EXPECTED_SHAPE))
def test_tree_shape_matches_the_research_plan(name):
    infosets, sequences, terminals = EXPECTED_SHAPE[name]
    sf = build_sequence_form(load_game(name))
    assert sf.num_infosets == infosets
    assert sf.num_sequences == (sequences,) * sf.num_players
    assert sf.num_terminals == terminals


def test_parameter_count_reconciliation():
    """The plan's "12-dim / 48-dim" is minimal; ours is the redundant softmax.

    The research plan counts one degree of freedom per binary information set.
    This module carries ``max_actions`` logits per information set instead, so
    the slot count is double. Both induce the same behaviour strategy; they do
    NOT induce the same reference measure, which is why the difference is
    written down rather than left for someone to rediscover while debugging a
    coverage threshold.
    """
    kuhn = build_sequence_form(load_game("kuhn"))
    kuhn3 = build_sequence_form(load_game("kuhn3"))
    assert (kuhn.num_infosets, kuhn3.num_infosets) == (12, 48)  # the plan's numbers
    assert kuhn.legal_mask.numel() == 24
    assert kuhn3.legal_mask.numel() == 96
    # Kuhn is binary everywhere, so every information set has exactly 2 actions.
    assert bool((kuhn.legal_mask.sum(dim=1) == 2).all())
    assert bool((kuhn3.legal_mask.sum(dim=1) == 2).all())


@pytest.mark.parametrize("name", sorted(EXPECTED_SHAPE))
def test_terminal_reach_mass_is_one_under_any_valid_profile(name):
    """Chance x every player's realization probability must sum to 1.

    Note this is NOT ``terminal_chance.sum()`` — that counts each deal once per
    terminal path below it (5 on 2p Kuhn, 13 on 3p Kuhn) and is meaningless on
    its own. The distribution only closes once the players' reach probabilities
    are folded in, which is precisely the invariant a mis-indexed sequence
    lookup would break.
    """
    sf = build_sequence_form(load_game(name))
    assert bool((sf.terminal_chance > 0).all())
    logits = torch.zeros(sf.num_infosets, sf.max_actions, dtype=torch.float64)
    plans = realization_plans(sf, behavior_from_logits(sf, logits))
    reach = sf.terminal_chance.clone()
    for player in range(sf.num_players):
        reach = reach * plans[player][sf.terminal_sequence[:, player]]
    assert float(reach.sum()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("name", sorted(EXPECTED_SHAPE))
def test_levels_partition_every_information_set(name):
    """``level_rows`` must cover each row exactly once, in ascending level."""
    sf = build_sequence_form(load_game(name))
    seen = torch.cat(sf.level_rows)
    assert sorted(seen.tolist()) == list(range(sf.num_infosets))
    for level, rows in enumerate(sf.level_rows):
        assert bool((sf.infoset_level[rows] == level).all())


@pytest.mark.parametrize("name", sorted(EXPECTED_SHAPE))
def test_sequence_indices_are_a_permutation_per_player(name):
    """Every non-empty sequence is assigned to exactly one (infoset, action)."""
    sf = build_sequence_form(load_game(name))
    for player in range(sf.num_players):
        rows = sf.rows_of(player)
        assigned = sf.sequence_of[rows]
        legal = assigned[assigned != NO_SEQUENCE].tolist()
        assert sorted(legal) == list(range(1, sf.num_sequences[player]))
        # And the empty sequence is never handed out as a child.
        assert EMPTY_SEQUENCE not in legal


@pytest.mark.parametrize("name", sorted(EXPECTED_SHAPE))
def test_parent_sequence_belongs_to_the_owning_player(name):
    """A row's parent must index into its OWN player's sequence vector."""
    sf = build_sequence_form(load_game(name))
    for player in range(sf.num_players):
        rows = sf.rows_of(player)
        parents = sf.parent_sequence[rows]
        assert bool((parents >= 0).all())
        assert bool((parents < sf.num_sequences[player]).all())
        # Level-0 rows hang off the empty sequence, by definition.
        level_zero = rows[sf.infoset_level[rows] == 0]
        assert bool((sf.parent_sequence[level_zero] == EMPTY_SEQUENCE).all())


def test_terminal_sequences_are_in_range():
    sf = build_sequence_form(load_game("kuhn3"))
    for player in range(sf.num_players):
        column = sf.terminal_sequence[:, player]
        assert bool((column >= 0).all())
        assert bool((column < sf.num_sequences[player]).all())


def test_legal_mask_agrees_with_the_sequence_table():
    sf = build_sequence_form(load_game("leduc"))
    assert bool(((sf.sequence_of != NO_SEQUENCE) == sf.legal_mask).all())


def test_row_of_key_round_trips(kuhn_sf):
    for row, key in enumerate(kuhn_sf.infoset_keys):
        player = int(kuhn_sf.infoset_player[row])
        assert kuhn_sf.row_of_key(player, key) == row


def test_row_of_key_raises_on_unknown(kuhn_sf):
    with pytest.raises(KeyError):
        kuhn_sf.row_of_key(0, "not-an-info-state")


@pytest.mark.parametrize("name", ["brps", "goofspiel5_ii", "oshi_zumo"])
def test_simultaneous_games_are_refused_not_converted(name):
    """AGENTS.md §11: no silent fallback. A simultaneous game has no turn-based
    sequence form, and quietly calling convert_to_turn_based would change the
    game being solved."""
    with pytest.raises(UnsupportedGameError, match="simultaneous"):
        build_sequence_form(load_game(name))


def test_imperfect_recall_is_refused():
    """Tic-tac-toe has no information-state string, so the loader falls back to
    the observation string — and two different move orders reach the same board.
    That is exactly the case where a sequence form cannot be built, and picking
    one of the two parent sequences would produce a silently wrong index.
    """
    with pytest.raises(PerfectRecallError, match="perfect recall"):
        build_sequence_form(load_game("ttt"))


@pytest.mark.slow
def test_liars_dice_builds():
    """The largest game the pACH programme plans to touch before Mahjong."""
    sf = build_sequence_form(load_game("liars_dice1"))
    assert sf.num_infosets == 24576
    assert sf.num_terminals == 147420
