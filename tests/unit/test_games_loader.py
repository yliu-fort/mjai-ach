"""Unit tests for the game loader (AGENTS.md §4, §5).

Verifies all eight canonical games load (D8's seven plus ``kuhn3``, D13), the
GameSpec fields match the values verified directly against open-spiel 2.0.1,
and the observation-encoding auto-selection picks information_state where
available and falls back to observation otherwise.
"""

from __future__ import annotations

import pytest

from mjai.games.loader import (
    GAME_STRINGS,
    GameLoadError,
    _parse_params,
    all_game_names,
    load_game,
    load_game_by_string,
)

# Expected (num_actions, max_length, obs_kind, is_simultaneous, num_players) per
# game, captured from the direct open-spiel probe during Step 2; kuhn3's row was
# probed the same way on 2026-07-26 (D13).
EXPECTED = {
    "brps": (3, 1, "information_state", True, 2),
    "kuhn": (2, 3, "information_state", False, 2),
    "kuhn3": (2, 5, "information_state", False, 3),
    "leduc": (3, 8, "information_state", False, 2),
    "ttt": (9, 9, "observation", False, 2),  # no info_state tensor -> falls back
    "goofspiel5_ii": (5, 5, "information_state", True, 2),
    "liars_dice1": (13, 13, "information_state", False, 2),
    "oshi_zumo": (6, 20, "observation", True, 2),  # no info_state tensor -> falls back
}


def test_all_eight_games_registered():
    assert set(GAME_STRINGS) == set(EXPECTED)
    assert len(all_game_names()) == 8


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_game_loads_with_correct_spec(name):
    spec = load_game(name)
    exp_actions, exp_len, exp_kind, exp_sim, exp_players = EXPECTED[name]
    assert spec.name == name
    assert spec.num_actions == exp_actions
    assert spec.max_game_length == exp_len
    assert spec.obs_kind == exp_kind
    assert spec.is_simultaneous == exp_sim
    assert spec.num_players == exp_players
    # is_zero_sum is True for CONSTANT_SUM too; kuhn3 is constant- not zero-sum.
    assert spec.is_zero_sum
    assert spec.obs_size > 0


def test_kuhn3_tree_size_matches_the_research_plan():
    """312 terminal histories over 24 deals, 48 information sets (D13).

    These are the numbers Generative-ach.md §1 quotes for the Phase-B decision
    gate; if OpenSpiel ever changes its 3p Kuhn parameters underneath us, every
    Step-0 ground-truth artifact silently goes stale, so pin them here.
    """
    spec = load_game("kuhn3")
    terminals = 0
    stack = [spec.new_state()]
    while stack:
        state = stack.pop()
        if state.is_terminal():
            terminals += 1
            continue
        stack.extend(state.child(a) for a in state.legal_actions())
    assert terminals == 312


def test_obs_tensor_size_matches_spec():
    """obs_tensor() on a fresh state has length == obs_size."""
    for name in EXPECTED:
        spec = load_game(name)
        state = spec.new_state()
        vec = spec.obs_tensor(state, player=0)
        assert len(vec) == spec.obs_size, f"{name}: len={len(vec)} != {spec.obs_size}"


def test_unknown_game_raises():
    with pytest.raises(GameLoadError, match="Unknown game"):
        load_game("not_a_game")


def test_unknown_game_lists_known():
    with pytest.raises(GameLoadError, match="kuhn"):
        load_game("nope")


def test_override_params_merge_with_baked_in():
    """Overrides merge with baked-in params (caller wins), not replace them.

    The registered oshi_zumo string is ``coins=5,size=3,horizon=20``. Overriding
    only ``coins=20`` must keep ``horizon=20`` (not fall back to the pyspiel
    default of 1000) while growing the action count.
    """
    small = load_game("oshi_zumo")  # coins=5 baked in
    big = load_game("oshi_zumo", coins=20)
    assert big.num_actions > small.num_actions  # 21 > 6
    assert big.max_game_length == small.max_game_length == 20  # horizon preserved


def test_load_game_by_string_for_unregistered():
    """Ad-hoc games load via load_game_by_string (e.g. non-biased RPS)."""
    spec = load_game_by_string("matrix_rps")
    assert spec.num_actions == 3
    assert spec.is_simultaneous
    assert spec.name == "matrix_rps"


def test_parse_params_types():
    """The internal param parser coerces bool/int/float and leaves strings."""
    p = _parse_params("goofspiel(imp_info=True,num_cards=5,points_order=descending)")
    assert p == {
        "imp_info": True,
        "num_cards": 5,
        "points_order": "descending",
    }
    assert _parse_params("kuhn_poker") == {}
    assert _parse_params("oshi_zumo(coins=5)") == {"coins": 5}


def test_repr_is_compact_and_informative():
    spec = load_game("kuhn")
    r = repr(spec)
    assert "kuhn" in r and "A=2" in r and "obs=information_state" in r


def test_new_state_returns_independent_states():
    spec = load_game("ttt")
    s1 = spec.new_state()
    s2 = spec.new_state()
    s1.apply_action(0)  # X plays top-left on s1 only
    assert s2.legal_actions() == list(range(9))  # s2 is untouched
    assert 0 not in s1.legal_actions()


def test_simultaneous_flag_drives_action_application():
    """Sanity: a simultaneous game's initial state reports SIMULTANEOUS."""
    sim = load_game("brps")
    seq = load_game("kuhn")
    assert sim.new_state().is_simultaneous_node()
    assert not seq.new_state().is_simultaneous_node()
