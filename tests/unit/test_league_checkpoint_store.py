"""Unit tests for CheckpointStore: pool + role-aware eviction (AGENTS.md §5)."""

from __future__ import annotations

import pytest

from mjai.agents.tabular import TabularPolicy
from mjai.league.checkpoint_store import CheckpointStore, PoolMember, Role


def _policy(seed: int = 0) -> TabularPolicy:
    return TabularPolicy(num_actions=3, seed=seed)


def test_empty_store_has_zero_size():
    s = CheckpointStore()
    assert len(s) == 0
    assert s.snapshot_summary() == {"main": 0, "main_exploiter": 0, "league_exploiter": 0}


def test_add_returns_member_with_id():
    s = CheckpointStore()
    p = _policy()
    m = s.add(p, Role.MAIN)
    assert isinstance(m, PoolMember)
    assert m.role == Role.MAIN
    assert m.member_id == 0
    assert len(s) == 1


def test_by_role_filters_correctly():
    s = CheckpointStore()
    s.add(_policy(), Role.MAIN)
    s.add(_policy(), Role.MAIN_EXPLOITER)
    s.add(_policy(), Role.LEAGUE_EXPLOITER)
    assert len(s.by_role(Role.MAIN)) == 1
    assert len(s.by_role(Role.MAIN_EXPLOITER)) == 1
    assert len(s.by_role(Role.LEAGUE_EXPLOITER)) == 1
    assert len(s.exploiters()) == 2


def test_main_history_sorted_oldest_first():
    s = CheckpointStore()
    m1 = s.add(_policy(), Role.MAIN)
    m2 = s.add(_policy(), Role.MAIN)
    m3 = s.add(_policy(), Role.MAIN)
    history = s.main_history()
    assert history == [m1, m2, m3]


def test_fifo_eviction_drops_oldest_main_first():
    s = CheckpointStore(capacity=3)
    m1 = s.add(_policy(), Role.MAIN)
    s.add(_policy(), Role.MAIN)
    s.add(_policy(), Role.MAIN)
    assert len(s) == 3
    # Adding a 4th main should evict m1 (oldest).
    s.add(_policy(), Role.MAIN)
    assert len(s) == 3
    assert m1 not in s.members


def test_exploiters_survive_when_main_is_evicted_first():
    """Even at capacity, exploiters aren't evicted while any main remains."""
    s = CheckpointStore(capacity=4)
    s.add(_policy(), Role.MAIN)
    s.add(_policy(), Role.MAIN)
    e1 = s.add(_policy(), Role.MAIN_EXPLOITER)
    e2 = s.add(_policy(), Role.LEAGUE_EXPLOITER)
    # Adding a 5th (another main) should evict a main, not an exploiter.
    s.add(_policy(), Role.MAIN)
    assert len(s) == 4
    assert e1 in s.members
    assert e2 in s.members
    assert len(s.by_role(Role.MAIN)) == 2  # evicted down from 3 to 2... wait


def test_capacity_validation():
    with pytest.raises(ValueError, match="capacity"):
        CheckpointStore(capacity=0)


def test_update_win_rate_records_value():
    s = CheckpointStore()
    m = s.add(_policy(), Role.MAIN)
    other = s.add(_policy(), Role.MAIN_EXPLOITER)
    s.update_win_rate(m.member_id, other.member_id, 0.7)
    assert m.win_rates[other.member_id] == 0.7


def test_update_win_rate_ignores_evicted_members():
    s = CheckpointStore(capacity=1)
    m = s.add(_policy(), Role.MAIN)
    s.add(_policy(), Role.MAIN)  # evicts m
    # Updating m should not raise; it's just silently ignored.
    s.update_win_rate(m.member_id, 999, 0.5)


def test_clear_empties_store():
    s = CheckpointStore()
    s.add(_policy(), Role.MAIN)
    s.clear()
    assert len(s) == 0


def test_exploiter_score_handles_unmeasured():
    s = CheckpointStore()
    m = s.add(_policy(), Role.MAIN_EXPLOITER)
    # No win_rates recorded => score is -inf (lowest priority for keeping).
    assert s._exploiter_score(m) == float("-inf")
    s.update_win_rate(m.member_id, 0, 0.6)
    s.update_win_rate(m.member_id, 1, 0.4)
    assert s._exploiter_score(m) == 0.5  # mean of {0.6, 0.4}


def test_members_property_returns_copy():
    """Callers can't mutate the internal list via the property."""
    s = CheckpointStore()
    s.add(_policy(), Role.MAIN)
    snapshot = s.members
    snapshot.clear()
    assert len(s) == 1  # internal list untouched
