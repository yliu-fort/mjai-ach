"""Unit tests for CheckpointStore: pool + role-aware quotas (AGENTS.md §5).

The pool is divided into a main-history quota (``capacity - 2``) plus one
reserved slot per exploiter role: exploiter adds REPLACE the same-role member
and never evict main history; main adds FIFO-evict only mains.
"""

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


def test_history_quota_fifo_evicts_oldest_main():
    """Mains beyond the capacity-2 history quota evict the oldest main."""
    s = CheckpointStore(capacity=4)  # history quota = 2
    m1 = s.add(_policy(), Role.MAIN)
    s.add(_policy(), Role.MAIN)
    assert len(s) == 2
    # A 3rd main overflows the quota: m1 (oldest) goes, exploiters untouched.
    s.add(_policy(), Role.MAIN)
    assert len(s.by_role(Role.MAIN)) == 2
    assert m1 not in s.members


def test_exploiters_survive_history_quota_eviction():
    """Main-add eviction never touches exploiters."""
    s = CheckpointStore(capacity=4)  # history quota = 2
    s.add(_policy(), Role.MAIN)
    s.add(_policy(), Role.MAIN)
    e1 = s.add(_policy(), Role.MAIN_EXPLOITER)
    e2 = s.add(_policy(), Role.LEAGUE_EXPLOITER)
    assert len(s) == 4
    for _ in range(5):  # hammer the history quota
        s.add(_policy(), Role.MAIN)
    assert len(s) == 4
    assert e1 in s.members
    assert e2 in s.members
    assert len(s.by_role(Role.MAIN)) == 2


def test_exploiter_add_replaces_same_role_member():
    """At most one snapshot per exploiter role: a promotion drops the old one."""
    s = CheckpointStore(capacity=4)
    old = s.add(_policy(seed=1), Role.MAIN_EXPLOITER)
    new = s.add(_policy(seed=2), Role.MAIN_EXPLOITER)
    assert old not in s.members
    assert new in s.members
    assert len(s.by_role(Role.MAIN_EXPLOITER)) == 1
    # ...independently per exploiter role.
    le = s.add(_policy(seed=3), Role.LEAGUE_EXPLOITER)
    assert new in s.members
    assert le in s.members


def test_exploiter_add_never_evicts_main_history():
    """A promotion at a full pool replaces its own role's slot, not a main."""
    s = CheckpointStore(capacity=4)  # quota: 2 mains + 2 exploiter slots
    mains = [s.add(_policy(seed=10 + i), Role.MAIN) for i in range(2)]
    s.add(_policy(), Role.MAIN_EXPLOITER)
    s.add(_policy(), Role.LEAGUE_EXPLOITER)
    assert len(s) == 4  # pool full
    # Re-promoting the league-exploiter must leave every main in place.
    s.add(_policy(seed=99), Role.LEAGUE_EXPLOITER)
    assert len(s) == 4
    assert all(m in s.members for m in mains)
    assert len(s.by_role(Role.LEAGUE_EXPLOITER)) == 1


def test_first_promotion_in_a_full_history_pool_stays_within_capacity():
    """History quota is capacity-2, so the first exploiter add always fits."""
    s = CheckpointStore(capacity=3)  # minimal: quota 1 + 2 exploiter slots
    s.add(_policy(), Role.MAIN)
    s.add(_policy(), Role.MAIN)  # evicts the first: history quota = 1
    assert len(s.by_role(Role.MAIN)) == 1
    s.add(_policy(), Role.MAIN_EXPLOITER)
    s.add(_policy(), Role.LEAGUE_EXPLOITER)
    assert len(s) == 3  # == capacity, never above


def test_capacity_validation():
    with pytest.raises(ValueError, match="capacity"):
        CheckpointStore(capacity=0)
    with pytest.raises(ValueError, match="capacity"):
        CheckpointStore(capacity=2)  # history quota would be 0


def test_removal_scrubs_win_rate_rows_pointing_at_it():
    """PFSP caches keyed by a removed member id are cleaned from all members."""
    s = CheckpointStore()
    keep = s.add(_policy(seed=1), Role.MAIN)
    gone = s.add(_policy(seed=2), Role.MAIN_EXPLOITER)
    s.update_win_rate(keep.member_id, gone.member_id, 0.7)
    s.add(_policy(seed=3), Role.MAIN_EXPLOITER)  # replaces `gone`
    assert gone.member_id not in keep.win_rates


def test_update_win_rate_records_value():
    s = CheckpointStore()
    m = s.add(_policy(), Role.MAIN)
    other = s.add(_policy(), Role.MAIN_EXPLOITER)
    s.update_win_rate(m.member_id, other.member_id, 0.7)
    assert m.win_rates[other.member_id] == 0.7


def test_update_win_rate_ignores_evicted_members():
    s = CheckpointStore(capacity=3)  # history quota = 1
    m = s.add(_policy(), Role.MAIN)
    s.add(_policy(), Role.MAIN)  # evicts m
    # Updating m should not raise; it's just silently ignored.
    s.update_win_rate(m.member_id, 999, 0.5)


def test_clear_empties_store():
    s = CheckpointStore()
    s.add(_policy(), Role.MAIN)
    s.clear()
    assert len(s) == 0


def test_members_property_returns_copy():
    """Callers can't mutate the internal list via the property."""
    s = CheckpointStore()
    s.add(_policy(), Role.MAIN)
    snapshot = s.members
    snapshot.clear()
    assert len(s) == 1  # internal list untouched
