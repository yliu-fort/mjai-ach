"""Unit tests for OpponentSampler: PFSP + role-based mixing (AGENTS.md §5)."""

from __future__ import annotations

import random

import pytest

from mjai.agents.tabular import TabularPolicy
from mjai.league.checkpoint_store import PoolMember, Role
from mjai.league.opponent_sampler import LeagueMix, OpponentSampler


def _member(role: Role, member_id: int, win_rate_vs_learner: float | None = None) -> PoolMember:
    m = PoolMember(
        policy=TabularPolicy(num_actions=3, seed=member_id), role=role, member_id=member_id
    )
    if win_rate_vs_learner is not None:
        # Sampler looks up candidate.win_rates[learner_id]; set against id 999.
        m.win_rates[999] = win_rate_vs_learner
    return m


def test_main_exploiter_always_plays_current_main():
    sampler = OpponentSampler(rng=random.Random(0))
    main = TabularPolicy(num_actions=3, seed=0)
    pool = [_member(Role.MAIN, 1), _member(Role.MAIN_EXPLOITER, 2)]
    for _ in range(10):
        opp = sampler.sample(
            pool, learner_role=Role.MAIN_EXPLOITER, current_main=main, learner_member_id=999
        )
        assert opp is main


def test_mix_weights_must_sum_to_one():
    with pytest.raises(ValueError, match=r"1\.0"):
        LeagueMix(current_main_weight=0.5, history_weight=0.3, exploiter_weight=0.1)


def test_empty_pool_returns_current_main():
    sampler = OpponentSampler(rng=random.Random(0))
    main = TabularPolicy(num_actions=3, seed=0)
    opp = sampler.sample([], learner_role=Role.MAIN, current_main=main, learner_member_id=None)
    assert opp is main


def test_empty_bucket_falls_back_to_current_main():
    """Drawing the 'history' bucket when no history exists => current main."""
    sampler = OpponentSampler(
        LeagueMix(current_main_weight=0.0, history_weight=1.0, exploiter_weight=0.0),
        rng=random.Random(0),
    )
    main = TabularPolicy(num_actions=3, seed=0)
    pool = [_member(Role.MAIN_EXPLOITER, 1)]  # no MAIN history
    opp = sampler.sample(pool, learner_role=Role.MAIN, current_main=main, learner_member_id=999)
    assert opp is main


def test_pfsp_over_samples_competitive_opponents():
    """Opponents near 50% win-rate get more samples than one-sided ones."""
    sampler = OpponentSampler(
        LeagueMix(current_main_weight=0.0, history_weight=1.0, exploiter_weight=0.0),
        rng=random.Random(0),
    )
    # Three history members: one dominating, one even, one dominated.
    pool = [
        _member(Role.MAIN, 1, win_rate_vs_learner=0.99),  # near-certain win
        _member(Role.MAIN, 2, win_rate_vs_learner=0.50),  # competitive
        _member(Role.MAIN, 3, win_rate_vs_learner=0.01),  # near-certain loss
    ]
    counts = {1: 0, 2: 0, 3: 0}
    for _ in range(600):
        opp = sampler.sample(pool, learner_role=Role.MAIN, current_main=None, learner_member_id=999)
        # opp is a Policy; find which member it came from by policy identity.
        for m in pool:
            if m.policy is opp:
                counts[m.member_id] += 1
                break
    # The competitive one (id 2) should be sampled most often.
    assert counts[2] > counts[1]
    assert counts[2] > counts[3]


def test_learner_excluded_from_own_pool():
    """The sampler won't return the learner's own pool entry."""
    sampler = OpponentSampler(
        LeagueMix(current_main_weight=0.0, history_weight=1.0, exploiter_weight=0.0),
        rng=random.Random(0),
    )
    me = _member(Role.MAIN, 42)
    others = [_member(Role.MAIN, 1), _member(Role.MAIN, 2)]
    pool = [me, *others]
    main_policy = TabularPolicy(num_actions=3, seed=0)
    for _ in range(50):
        opp = sampler.sample(
            pool, learner_role=Role.MAIN, current_main=main_policy, learner_member_id=42
        )
        if opp is not main_policy:
            assert opp is not me.policy  # never pick ourselves


def test_default_mix_is_50_30_20():
    m = LeagueMix()
    assert m.current_main_weight == 0.5
    assert m.history_weight == 0.3
    assert m.exploiter_weight == 0.2
