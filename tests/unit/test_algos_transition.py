"""Unit tests for producer tagging + identity routing on Batch (AGENTS.md §5).

The league's seat shuffle makes physical seat numbers meaningless for
routing; every transition carries the policy that produced it, and learners
train on their own samples via :meth:`Batch.for_producer`.
"""

from __future__ import annotations

import pytest

from mjai.agents.tabular import TabularPolicy
from mjai.algos.transition import Transition, make_batch
from mjai.utils import gpu_assert


@pytest.fixture(autouse=True)
def _cpu_mode():
    gpu_assert.reset_for_tests()
    gpu_assert.require_cpu()
    yield
    gpu_assert.reset_for_tests()


def _t(i: int, player: int, producer=None) -> Transition:
    return Transition(
        obs=[float(i)],
        legal_actions=[0, 1],
        action=i % 2,
        logprob=-0.7,
        value=0.0,
        reward=0.0,
        return_=float(i),
        advantage=0.0,
        player=player,
        producer=producer,
    )


def test_make_batch_dedupes_producers_by_identity():
    a = TabularPolicy(num_actions=2, seed=0)
    b = TabularPolicy(num_actions=2, seed=1)
    batch = make_batch([_t(0, 0, a), _t(1, 1, b), _t(2, 0, a)], num_actions=2)
    assert batch.producers == (a, b)
    assert batch.producer_idx is not None
    assert list(batch.producer_idx) == [0, 1, 0]


def test_for_producer_selects_only_that_policys_rows():
    a = TabularPolicy(num_actions=2, seed=0)
    b = TabularPolicy(num_actions=2, seed=1)
    batch = make_batch([_t(0, 0, a), _t(1, 1, b), _t(2, 1, a)], num_actions=2)
    sub_a = batch.for_producer(a)
    assert sub_a.size == 2
    assert list(sub_a.actions) == [0, 0]  # rows 0 and 2 (i % 2 == 0)
    # A's transitions here came from BOTH seats (rows 0 and 1) — the whole
    # point of identity routing under seat shuffle.
    assert sorted(int(p) for p in sub_a.players) == [0, 1]
    sub_b = batch.for_producer(b)
    assert sub_b.size == 1
    assert int(sub_b.players[0]) == 1


def test_for_producer_unknown_policy_returns_empty():
    a = TabularPolicy(num_actions=2, seed=0)
    stranger = TabularPolicy(num_actions=2, seed=9)
    batch = make_batch([_t(0, 0, a)], num_actions=2)
    sub = batch.for_producer(stranger)
    assert sub.size == 0


def test_for_producer_on_untagged_batch_fails_loudly():
    """No silent fallback to seat-0 filtering (AGENTS.md §11)."""
    batch = make_batch([_t(0, 0), _t(1, 1)], num_actions=2)
    assert batch.producers == ()
    with pytest.raises(RuntimeError, match="producer tags"):
        batch.for_producer(TabularPolicy(num_actions=2, seed=0))


def test_make_batch_rejects_mixed_tagged_and_untagged():
    a = TabularPolicy(num_actions=2, seed=0)
    with pytest.raises(ValueError, match="mixed producer tags"):
        make_batch([_t(0, 0, a), _t(1, 1)], num_actions=2)


def test_for_player_preserves_producer_tags_for_chained_routing():
    a = TabularPolicy(num_actions=2, seed=0)
    b = TabularPolicy(num_actions=2, seed=1)
    batch = make_batch([_t(0, 0, a), _t(1, 1, b), _t(2, 1, a)], num_actions=2)
    seat1 = batch.for_player(1)
    assert seat1.producers == (a, b)
    sub = seat1.for_producer(a)  # chained: seat filter THEN identity filter
    assert sub.size == 1
    assert int(sub.players[0]) == 1


def test_empty_batch_has_no_producer_tags():
    batch = make_batch([], num_actions=2)
    assert batch.producers == ()
    assert batch.producer_idx is None
