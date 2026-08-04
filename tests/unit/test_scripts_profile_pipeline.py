"""Unit tests for the pipeline profiler's batch routing (AGENTS.md §5, §8).

The profiler duplicates ``Trainer.step``'s dispatch because it needs the
rollout/update split that ``step`` does not expose. That duplication is what
rotted: it kept calling ``update_rule.step(collected)`` and reading
``collected.size`` after ``Collected`` grew per-learner parts. These tests pin
the contract that broke — routing by learner identity, skipping empty parts,
and pricing a round by ``sampled_steps`` rather than by summed part sizes.

CPU-only and deterministic (no torch, no game engine): the fakes below are the
whole fixture.
"""

from __future__ import annotations

import pytest

from mjai.algos.controller import Collected, LearnerBatch
from mjai.scripts.profile_pipeline import _apply_updates, _main_learner_batch


class _FakeBatch:
    def __init__(self, size: int) -> None:
        self.size = size


class _FakePolicy:
    def __init__(self, name: str) -> None:
        self.name = name


class _RecordingRule:
    """Stands in for an UpdateRule: records the batches it was stepped with."""

    def __init__(self, policy: _FakePolicy) -> None:
        self.policy = policy
        self.seen: list[_FakeBatch] = []

    def step(self, batch: _FakeBatch) -> None:
        self.seen.append(batch)


def _collected(*pairs: tuple[_FakePolicy, int], sampled_steps: int) -> Collected:
    parts = tuple(
        LearnerBatch(batch=_FakeBatch(size), learner=policy, label=policy.name)
        for policy, size in pairs
    )
    return Collected(parts=parts, sampled_steps=sampled_steps)


# --------------------------------------------------------------------------- #
# _apply_updates — the routing that the stale `update_rule.step(collected)` broke
# --------------------------------------------------------------------------- #


def test_each_part_updates_its_own_learner():
    main, exploiter = _FakePolicy("main"), _FakePolicy("main_exploiter")
    rules = {id(main): _RecordingRule(main), id(exploiter): _RecordingRule(exploiter)}
    collected = _collected((main, 4), (exploiter, 6), sampled_steps=20)

    consumed = _apply_updates(collected, rules)

    assert consumed == 10  # 4 + 6, summed over parts
    assert [b.size for b in rules[id(main)].seen] == [4]
    assert [b.size for b in rules[id(exploiter)].seen] == [6]


def test_empty_parts_are_skipped_not_stepped():
    """A kept learner that produced nothing must not take a gradient step."""
    main, idle = _FakePolicy("main"), _FakePolicy("league_exploiter")
    rules = {id(main): _RecordingRule(main), id(idle): _RecordingRule(idle)}

    consumed = _apply_updates(_collected((main, 3), (idle, 0), sampled_steps=9), rules)

    assert consumed == 3
    assert rules[id(idle)].seen == []


def test_unroutable_part_raises_instead_of_being_skipped():
    """No silent fallback (AGENTS.md §11): a missing rule is a wiring bug."""
    main, stranger = _FakePolicy("main"), _FakePolicy("orphan")
    rules = {id(main): _RecordingRule(main)}

    with pytest.raises(RuntimeError, match="no update rule"):
        _apply_updates(_collected((main, 2), (stranger, 2), sampled_steps=8), rules)


def test_two_learners_sharing_a_type_route_by_identity_not_equality():
    """Identity dispatch: two distinct policies must not collapse into one rule."""
    a, b = _FakePolicy("main"), _FakePolicy("main")  # same name, different objects
    rules = {id(a): _RecordingRule(a), id(b): _RecordingRule(b)}

    _apply_updates(_collected((a, 5), (b, 7), sampled_steps=12), rules)

    assert [x.size for x in rules[id(a)].seen] == [5]
    assert [x.size for x in rules[id(b)].seen] == [7]


def test_sampled_steps_is_not_the_summed_part_sizes():
    """The league drops frozen opponents' transitions, so the two diverge.

    Pricing a round by consumed samples would understate the simulation cost —
    exactly the confusion the old ``samples += batch.size`` line encoded.
    """
    main = _FakePolicy("main")
    rules = {id(main): _RecordingRule(main)}
    collected = _collected((main, 30), sampled_steps=100)

    consumed = _apply_updates(collected, rules)

    assert consumed == 30
    assert collected.sampled_steps == 100


# --------------------------------------------------------------------------- #
# _main_learner_batch — --ops profiles the Trainer's own update rule
# --------------------------------------------------------------------------- #


def test_main_learner_batch_picks_the_trainer_policys_share():
    main, exploiter = _FakePolicy("main"), _FakePolicy("main_exploiter")
    collected = _collected((exploiter, 9), (main, 4), sampled_steps=13)

    assert _main_learner_batch(collected, main).size == 4


def test_main_learner_batch_ignores_an_empty_main_part():
    main, exploiter = _FakePolicy("main"), _FakePolicy("main_exploiter")
    collected = _collected((main, 0), (exploiter, 9), sampled_steps=9)

    with pytest.raises(RuntimeError, match="no non-empty part"):
        _main_learner_batch(collected, main)


def test_main_learner_batch_refuses_another_learners_part():
    """Profiling the exploiter's batch under the main rule measures a fiction."""
    main, exploiter = _FakePolicy("main"), _FakePolicy("main_exploiter")
    collected = _collected((exploiter, 9), sampled_steps=9)

    with pytest.raises(RuntimeError, match="no non-empty part"):
        _main_learner_batch(collected, main)
