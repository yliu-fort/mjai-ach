"""Unit tests for the controller/Trainer composition (AGENTS.md §5)."""

from __future__ import annotations

import pytest

from mjai.agents.base import Policy
from mjai.agents.tabular import TabularPolicy
from mjai.algos.controller import (
    Collected,
    LearnerBatch,
    MirrorSelfPlay,
    RolloutRunnerProtocol,
    SelfPlayController,
    Trainer,
)
from mjai.algos.tabular_updates import TabularPPOUpdate
from mjai.algos.transition import Batch, Transition, make_batch


def _toy_batch(n: int = 4) -> Batch:
    ts = [
        Transition(
            obs=[float(i), 0.0],
            legal_actions=[0, 1],
            action=i % 2,
            logprob=-0.7,
            value=0.0,
            reward=0.0,
            return_=float(i),
            advantage=float(i) - 1.5,
        )
        for i in range(n)
    ]
    return make_batch(ts, num_actions=2)


class _ScriptedRunner:
    """A fake RolloutRunner: returns a fixed batch, records who played."""

    def __init__(self, batch: Batch | None = None) -> None:
        self.calls: list[tuple[Policy, Policy]] = []
        self._batch = batch or _toy_batch()

    def run_episode(
        self, learner: Policy, opponent: Policy, *, keep: tuple[Policy, ...] | None = None
    ) -> Batch:
        self.calls.append((learner, opponent))
        return self._batch


def test_mirror_controller_passes_learner_as_both_seats():
    runner = _ScriptedRunner()
    ctrl = MirrorSelfPlay(runner)
    p = TabularPolicy(num_actions=2, seed=0)
    ctrl.set_learner(p)
    ctrl.collect()
    assert len(runner.calls) == 1
    learner, opponent = runner.calls[0]
    assert learner is p and opponent is p  # mirror: same policy both seats


def test_mirror_name():
    assert MirrorSelfPlay(_ScriptedRunner()).name == "mirror"


def test_mirror_collect_before_set_learner_raises():
    ctrl = MirrorSelfPlay(_ScriptedRunner())
    with pytest.raises(RuntimeError, match="set_learner"):
        ctrl.collect()


def test_trainer_step_returns_round_summary():
    p = TabularPolicy(num_actions=2, seed=0)
    rule = TabularPPOUpdate(p)
    ctrl = MirrorSelfPlay(_ScriptedRunner())
    trainer = Trainer(policy=p, update_rule=rule, controller=ctrl)
    round_ = trainer.step()
    assert round_.batch_size == 4
    assert "policy_loss" in round_.stats_keys
    assert trainer.last_stats is not None


def test_trainer_step_actually_updates_policy():
    p = TabularPolicy(num_actions=2, seed=0)
    # Get a row reference for the only obs in the toy batch ([0.0, 0.0]).
    obs = [0.0, 0.0]
    before = p.get_logits(obs)[0]
    rule = TabularPPOUpdate(p, clip_eps=0.2)
    ctrl = MirrorSelfPlay(_ScriptedRunner())
    trainer = Trainer(policy=p, update_rule=rule, controller=ctrl)
    trainer.step()
    after = p.get_logits(obs)[0]
    # PPO moved the chosen-action logit by lr*clip(advantage).
    assert abs(after - before) > 0


def test_controller_set_learner_called_each_trainer_step():
    """The Trainer refreshes the controller's learner before each collect."""
    runner = _ScriptedRunner()
    ctrl = MirrorSelfPlay(runner)
    p = TabularPolicy(num_actions=2, seed=0)
    trainer = Trainer(policy=p, update_rule=TabularPPOUpdate(p), controller=ctrl)
    trainer.step()
    trainer.step()
    assert len(runner.calls) == 2


def test_rollout_runner_protocol_is_just_a_protocol():
    # The Protocol should accept any object with run_episode, not require inheritance.
    runner: RolloutRunnerProtocol = _ScriptedRunner()  # type-checks structurally
    assert hasattr(runner, "run_episode")


def test_mirror_round_prices_every_simulated_step():
    """Mirror keeps both seats, so retained samples == simulated cost."""
    p = TabularPolicy(num_actions=2, seed=0)
    ctrl = MirrorSelfPlay(_ScriptedRunner())
    round_ = Trainer(policy=p, update_rule=TabularPPOUpdate(p), controller=ctrl).step()
    assert round_.batch_size == round_.env_steps == 4


class _RotatingController(SelfPlayController):
    """Collects for a fixed cycle of learners, like the league's role rotation."""

    def __init__(self, learners: list[Policy]) -> None:
        self._learners = learners
        self._i = 0

    def set_learner(self, policy: Policy) -> None:
        self._learners[0] = policy

    def collect(self) -> Collected:
        learner = self._learners[self._i % len(self._learners)]
        self._i += 1
        label = "main" if learner is self._learners[0] else "other"
        part = LearnerBatch(batch=_toy_batch(), learner=learner, label=label)
        return Collected(parts=(part,), sampled_steps=8)

    @property
    def name(self) -> str:
        return "rotating"

    def learners(self) -> tuple[Policy, ...]:
        return tuple(self._learners)


def test_trainer_updates_the_policy_that_collected_the_batch():
    """A rotating controller must not train one learner on another's samples."""
    obs = [0.0, 0.0]
    main = TabularPolicy(num_actions=2, seed=0)
    other = TabularPolicy(num_actions=2, seed=0)
    ctrl = _RotatingController([main, other])
    trainer = Trainer(
        policy=main,
        update_rule=TabularPPOUpdate(main),
        controller=ctrl,
        extra_rules=[TabularPPOUpdate(other)],
    )
    main_before, other_before = main.get_logits(obs)[0], other.get_logits(obs)[0]
    trainer.step()  # round 0 -> main collects
    assert main.get_logits(obs)[0] != main_before
    assert other.get_logits(obs)[0] == other_before  # untouched by main's batch
    main_after = main.get_logits(obs)[0]
    trainer.step()  # round 1 -> other collects
    assert other.get_logits(obs)[0] != other_before
    assert main.get_logits(obs)[0] == main_after  # untouched by other's batch


def test_trainer_refuses_a_batch_from_a_learner_it_has_no_rule_for():
    """No silent fallback to the main rule (AGENTS.md §11)."""
    main = TabularPolicy(num_actions=2, seed=0)
    stranger = TabularPolicy(num_actions=2, seed=1)
    ctrl = _RotatingController([main, stranger])
    trainer = Trainer(policy=main, update_rule=TabularPPOUpdate(main), controller=ctrl)
    trainer.step()  # main's own round is fine
    with pytest.raises(RuntimeError, match="no\n?\\s*UpdateRule|UpdateRule"):
        trainer.step()


def test_round_reports_simulated_cost_separately_from_batch_size():
    main = TabularPolicy(num_actions=2, seed=0)
    ctrl = _RotatingController([main])
    round_ = Trainer(policy=main, update_rule=TabularPPOUpdate(main), controller=ctrl).step()
    assert round_.batch_size == 4  # retained
    assert round_.env_steps == 8  # simulated, including the discarded seat


class _DualPartController(SelfPlayController):
    """One round producing a part for EACH of two learners (league-style)."""

    def __init__(self, main: Policy, other: Policy) -> None:
        self._main, self._other = main, other

    def set_learner(self, policy: Policy) -> None:
        pass

    def collect(self) -> Collected:
        return Collected(
            parts=(
                LearnerBatch(batch=_toy_batch(2), learner=self._main, label="main"),
                LearnerBatch(batch=_toy_batch(3), learner=self._other, label="other"),
            ),
            sampled_steps=10,
        )

    @property
    def name(self) -> str:
        return "dual"

    def learners(self) -> tuple[Policy, ...]:
        return (self._main, self._other)


def test_trainer_dispatches_every_part_to_its_own_rule():
    """A multi-part round updates every kept learner on ITS OWN samples."""
    obs = [0.0, 0.0]
    main = TabularPolicy(num_actions=2, seed=0)
    other = TabularPolicy(num_actions=2, seed=0)
    trainer = Trainer(
        policy=main,
        update_rule=TabularPPOUpdate(main),
        controller=_DualPartController(main, other),
        extra_rules=[TabularPPOUpdate(other)],
    )
    main_before, other_before = main.get_logits(obs)[0], other.get_logits(obs)[0]
    round_ = trainer.step()
    assert main.get_logits(obs)[0] != main_before  # main's part updated main
    assert other.get_logits(obs)[0] != other_before  # other's part updated other
    assert round_.batch_size == 5  # 2 + 3 across parts
    assert round_.env_steps == 10
    assert trainer.last_stats is not None  # the main line's stats
    assert set(trainer.last_stats_by_label) == {"main", "other"}


def test_trainer_fails_loudly_when_every_part_is_empty():
    """Routing that drops everything is a controller bug, not a slow round (§11)."""
    from mjai.algos.transition import make_batch

    class _EmptyController(_DualPartController):
        def collect(self) -> Collected:
            empty = make_batch([], num_actions=2)
            return Collected(
                parts=(LearnerBatch(batch=empty, learner=self._main, label="main"),),
                sampled_steps=0,
            )

    main = TabularPolicy(num_actions=2, seed=0)
    other = TabularPolicy(num_actions=2, seed=0)
    trainer = Trainer(
        policy=main,
        update_rule=TabularPPOUpdate(main),
        controller=_EmptyController(main, other),
        extra_rules=[TabularPPOUpdate(other)],
    )
    with pytest.raises(RuntimeError, match="no trainable transitions"):
        trainer.step()
