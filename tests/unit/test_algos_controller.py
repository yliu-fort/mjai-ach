"""Unit tests for the controller/Trainer composition (AGENTS.md §5)."""

from __future__ import annotations

import pytest

from mjai.agents.base import Policy
from mjai.agents.tabular import TabularPolicy
from mjai.algos.controller import (
    MirrorSelfPlay,
    RolloutRunnerProtocol,
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

    def run_episode(self, learner: Policy, opponent: Policy) -> Batch:
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
