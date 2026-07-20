"""Unit tests for the tabular UpdateRules (AGENTS.md §5, Step 3)."""

from __future__ import annotations

import math

import pytest

from mjai.agents.tabular import TabularPolicy, _obs_to_key
from mjai.algos.tabular_updates import (
    TabularACHUpdate,
    TabularPPOUpdate,
    _explained_variance,
)
from mjai.algos.transition import Batch, Transition, make_batch
from mjai.algos.update_rule import AlgoConfig

OBS = [1.0, 0.0, 0.0]
NUM_ACTIONS = 3


def _batch(advantages: list[float], returns: list[float] | None = None, actions=None) -> Batch:
    n = len(advantages)
    returns = returns if returns is not None else [0.0] * n
    actions = actions if actions is not None else [0] * n
    ts = [
        Transition(
            obs=OBS,
            legal_actions=list(range(NUM_ACTIONS)),
            action=actions[i],
            logprob=math.log(1 / NUM_ACTIONS),
            value=0.0,
            reward=0.0,
            return_=returns[i],
            advantage=advantages[i],
        )
        for i in range(n)
    ]
    return make_batch(ts, num_actions=NUM_ACTIONS)


def test_empty_batch_returns_zero_stats():
    p = TabularPolicy(num_actions=NUM_ACTIONS, seed=0)
    rule = TabularACHUpdate(p)
    stats = rule.step(_batch([]))
    assert stats.policy_loss == 0.0 and stats.value_loss == 0.0


def test_rejects_non_tabular_policy():
    """A non-TabularPolicy instance (duck-typed) is rejected with a clear error."""

    class NotATabularPolicy:
        # Has the attribute the base class narrows, but is the wrong type.
        pass

    with pytest.raises(TypeError, match="TabularPolicy"):
        TabularACHUpdate(NotATabularPolicy())  # type: ignore[arg-type]


def test_ach_positive_advantage_increases_action_logit():
    """Hedge step: chosen action's logit grows when advantage > 0."""
    p = TabularPolicy(num_actions=NUM_ACTIONS, seed=0)
    before = p.get_logits(OBS)[0]
    rule = TabularACHUpdate(p, hedge_eta=1.0)
    rule.step(_batch([+1.0], actions=[0]))  # advantage +1 on action 0
    after = p.get_logits(OBS)[0]
    assert after > before


def test_ach_negative_advantage_decreases_action_logit():
    p = TabularPolicy(num_actions=NUM_ACTIONS, seed=0)
    before = p.get_logits(OBS)[1]
    rule = TabularACHUpdate(p, hedge_eta=1.0)
    rule.step(_batch([-1.0], actions=[1]))
    after = p.get_logits(OBS)[1]
    assert after < before


def test_ppo_positive_advantage_increases_action_logit():
    p = TabularPolicy(num_actions=NUM_ACTIONS, seed=0)
    before = p.get_logits(OBS)[0]
    rule = TabularPPOUpdate(p, clip_eps=0.2)
    rule.step(_batch([+1.0], actions=[0]))
    after = p.get_logits(OBS)[0]
    assert after > before


def test_ppo_clips_large_positive_advantage():
    """With clip_eps=0.2, an advantage of 10 only moves the logit by ~lr*0.2."""
    p = TabularPolicy(num_actions=NUM_ACTIONS, seed=0)
    before = p.get_logits(OBS)[0]
    rule = TabularPPOUpdate(p, AlgoConfig(learning_rate=1.0), clip_eps=0.2)
    rule.step(_batch([+10.0], actions=[0]))
    after = p.get_logits(OBS)[0]
    assert math.isclose(after - before, 0.2, abs_tol=1e-9)  # clipped to +0.2


def test_ppo_clips_large_negative_advantage():
    p = TabularPolicy(num_actions=NUM_ACTIONS, seed=0)
    before = p.get_logits(OBS)[0]
    rule = TabularPPOUpdate(p, AlgoConfig(learning_rate=1.0), clip_eps=0.2)
    rule.step(_batch([-10.0], actions=[0]))
    after = p.get_logits(OBS)[0]
    assert math.isclose(after - before, -0.2, abs_tol=1e-9)


def test_value_update_moves_toward_target():
    p = TabularPolicy(num_actions=NUM_ACTIONS, seed=0)
    p.values[_obs_to_key(OBS)] = 0.0
    rule = TabularPPOUpdate(p, AlgoConfig(learning_rate=0.1, value_coef=1.0))
    rule.step(_batch([0.0], returns=[1.0]))
    # v <- 0 + 0.1*1.0*(1.0 - 0) = 0.1
    assert math.isclose(p.value(OBS), 0.1, abs_tol=1e-9)


def test_stats_are_finite_and_reasonable():
    p = TabularPolicy(num_actions=NUM_ACTIONS, seed=0)
    rule = TabularACHUpdate(p)
    stats = rule.step(_batch([0.5, -0.5, 0.2]))
    for v in (stats.policy_loss, stats.value_loss, stats.entropy):
        assert math.isfinite(v)
    assert stats.entropy >= 0.0


def test_explained_variance_perfect_fit_is_one():
    ev = _explained_variance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert math.isclose(ev, 1.0, abs_tol=1e-9)


def test_explained_variance_zero_variance_target_is_zero():
    assert _explained_variance([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) == 0.0


def test_make_batch_builds_correct_shapes():
    b = _batch([0.1, 0.2, 0.3])
    assert b.size == 3
    assert b.obs.shape == (3, len(OBS))
    assert b.actions.shape == (3,)
    assert b.legal_mask.shape == (3, NUM_ACTIONS)
    assert b.legal_mask.all()  # all actions legal in this fixture


def test_make_batch_empty():
    b = make_batch([], num_actions=4)
    assert b.size == 0
    assert b.legal_mask.shape == (0, 4)


def test_ach_does_not_clip_unlike_ppo():
    """The headline algorithmic distinction (AGENTS.md §1 D4)."""
    # ACH with large advantage moves proportionally; PPO clips.
    p_ach = TabularPolicy(num_actions=NUM_ACTIONS, seed=0)
    p_ppo = TabularPolicy(num_actions=NUM_ACTIONS, seed=0)
    ach = TabularACHUpdate(p_ach, AlgoConfig(learning_rate=1.0), hedge_eta=1.0)
    ppo = TabularPPOUpdate(p_ppo, AlgoConfig(learning_rate=1.0), clip_eps=0.2)
    ach.step(_batch([+5.0], actions=[0]))
    ppo.step(_batch([+5.0], actions=[0]))
    # ACH moves by eta*adv = 5.0; PPO clips to 0.2.
    ach_delta = p_ach.get_logits(OBS)[0]
    ppo_delta = p_ppo.get_logits(OBS)[0]
    assert ach_delta > ppo_delta  # ACH moved much further
