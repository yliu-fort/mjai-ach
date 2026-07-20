"""Unit tests for the neural UpdateRules (AGENTS.md §5).

Run on CPU (dev env has torch+cpu while cu128 downloads). Each test forces CPU
via gpu_assert.require_cpu().
"""

from __future__ import annotations

import math

import pytest
import torch

from mjai.agents.mlp import MLPSharedActorCritic
from mjai.agents.tabular import TabularPolicy
from mjai.algos.nn_updates import NNACHUpdate, NNPPOUpdate, safe_log
from mjai.algos.transition import Batch, Transition, make_batch
from mjai.algos.update_rule import AlgoConfig
from mjai.utils import gpu_assert

OBS = [0.5, -0.2, 0.0, 1.0]
NUM_ACTIONS = 4


@pytest.fixture(autouse=True)
def _cpu_mode():
    gpu_assert.reset_for_tests()
    gpu_assert.require_cpu()
    yield
    gpu_assert.reset_for_tests()


def _batch(n: int, advantages=None, returns=None) -> Batch:
    advantages = advantages if advantages is not None else [0.5] * n
    returns = returns if returns is not None else [1.0] * n
    ts = [
        Transition(
            obs=OBS,
            legal_actions=list(range(NUM_ACTIONS)),
            action=i % NUM_ACTIONS,
            logprob=safe_log(1 / NUM_ACTIONS),
            value=0.0,
            reward=0.0,
            return_=returns[i],
            advantage=advantages[i],
        )
        for i in range(n)
    ]
    return make_batch(ts, num_actions=NUM_ACTIONS)


def test_rejects_non_mlp_policy():
    p = TabularPolicy(num_actions=4, seed=0)
    with pytest.raises(TypeError, match="MLPSharedActorCritic"):
        NNACHUpdate(p)  # type: ignore[arg-type]


def test_empty_batch_short_circuits():
    p = MLPSharedActorCritic(obs_size=4, num_actions=NUM_ACTIONS, seed=0)
    rule = NNACHUpdate(p)
    stats = rule.step(make_batch([], num_actions=NUM_ACTIONS))
    assert stats.policy_loss == 0.0


def test_ppo_returns_finite_stats_after_step():
    p = MLPSharedActorCritic(obs_size=4, num_actions=NUM_ACTIONS, seed=0)
    rule = NNPPOUpdate(p, AlgoConfig(learning_rate=1e-3), n_epochs=2)
    stats = rule.step(_batch(8))
    for v in (stats.policy_loss, stats.value_loss, stats.entropy, stats.approx_kl, stats.clip_frac):
        assert math.isfinite(v)
    # PPO reports clip_frac and approx_kl (ACH leaves them 0).
    assert stats.approx_kl >= 0.0


def test_ach_leaves_kl_and_clip_at_zero():
    """The ACH rule does not track PPO-specific diagnostics (AGENTS.md §1 D4)."""
    p = MLPSharedActorCritic(obs_size=4, num_actions=NUM_ACTIONS, seed=0)
    rule = NNACHUpdate(p, AlgoConfig(learning_rate=1e-3))
    stats = rule.step(_batch(8))
    assert stats.approx_kl == 0.0
    assert stats.clip_frac == 0.0
    # ACH does populate the extra dict with advantage stats.
    assert "adv_mean" in stats.extra


def test_ppo_step_changes_weights():
    p = MLPSharedActorCritic(obs_size=4, num_actions=NUM_ACTIONS, seed=0)
    before = p.policy_head.weight.detach().clone()
    # Non-constant advantages so normalization doesn't zero them out.
    advs = [0.5, -0.5, 0.3, -0.3, 0.8, -0.2, 0.1, -0.7]
    NNPPOUpdate(p, AlgoConfig(learning_rate=1e-2), n_epochs=2).step(_batch(8, advantages=advs))
    after = p.policy_head.weight.detach()
    assert not torch.allclose(before, after)


def test_ach_step_changes_weights():
    p = MLPSharedActorCritic(obs_size=4, num_actions=NUM_ACTIONS, seed=0)
    before = p.policy_head.weight.detach().clone()
    advs = [0.5, -0.5, 0.3, -0.3, 0.8, -0.2, 0.1, -0.7]
    NNACHUpdate(p, AlgoConfig(learning_rate=1e-2)).step(_batch(8, advantages=advs))
    after = p.policy_head.weight.detach()
    assert not torch.allclose(before, after)


def test_constant_advantages_produce_zero_policy_gradient():
    """Normalization zeroes identical advantages -> policy loss is ~0.

    This documents the intended behavior: only the value head learns from a
    constant-advantage batch; the policy gradient vanishes.
    """
    p = MLPSharedActorCritic(obs_size=4, num_actions=NUM_ACTIONS, seed=0)
    rule = NNACHUpdate(p, AlgoConfig(learning_rate=1e-2))
    stats = rule.step(_batch(8, advantages=[0.5] * 8))
    assert abs(stats.policy_loss) < 1e-6


def test_value_head_moves_toward_returns():
    """After enough steps with strong value coef, predicted values approach targets."""
    p = MLPSharedActorCritic(obs_size=4, num_actions=NUM_ACTIONS, hidden_sizes=(16,), seed=0)
    rule = NNPPOUpdate(p, AlgoConfig(learning_rate=5e-3, value_coef=1.0), n_epochs=1)
    # Train on a batch where returns are large positive.
    for _ in range(20):
        rule.step(_batch(16, returns=[5.0] * 16))
    with torch.no_grad():
        v = p.value(OBS)
    assert v > 0.5  # moved up from ~0 toward 5


def test_optimizer_state_roundtrip():
    p = MLPSharedActorCritic(obs_size=4, num_actions=NUM_ACTIONS, seed=0)
    rule = NNACHUpdate(p)
    rule.step(_batch(4))
    state = rule.state_dict()
    assert "optimizer" in state
    p2 = MLPSharedActorCritic(obs_size=4, num_actions=NUM_ACTIONS, seed=0)
    rule2 = NNACHUpdate(p2)
    rule2.load_state_dict(state)  # should not raise


def test_advantage_normalization_handles_constant_batch():
    """A batch with identical advantages must not divide by zero."""
    p = MLPSharedActorCritic(obs_size=4, num_actions=NUM_ACTIONS, seed=0)
    rule = NNACHUpdate(p, AlgoConfig(learning_rate=1e-3))
    stats = rule.step(_batch(8, advantages=[0.5] * 8))  # std = 0
    assert math.isfinite(stats.policy_loss)


def test_safe_log_handles_zero():
    assert math.isfinite(safe_log(0.0))
    assert math.isclose(safe_log(1.0), 0.0, abs_tol=1e-9)
