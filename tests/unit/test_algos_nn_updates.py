"""Unit tests for the neural UpdateRules (AGENTS.md §5).

Covers the paper-faithful ACH endpoint (Fu et al. ICLR 2022, Algorithm 2 /
Eq. 29, p24) and the reference PPO endpoint. Run on CPU (dev env has torch+cpu
while cu128 downloads). Each test forces CPU via gpu_assert.require_cpu().
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
MIXED_ADVS = [0.5, -0.5, 0.3, -0.3, 0.8, -0.2, 0.1, -0.7]


@pytest.fixture(autouse=True)
def _cpu_mode():
    gpu_assert.reset_for_tests()
    gpu_assert.require_cpu()
    yield
    gpu_assert.reset_for_tests()


def _policy(seed: int = 0) -> MLPSharedActorCritic:
    return MLPSharedActorCritic(obs_size=4, num_actions=NUM_ACTIONS, seed=seed)


def _ach(p: MLPSharedActorCritic, **cfg_kwargs) -> NNACHUpdate:
    """ACH with explicit SGD (paper H.3) and no entropy term unless asked."""
    base = {"optimizer": "sgd", "learning_rate": 1e-2, "entropy_coef": 0.0}
    return NNACHUpdate(p, AlgoConfig(**{**base, **cfg_kwargs}))


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


def _logp_under_current(p: MLPSharedActorCritic, action: int) -> float:
    """Exact log-prob of ``action`` under the policy's current weights (all-legal obs)."""
    with torch.no_grad():
        obs_t = torch.as_tensor(OBS, dtype=torch.float32, device=p.device).unsqueeze(0)
        logits, _ = p.forward(obs_t)
        return float(torch.log_softmax(logits[0], dim=-1)[action].item())


def _onpolicy_single_batch(
    p: MLPSharedActorCritic, action: int, advantage: float, *, logprob: float | None = None
) -> Batch:
    """One sample whose pi_old equals the current policy (ratio == 1) unless overridden.

    ``return_`` is set to the current value so the value-loss gradient is exactly
    zero; with ``entropy_coef=0`` the ONLY possible weight change is the ACH
    policy term — which isolates the gate.
    """
    t = Transition(
        obs=OBS,
        legal_actions=list(range(NUM_ACTIONS)),
        action=action,
        logprob=logprob if logprob is not None else _logp_under_current(p, action),
        value=0.0,
        reward=0.0,
        return_=p.value(OBS),
        advantage=advantage,
    )
    return make_batch([t], num_actions=NUM_ACTIONS)


def _shift_logit(p: MLPSharedActorCritic, action: int, delta: float) -> None:
    """Add a constant to one policy-head bias (moves the centered logit by ~delta)."""
    with torch.no_grad():
        p.policy_head.bias[action] += delta


def _head_weights(p: MLPSharedActorCritic) -> torch.Tensor:
    return p.policy_head.weight.detach().clone()


# ---- construction / interface ----


def test_rejects_non_mlp_policy():
    p = TabularPolicy(num_actions=4, seed=0)
    with pytest.raises(TypeError, match="MLPSharedActorCritic"):
        NNACHUpdate(p)  # type: ignore[arg-type]


def test_empty_batch_short_circuits():
    p = _policy()
    stats = _ach(p).step(make_batch([], num_actions=NUM_ACTIONS))
    assert stats.policy_loss == 0.0


# ---- optimizer + hyperparameter fidelity (paper H.3, p27-28) ----


def test_ach_optimizer_is_sgd_without_momentum():
    """Paper p27: 'stochastic gradient descent with a constant learning rate'."""
    rule = _ach(_policy(), learning_rate=1e-3)
    assert isinstance(rule.optimizer, torch.optim.SGD)
    assert rule.optimizer.param_groups[0]["momentum"] == 0.0
    assert rule.optimizer.param_groups[0]["lr"] == 1e-3


def test_ach_bare_construction_defaults_to_sgd():
    """No explicit config -> endpoint default is SGD (single paper-faithful ACH)."""
    rule = NNACHUpdate(_policy())
    assert isinstance(rule.optimizer, torch.optim.SGD)


def test_ach_rejects_non_sgd_optimizer():
    with pytest.raises(ValueError, match="SGD"):
        NNACHUpdate(_policy(), AlgoConfig(optimizer="adam"))


def test_ach_paper_default_hyperparams():
    """AlgoConfig defaults match p27 Table 7 / p28 Table 8."""
    cfg = AlgoConfig()
    assert cfg.eta == 1.0  # hedge coefficient (p27 Table 7)
    assert cfg.l_th == 2.0  # logit threshold (p28 Table 8)
    assert cfg.ratio_eps == 0.5  # vacuous when synchronous (p28)


def test_ppo_optimizer_is_adam_with_37details_eps():
    rule = NNPPOUpdate(_policy(), AlgoConfig(learning_rate=1e-3))
    assert isinstance(rule.optimizer, torch.optim.Adam)
    assert rule.optimizer.param_groups[0]["eps"] == 1e-5


# ---- generic stepping ----


def test_ppo_returns_finite_stats_after_step():
    p = _policy()
    rule = NNPPOUpdate(p, AlgoConfig(learning_rate=1e-3))
    stats = rule.step(_batch(8))
    for v in (stats.policy_loss, stats.value_loss, stats.entropy, stats.approx_kl, stats.clip_frac):
        assert math.isfinite(v)
    assert 0.0 <= stats.clip_frac <= 1.0


def test_ppo_step_changes_weights():
    p = _policy()
    before = _head_weights(p)
    NNPPOUpdate(p, AlgoConfig(learning_rate=1e-2)).step(_batch(8, advantages=MIXED_ADVS))
    assert not torch.allclose(before, p.policy_head.weight.detach())


def test_ach_step_changes_weights():
    p = _policy()
    before = _head_weights(p)
    _ach(p).step(_batch(8, advantages=MIXED_ADVS))
    assert not torch.allclose(before, p.policy_head.weight.detach())


def test_ach_reports_gate_off_fraction():
    stats = _ach(_policy()).step(_batch(8, advantages=MIXED_ADVS))
    assert "gate_off_frac" in stats.extra
    assert 0.0 <= stats.extra["gate_off_frac"] <= 1.0


# ---- ACH gate: advantage-sign-dependent, one-sided (p24 Algorithm 2) ----


def test_gate_blocks_push_beyond_upper_bound_but_allows_correction():
    """y_a >> +l_th: A>0 (push further up) is gated OFF; A<0 (pull back down)
    must keep its gradient — this is the one-sided property a symmetric
    |y|<=l_th gate would violate (audit F1)."""
    action = 0
    # A>0, logit already past the upper bound -> gated: weights unchanged.
    p = _policy()
    _shift_logit(p, action, +10.0)
    before = _head_weights(p)
    _ach(p).step(_onpolicy_single_batch(p, action, advantage=+1.0))
    assert torch.equal(before, p.policy_head.weight.detach())
    # A<0, same logit -> corrective direction allowed: weights change.
    p = _policy()
    _shift_logit(p, action, +10.0)
    before = _head_weights(p)
    _ach(p).step(_onpolicy_single_batch(p, action, advantage=-1.0))
    assert not torch.equal(before, p.policy_head.weight.detach())


def test_gate_blocks_push_beyond_lower_bound_but_allows_correction():
    """y_a << -l_th: A<0 (push further down) is gated OFF; A>0 (pull back up)
    keeps its gradient."""
    action = 1
    p = _policy()
    _shift_logit(p, action, -10.0)
    before = _head_weights(p)
    _ach(p).step(_onpolicy_single_batch(p, action, advantage=-1.0))
    assert torch.equal(before, p.policy_head.weight.detach())
    p = _policy()
    _shift_logit(p, action, -10.0)
    before = _head_weights(p)
    _ach(p).step(_onpolicy_single_batch(p, action, advantage=+1.0))
    assert not torch.equal(before, p.policy_head.weight.detach())


def test_ratio_gate_blocks_when_ratio_exceeds_bound():
    """pi/pi_old >= 1+eps with A>=0 gates the sample even when the logit is in range
    (vacuous under synchronous self-play, active under async staleness)."""
    action = 2
    p = _policy()
    before = _head_weights(p)
    stale_logp = _logp_under_current(p, action) - 5.0  # ratio = e^5 >> 1+eps
    _ach(p).step(_onpolicy_single_batch(p, action, +1.0, logprob=stale_logp))
    assert torch.equal(before, p.policy_head.weight.detach())


# ---- ACH: NO advantage normalization (paper p24) ----


def test_ach_policy_loss_scales_linearly_with_advantages():
    """Unnormalized loss is linear in A: scaling all advantages by 10 scales
    policy_loss by 10 (normalization would make it scale-invariant)."""
    advs = MIXED_ADVS
    s1 = _ach(_policy(seed=7)).step(_batch(8, advantages=advs))
    s2 = _ach(_policy(seed=7)).step(_batch(8, advantages=[10.0 * a for a in advs]))
    assert math.isclose(s2.policy_loss, 10.0 * s1.policy_loss, rel_tol=1e-4)


def test_ach_constant_advantages_give_nonzero_policy_loss():
    """Constant advantages are NOT normalized to zero (the old unified
    implementation zeroed them; the paper's loss uses raw A)."""
    stats = _ach(_policy()).step(_batch(8, advantages=[0.5] * 8))
    assert stats.policy_loss != 0.0


def test_ppo_constant_advantages_give_zero_policy_loss():
    """PPO keeps per-batch advantage normalization (37-details): constant A -> 0."""
    p = _policy()
    stats = NNPPOUpdate(p, AlgoConfig(learning_rate=1e-2)).step(_batch(8, advantages=[0.5] * 8))
    assert stats.policy_loss == 0.0


# ---- value head / persistence ----


def test_value_head_moves_toward_returns():
    """After enough steps with strong value coef, predicted values approach targets."""
    p = MLPSharedActorCritic(obs_size=4, num_actions=NUM_ACTIONS, hidden_sizes=(16,), seed=0)
    rule = NNPPOUpdate(p, AlgoConfig(learning_rate=5e-3, value_coef=1.0))
    for _ in range(20):
        rule.step(_batch(16, returns=[5.0] * 16))
    with torch.no_grad():
        v = p.value(OBS)
    assert v > 0.5  # moved up from ~0 toward 5


def test_optimizer_state_roundtrip():
    p = _policy()
    rule = _ach(p)
    rule.step(_batch(4))
    state = rule.state_dict()
    assert "optimizer" in state
    p2 = _policy()
    rule2 = _ach(p2)
    rule2.load_state_dict(state)  # should not raise


def test_safe_log_handles_zero():
    assert math.isfinite(safe_log(0.0))
    assert math.isclose(safe_log(1.0), 0.0, abs_tol=1e-9)
