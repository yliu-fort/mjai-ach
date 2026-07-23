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


# ---- ACH: A3 toggle (centered vs raw logit in the loss body) ----


def test_loss_body_logit_toggle_centered_vs_raw():
    """A3/U1 probe toggle: the gate stays on the centered logit while the loss
    body switches between centered (default) and raw logit. For a single
    on-policy sample, policy_loss == -eta * y_used * c * A / pi_old."""
    adv = 1.5
    losses: dict[bool, float] = {}
    for flag in (True, False):
        p = _policy(seed=3)
        _shift_logit(p, 1, +2.0)  # non-uniform policy -> raw logit != centered logit
        batch = _onpolicy_single_batch(p, 1, adv)
        losses[flag] = _ach(p, loss_centered_logits=flag).step(batch).policy_loss

    p = _policy(seed=3)
    _shift_logit(p, 1, +2.0)
    with torch.no_grad():
        obs_t = torch.as_tensor(OBS, dtype=torch.float32, device=p.device).unsqueeze(0)
        logits, _ = p.forward(obs_t)
    pi_old = float(torch.softmax(logits[0], dim=-1)[1].item())
    y_raw = float(logits[0, 1].item())
    y_cen = float((logits[0, 1] - logits[0].mean()).item())
    assert losses[False] == pytest.approx(-y_raw * adv / pi_old, rel=1e-4)
    assert losses[True] == pytest.approx(-y_cen * adv / pi_old, rel=1e-4)
    assert losses[False] != losses[True]


def test_loss_centered_logits_defaults_true():
    assert AlgoConfig().loss_centered_logits is True


# ---- ACH: A5 toggle (centered mean over legal actions only) ----


def _partial_legal_single_batch(p: MLPSharedActorCritic, action: int, legal: list[int]) -> Batch:
    """One on-policy sample where only ``legal`` actions are legal."""
    with torch.no_grad():
        obs_t = torch.as_tensor(OBS, dtype=torch.float32, device=p.device).unsqueeze(0)
        logits, _ = p.forward(obs_t)
        masked = logits[0].clone()
        for a in range(NUM_ACTIONS):
            if a not in legal:
                masked[a] = -torch.inf
        logprob = float(torch.log_softmax(masked, dim=-1)[action].item())
    t = Transition(
        obs=OBS,
        legal_actions=legal,
        action=action,
        logprob=logprob,
        value=0.0,
        reward=0.0,
        return_=p.value(OBS),
        advantage=1.5,
    )
    return make_batch([t], num_actions=NUM_ACTIONS)


def test_centered_mean_legal_only_changes_loss_only_with_illegal_actions():
    """A5 probe toggle: y_bar over legal actions only. With a partially-legal
    state the two means differ (so the loss differs); with all actions legal
    the toggle is a no-op."""
    losses: dict[bool, float] = {}
    for flag in (True, False):
        p = _policy(seed=5)
        _shift_logit(p, 3, +2.0)  # make the excluded illegal logit off-mean
        batch = _partial_legal_single_batch(p, 1, legal=[0, 1, 2])
        losses[flag] = _ach(p, centered_mean_legal_only=flag).step(batch).policy_loss
    assert losses[True] != losses[False]

    all_legal: dict[bool, float] = {}
    for flag in (True, False):
        p = _policy(seed=5)
        batch = _onpolicy_single_batch(p, 1, 1.5)
        all_legal[flag] = _ach(p, centered_mean_legal_only=flag).step(batch).policy_loss
    assert all_legal[True] == pytest.approx(all_legal[False], rel=1e-6)


def test_centered_mean_legal_only_defaults_false():
    assert AlgoConfig().centered_mean_legal_only is False


# ---- ACH: importance-weight / grad-norm telemetry (1/pi_old probe) ----


def test_ach_reports_importance_weight_and_grad_norm_telemetry():
    """The unbounded 1/pi_old probe needs iw/pterm/grad_norm on every update."""
    p = _policy()
    stats = _ach(p).step(_batch(8, advantages=MIXED_ADVS))
    for key in ("iw_max", "iw_mean", "pterm_max", "grad_norm"):
        assert key in stats.extra, key
        assert stats.extra[key] >= 0.0


def test_iw_max_tracks_the_rarest_sampled_action():
    """iw_max must equal 1/min(pi_old) over the batch — the blow-up driver."""
    p = _policy(seed=7)
    rare_lp = math.log(0.01)  # a very rare sampled action
    t_rare = Transition(
        obs=OBS,
        legal_actions=list(range(NUM_ACTIONS)),
        action=0,
        logprob=rare_lp,
        value=0.0,
        reward=0.0,
        return_=0.0,
        advantage=1.0,
    )
    t_common = Transition(
        obs=OBS,
        legal_actions=list(range(NUM_ACTIONS)),
        action=1,
        logprob=math.log(0.5),
        value=0.0,
        reward=0.0,
        return_=0.0,
        advantage=1.0,
    )
    stats = _ach(p).step(make_batch([t_rare, t_common], num_actions=NUM_ACTIONS))
    assert stats.extra["iw_max"] == pytest.approx(1.0 / 0.01, rel=1e-3)
    assert stats.extra["iw_mean"] == pytest.approx((1.0 / 0.01 + 1.0 / 0.5) / 2, rel=1e-3)


# ---- LayerNorm trunk: does it actually give l_th an absolute meaning? ----


def test_layernorm_normalizes_the_torso_output():
    """What LayerNorm actually guarantees: normalized FEATURES, not logits.

    Per sample the torso output has ~zero mean and ~unit variance. This is the
    real, checkable mechanism; it does NOT by itself bound the logit scale,
    because the policy head that follows is an unconstrained Linear
    (``logits = W @ LN(h) + b``). See docs/reproduce_report.md.
    """
    torch.manual_seed(0)
    obs = torch.randn(8, 4) * 5.0  # deliberately badly scaled input
    ln = MLPSharedActorCritic(
        obs_size=4, num_actions=NUM_ACTIONS, hidden_sizes=(16,), trunk_layernorm=True, seed=0
    )
    plain = MLPSharedActorCritic(
        obs_size=4, num_actions=NUM_ACTIONS, hidden_sizes=(16,), trunk_layernorm=False, seed=0
    )
    with torch.no_grad():
        f_ln = ln.torso(obs)
        f_plain = plain.torso(obs)
    assert f_ln.mean(dim=-1).abs().max().item() < 1e-5
    assert (f_ln.var(dim=-1, unbiased=False) - 1.0).abs().max().item() < 1e-3
    # The un-normalized torso does not have that property.
    assert f_plain.mean(dim=-1).abs().max().item() > 1e-3


def test_layernorm_is_off_by_default_and_changes_parameters():
    plain = MLPSharedActorCritic(obs_size=4, num_actions=NUM_ACTIONS, hidden_sizes=(8,), seed=0)
    assert plain.trunk_layernorm is False
    ln = MLPSharedActorCritic(
        obs_size=4, num_actions=NUM_ACTIONS, hidden_sizes=(8,), trunk_layernorm=True, seed=0
    )
    assert ln.trunk_layernorm is True
    assert len(ln.state_dict()) > len(plain.state_dict())


def test_gate_centered_logits_toggle_selects_the_gate_source():
    """gate_centered_logits=False thresholds the RAW logit instead of y - y_bar."""
    # Shift EVERY logit up by a constant: the raw logit is then far past +l_th
    # while the centered logit is unchanged (~0). Softmax — and hence the policy
    # — is identical, so only the gate source can distinguish the two.
    losses: dict[bool, float] = {}
    for flag in (True, False):
        p = _policy(seed=2)
        for a in range(NUM_ACTIONS):
            _shift_logit(p, a, +6.0)
        batch = _onpolicy_single_batch(p, 1, 1.0)  # positive advantage -> +l_th side
        losses[flag] = _ach(p, gate_centered_logits=flag, l_th=2.0).step(batch).policy_loss
    # Raw-gated run is blocked (raw logit > +l_th); centered-gated run is not.
    assert losses[False] == 0.0
    assert losses[True] != 0.0


def test_gate_centered_logits_defaults_true():
    assert AlgoConfig().gate_centered_logits is True


def test_grad_norm_is_pre_clip():
    """grad_norm reports the raw norm even when clipping would shrink it."""
    batch = _batch(8, advantages=MIXED_ADVS)
    unclipped = _ach(_policy(seed=11), max_grad_norm=0.0).step(batch).extra["grad_norm"]
    clipped = _ach(_policy(seed=11), max_grad_norm=1e-6).step(batch).extra["grad_norm"]
    assert clipped == pytest.approx(unclipped, rel=1e-6)


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
