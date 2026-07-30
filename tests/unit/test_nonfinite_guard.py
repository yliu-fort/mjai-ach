"""The divergence guard: a diverged network fails with a diagnosis, not a torch error.

Before this guard, a run whose logits overflowed died inside
``torch.multinomial`` with ``probability tensor contains either inf, nan or
element < 0`` and no indication of which update or which knob produced it — the
failure mode measured on 4 of 30 BRPS runs (docs/brps_mlp_nonconvergence.md §4).

Two raise sites, tested separately because they answer different questions:
:class:`~mjai.agents.mlp.MLPSharedActorCritic` says *the network is already
broken*, and :class:`~mjai.algos.nn_updates.NNActorCriticUpdate` says *this
update is what broke it*. Neither repairs anything (AGENTS.md §11).

CPU-only, like the rest of the fast suite.
"""

from __future__ import annotations

import math

import pytest
import torch

from mjai.agents.mlp import MLPSharedActorCritic
from mjai.agents.nonfinite import NonFiniteNetworkError, nonfinite_summary
from mjai.algos.nn_updates import NNActorCriticUpdate, safe_log
from mjai.algos.transition import Batch, Transition, make_batch
from mjai.algos.update_rule import AlgoConfig
from mjai.utils import gpu_assert

OBS = [0.5, -0.2, 0.0, 1.0]
NUM_ACTIONS = 3
LEGAL = list(range(NUM_ACTIONS))


@pytest.fixture(autouse=True)
def _cpu_mode():
    gpu_assert.reset_for_tests()
    gpu_assert.require_cpu()
    yield
    gpu_assert.reset_for_tests()


def _policy(seed: int = 0) -> MLPSharedActorCritic:
    return MLPSharedActorCritic(obs_size=len(OBS), num_actions=NUM_ACTIONS, seed=seed)


def _poison(policy: MLPSharedActorCritic, value: float) -> None:
    """Put ``value`` (inf or nan) into one policy-head bias entry, in place."""
    with torch.no_grad():
        policy.policy_head.bias[1] = value


# ---- nonfinite_summary (the message builder both sites share) ----


def test_summary_is_none_for_finite_tensor():
    assert nonfinite_summary("logits", torch.tensor([1.0, -2.0, 3.0])) is None


def test_summary_counts_nan_and_inf_separately():
    msg = nonfinite_summary("logits", torch.tensor([float("nan"), float("inf"), 1.0]))
    assert msg is not None
    assert "1 nan" in msg and "1 inf" in msg and "of 3" in msg


def test_summary_truncates_wide_tensors():
    msg = nonfinite_summary("param", torch.full((50,), float("nan")))
    assert msg is not None
    assert "50 nan" in msg
    assert msg.endswith(", ...")  # not 50 values in an exception message


def _batch(advantage: float, pi_old: float) -> Batch:
    """One transition carrying a chosen advantage and behavior probability.

    ``pi_old`` enters the ACH term as ``1/pi_old``, so a tiny value here is the
    unbounded importance weight the real failure rode in on.
    """
    ts = [
        Transition(
            obs=OBS,
            legal_actions=LEGAL,
            action=0,
            logprob=safe_log(pi_old),
            value=0.0,
            reward=0.0,
            return_=0.0,
            advantage=advantage,
        )
    ]
    return make_batch(ts, num_actions=NUM_ACTIONS)


def _ach(policy: MLPSharedActorCritic, **kwargs) -> NNActorCriticUpdate:
    base = {"optimizer": "sgd", "learning_rate": 1e-2, "entropy_coef": 0.0}
    return NNActorCriticUpdate(policy, AlgoConfig(**{**base, **kwargs}))


# ---- rollout side: refuse to sample from a diverged network ----


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
@pytest.mark.parametrize("method", ["act", "act_with_value"])
def test_sampling_raises_with_diagnosis(bad: float, method: str):
    policy = _policy()
    _poison(policy, bad)
    with pytest.raises(NonFiniteNetworkError) as exc:
        getattr(policy, method)(OBS, LEGAL, eval=False)
    msg = str(exc.value)
    assert f"obs={OBS}" in msg  # which decision point
    assert f"legal_actions={LEGAL}" in msg
    assert "param policy_head.bias" in msg  # which parameter
    assert "WEIGHTS are already non-finite" in msg
    assert "iw_clip" in msg and "brps_mlp_nonconvergence" in msg  # the hint


def test_minus_inf_logit_is_a_valid_distribution_and_is_left_to_the_update():
    """A ``-inf`` logit is the one non-finite weight that is NOT a broken sample.

    ``log_softmax`` sends it to ``-inf`` and ``exp`` to exactly 0, so the
    distribution over the rest is valid and that action is simply unreachable —
    the sampler has nothing to complain about, and complaining anyway would mean
    paying an unconditional ``isfinite`` scan on every decision point. The next
    update catches it (the entropy term hits ``0 * -inf``), which is the boundary
    this test pins so a future change does not move it silently.
    """
    policy = _policy()
    _poison(policy, float("-inf"))
    action, logprob, value = policy.act_with_value(OBS, LEGAL, eval=False)
    assert action in (0, 2) and math.isfinite(logprob) and math.isfinite(value)
    with pytest.raises(NonFiniteNetworkError) as exc:
        _ach(policy).step(_batch(advantage=1.0, pi_old=1 / NUM_ACTIONS))
    assert "entropy=nan" in str(exc.value)


def test_greedy_branch_is_guarded_too():
    """``eval=True`` takes an argmax, which is just as meaningless over nan."""
    policy = _policy()
    _poison(policy, float("nan"))
    with pytest.raises(NonFiniteNetworkError):
        policy.act_with_value(OBS, LEGAL, eval=True)


def test_exploring_branch_is_guarded_too():
    policy = _policy()
    _poison(policy, float("nan"))
    with pytest.raises(NonFiniteNetworkError):
        policy.act(OBS, LEGAL, eval=False, behavior_epsilon=0.1)


def test_healthy_policy_still_samples():
    """The guard must not fire on a normal network (it runs on every step)."""
    policy = _policy()
    for _ in range(20):
        action, logprob, value = policy.act_with_value(OBS, LEGAL, eval=False)
        assert action in LEGAL
        assert math.isfinite(logprob) and math.isfinite(value)


def test_diverged_weights_are_not_repaired():
    """AGENTS.md §11: the guard raises; it does not clamp, zero, or reinitialize."""
    policy = _policy()
    _poison(policy, float("inf"))
    with pytest.raises(NonFiniteNetworkError):
        policy.act(OBS, LEGAL, eval=False)
    assert torch.isinf(policy.policy_head.bias[1])


# ---- update side: name the update that diverged ----


def _sharpen(policy: MLPSharedActorCritic, action: int, logit: float) -> float:
    """Make ``action`` rare (logit = -|logit|) and return its exact log-prob.

    Zeroing the head's weight makes the logits the biases, so the resulting
    ``pi_old`` is both tiny and *consistent with the current policy* — which is
    what the real failure looked like: synchronous self-play, ratio ~= 1, so the
    ratio gate passes and the unbounded ``1/pi_old`` reaches the loss.
    """
    with torch.no_grad():
        policy.policy_head.weight.zero_()
        policy.policy_head.bias.zero_()
        policy.policy_head.bias[action] = -abs(logit)
        logits = torch.tensor(policy.action_logits(OBS, LEGAL))
        return float(torch.log_softmax(logits, dim=-1)[action].item())


def test_update_raises_when_the_policy_term_overflows():
    """Huge advantage x unbounded 1/pi_old overflows the loss itself."""
    policy = _policy()
    pi_old = math.exp(_sharpen(policy, action=0, logit=20.0))
    rule = _ach(policy)
    with pytest.raises(NonFiniteNetworkError) as exc:
        rule.step(_batch(advantage=1e30, pi_old=pi_old))
    msg = str(exc.value)
    assert "the update diverged" in msg
    assert "the forward overflowed before this step" in msg  # the loss, not the weights
    assert "iw_max" in msg  # the probe that explains it
    assert "brps_mlp_nonconvergence" in msg


def test_update_reports_a_weight_overflow_with_finite_loss():
    """Finite loss AND finite gradient, but ``w - lr*g`` leaves float32 range.

    The measured BRPS case: a scalar-only guard passes here and the run instead
    dies later in the rollout, blaming the wrong step.
    """
    policy = _policy()
    # Grad clipping off, so the step size is the one the lr asks for (the paper's
    # own setting, configs/exp/*: max_grad_norm: 0.0).
    rule = _ach(policy, learning_rate=1e38, max_grad_norm=0.0)
    with pytest.raises(NonFiniteNetworkError) as exc:
        rule.step(_batch(advantage=1e3, pi_old=1 / NUM_ACTIONS))
    msg = str(exc.value)
    assert "this optimizer step overflowed the weights" in msg
    assert "weights now: param" in msg


def test_update_names_pre_existing_divergence_separately():
    """Non-finite loss: the culprit is before this step, not this backward."""
    policy = _policy()
    _poison(policy, float("nan"))
    rule = _ach(policy)
    with pytest.raises(NonFiniteNetworkError) as exc:
        rule.step(_batch(advantage=1.0, pi_old=1 / NUM_ACTIONS))
    msg = str(exc.value)
    assert "the forward overflowed before this step" in msg
    assert "param policy_head.bias" in msg


def test_healthy_update_does_not_raise():
    """The guard must not false-positive on an ordinary update."""
    policy = _policy()
    rule = _ach(policy)
    stats = rule.step(_batch(advantage=0.5, pi_old=1 / NUM_ACTIONS))
    assert math.isfinite(stats.policy_loss)
    assert math.isfinite(stats.extra["grad_norm"])
