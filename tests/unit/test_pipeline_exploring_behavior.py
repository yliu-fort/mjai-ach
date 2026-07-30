"""Exploring behavior policy (mu) + V-trace off-policy correction.

The contract these pin down, in the order it matters:

1. ``behavior_epsilon=0`` changes **nothing** — same action, same logprob, same
   RNG consumption. The paper reproduction runs through this path.
2. The recorded logprob is ``log mu(a|s)``, not ``log pi(a|s)``. Getting this
   wrong is silent: ACH's ``1/pi_old`` would stop cancelling the sampling
   probability and the gradient would be biased with no error anywhere.
3. :func:`behavior_prob` and :func:`target_ratio` are exact inverses, including
   in the regime that motivated the second one's unusual form (rare actions,
   where ``mu -> eps/|legal|``).
4. V-trace at ``epsilon=0`` reduces **exactly** to GAE(lambda) in the advantage.
5. Deviating from the paper is never silent (ACHFidelityWarning).
"""

from __future__ import annotations

import math
import warnings

import pytest

from mjai.agents.base import behavior_prob, target_ratio
from mjai.agents.mlp import MLPSharedActorCritic
from mjai.agents.tabular import TabularPolicy
from mjai.algos.transition import Transition
from mjai.algos.update_rule import ACHFidelityWarning
from mjai.games.loader import load_game
from mjai.pipeline.rollout import RolloutConfig, RolloutWorkerCore

# --------------------------------------------------------------------------
# 3. the mixture and its inverse
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pi_a", [0.5, 0.1, 1e-3, 1e-9, 1e-15, 0.0])
@pytest.mark.parametrize("epsilon", [0.01, 0.1, 0.5])
@pytest.mark.parametrize("n_legal", [2, 13])
def test_behavior_prob_and_target_ratio_round_trip(pi_a: float, epsilon: float, n_legal: int):
    """``target_ratio(behavior_prob(pi)) == pi/mu`` to float64, even as pi -> 0.

    The rare-action regime is the whole point: that is where the naive
    ``(mu - u)/(1-eps)/mu`` form cancels catastrophically.
    """
    mu = behavior_prob(pi_a, n_legal, epsilon)
    got = target_ratio(mu, n_legal, epsilon)
    assert got == pytest.approx(pi_a / mu, abs=1e-12, rel=1e-9)
    assert got >= 0.0


def test_target_ratio_is_identity_when_on_policy():
    assert target_ratio(0.3, 5, 0.0) == 1.0


# --------------------------------------------------------------------------
# 1 & 2. the sampling path
# --------------------------------------------------------------------------


def _mlp() -> MLPSharedActorCritic:
    return MLPSharedActorCritic(obs_size=4, num_actions=3, hidden_sizes=(8,), seed=0, device="cpu")


def _tabular(obs: list[float] | None = None) -> TabularPolicy:
    """A tabular policy with a NON-uniform row at ``obs``.

    Mixing a uniform policy with the uniform distribution is the identity, so a
    freshly-built table cannot exercise the behavior-policy path at all.
    """
    from mjai.agents.tabular import _obs_to_key

    policy = TabularPolicy(num_actions=3, seed=0)
    if obs is not None:
        policy.logits[_obs_to_key(obs)] = [2.0, 0.0, -1.0]
    return policy


@pytest.mark.parametrize("make", [_mlp, lambda: _tabular([0.1, -0.2, 0.3, 0.4])])
def test_epsilon_zero_is_bit_identical(make):
    """The default path must be untouched — same action AND same logprob."""
    obs, legal = [0.1, -0.2, 0.3, 0.4], [0, 2]
    a1, lp1, v1 = make().act_with_value(obs, legal, eval=False)
    a2, lp2, v2 = make().act_with_value(obs, legal, eval=False, behavior_epsilon=0.0)
    assert (a1, lp1, v1) == (a2, lp2, v2)


@pytest.mark.parametrize("make", [_mlp, lambda: _tabular([0.1, -0.2, 0.3, 0.4])])
def test_recorded_logprob_is_the_behavior_probability(make):
    """log mu(a), not log pi(a) — the bug that would silently bias the gradient."""
    obs, legal, eps = [0.1, -0.2, 0.3, 0.4], [0, 2], 0.4
    policy = make()
    pi = dict(zip(legal, _legal_probs(policy, obs, legal), strict=True))
    action, logprob, _v = policy.act_with_value(obs, legal, eval=False, behavior_epsilon=eps)
    assert action in legal
    expected = behavior_prob(pi[action], len(legal), eps)
    assert math.exp(logprob) == pytest.approx(expected, rel=1e-5)
    # And it must differ from the on-policy value, or the test proves nothing.
    assert math.exp(logprob) != pytest.approx(pi[action], rel=1e-9)


def _legal_probs(policy, obs: list[float], legal: list[int]) -> list[float]:
    logits = policy.action_logits(obs, legal)
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def test_exploration_reaches_actions_the_policy_avoids():
    """The point of the knob: mass on actions pi has all but abandoned."""
    policy = _mlp()
    obs, legal = [0.0, 0.0, 0.0, 0.0], [0, 1, 2]
    # Drive one action's logit down so pi effectively never plays it.
    with __import__("torch").no_grad():
        policy.policy_head.bias[:] = __import__("torch").tensor([0.0, -30.0, 0.0])
    on_policy = [policy.act(obs, legal)[0] for _ in range(400)]
    exploring = [policy.act(obs, legal, behavior_epsilon=0.3)[0] for _ in range(400)]
    assert on_policy.count(1) == 0
    assert exploring.count(1) > 20  # ~0.3/3 * 400 = 40 expected


# --------------------------------------------------------------------------
# 4. V-trace
# --------------------------------------------------------------------------


def _episode(values: list[float], logprobs: list[float], n_legal: int = 2) -> list[Transition]:
    return [
        Transition(
            obs=[0.0],
            legal_actions=list(range(n_legal)),
            action=0,
            logprob=lp,
            value=v,
            reward=0.0,
            return_=0.0,
            player=0,
        )
        for v, lp in zip(values, logprobs, strict=True)
    ]


def _worker(**kwargs) -> RolloutWorkerCore:
    return RolloutWorkerCore(load_game("kuhn"), config=RolloutConfig(**kwargs))


def test_vtrace_reduces_exactly_to_gae_when_on_policy():
    """epsilon=0 => every ratio is 1 => the advantage IS GAE(lambda).

    This is what makes the estimator a strict generalization rather than a
    second, incomparable arm: any difference measured against a vtrace(eps=0)
    control is attributable to the exploration alone.
    """
    values = [0.2, -0.1, 0.35]
    logprobs = [math.log(0.5)] * 3
    payoff = 1.0

    gae_ts = _episode(values, logprobs)
    _worker(gae_lambda=0.95, advantage_estimator="gae")._assign_returns(gae_ts, [payoff, 0.0])
    vt_ts = _episode(values, logprobs)
    _worker(gae_lambda=0.95, advantage_estimator="vtrace", behavior_epsilon=0.0)._assign_returns(
        vt_ts, [payoff, 0.0]
    )

    for g, v in zip(gae_ts, vt_ts, strict=True):
        assert v.advantage == pytest.approx(g.advantage, rel=1e-12, abs=1e-12)
    # The value TARGET does change (MC return vs the V-trace target) -- documented
    # in _assign_vtrace, and the reason ablations use a vtrace(eps=0) control.
    assert vt_ts[0].return_ != pytest.approx(gae_ts[0].return_, abs=1e-9)


def test_vtrace_downweights_off_policy_steps():
    """A step the target policy would rarely take gets a shrunken correction."""
    values = [0.0, 0.0]
    eps, n_legal = 0.5, 2
    # An action pi almost never plays: mu(a) ~= eps/n_legal, so rho ~= 0.
    rare = math.log(behavior_prob(1e-6, n_legal, eps))
    common = math.log(behavior_prob(0.9, n_legal, eps))

    rare_ts = _episode(values, [rare, common], n_legal)
    _worker(advantage_estimator="vtrace", behavior_epsilon=eps)._assign_returns(rare_ts, [1.0, 0.0])
    common_ts = _episode(values, [common, common], n_legal)
    _worker(advantage_estimator="vtrace", behavior_epsilon=eps)._assign_returns(
        common_ts, [1.0, 0.0]
    )

    assert abs(rare_ts[0].advantage) < abs(common_ts[0].advantage)
    assert all(math.isfinite(t.advantage) and math.isfinite(t.return_) for t in rare_ts)


def test_unknown_estimator_fails_loudly():
    with pytest.raises(ValueError, match="advantage_estimator"):
        _worker(advantage_estimator="td0")._assign_returns(_episode([0.0], [0.0]), [1.0, 0.0])


# --------------------------------------------------------------------------
# 5. deviations are never silent
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [{"behavior_epsilon": 0.1}, {"advantage_estimator": "vtrace"}],
)
def test_deviation_warns_when_ach_has_weight(overrides):
    from mjai.scripts.experiment_build import ExperimentConfig, warn_if_rollout_ach_incompatible

    cfg = ExperimentConfig(game="kuhn", algo="ach", self_play_mode="mirror", **overrides)
    with pytest.warns(ACHFidelityWarning):
        warn_if_rollout_ach_incompatible(cfg)


def test_defaults_and_ppo_do_not_warn():
    from mjai.scripts.experiment_build import ExperimentConfig, warn_if_rollout_ach_incompatible

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        warn_if_rollout_ach_incompatible(
            ExperimentConfig(game="kuhn", algo="ach", self_play_mode="mirror")
        )
        warn_if_rollout_ach_incompatible(
            ExperimentConfig(game="kuhn", algo="ppo", self_play_mode="mirror", behavior_epsilon=0.1)
        )
