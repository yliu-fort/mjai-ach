"""Reach-tempered sample weights (``RolloutConfig.sample_weight_kappa``).

The mechanism, and what each group below pins:

1. **The weight is really ``reach(h)^-kappa``.** Not "the acting player's reach",
   not "the policy's reach" — the probability the SAMPLER had of producing that
   history, chance included. Getting the chance factor wrong is invisible in any
   aggregate metric, so it is checked against a hand-computed constant.
2. **kappa=0 changes nothing**, down to the emitted batch: no weights array, so
   every update rule keeps its original reduction (the golden fixture path).
3. **The weight reaches the loss**, all four terms of it, and the batch's
   effective sample size is reported so the variance it costs is visible.
4. **Nothing swallows it silently**: the tabular rules refuse a weighted batch
   and an ACH run says out loud that it is no longer the paper's objective.

Offline evidence that this weighting is the right one:
docs/liars_residual_floor.md §8.4-8.5.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mjai.agents.mlp import MLPSharedActorCritic
from mjai.agents.tabular import TabularPolicy
from mjai.algos.nn_losses import value_loss_and_entropy, weight_telemetry, weighted_mean
from mjai.algos.nn_updates import NNActorCriticUpdate
from mjai.algos.transition import Transition, make_batch
from mjai.algos.update_rule import ACHFidelityWarning, AlgoConfig
from mjai.games.loader import load_game
from mjai.pipeline.rollout import RolloutConfig, RolloutWorkerCore

# Kuhn deals one of 3 cards then one of the remaining 2, so every history's
# chance prefix is exactly 1/6 and the first decision point's sampling reach is
# that and nothing else. A weight that omitted the chance factor would read 1.0.
KUHN_CHANCE_REACH = 1.0 / 6.0


def _kuhn_run(**cfg) -> tuple[TabularPolicy, np.ndarray, np.ndarray]:
    """One Kuhn episode under a non-uniform tabular policy; returns the batch arrays."""
    spec = load_game("kuhn")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0)
    worker = RolloutWorkerCore(
        spec, config=RolloutConfig(n_episodes=1, target_samples=None, seed=11, **cfg)
    )
    batch = worker.run_episode(policy, policy)
    return policy, batch.logprobs, batch.weights


# --------------------------------------------------------------------------
# 1. the weight is the inverse sampling reach
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kappa", [0.5, 1.0])
def test_weight_is_the_inverse_sampling_reach(kappa: float):
    """``w_i = reach(h_i)^-kappa``, chance included, accumulated along the episode.

    Checked structurally rather than against a recomputed reach: within one
    episode the transitions are in temporal order, so each step's reach is the
    previous step's times the action probability that was actually sampled --
    which the batch already carries as ``logprobs``. The chain therefore pins
    the chance prefix (step 0), the per-step accumulation, and the exponent,
    with no reimplementation of the quantity under test.
    """
    _policy, logprobs, weights = _kuhn_run(sample_weight_kappa=kappa)
    assert weights is not None and len(weights) >= 2, "need a multi-step episode"
    assert weights[0] == pytest.approx(KUHN_CHANCE_REACH**-kappa, rel=1e-6)
    for i in range(1, len(weights)):
        step = math.exp(float(logprobs[i - 1])) ** -kappa
        assert weights[i] == pytest.approx(float(weights[i - 1]) * step, rel=1e-5)


def test_weight_grows_as_histories_get_rarer():
    """Monotone in depth: every extra action can only shrink the reach."""
    _policy, _logprobs, weights = _kuhn_run(sample_weight_kappa=1.0)
    assert all(weights[i] >= weights[i - 1] for i in range(1, len(weights)))


def test_clip_caps_the_weight():
    """The cap is what bounds the per-batch variance; it must actually bind."""
    _p, _lp, uncapped = _kuhn_run(sample_weight_kappa=1.0)
    cap = float(uncapped.max()) / 2.0
    _p, _lp, capped = _kuhn_run(sample_weight_kappa=1.0, sample_weight_clip=cap)
    assert capped.max() == pytest.approx(cap, rel=1e-6)
    assert (capped <= cap * (1 + 1e-6)).all()


# --------------------------------------------------------------------------
# 2. kappa=0 changes nothing
# --------------------------------------------------------------------------


def test_kappa_zero_emits_no_weights_and_leaves_the_batch_identical():
    """The default path must not even allocate a weights array."""
    _p, lp_off, w_off = _kuhn_run()
    _p, lp_on, w_on = _kuhn_run(sample_weight_kappa=0.0)
    assert w_off is None and w_on is None
    assert np.array_equal(lp_off, lp_on)  # same RNG stream, same episode


def test_weights_survive_producer_routing():
    """``Batch.for_producer`` slices the weights alongside everything else.

    A weight that got dropped at the routing boundary would silently disable
    the whole mechanism for league runs while leaving mirror runs working.
    """
    spec = load_game("kuhn")
    a, b = (
        TabularPolicy(num_actions=spec.num_actions, seed=0),
        TabularPolicy(num_actions=spec.num_actions, seed=1),
    )
    worker = RolloutWorkerCore(
        spec,
        config=RolloutConfig(n_episodes=4, target_samples=None, seed=3, sample_weight_kappa=1.0),
    )
    batch = worker.run_episode(a, b)
    sub = batch.for_producer(a)
    assert sub.weights is not None and len(sub.weights) == sub.size < batch.size


# --------------------------------------------------------------------------
# 3. the weight reaches the loss
# --------------------------------------------------------------------------


def test_weighted_mean_is_the_plain_mean_on_the_none_path():
    x = torch.tensor([1.0, -2.0, 3.5])
    assert weighted_mean(x, None) is not None
    assert float(weighted_mean(x, None)) == float(x.mean())


def test_weighted_mean_is_self_normalized():
    """Scaling every weight leaves the loss alone — only the ratios matter."""
    x = torch.tensor([1.0, -2.0, 3.5])
    w = torch.tensor([1.0, 4.0, 9.0])
    assert float(weighted_mean(x, w)) == pytest.approx(float(weighted_mean(x, 100.0 * w)), rel=1e-6)
    assert float(weighted_mean(x, w)) == pytest.approx(float((w * x).sum() / w.sum()), rel=1e-6)


def test_value_and_entropy_are_weighted_too():
    """Not just the policy term: a critic fit on the untempered distribution
    would supply the advantages the tempered policy term is aimed at."""
    logits = torch.tensor([[1.0, 0.0], [0.0, 3.0]])
    values = torch.tensor([0.5, -0.5])
    returns = torch.tensor([1.0, 1.0])
    mask = torch.ones(2, 2)
    w = torch.tensor([1.0, 99.0])
    v_flat, h_flat = value_loss_and_entropy(logits, values, returns, mask)
    v_w, h_w = value_loss_and_entropy(logits, values, returns, mask, w)
    assert float(v_w) != pytest.approx(float(v_flat), rel=1e-4)
    assert float(h_w) != pytest.approx(float(h_flat), rel=1e-4)


def _mlp_batch(weights: list[float] | None):
    ts = [
        Transition(
            obs=[0.1 * i, -0.2, 0.3, 0.4],
            legal_actions=[0, 1, 2],
            action=i % 3,
            logprob=math.log(0.3),
            value=0.1,
            reward=1.0,
            return_=1.0,
            advantage=1.0 - 0.5 * i,
            player=0,
            weight=1.0 if weights is None else weights[i],
        )
        for i in range(4)
    ]
    return make_batch(ts, num_actions=3)


def _ach(seed: int = 0) -> NNActorCriticUpdate:
    policy = MLPSharedActorCritic(
        obs_size=4, num_actions=3, hidden_sizes=(8,), seed=seed, device="cpu"
    )
    return NNActorCriticUpdate(policy, AlgoConfig(theta=1.0, learning_rate=0.1))


def test_weighted_batch_moves_the_policy_somewhere_else():
    """The end-to-end check: same data, different weights, different parameters."""
    flat = _ach()
    flat.step(_mlp_batch(None))
    tilted = _ach()
    tilted.step(_mlp_batch([1.0, 1.0, 1.0, 50.0]))
    a = flat.policy.policy_head.weight.detach()
    b = tilted.policy.policy_head.weight.detach()
    assert not torch.allclose(a, b, atol=1e-6)


def test_uniform_weights_reproduce_the_unweighted_step():
    """Weights that are all equal are the plain mean, self-normalization and all."""
    flat = _ach()
    flat.step(_mlp_batch(None))
    same = _ach()
    same.step(_mlp_batch([7.0, 7.0, 7.0, 7.0]))
    assert torch.allclose(
        flat.policy.policy_head.weight.detach(),
        same.policy.policy_head.weight.detach(),
        atol=1e-6,
    )


def test_effective_sample_size_is_reported():
    """``weight_effn`` is the cost side of the trade and must be visible."""
    stats = _ach().step(_mlp_batch([1.0, 1.0, 1.0, 50.0]))
    assert 1.0 <= stats.extra["weight_effn"] < 4.0
    assert stats.extra["weight_max_ratio"] == pytest.approx(50.0, rel=1e-4)
    assert "weight_effn" not in _ach().step(_mlp_batch(None)).extra


def test_weight_telemetry_is_empty_when_unweighted():
    assert weight_telemetry(None) == {}


# --------------------------------------------------------------------------
# 4. nothing swallows it silently
# --------------------------------------------------------------------------


def test_tabular_rules_refuse_a_weighted_batch():
    """Both of them, including the CFR+ wrapper that never reads the batch."""
    from mjai.algos.tabular_updates import TabularACHUpdate, TabularPPOUpdate

    spec = load_game("kuhn")
    rules = [
        TabularPPOUpdate(TabularPolicy(num_actions=3, seed=0), AlgoConfig()),
        TabularACHUpdate(TabularPolicy(num_actions=spec.num_actions, seed=0), spec, AlgoConfig()),
    ]
    for rule in rules:
        with pytest.raises(NotImplementedError, match="sample_weight_kappa"):
            rule.step(_mlp_batch([1.0, 2.0, 3.0, 4.0]))


def test_sample_weighting_warns_when_ach_has_weight():
    from mjai.scripts.experiment_build import ExperimentConfig, warn_if_rollout_ach_incompatible

    cfg = ExperimentConfig(
        game="kuhn", algo="ach", self_play_mode="mirror", sample_weight_kappa=0.5
    )
    with pytest.warns(ACHFidelityWarning, match="sample_weight_kappa"):
        warn_if_rollout_ach_incompatible(cfg)
