"""Exact per-information-set values, and the oracle-baseline rollout arm.

``V(I)`` is the one quantity in this repo that a learned critic is *supposed* to
approximate, so it is worth more than an internal-consistency check. Three
independent handles:

1. **The root identity.** Summing ``V(I) * cf(I)`` over a player's level-0
   information sets must reproduce :func:`expected_returns` exactly -- the
   backward induction and the multilinear terminal sum meeting in the middle.
2. **Monte Carlo.** Play the game and average the realized returns per
   information set. This is the only check that also validates the observation
   keying, which is where an exact-looking value silently attaches to the wrong
   state.
3. **The rollout actually uses it**, and does not when it is off.
"""

from __future__ import annotations

import math

import pytest
import torch

from mjai.agents.tabular import TabularPolicy
from mjai.algos.update_rule import ACHFidelityWarning
from mjai.eval.exact_value import ExactValueOracle
from mjai.games.loader import load_game
from mjai.pipeline.rollout import RolloutConfig, RolloutWorkerCore
from mjai.seqform.plan import expected_returns, infoset_values, realization_plans
from mjai.seqform.tree import build_sequence_form
from mjai.utils import gpu_assert


@pytest.fixture(autouse=True)
def _cpu_mode():
    gpu_assert.reset_for_tests()
    gpu_assert.require_cpu()
    yield
    gpu_assert.reset_for_tests()


def _random_behavior(sf, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    logits = torch.randn(sf.num_infosets, sf.max_actions, dtype=torch.float64)
    return torch.softmax(logits.masked_fill(~sf.legal_mask, float("-inf")), dim=1)


# --------------------------------------------------------------------------
# 1. the root identity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("game", ["kuhn", "leduc", "liars_dice1"])
def test_root_value_reproduces_the_multilinear_expected_return(game: str):
    """``sum_I V(I) * cf(I)`` over level 0 == ``E[u_p]``, to float64."""
    sf = build_sequence_form(load_game(game))
    behavior = _random_behavior(sf)
    plans = realization_plans(sf, behavior)
    values, cf = infoset_values(sf, behavior, plans)
    returns = expected_returns(sf, plans)
    for player in range(sf.num_players):
        rows = torch.nonzero((sf.infoset_level == 0) & (sf.infoset_player == player)).flatten()
        got = float((values[rows] * cf[rows]).sum())
        assert got == pytest.approx(float(returns[player]), abs=1e-12)


def test_shut_out_information_sets_report_zero_rather_than_dividing_by_zero():
    """cf(I) == 0 has no conditional expectation; it must not produce inf/nan."""
    sf = build_sequence_form(load_game("kuhn"))
    behavior = _random_behavior(sf)
    # Make player 1 deterministic so some of its successors carry zero mass.
    rows = sf.rows_of(1)
    mask = sf.legal_mask[rows]
    hard = torch.zeros_like(behavior[rows])
    hard[torch.arange(rows.numel()), mask.to(torch.int64).argmax(dim=1)] = 1.0
    behavior = behavior.clone()
    behavior[rows] = hard
    values, cf = infoset_values(sf, behavior)
    assert (cf == 0).any(), "test needs at least one shut-out information set"
    assert torch.isfinite(values).all()
    assert float(values[cf == 0].abs().max()) == 0.0


# --------------------------------------------------------------------------
# 2. Monte Carlo — the check that also validates the observation keying
# --------------------------------------------------------------------------


def test_oracle_value_matches_the_realized_average_return():
    """Play Kuhn and average the returns per information set; V(I) must match.

    Independent of every line in ``infoset_values``: the estimate comes from
    ``RolloutWorkerCore`` actually playing the game. Only information sets with
    enough visits are compared -- the rest are estimating a mean from a handful
    of +-1 payoffs, which is the sampler's noise and not a disagreement.
    """
    spec = load_game("kuhn")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0)
    for key in list(policy.logits):
        policy.logits[key] = [0.7, -0.4]
    oracle = ExactValueOracle(spec)
    oracle.refresh(policy, policy)

    worker = RolloutWorkerCore(
        spec, config=RolloutConfig(n_episodes=20000, target_samples=None, seed=7)
    )
    batch = worker.run_episode(policy, policy)
    total: dict[tuple[float, ...], list[float]] = {}
    for i in range(batch.size):
        total.setdefault(tuple(float(x) for x in batch.obs[i]), []).append(float(batch.returns[i]))

    compared = 0
    for key, seen in total.items():
        if len(seen) < 2000:
            continue
        empirical = sum(seen) / len(seen)
        stderr = (sum((x - empirical) ** 2 for x in seen) / len(seen)) ** 0.5 / len(seen) ** 0.5
        assert abs(oracle.value(list(key)) - empirical) < 5 * stderr + 1e-9
        compared += 1
    assert compared >= 4, f"only {compared} information sets were well-visited enough"


# --------------------------------------------------------------------------
# 3. the rollout uses it, and does not when it is off
# --------------------------------------------------------------------------


def _worker(oracle) -> RolloutWorkerCore:
    return RolloutWorkerCore(
        load_game("kuhn"),
        config=RolloutConfig(n_episodes=3, target_samples=None, seed=3),
        value_oracle=oracle,
    )


def test_rollout_records_the_oracle_baseline_not_the_network_one():
    spec = load_game("kuhn")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0)
    oracle = ExactValueOracle(spec)
    batch = _worker(oracle).run_episode(policy, policy)
    for i in range(batch.size):
        obs = [float(x) for x in batch.obs[i]]
        assert float(batch.values[i]) == pytest.approx(oracle.value(obs), abs=1e-6)
    assert oracle.refreshes == 1  # once per round, not once per episode


def test_no_oracle_is_the_untouched_path():
    spec = load_game("kuhn")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0)
    plain = _worker(None).run_episode(policy, policy)
    assert all(float(v) == 0.0 for v in plain.values)  # a fresh tabular value table
    assert plain.size > 0


def test_refresh_rejects_a_two_policy_profile():
    """V depends on BOTH strategies; a league round would need two tables."""
    spec = load_game("kuhn")
    oracle = ExactValueOracle(spec)
    a = TabularPolicy(num_actions=spec.num_actions, seed=0)
    b = TabularPolicy(num_actions=spec.num_actions, seed=1)
    with pytest.raises(ValueError, match="mirror self-play"):
        oracle.refresh(a, b)


def test_unknown_observation_fails_loudly():
    oracle = ExactValueOracle(load_game("kuhn"))
    with pytest.raises(KeyError, match="information-set enumeration"):
        oracle.value([math.pi] * 11)


# --------------------------------------------------------------------------
# governance
# --------------------------------------------------------------------------


def test_oracle_value_warns_and_league_is_refused():
    import warnings

    from mjai.scripts.experiment_build import ExperimentConfig, warn_if_rollout_ach_incompatible

    cfg = ExperimentConfig(game="kuhn", algo="ach", self_play_mode="mirror", oracle_value=True)
    with pytest.warns(ACHFidelityWarning, match="oracle_value"):
        warn_if_rollout_ach_incompatible(cfg)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_rollout_ach_incompatible(
            ExperimentConfig(game="kuhn", algo="ach", self_play_mode="mirror")
        )


def test_league_mode_refuses_the_oracle_rather_than_using_the_wrong_profile():
    import random

    from mjai.scripts.experiment_build import ExperimentConfig, build_controller

    spec = load_game("kuhn")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0)
    cfg = ExperimentConfig(
        game="kuhn", algo="ach", self_play_mode="league", policy_kind="tabular", oracle_value=True
    )
    with pytest.warns(ACHFidelityWarning), pytest.raises(ValueError, match="oracle_value needs"):
        build_controller(spec, policy, cfg, rng=random.Random(0))
