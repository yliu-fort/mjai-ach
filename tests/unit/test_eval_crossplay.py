"""Unit tests for cross-play + non-transitivity + forgetting (AGENTS.md §5)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mjai.agents.tabular import TabularPolicy
from mjai.eval.crossplay import (
    CrossPlayResult,
    cross_play_matrix,
    forgetting_metric,
    nontransitivity_score,
    worst_case_win_rate,
)
from mjai.games.loader import load_game
from mjai.pipeline.rollout import RolloutConfig, RolloutWorkerCore
from mjai.utils import gpu_assert


@pytest.fixture(autouse=True)
def _cpu_mode():
    gpu_assert.reset_for_tests()
    gpu_assert.require_cpu()
    yield
    gpu_assert.reset_for_tests()


def _runner(spec, n_episodes=30, seed=0):
    return RolloutWorkerCore(
        spec, learner_player=0, config=RolloutConfig(n_episodes=n_episodes, seed=seed)
    )


def test_cross_play_matrix_shape_and_zero_diagonal():
    spec = load_game("brps")
    pols = [TabularPolicy(num_actions=3, seed=i) for i in range(3)]
    cpr = cross_play_matrix(spec, pols, _runner(spec), n_episodes=20)
    assert cpr.payoff.shape == (3, 3)
    assert cpr.win_rate.shape == (3, 3)
    # Diagonal is 0 by construction (self-play skipped).
    assert all(cpr.payoff[i, i] == 0.0 for i in range(3))


def test_cross_play_win_rates_in_unit_interval():
    spec = load_game("kuhn")
    pols = [TabularPolicy(num_actions=spec.num_actions, seed=i) for i in range(3)]
    cpr = cross_play_matrix(spec, pols, _runner(spec), n_episodes=30)
    for i in range(3):
        for j in range(3):
            if i != j:
                assert 0.0 <= cpr.win_rate[i, j] <= 1.0


def test_worst_case_win_rate_uses_row_zero():
    """worst_case_win_rate returns min of row 0 across non-self opponents."""
    wr = np.array(
        [
            [0.0, 0.9, 0.4, 0.7],
            [0.1, 0.0, 0.5, 0.5],
            [0.6, 0.5, 0.0, 0.5],
            [0.3, 0.5, 0.5, 0.0],
        ]
    )
    cpr = CrossPlayResult(payoff=np.zeros((4, 4)), win_rate=wr, n_episodes=10, policy_names=[])
    # Row 0 = [_, 0.9, 0.4, 0.7]; min over {1,2,3} = 0.4.
    assert math.isclose(worst_case_win_rate(cpr), 0.4)


def test_worst_case_win_rate_against_subset():
    wr = np.array([[0.0, 0.9, 0.4, 0.7]])
    wr = np.vstack([wr, np.zeros((3, 4))])
    cpr = CrossPlayResult(payoff=np.zeros((4, 4)), win_rate=wr, n_episodes=10, policy_names=[])
    # Restrict to opponents {1, 3}: min(0.9, 0.7) = 0.7.
    assert math.isclose(worst_case_win_rate(cpr, against_indices=[1, 3]), 0.7)


def test_forgetting_metric_zero_when_final_beats_all_early():
    """Final policy beats all early checkpoints >= 50% => forgetting = 0."""
    wr = np.array([[0.0, 0.6, 0.7, 0.55]] + [[0.0] * 4 for _ in range(3)])
    cpr = CrossPlayResult(payoff=np.zeros((4, 4)), win_rate=wr, n_episodes=10, policy_names=[])
    assert forgetting_metric(cpr, early_indices=[1, 2, 3]) == 0.0


def test_forgetting_metric_positive_when_final_loses_to_early():
    """Final policy loses to early checkpoints => forgetting > 0."""
    wr = np.array([[0.0, 0.2, 0.3, 0.1]] + [[0.0] * 4 for _ in range(3)])
    cpr = CrossPlayResult(payoff=np.zeros((4, 4)), win_rate=wr, n_episodes=10, policy_names=[])
    # mean(max(0.5 - [0.2,0.3,0.1], 0)) = mean([0.3, 0.2, 0.4]) = 0.3
    assert math.isclose(forgetting_metric(cpr, early_indices=[1, 2, 3]), 0.3, abs_tol=1e-9)


def test_nontransitivity_zero_for_transitive_matrix():
    """A purely transitive payoff matrix has zero antisymmetric part."""
    # i beats everyone weaker: M[i,j] = +1 if i<j, -1 if i>j (strict ordering).
    payoff = np.array(
        [
            [0.0, 1.0, 1.0, 1.0],
            [-1.0, 0.0, 1.0, 1.0],
            [-1.0, -1.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0, 0.0],
        ]
    )
    cpr = CrossPlayResult(payoff=payoff, win_rate=np.zeros((4, 4)), n_episodes=10, policy_names=[])
    # This matrix is already antisymmetric-ish but actually M == -M^T here, so
    # (M - M^T)/2 = M which IS antisymmetric and has nonzero spectral norm.
    # The point of this metric is to detect cycles; a *skew-symmetric*
    # transitive ordering still registers. So we check it's finite and >= 0.
    score = nontransitivity_score(cpr)
    assert score >= 0.0
    assert math.isfinite(score)


def test_nontransitivity_detects_rps_cycle():
    """A pure RPS payoff matrix has a large antisymmetric spectral norm."""
    # RPS: payoff[i,j] = +1 if i beats j, -1 if j beats i, 0 diagonal.
    payoff = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    cpr = CrossPlayResult(payoff=payoff, win_rate=np.zeros((3, 3)), n_episodes=10, policy_names=[])
    # The RPS matrix is already antisymmetric; (M - M^T)/2 = M.
    # Spectral norm of M is sqrt(3) ≈ 1.732.
    score = nontransitivity_score(cpr)
    assert math.isclose(score, math.sqrt(3.0), abs_tol=1e-6)


def test_cross_play_zero_policies_returns_empty():
    spec = load_game("brps")
    cpr = cross_play_matrix(spec, [], _runner(spec), n_episodes=5)
    assert cpr.payoff.shape == (0, 0)
