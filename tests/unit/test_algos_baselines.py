"""Unit tests for the baseline solvers (AGENTS.md §5, §3).

These exercise the OpenSpiel reference solvers we wrap. They're marked
``slow`` because CFR+ runs for many iterations; the fast commit suite skips
them (per pyproject [tool.pytest] markers).
"""

from __future__ import annotations

import math

import numpy as np
import pyspiel
import pytest

from mjai.algos.baselines import (
    BRPS_EXACT_NASH,
    exact_nash_brps,
    solve_cfr_plus,
    solve_minimax,
    total_variation_distance,
)


def test_brps_exact_nash_is_valid_probability_vector():
    p = exact_nash_brps()
    assert p.shape == (3,)
    assert math.isclose(float(p.sum()), 1.0, abs_tol=1e-12)
    assert (p >= 0).all()


def test_brps_exact_nash_matches_known_values():
    # Known analytic Nash for biased RPS: (1/16, 10/16, 5/16).
    p = exact_nash_brps()
    expected = np.array([1, 10, 5]) / 16
    assert math.isclose(float(total_variation_distance(p, expected)), 0.0, abs_tol=1e-12)


def test_brps_constant_matches_module_level():
    assert exact_nash_brps() is not BRPS_EXACT_NASH  # returns a copy
    assert np.allclose(exact_nash_brps(), BRPS_EXACT_NASH)


@pytest.mark.slow
def test_cfr_plus_solves_kuhn_near_nash():
    """CFR+ on Kuhn should reach low exploitability after a few hundred iters."""
    game = pyspiel.load_game("kuhn_poker")
    sol = solve_cfr_plus(game, iterations=500)
    # Kuhn exploitability after CFR+ converges toward 0.
    assert sol.value is not None
    assert sol.value < 0.1  # well below the random-policy baseline
    # Strategy is populated for at least the 5 distinct info sets.
    assert len(sol.info_set_strategy) >= 5


@pytest.mark.slow
def test_cfr_plus_strategies_are_valid_distributions():
    game = pyspiel.load_game("kuhn_poker")
    sol = solve_cfr_plus(game, iterations=200)
    for info_state, action_probs in sol.info_set_strategy.items():
        total = sum(action_probs.values())
        assert math.isclose(total, 1.0, abs_tol=1e-6), f"{info_state}: sums to {total}"
        assert all(p >= 0.0 for p in action_probs.values())


def test_minimax_solves_tic_tac_toe_to_draw():
    """Tic-Tac-Toe minimax value is 0 (draw)."""
    game = pyspiel.load_game("tic_tac_toe")
    sol = solve_minimax(game)
    assert sol.value is not None
    assert math.isclose(sol.value, 0.0, abs_tol=1e-6)


def test_total_variation_distance_extremes():
    identical = total_variation_distance(np.array([0.5, 0.5]), np.array([0.5, 0.5]))
    disjoint = total_variation_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert math.isclose(identical, 0.0)
    assert math.isclose(disjoint, 1.0)


def test_total_variation_distance_symmetric():
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.4, 0.4, 0.2])
    d1 = total_variation_distance(p, q)
    d2 = total_variation_distance(q, p)
    assert math.isclose(d1, d2)
