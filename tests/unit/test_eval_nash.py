"""Unit tests for the equilibrium-distance eval wrappers (AGENTS.md §5, Step 7)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mjai.agents.tabular import TabularPolicy
from mjai.algos.baselines import BRPS_EXACT_NASH
from mjai.eval.nash import (
    best_metric_for,
    distance_to_brps_nash,
    evaluate_equilibrium,
    exploitability_of,
    nash_conv_of,
)
from mjai.games.loader import load_game
from mjai.utils import gpu_assert


@pytest.fixture(autouse=True)
def _cpu_mode():
    gpu_assert.reset_for_tests()
    gpu_assert.require_cpu()
    yield
    gpu_assert.reset_for_tests()


def test_best_metric_for_each_game():
    assert best_metric_for(load_game("brps")) == "exact_nash_brps"
    assert best_metric_for(load_game("kuhn")) == "exploitability"
    assert best_metric_for(load_game("leduc")) == "exploitability"
    assert best_metric_for(load_game("liars_dice1")) == "exploitability"
    # Simultaneous games fall back to nash_conv.
    assert best_metric_for(load_game("goofspiel5_ii")) == "nash_conv"
    assert best_metric_for(load_game("oshi_zumo")) == "nash_conv"


def test_exploitability_rejects_simultaneous_games():
    spec = load_game("goofspiel5_ii")
    p = TabularPolicy(num_actions=spec.num_actions, seed=0)
    with pytest.raises(ValueError, match="turn-based"):
        exploitability_of(spec, p)


def test_exploitability_uniform_policy_on_kuhn_is_finite_and_positive():
    """A uniform-random Kuhn policy has positive exploitability."""
    spec = load_game("kuhn")
    p = TabularPolicy(num_actions=spec.num_actions, seed=0)
    expl = exploitability_of(spec, p)
    assert math.isfinite(expl)
    assert expl > 0.0


@pytest.mark.slow
def test_nash_conv_finite_on_brps():
    spec = load_game("brps")
    p = TabularPolicy(num_actions=spec.num_actions, seed=0)
    nc = nash_conv_of(spec, p)
    assert math.isfinite(nc)
    assert nc >= 0.0


def test_distance_to_brps_nash_uniform_policy():
    """A uniform BRPS policy is far from the analytic NE (1/16, 10/16, 5/16)."""
    p = TabularPolicy(num_actions=3, seed=0)
    d = distance_to_brps_nash(p, num_actions=3)
    # Uniform = (1/3, 1/3, 1/3); TV distance to (1/16, 10/16, 5/16) is nonzero.
    assert d > 0.0
    # And the analytic distance value is exactly computable.
    uniform = np.array([1 / 3, 1 / 3, 1 / 3])
    expected = 0.5 * float(np.abs(uniform - BRPS_EXACT_NASH).sum())
    assert math.isclose(d, expected, abs_tol=1e-6)


def test_distance_to_brps_nash_zero_for_exact_nash_policy():
    """A policy playing exactly the NE has TV distance 0."""
    p = TabularPolicy(num_actions=3, seed=0)
    # Set logits so softmax = BRPS_EXACT_NASH. logit ∝ log(prob).
    obs = [0.0]
    for a in range(3):
        p.get_logits(obs)[a] = math.log(BRPS_EXACT_NASH[a])
    d = distance_to_brps_nash(p, num_actions=3)
    assert math.isclose(d, 0.0, abs_tol=1e-5)


def test_evaluate_equilibrium_returns_at_least_one_metric():
    for name in ["brps", "kuhn", "goofspiel5_ii"]:
        spec = load_game(name)
        p = TabularPolicy(num_actions=spec.num_actions, seed=0)
        metrics = evaluate_equilibrium(spec, p)
        assert len(metrics) >= 1, f"{name} returned no metrics"
        for v in metrics.values():
            assert math.isfinite(v)
