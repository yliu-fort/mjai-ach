"""Unit tests for the sampled NashConv estimator (AGENTS.md §5).

Covers: accuracy vs exact values on Kuhn and tic-tac-toe (with documented,
justified tolerances), seed determinism, and loud validation of the new
config knobs. All tests are CPU-only, seeded, and deterministic.
"""

from __future__ import annotations

import math

import pytest

from mjai.agents.tabular import TabularPolicy
from mjai.eval.nash import evaluate_equilibrium, nash_conv_of
from mjai.eval.sampled_nash import MIN_MC_SAMPLES, sampled_nash_conv
from mjai.games.loader import load_game
from mjai.scripts.experiment import ExperimentConfig
from mjai.utils import gpu_assert


@pytest.fixture(autouse=True)
def _cpu_mode():
    gpu_assert.reset_for_tests()
    gpu_assert.require_cpu()
    yield
    gpu_assert.reset_for_tests()


def test_sampled_vs_exact_on_kuhn():
    """Sampled NashConv of the uniform Kuhn policy matches the exact value.

    Tolerance justification: 4* the reported standard error (the estimator's
    zero-mean Monte-Carlo error, per the module docstring) + 0.05 for the
    documented one-sided BR-approximation bias. Kuhn has only 12 infosets per
    player, all heavily sampled at this budget, so the approximate BR ≈ the
    exact BR and the bias term is tiny. Seed is fixed ⇒ deterministic.
    Calibration anchor at (mc_samples=1600, seed=7): est=0.7975, se=0.162,
    exact=0.9167 — gap 0.12 sits well inside 4·SE.
    """
    spec = load_game("kuhn")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0)  # uniform: zero logits
    exact = nash_conv_of(spec, policy)
    res = sampled_nash_conv(spec, policy, mc_samples=1600, seed=7)
    assert res.nash_conv_std > 0.0
    assert abs(res.nash_conv - exact) <= 4 * res.nash_conv_std + 0.05


def test_sampled_ttt_uniform_matches_reference():
    """Sampled NashConv of the uniform TTT policy vs the exact reference.

    Reference 1.9197 is the exact open_spiel nash_conv of the uniform policy
    (measured offline — the full-tree traversal takes ~24 s, too slow for
    this suite). The estimator is CONSERVATIVE by construction (module
    docstring: a mis-ranked argmax, an unvisited state, or too few
    improvement passes can only lower the approximate BR's value), so the
    check is asymmetric:
      - upper: never above exact + 4*SE — there is no systematic upward
        channel, only zero-mean MC error;
      - lower: at least ~half of exact recovered at this budget.
    Calibration anchor at (mc_samples=800, n_passes=4, seed=3): est=1.1325,
    se=0.134 — deterministic given the seed.
    """
    exact_ref = 1.9197
    spec = load_game("ttt")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0)
    res = sampled_nash_conv(spec, policy, mc_samples=800, seed=3, n_passes=4)
    assert res.nash_conv <= exact_ref + 4 * res.nash_conv_std
    assert res.nash_conv >= 0.9


def test_deterministic_same_seed_same_result():
    """Same seed + same policy ⇒ bit-identical output; different seed differs."""
    spec = load_game("kuhn")
    policy_a = TabularPolicy(num_actions=spec.num_actions, seed=0)
    policy_b = TabularPolicy(num_actions=spec.num_actions, seed=0)
    first = sampled_nash_conv(spec, policy_a, mc_samples=200, seed=42)
    second = sampled_nash_conv(spec, policy_b, mc_samples=200, seed=42)
    assert first == second
    other_seed = sampled_nash_conv(spec, policy_a, mc_samples=200, seed=43)
    assert other_seed != second


def test_invalid_eval_estimator_rejected_loudly():
    """AGENTS.md §9: a bad estimator name must error, never silently default."""
    with pytest.raises(ValueError, match="eval_estimator"):
        ExperimentConfig(game="kuhn", algo="ach", self_play_mode="mirror", eval_estimator="mc")
    spec = load_game("kuhn")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0)
    with pytest.raises(ValueError, match="unknown eval estimator"):
        evaluate_equilibrium(spec, policy, estimator="bogus")


def test_eval_mc_samples_floor_validated():
    with pytest.raises(ValueError, match="eval_mc_samples"):
        ExperimentConfig(game="kuhn", algo="ach", self_play_mode="mirror", eval_mc_samples=4)
    spec = load_game("kuhn")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0)
    with pytest.raises(ValueError, match="mc_samples"):
        sampled_nash_conv(spec, policy, mc_samples=MIN_MC_SAMPLES - 1, seed=0)


def test_evaluate_equilibrium_sampled_keys_compatible():
    """Sampled mode keeps the ``nash_conv`` key and adds a standard error.

    For 2p0-sum turn-based games it also reports ``exploitability`` via the
    exact identity exploitability = nash_conv / 2 (OpenSpiel's definition).
    """
    spec = load_game("kuhn")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0)
    out = evaluate_equilibrium(spec, policy, estimator="sampled", mc_samples=200, seed=1)
    assert "nash_conv" in out
    assert "nash_conv_std" in out
    assert "exploitability" in out
    assert "exploitability_std" in out
    assert math.isclose(out["exploitability"], out["nash_conv"] / 2.0)
    for v in out.values():
        assert math.isfinite(v)
        assert v >= 0.0
