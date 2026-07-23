"""Unit tests for the equilibrium-distance eval wrappers (AGENTS.md §5, Step 7)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mjai.agents.mlp import MLPSharedActorCritic
from mjai.agents.tabular import TabularPolicy
from mjai.algos.baselines import BRPS_EXACT_NASH
from mjai.eval.nash import (
    _SKELETON_CACHE,
    best_metric_for,
    clear_skeleton_cache,
    distance_to_brps_nash,
    equilibrium_metrics_exact,
    evaluate_equilibrium,
    exploitability_of,
    nash_conv_of,
    tabular_view_of,
    use_cpp_backend,
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


# ---- fast route: one batched materialization instead of per-state queries ----


def _mlp(spec):
    from mjai.agents.mlp import MLPSharedActorCritic

    return MLPSharedActorCritic(
        obs_size=spec.obs_size, num_actions=spec.num_actions, hidden_sizes=(32,), seed=0
    )


@pytest.mark.parametrize("game", ["kuhn", "brps"])
def test_fast_route_agrees_with_reference_for_nn_policies(game: str):
    """The batched route must reproduce the per-state traversal's nash_conv.

    Not bit-for-bit: a batched float32 forward blocks differently than a
    one-row forward, so logits differ in the last ulp. The tolerance is many
    orders of magnitude below the metric's seed-to-seed spread.
    """
    spec = load_game(game)
    policy = _mlp(spec)
    reference = nash_conv_of(spec, policy)
    fast = equilibrium_metrics_exact(spec, policy)["nash_conv"]
    assert fast == pytest.approx(reference, rel=1e-6)


@pytest.mark.parametrize(
    "game",
    [
        "kuhn",
        "brps",
        # goofspiel5_ii costs ~13 s (reference traversal); the fast unit suite
        # must stay under 20 s total (AGENTS.md §5), so it rides with the
        # slow-marked larger games below.
        pytest.param("goofspiel5_ii", marks=pytest.mark.slow),
    ],
)
def test_fast_route_is_bit_identical_for_tabular_policies(game: str):
    """Dict lookups have no batching, so materialization must not move at all.

    Pinned against the PYTHON solver on purpose: the C++ best-response MDP is
    only reproducible to ~1 ulp across processes (see
    test_python_backend_is_bit_reproducible), so spanning it here would make
    this an intermittently-failing test rather than a statement about the
    materialization step.
    """
    spec = load_game(game)
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0, temperature=1.0)
    fast = equilibrium_metrics_exact(spec, policy, backend="python")["nash_conv"]
    assert fast == nash_conv_of(spec, policy)


@pytest.mark.slow
@pytest.mark.parametrize("game", ["leduc", "liars_dice1"])
def test_fast_route_agrees_with_reference_on_larger_games(game: str):
    spec = load_game(game)
    policy = _mlp(spec)
    assert equilibrium_metrics_exact(spec, policy)["nash_conv"] == pytest.approx(
        nash_conv_of(spec, policy), rel=1e-6
    )


def test_exploitability_is_derived_from_nash_conv_not_recomputed():
    """OpenSpiel: exploitability == NashConv / num_players for 2p constant-sum.

    Deriving it is what removes the second full-tree traversal per eval; this
    pins the identity against the reference implementation.
    """
    spec = load_game("kuhn")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0)
    metrics = equilibrium_metrics_exact(spec, policy)
    assert metrics["exploitability"] == metrics["nash_conv"] / 2
    assert metrics["exploitability"] == pytest.approx(exploitability_of(spec, policy), rel=1e-9)


def test_simultaneous_games_report_no_exploitability():
    """exploitability is undefined for simultaneous games (mjai.eval.nash contract)."""
    spec = load_game("brps")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0)
    assert "exploitability" not in equilibrium_metrics_exact(spec, policy)


def test_skeleton_cache_is_reused_and_clearable():
    """The enumeration is per game, paid once, and droppable (AGENTS.md §8)."""
    clear_skeleton_cache()
    spec = load_game("kuhn")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0)
    equilibrium_metrics_exact(spec, policy)
    first = _SKELETON_CACHE[spec.game_string]
    equilibrium_metrics_exact(spec, policy)
    assert _SKELETON_CACHE[spec.game_string] is first, "skeleton must be reused, not rebuilt"
    clear_skeleton_cache()
    assert spec.game_string not in _SKELETON_CACHE


def test_tabular_view_reflects_the_policy_it_was_built_from():
    """The shared skeleton must be rewritten per policy, never stale-aliased.

    The probability array is cached and overwritten in place, so this is the
    test that the cache cannot leak one policy's distribution into another
    policy's eval.
    """
    from mjai.agents.mlp import MLPSharedActorCritic

    spec = load_game("kuhn")
    one = MLPSharedActorCritic(
        obs_size=spec.obs_size, num_actions=spec.num_actions, hidden_sizes=(32,), seed=1
    )
    other = MLPSharedActorCritic(
        obs_size=spec.obs_size, num_actions=spec.num_actions, hidden_sizes=(32,), seed=2
    )
    first = tabular_view_of(spec, one).action_probability_array.copy()
    second = tabular_view_of(spec, other).action_probability_array.copy()
    assert not np.allclose(first, second), "different policies gave the same table"
    again = tabular_view_of(spec, one).action_probability_array.copy()
    assert np.array_equal(first, again), "rebuilding from the same policy must be stable"


# ---- exact-eval backend selection (python traversal vs C++ best-response MDP) ----


def test_backend_dispatch_is_by_game_type():
    """auto = C++ on turn-based games, Python on simultaneous ones."""
    assert use_cpp_backend(load_game("kuhn"), "auto") is True
    assert use_cpp_backend(load_game("leduc"), "auto") is True
    assert use_cpp_backend(load_game("brps"), "auto") is False
    assert use_cpp_backend(load_game("goofspiel5_ii"), "auto") is False
    # Explicit modes override the rule in both directions.
    assert use_cpp_backend(load_game("kuhn"), "python") is False
    assert use_cpp_backend(load_game("brps"), "cpp") is True


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown exact backend"):
        use_cpp_backend(load_game("kuhn"), "fortran")


@pytest.mark.parametrize("game", ["kuhn", "leduc"])
def test_cpp_backend_matches_the_python_traversal(game: str):
    """The fast solver must reproduce the reference traversal's nash_conv."""
    spec = load_game(game)
    policy = _mlp(spec)
    py = equilibrium_metrics_exact(spec, policy, backend="python")
    cpp = equilibrium_metrics_exact(spec, policy, backend="cpp")
    assert cpp["nash_conv"] == pytest.approx(py["nash_conv"], rel=1e-9)
    assert cpp["exploitability"] == pytest.approx(py["exploitability"], rel=1e-9)


def test_cpp_backend_survives_repeated_use():
    """Regression: the C++ solver does not own the policy it is handed.

    ``TabularBestResponseMDP`` keeps a raw reference, so passing the policy as
    a temporary lets Python free it under the solver — which SEGFAULTS the
    interpreter rather than raising. This loop (with forced collection between
    calls) crashes the whole suite if that lifetime bug ever comes back.
    """
    import gc

    spec = load_game("kuhn")
    values = []
    for seed in range(5):
        policy = MLPSharedActorCritic(
            obs_size=spec.obs_size, num_actions=spec.num_actions, hidden_sizes=(32,), seed=seed
        )
        values.append(equilibrium_metrics_exact(spec, policy, backend="cpp")["nash_conv"])
        gc.collect()
    assert all(math.isfinite(v) and v >= 0.0 for v in values)


def test_evaluate_equilibrium_forwards_the_backend_choice():
    spec = load_game("kuhn")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0)
    for backend in ("auto", "python", "cpp"):
        metrics = evaluate_equilibrium(spec, policy, exact_backend=backend)
        assert metrics["exploitability"] == pytest.approx(0.9166666667 / 2, rel=1e-6)


def test_python_backend_is_bit_reproducible_within_a_process():
    """The Python route is the one to use when exact reproducibility matters.

    The C++ best-response MDP sums over a hash-map iteration order that varies
    between processes, so its nash_conv can move by ~1 ulp (measured: 1 run in
    5 on Kuhn). That is ~1e-16 relative and irrelevant against seed-to-seed
    spread, but it means "same seed, same eval bits" only holds under
    eval_exact_backend="python".
    """
    spec = load_game("kuhn")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0, temperature=1.0)
    values = {
        equilibrium_metrics_exact(spec, policy, backend="python")["nash_conv"] for _ in range(5)
    }
    assert len(values) == 1


def test_cpp_and_python_backends_agree_to_floating_point_noise():
    """The two solvers must never differ by more than accumulation-order noise."""
    spec = load_game("kuhn")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0, temperature=1.0)
    cpp = equilibrium_metrics_exact(spec, policy, backend="cpp")["nash_conv"]
    py = equilibrium_metrics_exact(spec, policy, backend="python")["nash_conv"]
    assert cpp == pytest.approx(py, rel=1e-12)
