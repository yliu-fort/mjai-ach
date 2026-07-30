"""Unit tests for the running-average policy tracker (AGENTS.md §5, D16).

The load-bearing test here is
:func:`test_reproduces_openspiel_cfr_plus_average_exactly`. Averaging
realization plans is claimed to *be* the CFR average strategy, not an
approximation of it; OpenSpiel computes that same object by a completely
different route (accumulating reach-weighted behaviour inside its own tree
walk). Agreement to ~1e-16 is therefore a real check on the claim, and it is
what licenses reading the resulting curve as "the quantity ACH's O(T^-1/2)
theorem is about".
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from open_spiel.python.algorithms import cfr

from mjai.agents.tabular import TabularPolicy
from mjai.eval.average_policy import (
    RealizationAverage,
    average_plan_is_normalized,
    behavior_of,
)
from mjai.games.loader import load_game
from mjai.seqform import plan as P
from mjai.seqform.tree import build_sequence_form


@pytest.fixture(scope="module")
def kuhn_sf():
    return build_sequence_form(load_game("kuhn"))


@pytest.fixture(scope="module")
def kuhn3_sf():
    return build_sequence_form(load_game("kuhn3"))


def _read(sf, os_policy) -> torch.Tensor:
    behavior = torch.zeros(sf.num_infosets, sf.max_actions, dtype=torch.float64)
    for row, key in enumerate(sf.infoset_keys):
        behavior[row] = torch.tensor(list(os_policy.policy_for_key(key)), dtype=torch.float64)
    return behavior


def _random(sf, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(sf.num_infosets, sf.max_actions, dtype=torch.float64, generator=generator)
    return P.behavior_from_logits(sf, logits)


# --------------------------------------------------------------------------- #
# The claim: averaging realization plans IS the CFR average strategy
# --------------------------------------------------------------------------- #


def test_reproduces_openspiel_cfr_plus_average_exactly(kuhn_sf):
    """Two independent routes to the same object must agree to float64."""
    spec = load_game("kuhn")
    solver = cfr.CFRPlusSolver(spec.game)
    tracker = RealizationAverage(kuhn_sf)
    for iteration in range(1, 51):
        # The policy current at the START of iteration t, weighted by t — the
        # alignment OpenSpiel uses internally.
        tracker.update(_read(kuhn_sf, solver.current_policy()), weight=float(iteration))
        solver.evaluate_and_update_policy()
    theirs = float(P.nash_conv(kuhn_sf, _read(kuhn_sf, solver.average_policy())))
    assert tracker.nash_conv() == pytest.approx(theirs, abs=1e-14)


def test_post_update_alignment_is_a_visible_off_by_one(kuhn_sf):
    """Guards the docstring's warning with a number rather than a caution.

    Folding in the post-update policy is the natural thing to write and is
    wrong; the error is ~0.6% relative, small enough to pass for noise.
    """
    spec = load_game("kuhn")
    solver = cfr.CFRPlusSolver(spec.game)
    tracker = RealizationAverage(kuhn_sf)
    for iteration in range(1, 51):
        solver.evaluate_and_update_policy()
        tracker.update(_read(kuhn_sf, solver.current_policy()), weight=float(iteration))
    theirs = float(P.nash_conv(kuhn_sf, _read(kuhn_sf, solver.average_policy())))
    assert tracker.nash_conv() != pytest.approx(theirs, abs=1e-9)
    assert abs(tracker.nash_conv() - theirs) < 1e-3  # wrong, but only subtly


def test_average_of_one_iterate_is_that_iterate(kuhn3_sf):
    behavior = _random(kuhn3_sf, seed=0)
    tracker = RealizationAverage(kuhn3_sf)
    tracker.update(behavior)
    assert torch.allclose(tracker.average_behavior(), behavior, atol=1e-12)
    assert tracker.nash_conv() == pytest.approx(float(P.nash_conv(kuhn3_sf, behavior)), abs=1e-12)


def test_average_of_a_repeated_iterate_is_that_iterate(kuhn_sf):
    """Averaging must be idempotent on a constant sequence, at any weights."""
    behavior = _random(kuhn_sf, seed=1)
    tracker = RealizationAverage(kuhn_sf)
    for weight in (1.0, 7.0, 0.25):
        tracker.update(behavior, weight=weight)
    assert torch.allclose(tracker.average_behavior(), behavior, atol=1e-12)


# --------------------------------------------------------------------------- #
# Structural invariants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["kuhn", "kuhn3", "leduc"])
def test_averaged_plans_are_still_realization_plans(name):
    """A convex combination of realization plans is one: empty sequence at 1,
    flow constraints intact. If this breaks, the recovered behaviour is not a
    strategy and its NashConv means nothing."""
    sf = build_sequence_form(load_game(name))
    tracker = RealizationAverage(sf)
    for seed in range(4):
        tracker.update(_random(sf, seed), weight=float(seed + 1))
    plans = tracker.average_plans()
    assert average_plan_is_normalized(sf, plans)
    recovered = tracker.average_behavior()
    P.validate_behavior(sf, recovered)  # must not raise
    # And the recovered behaviour must regenerate exactly the averaged plans.
    for player, plan in enumerate(P.realization_plans(sf, recovered)):
        assert torch.allclose(plan, plans[player], atol=1e-12)


def test_unreached_information_sets_get_a_uniform_row(kuhn_sf):
    """The stated convention, exercised.

    "Unreached" is about the player's OWN reach, not the opponent's: P0's
    ``0pb`` rows sit behind P0's own *pass* at ``0``, so a P0 that always bets
    can never be there. Those rows have no bearing on exploitability and must
    come back uniform rather than 0/0.
    """
    behavior = torch.zeros(kuhn_sf.num_infosets, kuhn_sf.max_actions, dtype=torch.float64)
    behavior[:, 1] = 1.0  # always bet, so P0's own pass-then-face-a-bet rows die
    tracker = RealizationAverage(kuhn_sf)
    tracker.update(behavior)
    recovered = tracker.average_behavior()
    assert bool(torch.isfinite(recovered).all())
    P.validate_behavior(kuhn_sf, recovered)
    for key in ("0pb", "1pb", "2pb"):
        assert recovered[kuhn_sf.row_of_key(0, key)].tolist() == [0.5, 0.5]
    # A row the player does reach keeps its actual behaviour.
    assert recovered[kuhn_sf.row_of_key(0, "0")].tolist() == [0.0, 1.0]


def test_metrics_omit_exploitability_at_three_players(kuhn3_sf):
    """No minimax value at n >= 3, so no exploitability key (D14 wording)."""
    tracker = RealizationAverage(kuhn3_sf)
    tracker.update(_random(kuhn3_sf, seed=2))
    metrics = tracker.metrics()
    assert "avg_nash_conv" in metrics
    assert "avg_exploitability" not in metrics
    assert metrics["avg_iterates"] == 1.0


def test_metrics_include_exploitability_at_two_players(kuhn_sf):
    tracker = RealizationAverage(kuhn_sf)
    tracker.update(_random(kuhn_sf, seed=3))
    metrics = tracker.metrics()
    assert metrics["avg_exploitability"] == pytest.approx(metrics["avg_nash_conv"] / 2)


def test_empty_tracker_refuses_to_answer(kuhn_sf):
    with pytest.raises(ValueError, match="no iterates"):
        RealizationAverage(kuhn_sf).average_plans()


def test_non_positive_weight_is_rejected(kuhn_sf):
    tracker = RealizationAverage(kuhn_sf)
    with pytest.raises(ValueError, match="weight must be positive"):
        tracker.update(_random(kuhn_sf, seed=4), weight=0.0)


def test_invalid_behavior_is_rejected_before_it_pollutes_the_average(kuhn_sf):
    tracker = RealizationAverage(kuhn_sf)
    bad = _random(kuhn_sf, seed=5)
    bad[0, 0] += 0.3
    with pytest.raises(P.InvalidBehaviorError):
        tracker.update(bad)
    assert tracker.num_updates == 0


# --------------------------------------------------------------------------- #
# The Policy bridge
# --------------------------------------------------------------------------- #


def test_behavior_of_matches_a_direct_query(kuhn_sf):
    """``behavior_of`` must agree with asking the policy one row at a time."""
    spec = load_game("kuhn")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0, init_logit_std=1.0)
    batched = behavior_of(kuhn_sf, policy)
    for row, obs in enumerate(kuhn_sf.infoset_observation.tolist()):
        legal = torch.nonzero(kuhn_sf.legal_mask[row]).flatten().tolist()
        logits = np.asarray(policy.action_logits(obs, legal), dtype=np.float64)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        assert batched[row, legal].numpy() == pytest.approx(probs, abs=1e-6)


def test_behavior_of_produces_a_valid_strategy_on_leduc():
    """Leduc mixes 2- and 3-action rows, so the mask genuinely matters here."""
    spec = load_game("leduc")
    sf = build_sequence_form(spec)
    policy = TabularPolicy(num_actions=spec.num_actions, seed=1, init_logit_std=0.5)
    P.validate_behavior(sf, behavior_of(sf, policy))
