"""Unit tests for realization plans, exact payoffs and best responses (§5, D12).

Three things are worth more than the rest here:

- **Lemma 1 made testable.** ``<w, x_p> == value_p`` exactly, for a random
  strategy. That identity is what licenses a *linear* critic in pACH
  (Generative-ach.md §2.1); if it ever stops holding, the algorithm's central
  modelling assumption has broken and every downstream result is suspect.
- **The alpha family.** NashConv must be 0 across the whole known equilibrium
  segment and the game value must be exactly -1/18 (Kuhn 1950, 研究计划 §2.8).
- **D15.** An invalid policy must raise. OpenSpiel returns -6.7e-2 for the same
  input; a metric that answers for an impossible policy is worse than one that
  refuses.
"""

from __future__ import annotations

import pytest
import torch

from mjai.games.loader import load_game
from mjai.seqform import plan as P
from mjai.seqform.tree import EMPTY_SEQUENCE, build_sequence_form

# The Kuhn equilibrium family (研究计划 §2.8): P1 bets J with probability alpha,
# Q never, K with 3*alpha; after check-then-bet calls Q with alpha + 1/3, K
# always, folds J. P2's side is unique. Valid for alpha in [0, 1/3] — the upper
# end is exactly where 3*alpha saturates at probability 1.
KUHN_ALPHA_MAX = 1.0 / 3.0
KUHN_GAME_VALUE = -1.0 / 18.0


@pytest.fixture(scope="module")
def kuhn_sf():
    return build_sequence_form(load_game("kuhn"))


@pytest.fixture(scope="module")
def kuhn3_sf():
    return build_sequence_form(load_game("kuhn3"))


def kuhn_alpha_behavior(sf, alpha: float) -> torch.Tensor:
    """The alpha-family behaviour strategy, in this module's row order."""
    bet = {
        "0": alpha,
        "1": 0.0,
        "2": 3 * alpha,
        "0pb": 0.0,
        "1pb": alpha + 1 / 3,
        "2pb": 1.0,
        "0b": 0.0,
        "1b": 1 / 3,
        "2b": 1.0,
        "0p": 1 / 3,
        "1p": 0.0,
        "2p": 1.0,
    }
    behavior = torch.zeros(sf.num_infosets, sf.max_actions, dtype=torch.float64)
    for row, key in enumerate(sf.infoset_keys):
        p_bet = bet[key]
        behavior[row, 0] = 1.0 - p_bet
        behavior[row, 1] = p_bet
    return behavior


def random_behavior(sf, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(sf.num_infosets, sf.max_actions, dtype=torch.float64, generator=generator)
    return P.behavior_from_logits(sf, logits)


# --------------------------------------------------------------------------- #
# Realization plans
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["kuhn", "kuhn3", "leduc"])
def test_empty_sequence_has_realization_one(name):
    sf = build_sequence_form(load_game(name))
    for plan in P.realization_plans(sf, random_behavior(sf, seed=0)):
        assert float(plan[EMPTY_SEQUENCE]) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("name", ["kuhn", "kuhn3", "leduc"])
def test_realization_plans_satisfy_the_flow_constraints(name):
    """x(parent(I)) == sum over legal a of x(I, a), at every information set.

    This is the sequence-form polytope's defining equality. It is also what
    makes the polytope affine rather than full-dimensional, which is why a
    ridge critic on these coordinates is rank-deficient and why comparing
    critic coefficients requires projecting first (研究计划 §6.2 M1).
    """
    sf = build_sequence_form(load_game(name))
    plans = P.realization_plans(sf, random_behavior(sf, seed=1))
    for player in range(sf.num_players):
        rows = sf.rows_of(player)
        for row in rows.tolist():
            children = sf.sequence_of[row][sf.legal_mask[row]]
            got = float(plans[player][children].sum())
            want = float(plans[player][sf.parent_sequence[row]])
            assert got == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize("name", ["kuhn", "kuhn3", "leduc"])
def test_realization_probabilities_are_in_the_unit_interval(name):
    sf = build_sequence_form(load_game(name))
    for plan in P.realization_plans(sf, random_behavior(sf, seed=2)):
        assert bool(((plan >= -1e-15) & (plan <= 1 + 1e-15)).all())


# --------------------------------------------------------------------------- #
# Lemma 1: the payoff is exactly linear in one player's realization plan
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["kuhn", "kuhn3", "leduc"])
def test_payoff_is_exactly_linear_in_each_players_plan(name):
    """<w_p, x_p> == value_p. The premise of the whole linear-critic design."""
    sf = build_sequence_form(load_game(name))
    plans = P.realization_plans(sf, random_behavior(sf, seed=3))
    values = P.expected_returns(sf, plans)
    for player in range(sf.num_players):
        coefficients = P.sequence_payoff_coefficients(sf, plans, player)
        assert float(torch.dot(coefficients, plans[player])) == pytest.approx(
            float(values[player]), abs=1e-12
        )


def test_linearity_holds_along_an_interpolation(kuhn_sf):
    """Mixing one player's plan must move the payoff exactly linearly.

    A stronger statement than the identity above: it rules out a coefficient
    vector that happens to be right at one point by accident.
    """
    plans_a = P.realization_plans(kuhn_sf, random_behavior(kuhn_sf, seed=4))
    plans_b = P.realization_plans(kuhn_sf, random_behavior(kuhn_sf, seed=5))
    coefficients = P.sequence_payoff_coefficients(kuhn_sf, plans_a, player=0)
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        mixed = [(1 - t) * plans_a[0] + t * plans_b[0], plans_a[1]]
        want = float(P.expected_returns(kuhn_sf, mixed)[0])
        assert float(torch.dot(coefficients, mixed[0])) == pytest.approx(want, abs=1e-12)


# --------------------------------------------------------------------------- #
# Known ground truth: the Kuhn alpha family
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("alpha", [0.0, 1 / 12, 1 / 6, 0.25, KUHN_ALPHA_MAX])
def test_alpha_family_is_an_equilibrium_with_value_minus_one_eighteenth(kuhn_sf, alpha):
    behavior = kuhn_alpha_behavior(kuhn_sf, alpha)
    assert abs(float(P.nash_conv(kuhn_sf, behavior))) < 1e-12
    plans = P.realization_plans(kuhn_sf, behavior)
    values = P.expected_returns(kuhn_sf, plans)
    assert float(values[0]) == pytest.approx(KUHN_GAME_VALUE, abs=1e-12)
    assert float(values[1]) == pytest.approx(-KUHN_GAME_VALUE, abs=1e-12)


def test_off_family_strategy_is_exploitable(kuhn_sf):
    """Sanity: NashConv must be positive somewhere, or the metric is vacuous."""
    assert float(P.nash_conv(kuhn_sf, random_behavior(kuhn_sf, seed=6))) > 1e-3


def test_nash_conv_is_non_negative_on_valid_policies(kuhn3_sf):
    for seed in range(5):
        assert float(P.nash_conv(kuhn3_sf, random_behavior(kuhn3_sf, seed=seed))) > 0


# --------------------------------------------------------------------------- #
# D15: invalid policies raise instead of returning a plausible number
# --------------------------------------------------------------------------- #


def test_alpha_beyond_one_third_is_rejected(kuhn_sf):
    """At alpha = 0.4 the K-bet probability is 1.2. OpenSpiel accepts that and
    returns NashConv = -6.7e-2; we refuse (AGENTS.md D15)."""
    with pytest.raises(P.InvalidBehaviorError):
        P.nash_conv(kuhn_sf, kuhn_alpha_behavior(kuhn_sf, 0.4))


def test_unnormalized_rows_are_rejected(kuhn_sf):
    behavior = random_behavior(kuhn_sf, seed=7)
    behavior[3, 0] += 0.1
    with pytest.raises(P.InvalidBehaviorError, match="not 1"):
        P.nash_conv(kuhn_sf, behavior)


def test_mass_on_an_illegal_action_is_rejected():
    sf = build_sequence_form(load_game("leduc"))
    behavior = random_behavior(sf, seed=8)
    illegal = torch.nonzero(~sf.legal_mask, as_tuple=False)[0]
    behavior[illegal[0], illegal[1]] = 0.5
    with pytest.raises(P.InvalidBehaviorError, match="illegal action"):
        P.nash_conv(sf, behavior)


def test_wrong_shape_is_rejected(kuhn_sf):
    with pytest.raises(P.InvalidBehaviorError, match="shape"):
        P.validate_behavior(kuhn_sf, torch.zeros(3, 3, dtype=torch.float64))


def test_behavior_from_logits_is_valid_by_construction(kuhn3_sf):
    behavior = P.behavior_from_logits(
        kuhn3_sf, torch.full((kuhn3_sf.num_infosets, kuhn3_sf.max_actions), 3.0)
    )
    P.validate_behavior(kuhn3_sf, behavior)  # must not raise


def test_behavior_from_logits_zeroes_illegal_actions():
    """Leduc has 2- and 3-action information sets, so the mask really bites."""
    sf = build_sequence_form(load_game("leduc"))
    logits = torch.full((sf.num_infosets, sf.max_actions), 3.0, dtype=torch.float64)
    behavior = P.behavior_from_logits(sf, logits)
    assert float(behavior.masked_select(~sf.legal_mask).abs().max()) == 0.0
    P.validate_behavior(sf, behavior)


# --------------------------------------------------------------------------- #
# Best response
# --------------------------------------------------------------------------- #


def test_best_response_beats_or_matches_the_incumbent(kuhn3_sf):
    plans = P.realization_plans(kuhn3_sf, random_behavior(kuhn3_sf, seed=9))
    values = P.expected_returns(kuhn3_sf, plans)
    for player in range(kuhn3_sf.num_players):
        best = float(P.best_response_value(kuhn3_sf, plans, player))
        assert best >= float(values[player]) - 1e-12


def test_best_response_dominates_many_random_deviations(kuhn_sf):
    """No sampled alternative strategy may beat the computed best response.

    A cheap independent check on the backward induction: the exact optimum has
    to be an upper bound on anything we can stumble into.
    """
    plans = P.realization_plans(kuhn_sf, random_behavior(kuhn_sf, seed=10))
    best = float(P.best_response_value(kuhn_sf, plans, player=0))
    for seed in range(40):
        deviation = P.realization_plans(kuhn_sf, random_behavior(kuhn_sf, seed=100 + seed))
        value = float(P.expected_returns(kuhn_sf, [deviation[0], plans[1]])[0])
        assert value <= best + 1e-12


def test_best_response_to_the_alpha_family_is_the_game_value(kuhn_sf):
    """Against an equilibrium, even a best responder gets exactly -1/18."""
    plans = P.realization_plans(kuhn_sf, kuhn_alpha_behavior(kuhn_sf, 1 / 6))
    assert float(P.best_response_value(kuhn_sf, plans, 0)) == pytest.approx(
        KUHN_GAME_VALUE, abs=1e-12
    )


# --------------------------------------------------------------------------- #
# Autograd: the Oracle track differentiates through the exact expectation
# --------------------------------------------------------------------------- #


def test_gradient_flows_from_payoff_back_to_logits(kuhn3_sf):
    generator = torch.Generator().manual_seed(11)
    logits = torch.randn(
        kuhn3_sf.num_infosets,
        kuhn3_sf.max_actions,
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    plans = P.realization_plans(kuhn3_sf, P.behavior_from_logits(kuhn3_sf, logits))
    P.expected_returns(kuhn3_sf, plans)[0].backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())
    assert float(logits.grad.norm()) > 0


def test_autograd_matches_a_finite_difference(kuhn_sf):
    """Oracle-track gradients are checked against the definition, not trusted."""
    generator = torch.Generator().manual_seed(12)
    base = torch.randn(
        kuhn_sf.num_infosets, kuhn_sf.max_actions, dtype=torch.float64, generator=generator
    )

    def value(logits: torch.Tensor) -> torch.Tensor:
        plans = P.realization_plans(kuhn_sf, P.behavior_from_logits(kuhn_sf, logits))
        return P.expected_returns(kuhn_sf, plans)[0]

    logits = base.clone().requires_grad_(True)
    value(logits).backward()
    grad = logits.grad
    assert grad is not None
    eps = 1e-6
    for row, col in ((0, 0), (5, 1), (9, 0)):
        bumped_up, bumped_down = base.clone(), base.clone()
        bumped_up[row, col] += eps
        bumped_down[row, col] -= eps
        numeric = (float(value(bumped_up)) - float(value(bumped_down))) / (2 * eps)
        assert float(grad[row, col]) == pytest.approx(numeric, abs=1e-7)
