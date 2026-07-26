"""Three-way parity of the exact evaluators (AGENTS.md D14, 研究计划 §5.0(5)).

The research plan asks for "bit-for-bit" agreement between the base ACH stack,
our own exact evaluator and OpenSpiel. AGENTS.md D14 amends that claim, and this
suite is the amended form:

- **What is compared:** three *evaluators*, on ONE fixed policy. Not three
  solvers against each other. A solver comparison would confound "do we compute
  NashConv the same way" with "do we converge to the same equilibrium", and at
  n >= 3 the second question has no clean answer at all.
- **Where CFR sits:** it is a *test policy generator*, not a reference solution.
  Multiplayer CFR carries no Nash guarantee, so the 3p Kuhn cases below use its
  output only because it is a non-trivial, near-degenerate strategy — exactly
  where catastrophic cancellation would expose an indexing bug.
- **Tolerance is tiered, not zero,** and the tiers are set by measurement rather
  than by guesswork (AGENTS.md D14):

  ===================================  ==================  ==================
  Pair                                 Measured 2026-07-26  Tolerance here
  ===================================  ==================  ==================
  seqform vs OpenSpiel                 0 or 1 ulp          1e-12 absolute
  seqform vs base stack, either
  backend                              <= 1.5e-14 rel      1e-11 relative
  base stack python vs C++ backend     exactly 0           1e-12 absolute
  ===================================  ==================  ==================

  The middle row used to read 6e-10 .. 1.8e-8, which was **not** backend noise —
  the two base-stack backends agreed with each other to the bit even then. It
  was ``Policy.action_logits_batch`` returning float32, capping the base stack's
  exact evaluator regardless of which best-response solver ran underneath. That
  handoff is float64 as of 2026-07-26 and the gap fell four orders of magnitude
  to pure float64 round-off. Tabular policies now agree with an independent
  implementation exactly; an NN's residual error is its own float32 weights,
  which is a property of the model rather than of the metric.

Run this as a regression, not once: it is the standing guarantee that Step-0
ground truth and the base stack's training-time curves are the same quantity.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pyspiel
import pytest
import torch
from open_spiel.python import policy as ospolicy
from open_spiel.python.algorithms import cfr, exploitability

from mjai.agents.tabular import TabularPolicy
from mjai.eval.nash import equilibrium_metrics_exact
from mjai.games.loader import GameSpec, load_game
from mjai.seqform import plan as P
from mjai.seqform.tree import build_sequence_form

# D14 tolerances. See the module docstring for how each was measured.
TOL_EXACT = 1e-12  # seqform vs OpenSpiel: both float64 end to end
# seqform vs base stack: float64 round-off, worst case 1.5e-14 relative over
# random / uniform / near-equilibrium policies on all three games. Three orders
# of headroom for a different platform's BLAS, and still five orders tighter
# than the float32-era bound this replaces.
TOL_BASE_STACK = 1e-11

PARITY_GAMES = ["kuhn", "kuhn3", "leduc"]


def _decision_points(spec: GameSpec) -> dict[str, tuple[int, list[int], list[float]]]:
    """info-state string -> (player, legal actions, observation vector).

    Walked here rather than borrowed from ``mjai.eval`` so the bridge between
    the three legs does not itself come from one of the legs.
    """
    out: dict[str, tuple[int, list[int], list[float]]] = {}
    stack = [spec.new_state()]
    while stack:
        state = stack.pop()
        if state.is_terminal():
            continue
        if not state.is_chance_node():
            player = state.current_player()
            key = state.information_state_string(player)
            if key not in out:
                out[key] = (
                    player,
                    list(state.legal_actions(player)),
                    spec.obs_tensor(state, player),
                )
        actions = (
            [a for a, _ in state.chance_outcomes()]
            if state.is_chance_node()
            else state.legal_actions()
        )
        stack.extend(state.child(a) for a in actions)
    return out


def _random_table(spec: GameSpec, seed: int) -> dict[str, list[float]]:
    """A strictly positive random behaviour strategy, keyed by info-state string."""
    rng = np.random.default_rng(seed)
    table: dict[str, list[float]] = {}
    for key, (_player, legal, _obs) in _decision_points(spec).items():
        weights = rng.uniform(0.15, 1.0, size=len(legal))
        weights /= weights.sum()
        row = [0.0] * spec.num_actions
        for action, weight in zip(legal, weights, strict=True):
            row[action] = float(weight)
        table[key] = row
    return table


def _cfr_table(spec: GameSpec, iterations: int) -> dict[str, list[float]]:
    """CFR+ average strategy as a NON-TRIVIAL TEST POLICY (never a reference)."""
    solver = cfr.CFRPlusSolver(spec.game)
    for _ in range(iterations):
        solver.evaluate_and_update_policy()
    average = solver.average_policy()
    table: dict[str, list[float]] = {}
    for key, (player, legal, _obs) in _decision_points(spec).items():
        probs = average.action_probabilities(_state_for(spec, key), player)
        row = [0.0] * spec.num_actions
        for action in legal:
            row[action] = float(probs.get(action, 0.0))
        table[key] = row
    return table


def _state_for(spec: GameSpec, info_state: str) -> pyspiel.State:
    """Any state in the information set named ``info_state``."""
    stack = [spec.new_state()]
    while stack:
        state = stack.pop()
        if state.is_terminal():
            continue
        if not state.is_chance_node():
            if state.information_state_string(state.current_player()) == info_state:
                return state
            stack.extend(state.child(a) for a in state.legal_actions())
        else:
            stack.extend(state.child(a) for a, _ in state.chance_outcomes())
    raise KeyError(info_state)


# --------------------------------------------------------------------------- #
# The three legs, each fed from the same table
# --------------------------------------------------------------------------- #


def _leg_seqform(spec: GameSpec, table: dict[str, list[float]]) -> float:
    sf = build_sequence_form(spec)
    behavior = torch.zeros(sf.num_infosets, sf.max_actions, dtype=torch.float64)
    for row, key in enumerate(sf.infoset_keys):
        behavior[row] = torch.tensor(table[key], dtype=torch.float64)
    return float(P.nash_conv(sf, behavior))


def _leg_openspiel(spec: GameSpec, table: dict[str, list[float]]) -> float:
    tabular = ospolicy.TabularPolicy(spec.game)
    for key, row in tabular.state_lookup.items():
        tabular.action_probability_array[row] = table[key]
    return float(exploitability.nash_conv(spec.game, tabular))


def _leg_base_stack(spec: GameSpec, table: dict[str, list[float]], backend: str) -> float:
    policy = TabularPolicy(num_actions=spec.num_actions)
    for key, (_player, legal, obs) in _decision_points(spec).items():
        row = policy.get_logits(obs)
        for action in legal:
            row[action] = float(np.log(table[key][action]))
    return float(equilibrium_metrics_exact(spec, policy, backend=backend)["nash_conv"])


# --------------------------------------------------------------------------- #
# Parity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", PARITY_GAMES)
@pytest.mark.parametrize("seed", [0, 1])
def test_three_evaluators_agree_on_a_random_policy(name, seed):
    spec = load_game(name)
    table = _random_table(spec, seed)
    ours = _leg_seqform(spec, table)
    assert ours == pytest.approx(_leg_openspiel(spec, table), abs=TOL_EXACT)
    assert ours == pytest.approx(_leg_base_stack(spec, table, "python"), rel=TOL_BASE_STACK)
    assert ours == pytest.approx(_leg_base_stack(spec, table, "cpp"), rel=TOL_BASE_STACK)


@pytest.mark.parametrize("name", PARITY_GAMES)
def test_three_evaluators_agree_on_a_uniform_policy(name):
    """The degenerate case, where an off-by-one in the sequence index is most
    likely to cancel out of a random policy but not out of a uniform one."""
    spec = load_game(name)
    points = _decision_points(spec)
    table = {
        key: [1.0 / len(legal) if a in legal else 0.0 for a in range(spec.num_actions)]
        for key, (_p, legal, _o) in points.items()
    }
    ours = _leg_seqform(spec, table)
    assert ours == pytest.approx(_leg_openspiel(spec, table), abs=TOL_EXACT)
    assert ours == pytest.approx(_leg_base_stack(spec, table, "python"), rel=TOL_BASE_STACK)


@pytest.mark.parametrize("name", PARITY_GAMES)
@pytest.mark.parametrize("seed", [0, 1])
def test_base_stack_backends_agree_with_each_other_exactly(name, seed):
    """Isolates the two axes: backend choice is NOT what costs the precision.

    If this ever starts failing while the seqform comparison above still passes
    at 1e-6, the C++ solver's hash-map ordering has become the dominant term and
    D14's reproducibility clause needs revisiting.
    """
    spec = load_game(name)
    table = _random_table(spec, seed)
    assert _leg_base_stack(spec, table, "python") == pytest.approx(
        _leg_base_stack(spec, table, "cpp"), abs=TOL_EXACT
    )


def test_exact_eval_receives_float64_logits():
    """Pins the *reason* ``TOL_BASE_STACK`` is allowed to be 1e-11.

    ``action_logits_batch`` is the exact evaluator's only window onto a policy.
    While it returned float32 the base stack sat ~1e-8 away from an independent
    float64 implementation, whatever solver ran underneath. Narrowing it again
    would reopen that gap silently — the parity assertions above would simply
    start failing with no indication of why — so the dtype contract is asserted
    here, next to the tolerance it justifies.
    """
    import numpy as np

    spec = load_game("kuhn")
    policy = TabularPolicy(num_actions=spec.num_actions)
    mask = np.ones((1, spec.num_actions), dtype=bool)
    out = policy.action_logits_batch(np.zeros((1, spec.obs_size), dtype=np.float32), mask)
    assert np.asarray(out).dtype == np.float64


def test_tabular_policies_now_agree_with_seqform_to_round_off():
    """The sharpest statement the float64 handoff buys: not close, but exact.

    A tabular policy holds Python floats, so once nothing downcasts them the
    base stack and an independently written float64 evaluator are computing the
    same real number by two routes. Anything above round-off here is a bug in
    one of them, not a precision budget.
    """
    for name in PARITY_GAMES:
        spec = load_game(name)
        table = _random_table(spec, seed=3)
        ours = _leg_seqform(spec, table)
        theirs = _leg_base_stack(spec, table, "python")
        assert abs(ours - theirs) / abs(ours) < 1e-13


@pytest.mark.parametrize("name", ["kuhn", "kuhn3"])
def test_three_evaluators_agree_near_equilibrium(name):
    """CFR+ output as a test policy: NashConv near 0 is where cancellation bites.

    For kuhn3 this is emphatically NOT a claim that CFR found a Nash
    equilibrium — multiplayer CFR has no such guarantee (D14). It is only a
    strategy that all three evaluators must score identically.
    """
    spec = load_game(name)
    table = _cfr_table(spec, iterations=60)
    ours = _leg_seqform(spec, table)
    assert ours == pytest.approx(_leg_openspiel(spec, table), abs=TOL_EXACT)
    assert ours == pytest.approx(_leg_base_stack(spec, table, "python"), rel=TOL_BASE_STACK)
    # And the test policy really is the hard case: close to, but not at, zero.
    assert 0.0 <= ours < 0.2


def test_cfr_on_two_player_kuhn_approaches_the_known_game_value():
    """The one place CFR IS a reference: 2p zero-sum. Anchors the test policy."""
    spec = load_game("kuhn")
    table = _cfr_table(spec, iterations=400)
    assert _leg_seqform(spec, table) < 1e-2


# --------------------------------------------------------------------------- #
# The reproducibility half of D14
# --------------------------------------------------------------------------- #

_REPRO_SNIPPET = """
import torch
from mjai.games.loader import load_game
from mjai.seqform import plan as P
from mjai.seqform.tree import build_sequence_form

spec = load_game("kuhn3")
sf = build_sequence_form(spec)
generator = torch.Generator().manual_seed(17)
logits = torch.randn(sf.num_infosets, sf.max_actions, dtype=torch.float64, generator=generator)
print(float(P.nash_conv(sf, P.behavior_from_logits(sf, logits))).hex())
"""


@pytest.mark.slow
def test_seqform_nash_conv_is_bit_reproducible_across_processes():
    """D14's stronger half: the python route must be bit-stable, not just close.

    ``eval/nash.py`` records that the C++ best-response solver moves by ~1 ulp
    between processes because it sums over a hash-map order. Step-0 ground truth
    cannot be built on a number that does that, so our own route is required to
    reproduce exactly — and that requirement is tested, not assumed.
    """
    runs = {
        subprocess.run(
            [sys.executable, "-c", _REPRO_SNIPPET],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for _ in range(3)
    }
    assert len(runs) == 1, f"nash_conv differed across processes: {runs}"
