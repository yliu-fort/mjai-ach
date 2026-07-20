"""Reference baselines from OpenSpiel (AGENTS.md §3).

These are **not** part of the training pipeline — they produce ground-truth
strategies/values for evaluating the learned policies. Thin wrappers over
OpenSpiel's solver implementations so we don't reimplement game theory.

  - :func:`solve_cfr_plus` — exact Nash for any 2p0-sum extensive-form game
    that provides information_state_string (Kuhn, Leduc, Liar's-Dice, ...).
  - :func:`solve_minimax` — minimax value + optimal play for perfect-info
    games (Tic-Tac-Toe).
  - :func:`exact_nash_brps` — the known analytic Nash for biased RPS:
    (1/16, 10/16, 5/16).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyspiel

# Biased RPS exact Nash equilibrium, action order (R, P, S).
# Verified against the matrix_brps payoff {0,-25,50; 25,0,-5; -50,5,0}.
BRPS_EXACT_NASH = np.array([1.0 / 16, 10.0 / 16, 5.0 / 16], dtype=np.float64)


@dataclass(frozen=True)
class NashSolution:
    """A reference solution for a game (strategy + optional value)."""

    game_name: str
    # Per-info-set strategy: {info_state_string: {action: prob}}.
    info_set_strategy: dict[str, dict[int, float]]
    # Game value for player 0 (None if not computed).
    value: float | None = None


def solve_cfr_plus(game: pyspiel.Game, *, iterations: int = 1000) -> NashSolution:
    """Run CFR+ to convergence and return the average-strategy Nash.

    Works on any 2p0-sum sequential game providing information_state_string.
    For simultaneous-move games (BRPS, Goofspiel, Oshi-Zumo), prefer
    :func:`solve_cfr_plus_simultaneous` or use NashConv for evaluation instead.
    """
    from open_spiel.python.algorithms import cfr

    solver = cfr.CFRPlusSolver(game)
    for _ in range(iterations):
        solver.evaluate_and_update_policy()
    avg_policy = solver.average_policy()

    # Walk all infostates to materialize the tabular strategy.
    info_set_strategy: dict[str, dict[int, float]] = {}
    state = game.new_initial_state()
    _collect_info_states(state, avg_policy, info_set_strategy, visited=set())

    value = None
    try:
        from open_spiel.python.algorithms import exploitability

        value = float(exploitability.exploitability(game, avg_policy))
    except Exception:  # not all games support exploitability (simultaneous, etc.)
        pass
    return NashSolution(
        game_name=game.get_type().short_name,
        info_set_strategy=info_set_strategy,
        value=value,
    )


def _collect_info_states(
    state: pyspiel.State,
    # OpenSpiel's average_policy() returns a pyspiel.Policy; its stubs are
    # incomplete so we type as Any to call .action_probabilities without noise.
    policy: Any,
    out: dict[str, dict[int, float]],
    visited: set[str],
) -> None:
    """DFS over the game tree collecting each info-state's average strategy."""
    if state.is_terminal():
        return
    if state.is_chance_node():
        for action, _prob in state.chance_outcomes():
            child = state.child(action)
            _collect_info_states(child, policy, out, visited)
        return
    if state.is_simultaneous_node():
        # CFR on simultaneous nodes is approximated; skip for the baseline walker.
        return
    player = state.current_player()
    key = state.information_state_string(player)
    if key not in visited:
        visited.add(key)
        legal = state.legal_actions(player)
        probs = policy.action_probabilities(state, player)
        out[key] = {a: float(probs.get(a, 0.0)) for a in legal}
    for action in state.legal_actions(player):
        child = state.child(action)
        _collect_info_states(child, policy, out, visited)


def solve_minimax(game: pyspiel.Game) -> NashSolution:
    """Minimax value + optimal first-action for perfect-info games.

    Used for Tic-Tac-Toe: the minimax value is a draw (0.0). Returns the value
    in :attr:`NashSolution.value`; info_set_strategy is empty (minimax is a
    full-state search, not an info-set strategy).
    """
    from open_spiel.python.algorithms import minimax

    value, _best_action = minimax.alpha_beta_search(game=game)
    return NashSolution(
        game_name=game.get_type().short_name,
        info_set_strategy={},
        value=float(value),
    )


def exact_nash_brps() -> np.ndarray:
    """The analytic Nash equilibrium for biased RPS: (1/16, 10/16, 5/16).

    Action order is (Rock, Paper, Scissors) per pyspiel's matrix_brps.
    """
    return BRPS_EXACT_NASH.copy()


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """TV distance between two probability vectors; 0 == identical, 1 == disjoint."""
    return float(0.5 * np.abs(p - q).sum())
