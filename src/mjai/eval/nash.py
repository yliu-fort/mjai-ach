"""Equilibrium-distance metrics over a trained policy (AGENTS.md §3, Step 7).

Thin wrappers over OpenSpiel's solvers that turn a mjai :class:`Policy` into the
format OpenSpiel expects, then compute:
  - :func:`exploitability_of`  — for turn-based 2p0-sum games (Kuhn, Leduc,
    Liar's-Dice). Calls open_spiel.exploitability.exploitability.
  - :func:`nash_conv_of`       — for any game incl. simultaneous (Goofspiel,
    Oshi-Zumo, BRPS). Calls open_spiel.exploitability.nash_conv.
  - :func:`distance_to_brps_nash` — TV distance to the analytic BRPS NE.

The mjai Policy → OpenSpiel Policy adapter lives in :class:`_PolicyAdapter`.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pyspiel
from open_spiel.python import policy as ospolicy

from mjai.agents.base import Policy
from mjai.algos.baselines import BRPS_EXACT_NASH, total_variation_distance
from mjai.games.loader import GameSpec


class _PolicyAdapter(ospolicy.Policy):  # type: ignore[misc]
    """Adapts a mjai Policy into the open_spiel.python.policy.Policy interface
    expected by the OpenSpiel eval routines (exploitability / nash_conv).

    We only need ``action_probabilities`` for the eval use case.
    """

    def __init__(self, game: pyspiel.Game, policy: Policy) -> None:
        super().__init__(game, list(range(game.num_players())))
        self._policy = policy

    def action_probabilities(
        self, state: pyspiel.State, player_id: int | None = None
    ) -> dict[int, float]:
        p = state.current_player() if player_id is None else player_id
        # Use the game's observation encoding to match how the policy was trained.
        # information_state_tensor is what info-state-trained policies expect.
        try:
            obs = state.information_state_tensor(p)
        except Exception:
            obs = state.observation_tensor(p)
        obs_f = [float(x) for x in obs]
        legal = list(state.legal_actions(p))
        logits = self._policy.action_logits(obs_f, legal)
        # Softmax over legal actions only.
        mx = max(logits) if logits else 0.0
        exps = [np.exp(lg - mx) for lg in logits]
        s = sum(exps) or 1.0
        probs = {a: float(e / s) for a, e in zip(legal, exps, strict=True)}
        return probs


def exploitability_of(spec: GameSpec, policy: Policy) -> float:
    """Exploitability of ``policy`` in a 2p0-sum turn-based game.

    Raises ValueError for simultaneous or non-2p games (use nash_conv_of then).
    """
    if spec.is_simultaneous:
        raise ValueError(
            f"exploitability requires a turn-based game; {spec.name} is simultaneous. "
            f"Use nash_conv_of instead."
        )
    if spec.num_players != 2:
        raise ValueError(
            f"exploitability requires exactly 2 players; {spec.name} has {spec.num_players}."
        )
    from open_spiel.python.algorithms import exploitability

    adapter = _PolicyAdapter(spec.game, policy)
    return float(exploitability.exploitability(spec.game, adapter))


def nash_conv_of(spec: GameSpec, policy: Policy) -> float:
    """NashConv of ``policy`` — works for any game incl. simultaneous."""
    from open_spiel.python.algorithms import exploitability

    adapter = _PolicyAdapter(spec.game, policy)
    return float(exploitability.nash_conv(spec.game, adapter))


def distance_to_brps_nash(policy: Policy, *, num_actions: int = 3) -> float:
    """Total-variation distance between ``policy``'s BRPS mixed strategy and
    the analytic Nash equilibrium (1/16, 10/16, 5/16).

    The policy's strategy is its action distribution on BRPS's trivial
    initial observation ([0.0]); pass ``num_actions`` = 3.
    """
    # BRPS's single observation is the zero vector; get the policy's probs there.
    obs = [0.0]
    legal = list(range(num_actions))
    logits = policy.action_logits(obs, legal)
    mx = max(logits)
    exps = [np.exp(lg - mx) for lg in logits]
    s = sum(exps) or 1.0
    p = np.array([e / s for e in exps], dtype=np.float64)
    return total_variation_distance(p, BRPS_EXACT_NASH)


def best_metric_for(spec: GameSpec) -> str:
    """Return the most informative equilibrium metric available for ``spec``.

    'exploitability' for turn-based 2p0-sum, 'nash_conv' otherwise,
    'exact_nash_brps' for BRPS specifically.
    """
    if spec.name == "brps":
        return "exact_nash_brps"
    if not spec.is_simultaneous and spec.num_players == 2 and spec.is_zero_sum:
        return "exploitability"
    return "nash_conv"


def evaluate_equilibrium(spec: GameSpec, policy: Policy) -> dict[str, float]:
    """Run whichever equilibrium metric(s) apply, return as a dict.

    Always returns nash_conv when computable; adds exploitability for
    turn-based games and exact_nash_distance for BRPS.
    """
    out: dict[str, float] = {}
    if spec.name == "brps":
        out["exact_nash_distance"] = distance_to_brps_nash(policy, num_actions=spec.num_actions)
    # Some games / policy combos may not be evaluable by nash_conv; skip silently.
    with contextlib.suppress(Exception):
        out["nash_conv"] = nash_conv_of(spec, policy)
    if not spec.is_simultaneous and spec.num_players == 2:
        with contextlib.suppress(Exception):
            out["exploitability"] = exploitability_of(spec, policy)
    return out
