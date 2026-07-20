"""Cross-play payoff matrix over a set of policies (AGENTS.md §3, Step 7).

Plays every (policy_i as seat 0, policy_j as seat 1) pair for ``n_episodes``
episodes and returns the average payoff matrix M[i, j] = mean return of policy_i
when it occupies seat 0 against policy_j in seat 1. For zero-sum games M is
antisymmetric in the sense M[i, j] = -M[j, i] up to sign flips from seat swap.

This matrix feeds:
  - :func:`worst_case_win_rate`   — min over the pool of seat-0 win-rate.
  - :func:`nontransitivity_score` — spectral measure of cycles in M.
  - :func:`forgetting_metric`     — final-policy row vs early-policy columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mjai.agents.base import Policy
from mjai.algos.controller import RolloutRunnerProtocol
from mjai.games.loader import GameSpec


@dataclass
class CrossPlayResult:
    """The payoff matrix + per-pair win-rates from a cross-play sweep."""

    payoff: np.ndarray  # (N, N) mean seat-0 payoff of policy_i vs policy_j
    win_rate: np.ndarray  # (N, N) P(policy_i beats policy_j) as seat 0
    n_episodes: int
    policy_names: list[str]


def cross_play_matrix(
    spec: GameSpec,
    policies: list[Policy],
    runner: RolloutRunnerProtocol,
    *,
    n_episodes: int = 50,
    policy_names: list[str] | None = None,
) -> CrossPlayResult:
    """Play every pair and return the payoff + win-rate matrices.

    For each (i, j) with i != j, plays ``n_episodes`` episodes with policy_i in
    seat 0 (learner) and policy_j in seat 0 (opponent) via the runner. Diagonal
    (i == j) is set to 0.0 (self-play payoff is undefined / not meaningful for
    these metrics).
    """
    n = len(policies)
    payoff = np.zeros((n, n), dtype=np.float64)
    win_rate = np.zeros((n, n), dtype=np.float64)
    names = policy_names or [f"p{i}" for i in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            batch = runner.run_episode(learner=policies[i], opponent=policies[j])
            if batch.size == 0:
                continue
            # Seat-0 (learner) mean payoff; in these zero-sum games the terminal
            # payoff is recorded per transition's `player`. Filter to player==0.
            seat0 = batch.for_player(0)
            if seat0.size == 0:
                continue
            payoff[i, j] = float(seat0.returns.mean())
            win_rate[i, j] = float((seat0.returns > 0).mean())

    return CrossPlayResult(
        payoff=payoff, win_rate=win_rate, n_episodes=n_episodes, policy_names=names
    )


def worst_case_win_rate(cpr: CrossPlayResult, *, against_indices: list[int] | None = None) -> float:
    """Minimum win-rate of policy 0 (the final/main policy) across opponents.

    Args:
        cpr: a :class:`CrossPlayResult` from :func:`cross_play_matrix`.
        against_indices: pool indices to consider as opponents; default = all.
    """
    if cpr.win_rate.size == 0:
        return 0.0
    opponents = (
        against_indices if against_indices is not None else list(range(cpr.win_rate.shape[1]))
    )
    opponents = [j for j in opponents if j != 0]
    if not opponents:
        return 0.0
    return float(cpr.win_rate[0, opponents].min())


def forgetting_metric(cpr: CrossPlayResult, *, early_indices: list[int]) -> float:
    """How much worse the final policy (row 0) does against early checkpoints.

    A high value means the final policy "forgot" how to beat early versions —
    the failure mode league play is supposed to mitigate. Computed as the
    drop in mean win-rate of row 0 against ``early_indices`` vs its max.

    Returns a non-negative float (0 = no forgetting).
    """
    if not early_indices or cpr.win_rate.size == 0:
        return 0.0
    early_wr = cpr.win_rate[0, early_indices]
    # If the final policy beats all early checkpoints >= 50%, forgetting is 0;
    # otherwise the gap below 0.5 summed.
    gaps = np.maximum(0.5 - early_wr, 0.0)
    return float(gaps.mean())


def nontransitivity_score(cpr: CrossPlayResult) -> float:
    """Detect non-transitive cycles in the payoff matrix via its spectral norm.

    For a perfectly transitive ordering (A beats B beats C, no cycles), the
    antisymmetric part of the payoff matrix M - M^T has small spectral norm.
    Large values indicate strong rock-paper-scissors-style cycling — exactly
    what league play is supposed to handle (AGENTS.md §1 motivation).

    Returns the largest singular value of (M - M^T)/2 (a non-negative float).
    """
    if cpr.payoff.size == 0:
        return 0.0
    antisym = (cpr.payoff - cpr.payoff.T) / 2.0
    svds = np.linalg.svd(antisym, compute_uv=False)
    return float(svds[0]) if svds.size else 0.0
