"""Sampled NashConv estimator: Monte-Carlo approximate best response.

Why this exists: exact NashConv (``open_spiel.python.algorithms.exploitability.
nash_conv``) walks the FULL game tree. That is infeasible for
``oshi_zumo(coins=5,size=3,horizon=20)`` (the traversal hangs, verified
2026-07-22) and costs ~24 s per call on tic_tac_toe, which blows the per-eval
budget during training (a 12-eval run would add ~5 min inside one 300 s step).
This module estimates NashConv by trajectory sampling instead (AGENTS.md §4:
one metric = one module exposing a single entry function).

Estimator (2-player games only — all Phase-1 games are 2p). NashConv(π) =
Σ_i [ V_i(BR_i, π_-i) - V_i(π) ], approximated per player i by building an
approximate best response via ``n_passes`` rounds of SAMPLED POLICY ITERATION
on the best-response MDP (hero maximizes return against the fixed profile;
opponents + chance are the stochastic environment):

  Pass 1 — probe under π. Play ``n_probe`` episodes with everyone following π
  (these double as the π-vs-π sample for V_i(π) — reused at no extra cost). At
  every visited hero decision point (capped at ``max_points`` per pass),
  estimate Q_i(s, a) for each legal action a with ``k_cont`` Monte-Carlo
  continuations: after a is played (opponents act ~π at simultaneous nodes),
  everyone — including the hero's future moves — follows π to termination.
  Returns are pooled per infoset key, where the key is the policy's OWN
  observation vector (``GameSpec.obs_tensor``), so the BR is built over the
  same state partition the policy acts on. BR action at key I: argmax_a
  Q̂(I, a), ties → lowest action id.

  Passes 2..R — approximate policy improvement (rollout à la Bertsekas).
  Re-probe along trajectories of the CURRENT table (hero plays BR^{r-1},
  fallback π at unprobed keys), and re-estimate Q̂(s, a) with continuations
  where the hero's future moves follow BR^{r-1} instead of π. Argmax updates
  only the re-probed keys; the table persists across passes. One pass of
  one-step-greedy estimation with π continuations (pass 1 alone) badly
  undervalues winning lines in deep games like tic_tac_toe (a forced win in 3
  plies looks mediocre when the hero keeps playing randomly afterwards);
  iterated passes propagate value backwards — pass r effectively optimizes
  the last r plies — and the measured BR value climbs monotonically (up to MC
  noise) toward the true BR value.

  Phase B — value difference. Play ``n_match`` episodes of (BR^R vs π_-i);
  at infoset keys never probed the hero falls back to sampling from π.
  gain_i = mean(BR returns) - mean(probe returns); se_i from the two
  independent iid return samples (Welch). The reported estimate is
  max(0, Σ_i gain_i) with standard error sqrt(Σ_i se_i²).

Bias — read before trusting small budgets. The estimator leans CONSERVATIVE
(underestimates exact NashConv): a mis-ranked argmax, an unvisited infoset,
or too few improvement passes can only lower the approximate BR's true value,
and the BR is restricted to the policy's observation partition, which a full
best response could refine. There is no systematic UPWARD channel on the
value: Phase B measures actual episodes of the approximate BR, never the
noisy max-Q̂ values themselves (the classic "max of noisy estimates" bias
affects only action SELECTION, i.e. the downward channel above). The
remaining Monte-Carlo error is zero-mean and is quantified by the reported
standard error (per-player CLT over iid episode returns). The estimate is
clamped at 0 (NashConv's mathematical floor); a negative raw sum means
"within noise of 0". As mc_samples and n_passes → ∞ with a full-support
policy (softmax policies always have full support), sampled policy iteration
on the finite BR MDP converges to the true best response, so the estimate
converges to exact NashConv; at finite budget read it as a
lower-bound-leaning estimate ± 2 SE.

Determinism: ALL randomness (chance nodes, profile sampling, BR fallback
sampling) flows through one ``random.Random(seed)``; the policy's internal RNG
is never consumed (we softmax-sample from ``action_logits`` ourselves). Same
seed + same policy ⇒ bit-identical result.

Cost: per-player budgets derive from ``mc_samples`` N:
``n_probe = max(4, N//8)`` per pass, ``max_points = max(16, N//4)`` per pass,
``k_cont = max(4, N//64)``, ``n_match = max(8, N//2)``. Probe coverage matters
more than per-point argmax accuracy: an unprobed state means a Phase-B
fallback to π there (large value loss), while a noisy argmax at a probed
state is a small one. Measured wall time (2026-07, this machine): oshi_zumo
+ MLP(128) ≈ 5 s at N=400 with the default 3 passes (target <30 s), ttt +
tabular ≈ 2 s at N=400 (target <10 s).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pyspiel

from mjai.agents.base import Policy
from mjai.games.loader import GameSpec

MIN_MC_SAMPLES = 16  # below this the derived budgets collapse; reject loudly
DEFAULT_N_PASSES = 3  # 1 base probe + 2 improvement passes (see module docstring)
# Distinct sampled states kept per infoset key within one pass. Continuation
# budget is spread round-robin over them, so a key costs |A| x k_cont episodes
# regardless of how often it was visited (huge savings once the BR plays
# deterministically and revisits few states), while still averaging Q̂ over the
# on-profile distribution of states WITHIN an infoset — critical in
# imperfect-information games, where a single first-seen state would bias the
# action choice (e.g. Kuhn "hold Q vs a bet": call is right on average over the
# opponent's hidden card but wrong for one specific card).
MAX_VISITS_PER_KEY = 8


@dataclass(frozen=True)
class SampledNashResult:
    """Outcome of one sampled NashConv evaluation.

    Attributes:
        nash_conv: max(0, Σ_i gain_i) — the NashConv point estimate.
        nash_conv_std: standard error of the estimate, sqrt(Σ_i se_i²).
        per_player_gain: raw (unclamped) per-player BR-minus-profile value gaps.
        per_player_std: per-player standard errors of those gaps.
        n_episodes: total episodes simulated (probe + continuation + match),
            for cost accounting / profiling (AGENTS.md §8).
    """

    nash_conv: float
    nash_conv_std: float
    per_player_gain: tuple[float, ...]
    per_player_std: tuple[float, ...]
    n_episodes: int


@dataclass(frozen=True)
class _DecisionPoint:
    """A cloned decision state for the hero plus its infoset key and legal set."""

    state: pyspiel.State
    key: tuple[float, ...]
    legal: tuple[int, ...]


def _obs_key(spec: GameSpec, state: pyspiel.State, player: int) -> tuple[float, ...]:
    """The policy's observation vector as a hashable infoset key."""
    return tuple(spec.obs_tensor(state, player))


def _legal_probs(
    spec: GameSpec, policy: Policy, state: pyspiel.State, player: int
) -> tuple[list[int], list[float]]:
    """(legal actions, π(·|obs)) — softmax over the policy's logits."""
    # legal_actions takes the player positionally; pyspiel's pybind rejects the
    # keyword form on simultaneous-move (matrix-style) games.
    legal = list(state.legal_actions(player))
    logits = policy.action_logits(spec.obs_tensor(state, player), legal)
    mx = max(logits)
    exps = [math.exp(x - mx) for x in logits]
    total = sum(exps) or 1.0
    return legal, [e / total for e in exps]


def _draw(rng: random.Random, actions: list[int], probs: list[float]) -> int:
    """Sample one action from a categorical with the estimator's own RNG."""
    r = rng.random()
    cum = 0.0
    for a, p in zip(actions, probs, strict=True):
        cum += p
        if r <= cum:
            return a
    return actions[-1]  # float-rounding fallback


def _apply_chance(state: pyspiel.State, rng: random.Random) -> None:
    outcomes = state.chance_outcomes()
    actions, probs = zip(*outcomes, strict=True)
    state.apply_action(_draw(rng, list(actions), [float(p) for p in probs]))


def _hero_action(
    spec: GameSpec,
    state: pyspiel.State,
    rng: random.Random,
    hero: int | None,
    player: int,
    br_table: dict[tuple[float, ...], int],
    legal_probs: tuple[list[int], list[float]],
) -> int:
    """Pick the acting player's action at one node.

    Opponents always follow π. The hero plays the BR table, falling back to π
    at keys that were never probed (the documented downward-bias channel).
    ``hero=None`` means everyone follows π (pass-1 probes, plain rollouts).
    """
    legal, probs = legal_probs
    if hero is None or player != hero:
        return _draw(rng, legal, probs)
    action = br_table.get(_obs_key(spec, state, hero))
    if action is not None and action in legal:
        return action
    return _draw(rng, legal, probs)


def _rollout(
    spec: GameSpec,
    policy: Policy,
    rng: random.Random,
    state: pyspiel.State,
    hero: int | None,
    br_table: dict[tuple[float, ...], int],
) -> list[float]:
    """Finish the episode (hero on ``br_table``, others on π); return returns."""
    while not state.is_terminal():
        if state.is_chance_node():
            _apply_chance(state, rng)
        elif state.is_simultaneous_node():
            joint = [
                _hero_action(
                    spec, state, rng, hero, p, br_table, _legal_probs(spec, policy, state, p)
                )
                for p in range(spec.num_players)
            ]
            state.apply_actions(joint)
        else:
            p = state.current_player()
            action = _hero_action(
                spec, state, rng, hero, p, br_table, _legal_probs(spec, policy, state, p)
            )
            state.apply_action(action)
    return [float(r) for r in state.returns()]


def _play_episode(
    spec: GameSpec,
    policy: Policy,
    rng: random.Random,
    table_player: int | None,
    br_table: dict[tuple[float, ...], int],
    point_player: int | None,
) -> tuple[list[float], list[_DecisionPoint]]:
    """Play one episode; returns (per-player returns, decision points).

    ``table_player`` is who plays the BR table (None = everyone follows π);
    ``point_player`` is whose decision points get collected (None = collect
    none). The two differ on pass 1, when everyone follows π yet the hero's
    points must still be recorded. Chance nodes use the estimator's RNG.
    """
    state = spec.new_state()
    points: list[_DecisionPoint] = []
    while not state.is_terminal():
        if state.is_chance_node():
            _apply_chance(state, rng)
            continue
        if state.is_simultaneous_node():
            per_player = [_legal_probs(spec, policy, state, p) for p in range(spec.num_players)]
            if point_player is not None:
                legal, _ = per_player[point_player]
                pt = _DecisionPoint(
                    state.clone(), _obs_key(spec, state, point_player), tuple(legal)
                )
                points.append(pt)
            joint = [
                _hero_action(spec, state, rng, table_player, p, br_table, per_player[p])
                for p in range(spec.num_players)
            ]
            state.apply_actions(joint)
        else:
            p = state.current_player()
            legal_probs = _legal_probs(spec, policy, state, p)
            if point_player is not None and p == point_player:
                pt = _DecisionPoint(
                    state.clone(), _obs_key(spec, state, point_player), tuple(legal_probs[0])
                )
                points.append(pt)
            state.apply_action(
                _hero_action(spec, state, rng, table_player, p, br_table, legal_probs)
            )
    return [float(r) for r in state.returns()], points


def _continuation_return(
    spec: GameSpec,
    policy: Policy,
    rng: random.Random,
    point: _DecisionPoint,
    hero: int,
    action: int,
    br_table: dict[tuple[float, ...], int],
) -> float:
    """One MC continuation of Q_i(point, action): the hero commits ``action`` at
    the probed point (opponents act ~π at simultaneous nodes), then plays
    ``br_table`` (fallback π) against π to termination.
    """
    s = point.state.clone()
    if s.is_simultaneous_node():
        joint = []
        for p in range(spec.num_players):
            if p == hero:
                joint.append(action)
            else:
                legal, probs = _legal_probs(spec, policy, s, p)
                joint.append(_draw(rng, legal, probs))
        s.apply_actions(joint)
    else:
        s.apply_action(action)
    return _rollout(spec, policy, rng, s, hero, br_table)[hero]


def _argmax_action(cell: dict[int, list[float]]) -> int:
    """argmax over pooled mean continuation return; ties → lowest action id."""

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    return max(cell, key=lambda a: (mean(cell[a]), -a))


def _mean_var(xs: list[float]) -> tuple[float, float]:
    """Sample mean and unbiased variance of iid episode returns."""
    n = len(xs)
    mean = sum(xs) / n
    if n < 2:
        return mean, 0.0
    return mean, sum((x - mean) ** 2 for x in xs) / (n - 1)


def _probe_pass(
    spec: GameSpec,
    policy: Policy,
    rng: random.Random,
    hero: int,
    br_table: dict[tuple[float, ...], int],
    n_probe: int,
    max_points: int,
    k_cont: int,
) -> tuple[list[float], int]:
    """One probe+improvement pass; returns (π-vs-π returns if pass 1 else [],
    episode count). Updates ``br_table`` in place at re-probed keys.
    """
    hero_on_table: int | None = hero if br_table else None  # pass 1: everyone π
    points: list[_DecisionPoint] = []
    profile_returns: list[float] = []
    n_episodes = 0
    for _ in range(n_probe):
        returns, pts = _play_episode(spec, policy, rng, hero_on_table, br_table, hero)
        n_episodes += 1
        if not br_table:
            profile_returns.append(returns[hero])  # pass 1 doubles as V^π sample
        if len(points) < max_points:
            points.extend(pts[: max_points - len(points)])
    q: dict[tuple[float, ...], list[_DecisionPoint]] = {}
    for pt in points:
        visits = q.setdefault(pt.key, [])
        if len(visits) < MAX_VISITS_PER_KEY:
            visits.append(pt)
    for visits in q.values():
        legal = visits[0].legal  # same infoset ⇒ same legal set (perfect recall)
        cell: dict[int, list[float]] = {a: [] for a in legal}
        for a in legal:
            for j in range(k_cont):
                pt = visits[j % len(visits)]  # round-robin over sampled states
                cell[a].append(_continuation_return(spec, policy, rng, pt, hero, a, br_table))
                n_episodes += 1
        br_table[visits[0].key] = _argmax_action(cell)
    return profile_returns, n_episodes


def sampled_nash_conv(
    spec: GameSpec,
    policy: Policy,
    *,
    mc_samples: int = 400,
    seed: int = 0,
    n_passes: int = DEFAULT_N_PASSES,
) -> SampledNashResult:
    """Estimate NashConv(π) by Monte-Carlo approximate best response.

    Args:
        spec: the loaded game (must be 2-player; all Phase-1 games are).
        policy: the profile π; both seats play it (mirror/self-play eval).
        mc_samples: per-player MC episode budget unit N; derived per-phase
            budgets are documented in the module docstring.
        seed: master seed for every sampling draw; same seed + same policy ⇒
            bit-identical result.
        n_passes: probe/improvement passes of sampled policy iteration on the
            BR MDP (1 = one-step greedy with π continuations; the default 3
            handles 9-ply tic_tac_toe well).

    Raises:
        ValueError: for non-2-player games or degenerate budgets (loud, never
            a silent fallback — AGENTS.md §11).
    """
    if spec.num_players != 2:
        raise ValueError(
            f"sampled_nash_conv supports 2-player games only; {spec.name} has {spec.num_players}."
        )
    if mc_samples < MIN_MC_SAMPLES:
        raise ValueError(f"mc_samples must be >= {MIN_MC_SAMPLES}, got {mc_samples}")
    if n_passes < 1:
        raise ValueError(f"n_passes must be >= 1, got {n_passes}")
    n_probe = max(4, mc_samples // 8)
    max_points = max(16, mc_samples // 4)
    k_cont = max(4, mc_samples // 64)
    n_match = max(8, mc_samples // 2)

    rng = random.Random(seed)
    gains: list[float] = []
    ses: list[float] = []
    n_episodes = 0
    for hero in range(spec.num_players):
        br_table: dict[tuple[float, ...], int] = {}
        profile_returns: list[float] = []
        for _ in range(n_passes):
            pi_returns, used = _probe_pass(
                spec, policy, rng, hero, br_table, n_probe, max_points, k_cont
            )
            n_episodes += used
            profile_returns.extend(pi_returns)
        # Phase B: approximate BR vs profile.
        br_returns: list[float] = []
        for _ in range(n_match):
            returns, _pts = _play_episode(spec, policy, rng, hero, br_table, None)
            n_episodes += 1
            br_returns.append(returns[hero])
        m_br, v_br = _mean_var(br_returns)
        m_pi, v_pi = _mean_var(profile_returns)
        gains.append(m_br - m_pi)
        ses.append(math.sqrt(v_br / len(br_returns) + v_pi / len(profile_returns)))
    return SampledNashResult(
        nash_conv=max(0.0, sum(gains)),
        nash_conv_std=math.sqrt(sum(s * s for s in ses)),
        per_player_gain=tuple(gains),
        per_player_std=tuple(ses),
        n_episodes=n_episodes,
    )
