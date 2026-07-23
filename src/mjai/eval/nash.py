"""Equilibrium-distance metrics over a trained policy (AGENTS.md §3, Step 7).

Thin wrappers over OpenSpiel's solvers that turn a mjai :class:`Policy` into the
format OpenSpiel expects, then compute:
  - :func:`exploitability_of`  — for turn-based 2p0-sum games (Kuhn, Leduc,
    Liar's-Dice). Calls open_spiel.exploitability.exploitability.
  - :func:`nash_conv_of`       — for any game incl. simultaneous (Goofspiel,
    Oshi-Zumo, BRPS). Calls open_spiel.exploitability.nash_conv.
  - :func:`distance_to_brps_nash` — TV distance to the analytic BRPS NE.

Two routes to the same numbers:

  - **Reference** (:class:`_PolicyAdapter` + the two ``*_of`` functions): asks
    the policy for one state at a time, exactly as OpenSpiel's traversal
    demands. Simple, and the definition of correct.
  - **Fast** (:func:`equilibrium_metrics_exact`, used by the training loop):
    materializes the policy over every info state ONCE with a single batched
    query, then hands OpenSpiel a plain TabularPolicy. The reference route
    re-queries each info state ~18x (once per tree node that reaches it) and
    pays ~200 us per query for an NN; on Liar's Dice that is 122 s per eval on
    CPU and 468 s on GPU (one CUDA sync per legal action), versus ~15 s here.

The two agree to ~1e-8 relative, not bit-for-bit: a batched float32 forward
uses different BLAS blocking than a one-row forward, so the logits differ in
the last ulp. That is far below the metric's seed-to-seed noise, but it does
mean eval values are not comparable bit-for-bit with runs from before this
change. ``tests/unit/test_eval_nash.py`` pins the agreement.

**Reproducibility caveat.** The fast route's default best-response solver is
OpenSpiel's C++ ``TabularBestResponseMDP`` (7.9x on Liar's Dice), which sums
over a hash-map iteration order that is not stable across processes: repeating
the same eval in a fresh process moved nash_conv by 1 ulp in 1 run out of 5.
That is ~1e-16 relative — nothing against seed-to-seed spread — but it means
"same seed, same eval bits" holds only under ``eval_exact_backend="python"``.
Set that when you need exactly-reproducible curves; the default trades those
last two digits for the speedup.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Fast route: materialize the policy once per eval, then let OpenSpiel walk a
# plain TabularPolicy.
# ---------------------------------------------------------------------------

# Per-game state enumeration, keyed by the canonical game string. The game tree
# is invariant across evals, so the expensive part (walking every state, ~3.5 s
# on Liar's Dice) is paid once per process rather than once per eval point.
# Bounded by the number of exactly-evaluable games (D8: 7), and only populated
# for games that actually use the exact estimator. Measured resident cost:
# Liar's-Dice-1 ~16 MB, Goofspiel-5 ~2.6 MB, Leduc ~0.7 MB.
_SKELETON_CACHE: dict[str, tuple[ospolicy.TabularPolicy, np.ndarray]] = {}


def _state_obs(state: pyspiel.State, player: int) -> list[float]:
    """The observation the policy was trained on, for ``player`` at ``state``.

    Mirrors :meth:`_PolicyAdapter.action_probabilities` exactly — information
    state when the game provides one, observation tensor otherwise. The player
    is passed in rather than read from ``state.current_player()`` because
    simultaneous games report ``SIMULTANEOUS_PLAYER_ID`` (-2) there, which no
    tensor accessor accepts.
    """
    try:
        return list(state.information_state_tensor(player))
    except Exception:
        return list(state.observation_tensor(player))


def _row_players(tabular: ospolicy.TabularPolicy) -> list[int]:
    """Owning player per TabularPolicy row, recovered from its per-player index."""
    players = [0] * len(tabular.states)
    for player, info_states in enumerate(tabular.states_per_player):
        for info_state in info_states:
            players[tabular.state_lookup[info_state]] = player
    return players


def _skeleton(spec: GameSpec) -> tuple[ospolicy.TabularPolicy, np.ndarray]:
    """Cached (empty TabularPolicy, per-state observation matrix) for ``spec``."""
    hit = _SKELETON_CACHE.get(spec.game_string)
    if hit is None:
        tabular = ospolicy.TabularPolicy(spec.game)
        obs = np.asarray(
            [
                _state_obs(state, player)
                for state, player in zip(tabular.states, _row_players(tabular), strict=True)
            ],
            dtype=np.float32,
        )
        hit = (tabular, obs)
        _SKELETON_CACHE[spec.game_string] = hit
    return hit


def clear_skeleton_cache() -> None:
    """Drop the cached state enumerations (tests; long-lived multi-game processes)."""
    _SKELETON_CACHE.clear()


def tabular_view_of(spec: GameSpec, policy: Policy) -> ospolicy.TabularPolicy:
    """``policy`` as an OpenSpiel TabularPolicy, built with ONE batched query.

    .. warning::
       The returned object is the cached skeleton with its probability array
       overwritten in place — the next call for the same game invalidates it.
       Consume it before evaluating another policy; never store it.
    """
    tabular, obs = _skeleton(spec)
    mask = np.asarray(tabular.legal_actions_mask, dtype=bool)
    if not mask.any(axis=1).all():
        raise ValueError(
            f"{spec.name}: a decision state has no legal actions; refusing to "
            "normalize an empty action distribution (AGENTS.md: fail loudly)"
        )
    logits = np.asarray(policy.action_logits_batch(obs, mask), dtype=np.float64)
    logits = np.where(mask, logits, -np.inf)
    logits -= logits.max(axis=1, keepdims=True)
    exps = np.exp(logits)
    exps[~mask] = 0.0
    tabular.action_probability_array[:] = exps / exps.sum(axis=1, keepdims=True)
    return tabular


EXACT_BACKENDS = ("auto", "python", "cpp")


def _cpp_policy(tabular: ospolicy.TabularPolicy) -> pyspiel.Policy:
    """Convert a materialized TabularPolicy into OpenSpiel's C++ policy type."""
    probs = tabular.action_probability_array
    return pyspiel.TabularPolicy(
        {
            info: [
                (int(a), float(probs[row][a]))
                for a in np.flatnonzero(tabular.legal_actions_mask[row])
            ]
            for info, row in tabular.state_lookup.items()
        }
    )


def use_cpp_backend(spec: GameSpec, backend: str = "auto") -> bool:
    """Whether to solve ``spec`` with OpenSpiel's C++ best-response MDP.

    ``auto`` means "turn-based games only", which is a statement about where
    the C++ solver both works and wins, measured 2026-07-23:

      - turn-based: Liar's Dice 14.2 s -> 1.8 s (7.9x), Leduc 6.1x, Kuhn 12x,
        values identical or within 1e-15;
      - simultaneous: the C++ solver raises ``prob <= 1`` (an exact-1.0
        probability tripping a strict assertion) for EVERY trained MLP policy
        tried on Goofspiel-5, while accepting uniform tabular ones — i.e. it
        fails precisely on the policies we train. BRPS is also slower there,
        being tiny enough that the policy conversion dominates. Both games are
        cheap in Python anyway (0.00 s and 2.6 s).

    Where the C++ solver does answer it agrees with the Python route, on
    simultaneous games too; it never silently returns something different. So
    ``cpp`` is a safe thing to force — it either matches or raises.

    Dispatch is by game type, not by try/except: which backend ran is a
    property of the config, knowable before the run (AGENTS.md: no silent
    fallbacks).
    """
    if backend not in EXACT_BACKENDS:
        raise ValueError(f"unknown exact backend {backend!r}; want {' | '.join(EXACT_BACKENDS)}")
    if backend == "python":
        return False
    if backend == "cpp":
        return True
    return not spec.is_simultaneous


def equilibrium_metrics_exact(
    spec: GameSpec, policy: Policy, *, backend: str = "auto"
) -> dict[str, float]:
    """Exact ``nash_conv`` (+ ``exploitability`` when defined) in ONE traversal.

    OpenSpiel's own ``exploitability`` docstring states it "is equivalent to
    NashConv / num_players" for 2-player constant-sum games, so computing both
    metrics separately walks the same tree twice for no extra information —
    roughly 44% of the old eval cost. Here exploitability is derived from
    nash_conv; games outside that identity fall back to the reference call.

    ``backend`` selects the best-response solver (see :func:`use_cpp_backend`).
    Note that the C++ solver's own ``exploitability`` field is NOT used — it
    reports 0 for these games — so both routes derive it from nash_conv.
    """
    tabular = tabular_view_of(spec, policy)
    if use_cpp_backend(spec, backend):
        # LIFETIME: pyspiel.TabularBestResponseMDP keeps a raw reference to the
        # policy without extending its lifetime, so the C++ policy MUST stay in
        # a live Python local for as long as the solver is used. Passing it as
        # a temporary (`TabularBestResponseMDP(game, _cpp_policy(t)).nash_conv()`)
        # segfaults the interpreter — no exception, no traceback.
        cpp_policy = _cpp_policy(tabular)
        solver = pyspiel.TabularBestResponseMDP(spec.game, cpp_policy)
        nash_conv = float(solver.nash_conv().nash_conv)
        del solver, cpp_policy
    else:
        from open_spiel.python.algorithms import exploitability

        nash_conv = float(exploitability.nash_conv(spec.game, tabular))
    out = {"nash_conv": nash_conv}
    if spec.num_players != 2 or spec.is_simultaneous:
        return out
    if spec.is_zero_sum:
        out["exploitability"] = nash_conv / spec.num_players
    else:  # constant- but not zero-sum: no identity to lean on, ask OpenSpiel.
        out["exploitability"] = exploitability_of(spec, policy)
    return out


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


def evaluate_equilibrium(
    spec: GameSpec,
    policy: Policy,
    *,
    estimator: str = "exact",
    mc_samples: int = 400,
    seed: int = 0,
    exact_backend: str = "auto",
) -> dict[str, float]:
    """Run whichever equilibrium metric(s) apply, return as a dict.

    Always returns nash_conv when computable; adds exploitability for
    turn-based games and exact_nash_distance for BRPS. A metric that raises
    (unsupported game/policy combo) is skipped WITH a warning — never silently
    (AGENTS.md: no silent fallback).

    ``estimator`` selects the NashConv backend (AGENTS.md §9, config knob
    ``eval_estimator``): "exact" walks the full game tree via OpenSpiel —
    infeasible for oshi_zumo-scale games; "sampled" uses the Monte-Carlo
    approximate-BR estimator in :mod:`mjai.eval.sampled_nash` with a
    per-player budget of ``mc_samples`` episodes and additionally reports
    ``nash_conv_std`` (plus ``exploitability_std`` for 2p0-sum turn-based
    games, where exploitability = nash_conv / 2 is an exact identity that
    carries over to the estimates). Unknown estimators raise ValueError
    loudly.
    """
    import warnings

    if estimator not in ("exact", "sampled"):
        raise ValueError(f"unknown eval estimator {estimator!r}; want 'exact' | 'sampled'")
    out: dict[str, float] = {}
    if spec.name == "brps":
        out["exact_nash_distance"] = distance_to_brps_nash(policy, num_actions=spec.num_actions)
    if estimator == "sampled":
        from mjai.eval.sampled_nash import sampled_nash_conv

        try:
            res = sampled_nash_conv(spec, policy, mc_samples=mc_samples, seed=seed)
            out["nash_conv"] = res.nash_conv
            out["nash_conv_std"] = res.nash_conv_std
            if not spec.is_simultaneous and spec.num_players == 2 and spec.is_zero_sum:
                out["exploitability"] = res.nash_conv / 2.0
                out["exploitability_std"] = res.nash_conv_std / 2.0
        except Exception as e:
            warnings.warn(f"sampled nash_conv not computable for {spec.name}: {e}", stacklevel=2)
        return out
    try:
        out.update(equilibrium_metrics_exact(spec, policy, backend=exact_backend))
    except Exception as e:
        warnings.warn(f"exact equilibrium eval failed for {spec.name}: {e}", stacklevel=2)
    return out
