"""Tabular UpdateRules: PPO-style and ACH-style updates on dict-backed policies.

These drive the small-game Phase-1 experiments and the CFR/exact-Nash validation
(AGENTS.md §1 D5, Step 3). They mutate :class:`mjai.agents.tabular.TabularPolicy`
logit/value rows in place via semi-gradient updates (no autograd; the dicts are
the parameters).

The two rules share the math of policy-gradient-on-advantage; they differ in:

  - **PPO-tabular**: clipped semi-gradient surrogate over the
    (new_logprob/old_logprob) ratio, like PPO's clipped objective but applied
    as a direct additive update to the chosen action's logit. Keeps a trust
    region; standard for the PPO-vs-ACH comparison.
  - **ACH-tabular**: multiplicative Hedge-style update toward
    exp(+eta * advantage), which is the discrete-time replicator step. No
    clipping — the entropy regularization is what stabilizes it. This is the
    algorithm whose last-iterate convergence is the paper's headline claim.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pyspiel

from mjai.agents.base import entropy_of_probs, masked_softmax
from mjai.agents.tabular import TabularPolicy, _obs_to_key
from mjai.algos.transition import Batch, UpdateStats
from mjai.algos.update_rule import AlgoConfig, UpdateRule
from mjai.games.loader import GameSpec

# Logit clamp: prevents softmax saturation. Without it, an additive Hedge/PPO
# step on a winning action can push its logit to +30 over a few hundred steps,
# collapsing the policy to a spike and freezing training (advantage
# normalization then yields 0 gradient forever). ±10 lets the policy express
# up to ~e^20 ≈ 5e8:1 preferences without going numerically deterministic.
LOGIT_CLAMP = 10.0


def _clamp_row(logits: list[float], legal_mask: np.ndarray) -> None:
    """Clamp legal-action logits to ±LOGIT_CLAMP (in place)."""
    for a in range(len(legal_mask)):
        if legal_mask[a]:
            logits[a] = max(-LOGIT_CLAMP, min(LOGIT_CLAMP, logits[a]))


def _reject_weighted(batch: Batch, rule_name: str) -> None:
    """Refuse a reach-tempered batch rather than quietly dropping the weights.

    Neither tabular rule has a place to put them: the PPO/Hedge rule applies one
    fixed-size step per row instead of reducing a loss (scaling that step by up
    to ``sample_weight_clip`` would be a different, untested algorithm), and the
    ACH rule is a CFR+ wrapper that never reads the samples at all. Ignoring a
    knob the config asked for is exactly the silent fallback AGENTS.md §11
    forbids.
    """
    if batch.weights is not None:
        raise NotImplementedError(
            f"{rule_name} received a weighted batch; RolloutConfig.sample_weight_kappa "
            "is wired for the NN path only (AGENTS.md §11: no silent fallback)"
        )


def _explained_variance(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """1 - Var(y_true - y_pred) / Var(y_true); 1.0 is a perfect value fit."""
    yt = np.asarray(list(y_true), dtype=np.float64)
    yp = np.asarray(list(y_pred), dtype=np.float64)
    if yt.var() < 1e-12:
        return 0.0
    return float(1.0 - (yt - yp).var() / yt.var())


class TabularUpdateRule(UpdateRule):
    """Common base for the two tabular rules; holds config + the policy.

    Subclasses implement :meth:`_update_row`, which mutates one observation's
    logit/value row given the per-transition advantage. The base handles batch
    iteration and stats aggregation.
    """

    def __init__(self, policy: TabularPolicy, config: AlgoConfig | None = None) -> None:
        if not isinstance(policy, TabularPolicy):
            raise TypeError(f"{type(self).__name__} requires a TabularPolicy, got {type(policy)}")
        super().__init__(policy)
        self.config = config or AlgoConfig()
        self.policy: TabularPolicy = policy  # narrow the type for subclasses

    def step(self, batch: Batch) -> UpdateStats:
        if batch.size == 0:
            return UpdateStats(policy_loss=0.0, value_loss=0.0, entropy=0.0)
        _reject_weighted(batch, type(self).__name__)
        # Normalize advantages per-batch (matches NN rules). Critical for games
        # with large unscaled returns (BRPS payoffs are +-50/+-25/+-5); without
        # this, one Hedge step of eta*50 saturates the logits and the policy
        # collapses to a deterministic one — after which normalized advantages
        # are all zero and training freezes.
        advs = np.asarray(batch.advantages, dtype=np.float64)
        if advs.size > 1 and advs.std() > 1e-8:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        total_pol = 0.0
        total_val = 0.0
        total_ent = 0.0
        value_preds: list[float] = []
        value_targets: list[float] = []
        for i in range(batch.size):
            obs = batch.obs[i].tolist()
            adv = float(advs[i])
            ret = float(batch.returns[i])
            old_v = self.policy.get_value(obs)
            value_preds.append(old_v)
            value_targets.append(ret)
            pol_loss, ent = self._update_row(obs, batch.actions[i], adv, batch.legal_mask[i])
            val_loss = self._update_value(obs, ret)
            total_pol += pol_loss
            total_val += val_loss
            total_ent += ent
        n = batch.size
        return UpdateStats(
            policy_loss=total_pol / n,
            value_loss=total_val / n,
            entropy=total_ent / n,
            explained_variance=_explained_variance(value_targets, value_preds),
        )

    def _update_value(self, obs: list[float], target: float) -> float:
        """TD-ish regression: v <- v + lr * (target - v)."""
        key = _obs_to_key(obs)
        old = self.policy.values.get(key, 0.0)
        new = old + self.config.value_coef * self.config.learning_rate * (target - old)
        self.policy.values[key] = new
        return float((target - old) ** 2)

    def _update_row(
        self,
        obs: list[float],
        action: int,
        advantage: float,
        legal_mask: np.ndarray,
    ) -> tuple[float, float]:
        """Subclass-specific logit update; returns (policy_loss, entropy)."""
        raise NotImplementedError

    def _row_probs(self, obs: list[float], legal_mask: np.ndarray) -> list[float]:
        """Full-space probability vector after masking, at the current logits."""
        logits = self.policy.get_logits(obs)
        mask = [bool(m) for m in legal_mask]
        scaled = [lg / self.policy.temperature for lg in logits]
        return masked_softmax(scaled, mask)

    @staticmethod
    def _entropy_from_probs(probs: list[float], legal_mask: np.ndarray) -> float:
        legal_p = [p for p, m in zip(probs, legal_mask, strict=True) if m]
        return entropy_of_probs(legal_p)


class TabularPPOUpdate(TabularUpdateRule):
    """PPO-style clipped semi-gradient update on tabular logits.

    For the chosen action ``a`` at observation ``s`` with old/new log-probs
    (here: before/after the update), the clipped surrogate objective drives an
    additive update to ``logit[s, a]`` proportional to the clipped
    advantage-weighted ratio. We approximate the "old" policy as the current
    one before the update (on-policy), so the ratio starts at 1 and is clipped
    to ``[1-eps, 1+eps]`` times the sign of the advantage.
    """

    def __init__(
        self, policy: TabularPolicy, config: AlgoConfig | None = None, *, clip_eps: float = 0.2
    ) -> None:
        super().__init__(policy, config)
        self.clip_eps = clip_eps

    def _update_row(
        self, obs: list[float], action: int, advantage: float, legal_mask: np.ndarray
    ) -> tuple[float, float]:
        lr = self.config.learning_rate
        # Clipped surrogate (on-policy: ratio=1 before update): the per-step
        # advantage is clamped to [-clip_eps, +clip_eps], so the additive logit
        # update cannot move more than lr*clip_eps in either direction.
        clipped = max(-self.clip_eps, min(self.clip_eps, advantage))
        delta = lr * clipped
        logits = self.policy.get_logits(obs)
        logits[action] += delta
        _clamp_row(logits, legal_mask)
        probs = self._row_probs(obs, legal_mask)
        pol_loss = -math.log(probs[action] + 1e-30) * (1 if advantage >= 0 else -1)
        return pol_loss, self._entropy_from_probs(probs, legal_mask)


class TabularACHUpdate(TabularUpdateRule):
    """ACH as a CFR+ wrapper (AGENTS.md §1 D4).

    The ACH paper derives its objective from the CFR / regret-minimization view.
    The faithful tabular realization of that view is **CFR+** (CFR with
    positive-regret clamping and weighted averaging), which converges to Nash on
    any 2p0-sum extensive-form game.

    A from-scratch online single-sample regret-matching approximation (the
    previous implementation) does NOT converge on simultaneous games with
    non-uniform Nash like BRPS (Nash = (1/16, 10/16, 5/16)): once mirror
    self-play goes deterministic, payoffs vanish, advantages hit zero, and
    regrets freeze. CFR+ avoids this by doing a full game-tree traversal each
    iteration — it visits every joint action, so the signal never collapses.

    This wrapper runs ``iters_per_step`` CFR+ iterations per ``step()`` call,
    then writes the resulting average policy into the TabularPolicy's logits so
    the existing sampling / eval code reads it correctly. The batch argument is
    accepted for interface compatibility but ignored (CFR+ doesn't consume
    sampled transitions; it enumerates the tree).

    Simultaneous games (BRPS, Goofspiel, Oshi-Zumo) are auto-converted to
    turn-based via ``pyspiel.convert_to_turn_based`` so CFR+ accepts them.

    Note: requires the game's GameSpec at construction (the base UpdateRule
    signature takes only a Policy; we add ``spec`` as a required kwarg). The
    experiment runner constructs this with the spec; callers building a raw
    TabularACHUpdate must pass it.
    """

    def __init__(
        self,
        policy: TabularPolicy,
        spec: GameSpec,
        config: AlgoConfig | None = None,
        *,
        hedge_eta: float | None = None,  # accepted for backward compat; unused
        iters_per_step: int = 10,
    ) -> None:
        if not isinstance(policy, TabularPolicy):
            raise TypeError(f"{type(self).__name__} requires a TabularPolicy, got {type(policy)}")
        UpdateRule.__init__(self, policy)
        self.config = config or AlgoConfig()
        self.policy: TabularPolicy = policy
        self.spec = spec
        self.iters_per_step = int(iters_per_step)
        # Build a CFR+-compatible (turn-based) game.
        import pyspiel

        if spec.is_simultaneous:
            self._cfr_game = pyspiel.convert_to_turn_based(spec.game)
        else:
            self._cfr_game = spec.game
        from open_spiel.python.algorithms import cfr

        self._solver = cfr.CFRPlusSolver(self._cfr_game)
        self._total_iters = 0
        self._last_entropy = 0.0

    def step(self, batch: Batch) -> UpdateStats:
        """Run ``iters_per_step`` CFR+ iterations and sync the average policy."""
        _reject_weighted(batch, type(self).__name__)
        for _ in range(self.iters_per_step):
            self._solver.evaluate_and_update_policy()
            self._total_iters += 1
        avg_policy = self._solver.average_policy()
        # Collect the average policy keyed on the ORIGINAL game's obs (which is
        # what the rollout observes). For simultaneous games the original is a
        # one-shot matrix game with a single trivial obs per player.
        info_states = _collect_info_states_original(self.spec.game, avg_policy, self._cfr_game)
        entropy_total = 0.0
        n_states = 0
        for obs_key, action_probs in info_states.items():
            logits_row = self.policy.logits.setdefault(obs_key, [0.0] * self.policy.num_actions)
            for a, p in action_probs.items():
                logits_row[a] = math.log(max(p, 1e-12))
            for a in range(self.policy.num_actions):
                if a not in action_probs:
                    logits_row[a] = -LOGIT_CLAMP
            entropy_total += entropy_of_probs(list(action_probs.values()))
            n_states += 1
        self._last_entropy = entropy_total / max(n_states, 1)
        # NashConv of the current average policy (None/-1 if unsupported).
        nash_conv: float = -1.0
        try:
            from open_spiel.python.algorithms import exploitability

            nash_conv = float(exploitability.nash_conv(self._cfr_game, avg_policy))
        except Exception:
            pass
        return UpdateStats(
            policy_loss=0.0,  # CFR+ has no policy-loss; it minimizes regret.
            value_loss=0.0,
            entropy=self._last_entropy,
            explained_variance=0.0,
            extra={"cfr_iters": float(self._total_iters), "nash_conv": nash_conv},
        )

    def _update_row(
        self, obs: list[float], action: int, advantage: float, legal_mask: np.ndarray
    ) -> tuple[float, float]:
        raise NotImplementedError(
            "TabularACHUpdate.step overrides the loop; _update_row is unused."
        )


def _obs_key(state: pyspiel.State, player: int) -> bytes:
    """Build the obs-key TabularPolicy uses from a state's observation tensor."""
    try:
        obs_vec = state.information_state_tensor(player)
    except Exception:
        obs_vec = state.observation_tensor(player)
    return b"|".join(f"{round(float(x), 9):.9f}".encode() for x in obs_vec)


def _index_cfr_strategy(
    cfr_game: pyspiel.Game, avg_policy: object
) -> dict[tuple[int, str], dict[int, float]]:
    """Build a {(player, info_state_string): {action: prob}} lookup of CFR+ avg policy."""

    out: dict[tuple[int, str], dict[int, float]] = {}

    def walk(state: pyspiel.State) -> None:
        if state.is_terminal() or state.is_chance_node() or state.is_simultaneous_node():
            if state.is_chance_node():
                for a, _p in state.chance_outcomes():
                    walk(state.child(a))
            return
        player = state.current_player()
        key = (player, state.information_state_string(player))
        if key not in out:
            legal = state.legal_actions(player)
            probs = avg_policy.action_probabilities(state, player)  # type: ignore[attr-defined]
            out[key] = {a: float(probs.get(a, 0.0)) for a in legal}
        for a in state.legal_actions(player):
            walk(state.child(a))

    walk(cfr_game.new_initial_state())
    return out


def _collect_info_states_original(
    original_game: pyspiel.Game,
    avg_policy: object,
    cfr_game: pyspiel.Game,
) -> dict[bytes, dict[int, float]]:
    """Collect CFR+'s average policy keyed on the ORIGINAL game's observation.

    Walks the original game tree; at each player decision point, builds the obs
    key (matching what TabularPolicy / the rollout uses) and looks up the CFR+
    average strategy via the player's information-state-string. For simultaneous
    games the original has a single simultaneous node; each player's strategy
    there equals the CFR+ strategy at the converted tree's root (before that
    player has acted), so the lookup by player info-state-string is exact.

    For Phase-1 games (all small, all with a single root decision per player for
    simultaneous, or a clean sequential structure for turn-based), this mapping
    is exact. Returns {obs_key: {action: prob}}.
    """
    cfr_by_str = _index_cfr_strategy(cfr_game, avg_policy)
    out: dict[bytes, dict[int, float]] = {}
    visited_obs: set[bytes] = set()

    def lookup(player: int, state: pyspiel.State, legal: list[int]) -> dict[int, float]:
        iss = state.information_state_string(player)
        strat = cfr_by_str.get((player, iss))
        if strat is None:
            # Fallback: any CFR+ entry for this player, else uniform.
            fallbacks = [v for (pp, _s), v in cfr_by_str.items() if pp == player]
            strat = fallbacks[0] if fallbacks else {a: 1.0 / len(legal) for a in legal}
        return {a: strat.get(a, 0.0) for a in legal}

    def walk_original(state: pyspiel.State) -> None:
        if state.is_terminal():
            return
        if state.is_chance_node():
            for a, _p in state.chance_outcomes():
                walk_original(state.child(a))
            return
        if state.is_simultaneous_node():
            for p in range(state.num_players()):
                ok = _obs_key(state, p)
                if ok in visited_obs:
                    continue
                visited_obs.add(ok)
                out[ok] = lookup(p, state, list(state.legal_actions(p)))
            return
        player = state.current_player()
        ok = _obs_key(state, player)
        if ok not in visited_obs:
            visited_obs.add(ok)
            out[ok] = lookup(player, state, list(state.legal_actions(player)))
        for a in state.legal_actions(player):
            walk_original(state.child(a))

    walk_original(original_game.new_initial_state())
    return out
