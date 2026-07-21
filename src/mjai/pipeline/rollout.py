"""Rollout core: play episodes on an OpenSpiel env and yield a Batch.

This is the plain-Python (Ray-free) heart of the worker. Unit tests instantiate
it directly; ``pipeline._ray.RolloutWorker`` (Step 5) wraps it in a Ray actor.

It implements :class:`mjai.algos.controller.RolloutRunnerProtocol` so a
:class:`~mjai.algos.controller.MirrorSelfPlay` (or league) controller can call
``run_episode(learner, opponent)`` and get back a :class:`~mjai.algos.transition.Batch`.

Handles both turn-based and simultaneous-move games correctly:
  - turn-based: at each decision point, the acting player's policy chooses.
  - simultaneous: both policies choose independently; the joint action applies.
Chance nodes are auto-sampled (we never expose them to the policies).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pyspiel

from mjai.agents.base import Policy
from mjai.algos.transition import Batch, Transition, make_batch
from mjai.games.loader import GameSpec


@dataclass
class RolloutConfig:
    """How episodes are collected and how returns are computed."""

    n_episodes: int = 1  # episodes per run_episode call
    discount: float = 1.0  # IIG episodes are short; default undiscounted
    gae_lambda: float = 0.95  # GAE(λ) for advantage estimation (ACH paper §E)
    seed: int | None = None
    # Whether to pool transitions from BOTH players (True for self-play, where
    # the same learner occupies both seats) or only the learner's seat.
    pool_both_players: bool = True


@dataclass
class RolloutResult:
    """Diagnostic output alongside the batch."""

    batch: Batch
    episode_returns: list[list[float]] = field(default_factory=list)
    n_steps: int = 0


class RolloutWorkerCore:
    """Plays episodes of ``game_spec`` using two policies, returns a Batch.

    Args:
        game_spec: the loaded :class:`GameSpec`.
        learner_player: which seat (0 or 1) the ``learner`` argument controls.
            The opponent occupies the other seat. (For mirror self-play both
            seats get the same policy; ``learner_player`` still selects whose
            transitions get the "learner" tag, but with pool_both_players=True
            both are pooled regardless.)
        config: rollout hyperparameters.
    """

    def __init__(
        self,
        game_spec: GameSpec,
        *,
        learner_player: int = 0,
        config: RolloutConfig | None = None,
    ) -> None:
        self.game_spec = game_spec
        self.learner_player = learner_player
        self.config = config or RolloutConfig()
        self._rng = random.Random(self.config.seed)

    def run_episode(self, learner: Policy, opponent: Policy) -> Batch:
        """Play ``n_episodes`` episodes and return the pooled transitions.

        Implements RolloutRunnerProtocol: the controller calls this with the
        learner in the ``learner`` seat and (for mirror) the same policy as the
        opponent.
        """
        all_transitions: list[Transition] = []
        for _ in range(self.config.n_episodes):
            transitions, returns = self._play_one_episode(learner, opponent)
            self._assign_returns(transitions, returns)
            all_transitions.extend(transitions)
            self._last_returns = returns
        return make_batch(all_transitions, num_actions=self.game_spec.num_actions)

    def _play_one_episode(
        self, learner: Policy, opponent: Policy
    ) -> tuple[list[Transition], list[float]]:
        """Play one episode, recording (action, logprob, value) per decision point.

        Hot-path note (AGENTS.md §8): each decision point calls the policy ONCE
        via ``act_with_value`` (single forward for NN; single masked-softmax for
        tabular). The behavior-policy logprob recorded here is exact — it is the
        same value the old two-pass implementation recomputed in a second loop.
        """
        state = self.game_spec.new_state()
        # Record (player, obs, legal, action, logprob, value) at each decision
        # point in a single pass — no post-episode recompute loop.
        steps: list[tuple[int, list[float], list[int], int, float, float]] = []

        while not state.is_terminal():
            if state.is_chance_node():
                self._sample_chance(state)
                continue
            if state.is_simultaneous_node():
                joint, lps, per_player_values = self._simultaneous_actions(state, learner, opponent)
                # Record both players' transitions BEFORE applying (so obs is
                # the pre-step observation).
                for p in range(self.game_spec.num_players):
                    obs = self.game_spec.obs_tensor(state, p)
                    # Per-player legal set (positional id; kw form rejected by pyspiel).
                    legal = list(state.legal_actions(p))
                    steps.append((p, obs, legal, joint[p], lps[p], per_player_values[p]))
                state.apply_actions(joint)
            else:
                p = state.current_player()
                obs = self.game_spec.obs_tensor(state, p)
                legal = list(state.legal_actions())
                policy = self._policy_for(p, learner, opponent)
                a, lp, v = policy.act_with_value(obs, legal, eval=False)
                steps.append((p, obs, legal, a, lp, v))
                state.apply_action(a)

        returns = list(state.returns())
        transitions: list[Transition] = []
        for p, obs, legal, a, lp, v in steps:
            reward = returns[p]  # terminal-only rewards for these games
            transitions.append(
                Transition(
                    obs=obs,
                    legal_actions=legal,
                    action=a,
                    logprob=lp,
                    value=v,
                    reward=reward,
                    return_=reward,  # overwritten by _assign_returns if discounting
                    advantage=0.0,
                    player=p,
                )
            )
        return transitions, returns

    def _assign_returns(self, transitions: list[Transition], returns: list[float]) -> None:
        """Fill in return_ and advantage per transition using GAE(λ).

        For 2p0-sum games the transitions alternate between players. Each
        player's experience is a sub-trajectory; GAE is computed per-player
        using that player's value estimates and the terminal return as the
        episode's payoff. λ comes from ``self.config.gae_lambda``.
        """
        lam = self.config.gae_lambda
        # Group transitions by player; within each group the order matches the
        # temporal order of that player's decisions in the episode.
        by_player: dict[int, list[Transition]] = {}
        for t in transitions:
            by_player.setdefault(t.player, []).append(t)
        for player, ts in by_player.items():
            r = returns[player]
            n = len(ts)
            gae = 0.0
            # Walk backward through the player's transitions.
            for i in range(n - 1, -1, -1):
                t = ts[i]
                t.return_ = float(r)
                # TD residual: reward + gamma*V(next) - V(current).
                # For these games reward is 0 mid-episode and the terminal
                # payoff at the end; gamma=1 (undiscounted, short episodes).
                next_value = ts[i + 1].value if i + 1 < n else 0.0
                delta = 0.0 + next_value - t.value
                # On the last transition, attach the actual terminal return.
                if i == n - 1:
                    delta = r - t.value
                gae = delta + lam * gae
                t.advantage = gae

    def _policy_for(self, player: int, learner: Policy, opponent: Policy) -> Policy:
        return learner if player == self.learner_player else opponent

    def _sample_chance(self, state: pyspiel.State) -> None:
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes, strict=True)
        idx = self._rng.choices(range(len(actions)), weights=probs, k=1)[0]
        state.apply_action(actions[idx])

    def _simultaneous_actions(
        self, state: pyspiel.State, learner: Policy, opponent: Policy
    ) -> tuple[list[int], list[float], list[float]]:
        """Choose each player's action independently; return (actions, logprobs, values).

        Each player gets exactly one fused ``act_with_value`` call — the
        behavior-policy logprob and value baseline are captured here in the same
        call that samples the action, so the caller records them without a
        second forward (AGENTS.md §8).
        """
        n = self.game_spec.num_players
        actions: list[int] = []
        logprobs: list[float] = []
        values: list[float] = []
        for p in range(n):
            obs = self.game_spec.obs_tensor(state, p)
            # legal_actions accepts the player id positionally (kw form rejected
            # by pyspiel's pybind for matrix games).
            legal = list(state.legal_actions(p))
            policy = self._policy_for(p, learner, opponent)
            a, lp, v = policy.act_with_value(obs, legal, eval=False)
            actions.append(a)
            logprobs.append(lp)
            values.append(v)
        return actions, logprobs, values
