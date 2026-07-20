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
        state = self.game_spec.new_state()
        transitions: list[Transition] = []
        # Record (player, obs, legal, action, value) at each decision point.
        steps: list[tuple[int, list[float], list[int], int, float]] = []

        while not state.is_terminal():
            if state.is_chance_node():
                self._sample_chance(state)
                continue
            if state.is_simultaneous_node():
                joint = self._simultaneous_actions(state, learner, opponent)
                # Record both players' transitions BEFORE applying (so obs is
                # the pre-step observation).
                for p in range(self.game_spec.num_players):
                    obs = self.game_spec.obs_tensor(state, p)
                    # Per-player legal set (positional id; kw form rejected by pyspiel).
                    legal = list(state.legal_actions(p))
                    policy = self._policy_for(p, learner, opponent)
                    v = policy.value(obs)
                    steps.append((p, obs, legal, joint[p], v))
                state.apply_actions(joint)
            else:
                p = state.current_player()
                obs = self.game_spec.obs_tensor(state, p)
                legal = list(state.legal_actions())
                policy = self._policy_for(p, learner, opponent)
                v = policy.value(obs)
                a, lp = policy.act(obs, legal, eval=False)
                steps.append((p, obs, legal, a, v))
                state.apply_action(a)

        returns = list(state.returns())
        for p, obs, legal, a, v in steps:
            policy = self._policy_for(p, learner, opponent)
            # Recompute logprob under the (unchanged) policy for the record.
            _, lp = policy.act(obs, legal, eval=False)
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
        """Fill in return_ and advantage per transition.

        For these short IIG episodes the return IS the terminal payoff (rewards
        are terminal-only), so return_ = returns[player]. Advantage =
        return_ - value baseline (the ACH/PPO advantage with no GAE).
        """
        for t in transitions:
            r = returns[t.player]
            t.return_ = float(r)
            t.advantage = float(r) - t.value

    def _policy_for(self, player: int, learner: Policy, opponent: Policy) -> Policy:
        return learner if player == self.learner_player else opponent

    def _sample_chance(self, state: pyspiel.State) -> None:
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes, strict=True)
        idx = self._rng.choices(range(len(actions)), weights=probs, k=1)[0]
        state.apply_action(actions[idx])

    def _simultaneous_actions(
        self, state: pyspiel.State, learner: Policy, opponent: Policy
    ) -> list[int]:
        n = self.game_spec.num_players
        actions: list[int] = []
        for p in range(n):
            obs = self.game_spec.obs_tensor(state, p)
            # legal_actions accepts the player id positionally (kw form rejected
            # by pyspiel's pybind for matrix games).
            legal = list(state.legal_actions(p))
            policy = self._policy_for(p, learner, opponent)
            a, _ = policy.act(obs, legal, eval=False)
            actions.append(a)
        return actions
