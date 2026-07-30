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

import math
import random
from dataclasses import dataclass, field
from typing import Protocol

import pyspiel

from mjai.agents.base import Policy, target_ratio
from mjai.algos.transition import Batch, Transition, make_batch
from mjai.games.loader import GameSpec


class ValueOracle(Protocol):
    """An exact ``V(s)`` that replaces the critic's baseline in the advantage.

    Declared here as a structural type so the rollout stays independent of the
    thing that implements it (``mjai.eval.exact_value.ExactValueOracle`` needs
    the whole game tree, which no part of the pipeline should be pulled toward).
    ``refresh`` is called once per collection round -- collection is synchronous,
    so the profile is fixed for the whole round.
    """

    def refresh(self, learner: Policy, opponent: Policy) -> None: ...

    def value(self, obs: list[float]) -> float: ...


@dataclass
class RolloutConfig:
    """How episodes are collected and how returns are computed.

    Attributes:
        n_episodes: safety cap on episodes per :meth:`run_episode` call.
        gae_lambda: lambda for the per-player GAE (ACH paper App. E, p24; H.3 leaves
            it unspecified — spec assumption A1 follows the paper's other
            experiments at 0.95). Episodes are undiscounted (gamma=1): all Phase-1
            games are short with terminal-only rewards.
        seed: RNG seed for chance-node sampling (and seat shuffling, when on).
        target_samples: stop collecting whole episodes once the batch holds at
            least this many counted transitions (decision points). The paper's
            batch size is 64 samples (p28 Table 8); episodes are never truncated
            mid-game. ``None`` disables (collect exactly ``n_episodes``). Which
            transitions COUNT is decided per call by ``run_episode``'s ``keep``
            argument — not by physical seat, so the dose stays exact under seat
            shuffle (AGENTS.md §9: batch size is a config value, not a mode
            side effect).
        behavior_epsilon: sample actions from ``mu = (1-eps)*pi + eps*Uniform(legal)``
            instead of from ``pi``. 0.0 (default) is the paper's on-policy
            behavior (p24, ``mu_{p,t} = pi_{p,t}``); anything else is a
            deliberate deviation whose purpose is to flatten the information-set
            visitation ``rho`` (docs/liars_residual_floor.md). The recorded
            logprob is ``log mu(a|s)``, which is what makes ACH's ``1/pi_old``
            cancel the sampling probability exactly.
        advantage_estimator: "gae" (default, paper-faithful) or "vtrace". V-trace
            (Espeholt et al. 2018) corrects the value target and the advantage
            for the mismatch between ``mu`` and ``pi``; without it a run with
            ``behavior_epsilon > 0`` fits ``V^mu`` and estimates ``A^mu`` while
            the ACH update wants ``A^pi``.
        vtrace_rho_bar / vtrace_c_bar: the two V-trace truncations (IMPALA
            defaults 1.0). ``rho_bar`` caps the fixed point the value function
            converges to; ``c_bar`` caps how far a correction propagates back.
        sample_weight_kappa: temper the training distribution by weighting each
            sample with ``reach(h)^-kappa``, where ``reach(h)`` is the exact
            probability the sampler had of producing that history (chance x both
            players' behavior probabilities, all of them recorded here). A
            sample arrives with probability ``reach(h)``, so the weighted mass
            an information set receives is ``rho(I)^(1-kappa)`` up to the
            within-information-set correction measured in
            ``tools/history_weighting.py``. 0.0 (default) = the paper's
            on-policy weighting, and no ``Batch.weights`` is emitted at all.
            Rationale and the offline verification: docs/liars_residual_floor.md
            §8.4-8.5.
        sample_weight_clip: cap on the per-sample weight (``None`` = uncapped).
            ``reach(h)^-kappa`` is unbounded above; the cap is what bounds the
            per-batch variance the tempering buys coverage with. Measured
            offline: kappa=0.75 with clip 1e3 keeps effN 229 of the 1308
            an uncapped ideal weight reaches, at a weight range of 1e3.
        shuffle_seats: when True, each episode independently flips a fair coin
            (seeded) to decide which physical seat the ``learner`` occupies, so
            a learner sees BOTH perspectives against every opponent across
            episodes. Routing stays correct because every transition is tagged
            with the policy that produced it. Off by default: mirror self-play
            (same policy in both seats) would only perturb the RNG stream.
    """

    n_episodes: int = 1
    gae_lambda: float = 0.95
    seed: int | None = None
    target_samples: int | None = 64
    shuffle_seats: bool = False
    behavior_epsilon: float = 0.0
    advantage_estimator: str = "gae"
    vtrace_rho_bar: float = 1.0
    vtrace_c_bar: float = 1.0
    sample_weight_kappa: float = 0.0
    sample_weight_clip: float | None = None


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
        learner_player: the BASE seat (0 or 1) the ``learner`` argument controls.
            The opponent occupies the other seat. With ``shuffle_seats`` on, the
            learner's actual seat is re-drawn per episode (this value is what a
            coin-toss loss falls back to); transitions stay attributable via
            their ``producer`` tags, never via seat numbers. (For mirror
            self-play both seats get the same policy; transitions from BOTH
            players are always pooled — the paper trains the shared theta,omega
            on both seats' samples (p24); controllers route by producer
            afterwards via :meth:`Batch.for_producer`.)
        config: rollout hyperparameters.
    """

    def __init__(
        self,
        game_spec: GameSpec,
        *,
        learner_player: int = 0,
        config: RolloutConfig | None = None,
        value_oracle: ValueOracle | None = None,
    ) -> None:
        self.game_spec = game_spec
        self.learner_player = learner_player
        self.config = config or RolloutConfig()
        # Exact baseline for the advantage, in place of the critic's V. None
        # (default) is the paper's setup: the value comes from the network.
        self.value_oracle = value_oracle
        if self.config.shuffle_seats and game_spec.num_players != 2:
            raise ValueError(
                f"shuffle_seats is defined for 2-player games; "
                f"{game_spec.name} has {game_spec.num_players} (Phase-2 "
                "generalization would draw a seat permutation instead of a coin)"
            )
        self._rng = random.Random(self.config.seed)
        # Episodes played in the most recent run_episode call. The league
        # controller reads this to keep its promotion windows episode-counted.
        self.last_episode_count: int = 0

    def run_episode(
        self, learner: Policy, opponent: Policy, *, keep: tuple[Policy, ...] | None = None
    ) -> Batch:
        """Play episodes and return the pooled, producer-tagged transitions.

        Implements RolloutRunnerProtocol: the controller calls this with the
        learner in the (possibly shuffled) ``learner`` seat and (for mirror)
        the same policy as the opponent. Plays whole episodes until
        ``n_episodes`` is reached OR ``config.target_samples`` counted
        transitions have accumulated (never truncates an episode mid-game).

        ``keep`` decides which transitions count toward ``target_samples``:
          - ``None`` (mirror/tools default): every transition counts.
          - a tuple of live-learner policies: count per producer, and stop only
            when EVERY one of them has >= ``target_samples`` transitions, so
            each kept learner's update meets the protocol's batch size even
            when several learners share the round's episodes.
        The returned batch always holds ALL producers' transitions; the caller
        routes with :meth:`Batch.for_producer`.
        """
        if keep is not None and len(keep) == 0:
            raise ValueError("keep must be None (count all) or a non-empty tuple")
        keep_ids = {id(p) for p in keep} if keep else set()
        if self.value_oracle is not None:
            # Once per round, not per episode: collection is synchronous, so the
            # profile that generates every episode below is this one.
            self.value_oracle.refresh(learner, opponent)
        all_transitions: list[Transition] = []
        counted = 0
        per_producer: dict[int, int] = {}
        self.last_episode_count = 0
        for _ in range(self.config.n_episodes):
            target = self.config.target_samples
            if target is not None:
                if keep is None and counted >= target:
                    break
                if keep is not None and all(per_producer.get(pid, 0) >= target for pid in keep_ids):
                    break
            learner_seat = self._draw_learner_seat()
            transitions, returns = self._play_one_episode(learner, opponent, learner_seat)
            self._assign_returns(transitions, returns)
            all_transitions.extend(transitions)
            if keep is None:
                counted += len(transitions)
            else:
                for t in transitions:
                    pid = id(t.producer)
                    if pid in keep_ids:
                        per_producer[pid] = per_producer.get(pid, 0) + 1
            self._last_returns = returns
            self.last_episode_count += 1
        return make_batch(all_transitions, num_actions=self.game_spec.num_actions)

    def _draw_learner_seat(self) -> int:
        """This episode's learner seat: the base seat, or a coin-flip away."""
        if not self.config.shuffle_seats:
            return self.learner_player
        flip = self._rng.random() < 0.5
        return 1 - self.learner_player if flip else self.learner_player

    def _play_one_episode(
        self, learner: Policy, opponent: Policy, learner_seat: int
    ) -> tuple[list[Transition], list[float]]:
        """Play one episode, recording (action, logprob, value) per decision point.

        Hot-path note (AGENTS.md §8): each decision point calls the policy ONCE
        via ``act_with_value`` (single forward for NN; single masked-softmax for
        tabular). The behavior-policy logprob recorded here is exact — it is the
        same value the old two-pass implementation recomputed in a second loop.
        Every recorded step carries the acting policy itself, so the pooled
        batch can be routed by producer identity afterwards.
        """
        state = self.game_spec.new_state()
        # Record (player, producer, obs, legal, action, logprob, value, log_reach)
        # at each decision point in a single pass — no post-episode recompute loop.
        steps: list[tuple[int, Policy, list[float], list[int], int, float, float, float]] = []
        # log P(the sampler produces this history): chance outcomes times EVERY
        # player's behavior probabilities. Accumulated as the episode is played,
        # because that is the only place each factor is still available -- the
        # flattened batch has lost the trajectory, and `_sample_chance` is the
        # only thing that ever sees the chance probabilities.
        log_reach = 0.0

        while not state.is_terminal():
            if state.is_chance_node():
                log_reach += self._sample_chance(state)
                continue
            if state.is_simultaneous_node():
                joint, lps, per_player_values, actors = self._simultaneous_actions(
                    state, learner, opponent, learner_seat
                )
                # Record both players' transitions BEFORE applying (so obs is
                # the pre-step observation).
                for p in range(self.game_spec.num_players):
                    obs = self.game_spec.obs_tensor(state, p)
                    # Per-player legal set (positional id; kw form rejected by pyspiel).
                    legal = list(state.legal_actions(p))
                    steps.append(
                        (
                            p,
                            actors[p],
                            obs,
                            legal,
                            joint[p],
                            lps[p],
                            self._baseline(obs, per_player_values[p]),
                            log_reach,
                        )
                    )
                state.apply_actions(joint)
                log_reach += math.fsum(lps)  # both players moved out of this node
            else:
                p = state.current_player()
                obs = self.game_spec.obs_tensor(state, p)
                legal = list(state.legal_actions())
                policy = self._policy_for(p, learner, opponent, learner_seat)
                a, lp, v = policy.act_with_value(
                    obs, legal, eval=False, behavior_epsilon=self.config.behavior_epsilon
                )
                steps.append((p, policy, obs, legal, a, lp, self._baseline(obs, v), log_reach))
                state.apply_action(a)
                log_reach += lp

        returns = list(state.returns())
        transitions: list[Transition] = []
        for p, policy, obs, legal, a, lp, v, step_log_reach in steps:
            reward = returns[p]  # terminal-only rewards for these games
            transitions.append(
                Transition(
                    obs=obs,
                    legal_actions=legal,
                    action=a,
                    logprob=lp,
                    value=v,
                    reward=reward,
                    return_=reward,  # overwritten by _assign_returns below
                    advantage=0.0,
                    player=p,
                    producer=policy,
                    weight=self._sample_weight(step_log_reach),
                )
            )
        return transitions, returns

    def _baseline(self, obs: list[float], network_value: float) -> float:
        """The value that feeds the advantage: the network's, or the oracle's.

        Only the ADVANTAGE baseline is swapped. The value head is still trained
        on the returns by the update rule, so a shared-trunk run keeps whatever
        the value loss does to its features -- identical in the paired kappa=0
        control, which is the only thing this arm is read against.
        """
        if self.value_oracle is None:
            return network_value
        return self.value_oracle.value(obs)

    def _sample_weight(self, log_reach: float) -> float:
        """``min(reach(h)^-kappa, clip)`` — how many times this sample counts.

        Exactly 1.0 when the knob is off, so ``make_batch`` emits no weights and
        every update rule keeps its original reduction.

        Computed in log space: ``reach`` underflows float64 long before its
        negative power overflows, so the naive ``reach ** -kappa`` would read
        ``inf`` on histories the cap was meant to handle. An UNCAPPED weight can
        still overflow (``math.exp`` raises); that is deliberate -- the honest
        signal that this game needs ``sample_weight_clip`` set.
        """
        kappa = self.config.sample_weight_kappa
        if kappa == 0.0:
            return 1.0
        log_w = -kappa * log_reach
        cap = self.config.sample_weight_clip
        if cap is not None:
            log_w = min(log_w, math.log(cap))
        return math.exp(log_w)

    def _assign_returns(self, transitions: list[Transition], returns: list[float]) -> None:
        """Fill in return_ and advantage per transition, gamma=1.

        For 2p0-sum games the transitions alternate between players. Each
        player's experience is a sub-trajectory; the estimator runs per-player
        using that player's value estimates and the terminal return as the
        episode's payoff. gamma is fixed at 1 (undiscounted): all Phase-1 games
        are short with terminal-only rewards, and the paper's H.3 leaves gamma
        unspecified (spec assumption A1).

        Two estimators, selected by ``RolloutConfig.advantage_estimator``:
        ``"gae"`` (default, paper-faithful) and ``"vtrace"``.
        """
        estimator = self.config.advantage_estimator
        if estimator not in ("gae", "vtrace"):
            raise ValueError(
                f"unknown advantage_estimator {estimator!r}; want 'gae' | 'vtrace' "
                "(AGENTS.md §11: no silent fallback)"
            )
        # Group transitions by player; within each group the order matches the
        # temporal order of that player's decisions in the episode.
        by_player: dict[int, list[Transition]] = {}
        for t in transitions:
            by_player.setdefault(t.player, []).append(t)
        for player, ts in by_player.items():
            if estimator == "gae":
                self._assign_gae(ts, returns[player])
            else:
                self._assign_vtrace(ts, returns[player])

    def _assign_gae(self, ts: list[Transition], r: float) -> None:
        """GAE(lambda) advantages with the Monte-Carlo return as the value target.

        The paper's combination (p24): ``G`` is the sampled return and the
        advantage is GAE(lambda). Unchanged from the original implementation --
        this is the bit-identical default path.
        """
        lam = self.config.gae_lambda
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

    def _assign_vtrace(self, ts: list[Transition], r: float) -> None:
        """V-trace(lambda) targets and advantages (Espeholt et al. 2018), gamma=1.

        Needed once the behavior policy stops being the target policy: with
        ``behavior_epsilon > 0`` the returns are collected under ``mu``, so an
        uncorrected critic fits ``V^mu`` and GAE estimates ``A^mu`` while the
        ACH update wants ``A^pi``. V-trace reweights both by the per-step
        ratio ``pi(a|s)/mu(a|s)``.

        The ratio is recovered from the recorded ``log mu(a|s)`` by
        :func:`mjai.agents.base.target_ratio` -- exactly, and without a second
        network pass, because collection and update are synchronous here so the
        ``pi`` that acted IS the ``pi`` being trained.

        Two deliberate choices, both stated because they are not the only ones:

        - ``c_i = lambda * min(c_bar, ratio_i)`` folds the existing
          ``gae_lambda`` into the trace, so the estimator is a strict
          generalization: at ``epsilon = 0`` every ratio is 1 and the advantage
          below reduces **exactly** to GAE(lambda).
        - the advantage is the value residual ``v_i - V_i`` rather than
          IMPALA's ``rho_i * (r_i + gamma * v_{i+1} - V_i)``. The residual form
          is what makes the reduction above exact; IMPALA's form is one line
          away if a future arm wants it.

        Note the value TARGET does change at ``epsilon = 0``: GAE mode regresses
        V on the Monte-Carlo return, V-trace regresses it on ``v_i``. So an
        ablation isolating the effect of exploration should compare
        ``vtrace(eps>0)`` against ``vtrace(eps=0)``, not against the GAE default.
        """
        lam = self.config.gae_lambda
        rho_bar = self.config.vtrace_rho_bar
        c_bar = self.config.vtrace_c_bar
        eps = self.config.behavior_epsilon
        n = len(ts)
        v_next = 0.0  # bootstrap past the terminal
        value_next = 0.0  # V(s_{n}) = 0 at the terminal
        for i in range(n - 1, -1, -1):
            t = ts[i]
            reward = r if i == n - 1 else 0.0
            ratio = target_ratio(math.exp(t.logprob), len(t.legal_actions), eps)
            rho = min(rho_bar, ratio)
            c = lam * min(c_bar, ratio)
            delta = rho * (reward + value_next - t.value)
            v = t.value + delta + c * (v_next - value_next)
            t.return_ = float(v)
            t.advantage = float(v - t.value)
            v_next, value_next = v, t.value

    def _policy_for(
        self, player: int, learner: Policy, opponent: Policy, learner_seat: int
    ) -> Policy:
        return learner if player == learner_seat else opponent

    def _sample_chance(self, state: pyspiel.State) -> float:
        """Apply one sampled chance outcome; return its log probability.

        The log-prob is returned rather than discarded because chance is a
        factor of the history's sampling probability exactly like the players'
        actions are, and this is the only point in the pipeline where it exists.
        """
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes, strict=True)
        idx = self._rng.choices(range(len(actions)), weights=probs, k=1)[0]
        state.apply_action(actions[idx])
        return math.log(probs[idx])

    def _simultaneous_actions(
        self, state: pyspiel.State, learner: Policy, opponent: Policy, learner_seat: int
    ) -> tuple[list[int], list[float], list[float], list[Policy]]:
        """Choose each player's action independently; return (actions, logprobs, values, actors).

        Each player gets exactly one fused ``act_with_value`` call — the
        behavior-policy logprob and value baseline are captured here in the same
        call that samples the action, so the caller records them without a
        second forward (AGENTS.md §8). ``actors[p]`` is the policy that produced
        player p's action, for the producer tag.
        """
        n = self.game_spec.num_players
        actions: list[int] = []
        logprobs: list[float] = []
        values: list[float] = []
        actors: list[Policy] = []
        for p in range(n):
            obs = self.game_spec.obs_tensor(state, p)
            # legal_actions accepts the player id positionally (kw form rejected
            # by pyspiel's pybind for matrix games).
            legal = list(state.legal_actions(p))
            policy = self._policy_for(p, learner, opponent, learner_seat)
            a, lp, v = policy.act_with_value(
                obs, legal, eval=False, behavior_epsilon=self.config.behavior_epsilon
            )
            actions.append(a)
            logprobs.append(lp)
            values.append(v)
            actors.append(policy)
        return actions, logprobs, values, actors
