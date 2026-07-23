"""League self-play controller (AGENTS.md §1 D10, Step 6).

Implements :class:`mjai.algos.controller.SelfPlayController` by delegating
matchup decisions to a :class:`~mjai.league.manager.LeagueManager`. The Trainer
treats this exactly like MirrorSelfPlay — it doesn't know or care which is in
use (AGENTS.md §2, §4).

Each :meth:`collect` round picks a role (rotates through main, main-exploiter,
league-exploiter), asks the manager for the opponent, plays one batch of
episodes via the rollout runner, reports results back to the manager (for
promotion/reset), and returns every kept learner's transitions as
:class:`~mjai.algos.controller.LearnerBatch` parts.

Routing is by PRODUCER IDENTITY, never by physical seat. The rollout runner
may shuffle which seat the collector occupies each episode (perspective
coverage), and every transition carries the policy that produced it, so:

  - the collecting role always trains on its own samples, whichever seat it
    sat in — seat-0 filtering would keep the OPPONENT's samples on flipped
    episodes;
  - frozen opponents' transitions are dropped simply by not being kept;
  - with ``train_live_opponents`` on, an opponent that is itself a live
    learner (the main-exploiter always faces the live main) also gets its own
    part — on-policy anti-exploiter data for that learner instead of waste.

Win/promotion signals are computed from the collector's own part, so they stay
correct whichever seat the collector sat in (B2).
"""

from __future__ import annotations

import random

import numpy as np

from mjai.agents.base import Policy
from mjai.algos.controller import (
    Collected,
    LearnerBatch,
    RolloutRunnerProtocol,
    SelfPlayController,
)
from mjai.league.checkpoint_store import Role
from mjai.league.manager import LeagueManager


class LeagueSelfPlay(SelfPlayController):
    """Round-robin across main + main-exploiter + league-exploiter.

    Args:
        manager: the :class:`LeagueManager` holding the live policies + pool.
        runner: the rollout back-end (RolloutRunnerProtocol; Step 4's
            RolloutWorkerCore satisfies it).
        episodes_per_round: how many episodes to play per collect() call.
        role_schedule: optional explicit role order; default rotates
            [MAIN, MAIN_EXPLOITER, LEAGUE_EXPLOITER] round-robin so each
            collects every third round.
        train_live_opponents: when True, a round whose opponent is itself a
            live learner (e.g. every main-exploiter round, whose opponent is
            always the live main) yields a part for that opponent too, so its
            samples update its own weights. When False (default), only the
            collecting role's samples are kept and the live opponent's share
            is dropped like a frozen opponent's.
    """

    def __init__(
        self,
        manager: LeagueManager,
        runner: RolloutRunnerProtocol,
        *,
        episodes_per_round: int = 50,
        role_schedule: list[Role] | None = None,
        train_live_opponents: bool = False,
        rng: random.Random | None = None,
    ) -> None:
        self.manager = manager
        self.runner = runner
        self.episodes_per_round = episodes_per_round
        self.role_schedule = role_schedule or [
            Role.MAIN,
            Role.MAIN_EXPLOITER,
            Role.LEAGUE_EXPLOITER,
        ]
        self.train_live_opponents = train_live_opponents
        self.rng = rng or random.Random()
        self._round_idx: int = 0
        self._main: Policy | None = None
        # B7 telemetry: latest per-round true win-rate of each exploiter role
        # (fraction of the collector's own episodes with a positive return).
        self._last_true_winrate: dict[Role, float] = {}

    def set_learner(self, policy: Policy) -> None:
        """The Trainer passes the main agent here each step."""
        self._main = policy
        self.manager.main = policy  # keep the manager's main pointer fresh

    def learners(self) -> tuple[Policy, ...]:
        """The three policies :meth:`collect` may return a part for.

        The caller (which owns update rules; this layer must not — AGENTS.md
        §2) builds one rule per entry, so each role trains its own weights.
        """
        return (self.manager.main, self.manager.main_exploiter, self.manager.league_exploiter)

    def collect(self) -> Collected:
        if self._main is None:
            raise RuntimeError("LeagueSelfPlay.collect called before set_learner")
        role = self.role_schedule[self._round_idx % len(self.role_schedule)]
        self._round_idx += 1
        # Apply any reset this role earned last time it played, before its
        # weights are read for rollout (see LeagueManager.begin_round).
        self.manager.begin_round(role)

        learner = (
            self.manager.main
            if role == Role.MAIN
            else (
                self.manager.main_exploiter
                if role == Role.MAIN_EXPLOITER
                else self.manager.league_exploiter
            )
        )
        opponent = self.manager.opponent_for(role)
        if opponent is None:
            opponent = self.manager.main  # pool empty at the very start

        # The keep-set, by identity: the collecting role always keeps its own
        # samples; a live-learner opponent's samples are kept only when
        # train_live_opponents is on. (A self-play round has opponent IS
        # learner — one keep entry, both seats' samples, nothing dropped.)
        keep: list[Policy] = [learner]
        if (
            self.train_live_opponents
            and opponent is not learner
            and any(opponent is p for p in self.learners())
        ):
            keep.append(opponent)

        batch = self.runner.run_episode(learner=learner, opponent=opponent, keep=tuple(keep))

        # Route by producer identity: each kept learner gets exactly the
        # transitions ITS policy generated (both seats across shuffled
        # episodes); frozen opponents' transitions never become a part.
        parts: list[LearnerBatch] = []
        for p in keep:
            sub = batch.for_producer(p)
            if sub.size:
                parts.append(LearnerBatch(batch=sub, learner=p, label=self._label_for(p)))
        collector_part = next((pt for pt in parts if pt.learner is learner), None)

        # Report results back so the manager can promote/reset and so PFSP
        # bookkeeping stays current. The win signal is the mean real episode
        # return over the COLLECTOR'S OWN transitions — right under seat
        # shuffle, where the collector is not always seat 0 (B2).
        won = _collector_won(collector_part)
        n_episodes = self._episodes_this_round(collector_part)
        if role == Role.MAIN:
            self.manager.record_main_round(opponent=opponent, won=won, n_episodes=n_episodes)
        else:
            self._report_exploiter_round(role, opponent, collector_part, won, n_episodes)

        # sampled_steps is the FULL pooled batch: frozen/live opponents were
        # simulated too, and dropping their transitions does not refund the cost.
        return Collected(parts=tuple(parts), sampled_steps=batch.size)

    @property
    def name(self) -> str:
        return "league"

    def health_stats(self) -> dict[str, float]:
        """League health scalars for TensorBoard (B7; logged by the runner, §6)."""
        stats: dict[str, float] = {k: float(v) for k, v in self.manager.stats().items()}
        for role in (Role.MAIN_EXPLOITER, Role.LEAGUE_EXPLOITER):
            wr = self._last_true_winrate.get(role)
            if wr is not None:
                stats[f"exploiter_true_winrate/{role.value}"] = wr
        return stats

    # ---- internals ----

    def _label_for(self, policy: Policy) -> str:
        """The logging label for a kept policy; every kept policy is a live learner."""
        if policy is self.manager.main:
            return Role.MAIN.value
        if policy is self.manager.main_exploiter:
            return Role.MAIN_EXPLOITER.value
        if policy is self.manager.league_exploiter:
            return Role.LEAGUE_EXPLOITER.value
        raise RuntimeError(
            f"kept policy {type(policy).__name__} at {id(policy):#x} is not a live "
            "learner; keep-sets are built from learners(), so this is a controller "
            "bug (AGENTS.md §11)."
        )

    def _episodes_this_round(self, collector_part: LearnerBatch | None) -> int:
        """Episodes played in the collect round that produced ``collector_part``.

        RolloutWorkerCore exposes the exact count via ``last_episode_count``.
        Runners without that attribute fall back to the collector's retained
        transition count — exact for one-decision-per-episode games
        (BRPS/Goofspiel/Oshi-Zumo) and an upper bound otherwise.
        """
        exact = getattr(self.runner, "last_episode_count", None)
        if isinstance(exact, int) and exact > 0:
            return exact
        return collector_part.batch.size if collector_part is not None else 1

    def _report_exploiter_round(
        self,
        role: Role,
        opponent: Policy,
        part: LearnerBatch | None,
        won: bool | None,
        n_episodes: int,
    ) -> None:
        """Aggregate the round's real outcomes and feed one signal to the manager."""
        if won is None:
            return  # empty batch: nothing measured, nothing to report
        winrate = _collector_winrate(part)
        if winrate is not None:
            self._last_true_winrate[role] = winrate
        self.manager.record_exploiter_match(role, opponent, won, n_episodes=n_episodes)


def _collector_returns(part: LearnerBatch | None) -> np.ndarray:
    """Real episode returns of the collecting role — B2.

    Routing has already reduced the part to the collector's own transitions,
    and every transition carries its episode's terminal payoff for the seat
    its producer occupied — so the plain returns array is the signal, under
    any seat assignment. The old signal (mean advantage over the pooled
    two-seat batch) measured the sum of the two critics' errors, not who won.
    """
    if part is None or part.batch.size == 0:
        return np.zeros(0, dtype=np.float32)
    rets: np.ndarray = part.batch.returns
    return rets


def _collector_won(part: LearnerBatch | None) -> bool | None:
    """The round counts as won iff the collector's mean real return is positive."""
    rets = _collector_returns(part)
    if rets.size == 0:
        return None
    return bool(rets.mean() > 0.0)


def _collector_winrate(part: LearnerBatch | None) -> float | None:
    """Fraction of the collector's decisions whose episode return is positive."""
    rets = _collector_returns(part)
    if rets.size == 0:
        return None
    return float((rets > 0.0).mean())
