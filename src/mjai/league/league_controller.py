"""League self-play controller (AGENTS.md §1 D10, Step 6).

Implements :class:`mjai.algos.controller.SelfPlayController` by delegating
matchup decisions to a :class:`~mjai.league.manager.LeagueManager`. The Trainer
treats this exactly like MirrorSelfPlay — it doesn't know or care which is in
use (AGENTS.md §2, §4).

Each :meth:`collect` round picks a role (rotates through main, main-exploiter,
league-exploiter), asks the manager for the opponent, plays one batch of
episodes via the rollout runner, reports results back to the manager (for
promotion/reset), and returns the learner's transitions as a Batch.

The ``learner`` set via :meth:`set_learner` is taken to be the *main* agent;
the manager owns the two exploiters directly. Whose transitions are returned
depends on the role drawn this round (exploiters learn too — against their
narrower opponent sets).
"""

from __future__ import annotations

import random

import numpy as np

from mjai.agents.base import Policy
from mjai.algos.controller import RolloutRunnerProtocol, SelfPlayController
from mjai.algos.transition import Batch
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
    """

    def __init__(
        self,
        manager: LeagueManager,
        runner: RolloutRunnerProtocol,
        *,
        episodes_per_round: int = 50,
        role_schedule: list[Role] | None = None,
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
        self.rng = rng or random.Random()
        self._round_idx: int = 0
        self._main: Policy | None = None
        # B7 telemetry: latest per-round true win-rate of each exploiter role
        # (fraction of seat-0 episodes with a strictly positive return).
        self._last_true_winrate: dict[Role, float] = {}

    def set_learner(self, policy: Policy) -> None:
        """The Trainer passes the main agent here each step."""
        self._main = policy
        self.manager.main = policy  # keep the manager's main pointer fresh

    def collect(self) -> Batch:
        if self._main is None:
            raise RuntimeError("LeagueSelfPlay.collect called before set_learner")
        role = self.role_schedule[self._round_idx % len(self.role_schedule)]
        self._round_idx += 1

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

        batch = self.runner.run_episode(learner=learner, opponent=opponent)
        n_episodes = self._episodes_this_round(batch)

        # Report results back so the manager can promote/reset and so PFSP
        # bookkeeping stays current. The win signal is the mean real episode
        # return of the collecting role's seat (seat 0) — see _report_… (B2).
        won = _seat0_won(batch)
        if role == Role.MAIN:
            self.manager.record_main_round(opponent=opponent, won=won, n_episodes=n_episodes)
        else:
            self._report_exploiter_round(role, opponent, batch, won, n_episodes)

        # The rollout runner places the learner in seat 0 and the opponent in
        # seat 1. Return only the learner's transitions so each learner trains
        # on its own seat's data (the opponent seat belongs to a different
        # learner with a different objective).
        return batch.for_player(player=0)

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

    def _episodes_this_round(self, batch: Batch) -> int:
        """Episodes played in the collect round that produced ``batch``.

        RolloutWorkerCore exposes the exact count via ``last_episode_count``.
        Runners without that attribute fall back to the seat-0 transition
        count — exact for one-decision-per-episode games (BRPS/Goofspiel/
        Oshi-Zumo) and an upper bound otherwise.
        """
        exact = getattr(self.runner, "last_episode_count", None)
        if isinstance(exact, int) and exact > 0:
            return exact
        return int((batch.players == 0).sum()) if batch.size else 1

    def _report_exploiter_round(
        self, role: Role, opponent: Policy, batch: Batch, won: bool | None, n_episodes: int
    ) -> None:
        """Aggregate the round's real outcomes and feed one signal to the manager."""
        if won is None:
            return  # empty batch: nothing measured, nothing to report
        winrate = _seat0_winrate(batch)
        if winrate is not None:
            self._last_true_winrate[role] = winrate
        self.manager.record_exploiter_match(role, opponent, won, n_episodes=n_episodes)


def _seat0_returns(batch: Batch) -> np.ndarray:
    """Real episode returns of the collecting role (seat 0) — B2.

    Every transition carries its episode's terminal payoff as ``return_``;
    restricting to seat 0 avoids mixing the opponent's (negated) outcomes into
    the signal. The old signal (mean advantage over the pooled two-seat batch)
    measured the sum of the two critics' errors, not who won.
    """
    if batch.size == 0:
        return np.zeros(0, dtype=np.float32)
    rets: np.ndarray = batch.returns[batch.players == 0]
    return rets


def _seat0_won(batch: Batch) -> bool | None:
    """The round counts as won iff the mean seat-0 real return is positive."""
    rets = _seat0_returns(batch)
    if rets.size == 0:
        return None
    return bool(rets.mean() > 0.0)


def _seat0_winrate(batch: Batch) -> float | None:
    """Fraction of seat-0 decisions whose episode return is strictly positive."""
    rets = _seat0_returns(batch)
    if rets.size == 0:
        return None
    return float((rets > 0.0).mean())
