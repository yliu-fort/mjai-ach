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

        # Report results back so the manager can promote/reset.
        if role == Role.MAIN:
            self.manager.record_main_round()
        else:
            self._report_exploiter_round(role, opponent, batch)

        # The rollout runner places the learner in seat 0 and the opponent in
        # seat 1. Return only the learner's transitions so each learner trains
        # on its own seat's data (the opponent seat belongs to a different
        # learner with a different objective).
        return batch.for_player(player=0)

    @property
    def name(self) -> str:
        return "league"

    # ---- internals ----

    def _report_exploiter_round(self, role: Role, opponent: Policy, batch: Batch) -> None:
        """Aggregate the per-episode win/loss and feed one signal to the manager."""
        # In these zero-sum games the per-episode result is determined by the
        # terminal payoff for the learner (seat 0). We approximate "won" as
        # mean learner advantage > 0 over the batch.
        mean_adv = float(batch.advantages.mean()) if batch.size else 0.0
        won = mean_adv > 0.0
        self.manager.record_exploiter_match(role, opponent, won)
