"""Opponent sampler for the league (AGENTS.md §1 D10, Step 6).

Decides which pool member a learner plays next. The mixing weights are
role-dependent (AGENTS.md §1, Step 6 design):

  - main agent:           50% current main, 30% history, 20% exploiters
  - main-exploiter:       100% current main (it exists to expose the main)
  - league-exploiter:     50% current main, 30% history, 20% exploiters

Within the chosen category, opponents are drawn by PFSP: weight ∝
1/(|winrate-0.5|+ε), so we over-sample the most competitive opponents
(those whose win-rate against the learner is near 50%). Win-rates come from
the :class:`~mjai.league.checkpoint_store.CheckpointStore`'s cached values;
unmeasured opponents default to 0.5 (uniform) until played.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from mjai.agents.base import Policy
from mjai.league.checkpoint_store import PoolMember, Role

# PFSP smoothing: prevents division by zero and dampens over-concentration.
PFSP_EPS = 0.05


@dataclass(frozen=True)
class LeagueMix:
    """Per-role opponent-sampling mix (the 50/30/20 default, configurable).

    Values must sum to ~1.0 within floating tolerance; enforced at construction.
    """

    current_main_weight: float = 0.5
    history_weight: float = 0.3
    exploiter_weight: float = 0.2

    def __post_init__(self) -> None:
        total = self.current_main_weight + self.history_weight + self.exploiter_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"LeagueMix weights must sum to 1.0, got {total}")


class OpponentSampler:
    """Draws an opponent from the pool for a learner of a given role.

    Args:
        mix: the 50/30/20 mix used by main and league-exploiter roles.
        rng: random.Random for reproducible draws.
    """

    def __init__(self, mix: LeagueMix | None = None, *, rng: random.Random | None = None) -> None:
        self.mix = mix or LeagueMix()
        self.rng = rng or random.Random()

    def sample(
        self,
        pool: list[PoolMember],
        learner_role: Role,
        current_main: Policy | None,
        learner_member_id: int | None,
    ) -> Policy | None:
        """Pick an opponent Policy for a learner of ``learner_role``.

        Returns ``current_main`` (the live policy) when the 50% "current main"
        bucket is drawn, or a pool member otherwise. Returns None only if the
        pool and current_main are both empty (caller must handle).

        Args:
            pool: the live checkpoint store members.
            learner_role: the role of the agent that's about to play.
            current_main: the live main policy (the 50% bucket), if any.
            learner_member_id: this learner's own pool member id, so we don't
                sample ourselves.
        """
        if learner_role == Role.MAIN_EXPLOITER:
            # Main-exploiter only ever plays the current main.
            return current_main

        # Main and league-exploiter use the configured mix.
        bucket = self._draw_bucket()
        if bucket == "current_main":
            return current_main
        if bucket == "history":
            candidates = [
                m for m in pool if m.role == Role.MAIN and m.member_id != learner_member_id
            ]
        else:  # "exploiter"
            candidates = [
                m
                for m in pool
                if m.role in (Role.MAIN_EXPLOITER, Role.LEAGUE_EXPLOITER)
                and m.member_id != learner_member_id
            ]
        if not candidates:
            # Bucket empty: fall back to current main, else any pool member.
            return current_main if current_main is not None else (pool[0].policy if pool else None)
        return self._pfsp_pick(candidates, learner_member_id).policy

    def _draw_bucket(self) -> str:
        r = self.rng.random()
        cum = 0.0
        cum += self.mix.current_main_weight
        if r < cum:
            return "current_main"
        cum += self.mix.history_weight
        if r < cum:
            return "history"
        return "exploiter"

    def _pfsp_pick(self, candidates: list[PoolMember], learner_id: int | None) -> PoolMember:
        """Pick one candidate, weighting by 1/(|winrate-0.5|+eps)."""

        # Learner-vs-candidate win-rate; default 0.5 (uniform) if unmeasured.
        # The store caches win_rates keyed by *opponent* id; from the candidate's
        # perspective the learner is the opponent, so look up candidate's rate
        # vs the learner.
        def wr(c: PoolMember) -> float:
            if learner_id is None:
                return 0.5
            return c.win_rates.get(learner_id, 0.5)

        weights = [1.0 / (abs(wr(c) - 0.5) + PFSP_EPS) for c in candidates]
        return self.rng.choices(candidates, weights=weights, k=1)[0]
