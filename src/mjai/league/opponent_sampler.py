"""Opponent sampler for the league (AGENTS.md §1 D10, Step 6).

Decides which pool member a learner plays next. The opponent rule is
role-dependent (locked league design):

  - main agent:           50% current main, 30% history, 20% exploiters
  - main-exploiter:       100% current main (it exists to expose the main)
  - league-exploiter:     POOL MEMBERS ONLY — never the live main. It attacks
                          the league's past, not its present (AlphaStar-style
                          role split). History vs pool-exploiters are drawn in
                          the mix's own history:exploiter proportion,
                          renormalized (0.3:0.2 -> 60%/40% by default).

Within the chosen category, opponents are drawn by PFSP: weight ∝
1/(|winrate-0.5|+ε), so we over-sample the most competitive opponents
(those whose win-rate against the learner is near 50%). Win-rates come from
the :class:`~mjai.league.checkpoint_store.CheckpointStore`'s cached values;
unmeasured opponents default to 0.5 (uniform) until played. The
league-exploiter has no pool identity of its own, so its PFSP weights are all
the 0.5 default — i.e. uniform within the drawn category.
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
    """Opponent-sampling mix for the MAIN role (the 50/30/20 default).

    The league-exploiter reuses only the history:exploiter proportion of this
    mix (renormalized); its "current main" bucket is always zero by design.
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
        mix: the 50/30/20 mix used by the main role (the league-exploiter
            derives its pool-only proportions from it).
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

        Returns ``current_main`` (the live policy) when a main-role draw lands
        on the "current main" bucket (or has to fall back), or a pool member
        otherwise. For the league-exploiter the result is ALWAYS a pool member
        (None when the pool is empty — the caller treats that as a bug, since
        the manager seeds the pool at construction); for the main-exploiter it
        is always ``current_main``.

        Args:
            pool: the live checkpoint store members.
            learner_role: the role of the agent that's about to play.
            current_main: the live main policy (the main role's 50% bucket).
            learner_member_id: this learner's own pool member id, so we don't
                sample ourselves.
        """
        if learner_role == Role.MAIN_EXPLOITER:
            # Main-exploiter only ever plays the current main.
            return current_main
        if learner_role == Role.LEAGUE_EXPLOITER:
            return self._sample_league_exploiter(pool)

        # Main role: the configured 50/30/20 mix.
        bucket = self._draw_bucket()
        if bucket == "current_main":
            return current_main
        candidates = self._bucket_members(pool, bucket, learner_member_id)
        if not candidates:
            # Bucket empty: fall back to current main, else any pool member.
            return current_main if current_main is not None else (pool[0].policy if pool else None)
        return self._pfsp_pick(candidates, learner_member_id).policy

    # ---- internals ----

    def _sample_league_exploiter(self, pool: list[PoolMember]) -> Policy | None:
        """Draw strictly from the pool: history vs exploiters in the mix's own
        proportion, renormalized; a drawn-but-empty category yields the other
        pool category; the live main is never returned."""
        history_w, exploiter_w = self.mix.history_weight, self.mix.exploiter_weight
        total = history_w + exploiter_w
        # Degenerate mix (all weight on current_main): any pool split is
        # equally defensible — use 50/50 rather than inventing a new knob.
        p_history = history_w / total if total > 0 else 0.5
        first, second = (
            ("history", "exploiter")
            if self.rng.random() < p_history
            else (
                "exploiter",
                "history",
            )
        )
        for bucket in (first, second):
            candidates = self._bucket_members(pool, bucket, learner_member_id=None)
            if candidates:
                return self._pfsp_pick(candidates, learner_id=None).policy
        return None

    def _bucket_members(
        self, pool: list[PoolMember], bucket: str, learner_member_id: int | None
    ) -> list[PoolMember]:
        if bucket == "history":
            return [m for m in pool if m.role == Role.MAIN and m.member_id != learner_member_id]
        return [
            m
            for m in pool
            if m.role in (Role.MAIN_EXPLOITER, Role.LEAGUE_EXPLOITER)
            and m.member_id != learner_member_id
        ]

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
