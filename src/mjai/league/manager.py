"""League manager: role orchestration + promotion/reset (AGENTS.md §1, Step 6).

Owns the three live learners (main, main-exploiter, league-exploiter), the
:class:`~mjai.league.checkpoint_store.CheckpointStore`, and the
:class:`~mjai.league.opponent_sampler.OpponentSampler`. Decides per round:
  - which role collects a batch,
  - who that role's opponent is,
  - whether any exploiter should be promoted to the pool + reset.

Promotion thresholds (Step 6 design, configurable):
  - main_exploiter promoted when its rolling win-rate vs the current main >=
    ``main_exploiter_promo`` (default 0.55).
  - league_exploiter promoted when its win-rate vs >``league_exploiter_share``
    of the pool (default 0.70 share) is >= ``league_exploiter_promo``.

Reset policy (configurable knob, default: reset to current main weights per the
locked Step 6 design).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from mjai.agents.base import Policy
from mjai.league.checkpoint_store import CheckpointStore, Role
from mjai.league.opponent_sampler import LeagueMix, OpponentSampler


@dataclass(frozen=True)
class LeagueConfig:
    """All league knobs (Step 6 locked design + caller tuning)."""

    capacity: int = 16
    main_save_every_steps: int = 200
    main_exploiter_promo: float = 0.55  # >= this win-rate vs current main
    league_exploiter_promo: float = 0.55  # >= this win-rate vs pool members
    league_exploiter_share: float = 0.70  # fraction of pool it must beat
    promo_window: int = 20  # rolling-window size for win-rates
    mix: LeagueMix = field(default_factory=LeagueMix)
    # How an exploiter's weights are reset after promotion. Returns a fresh Policy.
    # Default: reset to the current main's weights (Step 6 locked decision).
    reset_mode: str = "to_main"  # "to_main" | "random"


class LeagueManager:
    """Holds the live policies + pool and decides matchups each round.

    The main agent is constructed externally and passed in; the manager owns
    the two exploiters (built via ``make_policy`` so they share the main's
    architecture).
    """

    def __init__(
        self,
        main: Policy,
        make_policy: Callable[[], Policy],
        copy_weights_into: Callable[[Policy, Policy], None],
        config: LeagueConfig | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config or LeagueConfig()
        self.main = main
        self._make_policy = make_policy
        self._copy_weights = copy_weights_into
        self.rng = rng or random.Random()
        self.store = CheckpointStore(capacity=self.config.capacity)
        self.sampler = OpponentSampler(self.config.mix, rng=self.rng)
        # Two exploiters, warm-started from the main per the locked default.
        self.main_exploiter = make_policy()
        self.league_exploiter = make_policy()
        self._copy_weights(self.main, self.main_exploiter)
        self._copy_weights(self.main, self.league_exploiter)
        # Rolling win-rate windows (list of 0/1 per recent episode).
        self._me_window: list[float] = []  # main-exploiter vs main
        self._le_window: dict[int, list[float]] = {}  # league-exploiter vs each pool member
        self._main_steps: int = 0

    # ---- per-round matchup decisions ----

    def opponent_for(self, role: Role) -> Policy | None:
        """Return the opponent policy for a learner of ``role`` this episode."""
        return self.sampler.sample(
            pool=self.store.members,
            learner_role=role,
            current_main=self.main,
            learner_member_id=None,  # live learners aren't pool members
        )

    # ---- result reporting (called after each episode) ----

    def record_main_round(self) -> None:
        """Call once per main-agent training step (drives periodic snapshots)."""
        self._main_steps += 1
        if self._main_steps % self.config.main_save_every_steps == 0:
            self.store.add(self._clone(self.main), Role.MAIN, train_step=self._main_steps)

    def record_exploiter_match(self, role: Role, opponent: Policy, won: bool) -> None:
        """Record an exploiter's result; trigger promotion/reset if threshold met."""
        if role == Role.MAIN_EXPLOITER:
            self._me_window.append(1.0 if won else 0.0)
            self._me_window = self._me_window[-self.config.promo_window :]
            if len(self._me_window) >= self.config.promo_window // 2:
                wr = sum(self._me_window) / len(self._me_window)
                if wr >= self.config.main_exploiter_promo:
                    self._promote(role, self._clone(self._policy_for_role(role)))
        elif role == Role.LEAGUE_EXPLOITER:
            # Need to identify which pool member we played; infer by identity.
            opp_id = self._find_member_id(opponent)
            window = self._le_window.setdefault(opp_id if opp_id is not None else -1, [])
            window.append(1.0 if won else 0.0)
            window[:] = window[-self.config.promo_window :]
            self._maybe_promote_league_exploiter()

    # ---- promotion internals ----

    def _maybe_promote_league_exploiter(self) -> None:
        """Promote the league exploiter if it beats >= share of the pool."""
        if not self._le_window:
            return
        beaten = 0
        total = 0
        for _opp_id, window in self._le_window.items():
            if len(window) < 3:
                continue
            total += 1
            wr = sum(window) / len(window)
            if wr >= self.config.league_exploiter_promo:
                beaten += 1
        if total == 0:
            return
        if beaten / total >= self.config.league_exploiter_share:
            self._promote(Role.LEAGUE_EXPLOITER, self._clone(self.league_exploiter))

    def _promote(self, role: Role, snapshot: Policy) -> None:
        """Save the exploiter snapshot to the pool and reset its live weights."""
        self.store.add(snapshot, role, train_step=self._main_steps)
        target = self.main_exploiter if role == Role.MAIN_EXPLOITER else self.league_exploiter
        if self.config.reset_mode == "random":
            fresh = self._make_policy()
            self._copy_weights(fresh, target)
        else:  # "to_main" — locked default
            self._copy_weights(self.main, target)
        # Clear the window so the reset exploiter re-earns its record.
        if role == Role.MAIN_EXPLOITER:
            self._me_window.clear()
        else:
            self._le_window.clear()

    # ---- helpers ----

    def _policy_for_role(self, role: Role) -> Policy:
        return {
            Role.MAIN: self.main,
            Role.MAIN_EXPLOITER: self.main_exploiter,
            Role.LEAGUE_EXPLOITER: self.league_exploiter,
        }[role]

    def _clone(self, policy: Policy) -> Policy:
        """Build a fresh policy and load the source's snapshot into it.

        Uses :meth:`Policy.snapshot_state` / :meth:`restore_state` rather than
        ``copy.deepcopy(policy)``. For NN policies the snapshot lives on CPU, so
        the clone's GPU footprint is exactly one policy's weights — no transient
        doubling from a full-module deepcopy (AGENTS.md §8).
        """
        fresh = self._make_policy()
        fresh.restore_state(policy.snapshot_state())
        return fresh

    def _find_member_id(self, policy: Policy) -> int | None:
        for m in self.store.members:
            if m.policy is policy:
                return m.member_id
        return None
