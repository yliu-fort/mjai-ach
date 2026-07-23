"""League manager: role orchestration + promotion/reset (AGENTS.md §1, Step 6).

Owns the three live learners (main, main-exploiter, league-exploiter), the
:class:`~mjai.league.checkpoint_store.CheckpointStore`, and the
:class:`~mjai.league.opponent_sampler.OpponentSampler`. Decides per round:
  - which role collects a batch,
  - who that role's opponent is,
  - whether any exploiter should be promoted to the pool + reset.

Promotion thresholds (Step 6 design, configurable):
  - main_exploiter promoted when its rolling win-rate vs the current main >=
    ``main_exploiter_promo`` (default 0.70).
  - league_exploiter promoted when its win-rate vs >``league_exploiter_share``
    of the pool (default 0.70 share) is >= ``league_exploiter_promo``
    (default 0.70).

Pool seeding: construction places a ``train_step=0`` GENESIS snapshot of the
initial main into the pool. The pool is therefore never empty, which makes
"the league-exploiter only ever faces pool members" a hard invariant from the
first round (no fallback to the live main — AGENTS.md §11) and gives PFSP a
main-line member id to bookkeep against from round one.

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
    # Main-snapshot cadence, counted in MAIN COLLECT ROUNDS (not env-steps, not
    # total rounds): a snapshot of the live main enters the pool every
    # ``main_save_every_rounds`` main rounds. Under the default 1:2 role
    # schedule one main round is every third collect().
    main_save_every_rounds: int = 200
    main_exploiter_promo: float = 0.70  # >= this win-rate vs current main
    league_exploiter_promo: float = 0.70  # >= this win-rate vs pool members
    league_exploiter_share: float = 0.70  # fraction of pool it must beat
    promo_window: int = 20  # rolling win-rate window size, counted in EPISODES
    mix: LeagueMix = field(default_factory=LeagueMix)
    # How an exploiter's weights are reset after promotion. Returns a fresh Policy.
    # Default: reset to the current main's weights (Step 6 locked decision).
    reset_mode: str = "to_main"  # "to_main" | "random"
    # EMA step for the PFSP win-rate bookkeeping written back to the store
    # after each main round vs a pool member (B4). 1.0 = last-result-only.
    win_rate_ema_alpha: float = 0.1


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
        # Rolling win-rate windows, counted in EPISODES (B8): each entry is one
        # episode's outcome (0/1) for the collecting role's seat.
        self._me_window: list[float] = []  # main-exploiter vs main
        self._le_window: dict[int, list[float]] = {}  # league-exploiter vs each pool member
        self._main_steps: int = 0
        # PFSP bookkeeping (B4): the live main's latest pool member id, and an
        # EMA of per-(opponent, main-member) outcomes written back to the store.
        self._main_member_id: int | None = None
        self._win_ema: dict[tuple[int, int], float] = {}
        # Telemetry counters (B7), surfaced via stats().
        self._promotions_total: int = 0
        self._main_snapshots_total: int = 0
        # Roles whose live weights are owed a reset, applied by begin_round.
        self._pending_reset: set[Role] = set()
        # Genesis snapshot: the initial main enters the pool at train_step=0.
        # The pool is never empty, so the league-exploiter's "pool members
        # only" rule is a hard invariant from round one (no live-main
        # fallback), and the main line has a pool identity for PFSP
        # bookkeeping immediately rather than after the first cadence hit.
        genesis = self.store.add(self._clone(self.main), Role.MAIN, train_step=0)
        self._main_member_id = genesis.member_id
        self._main_snapshots_total += 1

    # ---- per-round matchup decisions ----

    def opponent_for(self, role: Role) -> Policy | None:
        """Return the opponent policy for a learner of ``role`` this episode."""
        return self.sampler.sample(
            pool=self.store.members,
            learner_role=role,
            current_main=self.main,
            # B4: the live main borrows its latest snapshot's member id so PFSP
            # can weight candidates by their measured win-rate vs the main
            # line; exploiters have no pool entry and keep the 0.5 default.
            learner_member_id=self._main_member_id if role == Role.MAIN else None,
        )

    # ---- result reporting (called after each collect round) ----

    def record_main_round(
        self,
        *,
        opponent: Policy | None = None,
        won: bool | None = None,
        n_episodes: int = 0,
    ) -> None:
        """Call once per main-agent collect round (drives periodic snapshots).

        When the round's ``opponent`` was a pool member and the outcome is
        known, the result also feeds the PFSP win-rate bookkeeping (B4).
        """
        self._main_steps += 1
        if self._main_steps % self.config.main_save_every_rounds == 0:
            member = self.store.add(self._clone(self.main), Role.MAIN, train_step=self._main_steps)
            self._main_member_id = member.member_id
            self._main_snapshots_total += 1
        if opponent is not None and won is not None:
            self._record_pool_match(opponent, won, n_episodes, record=True)

    def record_exploiter_match(
        self, role: Role, opponent: Policy, won: bool, *, n_episodes: int = 1
    ) -> None:
        """Record an exploiter round's outcome; promote/reset if threshold met.

        ``n_episodes`` is how many episodes the round played; the rolling
        windows are episode-counted (B8), so one round contributes that many
        entries.
        """
        outcomes = [1.0 if won else 0.0] * max(1, n_episodes)
        if role == Role.MAIN_EXPLOITER:
            self._me_window.extend(outcomes)
            self._me_window = self._me_window[-self.config.promo_window :]
            if len(self._me_window) >= self.config.promo_window // 2:
                wr = sum(self._me_window) / len(self._me_window)
                if wr >= self.config.main_exploiter_promo:
                    self._promote(role, self._clone(self._policy_for_role(role)))
        elif role == Role.LEAGUE_EXPLOITER:
            # record=False: the LE's result must NOT be written into the
            # opponent's win-rate-vs-main bookkeeping (that table feeds PFSP
            # for the main line); we only need the pool-member resolution.
            opp_id = self._record_pool_match(opponent, won, n_episodes, record=False)
            if opp_id is None:
                # Opponent was the live main (or otherwise not in the pool).
                # B6: do NOT count it as a phantom "-1" pool member — only real
                # pool members enter the share denominator.
                return
            window = self._le_window.setdefault(opp_id, [])
            window.extend(outcomes)
            window[:] = window[-self.config.promo_window :]
            self._maybe_promote_league_exploiter()

    # ---- promotion internals ----

    def _record_pool_match(
        self, opponent: Policy, won: bool, n_episodes: int, *, record: bool
    ) -> int | None:
        """Resolve ``opponent`` to a pool id; optionally feed PFSP bookkeeping.

        Returns None when ``opponent`` is not a pool member (e.g. the live
        main). With ``record=True`` (main rounds only) the outcome updates an
        EMA stored on the OPPONENT member keyed by the main's latest member
        id — exactly what OpponentSampler's PFSP lookup reads. Exploiter
        rounds pass ``record=False``: their results are not the main line's
        win-rate and must not pollute that table.
        """
        opp_id = self._find_member_id(opponent)
        if opp_id is None or not record or self._main_member_id is None or n_episodes <= 0:
            return opp_id
        key = (opp_id, self._main_member_id)
        outcome = 1.0 if won else 0.0
        prev = self._win_ema.get(key)
        ema = outcome if prev is None else prev + self.config.win_rate_ema_alpha * (outcome - prev)
        self._win_ema[key] = ema
        self.store.update_win_rate(opp_id, self._main_member_id, ema)
        return opp_id

    def _maybe_promote_league_exploiter(self) -> None:
        """Promote the league exploiter if it beats >= share of the pool."""
        if not self._le_window:
            return
        live_ids = {m.member_id for m in self.store.members}
        beaten = 0
        total = 0
        for opp_id, window in self._le_window.items():
            if opp_id not in live_ids:
                continue  # stale entry for an evicted member — not a real opponent
            if len(window) < 3:  # minimum 3 episodes before a member counts
                continue
            total += 1
            wr = sum(window) / len(window)
            if wr >= self.config.league_exploiter_promo:
                beaten += 1
        if total == 0:
            return
        if beaten / total >= self.config.league_exploiter_share:
            self._promote(Role.LEAGUE_EXPLOITER, self._clone(self.league_exploiter))

    def begin_round(self, role: Role) -> None:
        """Apply any reset owed to ``role`` before it collects again.

        Promotion is decided from a round's OUTCOME, so it necessarily happens
        after that round's batch was collected — but the batch has not been
        learned from yet. Resetting the live weights right there would leave
        the caller holding samples drawn from weights that no longer exist,
        making the ensuing gradient step off-policy through no fault of the
        algorithm. Deferring the reset to the start of the role's next round
        keeps every batch on-policy for the weights it updates.
        """
        if role not in self._pending_reset:
            return
        self._pending_reset.discard(role)
        target = self._policy_for_role(role)
        if self.config.reset_mode == "random":
            self._copy_weights(self._make_policy(), target)
        else:  # "to_main" — locked default
            self._copy_weights(self.main, target)

    def _promote(self, role: Role, snapshot: Policy) -> None:
        """Save the exploiter snapshot to the pool and queue its reset."""
        self.store.add(snapshot, role, train_step=self._main_steps)
        self._promotions_total += 1
        # The reset itself lands in begin_round (see there for why).
        self._pending_reset.add(role)
        # Clear the window so the reset exploiter re-earns its record.
        if role == Role.MAIN_EXPLOITER:
            self._me_window.clear()
        else:
            self._le_window.clear()

    # ---- helpers ----

    def stats(self) -> dict[str, int]:
        """League health counters for the runner's telemetry (B7, AGENTS.md §6).

        The per-role pool composition is reported alongside the total size:
        under the store's role quotas a healthy pool holds a growing main
        history plus at most one snapshot per exploiter role, so any drift
        from that shape (e.g. history not growing between cadence hits) shows
        up here before it distorts sampling.
        """
        return {
            "pool_size": len(self.store),
            "promotions_total": self._promotions_total,
            "main_snapshots_total": self._main_snapshots_total,
            **{f"pool_{role}": count for role, count in self.store.snapshot_summary().items()},
        }

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
