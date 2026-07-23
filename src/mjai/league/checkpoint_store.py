"""League checkpoint store: the opponent pool (AGENTS.md §1, §8).

Holds the live set of saved policies the league samples opponents from. Three
roles (AGENTS.md §1 D10 / Step 6 design):

  - ``main``           — past snapshots of the main agent (history), FIFO-bounded.
  - ``main_exploiter`` — promoted when it beat the current main at >=threshold.
  - ``league_exploiter`` — promoted when it beats >threshold of the pool.

Quota policy (locked design): the pool is divided into a main-history quota of
``capacity - 2`` members plus exactly one reserved slot per exploiter role.

  - Adding a MAIN snapshot evicts the oldest MAIN snapshot once the history
    quota is exceeded — exploiters are never touched by main eviction.
  - Adding an exploiter REPLACES the existing member of the same role (at most
    one snapshot per exploiter role, ever): the old snapshot is dropped, and no
    main-history member is evicted to make room. Promotion therefore never
    erases the main line's past.

The bound ``len(pool) <= capacity`` holds by construction: mains are capped on
every main add, each exploiter role is capped at one on every exploiter add.
The store is size-bounded instead of weak-ref'd (AGENTS.md §8): live pool
members are needed immediately for sampling, so bounding is the memory
discipline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

from mjai.agents.base import Policy

#: Slots reserved for exploiters (one per exploiter role); the rest is history.
EXPLOITER_SLOTS = 2
#: Smallest legal capacity: history quota must fit at least one member.
MIN_CAPACITY = EXPLOITER_SLOTS + 1


class Role(StrEnum):
    """Which lineage a pool member belongs to."""

    MAIN = "main"
    MAIN_EXPLOITER = "main_exploiter"
    LEAGUE_EXPLOITER = "league_exploiter"


@dataclass
class PoolMember:
    """One saved policy in the league pool, with provenance + recent perf."""

    policy: Policy
    role: Role
    created_at: float = field(default_factory=time.time)
    train_step: int = 0
    # Cached cross-play win rates against other pool members (keyed by member id).
    # Updated lazily by the sampler; None means "not yet measured".
    win_rates: dict[int, float] = field(default_factory=dict)
    member_id: int = -1  # set by the store on add


class CheckpointStore:
    """The opponent pool, size-bounded with role-aware quotas.

    Args:
        capacity: max pool size (AGENTS.md Step 6: default 16). Must be >=
            :data:`MIN_CAPACITY` so the main-history quota is non-empty.
    """

    def __init__(self, *, capacity: int = 16) -> None:
        if capacity < MIN_CAPACITY:
            raise ValueError(
                f"capacity must be >= {MIN_CAPACITY} (main-history quota of "
                f"capacity-{EXPLOITER_SLOTS} must fit at least one member), got {capacity}"
            )
        self.capacity = capacity
        self.main_quota = capacity - EXPLOITER_SLOTS
        self._members: list[PoolMember] = []
        self._next_id: int = 0

    # ---- introspection ----

    def __len__(self) -> int:
        return len(self._members)

    @property
    def members(self) -> list[PoolMember]:
        """Read-only view of the pool (callers must not mutate the list)."""
        return list(self._members)

    def by_role(self, role: Role) -> list[PoolMember]:
        return [m for m in self._members if m.role == role]

    def main_history(self) -> list[PoolMember]:
        """Past main checkpoints, oldest-first (member_id breaks clock ties)."""
        return sorted(self.by_role(Role.MAIN), key=lambda m: (m.created_at, m.member_id))

    def exploiters(self) -> list[PoolMember]:
        return self.by_role(Role.MAIN_EXPLOITER) + self.by_role(Role.LEAGUE_EXPLOITER)

    # ---- mutation ----

    def add(self, policy: Policy, role: Role, *, train_step: int = 0) -> PoolMember:
        """Add a policy to the pool under its role's quota; returns the member.

        Exploiter adds replace the existing same-role member in place (capped
        at one per role) and never evict main history. Main adds FIFO-evict
        the oldest main once the history quota is exceeded and never touch
        exploiters. The caller is responsible for the promotion criterion
        (this method just stores + enforces the quotas).
        """
        if role != Role.MAIN:
            self._remove_all_of_role(role)
        member = PoolMember(
            policy=policy, role=role, train_step=train_step, member_id=self._next_id
        )
        self._next_id += 1
        self._members.append(member)
        if role == Role.MAIN:
            self._evict_history_over_quota()
        return member

    def _evict_history_over_quota(self) -> None:
        """FIFO-evict the oldest MAIN snapshots beyond the history quota."""
        while len(self.by_role(Role.MAIN)) > self.main_quota:
            self._remove(self.main_history()[0])

    def _remove_all_of_role(self, role: Role) -> None:
        for m in self.by_role(role):
            self._remove(m)

    def _remove(self, member: PoolMember) -> None:
        """Drop ``member`` and scrub win-rate rows that point at its id."""
        self._members.remove(member)
        for m in self._members:
            m.win_rates.pop(member.member_id, None)

    def update_win_rate(self, member_id: int, opponent_id: int, win_rate: float) -> None:
        """Record/refresh the cached win rate of ``member`` vs ``opponent``."""
        for m in self._members:
            if m.member_id == member_id:
                m.win_rates[opponent_id] = float(win_rate)
                return
        # Silently ignore updates for evicted members; the sampler will re-measure.

    def clear(self) -> None:
        """Drop everything (used by tests)."""
        self._members.clear()

    def snapshot_summary(self) -> dict[str, int]:
        """Counts by role, for logging."""
        return {role.value: len(self.by_role(role)) for role in Role}
