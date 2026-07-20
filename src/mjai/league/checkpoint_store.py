"""League checkpoint store: the opponent pool (AGENTS.md §1, §8).

Holds the live set of saved policies the league samples opponents from. Three
roles (AGENTS.md §1 D10 / Step 6 design):

  - ``main``           — past snapshots of the main agent (history), FIFO-evicted.
  - ``main_exploiter`` — promoted when it beat the current main at >=threshold.
  - ``league_exploiter`` — promoted when it beats >threshold of the pool.

Eviction policy (AGENTS.md §8): history is FIFO-bounded; exploiters are rare and
valuable so we keep the top-N by win-rate against the current main rather than
auto-evicting them. Weak refs are NOT used for live pool members (we need them
immediately for sampling); the store is size-bounded instead, which is the
memory discipline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

from mjai.agents.base import Policy


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
    """The opponent pool, size-bounded with role-aware eviction.

    Args:
        capacity: max pool size (AGENTS.md Step 6: default 16).
        exploiter_keep: how many exploiters of each kind to keep when over
            capacity (we evict the lowest-win-rate ones above this count).
    """

    def __init__(self, *, capacity: int = 16, exploiter_keep: int = 4) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.exploiter_keep = exploiter_keep
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
        """Past main checkpoints, oldest-first."""
        return sorted(self.by_role(Role.MAIN), key=lambda m: m.created_at)

    def exploiters(self) -> list[PoolMember]:
        return self.by_role(Role.MAIN_EXPLOITER) + self.by_role(Role.LEAGUE_EXPLOITER)

    # ---- mutation ----

    def add(self, policy: Policy, role: Role, *, train_step: int = 0) -> PoolMember:
        """Add a policy to the pool, then evict if over capacity.

        Returns the added member. Caller is responsible for the promotion
        criterion (this method just stores + bounds).
        """
        member = PoolMember(
            policy=policy, role=role, train_step=train_step, member_id=self._next_id
        )
        self._next_id += 1
        self._members.append(member)
        self._evict_if_needed()
        return member

    def _evict_if_needed(self) -> None:
        """Enforce the capacity, evicting history (FIFO) before exploiters."""
        while len(self._members) > self.capacity:
            # Prefer evicting the oldest MAIN checkpoint.
            mains = self.main_history()
            if mains:
                self._members.remove(mains[0])
                continue
            # No main left to evict — trim the weakest exploiter.
            exploiters = self.exploiters()
            if len(exploiters) <= 1:
                # Only exploiters remain and we're still over capacity; keep the
                # newest one (the pool can't go empty).
                newest = max(exploiters, key=lambda m: m.created_at, default=None)
                self._members = [newest] if newest is not None else []
                break
            weakest = min(exploiters, key=self._exploiter_score)
            self._members.remove(weakest)

    def _exploiter_score(self, m: PoolMember) -> float:
        """Higher = more worth keeping. Mean win rate; -inf if unmeasured."""
        if not m.win_rates:
            return float("-inf")
        return sum(m.win_rates.values()) / len(m.win_rates)

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
