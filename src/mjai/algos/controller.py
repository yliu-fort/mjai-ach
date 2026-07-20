"""Self-play controller interface + Trainer composition (AGENTS.md §2, §3).

The controller/Trainer/UpdateRule split keeps the dependency graph acyclic:

  - ``UpdateRule`` — pure gradient step on a Policy from a Batch. Stateless
    w.r.t. self-play topology.
  - ``SelfPlayController`` (this module, an ABC) — collects a Batch by deciding
    who plays whom each episode. ``MirrorSelfPlay`` lives here; league variants
    live in :mod:`mjai.league` (and depend only on this interface, never on a
    concrete UpdateRule — see import-linter contract "League does not depend on
    concrete algos").
  - ``Trainer`` — composes one Policy + one UpdateRule + one controller and
    runs the train loop. Pipeline code (Step 4+) builds Trainers; it never
    reaches into UpdateRules directly.

This module is the lowest point of the algos layer that knows about self-play,
so it's where league can hook in without circular imports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from mjai.agents.base import Policy
from mjai.algos.transition import Batch, UpdateStats
from mjai.algos.update_rule import UpdateRule


class BatchSink(Protocol):
    """Where a controller deposits the batch it collected this round."""

    def receive(self, batch: Batch) -> None: ...


class SelfPlayController(ABC):
    """Decides who plays whom and produces a Batch per :meth:`collect`.

    Implementations:
      - :class:`MirrorSelfPlay` (below) — both seats = current policy.
      - :class:`mjai.league.league_controller.LeagueSelfPlay` (Step 6) — mixed
        opponent pool with PFSP sampling.

    The controller is given the learner's current policy on each round via
    :meth:`set_learner` so it can use fresh weights for rollout without the
    Trainer having to know how the controller caches them.
    """

    @abstractmethod
    def set_learner(self, policy: Policy) -> None:
        """Inject the up-to-date learner policy for the next rollout round."""

    @abstractmethod
    def collect(self) -> Batch:
        """Play episodes and return the transitions as a :class:`Batch`."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Controller identifier (e.g. ``"mirror"``, ``"league"``)."""


@dataclass(frozen=True)
class TrainRound:
    """Result of one Trainer.step(): the batch consumed + the algo's stats."""

    batch_size: int
    stats_keys: tuple[str, ...]


class Trainer:
    """Owns one (policy, update_rule, controller) triple and runs train rounds.

    The Trainer is the only object the pipeline constructs; it hides the
    UpdateRule/controller wiring from callers. Adding an algorithm or a new
    self-play mode requires no Trainer edits (AGENTS.md §4).
    """

    def __init__(
        self,
        policy: Policy,
        update_rule: UpdateRule,
        controller: SelfPlayController,
    ) -> None:
        self.policy = policy
        self.update_rule = update_rule
        self.controller = controller
        self.controller.set_learner(policy)

    def step(self) -> TrainRound:
        """One train round: collect a batch under the current policy, then update."""
        self.controller.set_learner(self.policy)
        batch = self.controller.collect()
        stats = self.update_rule.step(batch)
        self._last_stats = stats
        return TrainRound(
            batch_size=batch.size,
            stats_keys=tuple(k for k in stats.__dict__ if k != "extra"),
        )

    @property
    def last_stats(self) -> UpdateStats | None:
        """Stats from the most recent :meth:`step` call."""
        return getattr(self, "_last_stats", None)


# -----------------------------------------------------------------------------


class MirrorSelfPlay(SelfPlayController):
    """Pure mirror self-play: both seats controlled by the current learner.

    This is the ACH paper's training topology (AGENTS.md §1 D4). The batch
    pools transitions from both players, since the same policy learns from
    every seat. The actual episode rollout is delegated to a RolloutRunner
    passed in at construction (Step 4 provides the OpenSpiel-backed runner).
    """

    def __init__(self, runner: RolloutRunnerProtocol) -> None:
        self._runner = runner
        self._learner: Policy | None = None

    def set_learner(self, policy: Policy) -> None:
        self._learner = policy

    def collect(self) -> Batch:
        if self._learner is None:
            raise RuntimeError("MirrorSelfPlay.collect called before set_learner")
        # Both seats play the learner; the runner handles env stepping.
        return self._runner.run_episode(learner=self._learner, opponent=self._learner)

    @property
    def name(self) -> str:
        return "mirror"


class RolloutRunnerProtocol(Protocol):
    """The contract a rollout back-end must satisfy for a controller.

    Implemented by :class:`mjai.pipeline.rollout.RolloutWorkerCore` (Step 4).
    Kept here (not imported) so this module has no downward dep into pipeline.
    """

    def run_episode(self, learner: Policy, opponent: Policy) -> Batch: ...
