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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from mjai.agents.base import Policy
from mjai.algos.transition import Batch, UpdateStats
from mjai.algos.update_rule import UpdateRule


class BatchSink(Protocol):
    """Where a controller deposits the batch it collected this round."""

    def receive(self, batch: Batch) -> None: ...


@dataclass(frozen=True)
class LearnerBatch:
    """One live learner's share of a collect round, with its logging label.

    ``learner`` is what makes a multi-learner controller safe. A controller
    that rotates roles produces batches belonging to different policies, and a
    gradient step is only valid on the policy that generated its samples — so
    the producing policy travels with the batch instead of being assumed.
    ``label`` namespaces that learner's TensorBoard stats (``train/<label>/*``);
    the Trainer's own policy is always logged under plain ``train/*``.
    """

    batch: Batch
    learner: Policy
    label: str


@dataclass(frozen=True)
class Collected:
    """One collect round's output: per-learner parts + the round's true cost.

    ``parts`` holds one :class:`LearnerBatch` per live learner whose policy
    acted this round and is allowed to train (the league's keep-set). Frozen
    opponents' transitions are dropped by never becoming a part. A mirror
    round has exactly one part pooling both seats; a league round has one part
    per kept learner (one, unless ``train_live_opponents`` routes a live
    opponent's share too).

    ``sampled_steps`` is every decision point the rollout actually played,
    including producers whose transitions were dropped from ``parts``. It is
    the environment-interaction cost of the round, which the part sizes are
    not: counting retained samples would price the same simulation differently
    per mode.
    """

    parts: tuple[LearnerBatch, ...]
    sampled_steps: int


class SelfPlayController(ABC):
    """Decides who plays whom and produces a :class:`Collected` per round.

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
    def collect(self) -> Collected:
        """Play episodes and return the round's transitions + provenance."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Controller identifier (e.g. ``"mirror"``, ``"league"``)."""

    def learners(self) -> tuple[Policy, ...]:
        """Every policy this controller may return a batch for, besides the
        Trainer's own.

        Declared here rather than discovered by ``isinstance`` (AGENTS.md §3.3)
        so the layer that owns update rules can build one per learner without
        the league layer having to know that update rules exist at all
        (AGENTS.md §2: league must not import concrete algos).
        """
        return ()


@dataclass(frozen=True)
class TrainRound:
    """Result of one Trainer.step(): what the round cost + the algo's stats.

    ``batch_size`` is the total samples the round's updates consumed (summed
    over parts); ``env_steps`` is the decision points the rollout played to
    produce them. They coincide under mirror self-play and diverge under any
    controller that discards a producer (e.g. the league dropping frozen
    opponents' transitions).
    """

    batch_size: int
    env_steps: int
    stats_keys: tuple[str, ...]


class Trainer:
    """Owns one (policy, update_rule, controller) triple and runs train rounds.

    The Trainer is the only object the pipeline constructs; it hides the
    UpdateRule/controller wiring from callers. Adding an algorithm or a new
    self-play mode requires no Trainer edits (AGENTS.md §4).

    A controller may collect for several learners (the league's main and its
    two exploiters) — across rounds AND within one round (a live opponent's
    share can come back as its own part). Each learner needs its OWN rule,
    because a gradient step is only valid on the policy that generated the
    batch — applying one learner's samples to another's parameters is an
    off-policy update with a behavior policy nobody chose. Pass those rules as
    ``extra_rules``; the Trainer dispatches each part on its learner's identity
    and refuses a batch from a learner it has no rule for (AGENTS.md §11: no
    silent fallback).
    """

    def __init__(
        self,
        policy: Policy,
        update_rule: UpdateRule,
        controller: SelfPlayController,
        *,
        extra_rules: Sequence[UpdateRule] = (),
    ) -> None:
        self.policy = policy
        self.update_rule = update_rule
        self.controller = controller
        self._rules: tuple[UpdateRule, ...] = (update_rule, *extra_rules)
        self.controller.set_learner(policy)

    def step(self) -> TrainRound:
        """One train round: collect, then update EVERY part's own learner."""
        self.controller.set_learner(self.policy)
        collected = self.controller.collect()
        stats_by_label: dict[str, UpdateStats] = {}
        main_stats: UpdateStats | None = None
        consumed = 0
        for part in collected.parts:
            if part.batch.size == 0:
                continue  # a kept learner produced nothing this round
            stats = self._rule_for(part.learner).step(part.batch)
            stats_by_label[part.label] = stats
            consumed += part.batch.size
            if part.learner is self.policy:
                main_stats = stats
        if not stats_by_label:
            raise RuntimeError(
                f"{type(self.controller).__name__} returned no trainable "
                f"transitions (sampled_steps={collected.sampled_steps}); routing "
                "dropped everything — a controller bug, not a slow round."
            )
        self._last_stats = main_stats
        self._last_stats_by_label = stats_by_label
        key_source = main_stats or next(iter(stats_by_label.values()))
        return TrainRound(
            batch_size=consumed,
            env_steps=collected.sampled_steps,
            stats_keys=tuple(k for k in key_source.__dict__ if k != "extra"),
        )

    def _rule_for(self, learner: Policy) -> UpdateRule:
        """The rule that owns ``learner``, matched by identity."""
        for rule in self._rules:
            if rule.policy is learner:
                return rule
        raise RuntimeError(
            f"{type(self.controller).__name__} collected a batch for a policy with no "
            f"UpdateRule ({type(learner).__name__} at {id(learner):#x}); the Trainer holds "
            f"{len(self._rules)} rule(s). Every learner a controller reports via learners() "
            "must get its own rule — see Trainer's docstring."
        )

    @property
    def last_stats(self) -> UpdateStats | None:
        """The MAIN policy's stats from the most recent :meth:`step` call.

        None on rounds where the main policy had no part (e.g. an exploiter's
        league round with live-opponent routing off). Rounds that update other
        learners report those via :meth:`last_stats_by_label` — ``train/*``
        scalars therefore always describe the main line, never whichever
        exploiter happened to collect (AGENTS.md §6).
        """
        return getattr(self, "_last_stats", None)

    @property
    def last_stats_by_label(self) -> dict[str, UpdateStats]:
        """Every learner's stats from the last step, keyed by part label."""
        return getattr(self, "_last_stats_by_label", {})


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

    def collect(self) -> Collected:
        if self._learner is None:
            raise RuntimeError("MirrorSelfPlay.collect called before set_learner")
        # Both seats play the learner; the runner handles env stepping.
        batch = self._runner.run_episode(learner=self._learner, opponent=self._learner)
        # Both seats belong to the learner, so nothing is discarded and the
        # retained batch IS the simulated cost.
        part = LearnerBatch(batch=batch, learner=self._learner, label="main")
        return Collected(parts=(part,), sampled_steps=batch.size)

    @property
    def name(self) -> str:
        return "mirror"


class RolloutRunnerProtocol(Protocol):
    """The contract a rollout back-end must satisfy for a controller.

    Implemented by :class:`mjai.pipeline.rollout.RolloutWorkerCore` (Step 4).
    Kept here (not imported) so this module has no downward dep into pipeline.

    ``last_episode_count`` is an optional capability: back-ends that expose it
    let controllers keep episode-accurate statistics; controllers must fall
    back gracefully when it is absent.

    ``keep`` names the live-learner policies whose transitions count toward the
    batch-size target (per-producer, min-rule); ``None`` counts everything
    (mirror default). The returned batch always pools every producer — routing
    is the controller's job, via :meth:`Batch.for_producer`.
    """

    last_episode_count: int

    def run_episode(
        self, learner: Policy, opponent: Policy, *, keep: tuple[Policy, ...] | None = None
    ) -> Batch: ...
