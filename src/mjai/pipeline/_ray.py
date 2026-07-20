"""Ray IMPALA executor: thin actor wrappers over the cores (AGENTS.md §1 D2, Step 5).

The Ray-free cores (RolloutWorkerCore, ParameterHub, LearnerCore) carry all the
real logic; this module adds ``@ray.remote`` actors that forward to them. Unit
tests exercise the cores directly; the Ray topology is wired here and exercised
in the (slow, Ray-marked) integration tests + via ``ray.init(local_mode=True)``
for debugging.

Topology (IMPALA-style, async):
  - N RolloutWorker actors on CPU: each pulls latest weights from the
    ParameterHub, plays ``episodes_per_round`` of self-play, returns a Batch.
  - 1 Learner (the in-process Trainer) on GPU: pulls batches, runs UpdateRule,
    publishes new weights to the hub.
  - ParameterHub as a Ray actor: zero-copy weight broadcast via the object store.
"""

from __future__ import annotations

from typing import Any

from mjai.agents.base import Policy
from mjai.algos.transition import Batch
from mjai.games.loader import GameSpec
from mjai.pipeline.parameter_hub import ParameterHub
from mjai.pipeline.rollout import RolloutConfig, RolloutWorkerCore


def _maybe_ray():
    """Import ray lazily so importing this module doesn't require ray installed."""
    import ray

    return ray


# --- Worker actor: wraps RolloutWorkerCore, refreshes weights each round ---


def make_worker_actor():  # pragma: no cover -- requires a running Ray cluster
    """Build a Ray-decorated RolloutWorker class.

    Kept as a function rather than a module-level decorator so that importing
    this module never triggers ``import ray`` (Ray init is the caller's job).
    """
    ray = _maybe_ray()

    @ray.remote
    class RolloutWorker:
        """One CPU rollout actor. Holds a RolloutWorkerCore + a policy handle.

        The policy is *stateless* across calls — each round the worker fetches
        fresh weights from the hub, plays episodes, and returns the Batch.
        """

        def __init__(self, spec: GameSpec, *, episodes_per_round: int = 50, seed: int = 0) -> None:
            self._core = RolloutWorkerCore(
                spec,
                learner_player=0,
                config=RolloutConfig(n_episodes=episodes_per_round, seed=seed),
            )
            self._policy: Policy | None = None

        def set_policy(self, policy: Policy) -> None:
            self._policy = policy

        def collect(self) -> Batch:
            assert self._policy is not None, "set_policy before collect"
            # Mirror self-play: learner is both seats.
            return self._core.run_episode(self._policy, self._policy)

    return RolloutWorker


def make_hub_actor():  # pragma: no cover -- requires a running Ray cluster
    """Build a Ray-decorated ParameterHub actor."""
    ray = _maybe_ray()

    @ray.remote
    class ParameterHubActor:
        def __init__(self) -> None:
            self._hub = ParameterHub()

        def publish(self, weights: Any) -> int:
            return self._hub.publish(weights)

        def latest(self):
            return self._hub.latest()

        def get(self, version: int | None = None):
            return self._hub.get(version)

        @property
        def version(self) -> int:
            return self._hub.version

    return ParameterHubActor


class LocalIMPALARunner:
    """In-process IMPALA topology for unit tests and small experiments.

    This is the Ray-free fallback: it runs the same async worker→learner→hub
    loop single-process. Useful when Ray is unavailable (e.g. CI sandbox) and
    for debugging the topology without actor overhead.

    The real Ray runner (Step 5 follow-up / Phase 3) swaps LocalIMPALARunner for
    one that uses the actors above; the trainer/policy/hub interfaces are
    identical.
    """

    def __init__(
        self,
        spec: GameSpec,
        policy: Policy,
        *,
        episodes_per_round: int = 50,
        n_workers: int = 2,
        seed: int = 0,
    ) -> None:
        self.spec = spec
        self.policy = policy
        self.hub = ParameterHub()
        self._workers = [
            RolloutWorkerCore(
                spec,
                learner_player=0,
                config=RolloutConfig(n_episodes=episodes_per_round, seed=seed + i),
            )
            for i in range(n_workers)
        ]

    def collect_round(self) -> list[Batch]:
        """Each worker plays a round with the current policy; returns the batches."""
        batches = [w.run_episode(self.policy, self.policy) for w in self._workers]
        return batches
