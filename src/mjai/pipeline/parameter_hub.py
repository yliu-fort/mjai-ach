"""Parameter hub: versioned weight store + broadcast (AGENTS.md §1 D2, Step 5).

A single source of truth for the learner's latest weights. Workers pull weights
from here at the start of each rollout round; the learner pushes new weights
after each step. Weights are small (state_dict for NN, dict for tabular) and go
into the Ray object store for zero-copy reads.

For tabular policies (Phase 1's primary path) weights are plain dicts; for NN
policies they are torch state_dicts. Both flow through the same interface.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class WeightVersion:
    """One versioned snapshot of learner weights + its version number."""

    version: int
    weights: Any  # dict[str, Any] for tabular; torch state_dict for NN


class ParameterHub:
    """Thread-safe versioned weight store.

    Lives outside Ray (the Ray actor wrapper in pipeline._ray holds an instance
    and forwards calls). The hub itself has no Ray dependency, so it can be
    unit-tested directly.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: WeightVersion | None = None
        self._history: list[WeightVersion] = []  # bounded below
        self._max_history = 8

    def publish(self, weights: Any) -> int:
        """Publish a new weight snapshot; returns the new version number.

        ``weights`` should be an independent snapshot (typically the output of
        ``Policy.snapshot_state()``, which for NN policies stores CPU tensors).
        The defensive ``deepcopy`` here keeps hub-held snapshots decoupled from
        the caller's object graph; for NN weights that means copying CPU
        tensors, not GPU memory — bounded history (8 versions) never pins GPU.
        """
        with self._lock:
            version = 0 if self._latest is None else self._latest.version + 1
            snap = WeightVersion(version=version, weights=copy.deepcopy(weights))
            self._latest = snap
            self._history.append(snap)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]
            return version

    def latest(self) -> WeightVersion | None:
        with self._lock:
            return self._latest

    def get(self, version: int | None = None) -> WeightVersion | None:
        """Fetch a specific version, or the latest if ``version`` is None."""
        with self._lock:
            if version is None or self._latest is None or version == self._latest.version:
                return self._latest
            # Linear scan over the bounded history.
            for snap in self._history:
                if snap.version == version:
                    return snap
            return None

    @property
    def version(self) -> int:
        with self._lock:
            return -1 if self._latest is None else self._latest.version

    @property
    def n_versions(self) -> int:
        with self._lock:
            return len(self._history)


def snapshot_policy_weights(policy: Any) -> Any:
    """Take an independent snapshot of a policy's trainable state.

    Thin delegate to :meth:`Policy.snapshot_state` — the policy itself picks the
    right device for the copy (NN: CPU tensors to avoid GPU-memory accumulation
    in long-lived stores; tabular: deep-copied dicts). This helper is kept for
    backwards compatibility with callers that already use the function form.
    """
    if hasattr(policy, "snapshot_state"):
        return policy.snapshot_state()
    raise TypeError(f"Cannot snapshot policy of type {type(policy)}; needs snapshot_state()")


def restore_policy_weights(policy: Any, snapshot: Any) -> None:
    """Inverse of :func:`snapshot_policy_weights`: write ``snapshot`` into ``policy``."""
    if hasattr(policy, "restore_state"):
        policy.restore_state(snapshot)
        return
    raise TypeError(f"Cannot restore policy of type {type(policy)}; needs restore_state()")
