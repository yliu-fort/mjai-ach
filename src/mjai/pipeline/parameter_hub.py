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
        """Publish a new weight snapshot; returns the new version number."""
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
    """Deep-copy a policy's trainable state into a plain structure.

    For TabularPolicy: {logits: ..., values: ...}. For NN (torch nn.Module):
    state_dict(). Returned object is what gets published to the hub.
    """
    if hasattr(policy, "logits") and hasattr(policy, "values"):
        return {
            "kind": "tabular",
            "logits": copy.deepcopy(policy.logits),
            "values": copy.deepcopy(policy.values),
        }
    if hasattr(policy, "state_dict"):
        # NN policy: state_dict is already a plain dict of tensors.
        return {"kind": "nn", "state_dict": copy.deepcopy(policy.state_dict())}
    raise TypeError(
        f"Cannot snapshot policy of type {type(policy)}; needs logits+values or state_dict"
    )


def restore_policy_weights(policy: Any, snapshot: Any) -> None:
    """Inverse of :func:`snapshot_policy_weights`: write ``snapshot`` into ``policy``."""
    if snapshot["kind"] == "tabular":
        policy.logits = copy.deepcopy(snapshot["logits"])
        policy.values = copy.deepcopy(snapshot["values"])
    elif snapshot["kind"] == "nn":
        policy.load_state_dict(copy.deepcopy(snapshot["state_dict"]))
    else:
        raise ValueError(f"Unknown snapshot kind: {snapshot['kind']}")
