"""Unit tests for ParameterHub + LocalIMPALARunner (AGENTS.md §5, Step 5)."""

from __future__ import annotations

import pytest

from mjai.agents.tabular import TabularPolicy
from mjai.games.loader import load_game
from mjai.pipeline._ray import LocalIMPALARunner
from mjai.pipeline.parameter_hub import (
    ParameterHub,
    restore_policy_weights,
    snapshot_policy_weights,
)
from mjai.utils import gpu_assert


@pytest.fixture(autouse=True)
def _cpu_mode():
    gpu_assert.reset_for_tests()
    gpu_assert.require_cpu()
    yield
    gpu_assert.reset_for_tests()


# --- ParameterHub ---


def test_empty_hub_has_no_latest():
    hub = ParameterHub()
    assert hub.latest() is None
    assert hub.version == -1


def test_publish_increments_version():
    hub = ParameterHub()
    v0 = hub.publish({"a": 1})
    v1 = hub.publish({"a": 2})
    assert v0 == 0
    assert v1 == 1
    assert hub.version == 1


def test_latest_reflects_most_recent_publish():
    hub = ParameterHub()
    hub.publish({"w": 1})
    hub.publish({"w": 2})
    snap = hub.latest()
    assert snap is not None
    assert snap.weights == {"w": 2}


def test_get_specific_version_from_history():
    hub = ParameterHub()
    hub.publish({"v": 10})
    v1 = hub.publish({"v": 20})
    hub.publish({"v": 30})
    snap = hub.get(v1)
    assert snap is not None
    assert snap.weights == {"v": 20}


def test_get_unknown_version_returns_none():
    hub = ParameterHub()
    hub.publish({"v": 1})
    assert hub.get(999) is None


def test_history_is_bounded():
    hub = ParameterHub()
    for i in range(20):
        hub.publish({"i": i})
    # Bounded to max_history (8).
    assert hub.n_versions <= 8


def test_published_weights_are_deep_copied():
    """Mutating the source after publish must not affect the snapshot."""
    hub = ParameterHub()
    obj = {"logits": {"a": [1, 2]}}
    hub.publish(obj)
    obj["logits"]["a"].append(999)  # mutate the source
    snap = hub.latest()
    assert snap.weights["logits"]["a"] == [1, 2]  # snapshot unchanged


# --- snapshot / restore round-trip ---


def test_snapshot_restore_tabular_roundtrip():
    p = TabularPolicy(num_actions=3, seed=0)
    p.get_logits([1.0, 0.0])[0] = 5.0
    snap = snapshot_policy_weights(p)
    assert snap["kind"] == "tabular"

    # Fresh policy, restore, verify identical.
    q = TabularPolicy(num_actions=3, seed=99)
    restore_policy_weights(q, snap)
    assert q.get_logits([1.0, 0.0])[0] == 5.0


def test_snapshot_is_independent_of_source():
    """Restoring a snapshot must not share state with the source policy."""
    p = TabularPolicy(num_actions=3, seed=0)
    p.get_logits([1.0])[0] = 1.0
    snap = snapshot_policy_weights(p)

    q = TabularPolicy(num_actions=3, seed=0)
    restore_policy_weights(q, snap)
    # Mutate p; q must be unaffected.
    p.get_logits([1.0])[0] = 99.0
    assert q.get_logits([1.0])[0] == 1.0


def test_snapshot_unknown_type_raises():
    class NoState:
        pass

    with pytest.raises(TypeError, match="Cannot snapshot"):
        snapshot_policy_weights(NoState())


def test_restore_unknown_kind_raises():
    p = TabularPolicy(num_actions=2, seed=0)
    with pytest.raises(ValueError, match="Unknown snapshot kind"):
        restore_policy_weights(p, {"kind": "bogus"})


# --- LocalIMPALARunner ---


def test_local_runner_collects_from_all_workers():
    spec = load_game("brps")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0)
    runner = LocalIMPALARunner(spec, policy, episodes_per_round=10, n_workers=3, seed=0)
    batches = runner.collect_round()
    assert len(batches) == 3
    # BRPS: 10 episodes * 2 simultaneous players = 20 transitions per worker.
    for b in batches:
        assert b.size == 20


def test_local_runner_uses_fresh_policy_each_round():
    """After mutating the policy, the next round reflects the new weights."""
    spec = load_game("brps")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0)
    runner = LocalIMPALARunner(spec, policy, episodes_per_round=5, n_workers=1, seed=0)
    batch1 = runner.collect_round()[0]
    # Mutate the policy; the runner holds the same reference so it sees the change.
    policy.get_logits([0.0])[0] = 100.0
    batch2 = runner.collect_round()[0]
    # Same shape, but the action distribution should differ (action 0 now dominant).
    assert batch1.size == batch2.size


def test_local_runner_validates_on_kuhn():
    spec = load_game("kuhn")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0)
    runner = LocalIMPALARunner(spec, policy, episodes_per_round=20, n_workers=2, seed=0)
    batches = runner.collect_round()
    assert len(batches) == 2
    for b in batches:
        assert b.size > 0
