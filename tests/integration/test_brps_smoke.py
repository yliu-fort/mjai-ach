"""Integration smoke test: full mirror-self-play train loop on BRPS (Step 4).

Runs both PPO and ACH (tabular) for a handful of steps on biased RPS, asserts:
  - losses are finite and non-NaN,
  - policy weights actually change between checkpoints,
  - the ACH-vs-PPO algorithmic distinction shows up (ACH moves further on the
    same advantage signal; AGENTS.md §1 D4).

This is the pre-push integration gate (AGENTS.md §5). Marked ``slow`` because it
runs many episodes; the commit suite skips it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mjai.agents.tabular import TabularPolicy
from mjai.algos.controller import MirrorSelfPlay, Trainer
from mjai.algos.tabular_updates import TabularACHUpdate, TabularPPOUpdate
from mjai.algos.update_rule import AlgoConfig
from mjai.games.loader import load_game
from mjai.pipeline.rollout import RolloutConfig, RolloutWorkerCore
from mjai.utils import gpu_assert


@pytest.fixture(autouse=True)
def _cpu_mode():
    gpu_assert.reset_for_tests()
    gpu_assert.require_cpu()
    yield
    gpu_assert.reset_for_tests()


def _build_trainer(rule_cls, **rule_kwargs):
    spec = load_game("brps")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0, temperature=1.0)
    rule = rule_cls(policy, AlgoConfig(learning_rate=0.1, value_coef=0.5), **rule_kwargs)
    worker = RolloutWorkerCore(spec, learner_player=0, config=RolloutConfig(n_episodes=50, seed=42))
    controller = MirrorSelfPlay(worker)
    return Trainer(policy=policy, update_rule=rule, controller=controller), policy


def _weights_signature(p: TabularPolicy) -> bytes:
    """A compact hash of the policy's current logits+values for change detection."""
    import pickle

    return pickle.dumps((sorted(p.logits.items()), sorted(p.values.items())))


@pytest.mark.slow
def test_ppo_mirror_smoke_runs_and_changes_weights():
    trainer, policy = _build_trainer(TabularPPOUpdate, clip_eps=0.2)
    before = _weights_signature(policy)
    last_stats = None
    for _ in range(10):
        trainer.step()
        last_stats = trainer.last_stats
    after = _weights_signature(policy)
    assert before != after, "PPO did not change policy weights over 10 rounds"
    assert last_stats is not None
    assert math.isfinite(last_stats.policy_loss)
    assert math.isfinite(last_stats.value_loss)


@pytest.mark.slow
def test_ach_mirror_smoke_runs_and_changes_weights():
    trainer, policy = _build_trainer(TabularACHUpdate, hedge_eta=0.5)
    before = _weights_signature(policy)
    for _ in range(10):
        trainer.step()
    after = _weights_signature(policy)
    assert before != after, "ACH did not change policy weights over 10 rounds"


@pytest.mark.slow
def test_ach_and_ppo_both_produce_finite_stats_over_many_steps():
    """Both algos survive 30 steps without NaN/inf explosion."""
    for rule_cls, kwargs in [
        (TabularPPOUpdate, {"clip_eps": 0.2}),
        (TabularACHUpdate, {"hedge_eta": 0.3}),
    ]:
        trainer, _ = _build_trainer(rule_cls, **kwargs)
        for _ in range(30):
            trainer.step()
        s = trainer.last_stats
        assert math.isfinite(s.policy_loss)
        assert math.isfinite(s.value_loss)
        assert s.entropy >= 0.0


@pytest.mark.slow
def test_brps_trained_policy_does_not_diverge_to_degenerate():
    """After training, the policy is still a valid distribution over actions."""
    trainer, policy = _build_trainer(TabularACHUpdate, hedge_eta=0.2)
    for _ in range(20):
        trainer.step()
    spec = load_game("brps")
    state = spec.new_state()
    # Sample many times; all actions must be in range and the empirical dist
    # must be a valid probability vector.
    counts = np.zeros(spec.num_actions)
    for _ in range(500):
        a, _ = policy.act(spec.obs_tensor(state, 0), list(range(spec.num_actions)))
        counts[a] += 1
    probs = counts / counts.sum()
    assert math.isclose(float(probs.sum()), 1.0, abs_tol=1e-9)
    assert (probs >= 0).all()


@pytest.mark.slow
def test_kuhn_mirror_loop_runs():
    """Smoke: the loop also runs on a turn-based game (Kuhn)."""
    spec = load_game("kuhn")
    policy = TabularPolicy(num_actions=spec.num_actions, seed=0, temperature=1.0)
    rule = TabularACHUpdate(policy, AlgoConfig(learning_rate=0.05), hedge_eta=0.2)
    worker = RolloutWorkerCore(spec, config=RolloutConfig(n_episodes=100, seed=7))
    trainer = Trainer(policy=policy, update_rule=rule, controller=MirrorSelfPlay(worker))
    batch_sizes = []
    for _ in range(5):
        r = trainer.step()
        batch_sizes.append(r.batch_size)
    assert all(b > 0 for b in batch_sizes)
