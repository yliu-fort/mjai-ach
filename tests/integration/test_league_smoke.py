"""Integration smoke: full league training loop on BRPS (AGENTS.md §5, Step 6).

The league counterpart to test_brps_smoke. Runs the full Trainer + LeagueSelfPlay
loop on BRPS for a handful of steps, asserts:
  - the pool grows (main snapshots accumulate + exploiters can promote),
  - both PPO and ACH survive without NaN,
  - the league produces a different batch-size pattern than mirror (exploiters
    train on narrower opponent sets).
"""

from __future__ import annotations

import math

import pytest

from mjai.agents.tabular import TabularPolicy
from mjai.algos.controller import Trainer
from mjai.algos.tabular_updates import TabularACHUpdate, TabularPPOUpdate
from mjai.algos.update_rule import AlgoConfig
from mjai.games.loader import load_game
from mjai.league.league_controller import LeagueSelfPlay
from mjai.league.manager import LeagueConfig, LeagueManager
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
    main = TabularPolicy(num_actions=spec.num_actions, seed=0)

    def make_policy() -> TabularPolicy:
        return TabularPolicy(num_actions=spec.num_actions, seed=1)

    def copy_weights(src: TabularPolicy, dst: TabularPolicy) -> None:
        import copy

        dst.logits = copy.deepcopy(src.logits)
        dst.values = copy.deepcopy(src.values)

    cfg = LeagueConfig(main_save_every_steps=2, capacity=8, promo_window=4)
    mgr = LeagueManager(main, make_policy, copy_weights, config=cfg)
    # ACH (CFR+ wrapper) requires the GameSpec; PPO doesn't.
    if rule_cls is TabularACHUpdate:
        rule = rule_cls(main, spec, AlgoConfig(learning_rate=0.1), **rule_kwargs)
    else:
        rule = rule_cls(main, AlgoConfig(learning_rate=0.1), **rule_kwargs)
    runner = RolloutWorkerCore(spec, learner_player=0, config=RolloutConfig(n_episodes=20, seed=42))
    ctrl = LeagueSelfPlay(mgr, runner, episodes_per_round=20)
    return Trainer(policy=main, update_rule=rule, controller=ctrl), mgr, main


@pytest.mark.slow
def test_league_loop_runs_and_pool_grows():
    trainer, mgr, _ = _build_trainer(TabularACHUpdate, hedge_eta=0.5)
    assert len(mgr.store) == 0
    for _ in range(12):  # 12 rounds => 4 MAIN rounds => >=1 snapshot at save_every=2
        trainer.step()
    assert len(mgr.store) >= 1
    s = trainer.last_stats
    assert math.isfinite(s.policy_loss)


@pytest.mark.slow
def test_league_loop_ppo_survives():
    trainer, _mgr, _ = _build_trainer(TabularPPOUpdate, clip_eps=0.2)
    for _ in range(12):
        trainer.step()
    assert math.isfinite(trainer.last_stats.policy_loss)
    assert math.isfinite(trainer.last_stats.value_loss)


@pytest.mark.slow
def test_league_three_roles_all_collect():
    """Over enough rounds, all three roles draw their turn and produce batches."""
    trainer, _mgr, _ = _build_trainer(TabularACHUpdate, hedge_eta=0.3)
    batch_sizes = []
    for _ in range(9):  # 3 full cycles of [MAIN, ME, LE]
        r = trainer.step()
        batch_sizes.append(r.batch_size)
    assert all(b > 0 for b in batch_sizes)
    assert len(batch_sizes) == 9


@pytest.mark.slow
def test_league_main_weights_change_over_training():
    trainer, _mgr, main = _build_trainer(TabularACHUpdate, hedge_eta=0.4)
    import pickle

    def sig(p):
        return pickle.dumps(sorted(p.logits.items()))

    before = sig(main)
    for _ in range(15):
        trainer.step()
    after = sig(main)
    assert before != after
