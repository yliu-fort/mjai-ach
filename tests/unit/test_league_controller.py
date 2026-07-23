"""Unit tests for LeagueSelfPlay controller (AGENTS.md §5, Step 6)."""

from __future__ import annotations

import random

import pytest

from mjai.agents.base import copy_weights
from mjai.agents.tabular import TabularPolicy
from mjai.algos.transition import Batch
from mjai.games.loader import load_game
from mjai.league.checkpoint_store import Role
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


def _make_league(game_name: str = "brps", **cfg_kwargs):
    spec = load_game(game_name)
    main = TabularPolicy(num_actions=spec.num_actions, seed=0)

    def make_policy() -> TabularPolicy:
        return TabularPolicy(num_actions=spec.num_actions, seed=1)

    # Default main_save_every_rounds=2; caller can override via cfg_kwargs.
    cfg_kwargs.setdefault("main_save_every_rounds", 2)
    cfg = LeagueConfig(**cfg_kwargs)
    mgr = LeagueManager(main, make_policy, copy_weights, config=cfg, rng=random.Random(0))
    runner = RolloutWorkerCore(spec, learner_player=0, config=RolloutConfig(n_episodes=10, seed=42))
    ctrl = LeagueSelfPlay(mgr, runner, episodes_per_round=10, rng=random.Random(0))
    return ctrl, mgr, main


def test_name_is_league():
    ctrl, _, _ = _make_league()
    assert ctrl.name == "league"


def test_collect_before_set_learner_raises():
    ctrl, _, _ = _make_league()
    with pytest.raises(RuntimeError, match="set_learner"):
        ctrl.collect()


def test_collect_returns_batch_with_learner_transitions():
    ctrl, _, main = _make_league()
    ctrl.set_learner(main)
    collected = ctrl.collect()
    assert isinstance(collected.batch, Batch)
    # BRPS, 10 episodes, 2 simultaneous players each => up to 20 transitions
    # before filtering; after filtering to seat 0 => ~10.
    assert collected.batch.size > 0


def test_collect_rotates_through_roles():
    """The default schedule cycles MAIN -> ME -> LE -> MAIN ..."""
    ctrl, _mgr, main = _make_league()
    ctrl.set_learner(main)
    ctrl.collect()  # round 0 -> MAIN
    assert ctrl._round_idx == 1
    ctrl.collect()  # round 1 -> MAIN_EXPLOITER
    ctrl.collect()  # round 2 -> LEAGUE_EXPLOITER
    assert ctrl._round_idx == 3
    ctrl.collect()  # round 3 -> MAIN again
    assert ctrl._round_idx == 4


def test_main_rounds_accumulate_snapshots():
    ctrl, mgr, main = _make_league(main_save_every_rounds=2)
    ctrl.set_learner(main)
    # MAIN is collected every 3rd round; force two MAIN rounds to trigger a snapshot.
    for _ in range(6):
        ctrl.collect()
    assert len(mgr.store.by_role(Role.MAIN)) >= 1


def test_set_learner_updates_manager_main_pointer():
    ctrl, mgr, _main = _make_league()
    new_main = TabularPolicy(num_actions=3, seed=99)
    ctrl.set_learner(new_main)
    assert mgr.main is new_main


def test_collect_filters_to_learner_seat_only():
    """Returned batch contains only seat-0 transitions (the learner's)."""
    ctrl, _, main = _make_league()
    ctrl.set_learner(main)
    batch = ctrl.collect().batch
    if batch.size > 0:
        # All transitions belong to player 0.
        assert (batch.players == 0).all()


def test_league_runs_full_loop_on_kuhn():
    ctrl, _mgr, main = _make_league("kuhn", main_save_every_rounds=2)
    ctrl.set_learner(main)
    for _ in range(6):
        collected = ctrl.collect()
        assert collected.batch.size >= 0  # no exception


def test_collect_names_the_role_that_produced_the_batch():
    """Each round reports its own collector, not the main agent.

    This is what lets the Trainer update the right weights: without it, two of
    every three rounds would apply an exploiter's samples to the main policy.
    """
    ctrl, mgr, main = _make_league()
    ctrl.set_learner(main)
    collectors = [ctrl.collect().learner for _ in range(6)]
    expected = [mgr.main, mgr.main_exploiter, mgr.league_exploiter] * 2
    assert collectors == expected


def test_learners_declares_every_collecting_role():
    ctrl, mgr, main = _make_league()
    ctrl.set_learner(main)
    assert set(map(id, ctrl.learners())) == {
        id(mgr.main),
        id(mgr.main_exploiter),
        id(mgr.league_exploiter),
    }


def test_sampled_steps_counts_the_discarded_seat_too():
    """The opponent seat was simulated; dropping it does not refund its cost."""
    ctrl, _, main = _make_league()
    ctrl.set_learner(main)
    collected = ctrl.collect()
    # BRPS is simultaneous: both seats act at every decision point, so the
    # full rollout is exactly twice the retained (seat-0-only) batch.
    assert collected.sampled_steps == 2 * collected.batch.size
