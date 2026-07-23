"""Unit tests for LeagueSelfPlay controller (AGENTS.md §5, Step 6)."""

from __future__ import annotations

import random

import pytest

from mjai.agents.base import copy_weights
from mjai.agents.tabular import TabularPolicy
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


def _make_league(
    game_name: str = "brps",
    *,
    shuffle_seats: bool = False,
    train_live_opponents: bool = False,
    n_episodes: int = 10,
    role_schedule: list[Role] | None = None,
    **cfg_kwargs,
):
    spec = load_game(game_name)
    main = TabularPolicy(num_actions=spec.num_actions, seed=0)

    def make_policy() -> TabularPolicy:
        return TabularPolicy(num_actions=spec.num_actions, seed=1)

    # Default main_save_every_rounds=2; caller can override via cfg_kwargs.
    cfg_kwargs.setdefault("main_save_every_rounds", 2)
    cfg = LeagueConfig(**cfg_kwargs)
    mgr = LeagueManager(main, make_policy, copy_weights, config=cfg, rng=random.Random(0))
    runner = RolloutWorkerCore(
        spec,
        learner_player=0,
        config=RolloutConfig(n_episodes=n_episodes, seed=42, shuffle_seats=shuffle_seats),
    )
    ctrl = LeagueSelfPlay(
        mgr,
        runner,
        episodes_per_round=n_episodes,
        role_schedule=role_schedule,
        train_live_opponents=train_live_opponents,
        rng=random.Random(0),
    )
    return ctrl, mgr, main


def _sole_producer(part_batch):
    """The single policy that produced every row of a routed part (homogeneity check)."""
    assert part_batch.size > 0
    assert part_batch.producer_idx is not None
    idxs = {int(i) for i in part_batch.producer_idx}
    assert len(idxs) == 1, f"part is not producer-homogeneous: {idxs}"
    return part_batch.producers[idxs.pop()]


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
    assert collected.parts
    # BRPS, 10 episodes, 2 simultaneous players each => up to 20 transitions
    # before routing; the collector's part is non-empty.
    assert collected.parts[0].batch.size > 0


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


def test_collect_routes_by_producer_identity_not_seat():
    """Every transition in a part was produced by THAT part's learner.

    On a main-exploiter round the opponent is the live main (a different
    object); with live-opponent routing off, only the exploiter's own samples
    are kept — whatever seats they came from.
    """
    ctrl, mgr, main = _make_league(
        shuffle_seats=True, n_episodes=20, role_schedule=[Role.MAIN_EXPLOITER]
    )
    ctrl.set_learner(main)
    collected = ctrl.collect()
    assert len(collected.parts) == 1  # only the collector is kept
    part = collected.parts[0]
    assert part.learner is mgr.main_exploiter
    assert part.label == "main_exploiter"
    assert _sole_producer(part.batch) is mgr.main_exploiter


def test_seat_shuffle_gives_collector_both_perspectives():
    """The fix for half-blindness: across episodes the collector acts from
    BOTH physical seats, and every one of those transitions lands in its part."""
    ctrl, mgr, main = _make_league(
        shuffle_seats=True, n_episodes=40, role_schedule=[Role.MAIN_EXPLOITER]
    )
    ctrl.set_learner(main)
    part = ctrl.collect().parts[0]
    assert part.learner is mgr.main_exploiter
    assert {int(p) for p in part.batch.players} == {0, 1}


def test_no_shuffle_keeps_collector_in_seat0():
    ctrl, _mgr, main = _make_league(n_episodes=20, role_schedule=[Role.MAIN_EXPLOITER])
    ctrl.set_learner(main)
    part = ctrl.collect().parts[0]
    assert {int(p) for p in part.batch.players} == {0}


def test_train_live_opponents_routes_the_live_mains_share_too():
    """ME round + live-opponent routing: the main's own transitions form a
    second part bound for the main's rule, instead of being dropped."""
    ctrl, mgr, main = _make_league(
        n_episodes=20, role_schedule=[Role.MAIN_EXPLOITER], train_live_opponents=True
    )
    ctrl.set_learner(main)
    collected = ctrl.collect()
    by_label = {p.label: p for p in collected.parts}
    assert set(by_label) == {"main_exploiter", "main"}
    assert by_label["main"].learner is mgr.main
    assert _sole_producer(by_label["main"].batch) is mgr.main
    # BRPS: one decision per seat per episode => the two shares are equal-sized.
    assert by_label["main"].batch.size == by_label["main_exploiter"].batch.size


def test_train_live_opponents_off_drops_the_live_mains_share():
    ctrl, _mgr, main = _make_league(
        n_episodes=20, role_schedule=[Role.MAIN_EXPLOITER], train_live_opponents=False
    )
    ctrl.set_learner(main)
    collected = ctrl.collect()
    assert [p.label for p in collected.parts] == ["main_exploiter"]


def test_self_play_round_keeps_both_seats_as_one_part():
    """MAIN round with an empty pool: opponent IS the live main, so both
    seats' transitions belong to it — one part, nothing dropped."""
    ctrl, mgr, main = _make_league(n_episodes=10, role_schedule=[Role.MAIN])
    ctrl.set_learner(main)
    collected = ctrl.collect()
    assert len(collected.parts) == 1
    part = collected.parts[0]
    assert part.learner is mgr.main
    # Retained == simulated: self-play has no foreign producer to drop.
    assert part.batch.size == collected.sampled_steps


def test_league_runs_full_loop_on_kuhn():
    ctrl, _mgr, main = _make_league("kuhn", main_save_every_rounds=2)
    ctrl.set_learner(main)
    for _ in range(6):
        collected = ctrl.collect()
        assert collected.parts  # no exception, non-empty routing


def test_collect_names_the_role_that_produced_the_batch():
    """Each round reports its own collector, not the main agent.

    This is what lets the Trainer update the right weights: without it, two of
    every three rounds would apply an exploiter's samples to the main policy.
    """
    ctrl, mgr, main = _make_league()
    ctrl.set_learner(main)
    collectors = [ctrl.collect().parts[0].learner for _ in range(6)]
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


def test_sampled_steps_counts_the_dropped_producers_too():
    """The opponent seat was simulated; dropping it does not refund its cost."""
    ctrl, _mgr, main = _make_league(n_episodes=10, role_schedule=[Role.MAIN_EXPLOITER])
    ctrl.set_learner(main)
    collected = ctrl.collect()
    # BRPS is simultaneous: both seats act at every decision point, and the ME
    # round's opponent (the live main) is dropped (routing off) — so the full
    # rollout is exactly twice the retained (collector-only) part.
    assert collected.sampled_steps == 2 * collected.parts[0].batch.size
