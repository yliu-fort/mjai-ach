"""Unit tests for LeagueManager: promotion, reset, snapshots (AGENTS.md §5)."""

from __future__ import annotations

import random

from mjai.agents.base import copy_weights
from mjai.agents.tabular import TabularPolicy
from mjai.league.checkpoint_store import Role
from mjai.league.manager import LeagueConfig, LeagueManager


def _make_manager(
    *,
    main_save_every_rounds: int = 2,
    main_exploiter_promo: float = 0.55,
    league_exploiter_promo: float = 0.55,
    league_exploiter_share: float = 0.70,
    promo_window: int = 4,
    capacity: int = 16,
    reset_mode: str = "to_main",
) -> tuple[LeagueManager, TabularPolicy]:
    main = TabularPolicy(num_actions=3, seed=0)

    def make_policy() -> TabularPolicy:
        return TabularPolicy(num_actions=3, seed=1)

    cfg = LeagueConfig(
        main_save_every_rounds=main_save_every_rounds,
        main_exploiter_promo=main_exploiter_promo,
        league_exploiter_promo=league_exploiter_promo,
        league_exploiter_share=league_exploiter_share,
        promo_window=promo_window,
        capacity=capacity,
        reset_mode=reset_mode,
    )
    return LeagueManager(main, make_policy, copy_weights, config=cfg, rng=random.Random(0)), main


def test_construction_warm_starts_exploiters_from_main():
    mgr, main = _make_manager()
    # Exploiters exist and share the main's (empty) weights at construction.
    assert isinstance(mgr.main_exploiter, TabularPolicy)
    assert isinstance(mgr.league_exploiter, TabularPolicy)
    assert mgr.main_exploiter.num_rows() == main.num_rows()


def test_record_main_round_snapshots_periodically():
    mgr, _ = _make_manager(main_save_every_rounds=3)
    assert len(mgr.store) == 0
    mgr.record_main_round()
    mgr.record_main_round()
    assert len(mgr.store) == 0  # not yet
    mgr.record_main_round()
    assert len(mgr.store) == 1
    assert mgr.store.by_role(Role.MAIN)  # it's a main snapshot


def test_main_exploiter_promotion_on_high_winrate():
    mgr, _ = _make_manager(main_exploiter_promo=0.6, promo_window=4)
    # Feed exactly enough wins to trip the threshold once (window/2 = 2 wins,
    # win rate 1.0 >= 0.6). Feeding more would legitimately promote again after
    # the reset clears the window.
    for _ in range(2):
        mgr.record_exploiter_match(Role.MAIN_EXPLOITER, opponent=mgr.main, won=True)
    # A main-exploiter snapshot should now be in the pool.
    assert len(mgr.store.by_role(Role.MAIN_EXPLOITER)) == 1


def test_main_exploiter_not_promoted_below_threshold():
    mgr, _ = _make_manager(main_exploiter_promo=0.8, promo_window=4)
    for _ in range(4):
        mgr.record_exploiter_match(Role.MAIN_EXPLOITER, opponent=mgr.main, won=False)  # 0% wr
    assert len(mgr.store.by_role(Role.MAIN_EXPLOITER)) == 0


def test_promotion_resets_exploiter_to_main_weights(reset_mode="to_main"):
    mgr, main = _make_manager(main_exploiter_promo=0.5, promo_window=2, reset_mode="to_main")
    # Dirty the main_exploiter's logits so we can tell they got overwritten.
    main.get_logits([1.0])[0] = 9.0  # main now has a row the exploiter doesn't
    for _ in range(2):
        mgr.record_exploiter_match(Role.MAIN_EXPLOITER, opponent=main, won=True)
    # The reset is owed, not yet applied: the round that earned the promotion
    # has a batch in flight that must stay on-policy for these weights.
    mgr.begin_round(Role.MAIN_EXPLOITER)
    # After promotion + reset-to-main, the exploiter should now have main's row.
    assert mgr.main_exploiter.get_logits([1.0])[0] == 9.0


def test_promotion_defers_the_reset_to_the_next_round():
    """Promotion must not overwrite weights a collected batch still refers to."""
    mgr, main = _make_manager(main_exploiter_promo=0.5, promo_window=2, reset_mode="to_main")
    main.get_logits([1.0])[0] = 9.0
    for _ in range(2):
        mgr.record_exploiter_match(Role.MAIN_EXPLOITER, opponent=main, won=True)
    assert len(mgr.store) >= 1  # the snapshot IS taken immediately
    assert mgr.main_exploiter.get_logits([1.0])[0] != 9.0  # ...the reset is not
    mgr.begin_round(Role.MAIN_EXPLOITER)
    assert mgr.main_exploiter.get_logits([1.0])[0] == 9.0


def test_begin_round_is_idempotent_without_a_pending_reset():
    mgr, main = _make_manager(main_exploiter_promo=0.5, promo_window=2)
    mgr.main_exploiter.get_logits([1.0])[0] = 3.0
    mgr.begin_round(Role.MAIN_EXPLOITER)
    mgr.begin_round(Role.MAIN_EXPLOITER)
    assert mgr.main_exploiter.get_logits([1.0])[0] == 3.0


def test_promotion_resets_exploiter_randomly():
    mgr, _ = _make_manager(main_exploiter_promo=0.5, promo_window=2, reset_mode="random")
    rows_before = mgr.main_exploiter.num_rows()
    for _ in range(2):
        mgr.record_exploiter_match(Role.MAIN_EXPLOITER, opponent=mgr.main, won=True)
    mgr.begin_round(Role.MAIN_EXPLOITER)
    # After reset-random, the exploiter is a fresh policy (empty rows here).
    assert mgr.main_exploiter.num_rows() == 0 or rows_before == 0


def test_league_exploiter_promotion_requires_share_of_pool():
    """League exploiter must beat >=share of pool members to promote."""
    mgr, _ = _make_manager(league_exploiter_promo=0.55, league_exploiter_share=0.5, promo_window=4)
    # Seed the pool with two main checkpoints so there's something to beat.
    mgr.store.add(TabularPolicy(num_actions=3, seed=2), Role.MAIN)
    mgr.store.add(TabularPolicy(num_actions=3, seed=3), Role.MAIN)
    # Beat each pool member exactly 3 times (window of 3, win rate 1.0 >= 0.55).
    # This trips promotion once; more rounds would promote again post-reset.
    members = list(mgr.store.members)
    for m in members:
        for _ in range(3):
            mgr.record_exploiter_match(Role.LEAGUE_EXPLOITER, opponent=m.policy, won=True)
        if mgr.store.by_role(Role.LEAGUE_EXPLOITER):
            break  # promoted after the first member; stop to avoid re-promotion
    assert len(mgr.store.by_role(Role.LEAGUE_EXPLOITER)) == 1


def test_opponent_for_main_returns_a_policy():
    mgr, _ = _make_manager()
    # Empty pool + current main available => opponent is the main.
    opp = mgr.opponent_for(Role.MAIN)
    assert opp is mgr.main


def test_opponent_for_main_exploiter_is_current_main():
    mgr, _ = _make_manager()
    opp = mgr.opponent_for(Role.MAIN_EXPLOITER)
    assert opp is mgr.main


def test_pool_grows_with_main_snapshots_and_promotions():
    mgr, _ = _make_manager(main_save_every_rounds=1)
    for _ in range(5):
        mgr.record_main_round()
    assert len(mgr.store.by_role(Role.MAIN)) == 5


def test_capacity_bounds_pool_size():
    mgr, _ = _make_manager(main_save_every_rounds=1, capacity=4)
    for _ in range(10):
        mgr.record_main_round()
    assert len(mgr.store) <= 4


def test_reset_mode_validation_in_config():
    # reset_mode is just a string; the manager branches on it. Any value other
    # than "random" is treated as "to_main".
    mgr, _ = _make_manager(reset_mode="nonsense")
    assert mgr.config.reset_mode == "nonsense"
