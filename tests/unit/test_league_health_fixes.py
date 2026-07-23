"""Behavior tests for the league health fixes B1/B2/B4/B6/B7/B8 (AGENTS.md §5).

Each test names the bug it pins down:
  - B1: weight copies go through the generic Policy.snapshot_state/restore_state
    interface — MLP round-trips exactly, mismatched kinds fail loudly, and a
    promotion reset leaves the exploiter identical to the main (KL = 0).
  - B2: the win signal is the mean REAL seat-0 return, not the pooled
    two-seat advantage mean.
  - B4: the live main's latest pool member id reaches the sampler, and main
    rounds vs pool members write win-rates back to the store (PFSP live).
  - B6: the live main is not a phantom "-1" pool member — an empty pool means
    no league-exploiter promotion, however many rounds it wins.
  - B7: league/* health scalars are emitted through the runner-passed writer.
  - B8: promotion windows are counted in EPISODES, not collect rounds.
"""

from __future__ import annotations

import dataclasses
import random

import pytest
import torch
from torch import nn

from mjai.agents.base import Policy, copy_weights
from mjai.agents.mlp import MLPSharedActorCritic
from mjai.agents.tabular import TabularPolicy
from mjai.algos.controller import MirrorSelfPlay
from mjai.algos.transition import Batch, Transition, make_batch
from mjai.games.loader import load_game
from mjai.league.checkpoint_store import Role
from mjai.league.league_controller import LeagueSelfPlay
from mjai.league.manager import LeagueConfig, LeagueManager
from mjai.league.opponent_sampler import LeagueMix, OpponentSampler
from mjai.pipeline.rollout import RolloutConfig, RolloutWorkerCore
from mjai.scripts.experiment_league import log_league_health


def _make_manager(
    *,
    main_save_every_rounds: int = 2,
    promo_window: int = 4,
    capacity: int = 16,
    share: float = 0.70,
) -> tuple[LeagueManager, TabularPolicy]:
    main = TabularPolicy(num_actions=3, seed=0)
    cfg = LeagueConfig(
        main_save_every_rounds=main_save_every_rounds,
        promo_window=promo_window,
        capacity=capacity,
        league_exploiter_share=share,
    )
    mgr = LeagueManager(
        main,
        lambda: TabularPolicy(num_actions=3, seed=1),
        copy_weights,
        config=cfg,
        rng=random.Random(0),
    )
    return mgr, main


def _mlp(seed: int) -> MLPSharedActorCritic:
    """Tiny CPU MLP (explicit device: no GPU assertion in unit tests)."""
    return MLPSharedActorCritic(
        obs_size=4, num_actions=3, hidden_sizes=(8,), activation=nn.ReLU, device="cpu", seed=seed
    )


def _params_equal(a: Policy, b: Policy) -> bool:
    sa, sb = a.snapshot_state(), b.snapshot_state()
    if sa.get("kind") == "nn" and sb.get("kind") == "nn":
        return all(
            torch.equal(x, y)
            for x, y in zip(sa["state_dict"].values(), sb["state_dict"].values(), strict=True)
        )
    return sa == sb


# ---- B1: generic snapshot/restore weight copies ----


def test_b1_mlp_copy_weights_roundtrip_exact():
    src, dst = _mlp(seed=0), _mlp(seed=1)
    assert not _params_equal(src, dst)  # distinct random inits
    copy_weights(src, dst)
    for key, val in src.state_dict().items():
        assert torch.equal(val, dst.state_dict()[key]), f"param {key} differs after copy"


def test_b1_copy_weights_kind_mismatch_fails_loudly():
    """No silent no-op: copying tabular state into an MLP raises (AGENTS.md §11)."""
    with pytest.raises(ValueError, match="kind mismatch"):
        copy_weights(TabularPolicy(num_actions=3), _mlp(seed=0))


def test_b1_promotion_reset_makes_exploiter_identical_to_main():
    """After a promotion reset the exploiter equals the main (KL = 0)."""
    main = _mlp(seed=0)
    seeds = iter(range(100, 999))
    mgr = LeagueManager(
        main,
        lambda: _mlp(seed=next(seeds)),
        copy_weights,
        config=LeagueConfig(promo_window=4, main_exploiter_promo=0.55),
        rng=random.Random(0),
    )
    # Main trains on; the frozen exploiter no longer matches it.
    with torch.no_grad():
        for p in main.parameters():
            p.add_(0.05)
    assert not _params_equal(main, mgr.main_exploiter)
    # One won round of 4 episodes fills the window past the threshold. The
    # reset is applied at the start of the role's next round, so that the batch
    # collected by the promoting round stays on-policy for its own weights.
    mgr.record_exploiter_match(Role.MAIN_EXPLOITER, opponent=main, won=True, n_episodes=4)
    mgr.begin_round(Role.MAIN_EXPLOITER)
    assert _params_equal(main, mgr.main_exploiter)
    obs = [0.25, -0.5, 1.0, 0.75]
    legal = [0, 1, 2]
    assert mgr.main_exploiter.action_logits(obs, legal) == pytest.approx(
        main.action_logits(obs, legal)
    )  # identical distributions => KL = 0


# ---- B2: win signal = mean real seat-0 return ----


def _fixed_batch(seat0_returns: list[float], seat1_return: float, adv: float) -> list[Transition]:
    """Two-seat transitions with known returns; advantages set to a misleading constant."""
    ts: list[Transition] = []
    for r in seat0_returns:
        ts.append(
            Transition(
                obs=[0.0],
                legal_actions=[0],
                action=0,
                logprob=0.0,
                value=0.0,
                reward=r,
                return_=r,
                advantage=adv,
                player=0,
            )
        )
    ts.append(
        Transition(
            obs=[0.0],
            legal_actions=[0],
            action=0,
            logprob=0.0,
            value=0.0,
            reward=seat1_return,
            return_=seat1_return,
            advantage=-adv,
            player=1,
        )
    )
    return ts


class _FixedRunner:
    """RolloutRunnerProtocol double: replays fixed transitions per collect.

    Tags producers the way the real runner would without seat shuffle: seat 0
    acted the learner, seat 1 the opponent.
    """

    def __init__(self, transitions: list[Transition], n_episodes: int) -> None:
        self._transitions = transitions
        self.last_episode_count = n_episodes

    def run_episode(
        self, learner: Policy, opponent: Policy, *, keep: tuple[Policy, ...] | None = None
    ) -> Batch:
        tagged = [
            dataclasses.replace(t, producer=learner if t.player == 0 else opponent)
            for t in self._transitions
        ]
        return make_batch(tagged, num_actions=1)


def _controller_against_fixed_batch(
    transitions: list[Transition], n_episodes: int
) -> tuple[LeagueSelfPlay, LeagueManager]:
    mgr, main = _make_manager(promo_window=20)
    runner = _FixedRunner(transitions, n_episodes)
    ctrl = LeagueSelfPlay(
        mgr,
        runner,
        role_schedule=[Role.MAIN_EXPLOITER],
        rng=random.Random(0),  # type: ignore[arg-type]
    )
    ctrl.set_learner(main)
    return ctrl, mgr


def test_b2_win_signal_uses_seat0_real_returns_not_advantages():
    # Seat-0 returns mostly positive; ALL advantages strongly negative; seat-1
    # return strongly negative. Old signal (pooled mean advantage > 0) => lose;
    # the fixed signal (mean seat-0 return > 0) => win.
    batch = _fixed_batch([1.0, 1.0, -1.0], seat1_return=-5.0, adv=-9.0)
    ctrl, mgr = _controller_against_fixed_batch(batch, n_episodes=3)
    ctrl.collect()
    assert mgr._me_window == [1.0, 1.0, 1.0]  # one win entry per episode (B8)


def test_b2_win_signal_negative_mean_return_counts_as_loss():
    # Seat-0 returns mostly negative; ALL advantages strongly positive.
    batch = _fixed_batch([-1.0, -1.0, 1.0], seat1_return=5.0, adv=9.0)
    ctrl, mgr = _controller_against_fixed_batch(batch, n_episodes=3)
    ctrl.collect()
    assert mgr._me_window == [0.0, 0.0, 0.0]


def test_b2_true_winrate_tracked_per_role():
    batch = _fixed_batch([1.0, -1.0, 1.0, 1.0], seat1_return=-5.0, adv=0.0)
    ctrl, _ = _controller_against_fixed_batch(batch, n_episodes=4)
    ctrl.collect()
    assert ctrl._last_true_winrate[Role.MAIN_EXPLOITER] == pytest.approx(0.75)


# ---- B4: PFSP wiring (member id to sampler + win-rate write-back) ----


def test_b4_main_member_id_reaches_sampler():
    mgr, main = _make_manager(main_save_every_rounds=1)
    mgr.record_main_round()  # live main enters the pool -> member id 0
    captured: dict[str, object] = {}

    class _SpySampler:
        mix = mgr.sampler.mix

        def sample(self, pool, learner_role, current_main, learner_member_id):
            captured["learner_member_id"] = learner_member_id
            return current_main

    mgr.sampler = _SpySampler()  # type: ignore[assignment]
    mgr.opponent_for(Role.MAIN)
    assert captured["learner_member_id"] == 0
    mgr.opponent_for(Role.MAIN_EXPLOITER)
    assert captured["learner_member_id"] is None  # exploiters have no pool entry


def test_b4_main_round_vs_pool_member_updates_win_rate():
    mgr, _ = _make_manager(main_save_every_rounds=1000)
    member = mgr.store.add(TabularPolicy(num_actions=3, seed=9), Role.MAIN)
    mgr._main_member_id = member.member_id  # simulate the main having entered the pool
    mgr.record_main_round(opponent=member.policy, won=True, n_episodes=10)
    assert member.win_rates[member.member_id] == pytest.approx(1.0)
    mgr.record_main_round(opponent=member.policy, won=False, n_episodes=10)
    # EMA(alpha=0.1): 1.0 -> 0.9.
    assert member.win_rates[member.member_id] == pytest.approx(0.9)


def test_b4_measured_win_rates_shift_pfsp_sampling():
    """A measured one-sided opponent is PFSP-sampled less than an unmeasured one."""
    mgr, main = _make_manager()
    dominant = mgr.store.add(TabularPolicy(num_actions=3, seed=3), Role.MAIN)
    unmeasured = mgr.store.add(TabularPolicy(num_actions=3, seed=4), Role.MAIN)
    mgr._main_member_id = 999  # the live main's (synthetic) member id
    mgr.store.update_win_rate(dominant.member_id, 999, 0.99)
    mgr.sampler = OpponentSampler(
        LeagueMix(current_main_weight=0.0, history_weight=1.0, exploiter_weight=0.0),
        rng=random.Random(0),
    )
    counts = {dominant.member_id: 0, unmeasured.member_id: 0}
    for _ in range(600):
        opp = mgr.opponent_for(Role.MAIN)
        for m in (dominant, unmeasured):
            if m.policy is opp:
                counts[m.member_id] += 1
    assert counts[unmeasured.member_id] > counts[dominant.member_id]


def test_b4_exploiter_results_do_not_pollute_main_win_rates():
    """LE rounds resolve the pool id but never write win-rate-vs-main rows."""
    mgr, main = _make_manager()
    member = mgr.store.add(TabularPolicy(num_actions=3, seed=3), Role.MAIN)
    mgr._main_member_id = 999
    for _ in range(3):
        mgr.record_exploiter_match(Role.LEAGUE_EXPLOITER, opponent=member.policy, won=True)
    assert member.win_rates == {}


# ---- B6: no phantom "-1" pool member ----


def test_b6_empty_pool_league_exploiter_cannot_promote():
    """Beating the live main every round never promotes with an empty pool."""
    mgr, main = _make_manager()
    for _ in range(10):
        mgr.record_exploiter_match(Role.LEAGUE_EXPLOITER, opponent=main, won=True, n_episodes=20)
    assert mgr.store.by_role(Role.LEAGUE_EXPLOITER) == []
    assert mgr._le_window == {}  # the live main opens no window at all
    assert mgr.stats()["promotions_total"] == 0


def test_b6_real_pool_members_still_allow_promotion():
    mgr, _ = _make_manager(share=0.70)
    member = mgr.store.add(TabularPolicy(num_actions=3, seed=3), Role.MAIN)
    mgr.record_exploiter_match(
        Role.LEAGUE_EXPLOITER, opponent=member.policy, won=True, n_episodes=3
    )
    assert len(mgr.store.by_role(Role.LEAGUE_EXPLOITER)) == 1  # 1/1 >= 0.70


def test_b6_evicted_member_windows_stop_counting():
    """Windows of evicted members are stale and leave the share denominator."""
    mgr, _ = _make_manager(share=0.75, capacity=2)
    m1 = mgr.store.add(TabularPolicy(num_actions=3, seed=3), Role.MAIN)
    m2 = mgr.store.add(TabularPolicy(num_actions=3, seed=4), Role.MAIN)
    # Lose to m1, beat m2: beaten 1 / total 2 = 0.5 < 0.75 — no promotion.
    mgr.record_exploiter_match(Role.LEAGUE_EXPLOITER, opponent=m1.policy, won=False, n_episodes=3)
    mgr.record_exploiter_match(Role.LEAGUE_EXPLOITER, opponent=m2.policy, won=True, n_episodes=3)
    assert mgr.store.by_role(Role.LEAGUE_EXPLOITER) == []
    # Capacity 2: adding a third main evicts m1 (oldest main, FIFO).
    m3 = mgr.store.add(TabularPolicy(num_actions=3, seed=5), Role.MAIN)
    assert m1 not in mgr.store.members
    # A one-episode result vs m3 triggers the promotion check. With m1's stale
    # window skipped: beaten 1 / total 1 = 1.0 >= 0.75 -> promote. (If the
    # stale window still counted: 1/2 = 0.5 < 0.75 -> no promotion.)
    mgr.record_exploiter_match(Role.LEAGUE_EXPLOITER, opponent=m3.policy, won=True, n_episodes=1)
    assert len(mgr.store.by_role(Role.LEAGUE_EXPLOITER)) == 1


# ---- B8: promotion windows counted in episodes ----


def test_b8_window_counts_episodes_not_rounds():
    """One round of >=window/2 episodes can already trip the threshold."""
    mgr, main = _make_manager(promo_window=20)  # min 10 episodes before a check
    mgr.record_exploiter_match(Role.MAIN_EXPLOITER, opponent=main, won=True, n_episodes=32)
    assert len(mgr.store.by_role(Role.MAIN_EXPLOITER)) == 1
    # Window is capped at promo_window EPISODES (20), not at 20 rounds.
    assert len(mgr._me_window) == 0  # cleared by the promotion reset


def test_b8_single_round_below_min_episodes_does_not_promote():
    mgr, main = _make_manager(promo_window=20)
    mgr.record_exploiter_match(Role.MAIN_EXPLOITER, opponent=main, won=True, n_episodes=5)
    assert mgr.store.by_role(Role.MAIN_EXPLOITER) == []  # 5 < window//2 episodes
    assert len(mgr._me_window) == 5  # five EPISODE entries from one round


# ---- B7: league health telemetry through the runner-passed writer ----


class _FakeWriter:
    """SummaryWriter stand-in capturing add_scalar calls (AGENTS.md §6)."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self.scalars.append((tag, float(scalar_value), global_step))

    def tags(self) -> set[str]:
        return {t for t, _, _ in self.scalars}


def _brps_league_controller() -> tuple[LeagueSelfPlay, LeagueManager, TabularPolicy]:
    spec = load_game("brps")
    main = TabularPolicy(num_actions=spec.num_actions, seed=0)
    mgr = LeagueManager(
        main,
        lambda: TabularPolicy(num_actions=spec.num_actions, seed=1),
        copy_weights,
        config=LeagueConfig(main_save_every_rounds=1),
        rng=random.Random(0),
    )
    runner = RolloutWorkerCore(spec, learner_player=0, config=RolloutConfig(n_episodes=8, seed=42))
    ctrl = LeagueSelfPlay(mgr, runner, episodes_per_round=8, rng=random.Random(0))
    ctrl.set_learner(main)
    return ctrl, mgr, main


def test_b7_health_scalars_logged_with_league_prefix():
    ctrl, mgr, _ = _brps_league_controller()
    writer = _FakeWriter()
    for step in range(3):  # MAIN, MAIN_EXPLOITER, LEAGUE_EXPLOITER rounds
        ctrl.collect()
        log_league_health(writer, step + 1, ctrl)
    tags = writer.tags()
    assert "league/pool_size" in tags
    assert "league/promotions_total" in tags
    assert "league/main_snapshots_total" in tags
    assert "league/exploiter_true_winrate/main_exploiter" in tags
    assert "league/exploiter_true_winrate/league_exploiter" in tags
    # Values are real: one main snapshot after the first (cadence-1) main round.
    pool = {t: v for t, v, _ in writer.scalars}
    assert pool["league/main_snapshots_total"] >= 1.0
    assert pool["league/pool_size"] >= 1.0
    assert mgr.stats()["main_snapshots_total"] >= 1


def test_b7_mirror_controller_emits_no_league_scalars():
    """Capability check, not isinstance (§3.3): mirror has no health_stats."""
    ctrl, _, _ = _brps_league_controller()
    mirror = MirrorSelfPlay(ctrl.runner)
    writer = _FakeWriter()
    log_league_health(writer, 1, mirror)
    assert writer.scalars == []
