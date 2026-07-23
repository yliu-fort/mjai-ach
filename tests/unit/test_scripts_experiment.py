"""Unit tests for ExperimentConfig league knobs (AGENTS.md §5, F2).

Covers: defaults preserve the previous hard-coded LeagueConfig/LeagueMix
behavior, every knob reaches the built LeagueManager/OpponentSampler via
build_controller, and invalid knob combinations fail loudly (§9).
"""

from __future__ import annotations

import random

import pytest

from mjai.games.loader import load_game
from mjai.league.checkpoint_store import Role
from mjai.league.league_controller import LeagueSelfPlay
from mjai.league.manager import LeagueConfig
from mjai.league.opponent_sampler import LeagueMix
from mjai.scripts.experiment import (
    ExperimentConfig,
    build_controller,
    build_policy,
    build_update_rule,
    resolve_theta,
)


def _base_cfg(**overrides: object) -> ExperimentConfig:
    """Minimal league-mode config (tabular policy keeps the test CPU-only)."""
    return ExperimentConfig(game="kuhn", algo="ach", self_play_mode="league", **overrides)  # type: ignore[arg-type]


def _build_league_controller(cfg: ExperimentConfig) -> LeagueSelfPlay:
    spec = load_game("kuhn")
    policy = build_policy(spec, cfg, seed=0)
    controller = build_controller(spec, policy, cfg, rng=random.Random(0))
    assert isinstance(controller, LeagueSelfPlay)
    return controller


def test_defaults_preserve_hardcoded_league_behavior():
    """No knobs passed -> LeagueConfig/LeagueMix identical to the code defaults."""
    league_cfg = _build_league_controller(_base_cfg()).manager.config
    reference = LeagueConfig(capacity=16)
    assert league_cfg.capacity == reference.capacity
    # B3: the pool cadence is an independent knob counted in main rounds;
    # the default matches LeagueConfig's, NOT save_every_steps.
    assert _base_cfg().league_main_save_every_rounds == 200
    assert league_cfg.main_save_every_rounds == reference.main_save_every_rounds == 200
    assert league_cfg.main_exploiter_promo == reference.main_exploiter_promo
    assert league_cfg.league_exploiter_promo == reference.league_exploiter_promo
    assert league_cfg.league_exploiter_share == reference.league_exploiter_share
    assert league_cfg.promo_window == reference.promo_window
    assert league_cfg.reset_mode == reference.reset_mode
    assert league_cfg.mix == LeagueMix()


def test_league_knobs_reach_manager_and_sampler():
    """Every YAML knob lands on the built LeagueConfig + OpponentSampler mix."""
    cfg = _base_cfg(
        league_mix_current_main=0.6,
        league_mix_history=0.25,
        league_mix_exploiter=0.15,
        league_main_exploiter_promo=0.6,
        league_league_exploiter_promo=0.65,
        league_exploiter_share=0.8,
        league_promo_window=10,
        league_reset_mode="random",
        league_main_save_every_rounds=7,
    )
    manager = _build_league_controller(cfg).manager
    assert manager.config.mix == LeagueMix(0.6, 0.25, 0.15)
    assert manager.sampler.mix == manager.config.mix  # sampler got the same mix
    assert manager.config.main_exploiter_promo == 0.6
    assert manager.config.league_exploiter_promo == 0.65
    assert manager.config.league_exploiter_share == 0.8
    assert manager.config.promo_window == 10
    assert manager.config.reset_mode == "random"
    assert manager.config.main_save_every_rounds == 7


def test_mix_weights_not_summing_to_one_raise():
    with pytest.raises(ValueError, match="sum to 1.0"):
        _base_cfg(league_mix_current_main=0.9)


def test_invalid_reset_mode_raises():
    with pytest.raises(ValueError, match="league_reset_mode"):
        _base_cfg(league_reset_mode="to_pool")


def test_valid_reset_modes_accepted():
    assert _base_cfg(league_reset_mode="to_main").league_reset_mode == "to_main"
    assert _base_cfg(league_reset_mode="random").league_reset_mode == "random"


def test_role_weight_knobs_reach_controller_schedule():
    """The YAML ratio becomes the controller's deterministic SWRR cycle."""
    ctrl = _build_league_controller(_base_cfg())
    assert ctrl.role_schedule == [
        Role.MAIN,
        Role.MAIN_EXPLOITER,
        Role.LEAGUE_EXPLOITER,
        Role.MAIN,
    ]
    ctrl = _build_league_controller(
        _base_cfg(
            league_role_weight_main=1.0,
            league_role_weight_main_exploiter=1.0,
            league_role_weight_league_exploiter=0.0,
        )
    )
    assert ctrl.role_schedule == [Role.MAIN, Role.MAIN_EXPLOITER]


def test_invalid_role_weights_raise():
    with pytest.raises(ValueError, match="league_role_weight_main must be > 0"):
        _base_cfg(league_role_weight_main=0.0)
    with pytest.raises(ValueError, match=">= 0"):
        _base_cfg(league_role_weight_main_exploiter=-0.1)


def test_league_capacity_below_three_raises():
    """Two pool slots are reserved for exploiters; the history quota needs >= 1."""
    with pytest.raises(ValueError, match="league_capacity"):
        _base_cfg(league_capacity=2)


def test_unknown_config_key_fails_loudly():
    """AGENTS.md §9: unknown YAML keys must error, not be silently ignored."""
    with pytest.raises(TypeError):
        ExperimentConfig(
            game="kuhn",
            algo="ach",
            self_play_mode="league",
            league_mix_current_mainn=0.5,  # typo must not parse
        )  # type: ignore[call-arg]


# ---- seat shuffle + identity routing knobs ----


def test_routing_knobs_default_and_reach_controller():
    """Seat shuffle defaults ON for league; live-opponent routing defaults OFF."""
    ctrl = _build_league_controller(_base_cfg())
    assert ctrl.runner.config.shuffle_seats is True
    assert ctrl.train_live_opponents is False
    cfg = _base_cfg(league_seat_shuffle=False, league_train_live_opponents=True)
    ctrl = _build_league_controller(cfg)
    assert ctrl.runner.config.shuffle_seats is False
    assert ctrl.train_live_opponents is True


def test_mirror_mode_never_shuffles_seats():
    """Mirror plays one policy in both seats; shuffling would only perturb its RNG."""
    cfg = ExperimentConfig(
        game="kuhn", algo="ach", self_play_mode="mirror", league_seat_shuffle=True
    )
    spec = load_game("kuhn")
    controller = build_controller(spec, build_policy(spec, cfg, seed=0), cfg, rng=random.Random(0))
    assert controller._runner.config.shuffle_seats is False


# ---- algo <-> theta resolution (the unified NN update rule) ----


def _algo_cfg(algo: str, **overrides: object) -> ExperimentConfig:
    """Mirror-mode config with an explicit algo (``_base_cfg`` pins algo=ach)."""
    return ExperimentConfig(game="kuhn", algo=algo, self_play_mode="mirror", **overrides)  # type: ignore[arg-type]


def test_algo_aliases_pin_theta():
    """ppo/ach are pinned aliases for the interpolation endpoints."""
    assert resolve_theta(_algo_cfg("ppo")) == 0.0
    assert resolve_theta(_algo_cfg("ach")) == 1.0


def test_explicit_theta_requires_the_theta_algo():
    """A config whose name disagrees with its update rule fails loudly (§9)."""
    with pytest.raises(ValueError, match="pins theta"):
        resolve_theta(_algo_cfg("ach", theta=0.5))
    with pytest.raises(ValueError, match="requires an explicit theta"):
        resolve_theta(_algo_cfg("theta"))
    assert resolve_theta(_algo_cfg("theta", theta=0.25)) == 0.25


def test_unknown_algo_is_rejected():
    with pytest.raises(ValueError, match="Unknown algo"):
        resolve_theta(_algo_cfg("reinforce"))


def test_theta_algo_has_no_tabular_implementation():
    """The interpolation is MLP-only; the tabular pair stays discrete (D5)."""
    cfg = _algo_cfg("theta", theta=0.5, policy_kind="tabular")
    spec = load_game("kuhn")
    policy = build_policy(spec, cfg, seed=0)
    with pytest.raises(ValueError, match="no tabular implementation"):
        build_update_rule(policy, cfg, spec)


def test_mlp_theta_reaches_the_update_rule():
    """Every algo name on the MLP path builds the one rule, carrying its theta."""
    from mjai.algos.nn_updates import NNActorCriticUpdate
    from mjai.utils import gpu_assert

    gpu_assert.reset_for_tests()
    gpu_assert.require_cpu()
    try:
        spec = load_game("kuhn")
        for algo, theta, want in (("ppo", None, 0.0), ("ach", None, 1.0), ("theta", 0.4, 0.4)):
            cfg = _algo_cfg(algo, theta=theta, policy_kind="mlp", optimizer="sgd")
            rule = build_update_rule(build_policy(spec, cfg, seed=0), cfg, spec)
            assert isinstance(rule, NNActorCriticUpdate)
            assert rule.config.theta == want
    finally:
        gpu_assert.reset_for_tests()


# ---- per-label train stat logging (identity routing, D5 of the fix) ----


class _FakeWriter:
    """SummaryWriter stand-in capturing the last value per tag (AGENTS.md §6)."""

    def __init__(self) -> None:
        self.scalars: dict[str, float] = {}

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self.scalars[tag] = float(scalar_value)


def test_log_train_stats_namespaces_non_main_learners():
    """The main line keeps plain train/*; other learners log per label."""
    from types import SimpleNamespace

    from mjai.algos.transition import UpdateStats
    from mjai.scripts.experiment import _log_train_stats

    main_stats = UpdateStats(policy_loss=1.0, value_loss=0.5, entropy=0.3)
    me_stats = UpdateStats(policy_loss=2.0, value_loss=0.6, entropy=0.4)
    trainer = SimpleNamespace(
        last_stats=main_stats,
        last_stats_by_label={"main": main_stats, "main_exploiter": me_stats},
    )
    writer = _FakeWriter()
    _log_train_stats(writer, 7, trainer)  # type: ignore[arg-type]
    assert writer.scalars["train/policy_loss"] == 1.0
    assert writer.scalars["train/main_exploiter/policy_loss"] == 2.0
    assert "train/main/policy_loss" not in writer.scalars  # identity skip: no double-log


def test_log_train_stats_handles_main_free_rounds():
    """On exploiter-only rounds nothing logs under plain train/* (sparse, not wrong)."""
    from types import SimpleNamespace

    from mjai.algos.transition import UpdateStats
    from mjai.scripts.experiment import _log_train_stats

    me_stats = UpdateStats(policy_loss=2.0, value_loss=0.6, entropy=0.4)
    trainer = SimpleNamespace(last_stats=None, last_stats_by_label={"main_exploiter": me_stats})
    writer = _FakeWriter()
    _log_train_stats(writer, 3, trainer)  # type: ignore[arg-type]
    assert "train/policy_loss" not in writer.scalars
    assert writer.scalars["train/main_exploiter/policy_loss"] == 2.0
