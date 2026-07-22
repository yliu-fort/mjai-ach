"""Unit tests for ExperimentConfig league knobs (AGENTS.md §5, F2).

Covers: defaults preserve the previous hard-coded LeagueConfig/LeagueMix
behavior, every knob reaches the built LeagueManager/OpponentSampler via
build_controller, and invalid knob combinations fail loudly (§9).
"""

from __future__ import annotations

import random

import pytest

from mjai.games.loader import load_game
from mjai.league.league_controller import LeagueSelfPlay
from mjai.league.manager import LeagueConfig
from mjai.league.opponent_sampler import LeagueMix
from mjai.scripts.experiment import ExperimentConfig, build_controller, build_policy


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


def test_unknown_config_key_fails_loudly():
    """AGENTS.md §9: unknown YAML keys must error, not be silently ignored."""
    with pytest.raises(TypeError):
        ExperimentConfig(
            game="kuhn",
            algo="ach",
            self_play_mode="league",
            league_mix_current_mainn=0.5,  # typo must not parse
        )  # type: ignore[call-arg]
