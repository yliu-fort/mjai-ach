"""Unit tests for the game-config loader (AGENTS.md §5, §9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mjai.config.game_config import (
    DEFAULT_GAMES_DIR,
    GameConfig,
    load_all_game_configs,
    load_game_config,
    resolve_to_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GAMES_DIR = REPO_ROOT / "configs" / "games"

EXPECTED_NAMES = {"brps", "kuhn", "leduc", "ttt", "goofspiel5_ii", "liars_dice1", "oshi_zumo"}


def test_default_games_dir_points_to_repo_configs():
    assert DEFAULT_GAMES_DIR == GAMES_DIR
    assert DEFAULT_GAMES_DIR.is_dir()


def test_all_seven_game_configs_present_and_load():
    cfgs = load_all_game_configs()
    assert set(cfgs) == EXPECTED_NAMES
    for name, cfg in cfgs.items():
        assert isinstance(cfg, GameConfig)
        assert cfg.name == name
        assert cfg.game_string  # non-empty


def test_each_config_resolves_to_a_working_spec():
    cfgs = load_all_game_configs()
    for _name, cfg in cfgs.items():
        spec = resolve_to_spec(cfg)
        assert spec.num_actions > 0
        assert spec.num_players == 2
        assert spec.is_zero_sum


def test_config_name_must_match_for_registered_resolution():
    """The goofspiel config must resolve with its params (imp_info etc.)."""
    cfgs = load_all_game_configs()
    spec = resolve_to_spec(cfgs["goofspiel5_ii"])
    assert spec.num_actions == 5
    assert spec.is_simultaneous  # goofspiel is simultaneous


def test_load_game_config_rejects_missing_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: nope\n")  # missing game_string
    with pytest.raises(ValueError, match="game_string"):
        load_game_config(bad)


def test_load_game_config_rejects_non_mapping(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_game_config(bad)


def test_load_all_returns_empty_for_missing_dir(tmp_path):
    assert load_all_game_configs(tmp_path / "nope") == {}


def test_notes_field_optional():
    cfgs = load_all_game_configs()
    # Every shipped config carries notes; this guards against silent drops.
    for cfg in cfgs.values():
        assert cfg.notes, f"{cfg.name} has no notes"
