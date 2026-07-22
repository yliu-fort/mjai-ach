"""Unit tests for tools/league_probe.py pure helpers (no training runs)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import league_probe  # noqa: E402


def test_parse_seeds_range_list_single():
    assert league_probe.parse_seeds("0-3") == [0, 1, 2, 3]
    assert league_probe.parse_seeds("0,2,5") == [0, 2, 5]
    assert league_probe.parse_seeds("3") == [3]
    assert league_probe.parse_seeds("1-2,7") == [1, 2, 7]
    with pytest.raises(ValueError, match="no seeds"):
        league_probe.parse_seeds(" , ")


def test_load_arm_config_all_four_arms_parse():
    for game in league_probe.GAMES:
        for mode in league_probe.MODES:
            cfg = league_probe.load_arm_config(game, mode)
            assert cfg.game == game
            assert cfg.self_play_mode == mode
            assert cfg.policy_kind == "mlp"


def test_load_arm_config_missing_game_fails_loudly():
    with pytest.raises(FileNotFoundError):
        league_probe.load_arm_config("no_such_game", "mirror")


def test_arm_dir_layout():
    d = league_probe.arm_dir(Path("/x"), "kuhn", "league", 7)
    assert d == Path("/x/kuhn_league/seed_7")


def test_interp_forward_fill():
    curve = [(10, 1.0), (20, 0.5), (30, 0.25)]
    assert league_probe.interp_forward(curve, [0, 10, 15, 25, 30, 99]) == [
        None,
        1.0,
        1.0,
        0.5,
        0.25,
        0.25,
    ]


def test_band_mean_min_max_and_seed_count():
    c1 = [(0, 1.0), (10, 0.6)]
    c2 = [(5, 0.8), (10, 0.4)]
    grid = [0, 5, 10]
    b = league_probe.band([c1, c2], grid)
    assert b["n_seeds"] == [1, 2, 2]
    assert b["mean"][0] == 1.0
    assert b["mean"][1] == pytest.approx(0.9)
    assert b["min"][2] == 0.4
    assert b["max"][2] == 0.6
    assert league_probe.band([], grid)["mean"] == [None, None, None]
