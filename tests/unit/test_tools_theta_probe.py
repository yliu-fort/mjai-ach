"""Unit tests for tools/theta_probe.py pure helpers (no training runs)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import theta_probe  # noqa: E402

SCAN_GAMES = {"brps", "kuhn", "liars_dice1"}


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def test_scan_games_are_the_three_notebook_games():
    """One notebook per game; the probe and the builder must not drift apart."""
    import build_theta_notebooks

    assert set(theta_probe.GAMES) == SCAN_GAMES
    assert set(build_theta_notebooks.GAMES) == SCAN_GAMES


def test_default_thetas_span_both_endpoints():
    assert theta_probe.DEFAULT_THETAS[0] == 0.0
    assert theta_probe.DEFAULT_THETAS[-1] == 1.0


def test_theta_tag_is_filesystem_safe_and_round_trips():
    assert theta_probe.theta_tag(0.0) == "0"
    assert theta_probe.theta_tag(0.25) == "0p25"
    assert theta_probe.theta_tag(1.0) == "1"
    for theta in theta_probe.DEFAULT_THETAS:
        tag = theta_probe.theta_tag(theta)
        assert "." not in tag
        assert float(tag.replace("p", ".")) == theta


def test_parse_thetas_accepts_lists_and_rejects_out_of_range():
    assert theta_probe.parse_thetas("0,0.5,1") == [0.0, 0.5, 1.0]
    assert theta_probe.parse_thetas(" 0.25 , 0.25 ") == [0.25]
    with pytest.raises(ValueError, match="theta must lie in"):
        theta_probe.parse_thetas("0,1.5")
    with pytest.raises(ValueError, match="no thetas"):
        theta_probe.parse_thetas(" , ")


def test_arm_dir_layout():
    d = theta_probe.arm_dir(Path("/x"), "kuhn", 0.25, 3)
    assert d == Path("/x/kuhn/theta_0p25/seed_3")


def test_load_base_config_is_the_ach_mirror_arm():
    """The scan's shared scaffolding comes from the ACH mirror config."""
    for game in theta_probe.GAMES:
        cfg = theta_probe.load_base_config(game)
        assert cfg.game == game
        assert cfg.policy_kind == "mlp"
        assert cfg.self_play_mode == "mirror"
        assert cfg.algo == "ach"  # run_arm replaces this with "theta"


def test_load_base_config_missing_game_fails_loudly():
    with pytest.raises(FileNotFoundError):
        theta_probe.load_base_config("no_such_game")


def test_final_value_averages_the_tail_not_the_last_point():
    """D5 convention: mean over the last final_frac of the x axis."""
    curve = [(0, 10.0), (25, 8.0), (50, 6.0), (75, 4.0), (100, 2.0)]
    assert theta_probe.final_value(curve, 0.25) == pytest.approx(3.0)  # x >= 75
    assert theta_probe.final_value(curve, 0.5) == pytest.approx(4.0)  # x >= 50
    assert theta_probe.final_value(curve, 1.0) == pytest.approx(6.0)  # all points
    assert theta_probe.final_value([], 0.1) is None


def test_final_value_degenerate_curve_falls_back_to_last_point():
    assert theta_probe.final_value([(5, 1.5)], 0.1) == pytest.approx(1.5)


def _summary(thetas: dict[float, list[float]]) -> dict[str, object]:
    """Build a minimal summary structure the renderers understand."""
    entry = {}
    for theta, finals in thetas.items():
        grid = [10, 20, 30]
        entry[f"theta_{theta_probe.theta_tag(theta)}"] = {
            "theta": theta,
            "seeds": [f"seed_{i}" for i in range(len(finals))],
            "done": [f"seed_{i}" for i in range(len(finals))],
            "final_per_seed": {f"seed_{i}": v for i, v in enumerate(finals)},
            "band": {
                "grid": grid,
                "mean": [max(finals) + 1, sum(finals) / len(finals), min(finals)],
                "min": [min(finals), min(finals), min(finals)],
                "max": [max(finals) + 2, max(finals), max(finals)],
                "n_seeds": [len(finals)] * 3,
            },
        }
    return {"kuhn": {"metric": "eval/exploitability", "final_frac": 0.1, "thetas": entry}}


def test_render_curves_and_final_write_pngs(tmp_path: Path):
    summary = _summary({0.0: [0.4, 0.5], 0.5: [0.3, 0.32], 1.0: [0.2, 0.25]})
    curves = theta_probe.render_curves(summary, "kuhn", tmp_path)
    finals = theta_probe.render_theta_final(summary, "kuhn", tmp_path)
    assert curves is not None and curves.is_file()
    assert finals is not None and finals.is_file()
    assert curves.name == "theta_curves_kuhn.png"
    assert finals.name == "theta_final_kuhn.png"


def test_renderers_return_none_without_data(tmp_path: Path):
    """An empty probe root must not raise or emit an empty figure."""
    assert theta_probe.render_curves({}, "kuhn", tmp_path) is None
    assert theta_probe.render_theta_final({}, "kuhn", tmp_path) is None
    assert theta_probe.render_telemetry("kuhn", tmp_path) is None


def test_summarize_on_empty_root_writes_empty_summary(tmp_path: Path):
    assert theta_probe.summarize(tmp_path) == {}
    assert (tmp_path / "summary.json").is_file()
