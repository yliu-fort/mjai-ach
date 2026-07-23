"""Unit tests for tools/league_probe.py pure helpers (no training runs)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import league_probe  # noqa: E402

ALL_SEVEN_GAMES = {
    "brps",
    "goofspiel5_ii",
    "liars_dice1",
    "oshi_zumo",
    "leduc",
    "kuhn",
    "ttt",
}


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def test_parse_seeds_range_list_single():
    assert league_probe.parse_seeds("0-3") == [0, 1, 2, 3]
    assert league_probe.parse_seeds("0,2,5") == [0, 2, 5]
    assert league_probe.parse_seeds("3") == [3]
    assert league_probe.parse_seeds("1-2,7") == [1, 2, 7]
    with pytest.raises(ValueError, match="no seeds"):
        league_probe.parse_seeds(" , ")


def test_games_cover_all_seven_phase1_games():
    """AGENTS.md D8: the A/B probe spans all 7 Phase-1 games."""
    assert len(league_probe.GAMES) == 7
    assert set(league_probe.GAMES) == ALL_SEVEN_GAMES


def test_eval_tag_chain_matches_plots_fallback():
    """Fallback order mirrors mjai.eval.plots (metric_key + fallback_keys)."""
    assert league_probe.EVAL_TAG_CHAIN == (
        "eval/exploitability",
        "eval/nash_conv",
        "eval/exact_nash_distance",
    )


def test_load_arm_config_all_arms_parse():
    """Every GAMES x MODES arm (7 x 2 = 14) has a parseable MLP exp YAML."""
    arms = [(game, mode) for game in league_probe.GAMES for mode in league_probe.MODES]
    assert len(arms) == 14
    for game, mode in arms:
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


def _patch_synthetic_tb(monkeypatch: pytest.MonkeyPatch, data: dict[str, dict[str, list]]):
    """Fake read_many: data maps tag -> {tb_dir_str: curve}."""

    def fake_read_many(tb_dirs, tag="eval/exploitability", workers=6):
        return {str(d): list(data.get(tag, {}).get(str(d), [])) for d in tb_dirs}

    monkeypatch.setattr(league_probe, "read_many", fake_read_many)


def test_read_curves_fallback_chain(monkeypatch: pytest.MonkeyPatch):
    """First tag with points wins per run; tagless runs stay empty."""
    data = {
        "eval/exploitability": {"/r/a/tb": [(0, 1.0), (10, 0.5)]},
        "eval/nash_conv": {"/r/b/tb": [(0, 0.7)]},
        "eval/exact_nash_distance": {"/r/c/tb": [(0, 0.2)]},
    }
    _patch_synthetic_tb(monkeypatch, data)
    tb_dirs = ["/r/a/tb", "/r/b/tb", "/r/c/tb", "/r/d/tb"]
    curves, used_tag = league_probe.read_curves_fallback(tb_dirs)
    assert curves["/r/a/tb"] == [(0, 1.0), (10, 0.5)]
    assert curves["/r/b/tb"] == [(0, 0.7)]
    assert curves["/r/c/tb"] == [(0, 0.2)]
    assert curves["/r/d/tb"] == []
    assert used_tag == {
        "/r/a/tb": "eval/exploitability",
        "/r/b/tb": "eval/nash_conv",
        "/r/c/tb": "eval/exact_nash_distance",
    }


def test_read_curves_fallback_prefers_exploitability_over_nash_conv(
    monkeypatch: pytest.MonkeyPatch,
):
    """A run with both tags resolves to exploitability (chain head)."""
    data = {
        "eval/exploitability": {"/r/a/tb": [(0, 1.0)]},
        "eval/nash_conv": {"/r/a/tb": [(0, 9.9)]},
    }
    _patch_synthetic_tb(monkeypatch, data)
    curves, used_tag = league_probe.read_curves_fallback(["/r/a/tb"])
    assert curves["/r/a/tb"] == [(0, 1.0)]
    assert used_tag["/r/a/tb"] == "eval/exploitability"


def test_summarize_falls_back_to_nash_conv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Regression: brps logs only nash_conv; its band must not be empty."""
    for arm in ("brps_mirror", "brps_league"):
        (tmp_path / arm / "seed_0" / "tb").mkdir(parents=True)
    (tmp_path / "brps_mirror" / "seed_0" / "DONE").write_text("ok\n")
    data = {
        "eval/nash_conv": {
            str(tmp_path / "brps_mirror" / "seed_0" / "tb"): [(0, 0.9), (10, 0.4)],
            str(tmp_path / "brps_league" / "seed_0" / "tb"): [(0, 0.8), (10, 0.3)],
        },
    }
    _patch_synthetic_tb(monkeypatch, data)
    summary = league_probe.summarize(root=tmp_path)
    for arm in ("brps_mirror", "brps_league"):
        entry = summary[arm]
        assert entry["tag"] == "eval/nash_conv"
        assert entry["band"] is not None
        assert entry["band"]["mean"][-1] in (0.4, 0.3)
        assert entry["final_per_seed"] == {"seed_0": entry["band"]["mean"][-1]}
    assert summary["brps_mirror"]["done"] == ["seed_0"]
    assert summary["brps_league"]["done"] == []  # no DONE marker
    on_disk = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["brps_mirror"]["tag"] == "eval/nash_conv"


def test_summarize_empty_root_writes_empty_summary(tmp_path: Path):
    summary = league_probe.summarize(root=tmp_path)
    assert summary == {}
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == {}


def _synthetic_arm(tag: str, final: float) -> dict[str, object]:
    return {
        "seeds": ["seed_0"],
        "done": ["seed_0"],
        "tag": tag,
        "final_per_seed": {"seed_0": final},
        "band": {
            "grid": [0, 10],
            "mean": [1.0, final],
            "min": [1.0, final],
            "max": [1.0, final],
            "n_seeds": [1, 1],
        },
    }


def test_render_figure_grid_covers_seven_games(tmp_path: Path):
    """The 2x4 grid renders all 7 game panels and hides the 8th (empty) one."""
    summary = {
        "kuhn_mirror": _synthetic_arm("eval/exploitability", 0.05),
        "brps_mirror": _synthetic_arm("eval/nash_conv", 0.2),
        "brps_league": _synthetic_arm("eval/nash_conv", 0.15),
    }
    out = league_probe.render_figure(summary, root=tmp_path)
    assert out is not None and out.is_file()
    fig, drew = league_probe.build_figure(summary)
    assert drew
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == len(league_probe.GAMES) == 7
    titles = {ax.get_title() for ax in visible}
    assert any("kuhn: exploitability" in t for t in titles)
    assert any("brps: nash_conv" in t for t in titles)
    # Games without data still get panels (grid slot), just no curves.
    assert any(t.startswith("ttt: ") for t in titles)


def test_render_figure_games_filter_gives_one_panel_per_game(tmp_path: Path):
    """A per-game ab_<game> notebook must not get 7 panels with 6 blank."""
    summary = {"kuhn_mirror": _synthetic_arm("eval/exploitability", 0.05)}
    fig, drew = league_probe.build_figure(summary, games=["kuhn"])
    assert drew
    assert [ax.get_title() for ax in fig.axes if ax.get_visible()] == [
        "kuhn: exploitability vs env-steps"
    ]
    out = league_probe.render_figure(summary, root=tmp_path, games=["kuhn"])
    assert out is not None and out.name == "ab_kuhn.png"


def test_renderers_never_touch_the_global_matplotlib_backend(tmp_path: Path):
    """matplotlib.use() from a notebook-imported helper kills inline plotting.

    The A/B notebook plots league telemetry inline AFTER calling render_figure;
    when this regresses, that cell silently emits no image at all.
    """
    before = matplotlib.get_backend()
    summary = {"kuhn_mirror": _synthetic_arm("eval/exploitability", 0.05)}
    league_probe.render_figure(summary, root=tmp_path)
    assert matplotlib.get_backend() == before
    assert not plt.get_fignums()  # and no figure leaked into pyplot's registry


def test_render_figure_no_data_returns_none(tmp_path: Path):
    assert league_probe.render_figure({}, root=tmp_path) is None
    assert not (tmp_path / "figs" / "ab_exploitability.png").exists()


def _fake_run_dir(root: Path, rows: list[dict], ckpt_steps: list[int]) -> Path:
    """Minimal run dir: train_curve.json + empty step_N checkpoint dirs."""
    run = root / "game_mode" / "seed_0"
    (run / "checkpoints").mkdir(parents=True)
    (run / "train_curve.json").write_text(json.dumps(rows), encoding="utf-8")
    for s in ckpt_steps:
        (run / "checkpoints" / f"step_{s}").mkdir()
        (run / "checkpoints" / f"step_{s}" / "manifest.json").write_text("{}", encoding="utf-8")
    return run


def test_mark_best_checkpoint_picks_lowest_and_copies(tmp_path):
    rows = [
        {"step": 10, "env_steps": 5000, "eval/nash_conv": 3.0},
        {"step": 20, "env_steps": 10000, "eval/nash_conv": 1.5},
        {"step": 30, "env_steps": 15000, "eval/nash_conv": 2.2},
    ]
    run = _fake_run_dir(tmp_path, rows, [10, 20, 30])
    info = league_probe.mark_best_checkpoint(run)
    assert info == {"tag": "eval/nash_conv", "value": 1.5, "step": 20, "env_steps": 10000}
    best = run / "checkpoints" / "best"
    assert (best / "manifest.json").is_file()
    assert json.loads((best / "best.json").read_text(encoding="utf-8"))["step"] == 20


def test_mark_best_checkpoint_tag_fallback_and_missing_curve(tmp_path):
    rows = [{"step": 10, "env_steps": 5000, "eval/exploitability": 0.4}]
    run = _fake_run_dir(tmp_path, rows, [10])
    info = league_probe.mark_best_checkpoint(run)
    assert info["tag"] == "eval/exploitability"  # first chain tag present wins
    # re-run replaces the previous best dir cleanly
    info2 = league_probe.mark_best_checkpoint(run)
    assert info2 == info
    missing = tmp_path / "empty_run"
    missing.mkdir()
    assert league_probe.mark_best_checkpoint(missing) is None


def test_mark_best_checkpoint_missing_step_dir_returns_none(tmp_path):
    rows = [{"step": 99, "env_steps": 5000, "eval/nash_conv": 1.0}]
    run = _fake_run_dir(tmp_path, rows, [10])  # no step_99 dir
    assert league_probe.mark_best_checkpoint(run) is None


def test_run_arm_accepts_a_device_override():
    """The A/B notebooks pin the device explicitly; the probe must honour it."""
    import inspect

    assert "device" in inspect.signature(league_probe.run_arm).parameters
