"""Unit tests for tools/ab_factor_probe.py pure helpers (no training runs).

Mirrors the structure of test_tools_theta_probe.py / test_tools_league_probe.py:
the read paths (``read_curves_fallback``, ``read_many_tags``) are monkeypatched
so no event files are needed, and the figure helpers draw onto an Agg backend.
``run_experiment`` is never invoked.
"""

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

import ab_factor_probe as ab  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# --------------------------------------------------------------------------- #
# config resolution: the arm fingerprint must reflect the override axis
# --------------------------------------------------------------------------- #


def test_load_config_missing_raises_loudly():
    # §9: unknown/missing config must fail, never silently fall back.
    with pytest.raises(FileNotFoundError):
        ab.load_config("does_not_exist_xyz")


def test_arm_dir_layout_carries_game_on_root():
    # <root>/<label>/seed_<seed> — root carries the game, label is the axis arm.
    root = Path("/tmp/probe/liars_dice1")
    assert ab.arm_dir(root, "ach_adam_3e4", 2) == root / "ach_adam_3e4" / "seed_2"


def test_arm_config_applies_overrides_and_sets_runtime_fields():
    cfg = ab.arm_config(
        "ach_adam_1e-3",
        "liars_dice1_ach_mlp_mirror",
        seed=3,
        overrides={"optimizer": "adam", "learning_rate": 0.001},
        total_env_steps=10000,
        eval_every_env_steps=2500,
        root=Path("runs/nb_sgd_adam/liars_dice1"),
    )
    assert cfg.optimizer == "adam"
    assert cfg.learning_rate == pytest.approx(1e-3)
    assert cfg.seed == 3
    assert cfg.total_env_steps == 10000
    # Cross-platform: out_dir joins with os.sep (backslash on Windows).
    assert cfg.out_dir.replace("\\", "/").endswith("ach_adam_1e-3/seed_3")
    assert cfg.verbose is False  # arm_cache must not fingerprint verbose noise


def test_arm_config_rejects_bogus_override_field():
    # An override naming a field ExperimentConfig lacks must raise, not no-op.
    with pytest.raises(TypeError):
        ab.arm_config(
            "bogus",
            "liars_dice1_ach_mlp_mirror",
            seed=0,
            overrides={"not_a_real_field": True},
            total_env_steps=10000,
            eval_every_env_steps=2500,
            root=Path("/tmp/probe/liars_dice1"),
        )


def test_arm_status_is_a_cache_verdict_not_a_run():
    # Pure: resolves config + asks arm_cache; no training, no events.
    st = ab.arm_status(
        "ach",
        "liars_dice1_ach_mlp_mirror",
        0,
        total_env_steps=10000,
        eval_every_env_steps=2500,
        root=Path("/tmp/probe/absent/liars_dice1"),
    )
    assert st.state == "missing"


# --------------------------------------------------------------------------- #
# summarize: aggregation + per-arm tagging, with the read path faked
# --------------------------------------------------------------------------- #


def _fake_fallback(data: dict[str, list[tuple[int, float]]]):
    """Stand in for read_curves_fallback: return the given curves + a fixed tag."""
    curves = {str(d): v for d, v in data.items()}
    used = {d: "eval/exploitability" for d in curves}
    return curves, used


def test_summarize_groups_by_arm_label_and_records_done(tmp_path, monkeypatch):
    # summarize globs <root>/* /seed_*/tb, so the tb dirs must exist on disk.
    for s in ("seed_0", "seed_1"):
        (tmp_path / "ach_sgd" / s / "tb").mkdir(parents=True)
    monkeypatch.setattr(
        ab,
        "read_curves_fallback",
        lambda dirs, tags: _fake_fallback(
            {
                tmp_path / "ach_sgd/seed_0/tb": [(0, 0.9), (10, 0.5)],
                tmp_path / "ach_sgd/seed_1/tb": [(0, 0.8), (10, 0.4)],
            }
        ),
    )
    (tmp_path / "ach_sgd/seed_0/DONE").write_text("ok\n", encoding="utf-8")
    (tmp_path / "ach_sgd/seed_1/DONE").write_text("ok\n", encoding="utf-8")

    summary = ab.summarize(tmp_path)
    arm = summary["ach_sgd"]
    assert arm["seeds"] == ["seed_0", "seed_1"]
    assert arm["done"] == ["seed_0", "seed_1"]
    assert arm["tag"] == "eval/exploitability"
    assert arm["final_per_seed"]["seed_0"] == pytest.approx(0.5)
    # band carries the shared grid + per-point mean.
    assert arm["band"]["grid"] == [0, 10]
    assert arm["band"]["mean"][1] == pytest.approx(0.45)  # mean of 0.5, 0.4

    # summary.json is the durable artifact the notebook does not have to rebuild.
    import json

    on_disk = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["ach_sgd"]["final_per_seed"]["seed_1"] == pytest.approx(0.4)


def test_summarize_empty_root_writes_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "read_curves_fallback", lambda dirs, tags: ({}, {}))
    assert ab.summarize(tmp_path) == {}


# --------------------------------------------------------------------------- #
# figure helpers: order, metric naming, empty-input safety, telemetry gating
# --------------------------------------------------------------------------- #


def test_metric_name_unifies_across_arms(monkeypatch):
    summary = {
        "a": {"tag": "eval/exploitability"},
        "b": {"tag": "eval/exploitability"},
    }
    assert ab._metric_name(summary) == "exploitability"


def test_metric_name_blank_when_tags_differ():
    summary = {"a": {"tag": "eval/exploitability"}, "b": {"tag": "eval/nash_conv"}}
    assert ab._metric_name(summary) == ""


def test_ordered_respects_caller_order_then_appends_extras():
    summary = {"a": {}, "b": {}, "c": {}}
    assert ab._ordered(summary, ["b", "a"]) == ["b", "a", "c"]


def test_build_curves_figure_handles_empty_summary():
    fig, drew = ab.build_curves_figure({}, title="t")
    assert drew is False


def test_build_telemetry_figure_no_dirs_returns_undrawn(tmp_path):
    # No arm dirs -> an empty figure, not a crash; telemetry is optional.
    monkeypatch_target = tmp_path  # empty root
    fig, drew = ab.build_telemetry_figure(monkeypatch_target)
    assert drew is False


def test_build_telemetry_figure_reads_through_read_many_tags(tmp_path, monkeypatch):
    # Real arm dir layout, but the read path is faked so no event files exist.
    arm = tmp_path / "ach/seed_0/tb"
    arm.mkdir(parents=True)
    fake = {
        str(arm): {
            "train/grad_norm": [(0, 1.0), (1, 2.0)],
            "train/clip_frac": [(0, 0.1)],
        }
    }
    monkeypatch.setattr(
        ab, "read_many_tags", lambda dirs, tags, max_points=2000, peak_tags=(): fake
    )
    fig, drew = ab.build_telemetry_figure(tmp_path)
    assert drew is True
    # one panel per requested tag, in order
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == list(ab.TELEMETRY_TAGS)
