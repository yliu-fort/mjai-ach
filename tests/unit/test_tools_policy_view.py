"""Unit tests for tools/policy_view.py — the rollout+view composition layer."""

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

import policy_view  # noqa: E402

from mjai.utils import gpu_assert  # noqa: E402

from .test_eval_policy_table import _checkpoint  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_mode():
    gpu_assert.reset_for_tests()
    gpu_assert.require_cpu()
    yield
    gpu_assert.reset_for_tests()
    plt.close("all")


def test_slug_makes_an_arm_label_filename_safe():
    assert policy_view.slug("mirror theta=0.5   seed=0") == "mirror_theta0p5_seed0"
    assert policy_view.slug("theta=1") == "theta1"
    assert policy_view.slug("league") == "league"
    # distinct labels must not collapse onto one filename
    assert policy_view.slug("theta=0.5") != policy_view.slug("theta=0")


def test_view_for_arm_carries_visits_end_to_end(tmp_path: Path):
    _checkpoint(tmp_path, "kuhn")
    view = policy_view.view_for_arm(tmp_path, checkpoint="last", episodes=40)
    assert view.visits is not None and view.visits.sum() > 0
    assert view.episodes == 40
    assert len(view.labels) == 12


def test_view_for_arm_without_sampling_leaves_visits_unset(tmp_path: Path):
    _checkpoint(tmp_path, "kuhn")
    view = policy_view.view_for_arm(tmp_path, checkpoint="last", episodes=0)
    assert view.visits is None


def test_sample_visits_keys_match_the_enumerated_observations(tmp_path: Path):
    """The join between sampled play and the enumerated table must actually hit."""
    from mjai.eval.nash import state_observations
    from mjai.eval.policy_table import observation_key
    from mjai.games.loader import load_game

    _checkpoint(tmp_path, "kuhn")
    counts = policy_view.sample_visits(tmp_path, checkpoint="last", episodes=40)
    enumerated = {observation_key(row) for row in state_observations(load_game("kuhn"))}
    assert counts, "self-play produced no decisions"
    assert set(counts) <= enumerated, "sampled observations must land on enumerated rows"


def test_render_arms_makes_one_panel_per_arm(tmp_path: Path):
    for name in ("mirror", "league"):
        _checkpoint(tmp_path / name, "kuhn")
    out, views, skipped = policy_view.render_arms(
        [("mirror", tmp_path / "mirror"), ("league", tmp_path / "league")],
        tmp_path / "figs" / "policy.png",
        checkpoint="last",
        episodes=20,
    )
    assert out is not None and out.is_file()
    assert set(views) == {"mirror", "league"} and not skipped


def test_render_arms_reports_unrenderable_arms_instead_of_dropping_them(tmp_path: Path):
    """A game whose tree will not enumerate must be named, not silently missing."""
    _checkpoint(tmp_path / "kuhn", "kuhn")
    _checkpoint(tmp_path / "oshi", "oshi_zumo")
    out, views, skipped = policy_view.render_arms(
        [("kuhn", tmp_path / "kuhn"), ("oshi", tmp_path / "oshi")],
        tmp_path / "figs" / "policy.png",
        checkpoint="last",
        episodes=0,
    )
    assert out is not None
    assert set(views) == {"kuhn"}
    assert "oshi" in skipped and "infeasible" in skipped["oshi"]


def test_render_arms_with_nothing_renderable_returns_none(tmp_path: Path):
    _checkpoint(tmp_path / "oshi", "oshi_zumo")
    out, views, skipped = policy_view.render_arms(
        [("oshi", tmp_path / "oshi")],
        tmp_path / "figs" / "policy.png",
        checkpoint="last",
        episodes=0,
    )
    assert out is None and not views and set(skipped) == {"oshi"}
