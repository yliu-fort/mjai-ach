"""Unit tests for mjai.eval.policy_table — the final-policy view (AGENTS.md §5)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
from matplotlib.figure import Figure

from mjai.agents.ckpt_io import CheckpointManifest, write_checkpoint
from mjai.agents.mlp import MLPSharedActorCritic
from mjai.eval import policy_table as pt
from mjai.games.loader import load_game
from mjai.utils import gpu_assert

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import policy_view as policy_view_tool  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_mode():
    gpu_assert.reset_for_tests()
    gpu_assert.require_cpu()
    yield
    gpu_assert.reset_for_tests()


def _checkpoint(run: Path, game: str, *, name: str = "step_10", step: int = 10) -> Path:
    """Write a real loadable MLP checkpoint, the way run_experiment does."""
    spec = load_game(game)
    policy = MLPSharedActorCritic(
        obs_size=spec.obs_size,
        num_actions=spec.num_actions,
        hidden_sizes=(8,),
        device="cpu",
        seed=step,
    )
    manifest = CheckpointManifest(
        game=spec.name,
        game_string=spec.game_string,
        algo="ach",
        self_play_mode="mirror",
        policy_kind="mlp",
        num_actions=spec.num_actions,
        obs_kind=spec.obs_kind,
        obs_size=spec.obs_size,
        train_step=step,
    )
    ckpt = run / "checkpoints" / name
    write_checkpoint(ckpt, manifest)
    policy.save(str(ckpt / manifest.weight_filename()))
    return ckpt


# ---- checkpoint selection ----


def test_resolve_checkpoint_prefers_best_then_falls_back_to_last(tmp_path: Path):
    _checkpoint(tmp_path, "kuhn", name="step_10", step=10)
    _checkpoint(tmp_path, "kuhn", name="step_20", step=20)
    assert pt.resolve_checkpoint(tmp_path, "best").name == "step_20"  # no best/ yet
    _checkpoint(tmp_path, "kuhn", name="best", step=10)
    assert pt.resolve_checkpoint(tmp_path, "best").name == "best"


def test_resolve_checkpoint_last_ignores_the_best_copy(tmp_path: Path):
    """checkpoints/best is a COPY of some step, so it must not rank as 'last'."""
    _checkpoint(tmp_path, "kuhn", name="step_10", step=10)
    _checkpoint(tmp_path, "kuhn", name="step_20", step=20)
    _checkpoint(tmp_path, "kuhn", name="best", step=10)
    assert pt.resolve_checkpoint(tmp_path, "last").name == "step_20"


def test_resolve_checkpoint_named_and_missing(tmp_path: Path):
    _checkpoint(tmp_path, "kuhn", name="step_10", step=10)
    assert pt.resolve_checkpoint(tmp_path, "step_10").name == "step_10"
    with pytest.raises(pt.PolicyViewError, match="no checkpoint"):
        pt.resolve_checkpoint(tmp_path, "step_999")
    with pytest.raises(pt.PolicyViewError, match="no step_"):
        pt.resolve_checkpoint(tmp_path / "nowhere", "last")


# ---- the view ----


def test_kuhn_view_is_the_full_twelve_row_table(tmp_path: Path):
    _checkpoint(tmp_path, "kuhn")
    view = pt.policy_view(tmp_path, checkpoint="last")
    assert view.game == "kuhn" and view.mode == "table"
    assert len(view.labels) == pt.ENUMERABLE["kuhn"] == 12
    assert view.probs.shape == (12, 2)
    assert view.action_names == ["Pass", "Bet"]  # not "Action(id=0, player=0)"
    assert sorted(set(view.players)) == [0, 1]
    # every row is a normalized distribution over its legal actions
    assert np.allclose(np.where(view.legal, view.probs, 0.0).sum(axis=1), 1.0)
    assert all(view.labels), "no row may be left unlabeled"


def test_labels_are_compacted_onto_one_line(tmp_path: Path):
    """Goofspiel's info-state strings are multi-line and unusable in a table."""
    _checkpoint(tmp_path, "goofspiel5_ii")
    view = pt.policy_view(tmp_path, checkpoint="last")
    assert not any("\n" in label for label in view.labels)
    assert view.mode == "heatmap"


def test_visit_counts_join_onto_the_enumerated_rows(tmp_path: Path):
    _checkpoint(tmp_path, "kuhn")
    counts = policy_view_tool.sample_visits(tmp_path, checkpoint="last", episodes=50)
    view = pt.policy_view(tmp_path, checkpoint="last", visit_counts=counts, episodes=50)
    assert view.visits is not None and view.episodes == 50
    assert view.visits.shape == (12,)
    assert view.visits.sum() > 0, "self-play must reach some info state"
    top = view.top_rows(3, player=0)
    assert top and all(view.players[i] == 0 for i in top)
    counts = [int(view.visits[i]) for i in top]
    assert counts == sorted(counts, reverse=True)


def test_top_rows_without_visits_says_so_by_returning_the_first_rows(tmp_path: Path):
    _checkpoint(tmp_path, "kuhn")
    view = pt.policy_view(tmp_path, checkpoint="last")
    assert view.visits is None
    assert view.top_rows(3, player=0) == [0, 1, 2]


def test_non_enumerable_game_refuses_instead_of_hanging(tmp_path: Path):
    """oshi_zumo's full enumeration did not finish in 90 s when measured."""
    _checkpoint(tmp_path, "oshi_zumo")
    assert "oshi_zumo" not in pt.ENUMERABLE
    with pytest.raises(pt.PolicyViewError, match="infeasible"):
        pt.policy_view(tmp_path, checkpoint="last")


def test_root_policy_works_for_the_non_enumerable_game_too(tmp_path: Path):
    _checkpoint(tmp_path, "oshi_zumo")
    roots = pt.root_policy(tmp_path, checkpoint="last")
    assert set(roots) == {0, 1}
    for dist in roots.values():
        assert sum(dist.values()) == pytest.approx(1.0)
        assert all(name.startswith("[P0]Bid") for name in dist)


def test_every_phase1_game_has_a_declared_view_mode():
    from mjai.games.loader import GAME_STRINGS

    assert set(pt.VIEW_MODE) == set(GAME_STRINGS)
    assert set(pt.ENUMERABLE) == set(GAME_STRINGS) - {"oshi_zumo"}
    assert pt.VIEW_MODE["oshi_zumo"] == "root"


# ---- BRPS: the one game whose Nash is known in closed form ----


def test_brps_nash_gap_measures_against_the_analytic_equilibrium(tmp_path: Path):
    _checkpoint(tmp_path, "brps")
    view = pt.policy_view(tmp_path, checkpoint="last")
    probs, tv = pt.brps_nash_gap(view)
    assert probs.shape == (3,) and probs.sum() == pytest.approx(1.0)
    assert 0.0 <= tv <= 1.0
    view_k = pt.policy_view(_kuhn_run(tmp_path), checkpoint="last")
    with pytest.raises(pt.PolicyViewError, match="BRPS-only"):
        pt.brps_nash_gap(view_k)


def _kuhn_run(tmp_path: Path) -> Path:
    run = tmp_path / "kuhn_run"
    _checkpoint(run, "kuhn")
    return run


# ---- export + rendering ----


def test_write_csv_round_trips_every_row(tmp_path: Path):
    _checkpoint(tmp_path, "kuhn")
    counts = policy_view_tool.sample_visits(tmp_path, checkpoint="last", episodes=20)
    view = pt.policy_view(tmp_path, checkpoint="last", visit_counts=counts, episodes=20)
    out = pt.write_csv(view, tmp_path / "out" / "kuhn.csv")
    with out.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 12
    assert {"player", "info_state", "visits", "visit_share", "Pass", "Bet"} <= set(rows[0])


def test_to_records_marks_illegal_actions_as_nan(tmp_path: Path):
    _checkpoint(tmp_path, "leduc")
    view = pt.policy_view(tmp_path, checkpoint="last")
    records = pt.to_records(view, rows=[0])
    # Fold is illegal at the opening state; a 0.0 there would read as "never folds"
    assert np.isnan(records[0]["Fold"])


@pytest.mark.parametrize(
    "game",
    [
        "brps",
        "kuhn",
        "leduc",
        # 24576 info states: ~2.9 s to enumerate, too slow for the pre-commit
        # suite but the only coverage of the top_k mode on a real game.
        pytest.param("liars_dice1", marks=pytest.mark.slow),
    ],
)
def test_plot_policy_draws_something_for_every_mode(tmp_path: Path, game: str):
    run = tmp_path / game
    _checkpoint(run, game)
    view = pt.policy_view(run, checkpoint="last")
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    pt.plot_policy(view, ax=ax, player=0)
    assert ax.has_data()
