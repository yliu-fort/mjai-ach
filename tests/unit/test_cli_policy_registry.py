"""Unit tests for the policy registry's filter/validate/page helpers (F5, §5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mjai.agents.ckpt_io import CheckpointManifest, write_checkpoint
from mjai.cli.policy_registry import (
    PolicyEntry,
    compatible_with,
    filter_labels,
    list_policies,
    page,
)


def _manifest(game: str, step: int, obs: int = 11, act: int = 2, created: float = 0.0):
    return CheckpointManifest(
        game=game,
        game_string=f"{game}_string",
        algo="ach",
        self_play_mode="mirror",
        policy_kind="mlp",
        num_actions=act,
        obs_kind="information_state",
        obs_size=obs,
        train_step=step,
        created_at=created,
    )


def _ckpt(root: Path, name: str, game: str, step: int, obs: int = 11, act: int = 2) -> Path:
    d = root / name
    write_checkpoint(d, _manifest(game, step, obs, act))
    return d


def test_list_policies_filters_to_selected_game(tmp_path):
    _ckpt(tmp_path, "a", "kuhn", 10)
    _ckpt(tmp_path, "b", "leduc", 20)
    _ckpt(tmp_path, "c", "kuhn", 30)
    kuhn = list_policies(tmp_path, game="kuhn")
    assert [e.manifest.train_step for e in kuhn] == [30, 10]
    assert all(e.manifest.game == "kuhn" for e in kuhn)
    assert len(list_policies(tmp_path, game="ttt")) == 0


def test_list_policies_sorted_by_step_descending(tmp_path):
    for name, step in [("a", 5), ("b", 100), ("c", 50)]:
        _ckpt(tmp_path, name, "kuhn", step)
    steps = [e.manifest.train_step for e in list_policies(tmp_path, game="kuhn")]
    assert steps == [100, 50, 5]


def test_list_policies_empty_on_missing_root(tmp_path):
    assert list_policies(tmp_path / "nope", game="kuhn") == []


def test_compatible_with_splits_on_obs_and_action_space(tmp_path):
    _ckpt(tmp_path, "good", "kuhn", 1, obs=11, act=2)
    _ckpt(tmp_path, "bad_obs", "kuhn", 2, obs=12, act=2)
    _ckpt(tmp_path, "bad_act", "kuhn", 3, obs=11, act=4)
    entries = list_policies(tmp_path, game="kuhn")
    ok, bad = compatible_with(entries, obs_size=11, num_actions=2)
    assert [e.manifest.train_step for e in ok] == [1]
    assert {e.manifest.train_step for e in bad} == {2, 3}


def test_filter_labels_case_insensitive_substring(tmp_path):
    _ckpt(tmp_path, "seed_0", "kuhn", 10)
    _ckpt(tmp_path, "seed_1", "kuhn", 20)
    entries = list_policies(tmp_path, game="kuhn")
    assert len(filter_labels(entries, "ACH")) == 2  # matches "ach" in every label
    assert len(filter_labels(entries, "step=20")) == 1
    assert len(filter_labels(entries, "SEED_0")) == 1  # label carries the rel path
    assert filter_labels(entries, "zzz") == []
    # Blank text is a no-op.
    assert len(filter_labels(entries, "  ")) == 2


def test_page_returns_window_and_remaining_count():
    entries = [
        PolicyEntry(path=Path(f"c{i}"), manifest=_manifest("kuhn", i), label=f"l{i}")
        for i in range(25)
    ]
    shown, remaining = page(entries)
    assert len(shown) == 20
    assert remaining == 5
    shown, remaining = page(entries[:3])
    assert len(shown) == 3
    assert remaining == 0


def test_page_rejects_nonpositive_size():
    with pytest.raises(ValueError, match="page size"):
        page([], 0)
