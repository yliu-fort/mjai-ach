"""Unit tests for the checkpoint manifest + I/O (AGENTS.md §5)."""

from __future__ import annotations

import dataclasses
import json
import time

import pytest

from mjai.agents.ckpt_io import (
    MANIFEST_NAME,
    CheckpointManifest,
    checkpoint_name,
    discover_checkpoints,
    read_manifest,
    write_checkpoint,
)


def _make(game="kuhn", step=100, score=None) -> CheckpointManifest:
    return CheckpointManifest(
        game=game,
        game_string="kuhn_poker",
        algo="ach",
        self_play_mode="mirror",
        policy_kind="tabular",
        num_actions=2,
        obs_kind="information_state",
        obs_size=11,
        train_step=step,
        eval_score=score,
    )


def test_write_and_read_roundtrip(tmp_path):
    m = _make(step=42, score=0.015)
    d = write_checkpoint(tmp_path / "run1", m)
    assert (d / MANIFEST_NAME).is_file()
    m2 = read_manifest(d)
    assert m2 == m


def test_manifest_serializes_to_json(tmp_path):
    m = _make()
    d = write_checkpoint(tmp_path / "r", m)
    data = json.loads((d / MANIFEST_NAME).read_text())
    assert data["game"] == "kuhn"
    assert data["algo"] == "ach"
    assert data["policy_kind"] == "tabular"


def test_weight_filename_dispatches_on_kind():
    tab = dataclasses.replace(_make(), policy_kind="tabular")
    nn = dataclasses.replace(_make(), policy_kind="mlp")
    assert tab.weight_filename() == "policy.json"
    assert nn.weight_filename() == "policy.pt"
    assert nn.is_neural
    assert not tab.is_neural


def test_read_manifest_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_manifest(tmp_path / "does_not_exist")


def test_read_manifest_corrupt_json_skipped_by_discover(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / MANIFEST_NAME).write_text("{not valid json")
    good = write_checkpoint(tmp_path / "good", _make(step=1))
    found = discover_checkpoints(tmp_path)
    dirs = [p for p, _ in found]
    assert good in dirs
    assert d not in dirs  # corrupt manifest skipped, not raised


def test_discover_finds_nested_checkpoints(tmp_path):
    write_checkpoint(tmp_path / "runs" / "a", _make(step=10))
    write_checkpoint(tmp_path / "runs" / "b" / "c", _make(step=20))
    write_checkpoint(tmp_path / "runs" / "b" / "c" / "d", _make(step=30))
    found = discover_checkpoints(tmp_path)
    assert len(found) == 3
    # Sorted by created_at ascending; with near-equal timestamps, stable.
    steps = [m.train_step for _, m in found]
    assert sorted(steps) == steps


def test_discover_empty_on_missing_root(tmp_path):
    assert discover_checkpoints(tmp_path / "nope") == []


def test_discover_empty_on_file_root(tmp_path):
    f = tmp_path / "notadir.txt"
    f.write_text("hi")
    assert discover_checkpoints(f) == []


def test_checkpoint_name_is_human_readable():
    m = _make(step=1000, score=0.0123)
    name = checkpoint_name(m)
    assert "kuhn" in name and "ach" in name and "mirror" in name
    assert "s1000" in name
    assert "0.0123" in name


def test_checkpoint_name_omits_score_when_none():
    m = _make(step=5, score=None)
    name = checkpoint_name(m)
    assert "s5" in name
    # No trailing underscore-from-score.
    assert name.rstrip().endswith("s5")


def test_created_at_is_realistic():
    before = time.time()
    m = _make()
    after = time.time()
    assert before <= m.created_at <= after
