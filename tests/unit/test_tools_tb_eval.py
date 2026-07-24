"""Unit tests for tools/tb_eval.py (downsample + multi-tag reader).

No training. The pure ``downsample`` cases are self-contained; the
``read_tags``/``read_many_tags`` round-trip emits a real (tiny) tfevents file
with a real ``SummaryWriter`` so the test exercises the actual lazy loader path
that the notebooks hit (AGENTS.md §6: one writer per run).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import tb_eval  # noqa: E402

# --------------------------------------------------------------------------- #
# downsample — pure, no I/O
# --------------------------------------------------------------------------- #


def test_downsample_disabled_when_max_points_le_zero():
    pts = [(i, float(i)) for i in range(5)]
    assert tb_eval.downsample(pts, 0) == pts
    assert tb_eval.downsample(pts, -3) == pts


def test_downsample_unchanged_when_under_cap():
    pts = [(0, 0.0), (1, 1.0), (2, 2.0)]
    assert tb_eval.downsample(pts, 10) == pts


def test_downsample_default_keeps_bucket_tail_and_exact_last_value():
    # 10 points -> 5 buckets; default keeps each bucket's LAST point.
    pts = [(i, float(i)) for i in range(10)]
    assert tb_eval.downsample(pts, 5) == [(1, 1.0), (3, 3.0), (5, 5.0), (7, 7.0), (9, 9.0)]


def test_downsample_preserve_peaks_keeps_max_magnitude_per_bucket():
    # 4 points -> 2 buckets; peak mode keeps the largest-|value| in each bucket,
    # carrying that point's OWN step (so the spike plots at the right x).
    pts = [(0, 0.0), (1, 5.0), (2, -9.0), (3, 1.0)]
    assert tb_eval.downsample(pts, 2, preserve_peaks=True) == [(1, 5.0), (2, -9.0)]


def test_downsample_does_not_change_input_length_property_below_threshold():
    # A short intermittent spike inside one bucket survives in peak mode and is
    # strided away in default mode — the exact asymmetry the grad_norm panel
    # relies on.
    pts = [(0, 1.0), (1, 1.0), (2, 100.0), (3, 1.0)]
    assert tb_eval.downsample(pts, 1, preserve_peaks=True) == [(2, 100.0)]
    assert tb_eval.downsample(pts, 1) == [(3, 1.0)]


# --------------------------------------------------------------------------- #
# read_tags / read_many_tags — real tiny tfevents round-trip
# --------------------------------------------------------------------------- #


@pytest.fixture
def tiny_tb_dir(tmp_path: Path) -> Path:
    """Two tags across two steps, written by a real SummaryWriter."""
    from torch.utils.tensorboard import SummaryWriter

    d = tmp_path / "seed_0" / "tb"
    d.mkdir(parents=True)
    w = SummaryWriter(str(d))
    w.add_scalar("train/grad_norm", 1.0, 0)
    w.add_scalar("train/clip_frac", 0.1, 0)
    w.add_scalar("train/grad_norm", 2.0, 1)
    w.close()
    return d


def test_read_tags_single_pass_returns_all_requested(tiny_tb_dir: Path):
    got = tb_eval.read_tags(tiny_tb_dir, ["train/grad_norm", "train/clip_frac"])
    assert set(got) == {"train/grad_norm", "train/clip_frac"}
    assert [s for s, _ in got["train/grad_norm"]] == [0, 1]
    assert got["train/grad_norm"][1][1] == pytest.approx(2.0)


def test_read_tags_absent_tag_is_absent_not_empty(tiny_tb_dir: Path):
    # "logged but empty" and "never logged" must stay distinguishable (docstring).
    got = tb_eval.read_tags(tiny_tb_dir, ["train/grad_norm", "train/never"])
    assert "train/never" not in got


def test_read_many_tags_parallel_single_pass_and_downsamples(tiny_tb_dir: Path):
    got = tb_eval.read_many_tags(
        [tiny_tb_dir],
        ["train/grad_norm", "train/clip_frac"],
        max_points=1,
        peak_tags=("train/grad_norm",),
    )
    inner = got[str(tiny_tb_dir)]
    # grad_norm had 2 points -> thinned to its largest-magnitude point (step 1).
    assert inner["train/grad_norm"] == [(1, pytest.approx(2.0))]
    # clip_frac had 1 point -> under cap, unchanged.
    assert inner["train/clip_frac"][0][0] == 0


def test_read_many_tags_empty_input_returns_empty_no_workers():
    # Must not spin up a process pool for zero jobs.
    assert tb_eval.read_many_tags([], ["train/grad_norm"]) == {}
