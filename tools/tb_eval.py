"""Fast eval/exploitability curve reader for TensorBoard event dirs.

EventAccumulator loads and accumulates every tag in a file; with 50MB event
files (one per reproduction run) that is minutes per run. This helper uses the
lazy EventFileLoader and keeps only the two eval tags, and a process-pool
wrapper parallelizes across runs.

Used by tools/summarize_reproduce.py and tools/compare_with_paper.py.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

EVAL_TAGS = ("eval/exploitability", "eval/nash_conv")


def read_eval_curve(
    tb_dir: str | Path, tag: str = "eval/exploitability"
) -> list[tuple[int, float]]:
    """Read one run's eval curve as [(env_step, value)], cheap tags only."""
    from tensorboard.backend.event_processing.event_file_loader import (
        EventFileLoader,
    )

    tb_dir = Path(tb_dir)
    events = sorted(tb_dir.glob("events.out.tfevents*"))
    if not events:
        return []
    points: list[tuple[int, float]] = []
    for ev_file in events:
        for event in EventFileLoader(str(ev_file)).Load():
            if event.HasField("summary"):
                for v in event.summary.value:
                    if v.tag != tag:
                        continue
                    if v.HasField("simple_value"):
                        points.append((int(event.step), float(v.simple_value)))
                    elif v.HasField("tensor"):
                        from tensorboard.util import tensor_util

                        arr = tensor_util.make_ndarray(v.tensor)
                        points.append((int(event.step), float(arr)))
    points.sort(key=lambda p: p[0])
    return points


def read_tags(tb_dir: str | Path, tags: Sequence[str]) -> dict[str, list[tuple[int, float]]]:
    """Read several tags in ONE pass over the event files.

    :func:`read_eval_curve` walks the whole file per tag, which is fine for the
    two eval tags but not for the ~15 per-update ``train/*`` scalars the league
    auditor needs. Returns a dict keyed by tag; tags absent from the file are
    absent from the result (an empty list would read as "logged but empty").
    """
    from tensorboard.backend.event_processing.event_file_loader import (
        EventFileLoader,
    )

    wanted = set(tags)
    out: dict[str, list[tuple[int, float]]] = {}
    for ev_file in sorted(Path(tb_dir).glob("events.out.tfevents*")):
        for event in EventFileLoader(str(ev_file)).Load():
            if not event.HasField("summary"):
                continue
            for v in event.summary.value:
                if v.tag not in wanted:
                    continue
                if v.HasField("simple_value"):
                    value = float(v.simple_value)
                elif v.HasField("tensor"):
                    from tensorboard.util import tensor_util

                    value = float(tensor_util.make_ndarray(v.tensor))
                else:
                    continue
                out.setdefault(v.tag, []).append((int(event.step), value))
    for points in out.values():
        points.sort(key=lambda p: p[0])
    return out


def downsample(
    points: list[tuple[int, float]], max_points: int, *, preserve_peaks: bool = False
) -> list[tuple[int, float]]:
    """Thin a ``[(step, value)]`` curve to at most ``max_points`` for DISPLAY.

    Read-side only — the event files keep their full per-update resolution
    (experiment.py logs ``train/*`` at every update on purpose, because the
    blow-ups the telemetry exists to catch are intermittent). This just bounds
    what a plot has to carry: a paper-budget run is ~1.5e5 updates per tag, and
    plotting or pickling every point across seeds x tags is the cost the
    notebooks were paying.

    ``max_points <= 0`` disables thinning. Points are bucketed into
    ``max_points`` contiguous index ranges and one representative is kept per
    bucket:

    - default: the bucket's LAST point (uniform stride; the exact tail value
      survives, which matters for "final" readouts);
    - ``preserve_peaks``: the bucket's largest-magnitude point, so an
      intermittent grad-norm spike is not strided away. The kept point carries
      its own step, so the spike still plots at the right x.
    """
    n = len(points)
    if max_points <= 0 or n <= max_points:
        return points
    out: list[tuple[int, float]] = []
    for b in range(max_points):
        lo = (b * n) // max_points
        hi = ((b + 1) * n) // max_points
        if lo >= hi:
            continue
        bucket = points[lo:hi]
        out.append(max(bucket, key=lambda sv: abs(sv[1])) if preserve_peaks else bucket[-1])
    return out


def _read_one(args: tuple[str, str]) -> tuple[str, list[tuple[int, float]]]:
    tb_dir, tag = args
    return tb_dir, read_eval_curve(tb_dir, tag)


def read_many(
    tb_dirs: list[str | Path], tag: str = "eval/exploitability", workers: int = 6
) -> dict[str, list[tuple[int, float]]]:
    """Parallel read of many runs' eval curves; keys are the tb_dir strings."""
    jobs = [(str(d), tag) for d in tb_dirs]
    if not jobs:
        return {}
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        return dict(pool.map(_read_one, jobs))


def _read_tags_downsampled(
    args: tuple[str, tuple[str, ...], int, tuple[str, ...]],
) -> tuple[str, dict[str, list[tuple[int, float]]]]:
    tb_dir, tags, max_points, peak_tags = args
    peaks = set(peak_tags)
    series = read_tags(tb_dir, tags)  # ONE pass over the file for all tags
    thinned = {
        tag: downsample(pts, max_points, preserve_peaks=tag in peaks) for tag, pts in series.items()
    }
    return tb_dir, thinned


def read_many_tags(
    tb_dirs: list[str | Path],
    tags: Sequence[str],
    *,
    max_points: int = 2000,
    peak_tags: Sequence[str] = (),
    workers: int = 6,
) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Parallel, single-pass, downsampled read of many per-update ``train/*`` curves.

    The telemetry cells used to call :func:`read_many` once per tag — N full
    scans of each (often 50-90 MB) event file — and then shipped and plotted
    every per-update point. This reads ALL ``tags`` in ONE pass per file
    (:func:`read_tags`) and thins each curve to ``max_points`` INSIDE the
    worker, so the full-resolution list never crosses the process boundary or
    reaches matplotlib. Tags in ``peak_tags`` keep their spikes
    (``preserve_peaks``); everything else is strided.

    Returns ``{tb_dir_str: {tag: [(step, value)]}}``; tags absent from a file
    are absent from its inner dict (never a misleading empty list).
    """
    jobs = [(str(d), tuple(tags), max_points, tuple(peak_tags)) for d in tb_dirs]
    if not jobs:
        return {}
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        return dict(pool.map(_read_tags_downsampled, jobs))


if __name__ == "__main__":
    # smoke: python tools/tb_eval.py <tb_dir>
    curve = read_eval_curve(sys.argv[1])
    print(
        f"{len(curve)} points; first={curve[0] if curve else None}; last={curve[-1] if curve else None}"
    )
