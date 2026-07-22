"""Fast eval/exploitability curve reader for TensorBoard event dirs.

EventAccumulator loads and accumulates every tag in a file; with 50MB event
files (one per reproduction run) that is minutes per run. This helper uses the
lazy EventFileLoader and keeps only the two eval tags, and a process-pool
wrapper parallelizes across runs.

Used by tools/summarize_reproduce.py and tools/compare_with_paper.py.
"""

from __future__ import annotations

import sys
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


if __name__ == "__main__":
    # smoke: python tools/tb_eval.py <tb_dir>
    curve = read_eval_curve(sys.argv[1])
    print(
        f"{len(curve)} points; first={curve[0] if curve else None}; last={curve[-1] if curve else None}"
    )
