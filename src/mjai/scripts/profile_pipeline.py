"""Profile the pipeline: per-component CPU + GPU breakdown (AGENTS.md §8).

Runs a short experiment under cProfile (CPU) and torch.profiler (GPU when
available), then prints and saves a per-component breakdown. Used to find
bottlenecks before optimizing (AGENTS.md §8: "Profile before optimizing").

Usage::

    uv run python -m mjai.scripts.profile_pipeline --game kuhn --steps 50
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile a short mjai training run.")
    parser.add_argument("--game", default="kuhn")
    parser.add_argument("--algo", default="ach")
    parser.add_argument("--mode", default="mirror")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--out", default="profiles")
    args = parser.parse_args(argv)

    from mjai.scripts.experiment import ExperimentConfig, run_experiment
    from mjai.utils import gpu_assert

    gpu_assert.require_cpu()  # profiling runs on CPU for portability
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(
        game=args.game,
        algo=args.algo,
        self_play_mode=args.mode,
        n_steps=args.steps,
        out_dir=str(out_dir / f"profile_{args.game}_{args.algo}_{args.mode}"),
    )

    # CPU profile via cProfile.
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    run_experiment(cfg)  # discard the returned run_dir; profiling only needs side effects
    pr.disable()
    wall = time.perf_counter() - t0

    # Text summary.
    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(30)
    text = buf.getvalue()

    print(f"profile_pipeline: {args.steps} steps of {args.game}/{args.algo}/{args.mode}")
    print(f"wall time: {wall:.2f}s")
    print(text)

    (out_dir / f"profile_{args.game}_{args.algo}_{args.mode}.txt").write_text(text)
    (out_dir / f"profile_{args.game}_{args.algo}_{args.mode}.prof").write_bytes(
        _serialize_profile(pr)
    )
    (out_dir / f"summary_{args.game}_{args.algo}_{args.mode}.json").write_text(
        json.dumps(
            {
                "game": args.game,
                "algo": args.algo,
                "mode": args.mode,
                "steps": args.steps,
                "wall_seconds": wall,
                "top_by_cumtime": _top_funcs(ps, 10),
            },
            indent=2,
        )
    )
    print(f"\nWrote profile artifacts under {out_dir}/")
    return 0


def _serialize_profile(pr: cProfile.Profile) -> bytes:
    """Serialize a cProfile Profile to bytes via its stats tuple."""
    import pickle

    # pstats attaches .stats at runtime; mypy can't see it.
    return pickle.dumps(getattr(pr, "stats", {}))


def _top_funcs(ps: pstats.Stats, n: int) -> list[dict[str, object]]:
    """Return the top-N functions by cumulative time as JSON-friendly dicts."""
    out: list[dict[str, object]] = []
    # pstats attaches .stats at runtime; mypy can't see it on the typed class.
    stats: dict[tuple[str, int, str], tuple[int, int, float, float, object]] = getattr(
        ps, "stats", {}
    )
    for func, stat_row in sorted(stats.items(), key=lambda kv: -kv[1][3])[:n]:
        filename, lineno, name = func
        _cc, nc, _tt, ct, _callers = stat_row
        out.append(
            {"function": name, "file": filename, "line": lineno, "cumtime": ct, "ncalls": nc}
        )
    return out


if __name__ == "__main__":
    sys.exit(main())
