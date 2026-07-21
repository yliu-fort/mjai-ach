"""Summarize ACH paper-reproduction runs into a single JSON (analysis tool).

Walks ``runs/reproduce/*_ach_mlp_mirror/seed_*/tb`` TensorBoard event files and
aggregates the ``eval/exploitability`` curve of each run: final value, best
(min) value, and values at 1e5 / 1e6 / 5e6 / 1e7 env-steps. Emits per-game
mean/min/max across seeds. This is a read-only analysis entry point; training
metrics themselves live only in TensorBoard (AGENTS.md §1 D9).

Usage (repo venv)::

    python tools/summarize_reproduce.py --root runs/reproduce --out summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

CHECKPOINT_STEPS = [100_000, 1_000_000, 5_000_000, 10_000_000]


def _run_curve(tb_dir: Path) -> list[tuple[int, float]]:
    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    tag = "eval/exploitability"
    if tag not in ea.Tags()["scalars"]:
        return []
    return [(int(s.step), float(s.value)) for s in ea.Scalars(tag)]


def _at(curve: list[tuple[int, float]], target: int) -> float | None:
    if not curve:
        return None
    step, value = min(curve, key=lambda p: abs(p[0] - target))
    return value if abs(step - target) <= target * 0.2 + 1 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--root", default="runs/reproduce")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    runs: dict[str, dict[str, Any]] = {}
    for seed_dir in sorted(root.glob("*_ach_mlp_mirror/seed_*")):
        game = seed_dir.parent.name.replace("_ach_mlp_mirror", "")
        seed = seed_dir.name.replace("seed_", "")
        tb = seed_dir / "tb"
        curve = _run_curve(tb) if tb.exists() else []
        done = (seed_dir / "DONE").exists()
        if not curve:
            runs.setdefault(game, {})[seed] = {"done": done, "n_evals": 0}
            continue
        runs.setdefault(game, {})[seed] = {
            "done": done,
            "n_evals": len(curve),
            "final": curve[-1][1],
            "best": min(v for _, v in curve),
            "at": {str(t): _at(curve, t) for t in CHECKPOINT_STEPS},
        }

    per_game: dict[str, Any] = {}
    for game, seeds in runs.items():
        finals = [r["final"] for r in seeds.values() if r.get("done") and "final" in r]
        per_game[game] = {
            "n_done": len(finals),
            "final_mean": sum(finals) / len(finals) if finals else None,
            "final_min": min(finals) if finals else None,
            "final_max": max(finals) if finals else None,
        }

    out = {"per_game": per_game, "runs": runs}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(per_game, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
