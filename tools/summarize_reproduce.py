"""Summarize ACH paper-reproduction runs into a single JSON (analysis tool).

Walks completed (DONE-marked) runs under ``runs/reproduce/*_ach_mlp_mirror/seed_*``
and aggregates each run's ``eval/exploitability`` curve: final value, best
(min) value, and values at 1e5 / 1e6 / 5e6 / 1e7 env-steps. Emits per-game
mean/min/max across seeds. Read-only analysis entry point; training metrics
themselves live only in TensorBoard (AGENTS.md §1 D9).

Usage (repo venv)::

    python tools/summarize_reproduce.py --root runs/reproduce --out summary.json
    python tools/summarize_reproduce.py --pattern '*_ach_mlp_league' --out league.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tb_eval import read_many

CHECKPOINT_STEPS = [100_000, 1_000_000, 5_000_000, 10_000_000]
# Run-dir glob under --root; the mirror arm is the default (paper protocol).
DEFAULT_PATTERN = "*_ach_mlp_mirror"


def pattern_suffix(pattern: str) -> str:
    """Strip the leading ``*`` from a run-dir glob, leaving the name suffix."""
    return pattern.lstrip("*")


def game_from_dirname(dirname: str, pattern: str = DEFAULT_PATTERN) -> str:
    """Recover the game name from a run dir like ``kuhn_ach_mlp_mirror``."""
    suffix = pattern_suffix(pattern)
    if suffix and dirname.endswith(suffix):
        return dirname[: -len(suffix)]
    return dirname


def _at(curve: list[tuple[int, float]], target: int) -> float | None:
    if not curve:
        return None
    step, value = min(curve, key=lambda p: abs(p[0] - target))
    return value if abs(step - target) <= target * 0.2 + 1 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--root", default="runs/reproduce")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Run-dir glob under --root (default: %(default)s).",
    )
    args = parser.parse_args()

    root = Path(args.root)
    seed_dirs = sorted(root.glob(f"{args.pattern}/seed_*"))
    done_dirs = [sd for sd in seed_dirs if (sd / "DONE").exists()]
    curves = read_many([sd / "tb" for sd in done_dirs])

    runs: dict[str, dict[str, Any]] = {}
    for sd in done_dirs:
        game = game_from_dirname(sd.parent.name, args.pattern)
        seed = sd.name.replace("seed_", "")
        curve = curves.get(str(sd / "tb"), [])
        entry: dict[str, Any] = {"done": True, "n_evals": len(curve)}
        if curve:
            entry.update(
                {
                    "final": curve[-1][1],
                    "best": min(v for _, v in curve),
                    "at": {str(t): _at(curve, t) for t in CHECKPOINT_STEPS},
                }
            )
        runs.setdefault(game, {})[seed] = entry

    per_game: dict[str, Any] = {}
    for game, seeds in runs.items():
        finals = [r["final"] for r in seeds.values() if "final" in r]
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
