"""Liar's Dice floor FIX (Phase C / B1+B4): does clipping 1/pi_old break the floor?

Phase B4 (tools/lth_floor_sweep.py) located the root cause: the unbounded 1/pi_old
forces the gate (l_th) to stay tight (loosen to >=8 and 1/pi_old blows up -> crash);
the tight gate caps sharpening at entropy ~1.0 = the ~0.20 floor. This tests the
fix: clip 1/pi_old (AlgoConfig.iw_clip) so the gate can be safely loosened.

Reads the fix arms and prints, vs the baseline (l_th=2, no clip, floor ~0.205):
  * current exploitability (tail-10%) and best-iterate,
  * policy entropy (does clipping let it sharpen below ~1.0?),
  * crash status (the unclipped l_th=8 crashed at 1.2e6).

Decisive read: if a l_th=8 + clip arm runs stably to 1e7 AND reaches
exploitability < 0.205 with entropy < ~1.0, clipping decouples the gate from
1/pi_old and breaks the floor -- the root cause is confirmed AND fixed.

All numbers in EXPLOITABILITY (= NashConv/2 at 2p). Usage::

    uv run python tools/fix_floor.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ARMS = [
    ("baseline l_th=2 (no clip)", "runs/avg_anchor/liars_dice1_seed0", None),
    (
        "B1 alone: l_th=2, iw20",
        "runs/ab_fix/liars_fix_lth2_iw20_seed0",
        "runs/ab_fix/lth2_iw20.stdout",
    ),
    ("l_th=4, iw20", "runs/ab_fix/liars_fix_lth4_iw20_seed0", "runs/ab_fix/lth4_iw20.stdout"),
    ("l_th=8, iw20", "runs/ab_fix/liars_fix_lth8_iw20_seed0", "runs/ab_fix/lth8_iw20.stdout"),
    ("l_th=8, iw100", "runs/ab_fix/liars_fix_lth8_iw100_seed0", "runs/ab_fix/lth8_iw100.stdout"),
]
CURRENT = "eval/exploitability"
ENTROPY = "entropy"


def _status(rows: list[dict], stdout: Path | None) -> str:
    """done / running / CRASHED — crash read from stdout signature, not point count."""
    if stdout is not None and stdout.exists():
        txt = stdout.read_text(encoding="utf-8", errors="ignore")
        if "Traceback" in txt or "SpielError" in txt:
            last_env = rows[-1].get("env_steps", 0) if rows else 0
            return f"CRASHED @{int(last_env):.1g}"
        if "mjai-train: done" in txt:
            return "done"
    n = len(rows)
    if n >= 95:
        return "done"
    return f"running({n}%)"


def _rows(run_dir: Path) -> list[dict] | None:
    f = run_dir / "train_curve.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def _tail(rows: list[dict], key: str, frac: float = 0.1) -> float:
    vals = [float(r[key]) for r in rows if key in r]
    if not vals:
        return float("nan")
    k = max(1, int(len(vals) * frac))
    return sum(vals[-k:]) / k


def _best(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r]
    return min(vals) if vals else float("nan")


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    baseline_floor = float("nan")
    print(f"\n{'arm':<28}{'current':>11}{'best-iter':>11}{'entropy':>10}{'pts':>6}  status")
    print("-" * 80)
    for label, run, stdout in ARMS:
        rows = _rows(Path(run))
        if rows is None:
            print(f"{label:<28}{'(missing)':>11}")
            continue
        cur = _tail(rows, CURRENT)
        best = _best(rows, CURRENT)
        ent = _tail(rows, ENTROPY) if ENTROPY in rows[0] else float("nan")
        n = len(rows)
        status = _status(rows, Path(stdout) if stdout else None)
        if "baseline" in label:
            baseline_floor = cur
        delta = ""
        if not math.isnan(baseline_floor) and not math.isnan(cur) and "baseline" not in label:
            d = cur - baseline_floor
            delta = f"  ({'+' if d >= 0 else ''}{d:.3f} vs base)"
        print(f"{label:<28}{cur:>11.4g}{best:>11.4g}{ent:>10.3f}{n:>6}  {status}{delta}")
    if not math.isnan(baseline_floor):
        print(f"\nbaseline floor (l_th=2, no clip): {baseline_floor:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
