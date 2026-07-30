"""Liar's Dice l_th sweep (Phase B4): is the gate what caps policy sharpening?

Phase A (tools/beta_floor_sweep.py) showed the ~0.18 exploitability floor and
the policy entropy (~1.0) are both beta-INVARIANT -- the policy is pinned at a
soft plateau by something other than the entropy bonus. This sweeps the ACH
logit-gate threshold l_th to test whether the GATE is that cap.

Reads l_th in {1, 2 (anchor), 4, 8, 1e6 (no gate)} and plots, against l_th:
  * current-policy exploitability (tail-10%) and best-iterate,
  * the policy entropy (the Phase-A clue: does raising l_th let it drop below ~1?).

Decisive read: if raising l_th drops entropy AND exploitability together, the
gate is the sharpening cap (Phase A prediction confirmed). The no-gate arm
(l_th=1e6) is expected to sharpen hard (entropy -> 0) but blow up (1/pi_old) --
it often crashes partway; its pre-crash points still show the sharpening.

All numbers in EXPLOITABILITY (= NashConv/2 at 2p), the repo/paper convention.

Usage::

    uv run python tools/lth_floor_sweep.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# l_th value -> run directory. l_th=2 is the committed anchor run.
LTHS = [
    (1.0, "runs/ab_lth/liars_lth1_seed0"),
    (2.0, "runs/avg_anchor/liars_dice1_seed0"),
    (4.0, "runs/ab_lth/liars_lth4_seed0"),
    (8.0, "runs/ab_lth/liars_lth8_seed0"),
    (1e6, "runs/ab_lth/liars_lth1e6_seed0"),
]
CURRENT = "eval/exploitability"
UNIFORM = "eval/avg_exploitability"
ENTROPY = "entropy"


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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="docs/figs/liars_lth_floor.png")
    args = p.parse_args(argv)
    out = Path(args.out)

    table: list[dict] = []
    for lth, run in LTHS:
        rows = _rows(Path(run))
        if rows is None:
            print(f"  (missing: l_th={lth} {run})")
            continue
        cur = _tail(rows, CURRENT)
        uni = _tail(rows, UNIFORM) if UNIFORM in rows[-1] else float("nan")
        best = _best(rows, CURRENT)
        ent = _tail(rows, ENTROPY) if ENTROPY in rows[0] else float("nan")
        table.append(
            {
                "lth": lth,
                "current": cur,
                "uniform": uni,
                "best": best,
                "entropy": ent,
                "n": len(rows),
            }
        )

    print(
        f"\n{'l_th':>9}{'current':>11}{'unif-avg':>11}{'best-iter':>11}{'entropy':>10}{'n_pts':>8}"
    )
    print("-" * 60)
    for r in table:
        tag = " (crashed)" if r["n"] < 80 else ""
        print(
            f"{r['lth']:>9g}   {r['current']:>9.4g}   {r['uniform']:>9.4g}"
            f"   {r['best']:>9.4g}   {r['entropy']:>8.3f}   {r['n']:>7}{tag}"
        )

    # Plot exploitability + entropy vs l_th (finite l_th only; no-gate is off-scale).
    fin = [r for r in table if r["lth"] < 1e5]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
    ls = [r["lth"] for r in fin]
    ax1.plot(ls, [r["current"] for r in fin], "o-", lw=1.8, label="current (tail-10%)")
    ax1.plot(ls, [r["uniform"] for r in fin], "s--", lw=1.4, label="uniform-avg")
    ax1.plot(ls, [r["best"] for r in fin], "^:", lw=1.4, label="best iterate")
    ax1.set_ylabel("exploitability")
    ax1.set_title("Liar's Dice: floor & entropy vs ACH gate threshold l_th (seed 0)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=9)
    ax2.plot(
        ls, [r["entropy"] for r in fin], "o-", color="tab:green", lw=1.8, label="policy entropy"
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("logit gate threshold l_th")
    ax2.set_ylabel("policy entropy (nats)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=9)
    # Annotate the no-gate point if it sharpened before crashing.
    ng = [r for r in table if r["lth"] >= 1e5]
    if ng and ng[0]["n"] > 0 and fin:
        ax2.annotate(
            f"no-gate: entropy {ng[0]['entropy']:.2f}\nthen 1/pi_old blows up",
            xy=(max(ls), fin[-1]["entropy"]),
            xytext=(0.5, 0.3),
            textcoords="axes fraction",
            fontsize=8,
            color="gray",
            arrowprops={"arrowstyle": "->", "color": "gray"},
        )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
