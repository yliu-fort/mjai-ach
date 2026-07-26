"""Liar's Dice β-sweep: is the ~0.18 exploitability floor the entropy regularizer?

Reads the β-sweep arms (β = 1e-2, 3e-3, 1e-3, 1e-4, 0) and plots, against β:

  * current-policy exploitability (tail-10% mean) — what ACH converges to,
  * uniform-average exploitability — the D16 average-strategy floor,
  * best-iterate exploitability (min over the eval curve) — the hardest cap.

The decisive read: if current exploitability -> 0 as β -> 0, the ~0.18 floor is
the entropy regularizer's soft-equilibrium fixed point (ACH is at its objective's
floor; the paper sits there too). If it plateaus near 0.18 even at β=0, the floor
is optimization error and Phase B (1/π_old clip, legalmean, critic, ...) owns it.

All numbers in EXPLOITABILITY (= NashConv/2 at 2p), the repo/paper convention —
NOT eval/nash_conv.

Usage::

    uv run python tools/beta_floor_sweep.py
    uv run python tools/beta_floor_sweep.py --anchor runs/avg_anchor/liars_dice1_seed0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# β value -> run directory. β=1e-2 is the committed anchor run; the rest live
# under runs/ab_beta/.
BETAS = [
    (1e-2, "runs/avg_anchor/liars_dice1_seed0"),
    (3e-3, "runs/ab_beta/liars_beta3e-3_seed0"),
    (1e-3, "runs/ab_beta/liars_beta1e-3_seed0"),
    (1e-4, "runs/ab_beta/liars_beta1e-4_seed0"),
    (0.0, "runs/ab_beta/liars_beta0_seed0"),
]
CURRENT = "eval/exploitability"
UNIFORM = "eval/avg_exploitability"
BEST = None  # computed as min over the curve, no extra column needed


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
    p.add_argument("--out", default="docs/figs/liars_beta_floor.png")
    args = p.parse_args(argv)
    out = Path(args.out)

    rows_table: list[dict] = []
    for beta, run in BETAS:
        rd = Path(run)
        rows = _rows(rd)
        if rows is None:
            print(f"  (missing: β={beta} {run})")
            continue
        cur = _tail(rows, CURRENT)
        uni = _tail(rows, UNIFORM) if UNIFORM in rows[-1] else float("nan")
        best = _best(rows, CURRENT)
        n = len(rows)
        rows_table.append({"beta": beta, "current": cur, "uniform": uni, "best": best, "n": n})

    rows_table.sort(key=lambda r: r["beta"], reverse=True)
    print(f"\n{'beta':>8}{'current':>12}{'unif-avg':>12}{'best-iter':>12}{'n_pts':>8}")
    print("-" * 56)
    for r in rows_table:
        print(
            f"{r['beta']:>8g}   {r['current']:>10.4g}   {r['uniform']:>10.4g}"
            f"   {r['best']:>10.4g}   {r['n']:>7}"
        )

    # Plot.
    bs = [r["beta"] for r in rows_table if r["beta"] > 0]
    cur = [r["current"] for r in rows_table if r["beta"] > 0]
    uni = [r["uniform"] for r in rows_table if r["beta"] > 0]
    bst = [r["best"] for r in rows_table if r["beta"] > 0]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(bs, cur, "o-", lw=1.8, label="current policy (tail-10%)")
    ax.plot(bs, uni, "s--", lw=1.4, label="uniform average (D16)")
    ax.plot(bs, bst, "^:", lw=1.4, label="best iterate (min over curve)")
    # β=0 point(s) if present — plot at a small x offset on the log axis.
    zrows = [r for r in rows_table if r["beta"] == 0]
    if zrows:
        x0 = min(bs) / 10 if bs else 1e-3
        for key in ("current", "uniform", "best"):
            ax.plot([x0], [zrows[0][key]], "x", color="gray")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("entropy coef β")
    ax.set_ylabel("exploitability")
    ax.set_title("Liar's Dice: exploitability floor vs entropy regularizer β (seed 0)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nsaved: {out}")

    # Verdict.
    if len(cur) >= 2:
        hi, lo = cur[0], cur[-1]  # β=1e-2 -> smallest β
        ratio = hi / lo if lo else float("inf")
        print(
            f"\ncurrent exploitability: β=1e-2 -> {hi:.4g}, smallest-β -> {lo:.4g} (ratio {ratio:.2f}x)"
        )
        if lo < hi / 3:
            print("VERDICT: floor DROPS with β -> the floor is (largely) the entropy regularizer.")
        else:
            print(
                "VERDICT: floor does NOT drop with β -> the floor is optimization error (Phase B)."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
