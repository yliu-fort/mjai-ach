"""If you can only afford to learn K information sets, which K should you pick?

A weighting is, operationally, a *ranking*: it decides which rows a
capacity- or sample-limited learner gets right and which it leaves ignorant.
This probe evaluates that ranking directly, with **no training at all** and no
optimizer to confound the answer:

  1. rank the information sets by a candidate weight, descending;
  2. keep the exact Nash strategy on the top ``K`` rows;
  3. replace the remaining rows with the uniform distribution -- a neutral
     stand-in for "the learner never got a signal here";
  4. measure the exact exploitability of the resulting policy.

The curve of exploitability versus ``K`` is then a training-free lower bound on
what a learner under that weighting can achieve, and the comparison between
weightings is budget-matched by construction.

``uniform`` weighting has no ranking, so it is scored as a random selection
(averaged over several seeded draws) -- which is exactly what "every row counts
the same" means when the budget binds.

Why this is the decisive test for ``cf``: in a two-player zero-sum game
``NashConv(pi) = BR_0(pi_1) + BR_1(pi_0)``, so the first-order sensitivity of
exploitability to player p's behaviour at ``I`` is ``own_reach_p(I) *
cf^{BR}(I)``. The ``own_reach`` factor is present whatever the opponent plays;
``cf`` weighting drops it, so it is not the exploitability sensitivity at all,
and this probe should show its ranking wasting budget on rows the player's own
strategy never reaches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tools.weighted_distill_probe import infoset_weights

from mjai.games.loader import load_game
from mjai.seqform.plan import nash_conv
from mjai.seqform.tree import build_sequence_form


def damaged(sf, target: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """``target`` on the rows in ``keep``, uniform-over-legal everywhere else."""
    mask = sf.legal_mask
    uniform = mask.to(torch.float64) / mask.sum(dim=1, keepdim=True)
    out = uniform.clone()
    out[keep] = target[keep]
    return out


def curve(sf, target: torch.Tensor, order: torch.Tensor, budgets: list[int]) -> list[float]:
    """Exploitability after keeping the first ``K`` rows of ``order``."""
    values = []
    for k in budgets:
        keep = order[:k]
        values.append(float(nash_conv(sf, damaged(sf, target, keep), validate=False)) / 2.0)
    return values


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="liars_dice1")
    ap.add_argument("--nash", type=Path, default=Path("runs/nash_liars_dice1_behavior.pt"))
    ap.add_argument(
        "--weightings", nargs="*", default=["reach", "rho:0.5", "rho:0.25", "cf", "cf:0.5"]
    )
    ap.add_argument("--random-draws", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("runs/exact_ach/starve_probe.json"))
    args = ap.parse_args()

    spec = load_game(args.game)
    sf = build_sequence_form(spec)
    target = torch.load(args.nash, weights_only=True).to(torch.float64)
    n = sf.num_infosets
    budgets = [k for k in (100, 300, 1000, 3000, 6000, 12000, 18000, 22000, n) if k <= n]

    print(
        f"{args.game}: {n} infosets, target exploitability {float(nash_conv(sf, target)) / 2:.3e}"
    )
    print(f"budgets K = {budgets}")

    results: dict[str, list[float]] = {"_budgets": [float(b) for b in budgets]}

    # Uniform weighting = no ranking = a random subset of the budget.
    draws = []
    for seed in range(args.random_draws):
        g = torch.Generator().manual_seed(seed)
        draws.append(curve(sf, target, torch.randperm(n, generator=g), budgets))
    results["uniform (random subset)"] = [sum(c) / len(c) for c in zip(*draws, strict=True)]

    for kind in args.weightings:
        w = infoset_weights(sf, target, kind)
        order = torch.argsort(w, descending=True)
        results[kind] = curve(sf, target, order, budgets)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    header = "".join(f"{b:>9}" for b in budgets)
    print(f"\n{'ranking':<24}{header}")
    for name, vals in results.items():
        if name.startswith("_"):
            continue
        print(f"{name:<24}" + "".join(f"{v:>9.4f}" for v in vals))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
