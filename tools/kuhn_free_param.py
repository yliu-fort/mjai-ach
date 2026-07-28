"""Where does a trained Kuhn policy sit on the one-parameter Nash family?

2p Kuhn's Nash equilibria are a **segment, not a point**. Player 0's strategy
carries a free parameter ``alpha`` in ``[0, 1/3]``::

    P0 opening      bet(J) = alpha      bet(Q) = 0          bet(K) = 3*alpha
    P0 after p-b    call(J) = 0         call(Q) = alpha+1/3 call(K) = 1
    P1 facing a bet call(J) = 0         call(Q) = 1/3       call(K) = 1
    P1 after a pass bet(J) = 1/3        bet(Q) = 0          bet(K) = 1

Player 1's half is **unique**; only three of the twelve rows move with alpha.
``tools/exact_ach.py``-adjacent check: every member of this family has NashConv
0 to float64 (verified in :func:`verify_family`), and alpha outside [0, 1/3]
leaves the simplex — which is exactly the invalid policy AGENTS.md D15 records
OpenSpiel accepting silently.

Two consequences this tool measures, because exploitability alone cannot see
either of them:

1. **Along the family exploitability is exactly flat**, so the free direction
   carries no gradient signal. A policy can drift along it forever without the
   headline metric moving. :func:`read_alphas` reports the three *independent*
   readings of alpha (from bet(J), bet(K)/3 and call(Q)-1/3): on the family they
   agree, off it they do not, so their spread is an on-manifold test that needs
   no optimization.
2. **Distance to the Nash set is not distance to "the" Nash.** Picking one
   member (say alpha=1/3) and measuring L1 to it — as `docs/kuhn_tie_rootcause.md`
   §1.3 does — conflates moving *along* the family (free, harmless) with moving
   *off* it (real error). :func:`decompose` splits them.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from mjai.games.loader import load_game
from mjai.seqform.plan import nash_conv
from mjai.seqform.tree import SequenceForm, build_sequence_form

# Kuhn actions (OpenSpiel): 0 = Pass/Fold, 1 = Bet/Call.
BET = 1

# The alpha-dependent rows, as (information-set key, slope, intercept): the
# family sets P(bet | key) = slope * alpha + intercept.
FAMILY_FREE: tuple[tuple[str, float, float], ...] = (
    ("0", 1.0, 0.0),  # P0 opens with the jack: the bluff frequency alpha
    ("2", 3.0, 0.0),  # P0 opens with the king: 3*alpha (value bets track bluffs)
    ("1pb", 1.0, 1 / 3),  # P0 calls with the queen after check-bet
)

# The rows the family pins down; these are the same at every alpha.
FAMILY_FIXED: dict[str, float] = {
    "1": 0.0,  # P0 never opens with the queen
    "0pb": 0.0,  # P0 folds the jack
    "2pb": 1.0,  # P0 calls with the king
    "0b": 0.0,  # P1 folds the jack to a bet
    "1b": 1 / 3,  # P1 calls the queen a third of the time
    "2b": 1.0,  # P1 always calls with the king
    "0p": 1 / 3,  # P1 bluffs the jack after a check
    "1p": 0.0,  # P1 checks behind with the queen
    "2p": 1.0,  # P1 always bets the king after a check
}

ALPHA_MAX = 1 / 3


def family_behavior(sf: SequenceForm, alpha: float) -> torch.Tensor:
    """The Nash-family member at ``alpha`` as a behaviour strategy."""
    idx = {k: i for i, k in enumerate(sf.infoset_keys)}
    b = torch.zeros(sf.num_infosets, sf.max_actions, dtype=torch.float64)
    probs = dict(FAMILY_FIXED)
    for key, slope, intercept in FAMILY_FREE:
        probs[key] = slope * alpha + intercept
    for key, p in probs.items():
        b[idx[key], BET] = p
        b[idx[key], 1 - BET] = 1.0 - p
    return b


def verify_family(sf: SequenceForm, n: int = 21) -> float:
    """Max NashConv over a grid of the family — should be float64 zero."""
    worst = 0.0
    for i in range(n):
        alpha = ALPHA_MAX * i / (n - 1)
        worst = max(worst, abs(float(nash_conv(sf, family_behavior(sf, alpha)))))
    return worst


def bet_probs(sf: SequenceForm, behavior: torch.Tensor) -> dict[str, float]:
    """P(bet | information set) for all twelve rows, keyed by information set."""
    return {k: float(behavior[i, BET]) for i, k in enumerate(sf.infoset_keys)}


def read_alphas(probs: dict[str, float]) -> dict[str, float]:
    """Three independent readings of alpha; equal iff the policy is on the family."""
    return {
        "alpha_from_bet_J": probs["0"],
        "alpha_from_bet_K": probs["2"] / 3.0,
        "alpha_from_call_Q": probs["1pb"] - 1 / 3,
    }


def fit_alpha(probs: dict[str, float]) -> float:
    """Least-squares alpha over the three free rows (closed form)."""
    num = sum(slope * (probs[key] - intercept) for key, slope, intercept in FAMILY_FREE)
    den = sum(slope * slope for _key, slope, _intercept in FAMILY_FREE)
    return num / den


def decompose(sf: SequenceForm, behavior: torch.Tensor) -> dict[str, float]:
    """Split the deviation into 'along the family' and 'off the family'.

    ``off_family`` is the mean absolute deviation left after fitting the best
    alpha — the part of the error that no choice of equilibrium can excuse.
    ``alpha_hat`` is where along the segment the policy sits, and
    ``alpha_spread`` is the disagreement between its three readings.
    """
    probs = bet_probs(sf, behavior)
    alphas = read_alphas(probs)
    alpha_hat = fit_alpha(probs)
    clipped = min(max(alpha_hat, 0.0), ALPHA_MAX)

    free_residual = [
        abs(probs[key] - (slope * clipped + intercept)) for key, slope, intercept in FAMILY_FREE
    ]
    fixed_residual = [abs(probs[key] - target) for key, target in FAMILY_FIXED.items()]
    residuals = free_residual + fixed_residual

    # Player 1's half of the tree is unique: its deviation is pure error.
    p1_keys = ("0b", "1b", "2b", "0p", "1p", "2p")
    p1_dev = [abs(probs[k] - FAMILY_FIXED[k]) for k in p1_keys]

    return {
        "alpha_hat": alpha_hat,
        "alpha_spread": max(alphas.values()) - min(alphas.values()),
        "off_family_mean_abs": sum(residuals) / len(residuals),
        "off_family_max_abs": max(residuals),
        "p1_mean_abs": sum(p1_dev) / len(p1_dev),
        "exploitability": float(nash_conv(sf, behavior, validate=False)) / 2.0,
        **alphas,
    }


def distance_to_family(sf: SequenceForm, behavior: torch.Tensor, n: int = 2001) -> float:
    """Mean-|.| distance to the NE **set** (grid-minimized over alpha).

    The honest version of 'distance to Nash' for a game whose equilibrium set is
    a segment. Compare with the distance to a single arbitrary member.
    """
    ours = torch.tensor([behavior[i, BET] for i in range(sf.num_infosets)], dtype=torch.float64)
    best = math.inf
    for i in range(n):
        alpha = ALPHA_MAX * i / (n - 1)
        theirs = family_behavior(sf, alpha)[:, BET]
        best = min(best, float((ours - theirs).abs().mean()))
    return best


def analyze_run(run_dir: Path) -> list[dict[str, float]]:
    """Decompose every checkpoint in ``run_dir/checkpoints``, ordered by step."""
    from mjai.agents.policy_factory import load_policy_from_checkpoint
    from mjai.eval.average_policy import behavior_of

    spec = load_game("kuhn")
    sf = build_sequence_form(spec)
    ckpts = sorted(
        (d for d in (run_dir / "checkpoints").iterdir() if d.is_dir()),
        key=lambda d: int(d.name.split("_")[-1]),
    )
    rows = []
    for d in ckpts:
        step = int(d.name.split("_")[-1])
        policy = load_policy_from_checkpoint(d, device="cpu")
        behavior = behavior_of(sf, policy)
        row = {"step": step, **decompose(sf, behavior)}
        row["dist_to_set"] = distance_to_family(sf, behavior)
        row["dist_to_alpha_third"] = float(
            (behavior[:, BET] - family_behavior(sf, ALPHA_MAX)[:, BET]).abs().mean()
        )
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path("runs/avg_anchor/kuhn_seed0"))
    ap.add_argument("--out", type=Path, default=Path("runs/exact_ach/kuhn_alpha_trajectory.json"))
    args = ap.parse_args()

    spec = load_game("kuhn")
    sf = build_sequence_form(spec)
    worst = verify_family(sf)
    print(f"family verification: max |NashConv| over 21 alphas = {worst:.3e}")

    rows = analyze_run(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    hdr = (
        f"{'step':>9} {'expl':>9} {'alpha_hat':>10} {'spread':>8} "
        f"{'off_fam':>8} {'p1_dev':>8} {'d(set)':>8} {'d(a=1/3)':>9}"
    )
    print(hdr)
    for r in rows[:: max(1, len(rows) // 20)] + rows[-1:]:
        print(
            f"{r['step']:>9} {r['exploitability']:>9.5f} {r['alpha_hat']:>10.4f} "
            f"{r['alpha_spread']:>8.4f} {r['off_family_mean_abs']:>8.4f} "
            f"{r['p1_mean_abs']:>8.4f} {r['dist_to_set']:>8.4f} {r['dist_to_alpha_third']:>9.4f}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
