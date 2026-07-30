"""Where does the exact dynamics' residual exploitability actually live?

``tools/exact_ach.py`` establishes that the noiseless ACH operator stops at a
floor. That is a number, not a location. This probe asks which information sets
own it, and cross-tabulates the answer against the two candidate mechanisms:

  - **temporal drift** -- how much a row's own step weight moved while the policy
    trained, i.e. the ``(w_h(s) - w_l(s)) / w_h(s)`` that Theorem 1's second term
    sums over (ICLR slide p8 brackets the weight per information set and over
    iterations, so the span it charges for is a row's drift in time, not the
    spread of reach across rows);
  - **reach** -- the row's on-policy visitation ``rho(I)``, which is the
    cross-information-set dynamic range ``docs/liars_residual_floor.md`` measured.

The attribution is causal rather than a regret proxy: for each group of rows, the
dynamics' behaviour on those rows is **repaired** to exact Nash and everything
else is left as the dynamics left it. The resulting drop in exploitability is
what that group was costing. Repairing a group is the counterfactual "suppose the
operator had got these rows right", which is exactly the question, and it reuses
the primitive ``tools/starve_probe.py`` already validated in the other direction
(keep Nash on the top rows, uniform elsewhere).

Reading it: if the floor is concentrated in the high-drift rows, the temporal
reading of the bound is the operative one. If it tracks reach deciles instead,
the cross-row dynamic range is. Both columns are printed side by side so the
comparison is not a matter of choosing which table to show.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tools.exact_ach import ExactAdvantage

from mjai.games.loader import load_game
from mjai.seqform.plan import nash_conv
from mjai.seqform.tree import SequenceForm, build_sequence_form


def repaired(policy: torch.Tensor, nash: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``policy`` everywhere except ``rows``, which take their exact Nash values."""
    out = policy.clone()
    out[rows] = nash[rows]
    return out


def expl(sf: SequenceForm, behavior: torch.Tensor) -> float:
    return float(nash_conv(sf, behavior, validate=False)) / 2.0


def decile_groups(key: torch.Tensor, live: torch.Tensor, groups: int) -> list[torch.Tensor]:
    """Split the live rows into ``groups`` equal-count bins, ascending in ``key``."""
    idx = torch.nonzero(live, as_tuple=False).flatten()
    order = idx[torch.argsort(key[idx])]
    return list(torch.chunk(order, groups))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="liars_dice1")
    ap.add_argument(
        "--policy",
        type=Path,
        required=True,
        help="a --save-policy bundle from tools/exact_ach.py (behavior + weight_lo/hi)",
    )
    ap.add_argument("--nash", type=Path, default=Path("runs/nash_liars_dice1_behavior.pt"))
    ap.add_argument("--groups", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("runs/exact_ach/floor_microscope.json"))
    args = ap.parse_args()

    spec = load_game(args.game)
    sf = build_sequence_form(spec)
    bundle = torch.load(args.policy, weights_only=True)
    policy = bundle["behavior"].to(torch.float64)
    lo, hi = bundle["weight_lo"].to(torch.float64), bundle["weight_hi"].to(torch.float64)
    nash = torch.load(args.nash, weights_only=True).to(torch.float64)

    base = expl(sf, policy)
    ceiling = expl(sf, nash)
    print(f"{args.game}: dynamics floor {base:.5f}, exact Nash {ceiling:.3e}")

    # Repairing every row at once must land on the Nash value: a check that the
    # repair operator and the saved bundle line up before any group is trusted.
    everything = torch.arange(sf.num_infosets)
    full = expl(sf, repaired(policy, nash, everything))
    if abs(full - ceiling) > 1e-9:
        raise AssertionError(f"repairing all rows gave {full:.3e}, expected {ceiling:.3e}")
    print(f"  repair-all check: {full:.3e} == Nash  OK")

    engine = ExactAdvantage(sf)
    adv, rho = engine.compute(policy)
    regret = adv.masked_fill(~sf.legal_mask, float("-inf")).max(dim=1).values.clamp(min=0.0)
    live = hi > 0.0
    drift = torch.where(live, 1.0 - lo / hi.clamp(min=1e-300), torch.zeros_like(hi))

    results: dict[str, object] = {
        "game": args.game,
        "floor": base,
        "nash": ceiling,
        "n_live": int(live.sum()),
    }

    for name, key in (("drift", drift), ("rho", rho)):
        groups = decile_groups(key, live, args.groups)
        rows = []
        for g, members in enumerate(groups):
            after = expl(sf, repaired(policy, nash, members))
            rows.append(
                {
                    "bin": g,
                    "n": int(members.numel()),
                    f"{name}_median": float(key[members].median()),
                    "expl_after_repair": after,
                    "drop": base - after,
                    "share_of_floor": (base - after) / base,
                    "regret_sum": float(regret[members].sum()),
                    "rho_sum": float(rho[members].sum()),
                }
            )
        results[name] = rows
        print(f"\n  grouped by {name} (ascending; bin 9 = highest)")
        print(f"    {'bin':>4}{'n':>7}{'median':>12}{'expl':>10}{'drop':>10}{'share':>8}")
        for r in rows:
            print(
                f"    {r['bin']:>4}{r['n']:>7}{r[f'{name}_median']:>12.3e}"
                f"{r['expl_after_repair']:>10.5f}{r['drop']:>10.5f}"
                f"{r['share_of_floor']:>8.1%}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
