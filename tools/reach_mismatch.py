"""Where does a trained policy's remaining exploitability live, and does ACH look there?

The hypothesis (H3): the residual RL floor is a **reach-weight mismatch**. ACH
learns from sampled rollouts, so the gradient an information set receives is
proportional to ``rho(I)`` -- the probability that a self-play episode visits it,
which multiplies **both** players' reach. A best response is under no such
obligation: it can steer the game into information sets the current policy
almost never enters, and its gain there is weighted by ``cfreach(I)`` -- chance
and the *opponent* only. Those two weightings differ by the learner's own reach,
which is exactly the quantity ACH's own guarantee has to pay for in the second,
``T``-independent term of Theorem 1 (``docs/paper_spec_ach.md`` §1.1).

So the measurement is: sort the information sets by how often training visits
them, then ask how much of the *available improvement* sits in the rarely
visited tail. Two weightings of the same per-information-set regret::

    regret(I)  = max_a A(I, a)                 exact, on-policy advantage
    R_sampled  = sum_I rho(I)     * regret(I)  what ACH's gradient sees
    R_counter  = sum_I cfreach(I) * regret(I)  what a best response can collect

If ``R_counter`` is dominated by information sets with negligible ``rho``, the
learner is structurally blind to its own worst weaknesses and no amount of
critic accuracy, batch size or capacity can fix it -- which is precisely the
pattern the previous ablations left unexplained.

Reads a checkpoint, writes a JSON summary. Analysis tool, not on the import path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tools.exact_ach import ExactAdvantage

from mjai.games.loader import load_game
from mjai.seqform.plan import nash_conv, realization_plans
from mjai.seqform.tree import build_sequence_form

_TINY = 1e-300


def analyze(game: str, checkpoint: Path | None) -> dict[str, object]:
    spec = load_game(game)
    sf = build_sequence_form(spec)
    engine = ExactAdvantage(sf)

    if checkpoint is None:
        behavior = torch.full((sf.num_infosets, sf.max_actions), 0.0, dtype=torch.float64)
        behavior = torch.softmax(behavior.masked_fill(~sf.legal_mask, float("-inf")), dim=1)
        label = "uniform"
    else:
        from mjai.agents.policy_factory import load_policy_from_checkpoint
        from mjai.eval.average_policy import behavior_of

        policy = load_policy_from_checkpoint(checkpoint, device="cpu")
        behavior = behavior_of(sf, policy)
        label = str(checkpoint)

    adv, rho = engine.compute(behavior)
    plans = realization_plans(sf, behavior)
    own_reach = torch.zeros(sf.num_infosets, dtype=torch.float64)
    for player in range(sf.num_players):
        rows = sf.rows_of(player)
        own_reach[rows] = plans[player].index_select(0, sf.parent_sequence[rows])
    cfreach = rho / own_reach.clamp(min=_TINY)

    regret = adv.masked_fill(~sf.legal_mask, float("-inf")).max(dim=1).values.clamp(min=0.0)
    r_sampled = float((rho * regret).sum())
    r_counter = float((cfreach * regret).sum())
    expl = float(nash_conv(sf, behavior, validate=False)) / 2.0

    # Sort by visitation ascending: how much counterfactual regret hides in the
    # tail that training barely touches?
    order = torch.argsort(rho)
    rho_sorted = rho[order]
    cr_sorted = (cfreach * regret)[order]
    cum_visits = torch.cumsum(rho_sorted, 0) / rho.sum().clamp(min=_TINY)
    cum_regret = torch.cumsum(cr_sorted, 0) / cr_sorted.sum().clamp(min=_TINY)

    # Regret mass held by the information sets receiving the least training.
    marks: dict[str, float] = {}
    for share in (0.001, 0.01, 0.05, 0.10, 0.25, 0.50):
        i = int(torch.searchsorted(cum_visits, torch.tensor(share, dtype=torch.float64)))
        i = min(i, sf.num_infosets - 1)
        marks[f"regret_in_bottom_{share:g}_of_visits"] = float(cum_regret[i])

    # And the converse: what share of visits the top-regret information sets get.
    order_r = torch.argsort(cr_sorted, descending=True)
    top = order_r[: max(1, sf.num_infosets // 100)]
    top_regret_share = float(cr_sorted[top].sum() / cr_sorted.sum().clamp(min=_TINY))
    top_visit_share = float(rho_sorted[top].sum() / rho.sum().clamp(min=_TINY))

    reached = rho > 1e-9
    return {
        "game": game,
        "policy": label,
        "num_infosets": sf.num_infosets,
        "exploitability": expl,
        "R_sampled": r_sampled,
        "R_counterfactual": r_counter,
        "mismatch_ratio": r_counter / r_sampled if r_sampled > 0 else float("inf"),
        "visitation": {
            "min": float(rho.min()),
            "median": float(rho.median()),
            "max": float(rho.max()),
            "frac_below_1e-6": float((rho < 1e-6).to(torch.float64).mean()),
            "frac_below_1e-4": float((rho < 1e-4).to(torch.float64).mean()),
            "effectively_unreached": int((~reached).sum()),
        },
        "concentration": marks,
        "top1pct_regret_infosets": {
            "regret_share": top_regret_share,
            "visit_share": top_visit_share,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="liars_dice1")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    res = analyze(args.game, args.checkpoint)
    print(json.dumps(res, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
