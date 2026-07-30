"""Do the rows that escape the gate's box own the ``1/pi_old`` gradient blow-up?

Two documents name two different things for Liar's Dice:
``docs/ach_operator_theory.md`` says the gate's box is a hard invariant for the
exact operator but only a soft attractor for the MLP (rows get carried past
their wall by updates belonging to other rows), and
``docs/liars_floor_ablation.md`` says the MLP's residual is driven by the
unbounded ``1/pi_old`` in the ACH loss. This probe asks whether those are one
mechanism: escaping the box means sharpening past what the gate permits, and a
sharpened row is exactly a row with a tiny ``pi_old`` waiting to be sampled.

**What would be circular, and is therefore not the test.** Separation and the
smallest action probability are algebraically tied: ``sep = log pi_max - log
pi_min``, so ``1/pi_min >= e^sep`` always. Correlating those two would prove
nothing. What is *not* implied is where the actual gradient VARIANCE lands,
because that carries a visitation and an advantage the separation knows nothing
about. Sampling ``a ~ pi`` and paying ``eta*y*c*A/pi_old`` gives a per-row
second moment

    V(s) = rho(s) * sum_a pi_a * (A(s,a) / pi_a)^2 = rho(s) * sum_a A(s,a)^2 / pi(a|s)

and the question is what share of ``sum_s V(s)`` sits in rows with
``sep(s) > 2*l_th``. Rows can overshoot and still contribute nothing (no
visitation, or no advantage left to pay for), so the share is a real
measurement.

**Measured outcome, recorded here so the statistic is not re-derived.** The
SHARE turned out not to discriminate: it is 92-100% for every arm including the
exact operator's own settled policy, because in any policy the variance is
dominated by whichever rows are most separated, over the wall or not. The
informative quantity is the LEVEL, which the share normalizes away -- at
l_th = 1 / 2 / 4 the MLP's total is 2.1e2 / 3.8e3 / 4.8e5, i.e. two orders of
magnitude per step of l_th, against box bounds falling 0.3596 / 0.1030 / 0.0054
over the same range. Both columns are printed; read the level.

Run::

    uv run python -m tools.box_escape_variance \\
        --run-dir runs/ab_fix/liars_fix_lth2_iw20_seed0 \\
        --policy runs/exact_ach/M_liars_rho0.5_policy.pt

Not on the ``mjai`` import path (AGENTS.md sec.4). float64 (D19).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tools.exact_ach import ExactAdvantage
from tools.gate_box_probe import _LIVE, load_run, row_stats

from mjai.games.loader import load_game
from mjai.seqform.plan import nash_conv
from mjai.seqform.tree import SequenceForm, build_sequence_form


def variance_per_row(
    sf: SequenceForm, behavior: torch.Tensor, engine: ExactAdvantage
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(V(s), rho(s))`` where ``V(s) = rho(s) * sum_a A(s,a)^2 / pi(a|s)``.

    The second moment of the ACH policy term's per-sample logit gradient, whose
    ``1/pi_old`` is what nothing in the loss bounds under synchronous self-play.
    Illegal slots contribute zero; ``pi`` is floored so an exactly-zero legal
    probability reports a large finite number rather than an inf that would
    swallow every share into one row.
    """
    adv, rho = engine.compute(behavior)
    safe = behavior.clamp(min=1e-12)
    per_action = torch.where(sf.legal_mask, adv * adv / safe, torch.zeros_like(adv))
    return rho * per_action.sum(dim=1), rho


def summarize(
    name: str, sf: SequenceForm, behavior: torch.Tensor, engine: ExactAdvantage, l_th: float
) -> dict[str, float]:
    v, rho = variance_per_row(sf, behavior, engine)
    live = rho > _LIVE
    sep = row_stats(behavior, sf.legal_mask)["sep"]
    wall = 2.0 * l_th
    over = (sep > wall + 1e-6) & live
    total = float(v[live].sum())
    share = float(v[over].sum()) / total if total > 0 else 0.0
    rec = {
        "name": name,
        "l_th": l_th,
        "expl": float(nash_conv(sf, behavior, validate=False)) / 2.0,
        "live_rows": int(live.sum()),
        "over_rows": int(over.sum()),
        "over_row_frac": float(over.sum()) / max(int(live.sum()), 1),
        "total_variance": total,
        "variance_share_over_wall": share,
        "mean_sep": float((rho[live] * sep[live]).sum() / rho[live].sum()),
    }
    print(
        f"{name:>34} | l_th {l_th:.0f} | expl {rec['expl']:.4f}"
        f" | rows over wall {rec['over_row_frac']:6.1%}"
        f" | VARIANCE over wall {share:6.1%}"
        f" | total var {total:.3e}"
    )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="liars_dice1")
    ap.add_argument("--run-dir", type=Path, action="append", default=[])
    ap.add_argument("--policy", type=Path, action="append", default=[])
    ap.add_argument("--l-th", type=float, default=2.0, help="for --policy bundles only")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sf = build_sequence_form(load_game(args.game))
    engine = ExactAdvantage(sf)
    print(f"game={args.game}   V(s) = rho(s) * sum_a A(s,a)^2 / pi(a|s)")
    print("-" * 132)

    out = []
    for d in args.run_dir:
        behavior, l_th, step = load_run(d, sf)
        out.append(summarize(f"MLP {d.name}", sf, behavior, engine, l_th))
        out[-1]["step"] = step
    for p in args.policy:
        behavior = torch.load(p, weights_only=True)["behavior"].to(torch.float64)
        out.append(summarize(f"exact {p.stem}", sf, behavior, engine, args.l_th))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"game": args.game, "arms": out}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
