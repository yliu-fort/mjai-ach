"""The best exploitability any policy inside the logit gate's box can achieve.

``docs/ach_operator_theory.md`` derives that the paper's one-sided logit gate
traps every persistently-signed logit in ``[-l_th, l_th]``, and
``tools/gate_box_probe.py`` (N3) confirms the exact dynamics rests exactly on
those walls. That makes the box a *constraint set*, and the obvious question
becomes: what is the best exploitability inside it?

    box_bound(l_th) = min over y in [-l_th, l_th]^A of  exploitability(softmax(y))

That is a LOWER BOUND on any gated run's floor, computed without reference to
the dynamics -- so comparing it to the measured 0.0990 splits the floor into
"bias the box forces on any policy" and "everything else the operator does".
Softmax is shift-invariant (ach_theory_sympy V5), so optimizing y over the box
covers exactly the policies whose per-row legal-action logit separation is at
most ``2*l_th``, which is the set the theory names.

``exploitability`` here is ``nash_conv/2`` on the exact sequence form, which is
differentiable in the behaviour strategy (its best-response DP is a ``max``), so
this is plain projected gradient descent. Adam is used because this is an
OPTIMIZATION over the constraint set, not a run of the ACH algorithm -- no
fidelity question arises and no ``ACHFidelityWarning`` applies.

Sanity anchor: ``--l-th inf`` must recover ~0, i.e. the unconstrained optimum is
Nash. If it does not, the optimizer is the problem, not the box.

Run::

    uv run python -m tools.box_bound --l-th 1 2 3 4 inf

Not on the ``mjai`` import path (AGENTS.md sec.4). float64 (D19).
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


def behaviour(sf: SequenceForm, y: torch.Tensor) -> torch.Tensor:
    return torch.softmax(y.masked_fill(~sf.legal_mask, float("-inf")), dim=1)


def expl(sf: SequenceForm, y: torch.Tensor) -> torch.Tensor:
    return nash_conv(sf, behaviour(sf, y), validate=False) / 2.0


def project(y: torch.Tensor, l_th: float, sf: SequenceForm) -> torch.Tensor:
    """Center each row's legal logits on their midrange, then clamp into the box.

    The raw one-sided gate's invariant set is ``y_a in [-l_th, l_th]``, so the
    reachable policies are exactly those with per-row legal separation at most
    ``2*l_th``. Centering on the midrange first is the shift that maximizes the
    surviving width -- softmax is shift-invariant (ach_theory_sympy V5), so a row
    that is feasible in shape must not be clipped merely for being offset.

    There is deliberately NO centered-gate mode here. The obvious candidate,
    ``{y : |y_a - ybar| <= l_th}``, is a strictly SMALLER set than the raw box
    (n=3, logits (0, 0, -2*l_th) has separation 2*l_th yet deviation 4*l_th/3),
    while the centered gate's actual resting separation is n*l_th/min(k, n-k),
    which is LARGER (ach_theory_sympy V9). A hard symmetric box is therefore the
    wrong model of it in both directions, and the measured separation of a real
    centered run (tools/gate_box_probe.py) is the ground truth instead.
    """
    if not math.isfinite(l_th):
        return y
    big = torch.where(sf.legal_mask, y, torch.full_like(y, -float("inf")))
    small = torch.where(sf.legal_mask, y, torch.full_like(y, float("inf")))
    mid = (big.max(dim=1, keepdim=True).values + small.min(dim=1, keepdim=True).values) / 2.0
    mid = torch.nan_to_num(mid, nan=0.0, posinf=0.0, neginf=0.0)
    return (y - mid).clamp(-l_th, l_th)


def optimize(
    sf: SequenceForm,
    l_th: float,
    *,
    init: torch.Tensor | None = None,
    steps: int = 3000,
    lr: float = 0.05,
) -> dict[str, float]:
    y = torch.zeros(sf.num_infosets, sf.max_actions, dtype=torch.float64) if init is None else init
    y = project(y.clone(), l_th, sf).requires_grad_(True)
    opt = torch.optim.Adam([y], lr=lr)
    best = float(expl(sf, y).detach())
    for _ in range(steps):
        opt.zero_grad()
        loss = expl(sf, y)
        loss.backward()
        opt.step()
        with torch.no_grad():
            y.copy_(project(y, l_th, sf))
        cur = float(loss.detach())
        best = min(best, cur)
    final = float(expl(sf, y).detach())
    with torch.no_grad():
        b = behaviour(sf, y)
        logp = torch.where(sf.legal_mask, torch.log(b.clamp(min=1e-300)), torch.nan)
        hi = torch.nan_to_num(logp, nan=-float("inf")).max(dim=1).values
        lo = torch.nan_to_num(logp, nan=float("inf")).min(dim=1).values
        sep_max = float((hi - lo).max())
    return {"l_th": l_th, "best": best, "final": final, "sep_max": sep_max}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="liars_dice1")
    ap.add_argument("--l-th", nargs="+", default=["1", "2", "3", "4", "inf"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument(
        "--init-policy",
        type=Path,
        default=None,
        help="warm-start from a tools/exact_ach.py --save-policy bundle",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sf = build_sequence_form(load_game(args.game))
    init = None
    if args.init_policy is not None:
        b = torch.load(args.init_policy, weights_only=True)["behavior"].to(torch.float64)
        init = torch.log(b.clamp(min=1e-300)).masked_fill(~sf.legal_mask, 0.0)

    print(f"game={args.game}  steps={args.steps}  lr={args.lr}  init={args.init_policy or 'zeros'}")
    print(f"{'l_th':>6} | {'box bound':>10} | {'final':>10} | {'max sep':>8}")
    print("-" * 46)
    results = []
    for tok in args.l_th:
        l_th = float("inf") if tok == "inf" else float(tok)
        r = optimize(sf, l_th, init=init, steps=args.steps, lr=args.lr)
        results.append(r)
        print(f"{tok:>6} | {r['best']:10.6f} | {r['final']:10.6f} | {r['sep_max']:8.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"game": args.game, "results": results}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
