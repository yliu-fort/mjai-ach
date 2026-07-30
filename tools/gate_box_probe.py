"""Is the exact dynamics' floor sitting on the walls of the logit gate's box?

``docs/ach_operator_theory.md`` derives that the paper's one-sided logit gate
traps every persistently-signed logit in ``[-l_th, l_th]``, so a converged row
must rest at separation

    sep(s) = max_a y(s,a) - min_a y(s,a)  ->  2 * l_th

over its legal actions, and the probability mass that separation forces onto the
off-support actions is a floor no amount of training removes:

    leak(s) = (n_s - k_s) / (k_s * e^{2*l_th} + (n_s - k_s))

This probe measures the separation and the support size on a policy the dynamics
actually settled on, cross-tabulated by reach so the answer is comparable to the
reach-decile attribution in ``tools/floor_microscope.py``. Turning the box into
an exploitability number is a separate job and belongs to ``tools/box_bound.py``,
which optimizes inside the box directly rather than estimating from the leak.

Separation is recoverable from a saved *behaviour* bundle even though the raw
logits are not: within a row ``log pi_a - log pi_b == y_a - y_b`` exactly
(softmax is shift-invariant, see ach_theory_sympy V5). The absolute mean logit
``ybar`` is NOT recoverable and is not needed -- the box's width is what the
theory predicts, not its position.

N3 in the pre-registration (`docs/ach_operator_theory.md` sec.10): if the
high-reach rows are not saturated at ``2 * l_th``, the theory is dead.

Run::

    uv run python -m tools.gate_box_probe --policy runs/exact_ach/M_liars_rho0.5_policy.pt

Not on the ``mjai`` import path (AGENTS.md sec.4). float64 (D19).
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

# A row is "live" if a self-play episode reaches it with non-negligible
# probability; unreached rows never received a gradient, so their logits say
# nothing about the gate (same threshold family as tools/exact_ach.py).
_LIVE = 1e-12
# An action counts as on-support if it holds at least this share of the row's
# mass. The box prediction is that off-support actions sit e^{-2 l_th} ~ 0.018
# below the on-support ones, so any cut well inside that gap gives the same k.
_SUPPORT = 0.05


def row_stats(behavior: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-row legal-action separation, support size, and legal action count."""
    logp = torch.where(mask, torch.log(behavior.clamp(min=1e-300)), torch.tensor(float("nan")))
    hi = torch.nan_to_num(logp, nan=-float("inf")).max(dim=1).values
    lo = torch.nan_to_num(logp, nan=float("inf")).min(dim=1).values
    n = mask.sum(dim=1)
    k = ((behavior >= _SUPPORT) & mask).sum(dim=1)
    return {"sep": hi - lo, "n": n, "k": k}


def predicted_leak(n: torch.Tensor, k: torch.Tensor, l_th: float) -> torch.Tensor:
    kk = k.clamp(min=1).to(torch.float64)
    nn = n.to(torch.float64)
    return (nn - kk) / (kk * torch.exp(torch.tensor(2.0 * l_th, dtype=torch.float64)) + (nn - kk))


def expl(sf: SequenceForm, behavior: torch.Tensor) -> float:
    return float(nash_conv(sf, behavior, validate=False)) / 2.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="liars_dice1")
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--l-th", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sf = build_sequence_form(load_game(args.game))
    behavior = torch.load(args.policy, weights_only=True)["behavior"].to(torch.float64)
    engine = ExactAdvantage(sf)
    _, rho_row = engine.compute(behavior)  # visitation is per-row, action-independent
    live = rho_row > _LIVE

    st = row_stats(behavior, sf.legal_mask)
    sep, n, k = st["sep"], st["n"], st["k"]
    target = 2.0 * args.l_th

    idx = torch.nonzero(live, as_tuple=False).flatten()
    order = idx[torch.argsort(rho_row[idx])]
    bins = list(torch.chunk(order, 10))

    print(f"game={args.game}  policy={args.policy}  l_th={args.l_th}  target sep={target}")
    print(f"live rows: {int(live.sum())} / {sf.num_infosets}   expl={expl(sf, behavior):.6f}")
    print()
    print(
        f"{'rho bin':>7} | {'rho med':>10} | {'sep med':>8} {'sep p10':>8} {'sep p90':>8}"
        f" | {'frac|sep-4|<.1':>14} | {'n med':>5} {'k med':>5} | {'leak_pred':>9}"
    )
    print("-" * 104)
    rows = []
    for b, chunk in enumerate(bins):
        s = sep[chunk]
        near = float(((s - target).abs() < 0.1).to(torch.float64).mean())
        rec = {
            "bin": b,
            "rho_med": float(rho_row[chunk].median()),
            "sep_med": float(s.median()),
            "sep_p10": float(s.quantile(0.10)),
            "sep_p90": float(s.quantile(0.90)),
            "frac_at_wall": near,
            "n_med": float(n[chunk].to(torch.float64).median()),
            "k_med": float(k[chunk].to(torch.float64).median()),
            "leak_pred": float(predicted_leak(n[chunk], k[chunk], args.l_th).mean()),
        }
        rows.append(rec)
        print(
            f"{b:>7} | {rec['rho_med']:10.3e} | {rec['sep_med']:8.4f} {rec['sep_p10']:8.4f}"
            f" {rec['sep_p90']:8.4f} | {rec['frac_at_wall']:14.3f}"
            f" | {rec['n_med']:5.1f} {rec['k_med']:5.1f} | {rec['leak_pred']:9.5f}"
        )

    # rho-weighted leak: the theory's floor estimate before any tree correction
    leak = predicted_leak(n, k, args.l_th)
    w = rho_row * live.to(torch.float64)
    weighted_leak = float((w * leak).sum() / w.sum())
    print()
    print(f"rho-weighted mean predicted leak = {weighted_leak:.6f}")
    print(f"rho-weighted mean separation     = {float((w * sep).sum() / w.sum()):.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "game": args.game,
                    "l_th": args.l_th,
                    "expl": expl(sf, behavior),
                    "bins": rows,
                    "weighted_leak": weighted_leak,
                },
                indent=2,
            )
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
