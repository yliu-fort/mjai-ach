"""Is the online ``reach^-kappa`` sample weight really ``rho^(1-kappa)`` per information set?

``docs/liars_residual_floor.md`` §8.4 proposes weighting each sampled
transition by the reach probability of its own history raised to ``-kappa``.
Every factor of that reach is known at rollout time (both players' recorded
action log-probs, and the chance probabilities the sampler itself drew from),
so unlike anything based on visit counts it does not degenerate as the game
grows. The claim was that this delivers an effective per-information-set weight
of ``rho(I)^(1-kappa)`` -- the tempering that reached exploitability 0.0092 in
the distillation control.

That claim is an approximation, and this module measures the error exactly. A
sample from information set ``I`` arrives with probability ``reach(h)`` for the
particular history ``h`` it came from, so the expected weight mass at ``I`` is

    W(I) = sum_{h in I} reach(h) * reach(h)^-kappa = sum_{h in I} reach(h)^(1-kappa)

whereas the target is ``rho(I)^(1-kappa) = (sum_{h in I} reach(h))^(1-kappa)``.
By the power-mean inequality ``W(I) >= rho(I)^(1-kappa)``, with equality only
when the information set holds a single history -- and the gap is bounded by
``|I|^kappa`` but is NOT constant across information sets, so it can distort the
ranking, which §7 showed is the axis that matters.

Computing ``W`` needs the per-history reaches, which the sequence form has
already aggregated away; this walks the real game tree instead. ``rho``
recomputed along the way is checked against :class:`ExactAdvantage`, which is
what makes the walk trustworthy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyspiel
import torch
from tools.exact_ach import ExactAdvantage

from mjai.games.loader import load_game
from mjai.seqform.tree import build_sequence_form


def per_history_weights(
    spec, sf, behavior: torch.Tensor, kappas: list[float]
) -> tuple[dict[float, torch.Tensor], torch.Tensor]:
    """Return ``({kappa: W}, rho)``, both indexed by sequence-form row."""
    if spec.is_simultaneous:
        raise ValueError(f"{spec.name} is simultaneous; this walk assumes turn-based play")
    index = {k: i for i, k in enumerate(sf.infoset_keys)}
    acc = {k: torch.zeros(sf.num_infosets, dtype=torch.float64) for k in kappas}
    rho = torch.zeros(sf.num_infosets, dtype=torch.float64)
    probs = behavior.tolist()

    def walk(state: pyspiel.State, chance: float, reaches: list[float]) -> None:
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, prob in state.chance_outcomes():
                walk(state.child(action), chance * prob, reaches)
            return
        player = state.current_player()
        row = index[state.information_state_string(player)]
        joint = chance
        for r in reaches:
            joint *= r
        rho[row] += joint
        for kappa in kappas:
            acc[kappa][row] += joint ** (1.0 - kappa)
        row_probs = probs[row]
        for action in state.legal_actions():
            p = row_probs[action]
            if p <= 0.0:
                continue  # an unreachable branch contributes nothing to either sum
            nxt = list(reaches)
            nxt[player] *= p
            walk(state.child(action), chance, nxt)

    walk(spec.game.new_initial_state(), 1.0, [1.0] * spec.num_players)
    return acc, rho


def effn(w: torch.Tensor) -> float:
    w = w / w.mean().clamp(min=1e-300)
    return float(w.sum() ** 2 / (w * w).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="liars_dice1")
    ap.add_argument("--nash", type=Path, default=Path("runs/nash_liars_dice1_behavior.pt"))
    ap.add_argument("--kappas", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--out", type=Path, default=Path("runs/exact_ach/history_weighting.json"))
    args = ap.parse_args()

    spec = load_game(args.game)
    sf = build_sequence_form(spec)
    target = torch.load(args.nash, weights_only=True).to(torch.float64)

    acc, rho_walk = per_history_weights(spec, sf, target, args.kappas)

    _adv, rho_ref = ExactAdvantage(sf).compute(target)
    reachable = rho_ref > 0
    err = ((rho_walk - rho_ref).abs() / rho_ref.clamp(min=1e-300))[reachable]
    print(f"walk validation: max relative error on rho = {float(err.max()):.3e}")

    results: dict[str, dict[str, float]] = {}
    print(
        f"\n{'kappa':>6} {'effN(W) online':>16} {'effN(rho^(1-k)) ideal':>23} "
        f"{'rank corr':>11} {'max W/ideal':>12}"
    )
    for kappa in args.kappas:
        w = acc[kappa]
        ideal = rho_ref.clamp(min=0.0).pow(1.0 - kappa)
        wn = w / w.mean()
        idn = ideal / ideal.mean()
        ratio = (wn / idn.clamp(min=1e-300))[reachable]
        order_w = torch.argsort(torch.argsort(w)).to(torch.float64)
        order_i = torch.argsort(torch.argsort(ideal)).to(torch.float64)
        corr = float(torch.corrcoef(torch.stack([order_w, order_i]))[0, 1])
        results[str(kappa)] = {
            "effN_online": effn(w),
            "effN_ideal": effn(ideal),
            "rank_spearman": corr,
            "max_ratio": float(ratio.max()),
            "median_ratio": float(ratio.median()),
        }
        print(
            f"{kappa:>6.2f} {effn(w):>16.1f} {effn(ideal):>23.1f} "
            f"{corr:>11.4f} {float(ratio.max()):>12.2f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    torch.save(acc, args.out.with_suffix(".pt"))
    print(f"\nwrote {args.out} (+ .pt with the weight vectors)")


if __name__ == "__main__":
    main()
