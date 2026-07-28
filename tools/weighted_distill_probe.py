"""Distil the same Nash into the same MLP, varying ONLY the per-information-set weight.

This is the controlled version of the question the RL ablations kept circling.
``docs/liars_machine_precision.md`` established that supervised distillation of a
CFR+ Nash into a 1024-wide MLP reaches exploitability 0.0033, while the RL run
that trains the *same architecture* on the *same game* floors at 0.146 -- a 45x
gap attributed to "the algorithm". This probe asks which part of the algorithm,
by holding everything else fixed:

  target      the same cached CFR+ Nash
  network     the same MLPSharedActorCritic
  optimiser   the same Adam -> L-BFGS schedule
  loss        the same cross-entropy

and varying only the weight each information set carries in that loss:

  ``uniform``   every information set counts once  (what distillation does)
  ``reach``     weighted by rho(I), the probability a self-play episode visits it
                (what an on-policy sampler delivers -- ACH's actual weighting)
  ``sqrt``      rho(I)**0.5, an intermediate point

If ``reach`` lands near the RL floor while ``uniform`` lands near 0.003, the
45x gap is the *weighting* and nothing else: not the critic, not the batch size,
not the capacity, not the entropy -- none of which this probe contains at all.

Analysis tool; writes JSON, changes nothing in ``src/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tools.exact_ach import ExactAdvantage

from mjai.agents.mlp import MLPSharedActorCritic
from mjai.games.loader import load_game
from mjai.seqform.plan import nash_conv, realization_plans
from mjai.seqform.tree import build_sequence_form

WEIGHTINGS = ("uniform", "reach", "sqrt", "cf")


def counterfactual_reach(sf, behavior: torch.Tensor) -> torch.Tensor:
    """Chance-and-opponent reach of each information set: CFR's regret weight.

    ``cf(I)`` is by definition independent of the owner's own strategy, so it is
    computed with the owner's rows replaced by uniform. That matters: dividing
    the visitation ``rho = own * cf`` by ``own`` would be 0/0 wherever the
    policy's own reach is exactly zero, which a CFR+ average policy routinely
    is. Under uniform own-play the divisor is a product of ``1/|legal|`` factors
    -- small but never zero.
    """
    engine = ExactAdvantage(sf)
    cf = torch.zeros(sf.num_infosets, dtype=torch.float64)
    for player in range(sf.num_players):
        rows = sf.rows_of(player)
        if rows.numel() == 0:
            continue
        probe = behavior.clone()
        mask = sf.legal_mask[rows]
        probe[rows] = mask.to(torch.float64) / mask.sum(dim=1, keepdim=True)
        _adv, rho_uniform_own = engine.compute(probe)
        plans = realization_plans(sf, probe)
        own = plans[player].index_select(0, sf.parent_sequence[rows])
        cf[rows] = rho_uniform_own[rows] / own.clamp(min=1e-300)
    return cf


def infoset_weights(sf, behavior: torch.Tensor, kind: str) -> torch.Tensor:
    """Per-information-set loss weight, normalized to mean 1.

    ``cf`` is the weight exploitability actually responds to (a best response
    collects an error at ``I`` in proportion to how often chance and the
    opponent take the game there). ``reach`` is what an on-policy sampler
    delivers, ``rho = own_reach * cf``: it discounts every information set by
    the learner's *own* probability of going there. The ratio between the two is
    therefore exactly ``1 / own_reach(I)`` -- the mismatch, in one expression.
    """
    if kind == "uniform":
        w = torch.ones(sf.num_infosets, dtype=torch.float64)
    elif kind == "cf":
        w = counterfactual_reach(sf, behavior)
    else:
        _adv, rho = ExactAdvantage(sf).compute(behavior)
        w = rho if kind == "reach" else rho.clamp(min=0.0).sqrt()
    return w / w.mean().clamp(min=1e-300)


def distill(
    spec,
    sf,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    width: int,
    epochs: int,
    lr: float,
    lbfgs_iters: int,
    seed: int,
) -> tuple[MLPSharedActorCritic, float]:
    obs = sf.infoset_observation.to(torch.float32)
    legal = sf.legal_mask
    tgt = target.to(torch.float32)
    w = weights.to(torch.float32)
    mlp = MLPSharedActorCritic(
        spec.obs_size, spec.num_actions, hidden_sizes=(width,), seed=seed, device="cpu"
    )

    def loss_fn() -> torch.Tensor:
        masked = mlp(obs)[0].masked_fill(~legal, -1e9)
        per_row = -(tgt * torch.log_softmax(masked, dim=-1)).sum(dim=-1)
        return (w * per_row).mean()

    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    for _ in range(epochs):
        loss = loss_fn()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    if lbfgs_iters:
        lopt = torch.optim.LBFGS(
            mlp.parameters(), lr=1.0, max_iter=25, line_search_fn="strong_wolfe"
        )
        for _ in range(lbfgs_iters):

            def closure() -> torch.Tensor:
                lopt.zero_grad()
                value = loss_fn()
                value.backward()
                return value

            lopt.step(closure)
    return mlp, float(loss_fn().detach())


def behavior_of_mlp(sf, mlp: MLPSharedActorCritic) -> torch.Tensor:
    with torch.no_grad():
        logits = mlp(sf.infoset_observation.to(torch.float32))[0].to(torch.float64)
    return torch.softmax(logits.masked_fill(~sf.legal_mask, float("-inf")), dim=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="liars_dice1")
    ap.add_argument("--nash", type=Path, default=Path("runs/nash_liars_dice1_behavior.pt"))
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--lbfgs-iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weightings", nargs="*", default=list(WEIGHTINGS))
    ap.add_argument("--out", type=Path, default=Path("runs/exact_ach/weighted_distill.json"))
    args = ap.parse_args()

    spec = load_game(args.game)
    sf = build_sequence_form(spec)
    target = torch.load(args.nash, weights_only=True).to(torch.float64)
    print(f"{args.game}: target Nash exploitability {float(nash_conv(sf, target)) / 2:.3e}")

    results: dict[str, dict[str, float]] = {}
    if args.out.is_file():
        results = json.loads(args.out.read_text(encoding="utf-8"))

    for kind in args.weightings:
        w = infoset_weights(sf, target, kind)
        covered = float((w > 1e-6).to(torch.float64).mean())
        mlp, final_loss = distill(
            spec,
            sf,
            target,
            w,
            width=args.width,
            epochs=args.epochs,
            lr=args.lr,
            lbfgs_iters=args.lbfgs_iters,
            seed=args.seed,
        )
        beh = behavior_of_mlp(sf, mlp)
        expl = float(nash_conv(sf, beh, validate=False)) / 2.0
        # True (unweighted) KL to the target, so the arms are comparable.
        kl = float(
            (target * (torch.log(target.clamp(min=1e-300)) - torch.log(beh.clamp(min=1e-300))))
            .masked_fill(~sf.legal_mask, 0.0)
            .sum(dim=1)
            .mean()
        )
        results[kind] = {
            "exploitability": expl,
            "unweighted_KL": kl,
            "weighted_train_loss": final_loss,
            "frac_infosets_with_weight_above_1e-6": covered,
        }
        print(
            f"{kind:>8}: expl {expl:.5f}  KL(Nash||MLP) {kl:.5f}  "
            f"covered {covered * 100:.1f}% of infosets",
            flush=True,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
