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
# plus the parametric forms "rho:K", "cf:K" and "own:K" (= own_reach^K * cf).


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


def own_reach(sf, behavior: torch.Tensor) -> torch.Tensor:
    """The acting player's OWN realization probability of reaching each row.

    Online-computable during a rollout at any game scale: it is the running
    product of that player's own recorded action probabilities so far in the
    episode. That is what makes the ``own:K`` family below implementable
    outside a toy game, unlike anything that needs visit counts.
    """
    plans = realization_plans(sf, behavior)
    own = torch.zeros(sf.num_infosets, dtype=torch.float64)
    for player in range(sf.num_players):
        rows = sf.rows_of(player)
        own[rows] = plans[player].index_select(0, sf.parent_sequence[rows])
    return own


def infoset_weights(sf, behavior: torch.Tensor, kind: str) -> torch.Tensor:
    """Per-information-set loss weight, normalized to mean 1.

    Two independent axes, which the ``cf`` result forced apart:

    - **Which information sets get the weight (the ordering).** ``rho`` is the
      on-policy visitation ``own_reach * cf``, and it is also -- to first order
      -- the sensitivity of exploitability to a perturbation at ``I``: behaviour
      at an information set the player's own strategy never reaches cannot be
      exploited (``docs/kuhn_free_parameter.md`` §1.3 measures exactly that).
      ``cf`` drops the ``own_reach`` factor, so it re-orders.
    - **How sharply the weight is concentrated (the exponent).** ``rho`` spans
      18 orders of magnitude on Liar's Dice; raising it to a power ``kappa < 1``
      keeps the ordering and compresses the range.

    Accepted names: ``uniform`` (= rho^0), ``reach`` (= rho^1), ``sqrt``
    (= rho^0.5), ``cf``, and the parametric forms ``rho:K`` / ``cf:K`` for any
    exponent K, which is what separates the two axes.
    """
    base, _, exponent = kind.partition(":")
    kappa = float(exponent) if exponent else None
    if base == "uniform":
        return torch.ones(sf.num_infosets, dtype=torch.float64)
    if base == "own":
        # w = own_reach^K * cf. K=1 is rho (what sampling delivers), K=0 is cf
        # (measured worse than uniform, §7) -- the interior is the part that is
        # both untested AND computable online without visit counts.
        assert kappa is not None, "the 'own' family needs an exponent, e.g. own:0.5"
        w = own_reach(sf, behavior).clamp(min=0.0).pow(kappa) * counterfactual_reach(sf, behavior)
        return w / w.mean().clamp(min=1e-300)
    if base == "cf":
        w = counterfactual_reach(sf, behavior)
        kappa = 1.0 if kappa is None else kappa
    else:  # rho and its aliases
        _adv, w = ExactAdvantage(sf).compute(behavior)
        if kappa is None:
            kappa = {"reach": 1.0, "rho": 1.0, "sqrt": 0.5}[base]
    w = w.clamp(min=0.0).pow(kappa)
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
    sampled_target: bool = False,
    minibatch: int | None = None,
    row_sampler: torch.Tensor | None = None,
) -> tuple[MLPSharedActorCritic, float]:
    obs = sf.infoset_observation.to(torch.float32)
    legal = sf.legal_mask
    tgt = target.to(torch.float32)
    w = weights.to(torch.float32)
    mlp = MLPSharedActorCritic(
        spec.obs_size, spec.num_actions, hidden_sizes=(width,), seed=seed, device="cpu"
    )
    gen = torch.Generator().manual_seed(seed)
    sampler = None if row_sampler is None else row_sampler.to(torch.float32)

    def loss_fn() -> torch.Tensor:
        rows = (
            None
            if minibatch is None or sampler is None
            else torch.multinomial(sampler, minibatch, replacement=True, generator=gen)
        )
        o, lg, tg, ww = (
            (obs, legal, tgt, w) if rows is None else (obs[rows], legal[rows], tgt[rows], w[rows])
        )
        masked = mlp(o)[0].masked_fill(~lg, -1e9)
        logp = torch.log_softmax(masked, dim=-1)
        if sampled_target:
            # ONE action drawn from the target per row, redrawn every step: the
            # same expected gradient, plus the zero-mean per-row noise an RL
            # sample carries and this probe otherwise does not. See `main`.
            drawn = torch.multinomial(tg, num_samples=1, generator=gen)
            per_row = -logp.gather(1, drawn).squeeze(1)
        else:
            per_row = -(tg * logp).sum(dim=-1)
        if rows is None:
            return (ww * per_row).mean()  # full batch: unchanged, w has mean 1
        # Minibatch: the drawn subset's weights do not average to 1, so the
        # reduction must self-normalize or the step size would vary with the
        # draw. Same choice the RL path makes (nn_losses.weighted_mean).
        return (ww * per_row).sum() / ww.sum()

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


def minibatch_arm(
    sf, behavior: torch.Tensor, kind: str, rho: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(per_sample_weight, row_sampler)`` for one minibatch arm.

    A full-batch arm applies ``w(I)`` with every row present. Drawing rows from
    ``rho`` instead already applies ``rho(I)``, so reproducing the same expected
    objective needs the per-sample weight ``w(I) / rho(I)``. That is what makes
    the two modes comparable, and it is also what turns the two arms RL can
    actually run into their familiar forms: ``reach`` (``w = rho``) becomes a
    flat weight of 1, and ``sqrt`` (``w = rho^0.5``) becomes ``rho^-0.5``.

    ``uniform_rows`` is the exception and the point of contrast: it draws rows
    uniformly at a flat weight, which reproduces the full-batch ``uniform`` arm
    WITHOUT paying a ``1/rho`` weight for it. No on-policy sampler can do that
    -- it is the coverage ceiling, included to bound what the other two are
    being compared against.
    """
    if kind == "uniform_rows":
        ones = torch.ones_like(rho)
        return ones, ones
    w = infoset_weights(sf, behavior, kind) / rho.clamp(min=1e-300)
    return w / w.mean().clamp(min=1e-300), rho


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
    ap.add_argument(
        "--sampled-target",
        action="store_true",
        help=(
            "replace each row's exact target distribution with ONE action drawn "
            "from it, redrawn every step. Same expected gradient, plus the "
            "estimation noise an RL sample carries. This is the disanalogy "
            "between this probe and the RL runs it was used to predict: with an "
            "exact target, up-weighting a rare information set delivers pure "
            "signal; with a sampled one it also multiplies that row's noise. "
            "Forces Adam-only (L-BFGS's line search needs a stationary loss)."
        ),
    )
    ap.add_argument(
        "--minibatch",
        type=int,
        default=None,
        help=(
            "draw N rows per step instead of using all of them. This is the last "
            "structural difference between the probe and RL: an on-policy batch "
            "contains only rows the policy actually reached, so a weight can only "
            "redistribute emphasis among THOSE -- it can never deliver a gradient "
            "to a row that was not sampled. Rows are drawn from rho (what a "
            "rollout delivers) and each arm's per-sample weight is w(I)/rho(I), "
            "so the EXPECTED weighted objective is identical to the same arm at "
            "full batch: 'reach' becomes weight 1 (what the paper does) and "
            "'sqrt' becomes rho^-0.5 (what the RL arm does). The special arm "
            "'uniform_rows' draws rows UNIFORMLY at weight 1 -- the coverage "
            "ceiling, which no online sampler can implement."
        ),
    )
    ap.add_argument("--out", type=Path, default=Path("runs/exact_ach/weighted_distill.json"))
    args = ap.parse_args()

    spec = load_game(args.game)
    sf = build_sequence_form(spec)
    target = torch.load(args.nash, weights_only=True).to(torch.float64)
    print(f"{args.game}: target Nash exploitability {float(nash_conv(sf, target)) / 2:.3e}")

    results: dict[str, dict[str, float]] = {}
    if args.out.is_file():
        results = json.loads(args.out.read_text(encoding="utf-8"))

    rho_rows = ExactAdvantage(sf).compute(target)[1].clamp(min=0.0) if args.minibatch else None

    for kind in args.weightings:
        sampler = None
        if args.minibatch:
            w, sampler = minibatch_arm(sf, target, kind, rho_rows)
            # Diagnostics describe the EFFECTIVE weighting (sampler x per-sample
            # weight), which is the quantity comparable across the two modes --
            # not `w`, which in minibatch mode is flat for the paper's own arm.
            w_eff = sampler * w
            w_eff = w_eff / w_eff.mean().clamp(min=1e-300)
        else:
            w = infoset_weights(sf, target, kind)
            w_eff = w
        covered = float((w_eff > 1e-6).to(torch.float64).mean())
        mlp, final_loss = distill(
            spec,
            sf,
            target,
            w,
            width=args.width,
            epochs=args.epochs,
            lr=args.lr,
            lbfgs_iters=0 if (args.sampled_target or args.minibatch) else args.lbfgs_iters,
            seed=args.seed,
            sampled_target=args.sampled_target,
            minibatch=args.minibatch,
            row_sampler=sampler,
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
        # Distribution shift: a weight computed from the TARGET is only the right
        # sensitivity weight while the learner still plays like the target. Ask
        # how much of the learned policy's own visitation lands in the rows this
        # arm barely trained -- errors there are what move the weight itself.
        engine = ExactAdvantage(sf)
        _adv, rho_learned = engine.compute(beh)
        _adv, rho_target = engine.compute(target)
        starved = w_eff < 1e-3
        leak_learned = float(rho_learned[starved].sum() / rho_learned.sum().clamp(min=1e-300))
        leak_target = float(rho_target[starved].sum() / rho_target.sum().clamp(min=1e-300))
        results[kind] = {
            "exploitability": expl,
            "unweighted_KL": kl,
            "weighted_train_loss": final_loss,
            "frac_infosets_with_weight_above_1e-6": covered,
            "effective_infosets": float(w_eff.sum() ** 2 / (w_eff * w_eff).sum()),
            "starved_rows": int(starved.sum()),
            "visits_into_starved_rows_learned": leak_learned,
            "visits_into_starved_rows_target": leak_target,
        }
        print(
            f"{kind:>10}: expl {expl:.5f}  KL {kl:.5f}  covered {covered * 100:5.1f}%  "
            f"effN {results[kind]['effective_infosets']:8.0f}  "
            f"leak {leak_learned:.2e} (target {leak_target:.2e})",
            flush=True,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
