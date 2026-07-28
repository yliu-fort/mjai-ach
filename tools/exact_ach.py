"""Exact (noiseless) ACH dynamics on the sequence-form tree — a research probe.

The question this answers: **what does ACH converge to when nothing else can be
blamed?** The RL pipeline's floor has already survived every ablation of its
*estimation* machinery — critic quality (4 ways), batch size (2 protocols),
network capacity, entropy coefficient (``docs/liars_floor_ablation.md``,
``docs/liars_machine_precision.md``). What has never been tested is the update
rule in isolation, because every test so far ran it through sampled rollouts, a
learned critic, and an MLP.

This module removes all three at once. It runs the *expected* ACH update

    y <- y + lr * rho * ( eta * c * A  -  beta * pi * (log pi + H) )

on a float64 logit table over the game's information sets, where

  - ``A`` is the **exact** on-policy advantage from a full-tree traversal (an
    infinitely-well-trained critic and an infinite batch),
  - ``rho`` is the **exact** on-policy visitation probability of the information
    set (what sampling would deliver in expectation),
  - ``c`` is the paper's advantage-sign-dependent one-sided logit gate,
  - the logit table is tabular, so there is no function approximation.

Why that expression is the ACH update in expectation: the per-sample loss is
``-c * eta * y(a) / pi_old(a) * A(a)`` (Eq. 29 p24), whose gradient w.r.t. the
logit ``y_j`` is ``-c * eta * A_j / pi_j`` on the sampled action only. Taking
the expectation over ``a ~ pi`` cancels the ``1/pi_old`` exactly, leaving
``-eta * c_j * A_j``. The entropy term ``beta * sum_b pi_b log pi_b``
contributes ``+beta * pi_j * (log pi_j + H)``. The per-information-set weight is
the probability a rollout visits it, which is what makes this the *sampled*
algorithm's expectation rather than a uniform-sweep variant of it.

So a floor that survives here is a property of the ACH **fixed point**, not of
any estimator feeding it. And each term can be switched off independently
(``--beta 0``, ``--no-gate``, ``--weighting uniform``) to see which one owns it.

Not a training pipeline and not on the ``mjai`` import path: this is an analysis
tool (AGENTS.md §4 "add a metric" does not apply — nothing here runs in a run
directory). All arithmetic is float64 (AGENTS.md D19).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from mjai.games.loader import load_game
from mjai.seqform.plan import nash_conv, realization_plans
from mjai.seqform.tree import SequenceForm, build_sequence_form

# Information sets whose on-policy visitation falls below this are treated as
# unvisited when computing an advantage: the exact advantage is a 0/0 there and
# a sampled run would never produce a gradient for them either.
_UNREACHED = 1e-300


@dataclass(frozen=True)
class AchParams:
    """The ACH hyperparameters this probe varies (paper values as defaults)."""

    eta: float = 1.0  # hedge coefficient (p27 Table 7)
    beta: float = 1e-2  # entropy coefficient (p28 Table 8)
    l_th: float = 2.0  # one-sided logit gate (p28 Table 8)
    lr: float = 1e-3  # SGD, constant (p27)
    gate: bool = True
    gate_centered: bool = False  # repo default: raw logits (LayerNorm arm)
    weighting: str = "reach"  # "reach" = on-policy visitation; "uniform" = flat


class ExactAdvantage:
    """Exact on-policy advantages and visitation weights for a behaviour strategy.

    Both come out of autograd on the exact multilinear payoff, which is what
    makes them exact rather than estimated::

        g[I, a]   = d E[u_p] / d behaviour[I, a] = own_reach(I) * CFV(I, a)
        rho[I, a] = the same derivative with every terminal utility replaced by
                    1, which is own_reach(I) * cfreach(I) -- the probability a
                    self-play episode visits I

    ``own_reach`` cancels in the advantage, so it is never divided out::

        A(I, a) = ( g[I, a] - sum_b pi_b g[I, b] ) / rho[I]

    ``rho`` is constant across the legal actions of a row by construction (the
    chance and opponent probabilities below a row sum to 1 whatever the owner
    does); :meth:`compute` asserts that, which is a strong end-to-end check on
    the whole derivative path.
    """

    def __init__(self, sf: SequenceForm) -> None:
        self.sf = sf
        # A copy of the tree whose payoffs are identically 1 for the row owner:
        # its derivative is the visitation probability. Built once.
        self._ones_utility = torch.ones_like(sf.terminal_utility)

    def _reach(self, behavior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The per-terminal reach probability, and the leaf it differentiates to.

        One forward pass shared by every vector-Jacobian product below: each
        quantity we want is ``d(w . reach)/d behaviour`` for a different weight
        vector ``w``, so the tree is walked once and back-propagated three times
        instead of being rebuilt per quantity.
        """
        b = behavior.detach().clone().requires_grad_(True)
        plans = realization_plans(self.sf, b)
        reach = self.sf.terminal_chance.clone()
        for p in range(self.sf.num_players):
            reach = reach * plans[p].index_select(0, self.sf.terminal_sequence[:, p])
        return reach, b

    def compute(self, behavior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(advantages [I, A], visitation [I])`` for ``behavior``."""
        sf = self.sf
        adv = torch.zeros_like(behavior)
        rho = torch.zeros(sf.num_infosets, dtype=torch.float64)
        reach, leaf = self._reach(behavior)
        # Visitation: weight every terminal by 1. Owner-independent, so one vjp
        # serves all players.
        (g_visits,) = torch.autograd.grad(
            reach, leaf, grad_outputs=torch.ones_like(reach), retain_graph=True
        )
        for player in range(sf.num_players):
            rows = sf.rows_of(player)
            if rows.numel() == 0:
                continue
            (g_full,) = torch.autograd.grad(
                reach,
                leaf,
                grad_outputs=sf.terminal_utility[:, player],
                retain_graph=player < sf.num_players - 1,
            )
            g = g_full[rows]
            r = g_visits[rows]
            mask = sf.legal_mask[rows]
            # rho is action-independent on legal slots; take the max and check.
            r_legal = torch.where(mask, r, torch.zeros_like(r))
            rho_rows = r_legal.max(dim=1).values
            spread = (r_legal - rho_rows.unsqueeze(1)).abs().masked_fill(~mask, 0.0)
            worst = float(spread.max()) if spread.numel() else 0.0
            if worst > 1e-9:
                raise AssertionError(
                    f"visitation is not action-independent (max spread {worst:.3e}); "
                    "the derivative path is wrong"
                )
            baseline = (behavior[rows] * g).sum(dim=1, keepdim=True)
            safe = rho_rows.clamp(min=_UNREACHED).unsqueeze(1)
            adv[rows] = torch.where(mask, (g - baseline) / safe, torch.zeros_like(g))
            rho[rows] = rho_rows
        return adv, rho


def masked_softmax(sf: SequenceForm, logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.masked_fill(~sf.legal_mask, float("-inf")), dim=1)


def ach_step(
    sf: SequenceForm,
    logits: torch.Tensor,
    engine: ExactAdvantage,
    params: AchParams,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One exact ACH update on the logit table. Returns (new logits, telemetry)."""
    pi = masked_softmax(sf, logits)
    adv, rho = engine.compute(pi)

    # Entropy gradient: beta * pi * (log pi + H), zero on illegal slots.
    log_pi = torch.log(pi.clamp(min=1e-300))
    entropy = -(pi * log_pi).masked_fill(~sf.legal_mask, 0.0).sum(dim=1, keepdim=True)
    ent_grad = params.beta * pi * (log_pi + entropy)
    ent_grad = ent_grad.masked_fill(~sf.legal_mask, 0.0)

    # Advantage-sign-dependent one-sided gate (p24 Algorithm 2).
    if params.gate:
        y_gate = logits - logits.mean(dim=1, keepdim=True) if params.gate_centered else logits
        gate = torch.where(adv >= 0, y_gate < params.l_th, y_gate > -params.l_th)
        c = gate.to(torch.float64)
    else:
        c = torch.ones_like(adv)

    weight = rho if params.weighting == "reach" else sf.legal_mask.any(dim=1).to(torch.float64)
    delta = weight.unsqueeze(1) * (params.eta * c * adv - ent_grad)
    new_logits = logits + params.lr * delta
    new_logits = new_logits.masked_fill(~sf.legal_mask, 0.0)

    with torch.no_grad():
        legal = sf.legal_mask
        gate_off = 1.0 - float(c.masked_select(legal).mean())
        mean_entropy = float(entropy.mean())
    return new_logits, {"gate_off_frac": gate_off, "entropy": mean_entropy}


@dataclass
class RunResult:
    game: str
    params: dict[str, float | str | bool]
    iters: int
    curve: list[tuple[int, float]] = field(default_factory=list)
    final_exploitability: float = math.nan
    best_exploitability: float = math.nan
    telemetry: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0


def run(
    game: str,
    params: AchParams,
    *,
    iters: int,
    eval_every: int,
    verbose: bool = True,
) -> RunResult:
    spec = load_game(game)
    sf = build_sequence_form(spec)
    engine = ExactAdvantage(sf)
    logits = torch.zeros(sf.num_infosets, sf.max_actions, dtype=torch.float64)
    divisor = float(spec.num_players) if spec.num_players == 2 and spec.is_zero_sum else 1.0

    result = RunResult(
        game=game,
        params={
            "eta": params.eta,
            "beta": params.beta,
            "l_th": params.l_th if params.gate else float("inf"),
            "lr": params.lr,
            "gate": params.gate,
            "gate_centered": params.gate_centered,
            "weighting": params.weighting,
        },
        iters=iters,
    )
    start = time.time()
    telemetry: dict[str, float] = {}
    best = math.inf
    for it in range(iters + 1):
        if it % eval_every == 0:
            pi = masked_softmax(sf, logits)
            expl = float(nash_conv(sf, pi, validate=False)) / divisor
            best = min(best, expl)
            result.curve.append((it, expl))
            if verbose:
                extra = " ".join(f"{k}={v:.3f}" for k, v in telemetry.items())
                print(f"  iter {it:>8} expl {expl:.6f}  {extra}", flush=True)
        if it == iters:
            break
        logits, telemetry = ach_step(sf, logits, engine, params)
    result.final_exploitability = result.curve[-1][1]
    result.best_exploitability = best
    result.telemetry = telemetry
    result.seconds = time.time() - start
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="kuhn")
    ap.add_argument("--iters", type=int, default=156_000, help="1e7 env-steps / batch 64")
    ap.add_argument("--eval-every", type=int, default=5_000)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1e-2)
    ap.add_argument("--l-th", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--gate-centered", action="store_true")
    ap.add_argument("--weighting", choices=("reach", "uniform"), default="reach")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    params = AchParams(
        eta=args.eta,
        beta=args.beta,
        l_th=args.l_th,
        lr=args.lr,
        gate=not args.no_gate,
        gate_centered=args.gate_centered,
        weighting=args.weighting,
    )
    print(f"exact ACH on {args.game}: {params}")
    res = run(args.game, params, iters=args.iters, eval_every=args.eval_every)
    print(
        f"final expl {res.final_exploitability:.6f}  best {res.best_exploitability:.6f}"
        f"  ({res.seconds:.1f}s)"
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res.__dict__, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
