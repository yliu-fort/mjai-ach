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
from tools.ach_weighting import WeightRange, own_reach

from mjai.eval.average_policy import RealizationAverage
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
    # "reach" = on-policy visitation rho (raw); "uniform" = flat 1.0 (raw);
    # "rho:K" = rho^K RENORMALIZED to mean 1 over reachable rows;
    # "frozen:K" = the same rho^K shape but computed ONCE from the initial policy
    # and held constant for every iteration.
    #
    # The renormalization is the point of the third form and it also fixes a
    # confound in the first two. `reach` applies a raw rho whose mean is ~1e-4
    # while `uniform` applies a raw 1.0, so those two arms differ by ~4 orders of
    # magnitude in AVERAGE step size as well as in shape -- part of what looks
    # like a weighting effect there is an effective-learning-rate effect (the
    # lr x100 arm probed only 2 of those decades). The `rho:K` family holds the
    # mean step size fixed at every K, so K moves the SHAPE and nothing else.
    #
    # `frozen:K` exists to separate the two readings of Theorem 1's second term
    # `Delta * sum_s (w_h(s) - w_l(s)) / w_h(s)`. The bound's weights are indexed
    # BY INFORMATION SET and bracketed OVER ITERATIONS -- the ICLR slide (p8)
    # states `w_t(s) = f_p^{mu_t}(s) in [w_l(s), w_h(s)]` for `t = 1..T` -- so the
    # span the term charges for is TEMPORAL (how much one row's own reach drifts
    # while the policy trains), not the spread of rho ACROSS rows. Freezing the
    # weights sets `w_h(s) = w_l(s)` exactly, zeroing the term, while leaving the
    # cross-row dynamic range untouched. Whichever reading is right, this arm
    # separates them in one run.
    weighting: str = "reach"
    # Cap on the renormalized weight, mirroring RolloutConfig.sample_weight_clip.
    weight_clip: float | None = None
    # `lr_t = lr / (1 + t / lr_decay_tau)`. Theorem 1's second term is a property
    # of the weights and is step-size independent; a discretization limit cycle's
    # amplitude is not. Decaying the step size therefore tells the two apart --
    # and 1/t keeps the total displacement divergent, so the iterate can still
    # travel arbitrarily far rather than freezing in place.
    lr_decay_tau: float | None = None


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


def step_weight(sf: SequenceForm, rho: torch.Tensor, params: AchParams) -> torch.Tensor:
    """Per-information-set step weight for one exact ACH update.

    ``reach`` and ``uniform`` are the raw historical arms, kept bit-identical.
    ``rho:K`` is the tempered family: ``rho^K`` renormalized to mean 1 over the
    rows a rollout can actually reach, so every K spends the same average step
    size and only the distribution of it across information sets changes. That
    is the isolation the RL runs could not do -- there the tempering also had a
    critic, a moving target and a 64-row batch riding on it.
    """
    legal_rows = sf.legal_mask.any(dim=1).to(torch.float64)
    if params.weighting == "reach":
        return rho
    if params.weighting == "uniform":
        return legal_rows
    base, _, exponent = params.weighting.partition(":")
    if base in {"mix", "mu"}:
        # The caller already evaluated rho under the mixed behaviour policy; the
        # exponent is the mixing coefficient, not a power. Normalize to mean 1 for
        # the same reason `rho:K` does -- raw reach has mean ~1e-4 on liars, so
        # comparing raw arms compares average step size as much as weight shape
        # (the confound §8.9 of docs/liars_residual_floor.md had to unwind).
        reach = rho > 0
        return (rho / rho[reach].mean().clamp(min=1e-300)) * legal_rows
    if base not in {"rho", "frozen"} or not exponent:
        raise ValueError(
            f"unknown weighting {params.weighting!r}; "
            "want reach | uniform | rho:K | frozen:K | mix:X"
        )
    w = rho.clamp(min=0.0).pow(float(exponent))
    if params.weight_clip is not None:
        w = w.clamp(max=params.weight_clip * w[rho > 0].min())
    reachable = rho > 0
    return (w / w[reachable].mean().clamp(min=1e-300)) * legal_rows


def ach_step(
    sf: SequenceForm,
    logits: torch.Tensor,
    engine: ExactAdvantage,
    params: AchParams,
    *,
    fixed_weight: torch.Tensor | None = None,
    lr: float | None = None,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    """One exact ACH update on the logit table.

    Returns ``(new logits, telemetry, weights)``, where ``weights`` carries both
    the weight this step applied and the weight Theorem 1 is written in, so the
    caller can accumulate each one's per-information-set range over iterations.
    """
    pi = masked_softmax(sf, logits)
    adv, rho = engine.compute(pi)

    # Appendix D's knob. `mu_t = x*Uniform + (1-x)*pi_t` is the behaviour policy the
    # weight is read off; the advantage stays on the current policy, because in
    # weighted CFR `mu_t` enters only through `w_t(s) = f_p^{mu_t}(s)` and the tree
    # is swept in full either way. At x = 1 the behaviour is stationary, so by
    # Corollary 1 (p23) the weight is constant in t and Theorem 1's second term
    # vanishes -- the paper's `Uniform` arm.
    #
    # Two forms, because the theorem's weight is the OWN-reach factor alone while
    # what an update multiplies an advantage by is the full product:
    #   `mu:x`  moves only the own-reach factor -- faithful to `w_t(s)`;
    #   `mix:x` moves the whole product -- a strictly stronger intervention.
    kind, _, arg = params.weighting.partition(":")
    behaviour_mixed = kind in {"mu", "mix"}
    if behaviour_mixed:
        legal = sf.legal_mask
        unif = legal.to(torch.float64) / legal.sum(dim=1, keepdim=True)
        mu = float(arg) * unif + (1.0 - float(arg)) * pi
        if kind == "mu":
            rho = own_reach(sf, mu) * (rho / own_reach(sf, pi).clamp(min=1e-300))
        else:
            _, rho = engine.compute(mu)
    else:
        mu = pi

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

    weight = step_weight(sf, rho, params) if fixed_weight is None else fixed_weight
    delta = weight.unsqueeze(1) * (params.eta * c * adv - ent_grad)
    new_logits = logits + (params.lr if lr is None else lr) * delta
    new_logits = new_logits.masked_fill(~sf.legal_mask, 0.0)

    with torch.no_grad():
        legal = sf.legal_mask
        gate_off = 1.0 - float(c.masked_select(legal).mean())
        mean_entropy = float(entropy.mean())
        # `applied` governs this operator; `w_theorem` is the quantity Theorem 1
        # brackets. They differ whenever an arm intervenes on the weight directly
        # (`rho:K`, `frozen:K`) or moves only one factor of it (`mu:x`), so both are
        # tracked rather than one standing in for the other.
        weights = {"applied": weight, "w_theorem": own_reach(sf, mu)}
    return new_logits, {"gate_off_frac": gate_off, "entropy": mean_entropy}, weights


def frozen_weight(
    sf: SequenceForm,
    engine: ExactAdvantage,
    logits: torch.Tensor,
    params: AchParams,
    frozen_ref: Path | None,
) -> torch.Tensor | None:
    """The constant weight vector for a ``frozen:K`` arm, or ``None`` for others.

    ``frozen_ref`` is what makes the comparison against ``rho:K`` clean. Freezing at
    the INITIAL policy changes two things at once -- the weights stop moving, and
    their shape becomes the one a uniform policy induces rather than the one the
    converged policy does. Pointing this at a ``rho:K`` run's own final behaviour
    holds the shape fixed and leaves "stops moving" as the only difference.
    """
    if not params.weighting.startswith("frozen:"):
        return None
    reference = masked_softmax(sf, logits)
    if frozen_ref is not None:
        reference = torch.load(frozen_ref, weights_only=True)["behavior"].to(torch.float64)
    _, rho0 = engine.compute(reference)
    return step_weight(sf, rho0, params)


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
    # ``--track-average``: (iter, uniform-average expl, linear-average expl).
    # The CURRENT policy is what ``curve`` holds and what the paper plots; the
    # AVERAGE is what Theorem 1's O(T^-1/2) bound is actually about (AGENTS.md
    # D16), so a current-iterate limit cycle says nothing about the theorem
    # until this column exists.
    avg_curve: list[tuple[int, float, float]] = field(default_factory=list)
    # Theorem 1's second term and the per-row temporal weight range behind it.
    # Per-weight-notion temporal range: "applied" (what this operator used) and
    # "w_theorem" (what Theorem 1 brackets). See tools/ach_weighting.py.
    weight_range: dict[str, dict[str, float]] = field(default_factory=dict)


def run(
    game: str,
    params: AchParams,
    *,
    iters: int,
    eval_every: int,
    verbose: bool = True,
    track_average: bool = False,
    save_policy: Path | None = None,
    frozen_ref: Path | None = None,
) -> RunResult:
    spec = load_game(game)
    sf = build_sequence_form(spec)
    engine = ExactAdvantage(sf)
    logits = torch.zeros(sf.num_infosets, sf.max_actions, dtype=torch.float64)
    divisor = float(spec.num_players) if spec.num_players == 2 and spec.is_zero_sum else 1.0
    # Averaging in REALIZATION-PLAN space, not behaviour space -- averaging
    # behaviour probabilities is not the average strategy and does not obey the
    # bound. Every iterate is folded in, not just the eval points: the average
    # of 20 checkpoints is a different object from the average of 200k iterates.
    trackers: dict[str, RealizationAverage] = (
        {"uniform": RealizationAverage(sf), "linear": RealizationAverage(sf)}
        if track_average
        else {}
    )
    fixed_weight = frozen_weight(sf, engine, logits, params, frozen_ref)
    delta_payoff = float(sf.terminal_utility.max() - sf.terminal_utility.min())
    ranges = {name: WeightRange(sf.num_infosets, delta_payoff) for name in ("applied", "w_theorem")}

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
            "frozen_ref": str(frozen_ref) if frozen_ref else "",
            "lr_decay_tau": params.lr_decay_tau if params.lr_decay_tau else 0.0,
            "delta_payoff": delta_payoff,
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
            avg = ""
            if trackers and trackers["uniform"].num_updates:
                scores = [
                    float(nash_conv(sf, tr.average_behavior(), validate=False)) / divisor
                    for tr in (trackers["uniform"], trackers["linear"])
                ]
                result.avg_curve.append((it, scores[0], scores[1]))
                avg = f"avg_u {scores[0]:.6f} avg_lin {scores[1]:.6f} "
            if verbose:
                extra = " ".join(f"{k}={v:.3f}" for k, v in telemetry.items())
                spans = {n: t.stats().get("span_mean") for n, t in ranges.items()}
                span_txt = "".join(f" {n}_span {v:.4f}" for n, v in spans.items() if v is not None)
                print(f"  iter {it:>8} expl {expl:.6f}  {avg}{extra}{span_txt}", flush=True)
        if it == iters:
            break
        if trackers:
            # The PRE-update iterate, weighted t for the linear average -- the
            # off-by-one RealizationAverage.update warns about (0.6% relative on
            # Kuhn after 50 CFR+ iterations, i.e. noise-sized and plot-visible).
            pre = masked_softmax(sf, logits)
            trackers["uniform"].update(pre)
            trackers["linear"].update(pre, weight=float(it + 1))
        lr_now = (
            params.lr
            if params.lr_decay_tau is None
            else params.lr / (1.0 + it / params.lr_decay_tau)
        )
        logits, telemetry, weights = ach_step(
            sf, logits, engine, params, fixed_weight=fixed_weight, lr=lr_now
        )
        for name, tracker in ranges.items():
            tracker.update(weights[name])
    result.final_exploitability = result.curve[-1][1]
    result.best_exploitability = best
    result.telemetry = telemetry
    result.weight_range = {n: t.stats() for n, t in ranges.items()}
    result.seconds = time.time() - start
    if save_policy is not None:
        # The behaviour the dynamics actually settled on, plus the per-row weight
        # range that produced it -- enough for an offline probe to ask *where* the
        # residual exploitability sits and whether those are the rows whose weight
        # moved the most while training.
        save_policy.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "behavior": masked_softmax(sf, logits),
                "weight_lo": ranges["applied"].lo,
                "weight_hi": ranges["applied"].hi,
            },
            save_policy,
        )
        if verbose:
            print(f"  saved policy + weight range to {save_policy}")
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
    ap.add_argument("--weighting", default="reach", help="reach | uniform | rho:K | frozen:K")
    ap.add_argument("--lr-decay-tau", type=float, default=None, help="lr_t = lr/(1+t/tau)")
    ap.add_argument("--track-average", action="store_true", help="also score the AVERAGE strategy")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--save-policy", type=Path, default=None, help="final behaviour + weight range")
    ap.add_argument(
        "--frozen-ref",
        type=Path,
        default=None,
        help="policy bundle whose reach defines frozen:K (default: the initial policy)",
    )
    args = ap.parse_args()

    params = AchParams(
        eta=args.eta,
        beta=args.beta,
        l_th=args.l_th,
        lr=args.lr,
        gate=not args.no_gate,
        gate_centered=args.gate_centered,
        weighting=args.weighting,
        lr_decay_tau=args.lr_decay_tau,
    )
    print(f"exact ACH on {args.game}: {params}")
    res = run(
        args.game,
        params,
        iters=args.iters,
        eval_every=args.eval_every,
        track_average=args.track_average,
        save_policy=args.save_policy,
        frozen_ref=args.frozen_ref,
    )
    print(
        f"final expl {res.final_exploitability:.6f}  best {res.best_exploitability:.6f}"
        f"  ({res.seconds:.1f}s)"
    )
    for name, stats in res.weight_range.items():
        print(f"  {name}: " + " ".join(f"{k}={v:.6g}" for k, v in stats.items()))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res.__dict__, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
