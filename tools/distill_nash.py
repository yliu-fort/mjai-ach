"""Distill exact tabular Nash into the MLP -> measure the REPRESENTATION ceiling.

The question: can a finite ReLU-MLP represent Liar's Dice's Nash strategy well
enough to reach low exploitability, or does representation cap it regardless of
the RL algorithm? Solve the exact Nash (tabular CFR+, ~1e-8), then SUPERVISE the
MLP on it (KL/NLL of Nash under the MLP's softmax over legal actions) -- no RL,
no critic, no gate, no 1/pi_old. The distilled MLP's exploitability IS the
representation ceiling for that width.

If the distilled floor ~1e-3 -> representation-limited: the MLP cannot reach
machine precision by any algorithm (docs/liars_machine_precision.md). If it is
~1e-6+ -> representation is fine and the RL algorithm owns the residual floor.

Exploitability via the in-house seqform nash_conv (= NashConv/2 at 2p), the
repo/paper convention. Usage::

    uv run python tools/distill_nash.py                 # width sweep
    uv run python tools/distill_nash.py --widths 256 512 --epochs 5000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from open_spiel.python.algorithms import cfr

from mjai.agents.mlp import MLPSharedActorCritic
from mjai.eval.average_policy import behavior_of
from mjai.games.loader import load_game
from mjai.seqform.plan import nash_conv
from mjai.seqform.tree import build_sequence_form


def solve_nash(spec, sf, iters: int):
    """Tabular CFR+ -> Nash behavior tensor aligned to the seqform's infoset rows.

    CFR+'s average_policy() is an OpenSpiel TabularPolicy keyed by info-state
    string; read it row-by-row via policy_for_key (behavior_of needs a mjai
    Policy with action_logits_batch, which the CFR+ policy is not). Alignment is
    checked downstream by nash_conv (must be ~0).
    """
    solver = cfr.CFRPlusSolver(spec.game)
    for _ in range(iters):
        solver.evaluate_and_update_policy()
    avg = solver.average_policy()
    behavior = torch.zeros(sf.num_infosets, sf.max_actions, dtype=torch.float64)
    for row, key in enumerate(sf.infoset_keys):
        behavior[row] = torch.tensor(list(avg.policy_for_key(key)), dtype=torch.float64)
    return behavior


def distill(
    spec,
    sf,
    nash_behavior,
    width,
    epochs,
    lr,
    seed=0,
    float64=False,
    eta_min=0.0,
    optimizer="adam",
    lbfgs_iters=300,
):
    """Supervised-fit an MLP of given width to the Nash behavior; return (mlp, final_loss).

    optimizer="adam": epochs of Adam (cosine lr). "lbfgs": a short Adam warmup (min(epochs,
    5000)) into the basin, then lbfgs_iters quasi-Newton polish steps -- L-BFGS breaks the
    Adam plateau on this smooth supervised loss and reaches ~2x lower KL (docs/
    liars_machine_precision.md sec 5.4). Each L-BFGS step does up to 25 line-search iters.
    """
    dtype = torch.float64 if float64 else torch.float32
    obs = sf.infoset_observation.to(dtype)  # [I, obs]
    legal = sf.legal_mask  # [I, A] bool
    target = nash_behavior.to(dtype)  # [I, A] probs over legal, 0 elsewhere
    mlp = MLPSharedActorCritic(
        spec.obs_size, spec.num_actions, hidden_sizes=(width,), seed=seed, device="cpu"
    )
    if float64:
        mlp = mlp.double()  # train in float64 to escape the ~1e-7 float32 precision cap

    def nll():
        masked = mlp(obs)[0].masked_fill(~legal, -1e9)
        return -(target * torch.log_softmax(masked, dim=-1)).sum(dim=-1).mean()

    warmup = min(epochs, 5000) if optimizer == "lbfgs" else epochs
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(warmup, 1), eta_min=eta_min)
    last_loss = float("nan")
    for _ in range(warmup):
        loss = nll()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        last_loss = float(loss)
    if optimizer == "lbfgs":
        opt = torch.optim.LBFGS(
            mlp.parameters(), lr=1.0, max_iter=25, line_search_fn="strong_wolfe"
        )
        for _ in range(lbfgs_iters):

            def closure():
                opt.zero_grad()
                loss = nll()
                loss.backward()
                return loss

            opt.step(closure)
        last_loss = float(nll())
    return mlp, last_loss


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default="liars_dice1")
    p.add_argument("--cfr-iters", type=int, default=2000)
    p.add_argument("--widths", type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=2e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--float64",
        action="store_true",
        help="Train MLP in float64 (escapes the ~1e-7 float32 cap).",
    )
    p.add_argument(
        "--eta-min", type=float, default=0.0, help="Cosine lr floor (use ~1e-6 to hold a small lr)."
    )
    p.add_argument(
        "--optimizer",
        choices=["adam", "lbfgs"],
        default="adam",
        help="lbfgs = Adam warmup + quasi-Newton polish (breaks the Adam KL plateau).",
    )
    p.add_argument("--lbfgs-iters", type=int, default=300, help="L-BFGS polish iterations.")
    p.add_argument(
        "--nash-cache",
        default="runs/nash_{game}_behavior.pt",
        help="Cache for the Nash behavior tensor (CFR+ on liars is ~1 iter/s).",
    )
    p.add_argument("--force-solve", action="store_true", help="Re-solve Nash even if cached.")
    args = p.parse_args(argv)

    spec = load_game(args.game)
    sf = build_sequence_form(spec)
    print(
        f"{args.game}: {sf.num_infosets} infosets, {sf.num_sequences} seqs, "
        f"{sf.max_actions} actions"
    )

    cache = Path(args.nash_cache.format(game=args.game))
    if cache.exists() and not args.force_solve:
        nash_beh = torch.load(cache, weights_only=True)
        print(f"loaded cached Nash behavior from {cache}")
    else:
        t0 = time.time()
        print(f"solving Nash (CFR+ {args.cfr_iters} iters, ~{args.cfr_iters}s)...", flush=True)
        nash_beh = solve_nash(spec, sf, args.cfr_iters)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(nash_beh, cache)
        print(f"  solved + cached to {cache} ({time.time() - t0:.0f}s)")
    nash_expl = float(nash_conv(sf, nash_beh)) / 2
    print(f"  Nash exploitability = {nash_expl:.3e} (target quality)")

    print(
        f"\n{'width':>6}{'params':>9}{'final NLL':>12}{'distilled expl':>18}  (vs Nash {nash_expl:.1e})"
    )
    print("-" * 60)
    results = []
    for w in args.widths:
        mlp, loss = distill(
            spec,
            sf,
            nash_beh,
            w,
            args.epochs,
            args.lr,
            args.seed,
            args.float64,
            args.eta_min,
            args.optimizer,
            args.lbfgs_iters,
        )
        expl = float(nash_conv(sf, behavior_of(sf, mlp))) / 2
        nparams = sum(par.numel() for par in mlp.parameters())
        print(f"{w:>6}{nparams:>9}{loss:>12.4f}{expl:>18.4e}")
        results.append((w, expl, loss))
    print("\nrepresentation ceiling per width (lowest exploitability an MLP of this")
    print("width can represent, independent of RL). If this bottoms out well above")
    print("1e-8, machine precision is unattainable for the MLP -- representation-limited.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
