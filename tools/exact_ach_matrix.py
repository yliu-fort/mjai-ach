"""Ablation matrix for the exact ACH dynamics (:mod:`tools.exact_ach`).

Each arm switches off exactly one suspect in the update rule, so the arm that
moves the floor is the arm that owns it:

  ``paper``      every term as published (beta=1e-2, one-sided gate l_th=2,
                 on-policy visitation weighting)
  ``beta0``      entropy off      -> is the floor the entropy bias?
  ``nogate``     logit gate off   -> is the floor the sharpening cap?
  ``beta0_nogate`` both off       -> the bare logit-space policy gradient
  ``uniform``    flat information-set weighting instead of on-policy reach
                 -> is the floor the reach-weight mismatch (Theorem 1's second,
                 T-independent term, ``docs/paper_spec_ach.md`` §1.1)?
  ``uniform_beta0_nogate`` all three off at once -> the control that must reach
                 Nash if the machinery is sound.

Budget is quoted in *updates*, matching the RL runs: 1e7 env-steps at batch 64
is ~156k updates.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tools.exact_ach import AchParams, run

ARMS: dict[str, AchParams] = {
    "paper": AchParams(),
    "beta0": AchParams(beta=0.0),
    "nogate": AchParams(gate=False),
    "beta0_nogate": AchParams(beta=0.0, gate=False),
    "uniform": AchParams(weighting="uniform"),
    "uniform_beta0_nogate": AchParams(beta=0.0, gate=False, weighting="uniform"),
    # Tempered family, mean-1 renormalized so K moves only the SHAPE of the
    # weighting (see AchParams.weighting). K=1 is what sampling delivers, K=0 is
    # flat; the interior is the reach^-kappa tempering that failed in RL. This
    # arm has exact advantages, no critic and every row present, so it isolates
    # whether the MOVING TARGET alone is what breaks the tempering.
    **{f"rho{k}": AchParams(weighting=f"rho:{k}") for k in (0.0, 0.25, 0.5, 0.75, 1.0)},
    # Who owns the ~0.099 the best-conditioned exact dynamics stops at? The
    # entropy regularizer biases the fixed point away from Nash and the l_th
    # gate truncates logit growth, so they are the two suspects -- but
    # docs/liars_residual_floor.md §3 concluded both were "inert on Liar's
    # Dice" from arms run under the RAW reach weighting, a regime where the
    # policy never sharpened enough to reach the gate at all (gate_off_frac
    # 0.000, entropy 0.508). Under rho:0.5 the gate is live (gate_off_frac
    # 0.43, entropy 0.199), so that conclusion does not carry over and the 2x2
    # has to be re-run here.
    "rho0.5_beta0": AchParams(beta=0.0, weighting="rho:0.5"),
    "rho0.5_nogate": AchParams(gate=False, weighting="rho:0.5"),
    "rho0.5_beta0_nogate": AchParams(beta=0.0, gate=False, weighting="rho:0.5"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="kuhn")
    ap.add_argument("--iters", type=int, default=156_000)
    ap.add_argument("--eval-every", type=int, default=10_000)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--lr", type=float, default=None, help="override lr on every arm")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or Path("runs/exact_ach") / f"{args.game}_matrix.json"
    results: dict[str, dict[str, object]] = {}
    if out.is_file():  # resume: keep arms already measured
        results = json.loads(out.read_text(encoding="utf-8"))

    for name in args.arms:
        params = ARMS[name]
        if args.lr is not None:
            params = AchParams(**{**params.__dict__, "lr": args.lr})
        print(f"\n=== {args.game} / {name}: {params}", flush=True)
        t = time.time()
        res = run(args.game, params, iters=args.iters, eval_every=args.eval_every)
        results[name] = {
            "params": res.params,
            "iters": res.iters,
            "final": res.final_exploitability,
            "best": res.best_exploitability,
            "curve": res.curve,
            "telemetry": res.telemetry,
            "seconds": res.seconds,
        }
        print(
            f"=== {name}: final {res.final_exploitability:.6f} "
            f"best {res.best_exploitability:.6f} ({time.time() - t:.0f}s)",
            flush=True,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nwrote {out}")
    print(f"{'arm':<24} {'final':>10} {'best':>10}")
    for name, r in results.items():
        print(f"{name:<24} {r['final']:>10.6f} {r['best']:>10.6f}")


if __name__ == "__main__":
    main()
