"""Can BRPS be tuned to convergence? A sweep in the cheap surrogate, then a shortlist.

The question this answers: `docs/brps_mlp_nonconvergence.md` says the paper's BRPS
arm dies for two independent reasons (§2 one-step overshoot, §5 the operator's own
limit cycle). Each was only ever tested one factor at a time, so "is there ANY
setting of the existing knobs that converges?" was still open.

Searching that in the RL pipeline is expensive. It does not have to be: the MLP's
amplification is ~isotropic (the two heads' parameter gradients are orthogonal, so
the NTK is `(||f||^2 + 1) * I` plus small torso cross-terms), which means the whole
MLP arm is the *tabular sampled* arm at

    lr_eff = lr * (hidden_size + 3)          (131 at the config's [128] + LayerNorm)

Verified before this sweep was trusted: at `lr_eff` the surrogate
(:mod:`tools.brps_noise`) reproduces the four measured RL arms in order and in
magnitude — surrogate exploitability 15.06 / 3.91 / 2.18 / 2.05 against the RL
arms' NashConv/2 of 33.4 / 3.2 / 2.4 / 1.5 (paper / no_ln / h1 / lr_matched), and
it diverges to non-finite logits on 1 of 3 seeds exactly like the paper arm does.

So the sweep runs in the surrogate, and the shortlist it produces is then confirmed
in the real pipeline (that confirmation is NOT in this file — it is
`tools/ab_factor_probe.py` with the printed `--overrides`).

Which knobs, and why these:

  ``eta``     scales the POLICY term only, so it is the one knob that changes the
              ratio `beta / (eta * |A|)` — the dimensionless number §5 shows owns
              the rotation-vs-contraction balance — WITHOUT slowing the entropy
              contraction the way `learning_rate` does. It is a paper knob
              (hedge coefficient eta(s), p27 Table 7).
  ``beta``    the only source of contraction (§5), at the price of a QRE bias.
  ``l_th``    clips the orbit without biasing the fixed point, but only if the box
              still contains the NE: `l_th >= log(10)/2 = 1.1513`.
  ``batch``   divides the sampling variance, which sets the stationary spread
              around the fixed point.

Reported: mean exploitability over the last window (the honest statistic for a
process that may still be orbiting), plus the config overrides that reproduce the
arm in the pipeline. Not on the ``mjai`` import path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from tools.brps_noise import NoiseParams, run

# hidden_sizes=[128] + trunk_layernorm=True, measured by tools/brps_logit_step.py.
AMPLIFICATION = 131.1
CONFIG_LR = 1e-3


def config_overrides(params: NoiseParams) -> dict[str, float]:
    """The `--overrides` dict that reproduces this surrogate arm in the pipeline."""
    out: dict[str, float] = {}
    lr_config = params.lr / AMPLIFICATION
    if abs(lr_config - CONFIG_LR) / CONFIG_LR > 1e-3:
        out["learning_rate"] = float(f"{lr_config:.3g}")
    if params.eta != 1.0:
        out["eta"] = params.eta
    if params.beta != 1e-2:
        out["entropy_coef"] = params.beta
    if params.l_th != 2.0:
        out["l_th"] = params.l_th
    if params.batch != 64:
        out["target_samples"] = params.batch
    if params.iw_clip is not None:
        out["iw_clip"] = params.iw_clip
    return out


def arm_stats(params: NoiseParams, updates: int, seeds: tuple[int, ...]) -> dict[str, float]:
    """Run one arm over ``seeds``; return the SECOND-half window + divergences.

    The second half, not the whole run: an arm that reaches its plateau after the
    transient would otherwise be scored on the transient it already left. The
    average-policy column is cumulative by construction (D16), so it is the one
    number here that does include the transient — as the theorem's does.
    """
    keys = ("expl_mean", "expl_max", "tv_mean", "pi_max_mean", "expl_avg_policy")
    acc: dict[str, list[float]] = {k: [] for k in keys}
    diverged = 0
    for seed in seeds:
        res = run(params, updates, seed, checkpoints=(updates // 2, updates))
        rows = res["rows"]
        if res["diverged_at"] is not None or len(rows) < 2:  # type: ignore[arg-type]
            diverged += 1
            continue
        row = rows[1]  # type: ignore[index]
        for k in keys:
            acc[k].append(row[k])
    out = {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}
    out["pi_max"] = out.pop("pi_max_mean")
    out["diverged"] = float(diverged)
    return out


def grid(etas: list[float], betas: list[float], l_ths: list[float], batches: list[int]):
    """The cross product, at the config's own lr_eff (architecture untouched)."""
    base = NoiseParams(lr=CONFIG_LR * AMPLIFICATION)
    for eta in etas:
        for beta in betas:
            for l_th in l_ths:
                for batch in batches:
                    yield replace(base, eta=eta, beta=beta, l_th=l_th, batch=batch)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--updates", type=int, default=156_250, help="1e7 env-steps at batch 64")
    ap.add_argument(
        "--env-steps",
        type=int,
        default=None,
        help="budget in env-steps instead of updates: updates = env_steps // batch. "
        "Required for a fair batch-size axis — a bigger batch buys less variance "
        "per update but fewer updates for the same sample budget.",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eta", type=float, nargs="+", default=[1.0, 0.1, 0.02])
    ap.add_argument("--beta", type=float, nargs="+", default=[0.01, 0.1, 0.5])
    ap.add_argument("--l-th", type=float, nargs="+", default=[2.0, 1.1513])
    ap.add_argument("--batch", type=int, nargs="+", default=[64])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    seeds = tuple(args.seeds)
    print(
        f"surrogate sweep: lr_eff = {CONFIG_LR * AMPLIFICATION:.4g} "
        f"(config lr {CONFIG_LR} x amp {AMPLIFICATION}), "
        f"{f'{args.env_steps} env-steps' if args.env_steps else f'{args.updates} updates'}, "
        f"seeds {list(seeds)}"
    )
    print(
        f"{'eta':>7}{'beta':>7}{'l_th':>8}{'batch':>7}{'expl_tail':>11}{'expl_max':>10}"
        f"{'expl_avg':>10}{'tv':>8}{'pi_max':>8}{'div':>5}   config overrides"
    )
    results = []
    for params in grid(args.eta, args.beta, args.l_th, args.batch):
        updates = args.updates if args.env_steps is None else args.env_steps // params.batch
        st = arm_stats(params, updates, seeds)
        ov = config_overrides(params)
        results.append({"params": params.__dict__ | {}, "stats": st, "overrides": ov})
        print(
            f"{params.eta:>7g}{params.beta:>7g}{params.l_th:>8.4g}{params.batch:>7d}"
            f"{st['expl_mean']:>11.4f}{st['expl_max']:>10.3f}{st['expl_avg_policy']:>10.4f}"
            f"{st['tv_mean']:>8.4f}{st['pi_max']:>8.3f}{int(st['diverged']):>5d}   {json.dumps(ov)}"
        )
    alive = [r for r in results if not np.isnan(r["stats"]["expl_mean"])]
    if alive:
        best = min(alive, key=lambda r: r["stats"]["expl_mean"])
        print(f"\nbest: expl {best['stats']['expl_mean']:.4f} at {json.dumps(best['overrides'])}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
