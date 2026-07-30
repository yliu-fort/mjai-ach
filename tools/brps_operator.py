"""Exact (noiseless) ACH dynamics on BRPS — the 3-dimensional case of tools/exact_ach.py.

Why BRPS deserves its own probe rather than a `--game brps` flag on
:mod:`tools.exact_ach`: that tool walks the sequence-form tree, and reaching a
simultaneous-move game there means `convert_to_turn_based`, which invents a
second decision node and changes what an information set *is*. BRPS does not
need any of it. It is one information set, three actions, and — because its
payoff matrix is antisymmetric and mirror self-play puts the same policy in both
seats — the state value is **exactly zero at every policy**, so the expected ACH
operator is a closed-form map on three logits:

    A(y)   = M @ softmax(y)                    (V = pi^T M pi = 0 for M = -M^T)
    c_a    = 1{y_a <  l_th}  if A_a >= 0       (paper's one-sided gate, p24)
             1{y_a > -l_th}  if A_a <  0
    y  <-  y + lr * ( eta * c * A  -  beta * pi * (log pi + H) )

The visitation weight `rho` of :mod:`tools.exact_ach` is identically 1 here (one
information set, always reached), so it does not appear.

What this separates: BRPS's Nash equilibrium (1/16, 10/16, 5/16) is **fully
mixed**, which is the degenerate case `docs/ach_operator_theory.md` §3 flags —
`sum_a A_a = 0` at equilibrium, so the gauge drift that owns the Liar's Dice
floor is absent, and the leak formula `(n-k)/(k e^{2 l_th} + (n-k))` gives 0 at
k = n. So if BRPS fails, it fails for a *different* reason than Liar's Dice, and
this file's job is to say whether that reason is already present in the
deterministic map (structure) or only appears once sampling noise is added
(statistics — that rung lives in ``tools/brps_noise.py``).

Metric conventions: with both seats on the same policy and V = 0,
`NashConv = 2 * max_a A_a` and `exploitability = NashConv / 2 = max_a A_a`
(memory: exploitability-vs-nashconv-units). Pure Rock gives NashConv 50, which
is the number `docs/league_investigation.md` §4.2 reports for the collapsed seed.

Not on the ``mjai`` import path: an analysis tool, all arithmetic float64
(AGENTS.md D19).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyspiel

# The analytic Nash equilibrium of BRPS (configs/games/brps.yaml).
NASH = np.array([1.0 / 16.0, 10.0 / 16.0, 5.0 / 16.0], dtype=np.float64)


def payoff_matrix() -> np.ndarray:
    """BRPS row payoffs, read from OpenSpiel rather than hard-coded (AGENTS.md §9).

    Asserts antisymmetry, because every closed form below (V = 0, NashConv =
    2 max_a A_a, one shared policy) rests on it.
    """
    game = pyspiel.load_game("matrix_brps")
    m = np.asarray(game.row_utilities(), dtype=np.float64).reshape(3, 3)
    if not np.allclose(m, -m.T, atol=0.0):
        raise AssertionError(f"BRPS matrix is not antisymmetric:\n{m}")
    return m


def softmax(y: np.ndarray) -> np.ndarray:
    z = y - y.max()
    e = np.exp(z)
    return e / e.sum()


def advantages(m: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """``A_a = Q_a - V`` with ``V = pi^T M pi = 0`` (antisymmetric M)."""
    return m @ pi


def exploitability(m: np.ndarray, pi: np.ndarray) -> float:
    """``max_a A_a`` — NashConv/2 for the symmetric mirror pair."""
    return float(advantages(m, pi).max())


def tv_to_nash(pi: np.ndarray) -> float:
    """Total-variation distance to the analytic NE (the geometric read)."""
    return float(0.5 * np.abs(pi - NASH).sum())


@dataclass(frozen=True)
class AchParams:
    """The paper's BRPS-arm values (configs/exp/brps_ach_mlp_mirror.yaml)."""

    lr: float = 1e-3
    eta: float = 1.0
    beta: float = 1e-2
    l_th: float = 2.0
    gate: bool = True
    payoff_scale: float = 1.0  # divide M by this: the scale-mismatch arm


def gate_mask(y: np.ndarray, adv: np.ndarray, params: AchParams) -> np.ndarray:
    """Advantage-sign-dependent one-sided gate on the RAW logit (repo default).

    ``gate_centered_logits=False`` in ``nn_losses.ach_policy_loss`` and in the
    BRPS config, so the threshold applies to ``y_a`` itself, not ``y_a - ybar``.
    """
    if not params.gate:
        return np.ones_like(y)
    return np.where(adv >= 0.0, y < params.l_th, y > -params.l_th).astype(np.float64)


def entropy_grad(pi: np.ndarray) -> np.ndarray:
    """``d/dy_j [ sum_b pi_b log pi_b ] = pi_j (log pi_j + H)`` (theory §2, V3)."""
    h = float(-(pi * np.log(pi)).sum())
    return pi * (np.log(pi) + h)


def ach_step(y: np.ndarray, m: np.ndarray, params: AchParams) -> np.ndarray:
    """One expected ACH update on the logit vector."""
    pi = softmax(y)
    adv = advantages(m, pi)
    c = gate_mask(y, adv, params)
    return y + params.lr * (params.eta * c * adv - params.beta * entropy_grad(pi))


def jacobian(y: np.ndarray, m: np.ndarray, params: AchParams) -> np.ndarray:
    """Jacobian of the *update map* at ``y`` (finite differences, central).

    Numeric rather than symbolic on purpose: the gate makes the map piecewise,
    and a finite difference reports the branch the point is actually in. The
    caller reads the eigenvalues of ``J - I`` (the vector field) — imaginary
    parts are rotation per step, real parts contraction per step.
    """
    eps = 1e-6
    n = y.size
    j = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        e = np.zeros(n)
        e[k] = eps
        j[:, k] = (ach_step(y + e, m, params) - ach_step(y - e, m, params)) / (2.0 * eps)
    return j


def box_worst_case(m: np.ndarray, l_th: float, grid: int = 601) -> tuple[float, np.ndarray]:
    """Worst (largest) exploitability reachable with every logit inside the gate box.

    The one-sided gate's invariant set is ``y in [-l_th, l_th]^n``: a logit the
    advantage pushes up is released at ``+l_th``, one pushed down at ``-l_th``.
    Since softmax is gauge-invariant, two logits suffice — fix ``y_0 = 0`` and
    sweep ``y_1, y_2`` over ``[-2 l_th, 2 l_th]`` (the reachable *separation*
    range) — but the box is stated on raw logits, so sweep the raw box directly
    with ``y_0`` at each of its three extremes to keep the argument honest.

    This is the mirror image of the Liar's Dice result: there the box was a
    *floor* (it forbade the sharpening Nash needed); here it is a *ceiling*, and
    the question is how much damage it still permits.
    """
    axis = np.linspace(-l_th, l_th, grid)
    best = -np.inf
    best_pi = NASH
    for y0 in (-l_th, 0.0, l_th):
        for y1 in axis:
            ys = np.stack([np.full_like(axis, y0), np.full_like(axis, y1), axis], axis=1)
            pis = np.exp(ys - ys.max(axis=1, keepdims=True))
            pis /= pis.sum(axis=1, keepdims=True)
            expl = (pis @ m.T).max(axis=1)
            k = int(expl.argmax())
            if expl[k] > best:
                best = float(expl[k])
                best_pi = pis[k]
    return best, best_pi


def nash_reachable(l_th: float) -> bool:
    """Does the box contain the NE? NE needs a logit spread of ``log 10 = 2.303``.

    Inside ``[-l_th, l_th]`` the largest expressible probability ratio is
    ``e^{2 l_th}``, so the NE (max/min ratio 10) needs ``l_th >= log(10)/2``.
    """
    return 2.0 * l_th >= float(np.log(10.0))


def run(
    params: AchParams,
    iters: int,
    y0: np.ndarray | None = None,
    record_every: int = 1000,
) -> dict[str, object]:
    """Iterate the expected operator; return the curve and the endpoint diagnosis."""
    m = payoff_matrix() / params.payoff_scale
    y = np.zeros(3, dtype=np.float64) if y0 is None else y0.astype(np.float64).copy()
    curve: list[dict[str, float]] = []
    for t in range(iters + 1):
        pi = softmax(y)
        if t % record_every == 0 or t == iters:
            curve.append(
                {
                    "iter": float(t),
                    # Report exploitability in the ORIGINAL payoff units, so a
                    # scaled arm stays comparable to the RL runs.
                    "expl": exploitability(payoff_matrix(), pi),
                    "tv": tv_to_nash(pi),
                    "gate_off_frac": float(1.0 - gate_mask(y, advantages(m, pi), params).mean()),
                    "y_spread": float(y.max() - y.min()),
                    "y_mean": float(y.mean()),
                }
            )
        if t < iters:
            y = ach_step(y, m, params)
    pi = softmax(y)
    eig = np.linalg.eigvals(jacobian(y, m, params) - np.eye(3))
    # Drop the gauge eigenvalue (softmax ignores y + k*1) before reading rates.
    order = np.argsort(-np.abs(eig))
    return {
        "params": asdict(params),
        "iters": iters,
        "final_pi": pi.tolist(),
        "final_expl": exploitability(payoff_matrix(), pi),
        "final_tv": tv_to_nash(pi),
        "final_y": y.tolist(),
        "eig_real": [float(np.real(eig[i])) for i in order],
        "eig_imag": [float(np.imag(eig[i])) for i in order],
        "curve": curve,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=156_000, help="1e7 env-steps at batch 64")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1e-2)
    ap.add_argument("--l-th", type=float, default=2.0)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--payoff-scale", type=float, default=1.0)
    ap.add_argument("--y0", type=float, nargs=3, default=None, help="initial logits")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    params = AchParams(
        lr=args.lr,
        eta=args.eta,
        beta=args.beta,
        l_th=args.l_th,
        gate=not args.no_gate,
        payoff_scale=args.payoff_scale,
    )
    y0 = None if args.y0 is None else np.asarray(args.y0, dtype=np.float64)
    res = run(params, args.iters, y0=y0)
    m = payoff_matrix()
    box_expl, box_pi = box_worst_case(m, args.l_th)
    res["box_worst_expl"] = box_expl
    res["box_worst_pi"] = box_pi.tolist()
    res["nash_inside_box"] = nash_reachable(args.l_th)

    print(f"exact ACH on BRPS: {params}")
    print(f"  NE = {NASH.tolist()}  expl(NE) = {exploitability(m, NASH):.3e}")
    print(
        f"  box(l_th={args.l_th}): NE reachable = {res['nash_inside_box']}, "
        f"worst expl inside = {box_expl:.4f} at pi = {np.round(box_pi, 4).tolist()}"
    )
    for row in res["curve"]:  # type: ignore[union-attr]
        print(
            f"  it {int(row['iter']):>8d}  expl {row['expl']:9.4f}  tv {row['tv']:.4f}  "
            f"spread {row['y_spread']:7.3f}  ymean {row['y_mean']:+.3f}  "
            f"gate_off {row['gate_off_frac']:.2f}"
        )
    print(f"  final pi = {np.round(res['final_pi'], 5).tolist()}")
    print(f"  vector-field eigenvalues (per step): real {np.round(res['eig_real'], 10).tolist()}")
    print(f"                                       imag {np.round(res['eig_imag'], 10).tolist()}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
