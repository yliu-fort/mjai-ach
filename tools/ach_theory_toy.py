"""Integrate the exact ACH operator on the toy game the theory is solved on.

``tools/ach_theory_sympy.py`` proves *static* facts: the gate's invariant box,
and the closed-form leak ``pi_X = 1/((n-1)e^{2 l_th}+1)`` at its corner. This
file asks the *dynamical* question -- does the operator actually go there? --
on the smallest game where the answer is known in closed form, before any of it
is claimed about Liar's Dice.

The game (``ach_theory_sympy.TOY_M``): rock-paper-scissors plus a strictly
dominated fourth action X. Its unique Nash is uniform on {R,P,S} with zero mass
on X -- *partially* mixed, which is the case the theory says carries a floor.
Plain RPS is fully mixed and is degenerate here (every advantage is zero at its
Nash, so there is nothing for the gate to pin).

Exploitability of a symmetric-zero-sum policy ``pi`` is ``max_b (M pi)_b``,
in the repo's units (= NashConv/2, see `exploitability-vs-nashconv-units`).

Pre-registered predictions (paper hyperparameters eta=1, beta=1e-2, lr=1e-3),
with ``k`` on-support actions out of ``n``:

  no gate         -> eps -> 0, while y separation and the mean logit both
                     diverge linearly (no fixed point in y at all)
  raw gate        -> resting separation 2*l_th, so
                     eps -> (n-k) / (k*e^{2*l_th} + (n-k))
  centered gate   -> resting separation n*l_th/min(k, n-k) >= 2*l_th, so it
                     floors LOWER than the raw gate, or not at all

The centered line is a correction. The first hand derivation took the binding
side to be the gate that closes FIRST and predicted n/(n-1) -- i.e. that
centered would floor higher. This file falsified that: the m=1 centered arm
rests at separation 4.000 = 4*l_th, not 4*l_th/3, because once one side's gate
closes the other keeps driving the separation. See ``ach_theory_sympy.v9``.

Run: ``uv run python -m tools.ach_theory_toy``.

Not on the ``mjai`` import path (AGENTS.md sec.4). float64 throughout (D19).
"""

from __future__ import annotations

import math

import torch
from tools.ach_theory_sympy import TOY_M


def toy_matrix(n_dominated: int = 1) -> torch.Tensor:
    """RPS (support k=3) plus ``n_dominated`` strictly dominated actions.

    Sweeping ``n_dominated`` moves ``n-k`` while holding ``k`` fixed, which is
    how the general leak formula ``(n-k)/(k*e^{2 l_th} + (n-k))`` gets tested
    against something other than the single case it was derived on.
    """
    n = 3 + n_dominated
    m = torch.zeros(n, n, dtype=torch.float64)
    rps = torch.tensor([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]], dtype=torch.float64)
    m[:3, :3] = rps
    m[:3, 3:] = 1.0  # every real action beats every dominated one
    m[3:, :3] = -1.0
    return m


_M = torch.tensor([[float(TOY_M[i, j]) for j in range(4)] for i in range(4)], dtype=torch.float64)
assert torch.equal(_M, toy_matrix(1)), "toy_matrix(1) must reproduce the symbolic TOY_M"


def exploitability(m: torch.Tensor, pi: torch.Tensor) -> float:
    """max_b (M pi)_b -- the best response's payoff in a symmetric zero-sum game."""
    return float((m @ pi).max())


def run(
    *,
    l_th: float | None,
    centered: bool = False,
    n_dominated: int = 1,
    eta: float = 1.0,
    beta: float = 1e-2,
    lr: float = 1e-3,
    steps: int = 2_000_000,
) -> dict[str, float]:
    """Iterate ``y <- y + lr*(eta*c*A - beta*pi*(log pi + H))`` from y = 0.

    Single information set, so the visitation weight w(s) is 1 and drops out --
    which is the point: this isolates the update rule from every weighting
    question `docs/liars_operator_floor.md` already settled.

    Log-space entropy (``xlogy``) so the UNGATED arm, whose whole prediction is
    that the logits diverge, stays finite instead of returning nan.
    """
    m = toy_matrix(n_dominated)
    n = m.shape[0]
    y = torch.zeros(n, dtype=torch.float64)
    c = torch.ones(n, dtype=torch.float64)
    for _ in range(steps):
        log_pi = torch.log_softmax(y, dim=0)
        pi = torch.exp(log_pi)
        q = m @ pi
        adv = q - (pi * q).sum()
        ent = -torch.xlogy(pi, pi).sum()
        ent_grad = beta * (torch.xlogy(pi, pi) + pi * ent)
        if l_th is not None:
            y_gate = y - y.mean() if centered else y
            c = torch.where(adv >= 0, y_gate < l_th, y_gate > -l_th).to(torch.float64)
        y = y + lr * (eta * c * adv - ent_grad)
    pi = torch.softmax(y, dim=0)
    return {
        "eps": exploitability(m, pi),
        "sep": float(y[:3].mean() - y[3:].mean()),
        "y_mean": float(y.mean()),
        "gate_off_frac": 0.0 if l_th is None else float(1.0 - c.mean()),
    }


def predicted(l_th: float, centered: bool, n_dominated: int) -> float:
    """Leaked mass at the box corner == exploitability in this toy family.

    Every dominated action leaks ``e^{-l}/Z``; a best response plays any of
    R/P/S and collects +1 against each leaked unit, so eps == total leak.
    """
    n, k = 3 + n_dominated, 3
    sep = n * l_th / min(k, n - k) if centered else 2.0 * l_th
    return (n - k) / (k * math.exp(sep) + (n - k))


def main() -> None:
    steps = 2_000_000
    print(f"toy = RPS + m dominated actions; eta=1, beta=1e-2, lr=1e-3, {steps:,} steps")
    print(f"{'arm':>26} | {'eps':>10} | {'predicted':>10} | {'sep':>7} | {'y_mean':>8} | gate_off")
    print("-" * 88)

    r = run(l_th=None, steps=steps)
    print(
        f"{'no gate (m=1)':>26} | {r['eps']:10.3e} | {'-> 0':>10} | {r['sep']:7.3f}"
        f" | {r['y_mean']:8.3f} | {r['gate_off_frac']:.3f}"
    )
    for centered in (False, True):
        for l_th in (1.0, 2.0, 3.0):
            r = run(l_th=l_th, centered=centered, steps=steps)
            p = predicted(l_th, centered, 1)
            tag = f"{'centered' if centered else 'raw'} gate l_th={l_th:.0f} (m=1)"
            print(
                f"{tag:>26} | {r['eps']:10.3e} | {p:10.3e} | {r['sep']:7.3f}"
                f" | {r['y_mean']:8.3f} | {r['gate_off_frac']:.3f}"
            )
    print()
    for m_dom in (1, 3, 10):
        r = run(l_th=2.0, n_dominated=m_dom, steps=steps)
        p = predicted(2.0, False, m_dom)
        tag = f"raw l_th=2, m={m_dom} (n={3 + m_dom}, k=3)"
        print(
            f"{tag:>26} | {r['eps']:10.3e} | {p:10.3e} | {r['sep']:7.3f}"
            f" | {r['y_mean']:8.3f} | {r['gate_off_frac']:.3f}"
        )


if __name__ == "__main__":
    main()
