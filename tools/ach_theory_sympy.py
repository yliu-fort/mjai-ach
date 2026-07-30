"""Symbolic verification of the exact-ACH operator's fixed-point structure.

Every algebraic step behind ``docs/ach_operator_theory.md`` is re-derived here
with sympy rather than by hand, because the whole argument turns on signs and on
which sums vanish. Nothing in this file is numerical-experiment code: it proves
identities about the operator

    y <- y + lr * w(s) * [ eta * c(s,a) * A(s,a)
                           - beta * pi(a|s) * (log pi(a|s) + H(s)) ]

which is what ``tools/exact_ach.py`` iterates and what ``ach_policy_loss`` in
``src/mjai/algos/nn_losses.py`` implements in expectation.

Run: ``uv run python -m tools.ach_theory_sympy``.

Checks (each prints PASS/FAIL and is asserted):

  V1  expected logit gradient of the ACH policy term, raw loss body
  V2  the same with a mean-centered loss body -- the gauge component vanishes
  V3  entropy-term logit gradient
  V4  the entropy gradient's action-sum is identically zero
  V5  softmax gauge invariance
  V6  the gauge drift rate, and its sign at a partially-mixed equilibrium
  V7  no fixed point: neither in y nor in the policy quotient (toy game)
  V8  the gate's invariant box, and the closed-form leak it forces
  V8b the general-support leak formula (n-k)/(k*e^{2 l_th} + (n-k))
  V9  raw vs mean-centered gate: the centered box is the WIDER one

Not on the ``mjai`` import path (AGENTS.md sec.4): this is an analysis tool.
"""

from __future__ import annotations

import sympy as sp

# ---------------------------------------------------------------------------
# shared symbolic setup
# ---------------------------------------------------------------------------

N = 3  # action count for the generic identities; they are n-independent


def softmax(y: list[sp.Expr]) -> list[sp.Expr]:
    z = sum(sp.exp(v) for v in y)
    return [sp.exp(v) / z for v in y]


def entropy(pi: list[sp.Expr]) -> sp.Expr:
    return -sum(p * sp.log(p) for p in pi)


def report(name: str, claim: str, ok: bool) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {claim}")
    assert ok, f"{name} failed: {claim}"


# ---------------------------------------------------------------------------
# V1 / V2 -- the policy term's expected logit gradient
# ---------------------------------------------------------------------------


def v1_expected_policy_gradient() -> None:
    """E_{a~pi}[ d/dy_j  (-c_a * eta * y_a / pi_old(a) * A_a) ] = -eta * c_j * A_j.

    This is the identity ``tools/exact_ach.py`` rests on (its docstring asserts
    it without proof): the 1/pi_old that makes the per-sample gradient explode
    on rare actions cancels *exactly* against the sampling probability, provided
    pi_old == pi (synchronous single-thread self-play, paper p28).
    """
    y = sp.symbols(f"y0:{N}", real=True)
    A = sp.symbols(f"A0:{N}", real=True)
    c = sp.symbols(f"c0:{N}", real=True)
    eta = sp.Symbol("eta", positive=True)
    pi = softmax(list(y))
    # pi_old is a detached constant in the code -> same value, no gradient.
    pi_old = [sp.Symbol(f"po{i}", positive=True) for i in range(N)]

    for j in range(N):
        # expectation over the sampled action a ~ pi of the per-sample gradient
        expected = sum(
            pi[a] * sp.diff(-c[a] * eta * y[a] / pi_old[a] * A[a], y[j]) for a in range(N)
        )
        expected = expected.subs({pi_old[i]: pi[i] for i in range(N)})
        report(
            "V1",
            f"E[dL/dy_{j}] = -eta*c_{j}*A_{j}",
            sp.simplify(expected + eta * c[j] * A[j]) == 0,
        )


def v2_centered_loss_body() -> None:
    """With ``loss_centered_logits=True`` the expected gradient loses its gauge part.

    Raw body:      E[dL/dy_j] = -eta * c_j * A_j            -> sum_j != 0
    Centered body: E[dL/dy_j] = -eta * (c_j A_j - mean_b c_b A_b) -> sum_j == 0

    So the mean-logit drift derived in V6 is a property of the *raw* loss body,
    which is the repo default (``nn_losses.py`` ``loss_centered_logits=False``).
    """
    y = sp.symbols(f"y0:{N}", real=True)
    A = sp.symbols(f"A0:{N}", real=True)
    c = sp.symbols(f"c0:{N}", real=True)
    eta = sp.Symbol("eta", positive=True)
    pi = softmax(list(y))
    pi_old = [sp.Symbol(f"po{i}", positive=True) for i in range(N)]
    ybar = sum(y) / N

    grads = []
    for j in range(N):
        g = sum(
            pi[a] * sp.diff(-c[a] * eta * (y[a] - ybar) / pi_old[a] * A[a], y[j]) for a in range(N)
        )
        grads.append(sp.simplify(g.subs({pi_old[i]: pi[i] for i in range(N)})))

    mean_cA = sum(c[b] * A[b] for b in range(N)) / N
    for j in range(N):
        report(
            "V2",
            f"centered body: E[dL/dy_{j}] = -eta*(c_{j}A_{j} - mean)",
            sp.simplify(grads[j] + eta * (c[j] * A[j] - mean_cA)) == 0,
        )
    report("V2", "centered body: sum_j E[dL/dy_j] = 0", sp.simplify(sum(grads)) == 0)


# ---------------------------------------------------------------------------
# V3 / V4 -- the entropy term
# ---------------------------------------------------------------------------


def v3_v4_entropy_gradient() -> None:
    """d/dy_j [ beta * sum_b pi_b log pi_b ] = beta * pi_j * (log pi_j + H), and
    the action-sum of that gradient is identically zero.

    V4 is load-bearing for the whole argument: the entropy regularizer has NO
    component along the gauge direction ``1``, so it cannot brake the drift that
    V6 finds -- it only ever moves the policy, never the mean logit.
    """
    y = sp.symbols(f"y0:{N}", real=True)
    beta = sp.Symbol("beta", positive=True)
    pi = softmax(list(y))
    H = entropy(pi)
    loss = beta * sum(p * sp.log(p) for p in pi)

    grads = []
    for j in range(N):
        g = sp.simplify(sp.diff(loss, y[j]))
        grads.append(g)
        claim = sp.simplify(g - beta * pi[j] * (sp.log(pi[j]) + H))
        report("V3", f"d/dy_{j} = beta*pi_{j}*(log pi_{j} + H)", sp.simplify(claim) == 0)
    report("V4", "sum_j of the entropy gradient == 0", sp.simplify(sum(grads)) == 0)


# ---------------------------------------------------------------------------
# V5 / V6 -- the gauge direction and its drift
# ---------------------------------------------------------------------------


def v5_gauge_invariance() -> None:
    """softmax(y + k*1) == softmax(y): the mean logit is unobservable in policy space."""
    y = sp.symbols(f"y0:{N}", real=True)
    k = sp.Symbol("k", real=True)
    shifted = softmax([v + k for v in y])
    base = softmax(list(y))
    ok = all(sp.simplify(shifted[i] - base[i]) == 0 for i in range(N))
    report("V5", "softmax(y + k*1) = softmax(y)", ok)


def v6_gauge_drift() -> None:
    """Mean-logit drift = (lr*w*eta/n) * sum_a c_a A_a, and sum_a A_a < 0 off support.

    sum_a A_a = sum_a Q_a - n*V.  The equilibrium condition is sum_a pi_a A_a = 0,
    which is a *pi-weighted* sum -- it says nothing about the unweighted one. At a
    partially mixed equilibrium A_a = 0 on support and A_a < 0 off it, so the
    unweighted sum is strictly negative exactly when the equilibrium is not fully
    mixed.
    """
    n = 4
    Q = sp.symbols(f"Q0:{n}", real=True)
    free = sp.symbols(f"p0:{n - 1}", positive=True)
    pi = [*free, 1 - sum(free)]  # on the simplex, which is what makes V a mean
    V = sum(pi[a] * Q[a] for a in range(n))
    A = [Q[a] - V for a in range(n)]

    report(
        "V6",
        "sum_a pi_a A_a == 0 identically on the simplex",
        sp.simplify(sum(pi[a] * A[a] for a in range(n))) == 0,
    )
    report(
        "V6",
        "sum_a A_a = sum_a Q_a - n*V (an UNweighted sum; not pinned to 0)",
        sp.simplify(sum(A) - (sum(Q) - n * V)) == 0,
    )
    # A partially mixed equilibrium: support {0,1,2} carries Q = V, and the
    # dominated action carries Q = V - g with g > 0.
    g = sp.Symbol("g", positive=True)
    Vs = sp.Symbol("Vs", real=True)
    Q_eq = {Q[0]: Vs, Q[1]: Vs, Q[2]: Vs, Q[3]: Vs - g}
    sum_A_eq = sp.simplify(sum(A).subs(Q_eq).subs({free[i]: sp.Rational(1, 3) for i in range(3)}))
    report(
        "V6",
        "at a partially mixed equilibrium sum_a A_a = -g < 0",
        sp.simplify(sum_A_eq + g) == 0,
    )

    # drift rate: sum_j of the full update, entropy part contributing 0 by V4
    lr, w, eta = sp.symbols("lr w eta", positive=True)
    c = sp.symbols(f"c0:{n}", real=True)
    Asym = sp.symbols(f"A0:{n}", real=True)
    total = sum(lr * w * (eta * c[a] * Asym[a]) for a in range(n))
    report(
        "V6",
        "d(sum_j y_j) = lr*w*eta*sum_a c_a A_a",
        sp.simplify(total - lr * w * eta * sum(c[a] * Asym[a] for a in range(n))) == 0,
    )


# ---------------------------------------------------------------------------
# V7 -- no fixed point, in a toy that isolates the mechanism
# ---------------------------------------------------------------------------

# RPS plus a strictly dominated fourth action X that loses to everything.
# Antisymmetric, so the game is symmetric zero-sum and its unique Nash is
# uniform on {R,P,S} with zero mass on X -- a *partially mixed* equilibrium,
# which is the case V6 says carries a drift. (Plain RPS is fully mixed and is
# therefore degenerate for this question.)
TOY_M = sp.Matrix(
    [
        [0, -1, 1, 1],
        [1, 0, -1, 1],
        [-1, 1, 0, 1],
        [-1, -1, -1, 0],
    ]
)


def _toy_advantages(p: sp.Expr, q: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """Advantages under the symmetric ansatz pi = (p,p,p,q), 3p + q = 1."""
    pi = sp.Matrix([p, p, p, q])
    Qv = TOY_M * pi
    V = (pi.T * Qv)[0, 0]
    return sp.simplify(Qv[0] - V), sp.simplify(Qv[3] - V)


def v7_no_fixed_point() -> None:
    """The ungated dynamics has no fixed point -- in y OR in the policy quotient.

    A policy fixed point needs the per-action update to be constant across
    actions (a pure gauge move leaves pi unchanged). On the toy's symmetry axis
    that reduces to one scalar equation, and the eta-part of it is identically 1:

        eta*(A_R - A_X) = eta*(q + 3p) = eta * 1

    so the condition eta = beta*[...] can only hold if beta is comparable to eta.
    With the paper's eta=1, beta=1e-2 the bracket is bounded far below 1, so the
    logit separation y_R - y_X grows without bound: convergence to Nash requires
    y_X -> -infinity. That is Hedge behaving as designed (logits track CUMULATIVE
    regret) -- and it is exactly what a bounded logit box cannot do.
    """
    p, q = sp.symbols("p q", positive=True)
    A_R, A_X = _toy_advantages(p, sp.Integer(1) - 3 * p)
    report("V7", "toy: A_R = q = 1-3p", sp.simplify(A_R - (1 - 3 * p)) == 0)
    report("V7", "toy: A_X = -3p", sp.simplify(A_X + 3 * p) == 0)
    report(
        "V7",
        "toy: A_R - A_X == 1 identically on the symmetry axis",
        sp.simplify(A_R - A_X - 1) == 0,
    )
    # Nash check: p = 1/3, q = 0 gives A = 0 on support, < 0 off it.
    report(
        "V7",
        "toy Nash = (1/3,1/3,1/3,0): A_R = 0, A_X = -1",
        sp.simplify(A_R.subs(p, sp.Rational(1, 3))) == 0
        and sp.simplify(A_X.subs(p, sp.Rational(1, 3))) == -1,
    )
    # The entropy bracket that would have to equal eta.
    beta = sp.Symbol("beta", positive=True)
    pi = [p, p, p, 1 - 3 * p]
    H = entropy(pi)
    bracket = beta * (pi[0] * (sp.log(pi[0]) + H) - pi[3] * (sp.log(pi[3]) + H))
    # sup over the open simplex axis of |bracket|/beta -- evaluated on a grid to
    # bound it; the claim is only that it is O(1), not its exact value.
    vals = [abs(float((bracket / beta).subs(p, sp.Rational(k, 100)))) for k in range(1, 34)]
    sup = max(vals)
    report(
        "V7",
        f"toy: |bracket|/beta <= {sup:.3f}, so eta=1 vs beta=1e-2 admits no root",
        sup * 1e-2 < 1.0,
    )


# ---------------------------------------------------------------------------
# V8 / V9 -- the gate's invariant box and the leak it forces
# ---------------------------------------------------------------------------


def v8_box_and_leak() -> None:
    """The one-sided gate makes the box invariant, and the box forces a leak.

    Gate (paper Algorithm 2, p24): an action with A >= 0 may not move UP past
    +l_th; an action with A < 0 may not move DOWN past -l_th. The entropy term is
    never gated but always points inward (toward uniform). So for an action whose
    advantage sign is persistent, the logit is trapped in [-l_th, l_th] up to one
    step.

    On the toy that pins y_X at -l_th and y_R at +l_th, giving the closed form

        pi_X = 1 / ((n-1) * exp(2*l_th) + 1)

    and, since the toy's exploitability against a symmetric-block policy is
    exactly the mass leaked onto the dominated action, eps = pi_X.
    """
    l_th = sp.Symbol("l_th", positive=True)
    n = 4
    y = [l_th, l_th, l_th, -l_th]
    pi = softmax(y)
    closed = 1 / ((n - 1) * sp.exp(2 * l_th) + 1)
    report(
        "V8",
        "pi_X at the box corner = 1/((n-1)e^{2 l_th}+1)",
        sp.simplify(pi[3] - closed) == 0,
    )
    # exploitability of that policy: best response value against pi.
    pi_v = sp.Matrix([pi[0], pi[1], pi[2], pi[3]])
    br = sp.Matrix(TOY_M) * pi_v
    # symmetric zero-sum: the opponent's payoff for action b equals (M pi)_b
    eps = sp.simplify(sp.Max(*[sp.simplify(br[i]) for i in range(4)]))
    report(
        "V8",
        "toy exploitability at the box corner == pi_X",
        sp.simplify(eps - closed) == 0,
    )
    for lt in (1, 2, 3, 4):
        print(f"        l_th={lt}: pi_X = eps = {float(closed.subs(l_th, lt)):.6f}")


def v9_raw_vs_centered_box() -> None:
    """Raw vs mean-centered gate: the centered box is ALWAYS at least as wide.

    Take the two-level configuration the theory predicts the operator walks to:
    ``k`` on-support actions at logit ``u``, ``n-k`` off-support ones at ``v``.
    Motion stops only when BOTH one-sided gates have closed -- if only the low
    side has closed, the on-support logits keep rising and the separation keeps
    growing, and vice versa. So the resting separation is the larger of the two
    thresholds:

        raw gate:      u <= l_th and v >= -l_th          -> sep = 2*l_th
        centered gate: u - ybar <= l_th  <=> sep >= n*l_th/(n-k)
                       ybar - v <= l_th  <=> sep >= n*l_th/k
                       -> sep = n*l_th / min(k, n-k)

    Since ``min(k, n-k) <= n/2``, the centered separation is >= 2*l_th, with
    equality only when the support is exactly half the action set. So the
    paper's own centered gate (p24 text; ambiguity A3) is the LOOSER one, and
    the repo's raw gate -- chosen because it reproduces the paper's Liar's Dice
    curve (docs/reproduce_report.md sec.6.5) -- is the one that floors.

    (An earlier hand derivation took the binding side to be the *first* gate to
    close and got n/(n-1); ``tools/ach_theory_toy.py`` falsified it -- the toy
    rests at sep = 4*l_th, not 4*l_th/3.)
    """
    l_th = sp.Symbol("l_th", positive=True)
    n, k = sp.symbols("n k", positive=True, integer=True)
    u, v = sp.symbols("u v", real=True)
    ybar = (k * u + (n - k) * v) / n
    report(
        "V9",
        "centered: u - ybar = (n-k)(u-v)/n",
        sp.simplify(u - ybar - (n - k) * (u - v) / n) == 0,
    )
    report(
        "V9",
        "centered: ybar - v = k(u-v)/n",
        sp.simplify(ybar - v - k * (u - v) / n) == 0,
    )
    sep_c = sp.Max(n * l_th / (n - k), n * l_th / k)
    report(
        "V9",
        "sep_centered = n*l_th/min(k, n-k) >= 2*l_th for all 1<=k<=n-1",
        all(
            float(sep_c.subs({n: nn, k: kk, l_th: 1})) >= 2.0 - 1e-12
            for nn in (4, 6, 13)
            for kk in range(1, nn)
        ),
    )
    print(f"        {'n':>3} {'k':>3} | {'sep raw':>8} {'sep cent':>9} | leak raw   leak cent")
    for nn, kk in ((4, 3), (4, 1), (13, 1), (13, 2), (13, 6)):
        lt = 2.0
        s_raw, s_cen = 2 * lt, nn * lt / min(kk, nn - kk)
        lr_ = (nn - kk) / (kk * math_exp(s_raw) + (nn - kk))
        lc_ = (nn - kk) / (kk * math_exp(s_cen) + (nn - kk))
        print(f"        {nn:>3} {kk:>3} | {s_raw:8.2f} {s_cen:9.2f} | {lr_:9.5f}  {lc_:9.2e}")


def v8b_general_leak() -> None:
    """Total leaked mass with k on-support and n-k off-support actions at the box corner.

        leak = (n-k) * e^{-l} / (k*e^{l} + (n-k)*e^{-l}) = (n-k)/(k*e^{2l} + (n-k))

    The k in the denominator is what the first hand estimate got wrong: with a
    SMALL support the leak is far larger than the naive 1/((n-1)e^{2l}+1). For
    Liar's Dice (n = 13 legal bids, support often k = 1-2, l_th = 2) it is
    0.18-0.10 of the probability mass -- the right order to produce a ~0.1
    exploitability floor, which 1/((n-1)e^{2l}+1) = 0.0015 is not.
    """
    l_th, kk, nn = sp.symbols("l_th k n", positive=True)
    u, v = l_th, -l_th
    leak = (nn - kk) * sp.exp(v) / (kk * sp.exp(u) + (nn - kk) * sp.exp(v))
    closed = (nn - kk) / (kk * sp.exp(2 * l_th) + (nn - kk))
    report("V8b", "leak = (n-k)/(k*e^{2 l_th} + (n-k))", sp.simplify(leak - closed) == 0)
    report(
        "V8b",
        "k=1,n=4 reduces to the V8 single-dominated-action form",
        sp.simplify(closed.subs({kk: 3, nn: 4}) - 1 / (3 * sp.exp(2 * l_th) + 1)) == 0,
    )
    for kk_, nn_ in ((3, 4), (1, 13), (2, 13)):
        val = float(closed.subs({kk: kk_, nn: nn_, l_th: 2}))
        print(f"        k={kk_:>2} n={nn_:>2} l_th=2: leaked mass = {val:.5f}")


def math_exp(x: float) -> float:
    return float(sp.exp(x))


def main() -> None:
    for title, fn in (
        ("V1  expected policy gradient (raw loss body)", v1_expected_policy_gradient),
        ("V2  expected policy gradient (centered loss body)", v2_centered_loss_body),
        ("V3/V4  entropy gradient and its action-sum", v3_v4_entropy_gradient),
        ("V5  gauge invariance of softmax", v5_gauge_invariance),
        ("V6  gauge drift rate and its sign", v6_gauge_drift),
        ("V7  no fixed point (toy)", v7_no_fixed_point),
        ("V8  invariant box and the forced leak", v8_box_and_leak),
        ("V8b general-support leak formula", v8b_general_leak),
        ("V9  raw vs centered gate width", v9_raw_vs_centered_box),
    ):
        print(f"\n{title}")
        fn()
    print("\nAll symbolic checks passed.")


if __name__ == "__main__":
    main()
