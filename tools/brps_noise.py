"""Sampled ACH on BRPS's three logits — the rung between the exact operator and RL.

:mod:`tools.brps_operator` iterates the *expected* ACH update. This file iterates
the update the pipeline actually applies: a batch of ``target_samples``
transitions, each contributing ``-eta * y(a) * c * A(a) / pi_old(a)`` (Eq. 29
p24), with the ``1/pi_old`` that cancels only *in expectation*. Everything else
is held identical to the exact rung — three tabular logits, no MLP, no LayerNorm,
optionally no critic — so whatever differs between the two rungs is owned by
**sampling variance alone**.

The batch is built the way :class:`~mjai.pipeline.rollout.RolloutWorkerCore`
builds it on BRPS under mirror self-play: each episode is one simultaneous node
and contributes **two** transitions (both seats, same shared policy, pooled —
``rollout.py`` docstring on ``learner_player``), and with ``gamma=1`` and a
one-step episode GAE collapses to ``A = r - V``. So ``target_samples=64`` means
32 episodes, and one update here is one update there.

Two critic modes, because "the critic is bad" is a competing explanation that has
to be killed rather than assumed away:

  ``oracle``  ``V = 0``, which on BRPS is the *exact* state value at every policy
              (antisymmetric payoffs, shared policy) — an infinitely good critic.
  ``scalar``  a single learned scalar trained by the same SGD step on the same
              MSE the pipeline uses, i.e. what the value head can express when
              the information-state tensor is the constant ``[0.]``.

Metric conventions and the ``NashConv = 2 * exploitability`` reading come from
:mod:`tools.brps_operator`. Float64 throughout (AGENTS.md D19); not on the
``mjai`` import path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tools.brps_operator import (
    entropy_grad,
    exploitability,
    payoff_matrix,
    softmax,
    tv_to_nash,
)


@dataclass(frozen=True)
class NoiseParams:
    """Defaults mirror configs/exp/brps_ach_mlp_mirror.yaml."""

    lr: float = 1e-3
    eta: float = 1.0
    beta: float = 1e-2
    l_th: float = 2.0
    gate: bool = True
    batch: int = 64  # counted transitions per update (target_samples)
    critic: str = "oracle"  # "oracle" (V=0, exact) or "scalar" (learned)
    value_coef: float = 1.0
    normalize_advantages: bool = False
    iw_clip: float | None = None
    payoff_scale: float = 1.0


def _sample_batch(
    pi: np.ndarray, m: np.ndarray, batch: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """One collection round: return (actions, rewards) over both seats' transitions.

    ``batch`` counted transitions = ``batch // 2`` episodes x 2 seats, which is
    what ``target_samples`` delivers on a one-node simultaneous game.
    """
    n_ep = max(1, batch // 2)
    a0 = rng.choice(3, size=n_ep, p=pi)
    a1 = rng.choice(3, size=n_ep, p=pi)
    r0 = m[a0, a1]
    actions = np.concatenate([a0, a1])
    rewards = np.concatenate([r0, -r0])  # antisymmetric: seat 1's payoff is -seat 0's
    return actions, rewards


def sampled_step(
    y: np.ndarray,
    v: float,
    m: np.ndarray,
    params: NoiseParams,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """One optimizer step on (logits, scalar critic) from one sampled batch."""
    pi = softmax(y)
    actions, rewards = _sample_batch(pi, m, params.batch, rng)
    adv = rewards - v
    if params.normalize_advantages and adv.size > 1 and adv.std() > 1e-8:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    # Per-sample gate: sign of THIS sample's advantage, current raw logit.
    y_a = y[actions]
    if params.gate:
        c = np.where(adv >= 0.0, y_a < params.l_th, y_a > -params.l_th).astype(np.float64)
    else:
        c = np.ones_like(adv)
    denom = pi[actions]
    if params.iw_clip is not None:
        denom = np.maximum(denom, 1.0 / params.iw_clip)
    # d/dy_j of -mean_i(eta * y(a_i) * c_i * A_i / pi_old(a_i)): scatter-add.
    per_sample = params.eta * c * adv / denom
    grad_pol = np.zeros(3, dtype=np.float64)
    np.add.at(grad_pol, actions, -per_sample / actions.size)
    grad_ent = params.beta * entropy_grad(pi)  # +beta*pi*(log pi + H)
    y_new = y - params.lr * (grad_pol + grad_ent)
    if params.critic == "scalar":
        v_new = v - params.lr * params.value_coef * 2.0 * float((v - rewards).mean())
    elif params.critic == "oracle":
        v_new = 0.0
    else:
        raise ValueError(f"unknown critic mode {params.critic!r} (want 'oracle' | 'scalar')")
    stats = {
        "gate_off_frac": float(1.0 - c.mean()),
        "iw_max": float(1.0 / denom.min()),
        "grad_pol_norm": float(np.linalg.norm(grad_pol)),
        "adv_std": float(adv.std()),
    }
    return y_new, v_new, stats


def run(
    params: NoiseParams,
    updates: int,
    seed: int,
    checkpoints: tuple[int, ...] = (4_688, 15_625, 156_250),
) -> dict[str, object]:
    """Iterate the sampled update; report the envelope, not just the endpoint.

    ``checkpoints`` default to the update counts of 3e5 / 1e6 / 1e7 env-steps at
    batch 64, so one run reads out at every RL budget we compare against. The
    envelope (mean/max over the window ending at each checkpoint) is the honest
    statistic for a process that orbits: an endpoint sample of an orbit says
    nothing about whether it is shrinking.
    """
    m_true = payoff_matrix()
    m = m_true / params.payoff_scale
    rng = np.random.default_rng(seed)
    y = np.zeros(3, dtype=np.float64)
    v = 0.0
    rows: list[dict[str, float]] = []
    window_start = 0
    buf: list[tuple[float, float, float, float]] = []
    collapse_updates = 0
    diverged_at: int | None = None
    # Running (uniform) average policy: ACH's O(T^-1/2) guarantee is about the
    # AVERAGE strategy, not the current one (AGENTS.md D16), so a run whose last
    # iterate orbits can still be converging in the sense the theorem states.
    # Tracked here because the surrogate is where it costs nothing; the pipeline
    # only has the tracker enabled on Kuhn.
    pi_sum = np.zeros(3, dtype=np.float64)
    for t in range(updates):
        if not np.isfinite(y).all():
            # The same death the pipeline dies (mjai.agents.nonfinite:
            # NonFiniteNetworkError, formerly a bare torch.multinomial error).
            # Reported rather than raised: a sweep wants "this arm diverged at
            # update N", not a traceback that kills the other arms.
            diverged_at = t
            break
        pi = softmax(y)
        expl = exploitability(m_true, pi)
        buf.append((expl, tv_to_nash(pi), float(y.max() - y.min()), float(pi.max())))
        collapse_updates += int(pi.max() > 0.9)
        pi_sum += pi
        y, v, stats = sampled_step(y, v, m, params, rng)
        if (t + 1) in checkpoints:
            arr = np.asarray(buf[window_start:])
            pi_avg = pi_sum / (t + 1)
            rows.append(
                {
                    "updates": float(t + 1),
                    "env_steps": float((t + 1) * params.batch),
                    "expl_mean": float(arr[:, 0].mean()),
                    "expl_max": float(arr[:, 0].max()),
                    "tv_mean": float(arr[:, 1].mean()),
                    "spread_max": float(arr[:, 2].max()),
                    "pi_max_mean": float(arr[:, 3].mean()),
                    "collapse_frac": float((arr[:, 3] > 0.9).mean()),
                    "gate_off_frac": stats["gate_off_frac"],
                    "iw_max": stats["iw_max"],
                    "adv_std": stats["adv_std"],
                    # Average policy over ALL updates so far (not the window).
                    "expl_avg_policy": exploitability(m_true, pi_avg),
                    "tv_avg_policy": tv_to_nash(pi_avg),
                }
            )
            window_start = len(buf)
    return {
        "params": asdict(params),
        "seed": seed,
        "updates": updates,
        "rows": rows,
        "final_pi": softmax(y).tolist() if diverged_at is None else [float("nan")] * 3,
        "collapse_frac_all": collapse_updates / max(1, updates),
        "diverged_at": diverged_at,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--updates", type=int, default=156_250, help="1e7 env-steps at batch 64")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1e-2)
    ap.add_argument("--l-th", type=float, default=2.0)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--critic", default="oracle", choices=("oracle", "scalar"))
    ap.add_argument("--normalize-advantages", action="store_true")
    ap.add_argument("--iw-clip", type=float, default=None)
    ap.add_argument("--payoff-scale", type=float, default=1.0)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    params = NoiseParams(
        lr=args.lr,
        eta=args.eta,
        beta=args.beta,
        l_th=args.l_th,
        gate=not args.no_gate,
        batch=args.batch,
        critic=args.critic,
        normalize_advantages=args.normalize_advantages,
        iw_clip=args.iw_clip,
        payoff_scale=args.payoff_scale,
    )
    print(f"sampled ACH on BRPS: {params}")
    results = []
    for seed in args.seeds:
        res = run(params, args.updates, seed)
        results.append(res)
        for row in res["rows"]:  # type: ignore[union-attr]
            print(
                f"  seed {seed}  {row['env_steps']:.1e} steps  "
                f"expl mean {row['expl_mean']:8.4f} max {row['expl_max']:8.3f}  "
                f"tv {row['tv_mean']:.4f}  spread_max {row['spread_max']:6.2f}  "
                f"pi_max {row['pi_max_mean']:.3f}  collapse {row['collapse_frac']:.2f}  "
                f"gate_off {row['gate_off_frac']:.2f}  iw_max {row['iw_max']:.1f}"
            )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
