"""How far does ONE ACH update move a logit? MLP vs the tabular/exact prediction.

Both :mod:`tools.brps_operator` (exact) and :mod:`tools.brps_noise` (sampled)
move a *logit table*, so their step is the textbook one:

    dy_a = lr * eta * A / pi_old(a)                (tabular logit-space step)

The pipeline moves *parameters*, and the logit that results is
``y = W_pi @ f(theta) + b_pi``. The chain rule maps the same loss gradient into
logit space with an extra factor — the squared norm of the logit's own parameter
gradient, i.e. the network's NTK diagonal at that input:

    dy_a = lr * eta * A / pi_old(a) * ||d y_a / d theta||^2

For BRPS that factor is not a small correction. The information-state tensor is
the constant ``[0.]`` (one feature, always zero), and ``trunk_layernorm=True``
(the ``docs/reproduce_report.md`` §6.5 deviation) *fixes* the torso output to
zero mean and unit variance, so ``||f||^2 = hidden_size`` exactly, whatever the
weights do. The head contribution alone is therefore ``hidden_size + 1``.

This probe measures the factor instead of trusting the estimate: it builds the
same policy and update rule the runner builds, feeds ONE hand-made transition
with a known advantage, and reads the realized ``dy``. Arms sweep the two knobs
the estimate says own it (``hidden_sizes``, ``trunk_layernorm``) plus the two the
config sets (``learning_rate``, advantage magnitude).

Read it against BRPS's payoff scale: an advantage of 25 (Rock vs Paper) with
``pi_old = 1/3`` and ``lr = 1e-3`` predicts a tabular step of 0.075 — and an MLP
step ~129x that, which is a jump straight past the ``l_th = 2`` gate box in a
single update. Entropy and the critic are off in this probe by construction; it
measures the policy term's transfer function, nothing else.

Not on the ``mjai`` import path (an analysis tool). CPU-only.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from mjai.agents.mlp import ACTIVATIONS, MLPSharedActorCritic
from mjai.algos.nn_updates import NNActorCriticUpdate
from mjai.algos.transition import Batch
from mjai.algos.update_rule import AlgoConfig
from mjai.utils.gpu_assert import require_cpu

OBS_SIZE = 1  # BRPS information_state_tensor is [0.]
NUM_ACTIONS = 3


def one_sample_batch(action: int, advantage: float, logprob: float) -> Batch:
    """A single BRPS transition with a chosen advantage, at obs = [0.]."""
    return Batch(
        obs=np.zeros((1, OBS_SIZE), dtype=np.float32),
        legal_actions=[list(range(NUM_ACTIONS))],
        actions=np.array([action], dtype=np.int64),
        logprobs=np.array([logprob], dtype=np.float32),
        values=np.zeros(1, dtype=np.float32),
        returns=np.array([advantage], dtype=np.float32),
        advantages=np.array([advantage], dtype=np.float32),
        legal_mask=np.ones((1, NUM_ACTIONS), dtype=bool),
        players=np.zeros(1, dtype=np.int8),
        num_actions=NUM_ACTIONS,
    )


def logits_of(policy: MLPSharedActorCritic) -> np.ndarray:
    with torch.no_grad():
        out = policy.action_logits_batch(
            np.zeros((1, OBS_SIZE), dtype=np.float32),
            np.ones((1, NUM_ACTIONS), dtype=bool),
        )
    return np.asarray(out, dtype=np.float64).reshape(-1)


def feature_norm_sq(policy: MLPSharedActorCritic) -> float:
    """``||f(theta)||^2`` at obs = [0.] — the head's share of the NTK diagonal."""
    with torch.no_grad():
        x = torch.zeros(1, OBS_SIZE, dtype=torch.float32, device=policy.device)
        f = policy.torso(x)
    return float((f * f).sum().item())


def ntk_diag(policy: MLPSharedActorCritic, action: int) -> float:
    """``||d y_action / d theta||^2`` over ALL parameters (autograd, exact)."""
    x = torch.zeros(1, OBS_SIZE, dtype=torch.float32, device=policy.device)
    y = policy.policy_head(policy.torso(x))[0, action]
    # The value head does not feed this logit, hence allow_unused (its grad is None).
    grads = torch.autograd.grad(
        y, [p for p in policy.parameters() if p.requires_grad], allow_unused=True
    )
    return float(sum((g * g).sum().item() for g in grads if g is not None))


def measure(
    *,
    hidden_sizes: tuple[int, ...],
    trunk_layernorm: bool,
    lr: float,
    advantage: float,
    l_th: float,
    seed: int = 0,
) -> dict[str, float]:
    """One update from a fresh policy; return the realized vs predicted logit step."""
    require_cpu()
    policy = MLPSharedActorCritic(
        OBS_SIZE,
        NUM_ACTIONS,
        hidden_sizes=hidden_sizes,
        activation=ACTIVATIONS["relu"],
        trunk_layernorm=trunk_layernorm,
        device="cpu",
        seed=seed,
    )
    # Entropy and critic off: this probe isolates the POLICY term's transfer.
    config = AlgoConfig(
        learning_rate=lr,
        optimizer="sgd",
        entropy_coef=0.0,
        value_coef=0.0,
        eta=1.0,
        l_th=l_th,
        theta=1.0,
        max_grad_norm=0.0,
    )
    rule = NNActorCriticUpdate(policy, config)
    y0 = logits_of(policy)
    pi0 = np.exp(y0 - y0.max())
    pi0 /= pi0.sum()
    action = int(np.argmin(pi0))  # the rarest action: where 1/pi_old bites hardest
    ntk = ntk_diag(policy, action)
    rule.step(one_sample_batch(action, advantage, float(np.log(pi0[action]))))
    y1 = logits_of(policy)
    dy = float(y1[action] - y0[action])
    tabular = lr * advantage / pi0[action]
    return {
        "hidden": float(hidden_sizes[0] if hidden_sizes else 0),
        "layernorm": float(trunk_layernorm),
        "lr": lr,
        "advantage": advantage,
        "pi_old": float(pi0[action]),
        "feat_norm_sq": feature_norm_sq(policy),
        "ntk_diag": ntk,
        "dy_tabular": tabular,
        "dy_mlp": dy,
        "amplification": dy / tabular if tabular else float("nan"),
        "y_after": float(y1[action]),
        "spread_after": float(y1.max() - y1.min()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--advantage", type=float, default=25.0, help="BRPS Rock-vs-Paper is 25")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l-th", type=float, default=2.0)
    ap.add_argument("--hidden", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    args = ap.parse_args()

    hdr = (
        f"{'arm':<22}{'pi_old':>8}{'||f||^2':>9}{'NTK':>9}"
        f"{'dy_tabular':>12}{'dy_mlp':>10}{'amplif':>9}{'y_after':>9}"
    )
    print(f"one ACH update, advantage={args.advantage}, lr={args.lr}, l_th={args.l_th}")
    print(hdr)
    rows = []
    for h in args.hidden:
        for ln in (True, False):
            r = measure(
                hidden_sizes=(h,),
                trunk_layernorm=ln,
                lr=args.lr,
                advantage=args.advantage,
                l_th=args.l_th,
            )
            rows.append(r)
            name = f"h={h} ln={'on' if ln else 'off'}"
            print(
                f"{name:<22}{r['pi_old']:>8.4f}{r['feat_norm_sq']:>9.2f}{r['ntk_diag']:>9.2f}"
                f"{r['dy_tabular']:>12.4f}{r['dy_mlp']:>10.4f}{r['amplification']:>9.2f}"
                f"{r['y_after']:>9.3f}"
            )
    # The config's own arm, plus the lr that would restore the tabular step.
    base = next(r for r in rows if r["hidden"] == 128.0 and r["layernorm"] == 1.0)
    print(
        f"\nBRPS config arm (h=128, LayerNorm on): amplification {base['amplification']:.1f}x\n"
        f"  => effective logit-space lr = {args.lr * base['amplification']:.4g} "
        f"(config lr {args.lr}); the lr that matches the tabular step is "
        f"{args.lr / base['amplification']:.3g}"
    )


if __name__ == "__main__":
    main()
