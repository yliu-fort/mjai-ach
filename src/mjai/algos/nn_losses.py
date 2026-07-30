"""Pure loss pieces for the theta-parameterized NN actor-critic update.

Stateless functions only — no optimizer, no policy, no side effects. They are
the two policy-improvement operators the ``theta`` knob interpolates between,
plus the pieces both share:

  - :func:`ppo_policy_loss` — clipped surrogate (``theta = 0`` endpoint).
  - :func:`ach_policy_loss` — paper-faithful ACH (``theta = 1`` endpoint):
    logit-space policy gradient with the advantage-sign-dependent one-sided
    logit gate (Fu et al., ICLR 2022, OpenReview ``DTXZqTNV5nW``, Algorithm 2 /
    Eq. 29 p24). Ground truth for every fidelity decision:
    ``docs/paper_spec_ach.md``.
  - :func:`value_loss_and_entropy` — the critic MSE + entropy term, identical
    in both (paper Eq. 29 and PPO agree on their form), so they are never
    interpolated.

Keeping them here means there is exactly ONE implementation of the ACH math in
the repo (AGENTS.md §1 D4) even though it is reachable at every theta.

Each loss returns ``(loss, telemetry)`` where telemetry maps a scalar name to a
0-dim tensor; the caller stacks them for a single device sync (AGENTS.md §8).
"""

from __future__ import annotations

import torch

from mjai.agents.mlp import MASK_VALUE
from mjai.algos.update_rule import AlgoConfig


def explained_variance(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Explained variance: 1 - Var(y_true - y_pred) / Var(y_true); 1.0 = perfect.

    Computed entirely on-device to avoid the two full-array D2H transfers a
    numpy variant would pay for (AGENTS.md §8). The trailing ``.item()`` is a
    single-scalar sync. Degenerate (single-sample or constant) batches
    report 0.0 — variance is undefined there.
    """
    if y_true.numel() <= 1:
        return 0.0
    yt_var = y_true.var()
    if yt_var.item() < 1e-12:
        return 0.0
    return float(1.0 - (y_true - y_pred).var() / yt_var)


def weighted_mean(x: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    """``x.mean()``, or the self-normalized weighted mean when weights are given.

    ``weights=None`` takes the *same* call the unweighted path always took, so
    the default trajectory stays bit-identical rather than merely
    arithmetically equal (``tests/unit/data/nn_updates_golden.json`` pins it).

    Self-normalized (divide by ``sum(w)``, not by ``len(x)``) on purpose: the
    weight ``reach(h)^-kappa`` has an arbitrary scale that grows with kappa, and
    dividing by the batch size would push that scale straight into the effective
    learning rate — turning a re-weighting experiment into a learning-rate
    experiment. The price is the usual self-normalized-importance-sampling
    bias, which vanishes as the batch grows; the mechanism variable it trades
    against (per-batch effective sample size) is logged as ``weight_effn``.
    """
    if weights is None:
        return x.mean()
    return (weights * x).sum() / weights.sum()


def normalize_advantages(adv: torch.Tensor) -> torch.Tensor:
    """Per-batch zero-mean/unit-std normalization (a PPO best practice).

    Off by default: the ACH paper's loss consumes raw GAE advantages (p24), and
    normalizing them would turn the hedge coefficient eta into a batch-adaptive
    learning rate. Enabled via ``AlgoConfig.normalize_advantages``, which warns
    when the ACH term carries any weight.
    """
    if adv.numel() <= 1:
        return adv
    std = adv.std()
    if std.item() < 1e-8:
        return adv - adv.mean()
    return (adv - adv.mean()) / (std + 1e-8)


def masked_log_probs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax over legal actions only (illegal logits pushed to MASK_VALUE)."""
    masked = logits + (1.0 - mask) * MASK_VALUE
    return torch.log_softmax(masked, dim=-1)


def value_loss_and_entropy(
    logits: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (value_mse, mean_entropy) for the combined loss.

    Paper Eq. 29 (p24): ``alpha/2 * [V(s;omega) - G]^2 + beta * sum_a pi(a|s)
    log pi(a|s)``. ``value_coef`` multiplies the plain MSE, so the paper's
    alpha=2.0 (p27 Table 7) corresponds to ``value_coef=1.0``. The entropy term
    enters the total loss as ``-entropy_coef * entropy`` (= ``+beta * sum_a pi
    log pi``), with beta=1e-2 (p28 Table 8).

    ``weights`` re-weights the critic and the entropy term alongside the policy
    term. That is the whole point of the knob rather than an afterthought: the
    critic is fit on the same concentrated visitation the policy is, so
    tempering only the policy would aim it at information sets whose advantages
    come from a critic that never saw them.
    """
    value_loss = weighted_mean((values - returns) ** 2, weights)
    logp_all = masked_log_probs(logits, mask)
    legal_probs = torch.exp(logp_all)
    entropy = -weighted_mean((legal_probs * logp_all).sum(dim=-1), weights)
    return value_loss, entropy


def ppo_policy_loss(
    *,
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PPO clipped surrogate — the ``theta = 0`` endpoint.

    Args:
        ratio: ``pi(a|s;theta) / pi_old(a|s)`` per sample.
        advantages: advantages for THIS term (normalized when
            ``AlgoConfig.normalize_advantages`` is set; the ACH term always
            keeps the raw ones).
        clip_eps: surrogate clip range (37-details default 0.2).
        weights: per-sample loss weights, or None for the plain mean.
    """
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -weighted_mean(torch.min(surr1, surr2), weights)
    with torch.no_grad():
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean()
    return loss, {"clip_frac": clip_frac}


def ach_policy_loss(
    *,
    logits: torch.Tensor,
    mask: torch.Tensor,
    actions: torch.Tensor,
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    old_probs: torch.Tensor,
    config: AlgoConfig,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Paper-faithful ACH policy loss — the ``theta = 1`` endpoint.

    Implements Algorithm 2 / Eq. 29 (p24) exactly — per sample
    ``[a, s, A(s,a), G, pi_old(a|s)]``::

        c = 1{pi(a|s;theta)/pi_old < 1+eps} * 1{y(a)-y_mean <  l_th}   if A >= 0
        c = 1{pi(a|s;theta)/pi_old > 1-eps} * 1{y(a)-y_mean > -l_th}   if A < 0
        L = -c * eta * y(a|s;theta) / pi_old(a|s) * A

    Fidelity notes (ground truth: docs/paper_spec_ach.md):

      - **Advantage-sign-dependent one-sided gate**: the gate only blocks
        *further* movement past the threshold in the direction the advantage
        pushes; the corrective direction is always allowed. A symmetric
        |y|<=l_th gate would also zero the gradient that pulls an overshot
        logit back — a different algorithm (see docs/audit_report.md F1).
      - **Ambiguity A3 (gate centered; loss body toggleable)**: the paper is
        explicit that the gate thresholds the mean-centered logit ``y - y_mean``
        ("the mean is subtracted from the policy output", p24). Both the gate
        and the loss body default to the RAW logit here, which is coherent
        because the MLP's trunk LayerNorm bounds the feature scale feeding the
        heads — that combination is what reproduces the paper's Liar's Dice
        curve (docs/reproduce_report.md §6.5). ``gate_centered_logits`` /
        ``loss_centered_logits`` restore the pre-LayerNorm reading.
      - **No advantage normalization** (the paper has none; eta=1.0 is the
        hedge learning rate, p27 Table 7).
      - **Ratio gate kept but vacuous** under synchronous single-threaded
        self-play, where pi == pi_old (p28 note); it only bites under async
        IMPALA-style sampling.
      - **Centered-mean action set toggleable** (A5-adjacent ambiguity): by
        default ``y_mean`` averages ALL logits; with
        ``centered_mean_legal_only=True`` it averages legal-action logits only,
        shielding the gate and the centered loss body from illegal-logit drift
        (matters when legal sets shrink, e.g. Liar's Dice).

    Telemetry: ``gate_off_frac`` plus the unbounded-importance-weight probes
    ``iw_max`` / ``iw_mean`` / ``pterm_max`` (nothing bounds ``1/pi_old`` under
    synchronous self-play, so these detect gradient blow-up driven by rare
    sampled actions).
    """
    # Gate logit: which actions enter y_mean is ambiguous (masking is never
    # discussed, A5) — all actions (historical default) or legal-only.
    if config.centered_mean_legal_only:
        n_legal = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        y_mean = (logits * mask).sum(dim=-1, keepdim=True) / n_legal
    else:
        y_mean = logits.mean(dim=-1, keepdim=True)
    centered = logits - y_mean
    y_gate_src = centered if config.gate_centered_logits else logits
    y_gate = y_gate_src.gather(1, actions.unsqueeze(1)).squeeze(1)
    y_loss_src = centered if config.loss_centered_logits else logits
    y_loss = y_loss_src.gather(1, actions.unsqueeze(1)).squeeze(1)
    # Advantage-sign-dependent one-sided gates (p24 Algorithm 2).
    gate_pos = (y_gate < config.l_th) & (ratio < 1.0 + config.ratio_eps)
    gate_neg = (y_gate > -config.l_th) & (ratio > 1.0 - config.ratio_eps)
    c = torch.where(advantages >= 0, gate_pos, gate_neg).float()
    denom = old_probs + 1e-8
    if config.iw_clip is not None:
        # Cap 1/pi_old at iw_clip (floor pi_old at 1/iw_clip). Bounds the
        # importance weight that otherwise blows up once the policy sharpens.
        denom = denom.clamp(min=1.0 / config.iw_clip)
    loss = -weighted_mean(config.eta * y_loss * c * advantages / denom, weights)

    with torch.no_grad():
        iw = 1.0 / denom
        pterm = (config.eta * y_loss * c * advantages / denom).abs()
        telemetry = {
            "gate_off_frac": 1.0 - c.mean(),
            "iw_max": iw.max(),
            "iw_mean": iw.mean(),
            "pterm_max": pterm.max(),
        }
    return loss, telemetry


def weight_telemetry(weights: torch.Tensor | None) -> dict[str, torch.Tensor]:
    """Dispersion of the per-sample weights — the cost side of the tempering.

    ``weight_effn`` is Kish's effective sample size ``(sum w)^2 / sum w^2``: how
    many samples this batch is really worth once the weights are applied. It is
    the counterweight to the coverage the weighting buys, and the number to read
    when a weighted arm underperforms — a batch of 64 with effN 3 has not been
    re-weighted so much as thrown away. Empty when unweighted, so the metric
    only appears on runs where it means something.
    """
    if weights is None:
        return {}
    with torch.no_grad():
        total = weights.sum()
        return {
            "weight_effn": total * total / (weights * weights).sum(),
            "weight_max_ratio": weights.max() / weights.min().clamp(min=1e-30),
        }


__all__ = [
    "ach_policy_loss",
    "explained_variance",
    "masked_log_probs",
    "normalize_advantages",
    "ppo_policy_loss",
    "value_loss_and_entropy",
    "weight_telemetry",
    "weighted_mean",
]
