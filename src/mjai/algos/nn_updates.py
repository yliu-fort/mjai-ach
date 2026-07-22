"""Neural UpdateRules: paper-faithful ACH and a reference PPO on torch policies.

These drive the NN half of the 2x2 matrix (AGENTS.md §1 D1) on
:class:`mjai.agents.mlp.MLPSharedActorCritic`. The two endpoints share only
semantics-free scaffolding (tensor conversion, masked log-probs, value/entropy
loss pieces, grad clipping, optimizer state I/O) via :class:`_NNUpdateBase`;
each owns its optimizer and its ``step()``.

  - :class:`NNACHUpdate` — the single ACH implementation (AGENTS.md §1 D4):
    paper-faithful per Fu et al., ICLR 2022 (OpenReview ``DTXZqTNV5nW``),
    Algorithm 2 + Eq. 29 (p24) with Appendix H.3 hyperparameters (p27-28).
    Ground truth for every fidelity decision: ``docs/paper_spec_ach.md``.
  - :class:`NNPPOUpdate` — reference PPO (clipped surrogate + value MSE +
    entropy bonus) for the PPO-vs-ACH comparison.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Any

import torch
from torch import nn

from mjai.agents.mlp import MASK_VALUE, MLPSharedActorCritic
from mjai.algos.transition import Batch, UpdateStats
from mjai.algos.update_rule import AlgoConfig, UpdateRule


def _explained_variance_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
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


def _normalize_advantages(adv: torch.Tensor) -> torch.Tensor:
    """Per-batch zero-mean/unit-std normalization (PPO only — ACH has none).

    The ACH paper's loss consumes raw GAE advantages (p24); normalizing them
    would turn the hedge coefficient eta into a batch-adaptive learning rate.
    """
    if adv.numel() <= 1:
        return adv
    std = adv.std()
    if std.item() < 1e-8:
        return adv - adv.mean()
    return (adv - adv.mean()) / (std + 1e-8)


class _NNUpdateBase(UpdateRule):
    """Semantics-free scaffolding shared by the PPO and ACH endpoints.

    Owns: optimizer construction (delegated to the endpoint), batch-to-tensor
    conversion, masked log-probs, the shared value-MSE + entropy-bonus loss
    pieces, grad-norm clipping, and optimizer state I/O. Owns NO algorithm
    semantics — the policy-improvement operator lives in each endpoint's
    ``step()``.
    """

    def __init__(self, policy: MLPSharedActorCritic, config: AlgoConfig | None = None) -> None:
        if not isinstance(policy, MLPSharedActorCritic):
            raise TypeError(
                f"{type(self).__name__} requires MLPSharedActorCritic, got {type(policy)}"
            )
        super().__init__(policy)
        self.config = config or AlgoConfig()
        self.policy: MLPSharedActorCritic = policy
        self.optimizer = self._make_optimizer()

    @abstractmethod
    def _make_optimizer(self) -> torch.optim.Optimizer:
        """Build the endpoint's optimizer (PPO: Adam; ACH: SGD, paper p27)."""
        ...

    # ---- batch-to-tensor helpers ----

    def _obs_tensor(self, batch: Batch) -> torch.Tensor:
        return torch.as_tensor(batch.obs, dtype=torch.float32, device=self.policy.device)

    def _legal_mask_tensor(self, batch: Batch) -> torch.Tensor:
        return torch.as_tensor(batch.legal_mask, dtype=torch.float32, device=self.policy.device)

    def _masked_logp(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = logits + (1.0 - mask) * MASK_VALUE
        return torch.log_softmax(masked, dim=-1)

    # ---- shared loss pieces (identical in the paper's Eq. 29 and in PPO) ----

    def _value_and_entropy(
        self, logits: torch.Tensor, values: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (value_mse, mean_entropy) for the combined loss.

        Paper Eq. 29 (p24): ``alpha/2 * [V(s;omega) - G]^2 + beta * sum_a pi(a|s)
        log pi(a|s)``. Our ``value_coef`` multiplies the plain MSE, so the
        paper's alpha=2.0 (p27 Table 7) corresponds to ``value_coef=1.0``. The
        entropy term enters the total loss as ``-entropy_coef * entropy``
        (= ``+beta * sum_a pi log pi``), with beta=1e-2 (p28 Table 8).
        """
        value_loss = ((values - returns) ** 2).mean()
        logp_all = self._masked_logp(logits, mask)
        legal_probs = torch.exp(logp_all)
        entropy = -(legal_probs * logp_all).sum(dim=-1).mean()
        return value_loss, entropy

    def _clip_grads(self) -> None:
        """Grad-norm clipping; disabled when ``max_grad_norm <= 0``.

        The ACH paper mentions no grad clipping, so the reproduction configs
        set ``max_grad_norm: 0.0``; PPO keeps the 37-details default 0.5.
        """
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)

    # ---- optimizer state I/O ----

    def state_dict(self) -> dict[str, object]:
        return {"optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if "optimizer" in state:
            # The optimizer entry is a dict[str, Any] produced by torch; cast
            # through Any since the value is typed as `object` at the boundary.
            opt_state: dict[str, Any] = state["optimizer"]  # type: ignore[assignment]
            self.optimizer.load_state_dict(opt_state)


class NNPPOUpdate(_NNUpdateBase):
    """Reference PPO endpoint: clipped surrogate + value MSE + entropy bonus.

    Intentional deviations from Huang et al. 2022 ("The 37 Implementation
    Details"), kept so PPO shares the ACH paper's single-update-per-batch
    regime and the comparison isolates the policy-improvement operator:

      - exactly ONE full-batch gradient step per batch (no minibatch shuffling,
        no multi-epoch sample reuse, hence no KL early-stop);
      - constant learning rate (no annealing);
      - PyTorch default weight init (no orthogonal init).

    Adam follows the 37-details eps recommendation (1e-5). Advantage
    normalization, clip_eps=0.2, action masking, and grad clipping (default
    0.5) are standard.
    """

    def __init__(
        self,
        policy: MLPSharedActorCritic,
        config: AlgoConfig | None = None,
        *,
        clip_eps: float = 0.2,
    ) -> None:
        self.clip_eps = float(clip_eps)
        super().__init__(policy, config)

    def _make_optimizer(self) -> torch.optim.Optimizer:
        opt = self.config.optimizer or "adam"
        if opt == "adam":
            return torch.optim.Adam(
                self.policy.parameters(), lr=self.config.learning_rate, eps=1e-5
            )
        if opt == "sgd":
            return torch.optim.SGD(self.policy.parameters(), lr=self.config.learning_rate)
        raise ValueError(f"Unknown optimizer {opt!r}; expected 'adam' or 'sgd'")

    def step(self, batch: Batch) -> UpdateStats:
        if batch.size == 0:
            return UpdateStats(policy_loss=0.0, value_loss=0.0, entropy=0.0)
        obs = self._obs_tensor(batch)
        mask = self._legal_mask_tensor(batch)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.policy.device)
        old_logp = torch.as_tensor(batch.logprobs, dtype=torch.float32, device=self.policy.device)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.policy.device)
        adv = _normalize_advantages(
            torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.policy.device)
        )

        # One full-batch gradient step (intentional; see class docstring).
        logits, values = self.policy(obs)
        logp_all = self._masked_logp(logits, mask)
        new_logp = logp_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        ratio = torch.exp(new_logp - old_logp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
        ppo_loss = -torch.min(surr1, surr2).mean()
        value_loss, entropy = self._value_and_entropy(logits, values, returns, mask)
        loss = ppo_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]  # torch's Tensor.backward is untyped
        self._clip_grads()
        self.optimizer.step()

        with torch.no_grad():
            clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean()
            stats_t = torch.stack(
                [ppo_loss.detach(), value_loss.detach(), entropy.detach(), clip_frac]
            )
        total_pol, total_val, total_ent, total_clip = stats_t.cpu().tolist()
        approx_kl = float((old_logp - new_logp.detach()).mean().item())
        ev = _explained_variance_torch(returns, values.detach())
        return UpdateStats(
            policy_loss=total_pol,
            value_loss=total_val,
            entropy=total_ent,
            approx_kl=approx_kl,
            clip_frac=total_clip,
            explained_variance=ev,
        )


class NNACHUpdate(_NNUpdateBase):
    """Paper-faithful ACH (Fu et al., ICLR 2022, OpenReview ``DTXZqTNV5nW``).

    Implements Algorithm 2 / Eq. 29 (p24) exactly -- per sample
    ``[a, s, A(s,a), G, pi_old(a|s)]``::

        c = 1{pi(a|s;theta)/pi_old < 1+eps} * 1{y(a)-y_mean <  l_th}   if A >= 0
        c = 1{pi(a|s;theta)/pi_old > 1-eps} * 1{y(a)-y_mean > -l_th}   if A < 0
        L = -c * eta * y(a|s;theta) / pi_old(a|s) * A
            + alpha/2 * (V-G)^2 + beta * sum_a pi log pi

    Fidelity notes (ground truth: docs/paper_spec_ach.md):

      - **Advantage-sign-dependent one-sided gate**: the gate only blocks
        *further* movement past the threshold in the direction the advantage
        pushes; the corrective direction is always allowed. A symmetric
        |y|<=l_th gate would also zero the gradient that pulls an overshot
        logit back -- a different algorithm (see docs/audit_report.md F1).
      - **Ambiguity A3 (gate centered; loss body toggleable)**: the gate always
        thresholds the mean-centered logit ``y - y_mean`` (paper is explicit:
        "the mean is subtracted from the policy output", p24). The loss body
        uses the centered logit by default (paper text); set
        ``AlgoConfig.loss_centered_logits=False`` for the literal Algorithm 2
        raw-logit reading. Mean-subtraction leaves softmax unchanged but
        spreads the gradient across actions (``g_a - mean_b g_b``).
      - **No advantage normalization** (the paper has none; eta=1.0 is the
        hedge learning rate, p27 Table 7).
      - **SGD with constant LR** (H.3, p27: "stochastic gradient descent with
        a constant learning rate", best lr=1e-3, no momentum mentioned).
      - **Ratio gate kept but vacuous** under synchronous single-threaded
        self-play, where pi == pi_old (p28 note); it only bites under async
        IMPALA-style sampling.
      - **One gradient step per mini-batch** (p24: "we update theta and omega
        once using a single mini-batch at each iteration").
    """

    def _make_optimizer(self) -> torch.optim.Optimizer:
        opt = self.config.optimizer or "sgd"
        if opt != "sgd":
            raise ValueError(
                f"Paper-faithful ACH uses SGD with a constant LR (p27 H.3); got {opt!r}. "
                "Set optimizer='sgd' in AlgoConfig/YAML."
            )
        return torch.optim.SGD(self.policy.parameters(), lr=self.config.learning_rate)

    def step(self, batch: Batch) -> UpdateStats:
        if batch.size == 0:
            return UpdateStats(policy_loss=0.0, value_loss=0.0, entropy=0.0)
        obs = self._obs_tensor(batch)
        mask = self._legal_mask_tensor(batch)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.policy.device)
        old_logp = torch.as_tensor(batch.logprobs, dtype=torch.float32, device=self.policy.device)
        # pi_old(a|s): the behavior-policy prob recorded by the rollout under the
        # exact policy that sampled the action (paper: samples arrive as
        # [a, s, A, G, pi_old(a|s)], p24 Algorithm 2).
        old_probs = torch.exp(old_logp)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.policy.device)
        # Raw GAE advantages — NO normalization (paper p24; see class docstring).
        adv = torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.policy.device)

        logits, values = self.policy(obs)
        logp_all = self._masked_logp(logits, mask)
        new_logp = logp_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        ratio = torch.exp(new_logp - old_logp)

        # Gate logit: always mean-centered (paper is explicit: the gate
        # thresholds on y(a) - y_mean, p24 Algorithm 2).
        centered = logits - logits.mean(dim=-1, keepdim=True)
        y_gate = centered.gather(1, actions.unsqueeze(1)).squeeze(1)
        # Loss-body logit: centered (paper text, default) or raw (literal
        # Algorithm 2) per AlgoConfig.loss_centered_logits -- A3/U1 probe toggle.
        y_loss_src = centered if self.config.loss_centered_logits else logits
        y_loss = y_loss_src.gather(1, actions.unsqueeze(1)).squeeze(1)
        # Advantage-sign-dependent one-sided gates (p24 Algorithm 2).
        gate_pos = (y_gate < self.config.l_th) & (ratio < 1.0 + self.config.ratio_eps)
        gate_neg = (y_gate > -self.config.l_th) & (ratio > 1.0 - self.config.ratio_eps)
        c = torch.where(adv >= 0, gate_pos, gate_neg).float()
        policy_loss = -(self.config.eta * y_loss * c * adv / (old_probs + 1e-8)).mean()

        value_loss, entropy = self._value_and_entropy(logits, values, returns, mask)
        loss = (
            policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
        )
        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grads()
        self.optimizer.step()

        with torch.no_grad():
            gate_off_frac = 1.0 - c.mean()
            stats_t = torch.stack(
                [policy_loss.detach(), value_loss.detach(), entropy.detach(), gate_off_frac]
            )
        total_pol, total_val, total_ent, gate_off = stats_t.cpu().tolist()
        approx_kl = float((old_logp - new_logp.detach()).mean().item())
        ev = _explained_variance_torch(returns, values.detach())
        return UpdateStats(
            policy_loss=total_pol,
            value_loss=total_val,
            entropy=total_ent,
            approx_kl=approx_kl,
            explained_variance=ev,
            extra={"gate_off_frac": gate_off},
        )


def safe_log(x: float) -> float:
    """Numerically safe log used by tests/fixtures to build logprob records."""
    return math.log(x + 1e-30)


__all__ = [
    "NNACHUpdate",
    "NNPPOUpdate",
    "safe_log",
]
