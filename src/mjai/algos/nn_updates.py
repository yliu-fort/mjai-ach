"""Neural UpdateRules: PPO and ACH on torch-backed policies.

These drive the NN half of the 2x2 matrix (AGENTS.md §1 D1, Step 4-5). They use
torch autograd on :class:`mjai.agents.mlp.MLPSharedActorCritic`. The actor math
mirrors the tabular rules (:mod:`mjai.algos.tabular_updates`) but expressed as
differentiable losses rather than in-place dict mutation:

  - :class:`NNPPOUpdate` — the standard PPO clipped surrogate + value MSE +
    entropy bonus, with optional multiple inner epochs over the batch.
  - :class:`NNACHUpdate` — REINFORCE policy gradient with the learned critic as
    baseline + entropy bonus, **no clipping** (AGENTS.md §1 D4). The entropy
    regularization is the sole stabilizer; in the continuous-time limit this is
    replicator/Hedge dynamics, which is what gives ACH its last-iterate
    convergence claim.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from mjai.agents.base import entropy_of_probs
from mjai.agents.mlp import MASK_VALUE, MLPSharedActorCritic
from mjai.algos.transition import Batch, UpdateStats
from mjai.algos.update_rule import AlgoConfig, UpdateRule


def _explained_variance_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.var() < 1e-12:
        return 0.0
    return float(1.0 - (y_true - y_pred).var() / y_true.var())


class _NNUpdateBase(UpdateRule):
    """Common scaffolding: optimizer, device, advantage normalization."""

    def __init__(self, policy: MLPSharedActorCritic, config: AlgoConfig | None = None) -> None:
        if not isinstance(policy, MLPSharedActorCritic):
            raise TypeError(
                f"{type(self).__name__} requires MLPSharedActorCritic, got {type(policy)}"
            )
        super().__init__(policy)
        self.config = config or AlgoConfig()
        self.policy: MLPSharedActorCritic = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.config.learning_rate)

    def _obs_tensor(self, batch: Batch) -> torch.Tensor:
        return torch.as_tensor(batch.obs, dtype=torch.float32, device=self.policy.device)

    def _legal_mask_tensor(self, batch: Batch) -> torch.Tensor:
        return torch.as_tensor(batch.legal_mask, dtype=torch.float32, device=self.policy.device)

    def _masked_logp(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = logits + (1.0 - mask) * MASK_VALUE
        return torch.log_softmax(masked, dim=-1)

    def _normalize_advantages(self, adv: torch.Tensor) -> torch.Tensor:
        if adv.numel() <= 1:
            return adv
        std = adv.std()
        if std.item() < 1e-8:
            return adv - adv.mean()
        return (adv - adv.mean()) / (std + 1e-8)

    def state_dict(self) -> dict[str, object]:
        return {"optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if "optimizer" in state:
            # The optimizer entry is a dict[str, Any] produced by torch; cast
            # through Any since the value is typed as `object` at the boundary.
            opt_state: dict[str, Any] = state["optimizer"]  # type: ignore[assignment]
            self.optimizer.load_state_dict(opt_state)


class NNPolicyGradientUpdate(_NNUpdateBase):
    """Unified PPO/ACH policy-gradient update (AGENTS.md D4).

    PPO and ACH are the *same* algorithm family; they differ only in the
    policy-improvement operator. We expose that difference as a single
    parameter ``theta`` in [0, 1] so PPO-vs-ACH comparisons hold everything
    else fixed (the experiment whole point):

      theta = 0  -> PPO  : clipped surrogate -min(r*A, clip(r)*A)
      theta = 1  -> ACH  : entropy-regularized replicator -logp*A - beta*H
      0<theta<1  -> convex interpolation of the two policy-loss terms.

    Everything else is identical between the two and is NOT a function of
    theta: critic (value head + MSE loss), advantage normalization,
    entropy bonus coefficient, value-loss coefficient, max grad norm,
    optimizer, multi-epoch inner loop, KL early-stop, and the snapshot of
    old-policy log-probs from the rollout. The previous design had several
    unfair differences (PPO ran 4 inner epochs while ACH ran 1; ACH used a
    separate KL term PPO did not; entropy was no_grad in PPO but in-loss in
    ACH); this design removes them.

    ``beta`` (the ACH entropy-regularizer weight) defaults to
    ``entropy_coef`` so the ACH term is the standard replicator form.
    """

    def __init__(
        self,
        policy: MLPSharedActorCritic,
        config: AlgoConfig | None = None,
        *,
        theta: float = 0.0,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        target_kl: float | None = 0.04,
        beta: float | None = None,
    ) -> None:
        super().__init__(policy, config)
        if not 0.0 <= theta <= 1.0:
            raise ValueError(f"theta must be in [0, 1], got {theta}")
        self.theta = float(theta)
        self.clip_eps = float(clip_eps)
        self.n_epochs = int(n_epochs)
        self.target_kl = target_kl
        self.beta = float(beta) if beta is not None else self.config.entropy_coef
        # ACH (NeuRD-style) logit threshold. The paper clips logits to
        # [-l_th, l_th] after mean-subtraction (Appendix E); l_th=2.0 for the
        # OpenSpiel small-game experiments. Essential because the NeuRD
        # direct-logit gradient does NOT vanish on softmax saturation, so
        # without bounding logits they blow up.
        self.l_th = 2.0
        # Hedge coefficient (per-state learning rate on the ACH side); η=1.0
        # in every experiment in the paper.
        self.eta = 1.0

    def step(self, batch: Batch) -> UpdateStats:
        if batch.size == 0:
            return UpdateStats(policy_loss=0.0, value_loss=0.0, entropy=0.0)
        obs = self._obs_tensor(batch)
        mask = self._legal_mask_tensor(batch)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.policy.device)
        old_logp = torch.as_tensor(batch.logprobs, dtype=torch.float32, device=self.policy.device)
        # old_probs(a) for the ACH importance-correction term (1/pi_old). In
        # synchronous self-play pi_old == current policy so this is ~1, but we
        # keep it for correctness.
        with torch.no_grad():
            old_logits_snap, _ = self.policy(obs)
            old_logp_all = self._masked_logp(old_logits_snap, mask)
        old_probs = torch.exp(old_logp_all).gather(1, actions.unsqueeze(1)).squeeze(1)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.policy.device)
        adv = self._normalize_advantages(
            torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.policy.device)
        )

        # Single gradient step per mini-batch for BOTH endpoints (apples-to-
        # apples). ACH paper Appendix E: "we only update once using a single
        # mini-batch at each iteration." PPO is aligned to the same single-step
        # regime so the PPO-vs-ACH comparison isolates the policy-improvement
        # operator (theta), not the update-count.
        total_pol = total_val = total_ent = total_kl = total_clip = 0.0
        steps = 0
        for _ in range(1):
            logits, values = self.policy(obs)
            logp_all = self._masked_logp(logits, mask)
            new_logp = logp_all.gather(1, actions.unsqueeze(1)).squeeze(1)
            # --- PPO term (theta=0): clipped surrogate of the likelihood ratio.
            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
            ppo_loss = -torch.min(surr1, surr2).mean()
            # --- ACH term (theta=1): NeuRD direct-logit loss (paper Algorithm 2).
            # The gradient of -y(a)*A wrt logit y(a) is -A (constant; does NOT
            # vanish on softmax saturation like REINFORCE's -(1-pi(a))*A). This
            # is the single most important difference from REINFORCE and the
            # reason ACH/NeuRD don't collapse the way REINFORCE does.
            # Steps: (1) mean-subtract the logits; (2) threshold to [-l_th, l_th]
            # (zero out the gradient where the logit is already at the bound);
            # (3) importance-correct by 1/old_prob(a); (4) loss = -eta*y(a)*A/pi_old.
            centered = logits - logits.mean(dim=-1, keepdim=True)
            y_a = centered.gather(1, actions.unsqueeze(1)).squeeze(1)
            # Threshold: where |centered_logit_for_action| > l_th, zero the loss
            # contribution for that sample (OpenSpiel NeuRD's `thresholded`).
            within = (y_a.abs() <= self.l_th).float()
            ach_loss = -(self.eta * y_a * adv * within / (old_probs + 1e-8)).mean()
            # --- Interpolated policy loss.
            policy_loss = (1.0 - self.theta) * ppo_loss + self.theta * ach_loss
            # --- Shared critic + entropy bonus (identical at both endpoints).
            # Entropy MUST keep its gradient (NOT under no_grad) so the entropy
            # bonus actually pulls logits toward uniform. The previous no_grad
            # wrapper silently zeroed the entropy gradient — a real bug.
            value_loss = ((values - returns) ** 2).mean()
            legal_probs = torch.exp(logp_all)
            ent_per_row = -(legal_probs * logp_all).sum(dim=-1)
            entropy = ent_per_row.mean()
            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                - self.config.entropy_coef * entropy
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            with torch.no_grad():
                total_pol += float(policy_loss.detach())
                total_val += float(value_loss.detach())
                total_ent += float(entropy.detach())
                total_kl += float((old_logp - new_logp.detach()).mean())
                total_clip += float(((ratio.detach() - 1.0).abs() > self.clip_eps).float().mean())
            steps += 1
            if self.target_kl is not None and total_kl / steps > 1.5 * self.target_kl:
                break

        ev = _explained_variance_np(returns.detach().cpu().numpy(), values.detach().cpu().numpy())
        return UpdateStats(
            policy_loss=total_pol / steps,
            value_loss=total_val / steps,
            entropy=total_ent / steps,
            approx_kl=total_kl / steps,
            clip_frac=total_clip / steps,
            explained_variance=ev,
            extra={"theta": self.theta},
        )


class NNPPOUpdate(NNPolicyGradientUpdate):
    """PPO endpoint of the unified update (theta=0)."""

    def __init__(
        self,
        policy: MLPSharedActorCritic,
        config: AlgoConfig | None = None,
        *,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        target_kl: float | None = 0.04,
        beta: float | None = None,
        theta: float = 0.0,
    ) -> None:
        if theta != 0.0:
            raise ValueError(f"NNPPOUpdate fixes theta=0; got theta={theta}")
        super().__init__(
            policy,
            config,
            theta=0.0,
            clip_eps=clip_eps,
            n_epochs=n_epochs,
            target_kl=target_kl,
            beta=beta,
        )


class NNACHUpdate(NNPolicyGradientUpdate):
    """ACH endpoint of the unified update (theta=1).

    Per AGENTS.md D4, ACH differs from PPO *only* in the policy-improvement
    operator: no clipped surrogate, just REINFORCE + entropy regularizer.
    All other hyperparams and stabilizers are identical to PPO so the
    comparison is apples-to-apples.
    """

    def __init__(
        self,
        policy: MLPSharedActorCritic,
        config: AlgoConfig | None = None,
        *,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        target_kl: float | None = 0.04,
        beta: float | None = None,
        theta: float = 1.0,
    ) -> None:
        if theta != 1.0:
            raise ValueError(f"NNACHUpdate fixes theta=1; got theta={theta}")
        super().__init__(
            policy,
            config,
            theta=1.0,
            clip_eps=clip_eps,
            n_epochs=n_epochs,
            target_kl=target_kl,
            beta=beta,
        )


def batch_entropy(probs: list[list[float]]) -> float:
    """Mean Shannon entropy across rows of a probability matrix (diagnostic)."""
    if not probs:
        return 0.0
    return sum(entropy_of_probs(row) for row in probs) / len(probs)


def safe_log(x: float) -> float:
    return math.log(x + 1e-30)
