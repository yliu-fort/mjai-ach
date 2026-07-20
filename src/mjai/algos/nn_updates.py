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


class NNPPOUpdate(_NNUpdateBase):
    """PPO: clipped surrogate + value MSE + entropy, multiple inner epochs."""

    def __init__(
        self,
        policy: MLPSharedActorCritic,
        config: AlgoConfig | None = None,
        *,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
    ) -> None:
        super().__init__(policy, config)
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs

    def step(self, batch: Batch) -> UpdateStats:
        if batch.size == 0:
            return UpdateStats(policy_loss=0.0, value_loss=0.0, entropy=0.0)
        obs = self._obs_tensor(batch)
        mask = self._legal_mask_tensor(batch)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.policy.device)
        old_logp = torch.as_tensor(batch.logprobs, dtype=torch.float32, device=self.policy.device)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.policy.device)
        adv = self._normalize_advantages(
            torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.policy.device)
        )

        total_pol = total_val = total_ent = total_kl = total_clip = 0.0
        steps = 0
        for _ in range(self.n_epochs):
            logits, values = self.policy(obs)
            logp_all = self._masked_logp(logits, mask)
            new_logp = logp_all.gather(1, actions.unsqueeze(1)).squeeze(1)
            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((values - returns) ** 2).mean()
            with torch.no_grad():
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
                total_pol += float(policy_loss)
                total_val += float(value_loss)
                total_ent += float(entropy)
                total_kl += float((old_logp - new_logp).mean())
                total_clip += float(((ratio - 1.0).abs() > self.clip_eps).float().mean())
            steps += 1

        ev = _explained_variance_np(returns.detach().cpu().numpy(), values.detach().cpu().numpy())
        return UpdateStats(
            policy_loss=total_pol / steps,
            value_loss=total_val / steps,
            entropy=total_ent / steps,
            approx_kl=total_kl / steps,
            clip_frac=total_clip / steps,
            explained_variance=ev,
        )


class NNACHUpdate(_NNUpdateBase):
    """ACH: REINFORCE + critic baseline + entropy bonus. **No clipping.**

    The policy loss is ``-mean(logp(a) * advantage) - entropy_coef * H``. The
    entropy term is what makes the continuous-time update replicator-like
    rather than best-response-like; this is the lever behind ACH's claimed
    last-iterate convergence (AGENTS.md §1 D4).
    """

    def __init__(
        self,
        policy: MLPSharedActorCritic,
        config: AlgoConfig | None = None,
    ) -> None:
        super().__init__(policy, config)

    def step(self, batch: Batch) -> UpdateStats:
        if batch.size == 0:
            return UpdateStats(policy_loss=0.0, value_loss=0.0, entropy=0.0)
        obs = self._obs_tensor(batch)
        mask = self._legal_mask_tensor(batch)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.policy.device)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.policy.device)
        adv = self._normalize_advantages(
            torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.policy.device)
        )

        logits, values = self.policy(obs)
        logp_all = self._masked_logp(logits, mask)
        new_logp = logp_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        policy_loss = -(new_logp * adv).mean()
        value_loss = ((values - returns) ** 2).mean()
        with torch.no_grad():
            legal_probs = torch.exp(logp_all)
            entropy = -(legal_probs * logp_all).sum(dim=-1).mean()
        loss = (
            policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
        )
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        ev = _explained_variance_np(returns.detach().cpu().numpy(), values.detach().cpu().numpy())
        return UpdateStats(
            policy_loss=float(policy_loss.detach()),
            value_loss=float(value_loss.detach()),
            entropy=float(entropy),
            explained_variance=ev,
            extra={"adv_mean": float(adv.mean().detach()), "adv_std": float(adv.std().detach())},
        )


def batch_entropy(probs: list[list[float]]) -> float:
    """Mean Shannon entropy across rows of a probability matrix (diagnostic)."""
    if not probs:
        return 0.0
    return sum(entropy_of_probs(row) for row in probs) / len(probs)


def safe_log(x: float) -> float:
    return math.log(x + 1e-30)
