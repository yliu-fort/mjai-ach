"""Tabular UpdateRules: PPO-style and ACH-style updates on dict-backed policies.

These drive the small-game Phase-1 experiments and the CFR/exact-Nash validation
(AGENTS.md §1 D5, Step 3). They mutate :class:`mjai.agents.tabular.TabularPolicy`
logit/value rows in place via semi-gradient updates (no autograd; the dicts are
the parameters).

The two rules share the math of policy-gradient-on-advantage; they differ in:

  - **PPO-tabular**: clipped semi-gradient surrogate over the
    (new_logprob/old_logprob) ratio, like PPO's clipped objective but applied
    as a direct additive update to the chosen action's logit. Keeps a trust
    region; standard for the PPO-vs-ACH comparison.
  - **ACH-tabular**: multiplicative Hedge-style update toward
    exp(+eta * advantage), which is the discrete-time replicator step. No
    clipping — the entropy regularization is what stabilizes it. This is the
    algorithm whose last-iterate convergence is the paper's headline claim.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from mjai.agents.base import entropy_of_probs, masked_softmax
from mjai.agents.tabular import TabularPolicy, _obs_to_key
from mjai.algos.transition import Batch, UpdateStats
from mjai.algos.update_rule import AlgoConfig, UpdateRule


def _explained_variance(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """1 - Var(y_true - y_pred) / Var(y_true); 1.0 is a perfect value fit."""
    yt = np.asarray(list(y_true), dtype=np.float64)
    yp = np.asarray(list(y_pred), dtype=np.float64)
    if yt.var() < 1e-12:
        return 0.0
    return float(1.0 - (yt - yp).var() / yt.var())


class TabularUpdateRule(UpdateRule):
    """Common base for the two tabular rules; holds config + the policy.

    Subclasses implement :meth:`_update_row`, which mutates one observation's
    logit/value row given the per-transition advantage. The base handles batch
    iteration and stats aggregation.
    """

    def __init__(self, policy: TabularPolicy, config: AlgoConfig | None = None) -> None:
        if not isinstance(policy, TabularPolicy):
            raise TypeError(f"{type(self).__name__} requires a TabularPolicy, got {type(policy)}")
        super().__init__(policy)
        self.config = config or AlgoConfig()
        self.policy: TabularPolicy = policy  # narrow the type for subclasses

    def step(self, batch: Batch) -> UpdateStats:
        if batch.size == 0:
            return UpdateStats(policy_loss=0.0, value_loss=0.0, entropy=0.0)
        total_pol = 0.0
        total_val = 0.0
        total_ent = 0.0
        value_preds: list[float] = []
        value_targets: list[float] = []
        for i in range(batch.size):
            obs = batch.obs[i].tolist()
            adv = float(batch.advantages[i])
            ret = float(batch.returns[i])
            old_v = self.policy.get_value(obs)
            value_preds.append(old_v)
            value_targets.append(ret)
            pol_loss, ent = self._update_row(obs, batch.actions[i], adv, batch.legal_mask[i])
            val_loss = self._update_value(obs, ret)
            total_pol += pol_loss
            total_val += val_loss
            total_ent += ent
        n = batch.size
        return UpdateStats(
            policy_loss=total_pol / n,
            value_loss=total_val / n,
            entropy=total_ent / n,
            explained_variance=_explained_variance(value_targets, value_preds),
        )

    def _update_value(self, obs: list[float], target: float) -> float:
        """TD-ish regression: v <- v + lr * (target - v)."""
        key = _obs_to_key(obs)
        old = self.policy.values.get(key, 0.0)
        new = old + self.config.value_coef * self.config.learning_rate * (target - old)
        self.policy.values[key] = new
        return float((target - old) ** 2)

    def _update_row(
        self,
        obs: list[float],
        action: int,
        advantage: float,
        legal_mask: np.ndarray,
    ) -> tuple[float, float]:
        """Subclass-specific logit update; returns (policy_loss, entropy)."""
        raise NotImplementedError

    def _row_probs(self, obs: list[float], legal_mask: np.ndarray) -> list[float]:
        """Full-space probability vector after masking, at the current logits."""
        logits = self.policy.get_logits(obs)
        mask = [bool(m) for m in legal_mask]
        scaled = [lg / self.policy.temperature for lg in logits]
        return masked_softmax(scaled, mask)

    @staticmethod
    def _entropy_from_probs(probs: list[float], legal_mask: np.ndarray) -> float:
        legal_p = [p for p, m in zip(probs, legal_mask, strict=True) if m]
        return entropy_of_probs(legal_p)


class TabularPPOUpdate(TabularUpdateRule):
    """PPO-style clipped semi-gradient update on tabular logits.

    For the chosen action ``a`` at observation ``s`` with old/new log-probs
    (here: before/after the update), the clipped surrogate objective drives an
    additive update to ``logit[s, a]`` proportional to the clipped
    advantage-weighted ratio. We approximate the "old" policy as the current
    one before the update (on-policy), so the ratio starts at 1 and is clipped
    to ``[1-eps, 1+eps]`` times the sign of the advantage.
    """

    def __init__(
        self, policy: TabularPolicy, config: AlgoConfig | None = None, *, clip_eps: float = 0.2
    ) -> None:
        super().__init__(policy, config)
        self.clip_eps = clip_eps

    def _update_row(
        self, obs: list[float], action: int, advantage: float, legal_mask: np.ndarray
    ) -> tuple[float, float]:
        lr = self.config.learning_rate
        # Clipped surrogate (on-policy: ratio=1 before update): the per-step
        # advantage is clamped to [-clip_eps, +clip_eps], so the additive logit
        # update cannot move more than lr*clip_eps in either direction.
        clipped = max(-self.clip_eps, min(self.clip_eps, advantage))
        delta = lr * clipped
        self.policy.get_logits(obs)[action] += delta
        probs = self._row_probs(obs, legal_mask)
        pol_loss = -math.log(probs[action] + 1e-30) * (1 if advantage >= 0 else -1)
        return pol_loss, self._entropy_from_probs(probs, legal_mask)


class TabularACHUpdate(TabularUpdateRule):
    """ACH-style Hedge/replicator update on tabular logits.

    Multiplicative update toward exp(+eta * advantage) for the chosen action
    (the discrete-time replicator step), with entropy regularization baked into
    the step size. **No clipping** — this is the algorithmic distinction from
    PPO that the project is studying (AGENTS.md §1 D4).
    """

    def __init__(
        self,
        policy: TabularPolicy,
        config: AlgoConfig | None = None,
        *,
        hedge_eta: float | None = None,
    ) -> None:
        super().__init__(policy, config)
        # Default eta ties it to the entropy coef so the two rules are comparable.
        self.eta = hedge_eta if hedge_eta is not None else max(self.config.entropy_coef, 0.05)

    def _update_row(
        self, obs: list[float], action: int, advantage: float, legal_mask: np.ndarray
    ) -> tuple[float, float]:
        # Hedge step: chosen action's logit += eta * advantage (additive in logit
        # space == multiplicative in probability space, which is the replicator).
        self.policy.get_logits(obs)[action] += self.eta * advantage
        probs = self._row_probs(obs, legal_mask)
        pol_loss = -math.log(probs[action] + 1e-30)
        return pol_loss, self._entropy_from_probs(probs, legal_mask)
