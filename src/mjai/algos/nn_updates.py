"""The NN UpdateRule: one theta-parameterized PPO/ACH actor-critic update.

``theta`` selects the policy-improvement operator on a continuum, on
:class:`mjai.agents.mlp.MLPSharedActorCritic`:

  - ``theta = 0`` — reference PPO: clipped surrogate (Schulman et al. 2017).
  - ``theta = 1`` — paper-faithful ACH (AGENTS.md §1 D4): Fu et al., ICLR 2022
    (OpenReview ``DTXZqTNV5nW``), Algorithm 2 + Eq. 29 (p24) with Appendix H.3
    hyperparameters (p27-28). Ground truth: ``docs/paper_spec_ach.md``.
  - in between — the convex combination of the two policy losses, which is
    what the theta-scan notebooks sweep.

Everything OUTSIDE the policy term is one shared, knob-driven scaffold
(optimizer, advantage treatment, epochs per batch, grad clipping), with
ACH-protocol defaults at every theta. That is deliberate: it makes theta the
only difference between a PPO arm and an ACH arm, and it makes each PPO best
practice an explicit, separately-testable knob rather than a package deal.
Turning on a knob the paper contradicts while ``theta > 0`` emits an
:class:`~mjai.algos.update_rule.ACHFidelityWarning`.

The loss math itself lives in :mod:`mjai.algos.nn_losses`, so there is exactly
one implementation of the ACH operator in the repo. ``theta=1`` with the
shipped ACH config is pinned bit-exactly by
``tests/unit/data/nn_updates_golden.json`` (see ``tools/gen_nn_golden.py``).
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import torch
from torch import nn

from mjai.agents.mlp import MLPSharedActorCritic
from mjai.algos.nn_losses import (
    ach_policy_loss,
    explained_variance,
    masked_log_probs,
    normalize_advantages,
    ppo_policy_loss,
    value_loss_and_entropy,
)
from mjai.algos.transition import Batch, UpdateStats
from mjai.algos.update_rule import ACHFidelityWarning, AlgoConfig, UpdateRule

# |logp_new - logp_old| above this counts a sample as off-policy. On the first
# inner epoch the two are produced by the same weights on the same observation,
# so an on-policy batch lands at 0.0 up to float32 noise: old_logp comes from
# the rollout's single-row forward, new_logp from the update's batched forward
# (different GEMM accumulation order). That noise was measured directly on a
# trained liars_dice1 mirror checkpoint (the worst game: largest logits) at
# max 1.43e-6 per sample. 2e-6 is the minimum cleanly-passing tolerance — ~1.2x
# that measured ceiling — so a true on-policy batch reads exactly 0.0 here
# while any real behavior/target mismatch (the pre-fix league ran median |KL|
# ~0.08 nats) is still caught with decades of headroom. Keep league_diagnose.py
# KL_TOL in sync with this value.
OFF_POLICY_TOL = 2e-6


def _warn_if_ach_incompatible(config: AlgoConfig) -> None:
    """Warn when a knob the ACH paper contradicts is on while ACH has weight.

    Silent at ``theta=0`` (no ACH term to be unfaithful to) and silent on the
    shipped ACH defaults — so a reproduction run never warns, and an A/B arm
    always says so out loud.
    """
    if config.theta <= 0.0:
        return
    issues: list[str] = []
    if (config.optimizer or "sgd") == "adam":
        issues.append(
            "optimizer='adam' (H.3 p27: 'stochastic gradient descent with a "
            "constant learning rate')"
        )
    if config.normalize_advantages:
        issues.append(
            "normalize_advantages=True (Eq. 29 p24 consumes raw GAE advantages; "
            "normalizing turns the hedge coefficient eta into a batch-adaptive "
            "learning rate)"
        )
    if config.n_epochs > 1:
        issues.append(
            f"n_epochs={config.n_epochs} (p24: 'we update theta and omega once "
            "using a single mini-batch at each iteration')"
        )
    if config.iw_clip is not None:
        issues.append(
            f"iw_clip={config.iw_clip} (Algorithm 2 / Eq. 29 carries the raw "
            "1/pi_old importance weight; capping it is a stabilizer the paper "
            "does not have)"
        )
    if config.n_critic_updates > 0:
        issues.append(
            f"n_critic_updates={config.n_critic_updates} (paper does 1 combined "
            "update; extra value-only updates are a critic-quality boost it does "
            "not have)"
        )
    if config.separate_critic:
        issues.append(
            "separate_critic=True (paper shares params between policy and value, "
            "App. E; an independent critic net is an architecture deviation)"
        )
    if issues:
        warnings.warn(
            f"ACH policy term is active (theta={config.theta}) alongside "
            f"PPO-best-practice knobs the ACH paper does not use: "
            f"{'; '.join(issues)}. This is a valid A/B arm but NOT a faithful "
            f"reproduction — do not compare it against the paper's curves.",
            ACHFidelityWarning,
            stacklevel=3,
        )


class NNActorCriticUpdate(UpdateRule):
    """One gradient step of the theta-interpolated PPO/ACH actor-critic loss.

    Per sample ``[a, s, A(s,a), G, pi_old(a|s)]`` the total loss is::

        L = (1-theta) * L_ppo + theta * L_ach
            + value_coef * (V - G)^2 - entropy_coef * H(pi)

    with ``L_ppo`` the clipped surrogate and ``L_ach`` the gated logit-space
    term of Eq. 29 (see :mod:`mjai.algos.nn_losses` for both, including the
    fidelity notes). The endpoints short-circuit: at ``theta=0`` the ACH term
    is never built, at ``theta=1`` the PPO term is never built, so each
    endpoint is bit-identical to a dedicated implementation and pays nothing
    for the other's presence.

    Intentional deviations on the PPO side, kept because the scaffolding is
    shared and its defaults follow ACH (each is a knob, listed with the field
    that restores the 37-details behavior):

      - one full-batch gradient step per batch, no minibatch shuffling and
        hence no KL early-stop (``n_epochs``);
      - raw rather than normalized advantages (``normalize_advantages``);
      - constant-LR SGD rather than Adam (``optimizer``, ``adam_eps``);
      - PyTorch default weight init (no orthogonal init — an architecture
        concern, so it is not a knob here).
    """

    def __init__(self, policy: MLPSharedActorCritic, config: AlgoConfig | None = None) -> None:
        if not isinstance(policy, MLPSharedActorCritic):
            raise TypeError(
                f"{type(self).__name__} requires MLPSharedActorCritic, got {type(policy)}"
            )
        super().__init__(policy)
        self.config = config or AlgoConfig()
        _warn_if_ach_incompatible(self.config)
        self.policy: MLPSharedActorCritic = policy
        self.optimizer = self._make_optimizer()
        # Optional INDEPENDENT critic (AlgoConfig.separate_critic): own params and
        # optimizer, so training it hard never drifts the policy (the shared-trunk
        # n_critic_updates does). Its V(s) supplies the advantage baseline.
        self.critic: MLPSharedActorCritic | None = None
        self.critic_opt: torch.optim.Optimizer | None = None
        if self.config.separate_critic:
            self.critic = MLPSharedActorCritic(
                policy.obs_size,
                policy.num_actions,
                hidden_sizes=self.config.critic_hidden_sizes,
                device=str(policy.device),
            )
            self.critic_opt = torch.optim.SGD(
                self.critic.parameters(), lr=self.config.learning_rate
            )

    # ---- scaffolding ----

    def _make_optimizer(self) -> torch.optim.Optimizer:
        """Build the configured optimizer; SGD (paper H.3 p27) is the default."""
        opt = self.config.optimizer or "sgd"
        if opt == "sgd":
            return torch.optim.SGD(self.policy.parameters(), lr=self.config.learning_rate)
        if opt == "adam":
            return torch.optim.Adam(
                self.policy.parameters(),
                lr=self.config.learning_rate,
                eps=self.config.adam_eps,
            )
        raise ValueError(f"Unknown optimizer {opt!r}; expected 'sgd' or 'adam'")

    def _clip_grads(self) -> None:
        """Grad-norm clipping; disabled when ``max_grad_norm <= 0``.

        The ACH paper mentions no grad clipping, so the reproduction configs
        set ``max_grad_norm: 0.0``; the 37-details PPO value is 0.5.
        """
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)

    def _term_grad_probe(
        self, terms: dict[str, torch.Tensor], theta: float
    ) -> dict[str, torch.Tensor]:
        """Per-policy-term gradient norms + their cosine, for mixed-theta debugging.

        The total ``grad_norm`` cannot say whether an update was driven by the
        PPO term or the ACH one, and at intermediate theta that is exactly the
        question: the two terms differ in gradient magnitude by orders of
        magnitude (ACH carries an unbounded ``1/pi_old``), so the blend weight
        theta is not the blend of *influence*. Reported per term:

          - ``grad_norm_<t>``        — the term's own norm, unweighted.
          - ``grad_norm_<t>_scaled`` — times its theta weight: what actually
            enters the update.
          - ``grad_cos_ppo_ach``     — cosine between the two term gradients
            (negative = the terms are pulling against each other).

        Only terms that were built are reported: at theta=0 there is no ACH
        gradient and at theta=1 no PPO one, and emitting 0.0 for a term that
        does not exist would read as a measured zero. The cosine needs both, so
        it appears only for 0 < theta < 1.

        Uses ``torch.autograd.grad``, which reads the graph without touching
        ``.grad``; the subsequent ``loss.backward()`` and optimizer step are
        unaffected.
        """
        params = [p for p in self.policy.parameters() if p.requires_grad]
        weights = {"ppo": 1.0 - theta, "ach": theta}
        out: dict[str, torch.Tensor] = {}
        flat: dict[str, torch.Tensor] = {}
        for name, loss in terms.items():
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            present = [g.reshape(-1) for g in grads if g is not None]
            if not present:  # a term touching no parameter has no direction
                continue
            vec = torch.cat(present)
            flat[name] = vec
            norm = vec.norm()
            out[f"grad_norm_{name}"] = norm
            out[f"grad_norm_{name}_scaled"] = norm * weights[name]
        if len(flat) == 2:
            a, b = flat["ppo"], flat["ach"]
            out["grad_cos_ppo_ach"] = torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)
        return out

    def _grad_norm(self) -> torch.Tensor:
        """Raw (pre-clip) global grad norm.

        Read before :meth:`_clip_grads` so it reflects the gradient the paper's
        unclipped setting actually applies. Central telemetry for the theta
        scan: the ACH term carries an unbounded ``1/pi_old`` factor while the
        PPO term is O(1), so gradient scale varies strongly with theta.
        """
        with torch.no_grad():
            grad_sqs = [
                (p.grad.detach() ** 2).sum() for p in self.policy.parameters() if p.grad is not None
            ]
            if not grad_sqs:
                return torch.zeros((), device=self.policy.device)
            return torch.sqrt(torch.stack(grad_sqs).sum())

    # ---- optimizer state I/O ----

    def state_dict(self) -> dict[str, object]:
        return {"optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if "optimizer" in state:
            # The optimizer entry is a dict[str, Any] produced by torch; cast
            # through Any since the value is typed as `object` at the boundary.
            opt_state: dict[str, Any] = state["optimizer"]  # type: ignore[assignment]
            self.optimizer.load_state_dict(opt_state)

    # ---- the update ----

    def step(self, batch: Batch) -> UpdateStats:
        if batch.size == 0:
            return UpdateStats(policy_loss=0.0, value_loss=0.0, entropy=0.0)
        device = self.policy.device
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=device)
        mask = torch.as_tensor(batch.legal_mask, dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=device)
        old_logp = torch.as_tensor(batch.logprobs, dtype=torch.float32, device=device)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=device)
        # Raw GAE advantages feed the ACH term unconditionally (paper p24); the
        # PPO term optionally sees the normalized copy.
        adv_raw = torch.as_tensor(batch.advantages, dtype=torch.float32, device=device)
        if self.critic is not None and self.critic_opt is not None:
            # Independent critic (AlgoConfig.separate_critic): train it hard on the
            # value loss (own params -- no policy drift), then set the advantage to
            # the MC baseline G - V_critic(s). Bypasses the rollout's GAE (the
            # flattened batch has lost the trajectory structure GAE needs); for
            # Liar's Dice terminal-only rewards this is a minor change.
            for _ in range(max(1, self.config.n_critic_updates)):
                v_c = self.critic(obs)[1]
                cvloss = self.config.value_coef * ((v_c - returns) ** 2).mean()
                self.critic_opt.zero_grad()
                cvloss.backward()
                self.critic_opt.step()
            with torch.no_grad():
                adv_raw = returns - self.critic(obs)[1]
        adv_ppo = adv_raw
        if self.config.theta < 1.0 and self.config.normalize_advantages:
            adv_ppo = normalize_advantages(adv_raw)
        # pi_old(a|s): the behavior-policy prob recorded by the rollout under
        # the exact policy that sampled the action (paper: samples arrive as
        # [a, s, A, G, pi_old(a|s)], p24 Algorithm 2).
        old_probs = torch.exp(old_logp)

        stats = UpdateStats(policy_loss=0.0, value_loss=0.0, entropy=0.0)
        # Optional extra value-only updates to fit V harder before the policy step
        # (AlgoConfig.n_critic_updates). The advantage for THIS batch was already
        # computed in the rollout; these updates improve V for FUTURE batches' GAE.
        for _ in range(self.config.n_critic_updates):
            logits_c, values_c = self.policy(obs)
            vloss, _ = value_loss_and_entropy(logits_c, values_c, returns, mask)
            vstep = self.config.value_coef * vloss
            self.optimizer.zero_grad()
            vstep.backward()  # type: ignore[no-untyped-call]  # torch's Tensor.backward is untyped
            self.optimizer.step()
        first_off_policy: float | None = None
        for _ in range(self.config.n_epochs):
            stats = self._gradient_step(
                obs=obs,
                mask=mask,
                actions=actions,
                old_logp=old_logp,
                old_probs=old_probs,
                returns=returns,
                adv_raw=adv_raw,
                adv_ppo=adv_ppo,
            )
            if first_off_policy is None:
                first_off_policy = stats.extra.get("off_policy_frac", 0.0)
        # Report the FIRST epoch's reading: later epochs are off-policy by
        # construction (the batch is stale w.r.t. the weights they update), so
        # only the first one answers "did the policy being updated collect this
        # batch?". At the ACH protocol's n_epochs=1 (paper p24) they coincide.
        if first_off_policy is not None:
            stats.extra["off_policy_frac"] = first_off_policy
        return stats

    def _gradient_step(
        self,
        *,
        obs: torch.Tensor,
        mask: torch.Tensor,
        actions: torch.Tensor,
        old_logp: torch.Tensor,
        old_probs: torch.Tensor,
        returns: torch.Tensor,
        adv_raw: torch.Tensor,
        adv_ppo: torch.Tensor,
    ) -> UpdateStats:
        """One forward/backward/optimizer step over the whole batch."""
        theta = self.config.theta
        logits, values = self.policy(obs)
        logp_all = masked_log_probs(logits, mask)
        new_logp = logp_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        ratio = torch.exp(new_logp - old_logp)

        telemetry: dict[str, torch.Tensor] = {
            # Structural probe, not a hyperparameter: the fraction of the batch
            # whose behavior log-prob disagrees with this policy's. A self-play
            # controller that hands one learner's transitions to a DIFFERENT
            # learner's rule shows up here as a nonzero value, at full update
            # resolution, without needing a paired run to notice.
            "off_policy_frac": ((new_logp - old_logp).detach().abs() > OFF_POLICY_TOL)
            .float()
            .mean(),
        }
        terms: dict[str, torch.Tensor] = {}
        policy_loss: torch.Tensor | None = None
        if theta < 1.0:
            ppo_loss, ppo_stats = ppo_policy_loss(
                ratio=ratio, advantages=adv_ppo, clip_eps=self.config.clip_eps
            )
            telemetry.update(ppo_stats)
            terms["ppo"] = ppo_loss
            policy_loss = ppo_loss if theta == 0.0 else (1.0 - theta) * ppo_loss
        if theta > 0.0:
            ach_loss, ach_stats = ach_policy_loss(
                logits=logits,
                mask=mask,
                actions=actions,
                ratio=ratio,
                advantages=adv_raw,
                old_probs=old_probs,
                config=self.config,
            )
            telemetry.update(ach_stats)
            terms["ach"] = ach_loss
            scaled = ach_loss if theta == 1.0 else theta * ach_loss
            policy_loss = scaled if policy_loss is None else policy_loss + scaled
        assert policy_loss is not None  # theta in [0, 1] always builds one term
        if self.config.probe_term_grad_norms:
            telemetry.update(self._term_grad_probe(terms, theta))

        value_loss, entropy = value_loss_and_entropy(logits, values, returns, mask)
        loss = (
            policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
        )
        self.optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]  # torch's Tensor.backward is untyped
        telemetry["grad_norm"] = self._grad_norm()
        self._clip_grads()
        self.optimizer.step()

        return self._make_stats(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            old_logp=old_logp,
            new_logp=new_logp,
            returns=returns,
            values=values,
            telemetry=telemetry,
        )

    def _make_stats(
        self,
        *,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        entropy: torch.Tensor,
        old_logp: torch.Tensor,
        new_logp: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
        telemetry: dict[str, torch.Tensor],
    ) -> UpdateStats:
        """Collect the step's scalars with a single device sync (AGENTS.md §8).

        ``clip_frac`` is promoted to a first-class field (PPO's own metric);
        the remaining telemetry — the ACH gate/importance-weight probes and the
        always-on ``grad_norm`` — rides in ``extra``.
        """
        names = sorted(telemetry)
        with torch.no_grad():
            stacked = torch.stack(
                [policy_loss.detach(), value_loss.detach(), entropy.detach()]
                + [telemetry[n] for n in names]
            )
        flat = stacked.cpu().tolist()
        extra = dict(zip(names, flat[3:], strict=True))
        return UpdateStats(
            policy_loss=flat[0],
            value_loss=flat[1],
            entropy=flat[2],
            approx_kl=float((old_logp - new_logp.detach()).mean().item()),
            clip_frac=extra.pop("clip_frac", 0.0),
            explained_variance=explained_variance(returns, values.detach()),
            extra=extra,
        )


def safe_log(x: float) -> float:
    """Numerically safe log used by tests/fixtures to build logprob records."""
    return math.log(x + 1e-30)


__all__ = [
    "NNActorCriticUpdate",
    "safe_log",
]
