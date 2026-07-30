"""Abstract :class:`Policy` interface (AGENTS.md §3, §4).

Every agent — tabular dict, MLP actor-critic, future Mahjong net — implements
this single interface. The interface is deliberately torch-free so the tabular
path has no heavy dependency; NN subclasses import torch internally.

The interface is **self-play-topology-agnostic**: a Policy does not know whether
it is playing itself, a frozen opponent, or a human. It only answers
"given this observation and these legal actions, what do you do?".
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Protocol


class ActionSampler(Protocol):
    """How an ``act()`` call turns logits/preferences into a sampled action.

    Decoupled from Policy so the same network can be sampled stochastically
    during data collection and greedily during evaluation.
    """

    def sample(self, logits: Any, legal_actions: list[int]) -> tuple[int, float]:
        """Return (action, logprob-of-action) under this sampler."""
        ...


class Policy(ABC):
    """Maps an observation + legal-action set to an action and a value estimate.

    Concrete subclasses:
      - :class:`mjai.agents.tabular.TabularPolicy` (dict-backed, no torch)
      - :class:`mjai.agents.mlp.MLPSharedActorCritic` (torch, Step 2 later)

    The two ``act`` overloads cover the two call sites:
      - ``act(obs, legal_actions)`` — used during self-play rollout (stochastic,
        returns a logprob so PPO/ACH can compute the policy loss).
      - ``act(obs, legal_actions, eval=True)`` — used during evaluation and the
        Play CLI (greedy/deterministic; logprob is unused).
    """

    @abstractmethod
    def act(
        self,
        obs: list[float],
        legal_actions: list[int],
        *,
        eval: bool = False,
        rng_key: Any = None,
        behavior_epsilon: float = 0.0,
    ) -> tuple[int, float]:
        """Choose an action.

        Args:
            obs: per-player observation vector (length = ``GameSpec.obs_size``).
            legal_actions: action ids legal at this state; never empty.
            eval: if True, act greedily (no exploration); else sample stochastically.
            rng_key: optional deterministic-state handle (seed/key) for reproducibility.
            behavior_epsilon: sample from ``mu = (1-eps)*pi + eps*Uniform(legal)``
                instead of ``pi``. 0.0 (default) is the paper's on-policy
                behavior and leaves this path bit-identical.

        Returns:
            (action_id, logprob). ``logprob`` is the log-probability of the
            chosen action under the **behavior** policy that sampled it -- which
            is ``pi`` unless ``behavior_epsilon`` is set (0.0 in eval mode is
            acceptable).
        """
        ...

    @abstractmethod
    def value(self, obs: list[float]) -> float:
        """Scalar value estimate V(obs). Used as the advantage baseline."""
        ...

    @abstractmethod
    def action_logits(self, obs: list[float], legal_actions: list[int]) -> list[float]:
        """Raw preferences per legal action (for entropy/stats; not normalized).

        Length must equal ``len(legal_actions)``. Order matches ``legal_actions``.
        """
        ...

    # ---- batched read API (eval hot path, AGENTS.md §8) ----
    #
    # Exact equilibrium eval materializes the whole policy over every info state
    # in the game. Asking one state at a time costs ~200 us per call for an NN
    # (a one-row forward plus one device sync per legal action), which dwarfs
    # the arithmetic: the same MLP answers 24k states in ~15 ms as one batch.
    # Subclasses with a vectorizable backbone override this; the default below
    # loops over ``action_logits`` so a new subclass is correct by construction.
    def action_logits_batch(self, obs_batch: Any, legal_mask: Any) -> Any:
        """Full-action-space logits for many observations at once.

        Args:
            obs_batch: ``(B, obs_size)`` float array of observations.
            legal_mask: ``(B, num_actions)`` bool array, True where legal. Every
                row must have at least one legal action.

        Returns:
            ``(B, num_actions)`` **float64** array of logits, with illegal entries
            set to ``-inf`` so the caller can softmax over the row directly.

        The output dtype is float64 because this method is the exact evaluator's
        only window onto a policy: :func:`mjai.eval.nash.tabular_view_of` softmaxes
        it and hands the result to a best-response solver. A float32 return here
        used to cap the base stack's NashConv at ~1e-8 relative — measured against
        an independent float64 implementation across Kuhn, 3p Kuhn and Leduc — no
        matter which solver ran underneath (AGENTS.md D14). Widening the *handoff*
        does not invent precision: a tabular policy holds Python floats, so its
        logits are now exact, while an MLP's logits stay as precise as its own
        float32 weights make them. It just stops the metric from adding error of
        its own to the model's.
        """
        import numpy as np

        mask = np.asarray(legal_mask, dtype=bool)
        out = np.full(mask.shape, -np.inf, dtype=np.float64)
        for i, row in enumerate(mask):
            legal = np.flatnonzero(row).tolist()
            out[i, legal] = self.action_logits(list(obs_batch[i]), legal)
        return out

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist to ``path`` using the canonical ckpt manifest (AGENTS.md §10)."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore in-place from ``path`` (produced by :meth:`save`)."""
        ...

    # ---- fused online-step API (hot-path optimization, AGENTS.md §8) ----
    #
    # During rollout the worker needs (action, logprob, value) at every decision
    # point. Naively that is two forwards (one in ``act``, one in ``value``).
    # Concrete subclasses fuse them into a single forward by overriding this
    # method; the default below falls back to ``act`` + ``value`` so a new
    # subclass is correct by construction (no silent degradation if a future
    # Policy forgets to override).
    def act_with_value(
        self,
        obs: list[float],
        legal_actions: list[int],
        *,
        eval: bool = False,
        rng_key: Any = None,
        behavior_epsilon: float = 0.0,
    ) -> tuple[int, float, float]:
        """Fused (action, logprob, value) in one call.

        Semantically identical to calling :meth:`act` then :meth:`value`, but
        subclasses with a shared backbone (e.g. the MLP actor-critic) fuse the
        two reads into a single forward pass. The rollout hot path must call
        this rather than ``act`` + ``value`` separately.
        """
        action, logprob = self.act(
            obs, legal_actions, eval=eval, rng_key=rng_key, behavior_epsilon=behavior_epsilon
        )
        value = self.value(obs)
        return action, logprob, value

    # ---- weight snapshot / restore (hot-path + GPU-memory optimization) ----
    #
    # The IMPALA parameter hub (pipeline.parameter_hub) and the league's
    # promotion logic (league.manager) need *independent* copies of a policy's
    # trainable state. Centralizing the snapshot semantics on the Policy ABC
    # means each subclass picks the right device for the copy:
    #   - tabular: dict deepcopy (CPU-only, tiny).
    #   - NN:      state_dict copied to CPU tensors (no GPU memory accumulation
    #             in the hub's bounded history; restored back to device on load).
    # These methods are the single source of truth — ParameterHub and
    # LeagueManager delegate to them rather than reaching into policy internals.

    @abstractmethod
    def snapshot_state(self) -> dict[str, Any]:
        """Return an independent snapshot of the policy's trainable state.

        The returned dict is fully decoupled from ``self`` — later mutations to
        ``self`` must not affect it. Storage location (CPU/GPU) is the subclass'
        choice; the NN arm stores CPU tensors to avoid GPU-memory accumulation
        in long-lived stores (hub history, league pool).
        """
        ...

    @abstractmethod
    def restore_state(self, snapshot: dict[str, Any]) -> None:
        """Restore in-place from a snapshot produced by :meth:`snapshot_state`."""
        ...

    # NOTE: parameters() / train() / eval_mode() are deliberately NOT on the
    # ABC. nn.Module already provides parameters()/train()/eval(); redefining
    # them here would create MRO/signature conflicts for NN subclasses
    # (MLPSharedActorCritic multiple-inherits nn.Module + Policy). Tabular
    # policies carry their state in dicts read by the UpdateRule, so they have
    # no need for these hooks.


def copy_weights(src: Policy, dst: Policy) -> None:
    """Copy trainable state ``src`` -> ``dst`` via the Policy snapshot interface.

    This is the single generic weight-copy path (AGENTS.md §3.3: behavior lives
    on the base-class interface, never in isinstance branches). It works for
    every Policy subclass — tabular and NN alike — and fails loudly
    (``ValueError`` from :meth:`Policy.restore_state`) if the two policies'
    snapshot kinds are incompatible. A silent no-op is impossible by
    construction: both methods are abstract, so every concrete policy has them.
    """
    dst.restore_state(src.snapshot_state())


# ---------------------------------------------------------------------------
# Exploring behavior policy (mu) -- ONE implementation of the mixture and its
# inverse, because the two must round-trip exactly.
#
# Sampling from mu instead of pi does NOT change what ACH estimates: the loss
# divides by ``pi_old(a|s)``, so recording ``pi_old = mu(a|s)`` makes the
# importance weight cancel the sampling probability exactly and the expected
# per-visit update is unchanged (paper Eq. 29 p24). What changes is the
# information-set visitation ``rho``, which is the point -- see
# docs/liars_residual_floor.md.
#
# The paper's behavior policy is ``mu = pi`` (p24), so any epsilon > 0 is a
# deliberate deviation and warns via ACHFidelityWarning.
# ---------------------------------------------------------------------------


def behavior_prob(pi_a: float, n_legal: int, epsilon: float) -> float:
    """``mu(a|s) = (1-eps) * pi(a|s) + eps / |legal|``."""
    return (1.0 - epsilon) * pi_a + epsilon / n_legal


def target_ratio(mu_a: float, n_legal: int, epsilon: float) -> float:
    """``pi(a|s) / mu(a|s)``, recovered from the behavior probability.

    Written as ``(1 - u/mu) / (1 - eps)`` with ``u = eps/|legal|`` rather than
    the algebraically equal ``((mu - u)/(1 - eps)) / mu``: the subtraction
    ``mu - u`` cancels catastrophically exactly where it matters (rare actions,
    where ``mu -> u``), while ``u/mu`` stays accurate there and the result
    degrades gracefully to 0. Returns 1.0 at ``epsilon = 0`` by construction.
    """
    if epsilon <= 0.0:
        return 1.0
    return (1.0 - (epsilon / n_legal) / mu_a) / (1.0 - epsilon)


def entropy_of_probs(probs: list[float]) -> float:
    """Shannon entropy in nats of a probability vector (used by both algos).

    Lives here rather than under algos/ because both PPO and ACH consume it
    and it is a pure function of the policy's output distribution.
    """
    h = 0.0
    for p in probs:
        if p > 0.0:
            h -= p * math.log(p)
    return h


def masked_softmax(logits: list[float], legal_mask: list[bool]) -> list[float]:
    """Numerically stable softmax over only the masked-True entries.

    Args:
        logits: one per action (full action space).
        legal_mask: same length; True where the action is legal.

    Returns:
        Probabilities over the full action space (illegal entries are 0.0).
    """
    max_logit = float("-inf")
    for lg, ok in zip(logits, legal_mask, strict=True):
        if ok and lg > max_logit:
            max_logit = lg
    if max_logit == float("-inf"):
        # No legal actions — caller violated the "legal_actions never empty"
        # contract. Return uniform over the full space as a defensive fallback
        # (never reaches an all-zero vector, which would break downstream sampling).
        n = len(legal_mask)
        return [1.0 / n] * n if n else []
    exps = [
        math.exp(lg - max_logit) if m else 0.0 for lg, m in zip(logits, legal_mask, strict=True)
    ]
    s = sum(exps)
    return [e / s for e in exps]
