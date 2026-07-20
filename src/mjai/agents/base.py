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
    ) -> tuple[int, float]:
        """Choose an action.

        Args:
            obs: per-player observation vector (length = ``GameSpec.obs_size``).
            legal_actions: action ids legal at this state; never empty.
            eval: if True, act greedily (no exploration); else sample stochastically.
            rng_key: optional deterministic-state handle (seed/key) for reproducibility.

        Returns:
            (action_id, logprob). ``logprob`` is the log-probability of the
            chosen action under the policy (0.0 in eval mode is acceptable).
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

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist to ``path`` using the canonical ckpt manifest (AGENTS.md §10)."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore in-place from ``path`` (produced by :meth:`save`)."""
        ...

    # ---- Optional hooks (default impls where sensible) ----

    def parameters(self) -> list[Any]:
        """Trainable parameters; empty for tabular (handled via dicts by the
        UpdateRule). NN subclasses override to return torch tensors.
        """
        return []

    def train(self) -> None:
        """Set training mode. Optional override (NN subclasses); default noop."""

    def eval_mode(self) -> None:
        """Set eval mode. Optional override (NN subclasses); default noop."""


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
