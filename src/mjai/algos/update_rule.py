"""Abstract :class:`UpdateRule` — the algorithm plug point (AGENTS.md §3, §4).

An UpdateRule takes a :class:`~mjai.algos.transition.Batch` (collected by the
self-play controller, which may be mirror or league) and performs one or more
gradient steps on the given :class:`~mjai.agents.base.Policy`. The rule is
**stateless w.r.t. self-play topology** — it does not know or care whether the
batch came from mirror self-play or a league opponent pool.

Adding an algorithm (AGENTS.md §4 "Algo rule") = new ``UpdateRule`` subclass.
No edits to Trainer or pipeline.

Two implementation families ship in Phase 1:
  - :mod:`mjai.algos.nn_updates` — :class:`NNPPOUpdate` (reference PPO) and
    :class:`NNACHUpdate` (paper-faithful ACH, AGENTS.md §1 D4) on torch MLPs.
  - :mod:`mjai.algos.tabular_updates` — tabular PPO / ACH (CFR+ wrapper) on
    dict-backed policies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from mjai.agents.base import Policy
from mjai.algos.transition import Batch, UpdateStats


@dataclass(frozen=True)
class AlgoConfig:
    """Shared hyperparameters consumed by every UpdateRule.

    All fields are wired from the experiment YAML (AGENTS.md §9 — no magic
    numbers in code). ACH-specific fields (``eta``, ``l_th``, ``ratio_eps``)
    are ignored by PPO; ``clip_eps`` lives on the PPO endpoint itself.

    Defaults follow the ACH paper's Appendix H.3 (p27-28) where applicable:
    eta=1.0, l_th=2.0 (p28 Table 8); ``ratio_eps`` is vacuous under synchronous
    single-threaded self-play (p28) and only matters for async sampling.

    Attributes:
        optimizer: ``"sgd"`` | ``"adam"`` | None. None = the endpoint's own
            default (ACH: SGD constant LR, paper p27; PPO: Adam eps=1e-5).
        max_grad_norm: grad-norm clip; ``<= 0`` disables (paper mentions no
            clipping, so the ACH reproduction configs disable it).
        gae_lambda: lambda for the rollout's per-player GAE (paper H.3 unspecified;
            0.95 follows the paper's Mahjong/FHP choice — spec assumption A1).
    """

    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95
    optimizer: str | None = None
    eta: float = 1.0
    l_th: float = 2.0
    ratio_eps: float = 0.5
    # ACH loss body uses the mean-centered logit (paper text, p24) when True;
    # False = raw logit (literal Algorithm 2). Probe toggle for spec ambiguity
    # A3/U1; the gate always uses the centered logit (paper is explicit there).
    loss_centered_logits: bool = True


class UpdateRule(ABC):
    """One gradient step per :meth:`step` call.

    Subclasses are constructed with their config + the Policy they update; they
    own their optimizer. The Trainer calls :meth:`step` repeatedly.
    """

    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    @abstractmethod
    def step(self, batch: Batch) -> UpdateStats:
        """Perform one update on ``self.policy`` using ``batch``.

        Returns stats to be logged to TensorBoard. Implementations may run
        multiple inner epochs over the batch (PPO does; ACH typically does one).
        """
        ...

    def state_dict(self) -> dict[str, object]:
        """Optimizer state, for checkpointing. Default: empty (tabular)."""
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore optimizer state. Default: noop (tabular)."""
