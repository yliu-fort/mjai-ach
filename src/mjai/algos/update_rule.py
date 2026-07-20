"""Abstract :class:`UpdateRule` — the algorithm plug point (AGENTS.md §3, §4).

An UpdateRule takes a :class:`~mjai.algos.transition.Batch` (collected by the
self-play controller, which may be mirror or league) and performs one or more
gradient steps on the given :class:`~mjai.agents.base.Policy`. The rule is
**stateless w.r.t. self-play topology** — it does not know or care whether the
batch came from mirror self-play or a league opponent pool.

Adding an algorithm (AGENTS.md §4 "Algo rule") = new ``UpdateRule`` subclass.
No edits to Trainer or pipeline.

Two implementations ship in Phase 1:
  - :class:`mjai.algos.ppo.PPOUpdate` — clipped surrogate + value MSE + entropy
  - :class:`mjai.algos.ach.ACHUpdate` — REINFORCE + critic baseline + entropy,
    no PPO clipping (the Hedge/replicator limit; AGENTS.md §1 D4)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from mjai.agents.base import Policy
from mjai.algos.transition import Batch, UpdateStats


@dataclass(frozen=True)
class AlgoConfig:
    """Shared hyperparameters consumed by every UpdateRule.

    Concrete rules add their own config dataclasses; this holds only the fields
    common to PPO and ACH so the Trainer can construct either from one source.
    """

    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    discount: float = 1.0  # IIG returns-to-go is undiscounted over an episode
    gae_lambda: float = 0.95  # only used by PPO's GAE; ACH uses plain returns


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
