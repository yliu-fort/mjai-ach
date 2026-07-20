"""Transition and batch data structures shared by all UpdateRules.

A :class:`Transition` is one player's experience at one decision point: the
observation they acted on, the legal action set, the action chosen, its
log-probability (under the behavior policy that sampled it), the value
baseline estimate, and the Monte-Carlo return they actually received.

Keeping these as plain dataclasses (not torch tensors) means the same batch
format works for both the tabular and NN paths; each UpdateRule converts to
whatever representation it needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Transition:
    """One (s, a, r, ...) experience record from a self-play rollout."""

    obs: list[float]  # per-player observation vector
    legal_actions: list[int]  # action ids legal at this state
    action: int  # chosen action id (must be in legal_actions)
    logprob: float  # log P(action | obs) under the behavior policy
    value: float  # V(obs) baseline estimate at sample time
    reward: float  # immediate reward received
    return_: float  # Monte-Carlo (or bootstrapped) return-to-go
    advantage: float = 0.0  # GAE/return - value; filled in by the UpdateRule
    player: int = 0  # which player this transition belongs to


@dataclass
class Batch:
    """A collection of :class:`Transition` records, materialized as numpy arrays.

    Built once from a list of transitions; UpdateRules read the pre-stacked
    arrays rather than re-converting on every call. Indexing is along axis 0.
    """

    obs: np.ndarray  # (B, obs_size) float32
    legal_actions: list[list[int]]  # length B
    actions: np.ndarray  # (B,) int64
    logprobs: np.ndarray  # (B,) float32  (behavior-policy log-probs)
    values: np.ndarray  # (B,) float32
    returns: np.ndarray  # (B,) float32
    advantages: np.ndarray  # (B,) float32
    legal_mask: np.ndarray  # (B, num_actions) bool — True where legal
    players: np.ndarray  # (B,) int8 — which seat each transition belongs to
    num_actions: int  # action-space width (for mask shape)

    @property
    def size(self) -> int:
        return int(self.obs.shape[0])

    def for_player(self, player: int) -> Batch:
        """Return a new Batch containing only transitions of ``player``.

        Used by the league controller to train each learner on its own seat's
        transitions only (the opponent seat belongs to a different learner).
        """
        if self.size == 0:
            return self
        mask = self.players == player
        if mask.all():
            return self
        import numpy as np

        idx = np.nonzero(mask)[0]
        return Batch(
            obs=self.obs[idx],
            legal_actions=[self.legal_actions[i] for i in idx],
            actions=self.actions[idx],
            logprobs=self.logprobs[idx],
            values=self.values[idx],
            returns=self.returns[idx],
            advantages=self.advantages[idx],
            legal_mask=self.legal_mask[idx],
            players=self.players[idx],
            num_actions=self.num_actions,
        )


def make_batch(transitions: Sequence[Transition], num_actions: int) -> Batch:
    """Stack a sequence of transitions into a :class:`Batch`.

    Args:
        transitions: the records to stack (may be empty).
        num_actions: width of the action space; drives the legal_mask shape.
    """
    n = len(transitions)
    if n == 0:
        return Batch(
            obs=np.zeros((0, 0), dtype=np.float32),
            legal_actions=[],
            actions=np.zeros((0,), dtype=np.int64),
            logprobs=np.zeros((0,), dtype=np.float32),
            values=np.zeros((0,), dtype=np.float32),
            returns=np.zeros((0,), dtype=np.float32),
            advantages=np.zeros((0,), dtype=np.float32),
            legal_mask=np.zeros((0, num_actions), dtype=bool),
            players=np.zeros((0,), dtype=np.int8),
            num_actions=num_actions,
        )
    legal_mask = np.zeros((n, num_actions), dtype=bool)
    for i, t in enumerate(transitions):
        for a in t.legal_actions:
            legal_mask[i, a] = True
    return Batch(
        obs=np.asarray([t.obs for t in transitions], dtype=np.float32),
        legal_actions=[list(t.legal_actions) for t in transitions],
        actions=np.asarray([t.action for t in transitions], dtype=np.int64),
        logprobs=np.asarray([t.logprob for t in transitions], dtype=np.float32),
        values=np.asarray([t.value for t in transitions], dtype=np.float32),
        returns=np.asarray([t.return_ for t in transitions], dtype=np.float32),
        advantages=np.asarray([t.advantage for t in transitions], dtype=np.float32),
        legal_mask=legal_mask,
        players=np.asarray([t.player for t in transitions], dtype=np.int8),
        num_actions=num_actions,
    )


@dataclass
class UpdateStats:
    """Returned by every UpdateRule.step(); logged to TensorBoard."""

    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float = 0.0  # PPO tracks this; ACH leaves 0
    clip_frac: float = 0.0  # PPO only
    explained_variance: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)
