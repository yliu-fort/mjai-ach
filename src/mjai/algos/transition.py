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

from mjai.agents.base import Policy


@dataclass
class Transition:
    """One (s, a, r, ...) experience record from a self-play rollout."""

    obs: list[float]  # per-player observation vector
    legal_actions: list[int]  # action ids legal at this state
    action: int  # chosen action id (must be in legal_actions)
    logprob: float  # log P(action | obs) under the behavior policy
    value: float  # V(obs) baseline estimate at sample time
    reward: float  # terminal payoff attached at every step (Phase-1 games have
    # zero mid-episode rewards; downstream consumers use return_/advantage)
    return_: float  # Monte-Carlo (or bootstrapped) return-to-go
    advantage: float = 0.0  # GAE/return - value; filled in by the UpdateRule
    player: int = 0  # which player this transition belongs to
    # How many times this sample COUNTS in the loss. 1.0 = on-policy (the paper:
    # a sample's influence is exactly the probability its history was reached).
    # The rollout sets it to reach(h)^-kappa when RolloutConfig.sample_weight_kappa
    # is on, which tempers the training distribution from rho to rho^(1-kappa)
    # (docs/liars_residual_floor.md §8.4-8.5).
    weight: float = 1.0
    # WHICH POLICY produced this transition (the behavior policy that acted at
    # the decision point). Tagged by the rollout runner; routes the sample to
    # the right UpdateRule regardless of the physical seat it came from —
    # seat numbers stop identifying learners once seats are shuffled.
    producer: Policy | None = None


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
    # Producer identity table: producer_idx[i] indexes ``producers`` to name
    # the Policy that generated transition i. Empty/None for hand-built
    # batches that predate tagging; the rollout runner always populates both.
    producers: tuple[Policy, ...] = ()
    producer_idx: np.ndarray | None = None  # (B,) int8
    # (B,) float32 per-sample loss weights, or None for the unweighted batch.
    # None is not "all ones": it selects the plain ``.mean()`` reduction the
    # update rules have always taken, so the default trajectory stays
    # bit-identical (tests/unit/data/nn_updates_golden.json). ``make_batch``
    # only populates it when some transition actually carries a weight != 1.
    weights: np.ndarray | None = None

    @property
    def size(self) -> int:
        return int(self.obs.shape[0])

    def for_player(self, player: int) -> Batch:
        """Return a new Batch containing only transitions of ``player``.

        Seat-based filter used by eval (cross-play payoff of a fixed seat).
        Training routes by producer identity instead — see :meth:`for_producer`.
        """
        if self.size == 0:
            return self
        mask = self.players == player
        if mask.all():
            return self
        import numpy as np

        return self._take(np.nonzero(mask)[0])

    def for_producer(self, policy: Policy) -> Batch:
        """Return a new Batch containing only transitions produced by ``policy``.

        Identity-based routing: a learner trains on the samples ITS OWN policy
        generated, whichever physical seat it occupied (seat shuffle) and
        whoever else acted in the same episodes (frozen opponents are simply
        never selected). Matched by object identity, not equality.
        """
        if self.producer_idx is None or not self.producers:
            raise RuntimeError(
                "Batch.for_producer on a batch without producer tags: the rollout "
                "runner tags every transition with the acting policy; hand-built "
                "batches must set Transition.producer too (AGENTS.md §11: no "
                "silent fallback)."
            )
        idx = next((i for i, p in enumerate(self.producers) if p is policy), None)
        import numpy as np

        if idx is None:
            # The policy produced nothing in this batch — an empty selection is
            # the honest answer (mirrors for_player on a seat with no rows).
            return self._take(np.zeros((0,), dtype=np.int64))
        return self._take(np.nonzero(self.producer_idx == idx)[0])

    def _take(self, idx: np.ndarray) -> Batch:
        """Row-select into a new Batch, preserving the producer table."""
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
            producers=self.producers,
            producer_idx=None if self.producer_idx is None else self.producer_idx[idx],
            weights=None if self.weights is None else self.weights[idx],
        )


def make_batch(transitions: Sequence[Transition], num_actions: int) -> Batch:
    """Stack a sequence of transitions into a :class:`Batch`.

    Args:
        transitions: the records to stack (may be empty).
        num_actions: width of the action space; drives the legal_mask shape.

    Producer tags are deduplicated BY IDENTITY into ``Batch.producers`` +
    ``producer_idx``. Tags are all-or-nothing: a batch mixing tagged and
    untagged transitions is a caller bug and fails loudly (§11).

    ``Batch.weights`` stays ``None`` while every transition carries the default
    weight 1.0, which is what keeps the unweighted path on its original
    reduction rather than on an arithmetically-equal weighted one.
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
    producers: list[Policy] = []
    producer_idx = np.full((n,), -1, dtype=np.int8)
    for i, t in enumerate(transitions):
        if t.producer is None:
            continue
        for j, p in enumerate(producers):
            if p is t.producer:
                producer_idx[i] = j
                break
        else:
            producers.append(t.producer)
            producer_idx[i] = len(producers) - 1
    n_tagged = int((producer_idx >= 0).sum())
    if 0 < n_tagged < n:
        raise ValueError(
            f"mixed producer tags: {n_tagged}/{n} transitions tagged; "
            "tag every transition or none (AGENTS.md §11)"
        )
    legal_mask = np.zeros((n, num_actions), dtype=bool)
    for i, t in enumerate(transitions):
        for a in t.legal_actions:
            legal_mask[i, a] = True
    weights = np.asarray([t.weight for t in transitions], dtype=np.float32)
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
        producers=tuple(producers),
        producer_idx=producer_idx if n_tagged == n else None,
        weights=None if bool((weights == 1.0).all()) else weights,
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
