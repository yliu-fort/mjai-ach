"""An exact ``V(s)`` for the rollout's advantage baseline — the oracle-critic arm.

``docs/liars_residual_floor.md`` §8.9 eliminated three of the four candidate
explanations for why reach-tempered sample weights help in every offline setting
and hurt in RL. The survivor is **advantage-estimation quality**: at information
sets the policy rarely reaches, the learned critic's error is not zero-mean
noise but bias, and a weight that emphasizes exactly those information sets
multiplies the bias along with everything else.

This replaces the learned baseline with the exact conditional expectation
:func:`mjai.seqform.plan.infoset_values` computes, so a paired kappa=0 / kappa>0
run differs from the RL arms only in that the critic is right. It is an
instrument, not an algorithm: it needs the whole game tree and it reads the
opponent's strategy, so nothing here is available at Mahjong scale or to any
agent that has to learn from play alone. Runs that switch it on emit an
``ACHFidelityWarning`` and are not comparable to any paper curve.

Cost, measured on ``liars_dice1`` (24576 information sets): one refresh is a
full-table policy forward plus the backward induction, 10.3 + 12.1 ms, and a
refresh happens once per gradient step -- about +0.95 h over a 153k-step run.
"""

from __future__ import annotations

import torch

from mjai.agents.base import Policy
from mjai.eval.average_policy import behavior_of
from mjai.games.loader import GameSpec
from mjai.seqform.plan import infoset_values
from mjai.seqform.tree import build_sequence_form


class ExactValueOracle:
    """Exact ``V(s)`` under the CURRENT joint strategy, refreshed per update.

    ``V`` is a property of the whole profile, not of one player, so
    :meth:`refresh` takes the single policy that occupies both seats. That
    restricts this to mirror self-play, and :meth:`refresh` says so rather than
    quietly returning a value computed against the wrong opponent.
    """

    def __init__(self, spec: GameSpec) -> None:
        self.spec = spec
        self.sf = build_sequence_form(spec)
        # Observation -> sequence-form row. Both sides of this lookup come from
        # the same GameSpec.obs_tensor call (mjai.seqform.tree fills
        # infoset_observation with it), so exact float equality is the right
        # match and a miss is a real bug rather than a rounding accident.
        self._row: dict[tuple[float, ...], int] = {
            tuple(float(x) for x in obs): i for i, obs in enumerate(self.sf.infoset_observation)
        }
        self._values = torch.zeros(self.sf.num_infosets, dtype=torch.float64)
        self.refreshes = 0

    def refresh(self, learner: Policy, opponent: Policy) -> None:
        """Recompute the value table for the profile about to be played."""
        if learner is not opponent:
            raise ValueError(
                "ExactValueOracle is defined for a joint strategy profile and is "
                "wired for mirror self-play, where one policy occupies both "
                "seats; got two different policies (a league round would need "
                "one behaviour table per seat)"
            )
        self._values = infoset_values(self.sf, behavior_of(self.sf, learner))[0]
        self.refreshes += 1

    def value(self, obs: list[float]) -> float:
        key = tuple(obs)
        row = self._row.get(key)
        if row is None:
            raise KeyError(
                f"observation not in {self.spec.name}'s information-set enumeration; "
                "the oracle and the rollout disagree about what a state looks like"
            )
        return float(self._values[row])
