"""Running-average policy in sequence-form coordinates (AGENTS.md D16).

ACH inherits CFR's guarantee, and that guarantee is about the **average**
strategy: `exploitability(avg) = O(T^-1/2)` (Fu et al. ICLR 2022, Theorem 1,
`docs/paper_spec_ach.md` §1.1). But `docs/reproduce_report.md` records that the
reproduction plots the *current* policy `pi = softmax(y)` — the paper's own
choice for its figures, and the right object for a last-iterate study, but not
the object the theorem covers. So the pipeline sanity anchor the research plan
asks for (Generative-ach.md §5.0(7), "a known O(T^-1/2) curve") does not exist
in the repo until something tracks the average. This module is that something.

**Averaging realization plans, not behaviour probabilities.** The two are not
the same and only the first is the CFR average strategy. Averaging behaviour
probabilities information-set by information set gives equal weight to an
information set the policy reaches half the time and one it reaches never;
averaging realization plans weights each by how often it is actually reached,
which is what makes the average a strategy whose exploitability is bounded.
Sequence form makes this a one-liner — the realization plan is linear in the
reach, so the reach-weighted average of behaviour strategies *is* the plain
arithmetic mean of realization plans.

Recovering behaviour from the averaged plan is `P(a | I) = x(I, a) / x(parent I)`.
Where the averaged parent reach is zero the information set was never reached by
any iterate, exploitability cannot depend on what is played there, and we emit
the uniform distribution — CFR's own convention, stated here because "uniform"
is a choice and not a derivation.

Note the reach in question is the player's **own**: a realization plan is the
product of that player's action probabilities along the sequence, with chance
and the opponents excluded. That is exactly what makes the average well defined
without reference to whoever the iterates happened to play against — and it is
why "the opponent never bets" does not make a row unreached, while "I never
pass" does.

This is an evaluation-side module. Nothing in `mjai.pach` may import it; the
layering enforces that (AGENTS.md §2).
"""

from __future__ import annotations

import numpy as np
import torch

from mjai.agents.base import Policy
from mjai.games.loader import GameSpec
from mjai.seqform.plan import realization_plans, validate_behavior
from mjai.seqform.tree import EMPTY_SEQUENCE, SequenceForm, build_sequence_form

# Parent reaches below this count as "never reached"; the recovered row is then
# uniform. Well clear of float64 round-off in a product of a few dozen factors.
_UNREACHED = 1e-300


def behavior_of(sf: SequenceForm, policy: Policy) -> torch.Tensor:
    """A :class:`Policy`'s behaviour strategy on ``sf``'s information sets.

    One batched query over the whole information-set enumeration, mirroring
    :func:`mjai.eval.nash.tabular_view_of`. ``action_logits_batch`` hands over
    float64 (AGENTS.md D14a), so nothing is truncated here; an NN policy's
    residual error is its own float32 weights, which is a property of the
    policy rather than of this measurement.
    """
    obs = sf.infoset_observation.numpy()
    mask = sf.legal_mask.numpy()
    logits = np.asarray(policy.action_logits_batch(obs, mask), dtype=np.float64)
    masked = torch.from_numpy(logits).masked_fill(~sf.legal_mask, float("-inf"))
    return torch.softmax(masked, dim=1)


class RealizationAverage:
    """Accumulates realization plans so the average strategy can be scored.

    Memory is one float64 vector per player — 33 doubles per seat on 3p Kuhn,
    1093 on Leduc — and is independent of how many iterates are folded in. No
    checkpoint pool, nothing to bound (AGENTS.md §8).

    Usage::

        tracker = RealizationAverage(sf)
        tracker.update(behavior_of(sf, policy))    # once per eval point
        curve_point = tracker.nash_conv()
    """

    def __init__(self, sf: SequenceForm) -> None:
        self.sf = sf
        self._sums = [torch.zeros(n, dtype=torch.float64) for n in sf.num_sequences]
        self._weight = 0.0
        self._count = 0

    @property
    def num_updates(self) -> int:
        """How many iterates have been folded in."""
        return self._count

    def update(self, behavior: torch.Tensor, *, weight: float = 1.0) -> None:
        """Fold one iterate's behaviour strategy into the average.

        ``weight`` supports linear averaging (weight = t), which CFR+ uses and
        which converges faster in practice. The default of 1.0 is the uniform
        average the theorem is stated for; if you change it, say so in the
        figure caption, because the two curves are different objects.

        **Which iterate gets which weight matters.** Reproducing OpenSpiel's
        CFR+ average means folding in the policy that was current at the *start*
        of iteration t with weight t. Folding in the post-update policy instead
        — the natural thing to write — is an off-by-one worth 3.3e-5 in NashConv
        after 50 CFR+ iterations on Kuhn, about 0.6% relative, which is small
        enough to look like noise and large enough to matter on a log-scale
        convergence plot. ``tests/unit/test_eval_average_policy.py`` pins both.
        """
        if weight <= 0:
            raise ValueError(f"weight must be positive, got {weight}")
        validate_behavior(self.sf, behavior)
        plans = realization_plans(self.sf, behavior)
        for player, plan in enumerate(plans):
            self._sums[player] = self._sums[player] + weight * plan.detach()
        self._weight += weight
        self._count += 1

    def average_plans(self) -> list[torch.Tensor]:
        """The averaged realization plan per player."""
        if self._weight <= 0:
            raise ValueError("no iterates have been averaged yet")
        return [total / self._weight for total in self._sums]

    def average_behavior(self) -> torch.Tensor:
        """Behaviour strategy recovered from the averaged realization plans.

        ``P(a | I) = x(I, a) / x(parent I)``, uniform on information sets no
        iterate ever reached (see the module docstring).
        """
        plans = self.average_plans()
        behavior = torch.zeros(self.sf.num_infosets, self.sf.max_actions, dtype=torch.float64)
        for row in range(self.sf.num_infosets):
            player = int(self.sf.infoset_player[row])
            legal = self.sf.legal_mask[row]
            children = self.sf.sequence_of[row][legal]
            parent_reach = float(plans[player][self.sf.parent_sequence[row]])
            if parent_reach <= _UNREACHED:
                behavior[row, legal] = 1.0 / int(legal.sum())
            else:
                behavior[row, legal] = plans[player][children] / parent_reach
        return behavior

    def nash_conv(self) -> float:
        """NashConv of the average strategy — the D16 anchor curve's y value."""
        from mjai.seqform.plan import nash_conv

        return float(nash_conv(self.sf, self.average_behavior()))

    def metrics(self) -> dict[str, float]:
        """Curve row fields, named to match :mod:`mjai.eval.nash`'s convention.

        ``exploitability`` is only emitted where the identity holds: 2-player
        constant-sum. At n >= 3 there is no minimax value to be exploitable
        relative to, so the key is simply absent rather than misleading.
        """
        value = self.nash_conv()
        out = {"avg_nash_conv": value, "avg_iterates": float(self._count)}
        if self.sf.num_players == 2:
            out["avg_exploitability"] = value / 2.0
        return out


class AveragePolicyTracker:
    """Run-scoped adapter: hand it a :class:`Policy` per eval point, get metrics.

    Owns the game's :class:`SequenceForm` (built once) and the weighting choice,
    so the training loop carries one object instead of three. Construction fails
    loudly on a game with no sequence form — a run that asked for the average
    anchor and silently did not get it would be worse than one that stopped
    (AGENTS.md §11).
    """

    def __init__(self, spec: GameSpec, *, weighting: str = "uniform") -> None:
        if weighting not in ("uniform", "linear"):
            raise ValueError(f"bad weighting {weighting!r}; want uniform|linear")
        self.spec = spec
        self.weighting = weighting
        self.sequence_form = build_sequence_form(spec)
        self.average = RealizationAverage(self.sequence_form)

    def observe(self, policy: Policy) -> dict[str, float]:
        """Fold ``policy`` into the average and return the curve-row metrics."""
        index = self.average.num_updates + 1
        weight = float(index) if self.weighting == "linear" else 1.0
        self.average.update(behavior_of(self.sequence_form, policy), weight=weight)
        return self.average.metrics()


def average_plan_is_normalized(sf: SequenceForm, plans: list[torch.Tensor]) -> bool:
    """Whether every averaged plan still starts from the empty sequence at 1.

    A convex combination of realization plans is a realization plan, so this
    must hold; it is exported so tests and diagnostics can assert it cheaply.
    """
    return all(abs(float(plan[EMPTY_SEQUENCE]) - 1.0) < 1e-12 for plan in plans)
