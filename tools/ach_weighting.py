"""The weight side of the exact ACH dynamics: reach factors and Theorem 1's range.

Split out of :mod:`tools.exact_ach` (AGENTS.md §3 rule 1) because it is a separate
responsibility: this module is about *what each information set's step weight is
and how much it moves*, while ``exact_ach`` is about the update the weight is
applied to.

The distinction the two helpers here exist to keep straight -- and the one the
repo's earlier reading of Theorem 1 lost -- is between:

  - the **theorem's weight** ``w_t(s) = f_p^{mu_t}(s)``, the reach under the
    *owner's own* behaviour only, and
  - the **applied product** ``f_p^{mu} * f_{-p}^{pi}``, which is what an update
    actually multiplies an advantage by.

Theorem 1 charges ``Delta * sum_s (w_h(s) - w_l(s)) / w_h(s)`` on the first of
those, bracketed **over iterations** for each information set separately (paper
p5; ICLR slides p8). Corollary 1 (p23) is the immediate consequence: hold the
behaviour policy still and ``w_l(s) = w_h(s)``, so the term vanishes and the
CFR-with-Hedge bound is recovered.
"""

from __future__ import annotations

import torch

from mjai.seqform.plan import realization_plans
from mjai.seqform.tree import SequenceForm


def own_reach(sf: SequenceForm, behavior: torch.Tensor) -> torch.Tensor:
    """``f_p(I)`` -- the owner's own probability of reaching each of its rows.

    This is the quantity Theorem 1's weight is. The opponent-and-chance factor is
    not part of ``w`` but of the counterfactual regret ``r^c_t(s, a) =
    f_{-p}^{pi_t}(s) A^{pi_t}(s, a)`` that ``w`` multiplies (paper p5), so only
    this factor is what Corollary 1 asks to hold still.
    """
    plans = realization_plans(sf, behavior)
    own = torch.zeros(sf.num_infosets, dtype=torch.float64)
    for player in range(sf.num_players):
        rows = sf.rows_of(player)
        if rows.numel() == 0:
            continue
        own[rows] = plans[player].index_select(0, sf.parent_sequence[rows])
    return own


class WeightRange:
    """Per-information-set range of a weight over the iterations it was applied.

    ``term2`` is reported for the *shape* of its dependence, never as a numeric
    prediction: it sums a per-row quantity of order 1 over every row, so on liars
    (24576 rows, payoff range 2) the bound is vacuous next to an exploitability of
    0.1. ``docs/liars_residual_floor.md`` §9.5 already declined to claim a
    quantitative correspondence and nothing here changes that -- what is claimed
    is the qualitative structure: a term that does not shrink with T, driven by how
    much the weights drift.
    """

    def __init__(self, num_infosets: int, delta: float) -> None:
        self.delta = delta
        self.lo = torch.full((num_infosets,), float("inf"), dtype=torch.float64)
        self.hi = torch.zeros(num_infosets, dtype=torch.float64)

    def update(self, weight: torch.Tensor) -> None:
        self.lo = torch.minimum(self.lo, weight)
        self.hi = torch.maximum(self.hi, weight)

    def stats(self) -> dict[str, float]:
        # Rows the schedule never gave any weight to are outside the theorem's
        # (0, 1] condition; it has nothing to say about them and neither do we.
        live = self.hi > 0.0
        if not bool(live.any()):
            return {"rows": 0.0}
        ratio = 1.0 - (self.lo[live] / self.hi[live])
        return {
            "rows": float(live.sum()),
            "span_mean": float(ratio.mean()),
            "span_max": float(ratio.max()),
            "term2": self.delta * float(ratio.sum()),
        }
