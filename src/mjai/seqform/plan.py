"""Realization plans, exact expected payoffs, best responses and NashConv.

Everything here is float64 (AGENTS.md D19) and differentiable end to end, so the
research plan's Oracle track can put ``torch.autograd`` straight through the
exact expectation instead of estimating a gradient (研究计划 §3.3). No sampling
happens anywhere in this module.

The three exported quantities:

- :func:`realization_plans` — behaviour strategy -> one realization vector per
  player, by a forward pass down the sequence tree.
- :func:`expected_returns` — the multilinear terminal sum, exactly.
- :func:`best_response_value` / :func:`nash_conv` — the exact best response by a
  backward pass over the same tree. This is the primary equilibrium certificate
  (研究计划 §4.1) and one leg of the D14 parity check.

**NashConv, not exploitability.** Exploitability = NashConv / n is an identity
for 2-player constant-sum games only. 3p Kuhn is constant-sum but multiplayer,
where there is no minimax value to be exploitable *relative to*; this module
therefore reports NashConv and leaves the division to callers who have checked
they are allowed to do it.
"""

from __future__ import annotations

import torch

from mjai.seqform.tree import EMPTY_SEQUENCE, NO_SEQUENCE, SequenceForm

# Row sums of a behaviour strategy must be 1 to this tolerance. Tight enough to
# catch a real bug, loose enough for float64 round-off in a 13-action row.
_SIMPLEX_TOL = 1e-9


class InvalidBehaviorError(ValueError):
    """A purported behaviour strategy is not a distribution over legal actions.

    AGENTS.md D15: OpenSpiel's own ``nash_conv`` silently accepts probabilities
    greater than 1 and returns a *negative* NashConv (measured on the Kuhn
    alpha-family at alpha = 0.4, where 3*alpha = 1.2, giving -6.7e-2). A metric
    that reports a plausible-looking number for an impossible policy is worse
    than one that raises, so we raise.
    """


def validate_behavior(sf: SequenceForm, behavior: torch.Tensor) -> None:
    """Check ``behavior`` is a distribution over legal actions at every row.

    Args:
        sf: the game's sequence form.
        behavior: float ``[num_infosets, max_actions]``.

    Raises:
        InvalidBehaviorError: on a shape mismatch, a negative probability, mass
            on an illegal action, or a row that does not sum to 1.
    """
    if behavior.shape != (sf.num_infosets, sf.max_actions):
        raise InvalidBehaviorError(
            f"{sf.game_name}: behaviour has shape {tuple(behavior.shape)}, want "
            f"{(sf.num_infosets, sf.max_actions)}"
        )
    if bool((behavior < 0).any()):
        raise InvalidBehaviorError(f"{sf.game_name}: negative action probability")
    illegal_mass = behavior.masked_select(~sf.legal_mask)
    if illegal_mass.numel() and bool((illegal_mass.abs() > _SIMPLEX_TOL).any()):
        raise InvalidBehaviorError(
            f"{sf.game_name}: non-zero probability on an illegal action "
            f"(max {float(illegal_mass.abs().max()):.3e})"
        )
    row_sums = behavior.masked_fill(~sf.legal_mask, 0.0).sum(dim=1)
    worst = float((row_sums - 1.0).abs().max())
    if worst > _SIMPLEX_TOL:
        row = int((row_sums - 1.0).abs().argmax())
        raise InvalidBehaviorError(
            f"{sf.game_name}: information set {sf.infoset_keys[row]!r} sums to "
            f"{float(row_sums[row]):.12f}, not 1 (off by {worst:.3e})"
        )


def behavior_from_logits(sf: SequenceForm, logits: torch.Tensor) -> torch.Tensor:
    """Masked softmax over legal actions, row by row.

    This is the map the pACH mother's output goes through: theta is a logit
    table, and the behaviour strategy is its row-wise softmax restricted to
    legal actions. Illegal slots come back exactly 0, so the result passes
    :func:`validate_behavior` by construction.
    """
    if logits.shape != (sf.num_infosets, sf.max_actions):
        raise InvalidBehaviorError(
            f"{sf.game_name}: logits have shape {tuple(logits.shape)}, want "
            f"{(sf.num_infosets, sf.max_actions)}"
        )
    masked = logits.to(torch.float64).masked_fill(~sf.legal_mask, float("-inf"))
    return torch.softmax(masked, dim=1)


def realization_plans(sf: SequenceForm, behavior: torch.Tensor) -> list[torch.Tensor]:
    """Realization probability per sequence, one float64 vector per player.

    ``x[0] = 1`` (the empty sequence) and ``x[(I, a)] = x[parent(I)] * P(a | I)``.
    Computed level by level so the cost is the tree's own-action depth rather
    than its information-set count, and out of place so autograd can follow it.
    """
    behavior = behavior.to(torch.float64)
    flat = behavior.reshape(-1)
    plans = [
        torch.cat([torch.ones(1, dtype=torch.float64), torch.zeros(n - 1, dtype=torch.float64)])
        for n in sf.num_sequences
    ]
    for rows in sf.level_rows:
        if rows.numel() == 0:
            continue
        for player in range(sf.num_players):
            player_rows = rows[sf.infoset_player[rows] == player]
            if player_rows.numel() == 0:
                continue
            mask = sf.legal_mask[player_rows]
            # Flatten the (row, action) grid down to the legal pairs only.
            pair_rows, pair_actions = torch.nonzero(mask, as_tuple=True)
            global_rows = player_rows[pair_rows]
            parents = sf.parent_sequence[global_rows]
            targets = sf.sequence_of[global_rows, pair_actions]
            slots = global_rows * sf.max_actions + pair_actions
            values = plans[player].index_select(0, parents) * flat.index_select(0, slots)
            plans[player] = plans[player].scatter(0, targets, values)
    return plans


def expected_returns(sf: SequenceForm, plans: list[torch.Tensor]) -> torch.Tensor:
    """Exact expected return per player: float64 ``[num_players]``.

    The multilinear terminal sum. With one term per terminal history this is
    30 products on 2p Kuhn and 312 on 3p Kuhn — there is no approximation here
    and no sampling error to quote.
    """
    reach = sf.terminal_chance.clone()
    for player in range(sf.num_players):
        reach = reach * plans[player].index_select(0, sf.terminal_sequence[:, player])
    return reach @ sf.terminal_utility


def sequence_payoff_coefficients(
    sf: SequenceForm, plans: list[torch.Tensor], player: int
) -> torch.Tensor:
    """Payoff coefficient per sequence of ``player``, holding others fixed.

    ``w[sigma] = sum over terminals reached through sigma of chance * util *
    prod over other players of their realization probability``. By construction
    ``<w, x_player> == expected_returns(...)[player]``, which is Lemma 1 made
    concrete: the game is exactly linear in one player's realization plan, so a
    linear critic on these coordinates is realizable rather than approximate.
    """
    weight = sf.terminal_chance * sf.terminal_utility[:, player]
    for other in range(sf.num_players):
        if other == player:
            continue
        weight = weight * plans[other].index_select(0, sf.terminal_sequence[:, other])
    coefficients = torch.zeros(sf.num_sequences[player], dtype=torch.float64)
    return coefficients.index_add(0, sf.terminal_sequence[:, player], weight)


def best_response_value(sf: SequenceForm, plans: list[torch.Tensor], player: int) -> torch.Tensor:
    """Exact best-response value for ``player`` against the others' plans.

    Maximizes the linear objective of :func:`sequence_payoff_coefficients` over
    the sequence-form polytope by backward induction:

        V(sigma) = w(sigma) + sum over information sets I with parent sigma of
                   max over legal a of V(sigma_{I,a})

    Walking ``level_rows`` in reverse guarantees every child value is final
    before its parent consumes it. Exact, not iterative — no convergence
    threshold to argue about.
    """
    values = sequence_payoff_coefficients(sf, plans, player)
    for rows in reversed(sf.level_rows):
        player_rows = rows[sf.infoset_player[rows] == player]
        if player_rows.numel() == 0:
            continue
        children = sf.sequence_of[player_rows]  # [k, max_actions], NO_SEQUENCE if illegal
        gathered = values.index_select(0, children.clamp(min=0).reshape(-1))
        child_values = gathered.reshape(children.shape).masked_fill(
            children == NO_SEQUENCE, float("-inf")
        )
        best = child_values.max(dim=1).values
        values = values.index_add(0, sf.parent_sequence[player_rows], best)
    return values[EMPTY_SEQUENCE]


def infoset_values(
    sf: SequenceForm, behavior: torch.Tensor, plans: list[torch.Tensor] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact ``(V(I), cf(I))`` per information set under ``behavior``.

    ``V(I)`` is what a critic's ``V(s)`` estimates and never quite reaches: the
    owner's expected return **conditional on the play having arrived at I**. It
    is the same backward induction as :func:`best_response_value` with the
    ``max`` replaced by the owner's own expectation, run twice -- once with the
    terminal utilities and once with those utilities replaced by 1:

        V(sigma)  = w(sigma)  + sum_{I: parent(I)=sigma} cfv(I)
        cfv(I)    = sum_a P(a|I) * V(sigma_{I,a})

    The second run returns ``cf(I)``, the counterfactual mass reaching I (chance
    times every OTHER player's realization probability). Their ratio is the
    conditional expectation, because the owner's own reach is by definition
    constant across the histories of one of its information sets and so cancels:

        V(I) = cfv_utility(I) / cf(I)

    Exact, non-iterative and float64. ``cf(I) == 0`` -- an information set the
    opponents' strategy shuts out entirely -- returns ``V(I) = 0``; there is no
    conditional expectation to report there, and no sampler would ever ask.

    Returns ``(values, cf)``, both ``[num_infosets]``, indexed by sequence-form
    row. ``plans`` may be passed in when the caller already has them.
    """
    behavior = behavior.to(torch.float64)
    if plans is None:
        plans = realization_plans(sf, behavior)
    values = torch.zeros(sf.num_infosets, dtype=torch.float64)
    cf = torch.zeros(sf.num_infosets, dtype=torch.float64)
    for player in range(sf.num_players):
        util = sequence_payoff_coefficients(sf, plans, player)
        # Same coefficients with utility == 1: the counterfactual probability
        # mass, which is the normalizer the conditional expectation needs.
        mass = _sequence_mass_coefficients(sf, plans, player)
        for rows in reversed(sf.level_rows):
            player_rows = rows[sf.infoset_player[rows] == player]
            if player_rows.numel() == 0:
                continue
            probs = behavior[player_rows]
            children = sf.sequence_of[player_rows]
            legal = children != NO_SEQUENCE
            safe = children.clamp(min=0)
            for source, target in ((util, values), (mass, cf)):
                child = source.index_select(0, safe.reshape(-1)).reshape(children.shape)
                target[player_rows] = (probs * child).masked_fill(~legal, 0.0).sum(dim=1)
            parents = sf.parent_sequence[player_rows]
            util = util.index_add(0, parents, values[player_rows])
            mass = mass.index_add(0, parents, cf[player_rows])
    return values / cf.clamp(min=1e-300), cf


def _sequence_mass_coefficients(
    sf: SequenceForm, plans: list[torch.Tensor], player: int
) -> torch.Tensor:
    """:func:`sequence_payoff_coefficients` with every terminal utility set to 1."""
    weight = sf.terminal_chance.clone()
    for other in range(sf.num_players):
        if other == player:
            continue
        weight = weight * plans[other].index_select(0, sf.terminal_sequence[:, other])
    coefficients = torch.zeros(sf.num_sequences[player], dtype=torch.float64)
    return coefficients.index_add(0, sf.terminal_sequence[:, player], weight)


def nash_conv(sf: SequenceForm, behavior: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
    """Sum over players of (best-response value - value under ``behavior``).

    The primary equilibrium certificate (研究计划 §4.1). Zero exactly at a Nash
    equilibrium, positive otherwise; for 2p constant-sum games it is twice the
    exploitability, but see the module docstring before dividing at n >= 3.

    ``validate`` runs :func:`validate_behavior` first (D15). Turn it off only
    inside a loop that has already validated the same rows.
    """
    if validate:
        validate_behavior(sf, behavior)
    plans = realization_plans(sf, behavior)
    values = expected_returns(sf, plans)
    total = torch.zeros((), dtype=torch.float64)
    for player in range(sf.num_players):
        total = total + best_response_value(sf, plans, player) - values[player]
    return total
