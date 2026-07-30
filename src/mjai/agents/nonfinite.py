"""The divergence guard: what a run does when a network produces inf/nan.

One concern, two callers, so it lives in its own module rather than in either of
them: :mod:`mjai.agents.mlp` refuses to sample from a diverged network, and
:class:`mjai.algos.nn_updates.NNActorCriticUpdate` refuses to continue past the
update that diverged it. Both raise; nothing here clamps, zeroes, skips a batch,
or reinitializes anything (AGENTS.md §11 — a diverged run is over, and continuing
it would report numbers produced by a broken network).

The module sits in the ``agents`` layer, below ``algos``, so the update rule can
reach it. That is also why :func:`assert_finite_update` takes plain floats rather
than an :class:`~mjai.algos.transition.UpdateStats`: the scalars it reads are all
host-side by then, and taking the dataclass would point this layer upward.

Motivating failure (docs/brps_mlp_nonconvergence.md §4): 4 of 30 BRPS runs died
inside ``torch.multinomial`` with ``probability tensor contains either inf, nan
or element < 0`` — no update named, no knob implicated.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn


class NonFiniteNetworkError(RuntimeError):
    """A network produced inf/nan, so the run is over — raised, never repaired.

    ``HINT`` is the shared tail of every message this module builds: what a
    diverged run should read first. It lives on the exception so the raise sites
    cannot drift apart.
    """

    HINT = (
        "The usual cause is the ACH policy term's unbounded importance weight: "
        "1/pi_old(a|s) has no bound under the paper-faithful default "
        "(AlgoConfig.iw_clip=None), so one rare action sampled at pi_old ~ 1e-3 "
        "contributes a gradient ~1e4 and overflows the float32 logits in a single "
        "SGD step. Read the run's train/iw_max and train/pterm_max telemetry at the "
        "last update before this one. Root-cause write-up and the measured example "
        "(4 of 30 BRPS runs): docs/brps_mlp_nonconvergence.md §4. Setting "
        "AlgoConfig.iw_clip bounds it, at the cost of paper fidelity."
    )


def nonfinite_summary(name: str, t: torch.Tensor) -> str | None:
    """``"name: 2 nan, 1 inf, of 3; first values=[...]"`` — None when all-finite.

    Truncates at 8 values: the caller is building an exception message, and a
    wide action space or a hidden layer would otherwise bury the diagnosis.
    """
    if bool(torch.isfinite(t).all()):
        return None
    n_nan = int(torch.isnan(t).sum().item())
    n_inf = int(torch.isinf(t).sum().item())
    flat = t.detach().reshape(-1)[:8].tolist()
    tail = ", ..." if t.numel() > 8 else ""
    return f"{name}: {n_nan} nan, {n_inf} inf, of {t.numel()}; first values={flat}{tail}"


def nonfinite_params(module: nn.Module) -> list[str]:
    """Per-parameter reports for every non-finite tensor (empty when healthy).

    Two passes on purpose. The first is one fused ``isfinite().all()`` per
    parameter reduced to a single scalar — one device sync for the whole model,
    which is what makes this affordable once per update. The second pass builds
    messages and only ever runs on a model that is already broken.
    """
    named = list(module.named_parameters())
    if not named:
        return []
    flags = torch.stack([torch.isfinite(p.detach()).all() for _, p in named])
    if bool(flags.all().item()):
        return []
    return [r for r in (nonfinite_summary(f"param {n}", p) for n, p in named) if r is not None]


def nonfinite_action_dist_error(
    module: nn.Module,
    *,
    logits: torch.Tensor,
    probs: torch.Tensor,
    obs: list[float],
    legal_actions: list[int],
) -> NonFiniteNetworkError:
    """Build (do not raise) the diagnosis for a decision point that went bad.

    Returned rather than raised so the call site can attach ``from exc`` when it
    is translating torch's own error, and so every scan in here runs only on the
    failure path. That matters: the callers are the rollout's hot loop, and an
    unconditional ``isfinite`` check there measured +24% per decision point
    (40.2 -> 49.9 us on CPU, 128-wide net). What the callers do instead is
    validate the scalars they *already* sync — the sampled action's log-prob and
    the value — which costs nothing and is equally conclusive, because a
    non-finite logit anywhere makes the whole log-softmax non-finite.
    """
    reports = [
        r
        for r in (nonfinite_summary("logits", logits), nonfinite_summary("probs", probs))
        if r is not None
    ] or ["logits and probs are finite; the non-finite value came out of the sampled log-prob"]
    params = nonfinite_params(module)
    where = "the WEIGHTS are already non-finite" if params else "the weights are still finite"
    detail = " (" + "; ".join(params) + ")" if params else ""
    return NonFiniteNetworkError(
        f"non-finite action distribution at obs={obs}, legal_actions={legal_actions}; "
        + "; ".join(reports)
        + f". Diagnosis: {where}{detail}. {NonFiniteNetworkError.HINT}"
    )


def assert_finite_update(
    module: nn.Module,
    *,
    forward_scalars: Mapping[str, float],
    grad_norm: float,
    probes: Mapping[str, float],
) -> None:
    """Raise if the update just applied diverged, localizing it in time.

    Args:
        module: the network the optimizer stepped.
        forward_scalars: scalars computed from the PRE-step weights (losses,
            entropy, approx_kl). Entropy and approx_kl are load-bearing: they
            read the log-softmax, so they catch a broken forward that the gated
            ACH policy term can hide — a nan logit makes every gate comparison
            False, which zeroes that term to a finite 0.0.
        grad_norm: this step's gradient norm (post-backward, pre-clip).
        probes: telemetry to quote, e.g. ``iw_max`` / ``pterm_max``.

    Three distinguishable cases, and the message says which:

      - a non-finite forward scalar — *the forward overflowed before this step*,
        so this update is only the messenger. Note what it does NOT mean: since
        this guard runs after every update, the weights on entry were finite
        (unless the run resumed from a diverged checkpoint), so the culprit is
        either this batch's ``advantage / pi_old`` or weights that are finite but
        large enough for the forward to leave float32 range — which is what the
        measured BRPS run looked like (``value_loss=inf``, a torso bias at
        -5.8e14, weights finite on entry);
      - finite forward, non-finite ``grad_norm`` — *this step's backward
        overflowed*;
      - both finite, non-finite weights — *this optimizer step overflowed the
        weights* (finite ``g``, but ``w - lr*g`` out of float32 range), the case
        a scalar-only guard misses entirely.

    The reported parameters are always the POST-step ones (``weights now``): by
    the time a nan gradient has been applied they say what the run holds, not
    what it held on entry.
    """
    readings = {**dict(forward_scalars), "grad_norm": grad_norm}
    bad = {k: v for k, v in readings.items() if not math.isfinite(v)}
    params = nonfinite_params(module)
    if not bad and not params:
        return
    if any(not math.isfinite(v) for v in forward_scalars.values()):
        cause = (
            "the forward overflowed before this step, so the divergence is earlier — if the "
            "weights were finite on entry (this guard checks them after every update, so "
            "they are, unless the run resumed from a diverged checkpoint) that points at "
            "this batch's advantage / pi_old, or at weights large enough to overflow the "
            "forward on their own"
        )
    elif not math.isfinite(grad_norm):
        cause = "this step's backward overflowed"
    else:
        cause = "this optimizer step overflowed the weights (finite loss and gradient)"
    quoted = ", ".join(f"{k}={v}" for k, v in (bad or readings).items())
    raise NonFiniteNetworkError(
        f"the update diverged: {quoted} ({cause}); "
        + (f"this update's probes: {dict(probes)}; " if probes else "")
        + (f"weights now: {'; '.join(params)}. " if params else "weights still finite. ")
        + NonFiniteNetworkError.HINT
    )


__all__ = [
    "NonFiniteNetworkError",
    "assert_finite_update",
    "nonfinite_action_dist_error",
    "nonfinite_params",
    "nonfinite_summary",
]
