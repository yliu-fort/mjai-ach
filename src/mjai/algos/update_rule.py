"""Abstract :class:`UpdateRule` — the algorithm plug point (AGENTS.md §3, §4).

An UpdateRule takes a :class:`~mjai.algos.transition.Batch` (collected by the
self-play controller, which may be mirror or league) and performs one or more
gradient steps on the given :class:`~mjai.agents.base.Policy`. The rule is
**stateless w.r.t. self-play topology** — it does not know or care whether the
batch came from mirror self-play or a league opponent pool.

Adding an algorithm (AGENTS.md §4 "Algo rule") = new ``UpdateRule`` subclass.
No edits to Trainer or pipeline.

Two implementation families ship in Phase 1:
  - :mod:`mjai.algos.nn_updates` — :class:`NNActorCriticUpdate`, one
    theta-parameterized rule spanning reference PPO (``theta=0``) and
    paper-faithful ACH (``theta=1``, AGENTS.md §1 D4) on torch MLPs.
  - :mod:`mjai.algos.tabular_updates` — tabular PPO / ACH (CFR+ wrapper) on
    dict-backed policies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from mjai.agents.base import Policy
from mjai.algos.transition import Batch, UpdateStats


class ACHFidelityWarning(UserWarning):
    """A scaffolding knob is set to a value the ACH paper does not use.

    Raised (as a warning) when the ACH policy term carries weight
    (``theta > 0``) while a PPO-best-practice knob is enabled that the paper's
    Appendix H.3 protocol contradicts — Adam instead of constant-LR SGD,
    per-batch advantage normalization, or more than one epoch per mini-batch.
    Such a run is a deliberate A/B arm, not a reproduction; the warning exists
    so that distinction is never silent (AGENTS.md §11).
    """


@dataclass(frozen=True)
class AlgoConfig:
    """Shared hyperparameters consumed by every UpdateRule.

    All fields are wired from the experiment YAML (AGENTS.md §9 — no magic
    numbers in code). ACH-specific fields (``eta``, ``l_th``, ``ratio_eps``)
    are inert at ``theta=0``; ``clip_eps`` is inert at ``theta=1``.

    Defaults follow the ACH paper's Appendix H.3 (p27-28) where applicable:
    eta=1.0, l_th=2.0 (p28 Table 8); ``ratio_eps`` is vacuous under synchronous
    single-threaded self-play (p28) and only matters for async sampling.

    **Scaffolding defaults follow ACH, at every theta.** The optimizer, the
    advantage treatment, the epoch count and the grad-clip setting are one
    shared, knob-driven scaffold rather than a per-algorithm one, so a
    PPO-vs-ACH comparison varies only ``theta`` unless a knob is turned on
    deliberately. Knobs that contradict the paper's protocol emit an
    :class:`ACHFidelityWarning` when the ACH term carries weight.

    Attributes:
        theta: PPO/ACH interpolation of the POLICY term, 0 = PPO clipped
            surrogate, 1 = paper-faithful ACH. Intermediate values take the
            convex combination ``(1-theta)*L_ppo + theta*L_ach``. The value
            and entropy terms have the same form in both and are never
            interpolated.
        optimizer: ``"sgd"`` | ``"adam"`` | None (None = ``"sgd"``, paper p27).
        adam_eps: Adam epsilon; 1e-5 per the 37-details recommendation.
            Ignored when the optimizer is SGD.
        max_grad_norm: grad-norm clip; ``<= 0`` disables (paper mentions no
            clipping, so the ACH reproduction configs disable it).
        normalize_advantages: per-batch advantage normalization for the PPO
            term (a 37-details best practice). The ACH term ALWAYS consumes
            raw GAE advantages regardless, since normalizing them would turn
            eta into a batch-adaptive learning rate (paper p24).
        n_epochs: gradient steps taken per collected batch. The paper updates
            once per mini-batch (p24), so >1 is a PPO-side A/B knob.
        clip_eps: PPO surrogate clip range (37-details default 0.2).
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
    # False (default) = raw logit, the literal Algorithm 2 form. Raw logits are
    # the default because they are paired with the MLP's trunk LayerNorm, which
    # supplies the logit-scale stability that manual centering was standing in
    # for -- the combination is what reproduces the paper's Liar's Dice curve
    # (docs/reproduce_report.md §6.5). Set True for the pre-LayerNorm behavior.
    loss_centered_logits: bool = False
    # ACH centered-logit mean y_bar over LEGAL actions only when True; False =
    # mean over all actions (historical behavior). The paper never discusses
    # action masking (spec assumption A5), so which actions enter y_bar is
    # ambiguous; illegal-logit drift distorts the gate most in games with
    # shrinking legal sets (Liar's Dice). Probe toggle for the gap
    # investigation (docs/reproduce_report.md liars +0.15 bias).
    centered_mean_legal_only: bool = False
    # ACH gate thresholds the mean-centered logit (paper p24 is explicit) when
    # True. False (default) = threshold the RAW logit, which is coherent because
    # the MLP's trunk LayerNorm stabilizes the scale feeding the heads:
    # architecture-level normalization instead of manual centering. Set True
    # together with ``loss_centered_logits`` for the pre-LayerNorm behavior.
    gate_centered_logits: bool = False
    # ---- PPO/ACH interpolation + shared-scaffolding knobs ----
    # theta=1 (default) is paper-faithful ACH; theta=0 is the PPO clipped
    # surrogate. See the class docstring for what is and is not interpolated.
    theta: float = 1.0
    clip_eps: float = 0.2
    # Scaffolding knobs. Defaults are the ACH-protocol values at EVERY theta
    # (so PPO inherits ACH's scaffolding unless told otherwise); flipping one
    # while theta>0 warns (ACHFidelityWarning) rather than silently producing a
    # run that looks like a reproduction but is not.
    normalize_advantages: bool = False
    n_epochs: int = 1
    adam_eps: float = 1e-5

    def __post_init__(self) -> None:
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError(f"theta must lie in [0, 1], got {self.theta}")
        if self.n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {self.n_epochs}")


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
