"""League wiring + telemetry for the experiment runner (AGENTS.md §3.1).

Split out of :mod:`mjai.scripts.experiment` to keep that module under the
500-line cap: the league branch of ``build_controller`` and the ``league/*``
TensorBoard health scalars (B7) live here. The module depends on the abstract
controller interface only for typing the telemetry entry point — it never
reaches into UpdateRules (AGENTS.md §2).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from mjai.agents.base import Policy, copy_weights
from mjai.algos.controller import SelfPlayController
from mjai.league.manager import LeagueConfig, LeagueManager
from mjai.league.opponent_sampler import LeagueMix

if TYPE_CHECKING:
    from mjai.scripts.experiment import ExperimentConfig


class _ScalarWriter(Protocol):
    """Structural type for the one writer method we use (real: SummaryWriter).

    The writer is always created by the runner and passed down (AGENTS.md §6);
    tests substitute a lightweight fake.
    """

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None: ...


def build_league_manager(
    policy: Policy,
    make_policy: Callable[[], Policy],
    cfg: ExperimentConfig,
    *,
    rng: random.Random,
) -> LeagueManager:
    """Wire the full LeagueManager from the experiment config (AGENTS.md §9).

    Weight copies go through the generic :func:`mjai.agents.base.copy_weights`
    (Policy.snapshot_state/restore_state), which works for tabular and NN
    policies alike and fails loudly on incompatible kinds — no isinstance
    branches, no silent no-ops (B1, AGENTS.md §3.3, §11).
    """
    # All league knobs come from the YAML config — no magic numbers (§9).
    # LeagueMix arg order: current_main, history, exploiter.
    mix = LeagueMix(cfg.league_mix_current_main, cfg.league_mix_history, cfg.league_mix_exploiter)
    league_cfg = LeagueConfig(
        capacity=cfg.league_capacity,
        main_save_every_rounds=cfg.league_main_save_every_rounds,
        main_exploiter_promo=cfg.league_main_exploiter_promo,
        league_exploiter_promo=cfg.league_league_exploiter_promo,
        league_exploiter_share=cfg.league_exploiter_share,
        promo_window=cfg.league_promo_window,
        mix=mix,
        reset_mode=cfg.league_reset_mode,
    )
    return LeagueManager(policy, make_policy, copy_weights, config=league_cfg, rng=rng)


def log_league_health(writer: _ScalarWriter, step: int, controller: SelfPlayController) -> None:
    """Push the league health scalars to TensorBoard (B7, AGENTS.md §6).

    Capability-based, not isinstance-based (§3.3): controllers that expose
    ``health_stats()`` (LeagueSelfPlay) get their scalars logged; mirror
    controllers simply have nothing to report.
    """
    health = getattr(controller, "health_stats", None)
    if health is None:
        return
    for key, value in health().items():
        writer.add_scalar(f"league/{key}", float(value), step)
