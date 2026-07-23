"""Experiment runner: config -> Trainer -> train loop -> checkpoints (Step 8).

The single place that wires together game + policy + algo + controller into a
running experiment. Used by scripts/train.py and the one-click notebook.

Reads an :class:`ExperimentConfig` (loaded from YAML), builds the appropriate
Trainer (mirror or league), runs the train loop for N steps, snapshots the main
policy periodically to disk via the canonical ckpt manifest, and logs scalars
to a TensorBoard SummaryWriter (AGENTS.md §1 D9: TensorBoard only).

The config schema and the policy/rule/controller builders live in
:mod:`mjai.scripts.experiment_build` (§3 line cap) and are re-exported here,
so ``from mjai.scripts.experiment import ExperimentConfig, run_experiment``
remains the entry point.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from mjai.agents.base import Policy
from mjai.agents.ckpt_io import CheckpointManifest, write_checkpoint
from mjai.algos.controller import Trainer
from mjai.algos.transition import UpdateStats
from mjai.games.loader import GameSpec, load_game
from mjai.scripts.experiment_build import (
    ALGO_THETA,
    ExperimentConfig,
    build_controller,
    build_policy,
    build_update_rule,
    resolve_theta,
)
from mjai.scripts.experiment_eval import (
    build_eval_row,
    log_eval_scalars,
    print_eval_row,
    write_curve,
)
from mjai.scripts.experiment_league import log_league_health

__all__ = [
    "ALGO_THETA",
    "ExperimentConfig",
    "build_controller",
    "build_policy",
    "build_update_rule",
    "resolve_theta",
    "run_experiment",
]


def run_experiment(cfg: ExperimentConfig) -> Path:
    """Run one full experiment; returns the output directory.

    Snapshots the main policy to disk periodically, logs scalars to TensorBoard,
    and writes the full config as ``config.json`` at the start (AGENTS.md §9).

    Two loop protocols (see :class:`ExperimentConfig`):

      - legacy round mode (``total_env_steps is None``): ``n_steps`` rounds,
        checkpoint every ``save_every_steps`` rounds, evaluate every
        ``eval_every_steps`` rounds when ``eval_during_training`` is set.
      - env-step mode (``total_env_steps`` set; paper Appendix G, p25-26): run
        until ``total_env_steps`` decision-point samples have been collected
        (paper: 1e7), evaluate the CURRENT policy every
        ``eval_every_env_steps`` env-steps (paper: 1e5) and checkpoint at the
        same cadence. Eval scalars are logged to TensorBoard keyed by
        env-steps (D9) and mirrored into the legacy ``train_curve.json``.
    """
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)  # noqa: NPY002 -- global seed for any 3rd-party RNG pulls; intentional.
    rng = random.Random(cfg.seed)

    spec = load_game(cfg.game)
    policy = build_policy(spec, cfg, seed=cfg.seed)
    rule = build_update_rule(policy, cfg, spec)
    controller = build_controller(spec, policy, cfg, rng=rng)
    # A league controller collects for its exploiters as well as the main
    # agent; each needs its own rule so a role's samples update that role's
    # weights. This layer sits above both algos and league, so it is the one
    # place allowed to pair them (AGENTS.md §2).
    extra_rules = [
        build_update_rule(learner, cfg, spec)
        for learner in controller.learners()
        if learner is not policy
    ]
    trainer = Trainer(
        policy=policy, update_rule=rule, controller=controller, extra_rules=extra_rules
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(_cfg_to_dict(cfg), indent=2))

    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    curve_path = out_dir / "train_curve.json"
    curve_rows: list[dict[str, object]] = []
    step, env_steps, stats = _train_loop(
        cfg, trainer, writer, out_dir, spec, policy, curve_path, curve_rows
    )
    eval_ran = cfg.total_env_steps is not None or cfg.eval_during_training
    if eval_ran and (not curve_rows or curve_rows[-1]["step"] != step):
        _eval_and_record(
            cfg,
            writer,
            curve_path,
            curve_rows,
            spec,
            policy,
            stats,
            step,
            env_steps,
            checkpoint=False,
        )
    _save_checkpoint(out_dir, spec, cfg, policy, step)
    writer.close()
    return out_dir


def _train_loop(
    cfg: ExperimentConfig,
    trainer: Trainer,
    writer: SummaryWriter,
    out_dir: Path,
    spec: GameSpec,
    policy: Policy,
    curve_path: Path,
    curve_rows: list[dict[str, object]],
) -> tuple[int, int, UpdateStats | None]:
    """Run the train rounds; returns (rounds_run, env_steps, last_stats).

    Legacy round mode stops at ``n_steps`` rounds; env-step mode stops at
    ``total_env_steps`` collected decision-point samples (paper: 1e7) and
    evaluates/checkpoints every ``eval_every_env_steps`` (paper: 1e5, p25-26).
    """
    env_steps = 0
    next_eval_at = cfg.eval_every_env_steps
    stats: UpdateStats | None = None
    step = 0
    bar = None
    if cfg.progress_bar:
        from tqdm.auto import tqdm  # type: ignore[import-untyped]

        bar = tqdm(
            total=cfg.total_env_steps,
            desc=f"{cfg.game}/{cfg.algo}/{cfg.self_play_mode}/s{cfg.seed}",
            unit="env-step",
            dynamic_ncols=True,
        )
    while _should_continue(cfg, step, env_steps):
        step += 1
        round_ = trainer.step()
        # 1 env-step = 1 decision point the rollout actually played, INCLUDING
        # seats whose transitions the controller discarded. Counting only the
        # retained samples would let a mode that drops a seat buy twice the
        # simulation for the same nominal budget (mirror keeps both seats,
        # league keeps one), which makes the two modes' curves incomparable.
        env_steps += round_.env_steps
        if bar is not None:
            bar.update(round_.env_steps)
        stats = trainer.last_stats
        if stats:
            _log_stats(writer, step, stats)
        # The round's two costs, logged separately so an audit never has to
        # infer one from the other (tools/league_diagnose.py reads both).
        writer.add_scalar("train/sampled_steps", float(round_.env_steps), step)
        writer.add_scalar("train/batch_size", float(round_.batch_size), step)
        log_league_health(writer, step, trainer.controller)  # B7: league/* scalars
        if cfg.verbose and step % max(1, cfg.n_steps // 20) == 0:
            _print_progress(cfg, step, stats, env_steps=env_steps)
        if cfg.total_env_steps is not None:
            if env_steps >= next_eval_at:
                next_eval_at += cfg.eval_every_env_steps
                _eval_and_record(
                    cfg,
                    writer,
                    curve_path,
                    curve_rows,
                    spec,
                    policy,
                    stats,
                    step,
                    env_steps,
                    checkpoint=True,
                )
        else:
            if step % cfg.save_every_steps == 0:
                _save_checkpoint(out_dir, spec, cfg, policy, step)
            if cfg.eval_during_training and step % cfg.eval_every_steps == 0:
                _eval_and_record(
                    cfg,
                    writer,
                    curve_path,
                    curve_rows,
                    spec,
                    policy,
                    stats,
                    step,
                    env_steps,
                    checkpoint=False,
                )
    if bar is not None:
        bar.close()
    return step, env_steps, stats


def _should_continue(cfg: ExperimentConfig, step: int, env_steps: int) -> bool:
    """Loop condition for the two protocols (env-step mode takes precedence)."""
    if cfg.total_env_steps is not None:
        return env_steps < cfg.total_env_steps
    return step < cfg.n_steps


def _eval_and_record(
    cfg: ExperimentConfig,
    writer: SummaryWriter,
    curve_path: Path,
    curve_rows: list[dict[str, object]],
    spec: GameSpec,
    policy: Policy,
    stats: UpdateStats | None,
    step: int,
    env_steps: int,
    *,
    checkpoint: bool,
) -> None:
    """Evaluate the current policy, append the curve row, log to TensorBoard."""
    if checkpoint:
        _save_checkpoint(Path(cfg.out_dir), spec, cfg, policy, step)
    row = build_eval_row(
        spec,
        policy,
        stats,
        step,
        env_steps,
        eval_estimator=cfg.eval_estimator,
        eval_mc_samples=cfg.eval_mc_samples,
        seed=cfg.seed,
        eval_exact_backend=cfg.eval_exact_backend,
    )
    curve_rows.append(row)
    write_curve(curve_path, curve_rows)
    log_eval_scalars(writer, row, env_steps)
    if cfg.verbose:
        print_eval_row(row)


def _print_progress(
    cfg: ExperimentConfig, step: int, stats: UpdateStats | None, *, env_steps: int
) -> None:
    """One-line training progress: step, losses, entropy (AGENTS.md §6 friendly)."""
    if stats is None:
        print(f"  [{cfg.game}/{cfg.algo}/{cfg.self_play_mode}] step {step}/{cfg.n_steps}")
        return
    parts = [
        f"step {step}"
        + (
            f" (env {env_steps}/{cfg.total_env_steps})"
            if cfg.total_env_steps
            else f"/{cfg.n_steps}"
        ),
        f"pol_loss={stats.policy_loss:+.4f}",
        f"val_loss={stats.value_loss:.4f}",
        f"entropy={stats.entropy:.3f}",
    ]
    if stats.approx_kl:
        parts.append(f"kl={stats.approx_kl:.4f}")
    if stats.clip_frac:
        parts.append(f"clip={stats.clip_frac:.2f}")
    if "gate_off_frac" in stats.extra:
        parts.append(f"gated={stats.extra['gate_off_frac']:.2f}")
    if stats.explained_variance:
        parts.append(f"vR2={stats.explained_variance:.2f}")
    print(f"  [{cfg.game}/{cfg.algo}/{cfg.self_play_mode}] " + " ".join(parts))


def _log_stats(writer: SummaryWriter, step: int, stats: UpdateStats) -> None:
    """Push scalar stats to TensorBoard (AGENTS.md §1 D9)."""
    for key in (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_frac",
        "explained_variance",
    ):
        val = getattr(stats, key, None)
        if val is not None:
            writer.add_scalar(f"train/{key}", float(val), step)
    # Rule-specific telemetry (ACH: gate_off_frac, iw_max/iw_mean, pterm_max,
    # grad_norm). Previously computed and dropped; needed at full update
    # resolution because the blow-ups being probed are intermittent.
    for key, val in stats.extra.items():
        writer.add_scalar(f"train/{key}", float(val), step)


def _save_checkpoint(
    out_dir: Path, spec: GameSpec, cfg: ExperimentConfig, policy: Policy, step: int
) -> None:
    """Persist the main policy + manifest under ``out_dir``."""
    ckpt_dir = out_dir / "checkpoints" / f"step_{step}"
    manifest = CheckpointManifest(
        game=spec.name,
        game_string=spec.game_string,
        algo=cfg.algo,
        self_play_mode=cfg.self_play_mode,
        policy_kind=cfg.policy_kind,
        num_actions=spec.num_actions,
        obs_kind=spec.obs_kind,
        obs_size=spec.obs_size,
        train_step=step,
    )
    write_checkpoint(ckpt_dir, manifest)
    policy.save(str(ckpt_dir / manifest.weight_filename()))


def _cfg_to_dict(cfg: ExperimentConfig) -> dict[str, object]:
    import dataclasses

    return dataclasses.asdict(cfg)
