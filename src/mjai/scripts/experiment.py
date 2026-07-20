"""Experiment runner: config -> Trainer -> train loop -> checkpoints (Step 8).

The single place that wires together game + policy + algo + controller into a
running experiment. Used by scripts/train.py and the one-click notebook.

Reads an :class:`ExperimentConfig` (loaded from YAML), builds the appropriate
Trainer (mirror or league), runs the train loop for N steps, snapshots the main
policy periodically to disk via the canonical ckpt manifest, and logs scalars
to a TensorBoard SummaryWriter (AGENTS.md §1 D9: TensorBoard only).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from mjai.agents.base import Policy
from mjai.agents.ckpt_io import CheckpointManifest, write_checkpoint
from mjai.algos.controller import MirrorSelfPlay, SelfPlayController, Trainer
from mjai.algos.tabular_updates import TabularACHUpdate, TabularPPOUpdate
from mjai.algos.transition import UpdateStats
from mjai.algos.update_rule import AlgoConfig, UpdateRule
from mjai.games.loader import GameSpec, load_game
from mjai.league.league_controller import LeagueSelfPlay
from mjai.league.manager import LeagueConfig, LeagueManager
from mjai.pipeline.rollout import RolloutConfig, RolloutWorkerCore


@dataclass(frozen=True)
class ExperimentConfig:
    """All knobs for one experiment (one cell of the 2x2 matrix)."""

    game: str  # short name, e.g. "kuhn"
    algo: str  # "ppo" | "ach"
    self_play_mode: str  # "mirror" | "league"
    policy_kind: str = "tabular"  # "tabular" | "mlp"
    n_steps: int = 500
    episodes_per_round: int = 50
    save_every_steps: int = 100
    eval_every_steps: int = 100
    # When True, run_experiment prints per-step training stats and per-eval
    # equilibrium metrics so the notebook / CLI shows live progress.
    verbose: bool = False
    seed: int = 0
    out_dir: str = "runs/default"
    # When True, evaluate the current policy every ``eval_every_steps`` and
    # append to train_curve.json. Required for the notebook's training-curve
    # plots (AGENTS.md Fig 2 reproduction). Off by default to keep fast smoke
    # runs cheap.
    eval_during_training: bool = False
    # Algo + rollout + league sub-configs are built in code from these scalars
    # (kept flat here for YAML simplicity; richer configs can extend later).
    learning_rate: float = 0.1
    entropy_coef: float = 0.01
    hedge_eta: float | None = None  # ACH only
    clip_eps: float = 0.2  # PPO only
    league_capacity: int = 16


def build_policy(spec: GameSpec, cfg: ExperimentConfig, *, seed: int) -> Policy:
    """Construct the policy of the configured kind for ``spec``."""
    if cfg.policy_kind == "tabular":
        from mjai.agents.tabular import TabularPolicy

        return TabularPolicy(num_actions=spec.num_actions, seed=seed, temperature=1.0)
    if cfg.policy_kind == "mlp":
        from mjai.utils import gpu_assert

        gpu_assert.require_cpu()  # notebook/smoke default; override for GPU runs
        from mjai.agents.mlp import MLPSharedActorCritic

        return MLPSharedActorCritic(obs_size=spec.obs_size, num_actions=spec.num_actions, seed=seed)
    raise ValueError(f"Unknown policy_kind: {cfg.policy_kind}")


def build_update_rule(policy: Policy, cfg: ExperimentConfig, spec: GameSpec) -> UpdateRule:
    """Construct the configured UpdateRule on ``policy`` for ``spec``."""
    algo_cfg = AlgoConfig(learning_rate=cfg.learning_rate, entropy_coef=cfg.entropy_coef)
    if cfg.policy_kind == "tabular":
        if cfg.algo == "ppo":
            return TabularPPOUpdate(policy, algo_cfg, clip_eps=cfg.clip_eps)  # type: ignore[arg-type]
        if cfg.algo == "ach":
            # TabularACHUpdate wraps CFR+ (AGENTS.md §1 D4) and needs the game
            # spec to build the solver.
            return TabularACHUpdate(policy, spec, algo_cfg, hedge_eta=cfg.hedge_eta)  # type: ignore[arg-type]
    elif cfg.policy_kind == "mlp":
        from mjai.algos.nn_updates import NNACHUpdate, NNPPOUpdate

        if cfg.algo == "ppo":
            return NNPPOUpdate(policy, algo_cfg, clip_eps=cfg.clip_eps)  # type: ignore[arg-type]
        if cfg.algo == "ach":
            return NNACHUpdate(policy, algo_cfg)  # type: ignore[arg-type]
    raise ValueError(f"Unknown algo/policy combo: {cfg.algo}/{cfg.policy_kind}")


def build_controller(
    spec: GameSpec, policy: Policy, cfg: ExperimentConfig, *, rng: random.Random
) -> SelfPlayController:
    """Build the mirror or league controller + return it."""
    runner = RolloutWorkerCore(
        spec,
        learner_player=0,
        config=RolloutConfig(n_episodes=cfg.episodes_per_round, seed=cfg.seed),
    )
    if cfg.self_play_mode == "mirror":
        return MirrorSelfPlay(runner)
    if cfg.self_play_mode == "league":

        def make_policy() -> Policy:
            return build_policy(spec, cfg, seed=rng.randint(0, 10**9))

        def copy_weights(src: Policy, dst: Policy) -> None:
            import copy

            if (
                hasattr(src, "logits")
                and hasattr(dst, "logits")
                and hasattr(src, "values")
                and hasattr(dst, "values")
            ):
                dst.logits = copy.deepcopy(src.logits)
                dst.values = copy.deepcopy(src.values)

        league_cfg = LeagueConfig(
            capacity=cfg.league_capacity, main_save_every_steps=cfg.save_every_steps
        )
        mgr = LeagueManager(policy, make_policy, copy_weights, config=league_cfg, rng=rng)
        return LeagueSelfPlay(mgr, runner, episodes_per_round=cfg.episodes_per_round, rng=rng)
    raise ValueError(f"Unknown self_play_mode: {cfg.self_play_mode}")


def run_experiment(cfg: ExperimentConfig) -> Path:
    """Run one full experiment; returns the output directory.

    Snapshots the main policy to disk periodically, logs scalars to TensorBoard,
    and writes the full config as ``config.json`` at the start (AGENTS.md §9).
    When ``cfg.eval_during_training`` is True, also evaluates the current policy
    every ``eval_every_steps`` and appends to ``train_curve.json`` (used by the
    notebook's training-curve plots — AGENTS.md Fig 2 reproduction).
    """
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)  # noqa: NPY002 -- global seed for any 3rd-party RNG pulls; intentional.
    rng = random.Random(cfg.seed)

    spec = load_game(cfg.game)
    policy = build_policy(spec, cfg, seed=cfg.seed)
    rule = build_update_rule(policy, cfg, spec)
    controller = build_controller(spec, policy, cfg, rng=rng)
    trainer = Trainer(policy=policy, update_rule=rule, controller=controller)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(_cfg_to_dict(cfg), indent=2))

    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    curve_path = out_dir / "train_curve.json"
    curve_rows: list[dict[str, object]] = []
    step = 0
    for step in range(1, cfg.n_steps + 1):
        trainer.step()
        stats = trainer.last_stats
        if stats:
            _log_stats(writer, step, stats)
        if cfg.verbose and step % max(1, cfg.n_steps // 20) == 0:
            _print_progress(cfg, step, stats)
        if step % cfg.save_every_steps == 0:
            _save_checkpoint(out_dir, spec, cfg, policy, step)
        if cfg.eval_during_training and step % cfg.eval_every_steps == 0:
            row = _eval_during_training(spec, policy, stats, step)
            curve_rows.append(row)
            _write_curve(curve_path, curve_rows)
            if cfg.verbose:
                _print_eval_row(row)
    if cfg.eval_during_training and (not curve_rows or curve_rows[-1]["step"] != step):
        last_row = _eval_during_training(spec, policy, stats, step)
        curve_rows.append(last_row)
        _write_curve(curve_path, curve_rows)
        if cfg.verbose:
            _print_eval_row(last_row)
    _save_checkpoint(out_dir, spec, cfg, policy, step)
    writer.close()
    return out_dir


def _print_progress(cfg: ExperimentConfig, step: int, stats: UpdateStats | None) -> None:
    """One-line training progress: step, losses, entropy (AGENTS.md §6 friendly)."""
    if stats is None:
        print(f"  [{cfg.game}/{cfg.algo}/{cfg.self_play_mode}] step {step}/{cfg.n_steps}")
        return
    parts = [
        f"step {step}/{cfg.n_steps}",
        f"pol_loss={stats.policy_loss:+.4f}",
        f"val_loss={stats.value_loss:.4f}",
        f"entropy={stats.entropy:.3f}",
    ]
    if stats.approx_kl:
        parts.append(f"kl={stats.approx_kl:.4f}")
    if stats.clip_frac:
        parts.append(f"clip={stats.clip_frac:.2f}")
    if stats.explained_variance:
        parts.append(f"vR2={stats.explained_variance:.2f}")
    print(f"  [{cfg.game}/{cfg.algo}/{cfg.self_play_mode}] " + " ".join(parts))


def _print_eval_row(row: dict[str, object]) -> None:
    """Pretty-print an eval row's equilibrium metrics + BRPS probe."""
    bits = [f"step={row['step']}"]
    for k in ("eval/exploitability", "eval/nash_conv", "eval/exact_nash_distance"):
        if k in row:
            bits.append(f"{k.removeprefix('eval/')}={float(row[k]):.4g}")  # type: ignore[arg-type]
    if "brps/nash_distance" in row:
        bits.append(f"brps_nash_d={float(row['brps/nash_distance']):.4g}")  # type: ignore[arg-type]
        bits.append(
            f"P(R,P,S)="
            f"({float(row['brps/P_R']):.3f},{float(row['brps/P_P']):.3f},{float(row['brps/P_S']):.3f})"  # type: ignore[arg-type]
        )
    print("    eval: " + " ".join(bits))


def _eval_during_training(
    spec: GameSpec, policy: Policy, stats: UpdateStats | None, step: int
) -> dict[str, object]:
    """Compute equilibrium metrics + per-action BRPS probe for the curve row.

    Failures inside the equilibrium evaluator are caught — we want a training
    curve even when one metric isn't computable for a game.
    """
    import contextlib

    row: dict[str, object] = {"step": step}
    if stats is not None:
        for k in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_frac",
            "explained_variance",
        ):
            v = getattr(stats, k, None)
            if v is not None:
                row[k] = float(v)
    # Equilibrium metrics (best available for this game).
    from mjai.eval.nash import evaluate_equilibrium

    with contextlib.suppress(Exception):
        row.update({f"eval/{k}": v for k, v in evaluate_equilibrium(spec, policy).items()})
    # BRPS-specific probe: P(R), P(P), P(S) at the trivial observation, so the
    # notebook can plot the policy trajectory (AGENTS.md Fig 1).
    if spec.name == "brps":
        with contextlib.suppress(Exception):
            from mjai.eval.nash import distance_to_brps_nash

            obs = [0.0]
            legal = list(range(spec.num_actions))
            logits = policy.action_logits(obs, legal)
            import math

            mx = max(logits)
            exps = [math.exp(x - mx) for x in logits]
            s = sum(exps) or 1.0
            probs = [e / s for e in exps]
            padded = [*probs, 0.0, 0.0, 0.0]
            row["brps/P_R"], row["brps/P_P"], row["brps/P_S"] = padded[:3]
            row["brps/nash_distance"] = distance_to_brps_nash(policy, num_actions=spec.num_actions)
    return row


def _write_curve(path: Path, rows: list[dict[str, object]]) -> None:
    """Persist the training-curve rows as JSON (overwritten each eval)."""
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


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
