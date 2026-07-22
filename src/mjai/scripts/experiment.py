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
from dataclasses import dataclass, field
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
    """All knobs for one experiment (one cell of the 2x2 matrix).

    Env-step mode (paper Appendix G protocol, p25-26): when ``total_env_steps`` is
    set, the train loop runs until that many decision-point samples have been
    collected (paper: 1e7) and evaluates every ``eval_every_env_steps``
    env-steps (paper: 1e5). When ``total_env_steps`` is None, the legacy
    round-based fields (``n_steps`` / ``eval_every_steps`` /
    ``save_every_steps``) drive the loop instead. One env-step = one sampled
    decision point = one batch transition (spec assumption A2).
    """

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
    # runs cheap. In env-step mode evaluation always runs (paper protocol).
    eval_during_training: bool = False
    # Algo + rollout + league sub-configs are built in code from these scalars
    # (kept flat here for YAML simplicity; richer configs can extend later).
    learning_rate: float = 0.0001  # NN-tuned; tabular overrides at call site
    entropy_coef: float = 0.05  # NN-tuned; paper ACH uses beta=1e-2 (p28 Table 8)
    hedge_eta: float | None = None  # tabular ACH (CFR+ wrapper) only
    clip_eps: float = 0.2  # PPO only
    league_capacity: int = 16
    # ---- MLP architecture (paper p25: 1x128 FC + ReLU, two linear heads) ----
    hidden_sizes: list[int] = field(default_factory=lambda: [128])
    activation: str = "relu"  # "relu" | "tanh"
    # Explicit torch device override (e.g. "cpu", "cuda:0"); None = gpu_assert
    # resolution (GPU default, fail loudly unless --cpu / MJAI_CPU=1; D6).
    device: str | None = None
    # ---- AlgoConfig wiring (AGENTS.md §9; ACH defaults from p27-28) ----
    value_coef: float = 0.5  # paper ACH alpha=2.0 <=> value_coef=1.0 (alpha/2 * MSE form)
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5  # <=0 disables; paper mentions no clipping
    optimizer: str | None = None  # None = per-algo default (ach->sgd, ppo->adam)
    eta: float = 1.0  # hedge coefficient eta(s) (p27 Table 7)
    l_th: float = 2.0  # one-sided logit gate threshold (p28 Table 8)
    ratio_eps: float = 0.5  # ratio gate; vacuous when synchronous (p28)
    loss_centered_logits: bool = True  # False = raw logit in ACH loss (A3 probe)
    # ---- Sampling / env-step protocol (paper: batch 64, 1e5 eval, 1e7 total) ----
    target_samples: int | None = None  # per-round batch size in samples (p28: 64)
    total_env_steps: int | None = None  # set -> env-step mode (paper: 1e7)
    eval_every_env_steps: int = 100000  # paper p25: evaluate every 1e5 steps


def build_policy(spec: GameSpec, cfg: ExperimentConfig, *, seed: int) -> Policy:
    """Construct the policy of the configured kind for ``spec``.

    MLP runs resolve the device via gpu_assert (D6: GPU by default, loud error
    unless CPU was explicitly requested via ``--cpu`` / ``MJAI_CPU=1`` or an
    explicit ``cfg.device``). No silent CPU fallback.
    """
    if cfg.policy_kind == "tabular":
        from mjai.agents.tabular import TabularPolicy

        return TabularPolicy(num_actions=spec.num_actions, seed=seed, temperature=1.0)
    if cfg.policy_kind == "mlp":
        from mjai.agents.mlp import ACTIVATIONS, MLPSharedActorCritic

        if cfg.activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {cfg.activation!r}; expected one of {sorted(ACTIVATIONS)}"
            )
        return MLPSharedActorCritic(
            obs_size=spec.obs_size,
            num_actions=spec.num_actions,
            hidden_sizes=tuple(cfg.hidden_sizes),
            activation=ACTIVATIONS[cfg.activation],
            device=cfg.device,
            seed=seed,
        )
    raise ValueError(f"Unknown policy_kind: {cfg.policy_kind}")


def build_update_rule(policy: Policy, cfg: ExperimentConfig, spec: GameSpec) -> UpdateRule:
    """Construct the configured UpdateRule on ``policy`` for ``spec``.

    Every AlgoConfig field is wired from the experiment YAML (AGENTS.md §9).
    The optimizer defaults per endpoint: ACH → SGD (paper H.3, p27), PPO →
    Adam; an explicit ``cfg.optimizer`` overrides either (NNACHUpdate rejects
    anything but SGD — paper-faithful, single ACH implementation).
    """
    optimizer = cfg.optimizer or ("sgd" if cfg.algo == "ach" else "adam")
    algo_cfg = AlgoConfig(
        learning_rate=cfg.learning_rate,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        max_grad_norm=cfg.max_grad_norm,
        gae_lambda=cfg.gae_lambda,
        optimizer=optimizer,
        eta=cfg.eta,
        l_th=cfg.l_th,
        ratio_eps=cfg.ratio_eps,
        loss_centered_logits=cfg.loss_centered_logits,
    )
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
        config=RolloutConfig(
            n_episodes=cfg.episodes_per_round,
            gae_lambda=cfg.gae_lambda,
            seed=cfg.seed,
            target_samples=cfg.target_samples,
        ),
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
    trainer = Trainer(policy=policy, update_rule=rule, controller=controller)

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
    while _should_continue(cfg, step, env_steps):
        step += 1
        env_steps += trainer.step().batch_size  # 1 env-step = 1 sampled decision point
        stats = trainer.last_stats
        if stats:
            _log_stats(writer, step, stats)
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
    row = _eval_during_training(spec, policy, stats, step, env_steps)
    curve_rows.append(row)
    _write_curve(curve_path, curve_rows)
    _log_eval_scalars(writer, row, env_steps)
    if cfg.verbose:
        _print_eval_row(row)


def _log_eval_scalars(writer: SummaryWriter, row: dict[str, object], env_steps: int) -> None:
    """Log equilibrium metrics to TensorBoard keyed by env-steps (AGENTS.md D9).

    The paper's Fig 10 x-axis is training steps (p25-26), so eval curves live
    on the env-step axis; ``train_curve.json`` remains for the notebook.
    """
    for k, v in row.items():
        if k.startswith("eval/") and isinstance(v, int | float):
            writer.add_scalar(k, float(v), env_steps)


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
    spec: GameSpec, policy: Policy, stats: UpdateStats | None, step: int, env_steps: int
) -> dict[str, object]:
    """Compute equilibrium metrics + per-action BRPS probe for the curve row.

    Metric failures are NOT silently swallowed (AGENTS.md: no silent fallback):
    they emit a ``warnings.warn`` and leave an ``eval/error`` field in the row,
    so a missing column in the curve is always traceable.
    """
    import warnings

    row: dict[str, object] = {"step": step, "env_steps": env_steps}
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

    try:
        row.update({f"eval/{k}": v for k, v in evaluate_equilibrium(spec, policy).items()})
    except Exception as e:
        warnings.warn(f"equilibrium eval failed at step {step}: {e}", stacklevel=2)
        row["eval/error"] = str(e)
    # BRPS-specific probe: P(R), P(P), P(S) at the trivial observation, so the
    # notebook can plot the policy trajectory (AGENTS.md Fig 1).
    if spec.name == "brps":
        try:
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
        except Exception as e:
            warnings.warn(f"BRPS probe failed at step {step}: {e}", stacklevel=2)
            row["brps/error"] = str(e)
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
