"""Config schema + object construction for one experiment (AGENTS.md §9).

Split out of :mod:`mjai.scripts.experiment` so both modules stay under the
500-line AST cap (AGENTS.md §3 rule 1). Owns the YAML-facing
:class:`ExperimentConfig` dataclass and the three builders that turn it into
live objects — policy, update rule, self-play controller. The train loop that
consumes them stays in :mod:`mjai.scripts.experiment`, which re-exports these
names so existing imports keep working.

This module changes for exactly one reason: a new configurable knob.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from mjai.agents.base import Policy
from mjai.algos.controller import MirrorSelfPlay, SelfPlayController
from mjai.algos.tabular_updates import TabularACHUpdate, TabularPPOUpdate
from mjai.algos.update_rule import AlgoConfig, UpdateRule
from mjai.games.loader import GameSpec
from mjai.league.checkpoint_store import Role
from mjai.league.league_controller import LeagueSelfPlay
from mjai.pipeline.rollout import RolloutConfig, RolloutWorkerCore
from mjai.scripts.experiment_league import build_league_manager


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
    # "ppo" (theta=0) | "ach" (theta=1) | "theta" (explicit ``theta`` field).
    # On the MLP path all three build the same rule; the first two are pinned
    # aliases so a config cannot silently disagree with its own name.
    algo: str
    self_play_mode: str  # "mirror" | "league"
    policy_kind: str = "tabular"  # "tabular" | "mlp"
    n_steps: int = 500
    episodes_per_round: int = 50
    save_every_steps: int = 100
    eval_every_steps: int = 100
    # When True, run_experiment prints per-step training stats and per-eval
    # equilibrium metrics so the notebook / CLI shows live progress.
    verbose: bool = False
    # tqdm bar over env-steps in the train loop (notebooks); off by default so
    # batch/CI runs and unit tests stay quiet.
    progress_bar: bool = False
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
    league_capacity: int = 16  # pool size; >= 3 (2 slots reserved for exploiters)
    # ---- League strategy knobs (defaults mirror LeagueConfig/LeagueMix; F2) ----
    # Opponent mix for the MAIN role. The league-exploiter never faces the live
    # main: it draws pool members only, history vs pool-exploiters in the same
    # history:exploiter proportion renormalized (0.3:0.2 -> 60%/40%).
    league_mix_current_main: float = 0.5  # P(opponent = live main), main role only
    league_mix_history: float = 0.3  # P(opponent = past main snapshot from pool)
    league_mix_exploiter: float = 0.2  # P(opponent = promoted exploiter from pool)
    league_main_exploiter_promo: float = 0.70  # promote main-exploiter at this WR vs main
    league_league_exploiter_promo: float = 0.70  # promote league-exploiter at this WR
    league_exploiter_share: float = 0.70  # pool fraction league-exploiter must beat
    league_promo_window: int = 20  # rolling win-rate window size (episodes)
    league_reset_mode: str = "to_main"  # exploiter reset after promotion: to_main|random
    # Data-collection ratio main : main-exploiter : league-exploiter
    # (AlphaStar-style default 1 : 0.5 : 0.5 — the main line collects half of
    # all samples). Turned into a deterministic smooth-WRR cycle
    # (mjai.league.league_controller.role_cycle); main must be > 0, an
    # exploiter may be 0 (ablation: that role never collects).
    league_role_weight_main: float = 1.0
    league_role_weight_main_exploiter: float = 0.5
    league_role_weight_league_exploiter: float = 0.5
    # Per-episode seat shuffle for the collecting role (perspective coverage):
    # without it a learner only ever sees the game from seat 0, so frozen
    # opponents are never faced from seat 1 — the policy is half-blind. Routing
    # is by producer identity, so shuffled episodes still train the right rule.
    league_seat_shuffle: bool = True
    # When True, a round whose opponent is itself a live learner (every
    # main-exploiter round faces the live main) also routes that opponent's
    # transitions to ITS OWN update
    # rule — on-policy anti-exploiter data instead of dropped samples. This
    # changes the league dynamics (the main line patches exploiter-found holes
    # faster), so it is an explicit knob, off by default. (The league-exploiter
    # never faces a live learner: its opponents are always frozen pool members.)
    league_train_live_opponents: bool = False
    # Main-snapshot cadence for the league pool, counted in MAIN COLLECT ROUNDS
    # (B3; under the default 1:0.5:0.5 role weights, every other collect is a
    # main round). Independent of ``save_every_steps`` (legacy round-mode disk
    # checkpoint cadence) — reusing that knob coupled pool history to an
    # unrelated unit and starved the pool at probe scale.
    league_main_save_every_rounds: int = 200
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
    optimizer: str | None = None  # None = SGD (paper H.3 p27), at every theta
    eta: float = 1.0  # hedge coefficient eta(s) (p27 Table 7)
    l_th: float = 2.0  # one-sided logit gate threshold (p28 Table 8)
    ratio_eps: float = 0.5  # ratio gate; vacuous when synchronous (p28)
    # Default ACH shape since docs/reproduce_report.md §6.5: trunk LayerNorm
    # supplies logit-scale stability, so gate and loss body both use the raw
    # logit. Set all three to the old values for the pre-LayerNorm behavior.
    loss_centered_logits: bool = False  # True = mean-centered logit in the ACH loss
    gate_centered_logits: bool = False  # True = ACH gate on the mean-centered logit
    trunk_layernorm: bool = True  # LayerNorm at the torso end (normalizes features)
    centered_mean_legal_only: bool = False  # ACH y_bar over legal actions only (A5 probe)
    # ---- PPO<->ACH interpolation + shared-scaffolding knobs (NN path only) ----
    # theta is settable only with algo="theta"; algo="ppo"/"ach" pin it to
    # 0.0/1.0 so a config's name can never disagree with its update rule.
    theta: float | None = None
    # PPO best practices, off by default so the scaffolding follows the ACH
    # protocol at every theta. Enabling one while the ACH term has weight emits
    # an ACHFidelityWarning (mjai.algos.update_rule).
    normalize_advantages: bool = False  # per-batch advantage normalization (PPO term)
    n_epochs: int = 1  # gradient steps per collected batch (paper: 1, p24)
    adam_eps: float = 1e-5  # Adam epsilon (37-details); ignored under SGD
    # Log the PPO and ACH policy terms' gradient norms separately (+ their
    # cosine) as train/grad_norm_{ppo,ach}[_scaled] and train/grad_cos_ppo_ach.
    # Two extra backward passes per update; the update itself is unchanged.
    probe_term_grad_norms: bool = False
    # ---- Sampling / env-step protocol (paper: batch 64, 1e5 eval, 1e7 total) ----
    target_samples: int | None = None  # per-round batch size in samples (p28: 64)
    total_env_steps: int | None = None  # set -> env-step mode (paper: 1e7)
    eval_every_env_steps: int = 100000  # paper p25: evaluate every 1e5 steps
    # ---- Equilibrium eval estimator (AGENTS.md §9; mjai.eval.sampled_nash) ----
    # "exact" walks the full game tree (open_spiel nash_conv) — infeasible for
    # oshi_zumo-scale games; "sampled" uses a Monte-Carlo approximate best
    # response with a per-player budget of eval_mc_samples episodes per eval.
    # The eval seed is cfg.seed itself (common random numbers across the curve
    # => reproducible, lower-variance eval-to-eval comparisons).
    eval_estimator: str = "exact"
    eval_mc_samples: int = 400
    # Best-response solver for the "exact" estimator (mjai.eval.nash):
    # "auto" = OpenSpiel's C++ MDP solver on turn-based games (7.9x on Liar's
    # Dice) and the Python traversal on simultaneous ones, where the C++ solver
    # raises on trained MLP policies; "python" / "cpp" force one route.
    # NOTE: the C++ solver is only reproducible to ~1 ulp across processes (its
    # summation order follows a hash map). Use "python" when eval curves must
    # be bit-identical run to run.
    eval_exact_backend: str = "auto"
    # ---- Average-strategy anchor (AGENTS.md D16) ----
    # ACH's O(T^-1/2) bound is about the AVERAGE strategy; the curves this repo
    # plots are the current policy (docs/reproduce_report.md), which is the right
    # object for a last-iterate study but not the one the theorem covers. Setting
    # this additionally tracks the running-average strategy in sequence-form
    # coordinates and emits eval/avg_nash_conv (+ eval/avg_exploitability at 2
    # players). Off by default: it costs exact NashConv(s) per eval point and
    # only makes sense on exactly-enumerable games.
    track_average_policy: bool = False
    # Weighting for that average: "uniform" is the one the theorem is stated
    # for; "linear" (weight = eval index) is CFR+'s and converges faster; "both"
    # emits uniform under eval/avg_* and linear under eval/avg_*_lin from a
    # single run. They are different curves — say which one a figure shows.
    average_policy_weighting: str = "uniform"

    def __post_init__(self) -> None:
        # League knob validation (AGENTS.md §9: invalid config fails loudly).
        total = self.league_mix_current_main + self.league_mix_history + self.league_mix_exploiter
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"league mix weights must sum to 1.0 (tolerance 1e-6), got {total}")
        if self.league_reset_mode not in ("to_main", "random"):
            raise ValueError(
                f"bad league_reset_mode {self.league_reset_mode!r}; want to_main|random"
            )
        if self.league_capacity < 3:
            raise ValueError(
                f"league_capacity must be >= 3 (2 pool slots are reserved for the "
                f"exploiters; the main-history quota capacity-2 must be non-empty), "
                f"got {self.league_capacity}"
            )
        role_weights = {
            "main": self.league_role_weight_main,
            "main_exploiter": self.league_role_weight_main_exploiter,
            "league_exploiter": self.league_role_weight_league_exploiter,
        }
        for name, w in role_weights.items():
            if w < 0:
                raise ValueError(f"league_role_weight_{name} must be >= 0, got {w}")
        if self.league_role_weight_main <= 0:
            raise ValueError(
                "league_role_weight_main must be > 0: pool snapshots and PFSP "
                "bookkeeping are driven by main collect rounds"
            )
        if self.eval_estimator not in ("exact", "sampled"):
            raise ValueError(f"bad eval_estimator {self.eval_estimator!r}; want exact|sampled")
        if self.eval_exact_backend not in ("auto", "python", "cpp"):
            raise ValueError(
                f"bad eval_exact_backend {self.eval_exact_backend!r}; want auto|python|cpp"
            )
        if self.average_policy_weighting not in ("uniform", "linear", "both"):
            raise ValueError(
                f"bad average_policy_weighting {self.average_policy_weighting!r}; "
                f"want uniform|linear|both"
            )
        if self.eval_mc_samples < 16:
            raise ValueError(
                f"eval_mc_samples must be >= 16 for the derived probe/match budgets, "
                f"got {self.eval_mc_samples}"
            )


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
            trunk_layernorm=cfg.trunk_layernorm,
            device=cfg.device,
            seed=seed,
        )
    raise ValueError(f"Unknown policy_kind: {cfg.policy_kind}")


# algo names that pin theta. "theta" itself reads ExperimentConfig.theta.
ALGO_THETA: dict[str, float] = {"ppo": 0.0, "ach": 1.0}


def resolve_theta(cfg: ExperimentConfig) -> float:
    """Map ``algo`` (+ ``theta``) onto the update rule's interpolation weight.

    ``algo: ppo`` and ``algo: ach`` are pinned aliases for theta 0 and 1; only
    ``algo: theta`` may carry an explicit value. Setting ``theta`` alongside a
    pinned alias is a contradiction and fails loudly (AGENTS.md §9).
    """
    if cfg.algo in ALGO_THETA:
        if cfg.theta is not None:
            raise ValueError(
                f"algo={cfg.algo!r} pins theta={ALGO_THETA[cfg.algo]}; "
                f"remove theta={cfg.theta} or set algo='theta' to sweep it"
            )
        return ALGO_THETA[cfg.algo]
    if cfg.algo == "theta":
        if cfg.theta is None:
            raise ValueError("algo='theta' requires an explicit theta in [0, 1]")
        return float(cfg.theta)
    raise ValueError(f"Unknown algo {cfg.algo!r}; expected 'ppo' | 'ach' | 'theta'")


def build_update_rule(policy: Policy, cfg: ExperimentConfig, spec: GameSpec) -> UpdateRule:
    """Construct the configured UpdateRule on ``policy`` for ``spec``.

    Every AlgoConfig field is wired from the experiment YAML (AGENTS.md §9).
    On the MLP path all three algo names build the same theta-parameterized
    rule; the scaffolding (optimizer, advantage treatment, epochs) follows the
    ACH protocol by default regardless of theta, so a PPO arm and an ACH arm
    differ only in the policy term unless a knob is set deliberately.
    """
    theta = resolve_theta(cfg)
    algo_cfg = AlgoConfig(
        learning_rate=cfg.learning_rate,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        max_grad_norm=cfg.max_grad_norm,
        gae_lambda=cfg.gae_lambda,
        optimizer=cfg.optimizer or "sgd",
        eta=cfg.eta,
        l_th=cfg.l_th,
        ratio_eps=cfg.ratio_eps,
        loss_centered_logits=cfg.loss_centered_logits,
        centered_mean_legal_only=cfg.centered_mean_legal_only,
        gate_centered_logits=cfg.gate_centered_logits,
        theta=theta,
        clip_eps=cfg.clip_eps,
        normalize_advantages=cfg.normalize_advantages,
        n_epochs=cfg.n_epochs,
        adam_eps=cfg.adam_eps,
        probe_term_grad_norms=cfg.probe_term_grad_norms,
    )
    if cfg.policy_kind == "tabular":
        # The tabular pair predates the interpolation and stays discrete: PPO
        # is a tabular clipped surrogate, ACH wraps CFR+ (AGENTS.md §1 D4/D5).
        if cfg.algo == "ppo":
            return TabularPPOUpdate(policy, algo_cfg, clip_eps=cfg.clip_eps)  # type: ignore[arg-type]
        if cfg.algo == "ach":
            # TabularACHUpdate wraps CFR+ and needs the game spec to build the solver.
            return TabularACHUpdate(policy, spec, algo_cfg, hedge_eta=cfg.hedge_eta)  # type: ignore[arg-type]
        raise ValueError(
            f"algo={cfg.algo!r} has no tabular implementation; "
            "the theta interpolation is MLP-only (policy_kind: mlp)"
        )
    if cfg.policy_kind == "mlp":
        from mjai.algos.nn_updates import NNActorCriticUpdate

        return NNActorCriticUpdate(policy, algo_cfg)  # type: ignore[arg-type]
    raise ValueError(f"Unknown policy_kind: {cfg.policy_kind}")


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
            # League rounds shuffle the collector's seat per episode so every
            # opponent is faced from both perspectives; routing by producer
            # identity keeps each learner's dose exact regardless of seat.
            # Mirror plays the same policy in both seats, so shuffling would
            # only perturb its RNG stream — it stays off there.
            shuffle_seats=cfg.league_seat_shuffle if cfg.self_play_mode == "league" else False,
        ),
    )
    if cfg.self_play_mode == "mirror":
        return MirrorSelfPlay(runner)
    if cfg.self_play_mode == "league":

        def make_policy() -> Policy:
            return build_policy(spec, cfg, seed=rng.randint(0, 10**9))

        # League wiring (incl. the B1-generic weight copy) lives in
        # experiment_league (§3.1: keeps this module under the line cap).
        mgr = build_league_manager(policy, make_policy, cfg, rng=rng)
        return LeagueSelfPlay(
            mgr,
            runner,
            episodes_per_round=cfg.episodes_per_round,
            role_weights={
                Role.MAIN: cfg.league_role_weight_main,
                Role.MAIN_EXPLOITER: cfg.league_role_weight_main_exploiter,
                Role.LEAGUE_EXPLOITER: cfg.league_role_weight_league_exploiter,
            },
            train_live_opponents=cfg.league_train_live_opponents,
            rng=rng,
        )
    raise ValueError(f"Unknown self_play_mode: {cfg.self_play_mode}")


__all__ = [
    "ALGO_THETA",
    "ExperimentConfig",
    "build_controller",
    "build_policy",
    "build_update_rule",
    "resolve_theta",
]
