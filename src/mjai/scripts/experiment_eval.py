"""Eval-during-training helpers: build, log, print, and persist eval rows.

Split out of :mod:`mjai.scripts.experiment` so both modules stay under the
500-line AST cap (AGENTS.md §3 rule 1). Owns everything about one eval row:
equilibrium metrics + BRPS probe (``build_eval_row``), TensorBoard logging of
``eval/*`` scalars keyed by env-steps (``log_eval_scalars``), the console
pretty-print (``print_eval_row``), and ``train_curve.json`` persistence
(``write_curve``). Called only from ``experiment._eval_and_record``.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from mjai.agents.base import Policy
from mjai.algos.transition import UpdateStats
from mjai.games.loader import GameSpec


def build_eval_row(
    spec: GameSpec,
    policy: Policy,
    stats: UpdateStats | None,
    step: int,
    env_steps: int,
    *,
    eval_estimator: str = "exact",
    eval_mc_samples: int = 400,
    seed: int = 0,
    eval_exact_backend: str = "auto",
) -> dict[str, object]:
    """Compute equilibrium metrics + per-action BRPS probe for the curve row.

    ``eval_estimator`` / ``eval_mc_samples`` select the NashConv backend
    ("exact" full-tree vs "sampled" Monte-Carlo approximate BR; see
    :func:`mjai.eval.nash.evaluate_equilibrium`). Metric failures are NOT
    silently swallowed (AGENTS.md: no silent fallback): they emit a
    ``warnings.warn`` and leave an ``eval/error`` field in the row, so a
    missing column in the curve is always traceable.
    """
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
        # Rule-specific telemetry (ACH gate/importance-weight probes). This is
        # the last update's snapshot, not an interval aggregate — TensorBoard
        # holds the full-resolution series; these are for quick curve reads.
        for k, v in stats.extra.items():
            row[k] = float(v)
    # Equilibrium metrics (best available for this game).
    from mjai.eval.nash import evaluate_equilibrium

    try:
        metrics = evaluate_equilibrium(
            spec,
            policy,
            estimator=eval_estimator,
            mc_samples=eval_mc_samples,
            seed=seed,
            exact_backend=eval_exact_backend,
        )
        row.update({f"eval/{k}": v for k, v in metrics.items()})
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


def log_eval_scalars(writer: SummaryWriter, row: dict[str, object], env_steps: int) -> None:
    """Log equilibrium metrics to TensorBoard keyed by env-steps (AGENTS.md D9).

    The paper's Fig 10 x-axis is training steps (p25-26), so eval curves live
    on the env-step axis; ``train_curve.json`` remains for the notebook.
    """
    for k, v in row.items():
        if k.startswith("eval/") and isinstance(v, int | float):
            writer.add_scalar(k, float(v), env_steps)


def print_eval_row(row: dict[str, object]) -> None:
    """Pretty-print an eval row's equilibrium metrics + BRPS probe."""
    bits = [f"step={row['step']}"]
    for k in ("eval/exploitability", "eval/nash_conv", "eval/exact_nash_distance"):
        if k in row:
            text = f"{k.removeprefix('eval/')}={float(row[k]):.4g}"  # type: ignore[arg-type]
            if f"{k}_std" in row:  # sampled estimator reports a standard error
                text += f"±{float(row[f'{k}_std']):.2g}"  # type: ignore[arg-type]
            bits.append(text)
    if "brps/nash_distance" in row:
        bits.append(f"brps_nash_d={float(row['brps/nash_distance']):.4g}")  # type: ignore[arg-type]
        bits.append(
            f"P(R,P,S)="
            f"({float(row['brps/P_R']):.3f},{float(row['brps/P_P']):.3f},{float(row['brps/P_S']):.3f})"  # type: ignore[arg-type]
        )
    print("    eval: " + " ".join(bits))


def write_curve(path: Path, rows: list[dict[str, object]]) -> None:
    """Persist the training-curve rows as JSON (overwritten each eval)."""
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
