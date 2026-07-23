"""League/mirror run auditor: a health card from a finished run's TB log.

Answers "why is this league run converging slower than its mirror twin?" from
data that already exists on disk — no retraining. Four things it checks, each
one a defect that a plain exploitability curve cannot distinguish from "league
is simply a harder objective":

  1. **On-policy invariant.** Every gradient step must be applied to the policy
     that collected its batch. A self-play controller that rotates roles but
     hands every batch to one learner's UpdateRule violates this, and it shows
     up as ``train/off_policy_frac > 0`` (new runs) or a nonzero
     ``train/approx_kl`` (older runs, which predate that scalar). ``approx_kl``
     is read from the same forward pass that produces the gradient, BEFORE the
     optimizer step, so an on-policy batch gives exactly 0.0 and any nonzero
     value is a real behavior/target mismatch rather than staleness.
  2. **Budget currency.** ``env_steps`` counts the samples a round RETAINED.
     Mirror keeps both seats, league keeps one, so the same nominal env-step
     budget buys different numbers of updates, different batch sizes and
     different amounts of simulation. All three are reported so a curve is
     never read against the wrong x-axis.
  3. **Per-role telemetry.** With a period-3 role schedule the residue of the
     update index identifies the role, so gate/gradient/importance-weight
     statistics can be split by role.
  4. **League churn.** Promotion rate and snapshot cadence relative to the pool
     capacity: a pool that turns over faster than it is filled has no history
     left to sample from.

Usage::

    python tools/league_diagnose.py runs/nb_ab/liars_dice1/liars_dice1_league/seed_0
    python tools/league_diagnose.py --glob 'runs/nb_ab/*/*/seed_*' --json card.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tb_eval import read_tags

# Role schedule of LeagueSelfPlay, in collect() order. Update index i (1-based,
# as logged) ran role_schedule[(i - 1) % 3], so MAIN sits at i % 3 == 1.
ROLE_ORDER = ("main", "main_exploiter", "league_exploiter")
ROLE_PERIOD = len(ROLE_ORDER)
# Distinct short labels for the compact per-role table ("main" and
# "main_exploiter" share a prefix, so they cannot be abbreviated by splitting).
ROLE_ABBREV = {"main": "MAIN", "main_exploiter": "ME", "league_exploiter": "LE"}

# A batch is off-policy if its behavior log-probs disagree with the updating
# policy's. Same tolerance as mjai.algos.nn_updates.OFF_POLICY_TOL (2e-6: 1.2x
# the measured max float32 noise from single-row vs batched forward).
KL_TOL = 2e-6
# Fraction of updates allowed to be off-policy before the invariant fails.
OFF_POLICY_BUDGET = 1e-3
# Promotions per exploiter round above this means the pool is churning, not
# accumulating: every promotion evicts a member from a capacity-bounded store.
PROMO_RATE_BUDGET = 0.10

PER_ROLE_TAGS = (
    "train/gate_off_frac",
    "train/grad_norm",
    "train/iw_max",
    "train/entropy",
    "train/policy_loss",
)
TAGS = (
    "train/approx_kl",
    "train/off_policy_frac",
    "train/sampled_steps",
    "train/batch_size",
    "league/pool_size",
    "league/promotions_total",
    "league/main_snapshots_total",
    "league/pool_main",
    "league/pool_main_exploiter",
    "league/pool_league_exploiter",
    "league/exploiter_true_winrate/main_exploiter",
    "league/exploiter_true_winrate/league_exploiter",
    "eval/exploitability",
    "eval/nash_conv",
    *PER_ROLE_TAGS,
)


def _stats(values: list[float]) -> dict[str, float]:
    """mean/median/p90/max of a non-empty list."""
    ordered = sorted(values)
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max": ordered[-1],
    }


def _identity(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {"run": str(run_dir)}
    cfg = json.loads(cfg_path.read_text())
    keys = (
        "game",
        "algo",
        "theta",
        "self_play_mode",
        "policy_kind",
        "seed",
        "target_samples",
        "total_env_steps",
        "league_capacity",
        "league_main_save_every_rounds",
        "league_promo_window",
    )
    return {"run": str(run_dir), **{k: cfg.get(k) for k in keys}}


def _off_policy_series(series: dict[str, list[tuple[int, float]]]) -> tuple[list[tuple[int, bool]], str]:
    """Per-update (step, is_off_policy) plus which scalar it came from.

    Prefers the explicit ``off_policy_frac`` probe; falls back to ``approx_kl``
    for runs logged before that scalar existed. Both are read pre-step, so both
    are exactly 0 on an on-policy batch.
    """
    if "train/off_policy_frac" in series:
        return [(s, v > 0.0) for s, v in series["train/off_policy_frac"]], "off_policy_frac"
    if "train/approx_kl" in series:
        return [(s, abs(v) > KL_TOL) for s, v in series["train/approx_kl"]], "approx_kl (fallback)"
    return [], "none"


def _role_phase(flags: list[tuple[int, bool]]) -> dict[int, str] | None:
    """Map residue (step % 3) -> role name, or None if the pattern is not clean.

    Self-validating on purpose: rather than trusting that the schedule is the
    default 3-role rotation, it requires exactly one residue to be essentially
    all-on-policy and the other two to be predominantly off-policy. Anything
    else (mirror, a custom schedule, an already-fixed league) returns None and
    the per-role split is simply skipped instead of being invented.
    """
    if not flags:
        return None
    rates: dict[int, float] = {}
    for residue in range(ROLE_PERIOD):
        vals = [off for step, off in flags if step % ROLE_PERIOD == residue]
        if not vals:
            return None
        rates[residue] = sum(vals) / len(vals)
    clean = [r for r, rate in rates.items() if rate <= 0.01]
    dirty = [r for r, rate in rates.items() if rate >= 0.50]
    if len(clean) != 1 or len(dirty) != ROLE_PERIOD - 1:
        return None
    main = clean[0]
    return {(main + i) % ROLE_PERIOD: name for i, name in enumerate(ROLE_ORDER)}


def _budget(series: dict[str, list[tuple[int, float]]], ident: dict[str, Any]) -> dict[str, Any]:
    """The three currencies a curve can legitimately be plotted against."""
    curve = series.get("eval/exploitability") or series.get("eval/nash_conv") or []
    any_train = next((series[t] for t in PER_ROLE_TAGS if t in series), [])
    updates = len(any_train)
    if "train/batch_size" in series:
        retained: int | None = int(sum(v for _, v in series["train/batch_size"]))
    else:
        # Pre-fix runs log neither batch size nor simulated steps, and their
        # env_steps counted RETAINED samples — so the eval scalars' step key is
        # the only reading available. A run with no eval point yet has none at
        # all: report None rather than 0, which would read as "retained
        # nothing" and fail the batch-size check for the wrong reason.
        retained = int(curve[-1][0]) if curve else None
    out: dict[str, Any] = {
        "updates": updates,
        "retained_samples": retained,
        "mean_retained_batch": (retained / updates) if (retained and updates) else None,
        "target_samples": ident.get("target_samples"),
    }
    if "train/sampled_steps" in series:
        out["simulated_decision_points"] = int(sum(v for _, v in series["train/sampled_steps"]))
        out["simulated_estimated"] = False
    elif updates and ident.get("target_samples"):
        # Pre-fix runs do not log what the rollout actually simulated. The
        # rollout collects whole episodes until the POOLED batch reaches
        # target_samples, so updates * target_samples is a lower bound on the
        # decision points played (it ignores the final episode's overshoot).
        out["simulated_decision_points"] = updates * int(ident["target_samples"])
        out["simulated_estimated"] = True
    return out


def _league_health(series: dict[str, list[tuple[int, float]]], updates: int) -> dict[str, Any]:
    def last(tag: str) -> float | None:
        pts = series.get(tag)
        return pts[-1][1] if pts else None

    def mean(tag: str) -> float | None:
        pts = series.get(tag)
        return statistics.mean(v for _, v in pts) if pts else None

    promotions = last("league/promotions_total")
    exploiter_rounds = updates * (ROLE_PERIOD - 1) / ROLE_PERIOD if updates else 0
    return {
        "pool_size": last("league/pool_size"),
        "pool_composition": {
            role: last(f"league/pool_{role}")
            for role in ROLE_ORDER
            if f"league/pool_{role}" in series
        }
        or None,
        "promotions_total": promotions,
        "main_snapshots_total": last("league/main_snapshots_total"),
        "promotions_per_exploiter_round": (
            promotions / exploiter_rounds if promotions is not None and exploiter_rounds else None
        ),
        "main_exploiter_winrate_mean": mean("league/exploiter_true_winrate/main_exploiter"),
        "league_exploiter_winrate_mean": mean("league/exploiter_true_winrate/league_exploiter"),
    }


def _checks(card: dict[str, Any]) -> list[dict[str, Any]]:
    """Pass/fail verdicts. A check whose input is missing is reported SKIP."""
    out: list[dict[str, Any]] = []

    def add(name: str, ok: bool | None, detail: str) -> None:
        out.append({"check": name, "verdict": "SKIP" if ok is None else ("PASS" if ok else "FAIL"), "detail": detail})

    frac = card["on_policy"]["off_policy_fraction"]
    add(
        "updates_are_on_policy",
        None if frac is None else frac <= OFF_POLICY_BUDGET,
        f"{frac:.4f} of updates applied a batch the updated policy did not collect"
        if frac is not None
        else "no off_policy_frac / approx_kl scalar in this run",
    )

    budget = card["budget"]
    target = budget.get("target_samples")
    mean_batch = budget.get("mean_retained_batch")
    add(
        "batch_reaches_target",
        None if not target or mean_batch is None else mean_batch >= 0.9 * target,
        f"mean retained batch {mean_batch:.1f} vs target_samples {target}"
        if mean_batch is not None
        else "no eval point yet, so retained samples are unknown",
    )

    if card["identity"].get("self_play_mode") == "league":
        rate = card["league"]["promotions_per_exploiter_round"]
        add(
            "promotion_rate_sane",
            None if rate is None else rate <= PROMO_RATE_BUDGET,
            f"{rate:.3f} promotions per exploiter round (budget {PROMO_RATE_BUDGET})"
            if rate is not None
            else "no promotions_total scalar",
        )
        comp = card["league"]["pool_composition"]
        add(
            "pool_keeps_main_history",
            None if not comp else bool(comp.get("main", 0)),
            f"pool composition {comp}" if comp else "per-role pool composition not logged",
        )
    return out


def diagnose(run_dir: Path) -> dict[str, Any]:
    """Build one run's health card."""
    series = read_tags(run_dir / "tb", TAGS)
    ident = _identity(run_dir)
    flags, source = _off_policy_series(series)
    phase = _role_phase(flags)
    budget = _budget(series, ident)

    on_policy: dict[str, Any] = {
        "source": source,
        "n_updates": len(flags),
        "off_policy_fraction": (sum(off for _, off in flags) / len(flags)) if flags else None,
    }
    if "train/approx_kl" in series:
        nonzero = [abs(v) for _, v in series["train/approx_kl"] if abs(v) > KL_TOL]
        on_policy["abs_kl_when_off_policy"] = _stats(nonzero) if nonzero else None
    # Both per-role tables are keyed in ROLE_ORDER, not residue order, so the
    # columns read main -> main-exploiter -> league-exploiter everywhere.
    residue_of = {role: r for r, role in phase.items()} if phase else {}
    if phase:
        on_policy["by_role"] = {
            role: {
                "n": sum(1 for s, _ in flags if s % ROLE_PERIOD == residue_of[role]),
                "off_policy_fraction": statistics.mean(
                    [float(off) for s, off in flags if s % ROLE_PERIOD == residue_of[role]]
                ),
            }
            for role in ROLE_ORDER
        }

    per_role: dict[str, Any] = {}
    if phase:
        for tag in PER_ROLE_TAGS:
            if tag not in series:
                continue
            per_role[tag.removeprefix("train/")] = {
                role: statistics.mean(
                    [v for s, v in series[tag] if s % ROLE_PERIOD == residue_of[role]]
                )
                for role in ROLE_ORDER
            }

    card: dict[str, Any] = {
        "identity": ident,
        "on_policy": on_policy,
        "budget": budget,
        "per_role_mean": per_role or None,
        "league": _league_health(series, budget["updates"]),
    }
    card["checks"] = _checks(card)
    return card


def render(card: dict[str, Any]) -> str:
    """Human-readable one-run summary."""
    ident = card["identity"]
    lines = [
        f"=== {ident['run']}",
        f"    {ident.get('game')} / {ident.get('algo')} / {ident.get('self_play_mode')} / seed {ident.get('seed')}",
    ]
    op = card["on_policy"]
    frac = op["off_policy_fraction"]
    lines.append(
        f"  on-policy   : {op['n_updates']} updates, off-policy "
        f"{'n/a' if frac is None else f'{frac:.3f}'}  (source: {op['source']})"
    )
    for role, sub in (op.get("by_role") or {}).items():
        lines.append(f"      {role:18s} n={sub['n']:6d}  off-policy {sub['off_policy_fraction']:.3f}")
    if op.get("abs_kl_when_off_policy"):
        k = op["abs_kl_when_off_policy"]
        lines.append(
            f"      |KL| when off-policy: median {k['median']:.3f}  p90 {k['p90']:.3f}  max {k['max']:.3f}"
        )
    b = card["budget"]
    sim = b.get("simulated_decision_points")
    batch = b["mean_retained_batch"]
    lines.append(
        f"  budget      : {b['updates']} updates | {b['retained_samples']} retained samples "
        f"(batch {'n/a' if batch is None else f'{batch:.1f}'}/{b.get('target_samples')}) | "
        f"{sim if sim is not None else 'n/a'} simulated"
        + (" (estimated lower bound)" if b.get("simulated_estimated") else "")
    )
    for name, by_role in (card.get("per_role_mean") or {}).items():
        cells = "  ".join(f"{ROLE_ABBREV[r]}={v:.3f}" for r, v in by_role.items())
        lines.append(f"      {name:16s} {cells}")
    lg = card["league"]
    if lg.get("promotions_total") is not None:
        lines.append(
            f"  league      : pool {lg['pool_size']} | promotions {lg['promotions_total']:.0f} "
            f"({lg['promotions_per_exploiter_round']:.3f}/exploiter-round) | "
            f"main snapshots {lg['main_snapshots_total']:.0f} | composition {lg['pool_composition']}"
        )
    for c in card["checks"]:
        lines.append(f"  [{c['verdict']:4s}] {c['check']}: {c['detail']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*", type=Path, help="run directories (each holding tb/ and config.json)")
    ap.add_argument("--glob", help="shell glob for run dirs, relative to cwd")
    ap.add_argument("--json", type=Path, help="write all cards to this JSON file")
    args = ap.parse_args(argv)

    dirs = list(args.run_dirs)
    if args.glob:
        dirs += [p for p in sorted(Path().glob(args.glob)) if (p / "tb").is_dir()]
    if not dirs:
        ap.error("no run directories given (pass paths or --glob)")

    cards = []
    failures = 0
    for run_dir in dirs:
        card = diagnose(run_dir)
        cards.append(card)
        print(render(card))
        failures += sum(1 for c in card["checks"] if c["verdict"] == "FAIL")
    if args.json:
        args.json.write_text(json.dumps(cards, indent=2))
        print(f"\nwrote {args.json} ({len(cards)} runs)")
    print(f"\n{failures} failed check(s) across {len(cards)} run(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
