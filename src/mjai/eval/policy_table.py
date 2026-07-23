"""The final policy itself, as a table or a plot (AGENTS.md §4 "add a metric").

The eval curves say how far a run ended from Nash; they never show WHAT it
learned to play. This module materializes a finished run's policy over the game
tree and hands back something readable — a table for a 12-info-state game, a
heatmap for a 2000-row one, action marginals plus the most-visited rows for a
25000-row one.

Which of those is appropriate is a property of the game, not a preference, so
it is decided from measured enumeration cost (2026-07-23, this machine):

    brps               2 info states,      3 actions,   0.00 s
    kuhn              12                   2            0.00 s
    leduc            936                   3            0.06 s
    goofspiel5_ii   2124                   5            0.16 s
    liars_dice1    24576                  13            2.43 s
    ttt           294778                   9            6.04 s
    oshi_zumo        -- full enumeration did not finish in 90 s --

:data:`ENUMERABLE` is that table minus oshi_zumo. Asking for a full view of a
game outside it raises rather than hanging (it is the same size wall that makes
oshi_zumo use the sampled equilibrium estimator); :func:`root_policy` still
works there, since the opening distribution costs one forward pass.

Nothing here re-implements traversal: the policy is materialized by
:func:`mjai.eval.nash.tabular_view_of`, one batched forward over every info
state.

Visit counts are taken as INPUT rather than sampled here. ``mjai.eval`` sits
below ``mjai.pipeline`` in the layering, so it must not reach for a rollout
worker; :func:`visit_counts_from_batch` turns a batch the caller collected
into the counts, and ``tools/policy_view.py`` does that composition for the
notebooks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mjai.agents.base import Policy
from mjai.agents.ckpt_io import discover_checkpoints, read_manifest
from mjai.agents.policy_factory import load_policy_from_checkpoint
from mjai.algos.baselines import BRPS_EXACT_NASH, total_variation_distance
from mjai.eval.nash import row_players, state_observations, tabular_view_of
from mjai.games.loader import GameSpec, load_game

# Games whose full info-state enumeration is affordable, with the measured
# row count (see the module docstring). oshi_zumo is deliberately absent.
ENUMERABLE: dict[str, int] = {
    "brps": 2,
    "kuhn": 12,
    "leduc": 936,
    "goofspiel5_ii": 2124,
    "liars_dice1": 24576,
    "ttt": 294778,
}

# How to show each game's final policy, given the size above.
#   table      full per-info-state table
#   heatmap    info-state x action image + action marginals
#   top_k      action marginals + the most-visited info states
#   root       opening distribution only (enumeration infeasible)
VIEW_MODE: dict[str, str] = {
    "brps": "table",
    "kuhn": "table",
    "leduc": "heatmap",
    "goofspiel5_ii": "heatmap",
    "liars_dice1": "top_k",
    "ttt": "top_k",
    "oshi_zumo": "root",
}

_WHITESPACE = re.compile(r"\s+")


class PolicyViewError(ValueError):
    """The requested view cannot be produced for this game/run."""


@dataclass(frozen=True)
class PolicyView:
    """A finished run's policy over the info states it owns.

    Attributes:
        game: short game name.
        checkpoint: the checkpoint directory the policy came from.
        labels: one compacted info-state string per row.
        players: owning player per row.
        probs: ``(n_rows, n_actions)`` action probabilities; illegal = 0.
        legal: ``(n_rows, n_actions)`` legality mask.
        action_names: per-action display names.
        visits: per-row self-play visit counts, or None when not sampled.
        episodes: how many episodes produced ``visits``.
    """

    game: str
    checkpoint: Path
    labels: list[str]
    players: list[int]
    probs: np.ndarray
    legal: np.ndarray
    action_names: list[str]
    visits: np.ndarray | None = None
    episodes: int = 0

    @property
    def mode(self) -> str:
        """The display mode this game calls for (see :data:`VIEW_MODE`)."""
        return VIEW_MODE.get(self.game, "top_k")

    def action_marginals(self, player: int | None = None) -> np.ndarray:
        """Mean action probability over rows (optionally one player's rows).

        Averaged over info states, NOT over play: an info state reached once
        per thousand episodes counts as much as the opening. Read it as "what
        this policy tends to say", and use the visit-weighted table when you
        want "what it actually does".
        """
        rows = self._rows_for(player)
        return self.probs[rows].mean(axis=0) if len(rows) else np.zeros(self.probs.shape[1])

    def _rows_for(self, player: int | None) -> np.ndarray:
        players = np.asarray(self.players)
        return np.flatnonzero(players == player) if player is not None else np.arange(len(players))

    def top_rows(self, k: int, player: int | None = None) -> list[int]:
        """Row indices of the ``k`` most-visited info states (ties broken by row).

        Falls back to the first ``k`` rows when the view carries no visit
        counts — stated rather than silently pretending to rank by frequency.
        """
        rows = self._rows_for(player)
        if self.visits is None:
            return list(rows[:k])
        order = np.argsort(-self.visits[rows], kind="stable")
        return [int(rows[i]) for i in order[:k] if self.visits[rows[i]] > 0]


def _compact(label: str) -> str:
    """Collapse an info-state string onto one line (goofspiel's are multi-line)."""
    return _WHITESPACE.sub(" ", label).strip()


def _first_decision_state(spec: GameSpec) -> Any:
    """The opening decision state, chance resolved along its most likely branch."""
    state = spec.new_state()
    while state.is_chance_node():
        state.apply_action(max(state.chance_outcomes(), key=lambda o: o[1])[0])
    return state


def _action_names(spec: GameSpec) -> list[str]:
    """Per-action display names: "Pass"/"Bet", "1-3", "x(0,0)" — not raw ids.

    The STATE's ``action_to_string`` is what carries the readable names; the
    game-level one returns ``Action(id=0, player=0)`` for most of these games.
    Falls back per action, since ids that cannot occur at the opening state
    (Liar's Dice "call") raise there.
    """
    state = _first_decision_state(spec)
    names = []
    for a in range(spec.num_actions):
        for source in (state, spec.game):
            try:
                names.append(str(source.action_to_string(0, a)))
                break
            except Exception:  # not expressible from this source
                continue
        else:
            names.append(str(a))
    return names


def resolve_checkpoint(run_dir: str | Path, checkpoint: str = "best") -> Path:
    """Locate a run's checkpoint directory.

    ``best`` is the SOTA snapshot the probes copy aside (lowest equilibrium
    metric over the run); ``last`` is the final one; anything else is treated
    as a checkpoint directory name.
    """
    run = Path(run_dir)
    ckpt_root = run / "checkpoints"
    if checkpoint == "best":
        best = ckpt_root / "best"
        if best.is_dir():
            return best
        checkpoint = "last"  # no best marker (unfinished arm) -> newest
    if checkpoint == "last":
        # Rank the periodic snapshots by train step. discover_checkpoints sorts
        # by manifest created_at and also picks up checkpoints/best, whose
        # manifest was copied verbatim from whichever step it came from —
        # ordering by that would sometimes return "best" as "last".
        steps = [
            (m.train_step, d)
            for d, m in discover_checkpoints(ckpt_root)
            if d.name.startswith("step_")
        ]
        if not steps:
            raise PolicyViewError(f"{run}: no step_* checkpoints under {ckpt_root}")
        return max(steps)[1]
    target = ckpt_root / checkpoint
    if not target.is_dir():
        raise PolicyViewError(f"{run}: no checkpoint {checkpoint!r} under {ckpt_root}")
    return target


def load_run_policy(run_dir: str | Path, checkpoint: str = "best") -> tuple[Policy, GameSpec, Path]:
    """Rebuild a run's policy plus its game spec. Eval always plays on CPU."""
    ckpt = resolve_checkpoint(run_dir, checkpoint)
    policy = load_policy_from_checkpoint(ckpt, device="cpu")
    return policy, load_game(read_manifest(ckpt).game), ckpt


def observation_key(obs: Any) -> bytes:
    """Canonical dict key for one observation vector.

    The float32 bytes, which is exactly what
    :func:`mjai.eval.nash.state_observations` stores per enumerated info state
    — so sampled decisions and enumerated rows join without a second notion of
    state identity.
    """
    return np.asarray(obs, dtype=np.float32).tobytes()


def visit_counts_from_batch(batch: Any) -> dict[bytes, int]:
    """Decisions per observation in a collected batch, keyed by observation.

    Takes an already-collected :class:`~mjai.algos.transition.Batch` because
    collecting one needs a rollout worker, which lives a layer ABOVE this
    module. ``tools/policy_view.py`` is where the two are composed.
    """
    counts: dict[bytes, int] = {}
    for row in np.asarray(batch.obs, dtype=np.float32):
        key = row.tobytes()
        counts[key] = counts.get(key, 0) + 1
    return counts


def policy_view(
    run_dir: str | Path,
    *,
    checkpoint: str = "best",
    visit_counts: dict[bytes, int] | None = None,
    episodes: int = 0,
) -> PolicyView:
    """Materialize a finished run's policy over every info state.

    Args:
        run_dir: the arm directory (containing ``checkpoints/``).
        checkpoint: ``best`` | ``last`` | a checkpoint directory name.
        visit_counts: observation-keyed decision counts from self-play (see
            :func:`visit_counts_from_batch`), used to rank info states by how
            often they are actually reached. None leaves the view unranked.
        episodes: how many episodes produced ``visit_counts``; recorded on the
            view so the notebook can state the sample size.

    Raises:
        PolicyViewError: for a game outside :data:`ENUMERABLE` — enumerating
            it is what does not terminate, so it fails instead of hanging.
    """
    policy, spec, ckpt = load_run_policy(run_dir, checkpoint)
    if spec.name not in ENUMERABLE:
        raise PolicyViewError(
            f"{spec.name}: full info-state enumeration is infeasible "
            f"(>90 s and still running when measured), so there is no policy "
            f"table to build. Use root_policy() for the opening distribution."
        )
    tabular = tabular_view_of(spec, policy)
    # tabular_view_of hands back the CACHED skeleton with its probability array
    # overwritten in place; copy everything out before anything else evaluates.
    probs = np.asarray(tabular.action_probability_array, dtype=np.float64).copy()
    legal = np.asarray(tabular.legal_actions_mask, dtype=bool).copy()
    players = list(row_players(tabular))
    labels = [""] * len(players)
    for info, row in tabular.state_lookup.items():
        labels[row] = _compact(str(info))

    visits = None
    if visit_counts is not None:
        # Same per-info-state observation matrix the materialization enumerated
        # on (cached per game), so rows join to visits without re-walking the
        # tree and without a second notion of state identity.
        obs = state_observations(spec)
        visits = np.asarray([visit_counts.get(row.tobytes(), 0) for row in obs], dtype=np.int64)

    return PolicyView(
        game=spec.name,
        checkpoint=ckpt,
        labels=labels,
        players=players,
        probs=probs,
        legal=legal,
        action_names=_action_names(spec),
        visits=visits,
        episodes=episodes,
    )


def root_policy(run_dir: str | Path, *, checkpoint: str = "best") -> dict[int, dict[str, float]]:
    """Each player's action distribution at the opening state.

    One forward pass per player, so it works for every game including the ones
    whose full tree cannot be enumerated — for oshi_zumo the opening coin bid
    is the single most informative thing about a learned policy anyway.
    """
    policy, spec, _ = load_run_policy(run_dir, checkpoint)
    state = _first_decision_state(spec)
    names = _action_names(spec)
    out: dict[int, dict[str, float]] = {}
    for player in range(spec.num_players):
        legal = list(state.legal_actions(player))
        if not legal:
            continue
        logits = policy.action_logits(spec.obs_tensor(state, player), legal)
        arr = np.asarray(logits, dtype=np.float64)
        arr -= arr.max()
        exps = np.exp(arr)
        out[player] = {names[a]: float(p) for a, p in zip(legal, exps / exps.sum(), strict=True)}
    return out


def brps_nash_gap(view: PolicyView) -> tuple[np.ndarray, float]:
    """(seat-0 action distribution, TV distance to the analytic BRPS NE).

    BRPS has one info state per seat, so "the final policy" is literally three
    numbers and the paper's target (1/16, 10/16, 5/16) is known exactly.
    """
    if view.game != "brps":
        raise PolicyViewError(f"brps_nash_gap is BRPS-only; got {view.game!r}")
    rows = view._rows_for(0)
    probs = view.probs[rows[0]][: len(BRPS_EXACT_NASH)]
    return probs, float(total_variation_distance(probs, BRPS_EXACT_NASH))


def to_records(view: PolicyView, rows: list[int] | None = None) -> list[dict[str, object]]:
    """Rows as plain dicts: player, info state, visits, one column per action."""
    selected = range(len(view.labels)) if rows is None else rows
    out: list[dict[str, object]] = []
    for i in selected:
        rec: dict[str, object] = {"player": view.players[i], "info_state": view.labels[i]}
        if view.visits is not None:
            rec["visits"] = int(view.visits[i])
            rec["visit_share"] = float(view.visits[i]) / max(int(view.visits.sum()), 1)
        for a, name in enumerate(view.action_names):
            rec[name] = float(view.probs[i][a]) if view.legal[i][a] else float("nan")
        out.append(rec)
    return out


def plot_policy(view: PolicyView, *, ax: Any, player: int = 0) -> None:
    """Draw the mode-appropriate picture of ``view`` into ``ax``.

    ``table`` games get a bar chart of the per-info-state distributions (a
    12-row table is also printed alongside, so the plot is the shape summary);
    ``heatmap`` games get an info-state x action image; ``top_k`` games get
    action marginals, since an image 24576 rows tall reads as noise.
    """
    rows = view._rows_for(player)
    if view.mode == "table" and len(rows) <= 8:
        _plot_per_state_bars(view, ax, rows)
    elif view.mode == "heatmap":
        _plot_heatmap(view, ax, rows)
    else:
        _plot_marginals(view, ax, rows)


def _plot_per_state_bars(view: PolicyView, ax: Any, rows: np.ndarray) -> None:
    """Grouped bars: one group per info state, one bar per action."""
    n_act = len(view.action_names)
    width = 0.8 / max(n_act, 1)
    x = np.arange(len(rows))
    for a, name in enumerate(view.action_names):
        vals = [view.probs[i][a] if view.legal[i][a] else 0.0 for i in rows]
        ax.bar(x + a * width - 0.4 + width / 2, vals, width, label=name)
    if view.game == "brps":  # analytic NE is known exactly -- show the target
        for value, color in zip(
            BRPS_EXACT_NASH, ("tab:blue", "tab:orange", "tab:green"), strict=False
        ):
            ax.axhline(value, color=color, ls="--", lw=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [view.labels[i] or "(root)" for i in rows], rotation=45, ha="right", fontsize=7
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("action probability")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)


def _plot_heatmap(view: PolicyView, ax: Any, rows: np.ndarray) -> None:
    """Info-state x action image, rows ordered by visits when available."""
    order = view.top_rows(len(rows), player=None) if view.visits is not None else list(rows)
    order = [i for i in order if i in set(rows.tolist())] or list(rows)
    data = np.where(view.legal[order], view.probs[order], np.nan)
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks(range(len(view.action_names)))
    ax.set_xticklabels(view.action_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"info state (n={len(order)})")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_marginals(view: PolicyView, ax: Any, rows: np.ndarray) -> None:
    """Mean action probability over info states, plus the visit-weighted mean.

    The two differ exactly when the policy behaves differently on the states it
    actually reaches — which is the interesting case, so both are drawn rather
    than picking one and calling it "the" action distribution.
    """
    marg = view.probs[rows].mean(axis=0) if len(rows) else np.zeros(len(view.action_names))
    x = np.arange(len(view.action_names))
    ax.bar(x - 0.2, marg, 0.4, label="mean over info states")
    if view.visits is not None and view.visits[rows].sum() > 0:
        w = view.visits[rows].astype(np.float64)
        weighted = (view.probs[rows] * w[:, None]).sum(axis=0) / w.sum()
        ax.bar(x + 0.2, weighted, 0.4, label="visit-weighted")
    ax.set_xticks(x)
    ax.set_xticklabels(view.action_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("action probability")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)


def write_csv(view: PolicyView, path: str | Path) -> Path:
    """Dump the FULL table to CSV — the escape hatch for un-plottable games."""
    import csv

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = to_records(view)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]) if records else ["info_state"])
        writer.writeheader()
        writer.writerows(records)
    return out


__all__ = [
    "ENUMERABLE",
    "VIEW_MODE",
    "PolicyView",
    "PolicyViewError",
    "brps_nash_gap",
    "load_run_policy",
    "observation_key",
    "plot_policy",
    "policy_view",
    "resolve_checkpoint",
    "root_policy",
    "to_records",
    "visit_counts_from_batch",
    "write_csv",
]
