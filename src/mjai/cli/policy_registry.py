"""Policy registry: discover saved policies from run dirs (AGENTS.md §1 D10).

Discovery + the pure filtering/paging helpers behind the Play CLI's picker
(F5): filter to the selected game, drop checkpoints whose obs/action space
does not match, substring-filter the visible list, and page the newest ~20.
These functions are deliberately UI-free so they are unit-testable (§5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mjai.agents.ckpt_io import CheckpointManifest, discover_checkpoints

# Default page size for the CLI's policy picker; the remainder is reachable
# via substring filtering (F5).
DEFAULT_PAGE_SIZE = 20


@dataclass(frozen=True)
class PolicyEntry:
    """One loadable policy in the CLI's picker."""

    path: Path
    manifest: CheckpointManifest
    label: str  # human-readable name for the menu


def list_policies(root: str | Path = "runs", *, game: str | None = None) -> list[PolicyEntry]:
    """Discover saved policies under ``root``, newest first by train step.

    Args:
        root: run tree to scan. Empty list if ``root`` doesn't exist.
        game: if set, keep only checkpoints whose manifest was trained on this
            game short name (F5: the picker never shows cross-game policies).

    Sort: ``train_step`` descending, ties broken by ``created_at`` descending.
    """
    entries: list[PolicyEntry] = []
    for path, manifest in discover_checkpoints(root):
        if game is not None and manifest.game != game:
            continue
        entries.append(
            PolicyEntry(path=path, manifest=manifest, label=_label(path, manifest, root))
        )
    entries.sort(key=lambda e: (e.manifest.train_step, e.manifest.created_at), reverse=True)
    return entries


def _label(path: Path, manifest: CheckpointManifest, root: str | Path) -> str:
    score = "" if manifest.eval_score is None else f" ({manifest.eval_score:.3g})"
    head = f"{manifest.game}/{manifest.algo}/{manifest.self_play_mode} step={manifest.train_step}{score}"
    try:
        rel = path.relative_to(root)
    except ValueError:  # different drive / not under root; show the full path
        rel = path
    return f"{head}  [{rel}]"


def compatible_with(
    entries: list[PolicyEntry], *, obs_size: int, num_actions: int
) -> tuple[list[PolicyEntry], list[PolicyEntry]]:
    """Split entries into (compatible, incompatible) for a loaded game.

    Compatibility means the checkpoint's observation length and action-space
    size match the game's spec — loading an incompatible checkpoint would
    either crash or, worse, silently play garbage (§11: fail loudly upstream).
    """
    ok: list[PolicyEntry] = []
    bad: list[PolicyEntry] = []
    for e in entries:
        if e.manifest.obs_size == obs_size and e.manifest.num_actions == num_actions:
            ok.append(e)
        else:
            bad.append(e)
    return ok, bad


def filter_labels(entries: list[PolicyEntry], text: str) -> list[PolicyEntry]:
    """Case-insensitive substring filter over the menu label."""
    needle = text.strip().lower()
    if not needle:
        return list(entries)
    return [e for e in entries if needle in e.label.lower()]


def page(
    entries: list[PolicyEntry], size: int = DEFAULT_PAGE_SIZE
) -> tuple[list[PolicyEntry], int]:
    """Return (first ``size`` entries, how many more remain unshown)."""
    if size <= 0:
        raise ValueError(f"page size must be positive, got {size}")
    return entries[:size], max(0, len(entries) - size)
