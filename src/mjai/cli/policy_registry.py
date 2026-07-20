"""Policy registry: discover saved policies from run dirs (AGENTS.md §1 D10)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mjai.agents.ckpt_io import CheckpointManifest, discover_checkpoints


@dataclass(frozen=True)
class PolicyEntry:
    """One loadable policy in the CLI's picker."""

    path: Path
    manifest: CheckpointManifest
    label: str  # human-readable name for the menu


def list_policies(root: str | Path = "runs") -> list[PolicyEntry]:
    """Discover every saved policy under ``root`` via checkpoint manifests.

    Returns entries sorted newest-first. Empty list if ``root`` doesn't exist.
    """
    entries: list[PolicyEntry] = []
    for path, manifest in discover_checkpoints(root):
        algo = manifest.algo
        mode = manifest.self_play_mode
        step = manifest.train_step
        score = "" if manifest.eval_score is None else f" ({manifest.eval_score:.3g})"
        label = f"{manifest.game}/{algo}/{mode} step={step}{score}"
        entries.append(PolicyEntry(path=path, manifest=manifest, label=label))
    # Newest first (discover_checkpoints sorts oldest-first).
    return list(reversed(entries))
