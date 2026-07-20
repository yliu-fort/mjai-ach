"""Canonical checkpoint manifest + I/O (AGENTS.md §4, §10).

A checkpoint on disk is a directory with:
  - ``manifest.json`` — provenance metadata (game, algo, self-play mode, train
    step, eval score, obs kind/size, num_actions). The single source of truth
    that both training and the Play CLI read to discover policies.
  - ``policy.<ext>`` — the weight blob. Extension/format is policy-kind-specific:
      - tabular -> ``policy.json`` (or ``.pkl``) via :meth:`TabularPolicy.save`
      - NN      -> ``policy.pt`` (torch ``state_dict``) — added in Step 2 NN.

This module owns the manifest schema and the load-by-manifest dispatch. Policy
classes own their own weight serialization (called back via ``Policy.save``).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CheckpointManifest:
    """Provenance for a saved policy. Written as ``manifest.json``."""

    game: str  # GameSpec.name, e.g. "kuhn"
    game_string: str  # canonical pyspiel string
    algo: str  # "ppo" | "ach" | "cfr" | ...
    self_play_mode: str  # "mirror" | "league" | "baseline"
    policy_kind: str  # "tabular" | "mlp" | ...
    num_actions: int
    obs_kind: str  # "information_state" | "observation"
    obs_size: int
    train_step: int = 0
    eval_score: float | None = None  # e.g. exploitability; None if not computed
    created_at: float = field(default_factory=time.time)
    notes: str = ""

    @property
    def is_neural(self) -> bool:
        return self.policy_kind != "tabular"

    def weight_filename(self) -> str:
        return "policy.pt" if self.is_neural else "policy.json"


MANIFEST_NAME = "manifest.json"


def write_checkpoint(
    directory: str | os.PathLike[str],
    manifest: CheckpointManifest,
) -> Path:
    """Create ``directory/`` and write ``manifest.json``. Returns the directory.

    The caller is responsible for writing the weight blob immediately after,
    using ``manifest.weight_filename()`` and the Policy's own ``save()``.
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    out = d / MANIFEST_NAME
    out.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return d


def read_manifest(directory: str | os.PathLike[str]) -> CheckpointManifest:
    """Load a ``CheckpointManifest`` from a checkpoint directory."""
    p = Path(directory) / MANIFEST_NAME
    if not p.is_file():
        raise FileNotFoundError(f"No {MANIFEST_NAME} in {directory!r}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return CheckpointManifest(**data)


def discover_checkpoints(root: str | os.PathLike[str]) -> list[tuple[Path, CheckpointManifest]]:
    """Walk ``root`` and return every ``(dir, manifest)`` pair found.

    Used by the Play CLI's policy registry to list all playable policies.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    out: list[tuple[Path, CheckpointManifest]] = []
    for dirpath, _dirs, files in os.walk(root_path):
        if MANIFEST_NAME in files:
            try:
                m = read_manifest(dirpath)
                out.append((Path(dirpath), m))
            except (OSError, ValueError, TypeError):
                continue  # corrupt manifest; skip
    out.sort(key=lambda x: x[1].created_at)
    return out


def checkpoint_name(manifest: CheckpointManifest, *, slug_max: int = 40) -> str:
    """Human-readable short name for display in the Play CLI's policy picker."""
    score = "" if manifest.eval_score is None else f"_{manifest.eval_score:.3g}"
    return (
        f"{manifest.game}_{manifest.algo}_{manifest.self_play_mode}_s{manifest.train_step}{score}"
    )
