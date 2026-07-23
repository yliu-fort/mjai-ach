"""Rebuild a :class:`Policy` from a checkpoint directory (AGENTS.md §1 D10).

Single entry point for turning an on-disk checkpoint back into a usable
policy — the Play CLI, eval scripts, and notebooks all go through here.
Callers receive the abstract :class:`mjai.agents.base.Policy`; they never
branch on ``policy_kind`` or downcast themselves (§3.3).

Architecture provenance for neural checkpoints, in priority order:

1. The ``policy.pt.meta.json`` sidecar written by ``MLP.save`` (authoritative:
   hidden_sizes, activation, obs_size, num_actions).
2. Legacy checkpoints whose sidecar is missing or predates the ``activation``
   key: derive from the run's dumped ``config.json`` (found by walking up
   ancestor directories — the ``<run>/checkpoints/step_N`` layout puts it two
   levels up), after cross-checking every overlapping field against the
   manifest/sidecar.
3. Anything still unknown or inconsistent is a loud :class:`CheckpointLoadError`
   — never a silent default (§1 D6, §11).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mjai.agents.base import Policy
from mjai.agents.ckpt_io import CheckpointManifest, read_manifest
from mjai.agents.tabular import TabularPolicy

# How many ancestor directories to search for a run's dumped config.json.
# Covers ``<run>/checkpoints/step_N`` (2 levels) plus headroom for league nests.
_CONFIG_SEARCH_DEPTH = 4

# Basename of the config dump written next to a run's checkpoints/ (AGENTS.md §9).
_RUN_CONFIG_NAME = "config.json"


class CheckpointLoadError(RuntimeError):
    """A checkpoint could not be reconstructed into a Policy.

    The message is user-readable (cause + remedy) so the Play CLI can print it
    as a one-line error instead of surfacing a torch/key-error traceback.
    """


@dataclass(frozen=True)
class _MlpArch:
    """Reconstruction parameters for an MLP checkpoint."""

    hidden_sizes: tuple[int, ...]
    activation: str  # key into mlp.ACTIVATIONS
    trunk_layernorm: bool


def load_policy_from_checkpoint(
    directory: str | os.PathLike[str], *, device: str | None = None
) -> Policy:
    """Rebuild the Policy stored in a checkpoint directory.

    Args:
        directory: checkpoint dir containing ``manifest.json`` + weight blob.
        device: optional torch device override for neural policies; None uses
            gpu_assert resolution (the Play CLI opts into CPU explicitly).

    Raises:
        CheckpointLoadError: user-readable reason when anything is missing,
            inconsistent, or unknown. Never a raw torch/KeyError traceback.
    """
    d = Path(directory)
    manifest = _read_manifest_loud(d)
    policy = _construct_policy(d, manifest, device=device)
    weight = d / manifest.weight_filename()
    if not weight.is_file():
        raise CheckpointLoadError(
            f"{d}: missing weight file {weight.name!r}; the checkpoint is incomplete."
        )
    try:
        policy.load(str(weight))
    except Exception as e:
        raise CheckpointLoadError(
            f"{d}: weights in {weight.name!r} do not fit the reconstructed "
            f"{manifest.policy_kind} architecture ({e}). The checkpoint may be "
            f"corrupt or written by an incompatible mjai version."
        ) from e
    return policy


def _read_manifest_loud(d: Path) -> CheckpointManifest:
    try:
        return read_manifest(d)
    except FileNotFoundError as e:
        raise CheckpointLoadError(f"{d}: no manifest.json — not a checkpoint directory.") from e
    except (ValueError, TypeError) as e:
        raise CheckpointLoadError(f"{d}: manifest.json is unreadable ({e}).") from e


def _construct_policy(d: Path, manifest: CheckpointManifest, *, device: str | None) -> Policy:
    if manifest.policy_kind == "tabular":
        return TabularPolicy(num_actions=manifest.num_actions, seed=0)
    if manifest.policy_kind == "mlp":
        arch = _mlp_architecture(d, manifest)
        # Lazy import: keeps torch out of the tabular-only load path (and out
        # of `import mjai.agents.policy_factory` for tabular-only callers).
        from mjai.agents.mlp import ACTIVATIONS, MLPSharedActorCritic

        activation = ACTIVATIONS.get(arch.activation)
        if activation is None:
            known = ", ".join(sorted(ACTIVATIONS))
            raise CheckpointLoadError(
                f"{d}: unknown activation {arch.activation!r} (known: {known}); "
                f"the checkpoint was written by an incompatible mjai version."
            )
        return MLPSharedActorCritic(
            obs_size=manifest.obs_size,
            num_actions=manifest.num_actions,
            hidden_sizes=arch.hidden_sizes,
            activation=activation,
            trunk_layernorm=arch.trunk_layernorm,
            device=device,
            # Weights are overwritten by load() right after; seeding the init
            # would only clobber the process-global torch RNG for nothing.
            seed=None,
        )
    raise CheckpointLoadError(
        f"{d}: unknown policy_kind {manifest.policy_kind!r} in manifest.json; "
        f"this mjai version cannot rebuild it."
    )


def _mlp_architecture(d: Path, manifest: CheckpointManifest) -> _MlpArch:
    """Resolve (hidden_sizes, activation) from sidecar first, run config second."""
    meta = _read_sidecar(d, manifest)
    config = _find_run_config(d)
    _cross_check_provenance(d, manifest, meta, config)

    hidden = _derive_hidden_sizes(d, meta, config)
    activation = _derive_activation(d, meta, config)
    ln = _derive_trunk_layernorm(meta, config)
    return _MlpArch(hidden_sizes=hidden, activation=activation, trunk_layernorm=ln)


def _derive_trunk_layernorm(meta: dict[str, Any] | None, config: dict[str, Any] | None) -> bool:
    """Sidecar first, run config second, else False (historical architecture).

    Unlike the activation, a wrong guess here cannot load silently: LayerNorm
    adds parameters, so a mismatch fails the state_dict load with a loud
    CheckpointLoadError. That makes False a safe last resort for the many
    pre-LayerNorm checkpoints rather than a silent-default hazard (§11).
    """
    if meta is not None and meta.get("trunk_layernorm") is not None:
        return bool(meta["trunk_layernorm"])
    if config is not None and config.get("trunk_layernorm") is not None:
        return bool(config["trunk_layernorm"])
    return False


def _read_sidecar(d: Path, manifest: CheckpointManifest) -> dict[str, Any] | None:
    """The ``policy.pt.meta.json`` written by MLP.save; None if absent."""
    # Mirror MLP.save's naming: Path("policy.pt").with_suffix(".pt.meta.json").
    sidecar = (d / manifest.weight_filename()).with_suffix(".pt.meta.json")
    if not sidecar.is_file():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CheckpointLoadError(f"{d}: corrupt sidecar {sidecar.name} ({e}).") from e
    if not isinstance(data, dict):
        raise CheckpointLoadError(f"{d}: sidecar {sidecar.name} is not a JSON object.")
    return data


def _find_run_config(d: Path) -> dict[str, Any] | None:
    """Nearest ancestor ``config.json`` that looks like a dumped ExperimentConfig.

    A file counts only if it carries ``policy_kind`` (every run dump has it);
    unrelated config.json files are skipped. Unparseable files are skipped too
    — they are not the provenance we are looking for.
    """
    current = d
    for _ in range(_CONFIG_SEARCH_DEPTH):
        current = current.parent
        candidate = current / _RUN_CONFIG_NAME
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and "policy_kind" in data:
            return data
    return None


def _cross_check_provenance(
    d: Path,
    manifest: CheckpointManifest,
    meta: dict[str, Any] | None,
    config: dict[str, Any] | None,
) -> None:
    """Fail loudly when provenance sources disagree on an overlapping field."""

    def _check(field: str, a: Any, b: Any, a_name: str, b_name: str) -> None:
        if a is not None and b is not None and a != b:
            raise CheckpointLoadError(
                f"{d}: conflicting provenance for {field!r}: {a_name} says {a!r} "
                f"but {b_name} says {b!r}. Refusing to guess (AGENTS.md §11)."
            )

    if meta is not None:
        _check("obs_size", meta.get("obs_size"), manifest.obs_size, "sidecar", "manifest")
        _check("num_actions", meta.get("num_actions"), manifest.num_actions, "sidecar", "manifest")
    if config is not None:
        _check("game", config.get("game"), manifest.game, "run config.json", "manifest")
        if meta is not None:
            _check(
                "hidden_sizes",
                config.get("hidden_sizes"),
                meta.get("hidden_sizes"),
                "run config.json",
                "sidecar",
            )
            _check(
                "activation",
                _norm_str(config.get("activation")),
                _norm_str(meta.get("activation")),
                "run config.json",
                "sidecar",
            )


def _norm_str(value: Any) -> str | None:
    return value.lower() if isinstance(value, str) else None


def _derive_hidden_sizes(
    d: Path, meta: dict[str, Any] | None, config: dict[str, Any] | None
) -> tuple[int, ...]:
    raw: Any = None
    if meta is not None and meta.get("hidden_sizes") is not None:
        raw = meta["hidden_sizes"]
    elif config is not None and config.get("hidden_sizes") is not None:
        raw = config["hidden_sizes"]
    if raw is None:
        raise CheckpointLoadError(
            f"{d}: cannot determine the MLP hidden sizes — no sidecar "
            f"policy.pt.meta.json and no run config.json with 'hidden_sizes' "
            f"found nearby. Re-save the checkpoint with the current mjai version."
        )
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(h, int) or isinstance(h, bool) or h <= 0 for h in raw)
    ):
        raise CheckpointLoadError(
            f"{d}: invalid hidden_sizes {raw!r} in checkpoint metadata "
            f"(expected a non-empty list of positive ints)."
        )
    return tuple(raw)


def _derive_activation(d: Path, meta: dict[str, Any] | None, config: dict[str, Any] | None) -> str:
    raw = _norm_str(meta.get("activation")) if meta is not None else None
    if raw is None and config is not None:
        raw = _norm_str(config.get("activation"))
    if raw is None:
        raise CheckpointLoadError(
            f"{d}: cannot determine the MLP activation — the sidecar predates "
            f"the 'activation' key and no run config.json records it. Weights "
            f"would load into the wrong nonlinearity silently, so this load is "
            f"refused (AGENTS.md §11); re-save the checkpoint with the current "
            f"mjai version."
        )
    return raw
