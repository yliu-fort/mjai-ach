"""GPU device selection with no silent degradation (AGENTS.md §1 D6).

Training defaults to CUDA. If CUDA is unavailable we **fail loudly** unless the
caller explicitly opts into CPU via:
  - the ``--cpu`` CLI flag (handled by the caller, which calls :func:`require_cpu`),
  - the ``MJAI_CPU=1`` environment variable, or
  - calling :func:`resolve_device` after :func:`require_cpu`.

This module never silently falls back. A run that lands on CPU when the user
expected GPU is always either (a) explicitly requested or (b) a hard error.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

# Environment variable that forces CPU. Checked lazily inside resolve_device so
# tests can monkeypatch it.
CPU_ENV_VAR = "MJAI_CPU"

# Module-level flag, flipped by require_cpu() / CLI --cpu parsing. Kept separate
# from the env var so a caller can force CPU programmatically without touching env.
_cpu_forced: bool = False


class GPUUnavailableError(RuntimeError):
    """Raised when GPU is required but unavailable and CPU was not opted into."""


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved device + how we got there, for logging."""

    device: str  # "cuda" | "cuda:<i>" | "cpu"
    is_cuda: bool
    reason: str  # human-readable provenance for the SummaryWriter / logs


def require_cpu() -> None:
    """Programmatically force CPU mode (e.g. CLI ``--cpu`` flag was passed).

    Idempotent. Does not read or write the environment; pair with the ``MJAI_CPU``
    env var if you need both. After this call, :func:`resolve_device` returns CPU
    without raising.
    """
    # Single programmatic toggle; a class would be over-engineering.
    global _cpu_forced  # noqa: PLW0603
    _cpu_forced = True


def _is_cpu_requested() -> bool:
    """True if CPU was explicitly requested via flag or env."""
    if _cpu_forced:
        return True
    return os.environ.get(CPU_ENV_VAR, "").lower() in ("1", "true", "yes")


@lru_cache(maxsize=1)
def _cuda_available() -> bool:
    """Wrapped so tests can monkeypatch without importing torch at module load.

    Importing torch here (not at module top) keeps ``import mjai.utils`` cheap
    when only the CPU helpers are needed.
    """
    try:
        import torch  # lazy on purpose so `import mjai.utils` is cheap without torch.

        return bool(torch.cuda.is_available())
    except ImportError as e:  # pragma: no cover - torch is a hard dep in practice
        raise RuntimeError(
            "PyTorch is not installed. Install with `uv sync` (see pyproject.toml)."
        ) from e


def resolve_device(preferred_index: int | None = None) -> DeviceInfo:
    """Resolve the torch device to use, failing loudly on silent-degradation risk.

    Args:
        preferred_index: if set and CUDA is available, use ``cuda:<index>``;
            otherwise ``cuda`` (default device).

    Returns:
        DeviceInfo describing the chosen device and why.

    Raises:
        GPUUnavailableError: if CPU was NOT explicitly requested but CUDA is
            unavailable. This is the "no silent degradation" guarantee.
    """
    if _is_cpu_requested():
        return DeviceInfo(device="cpu", is_cuda=False, reason="cpu explicitly requested")

    if not _cuda_available():
        raise GPUUnavailableError(
            "GPU (CUDA) is unavailable. Training defaults to GPU and will not "
            "silently fall back to CPU. To run on CPU, pass --cpu or set "
            f"{CPU_ENV_VAR}=1."
        )

    device = "cuda" if preferred_index is None else f"cuda:{preferred_index}"
    return DeviceInfo(
        device=device,
        is_cuda=True,
        reason=f"cuda available; using {device}",
    )


def reset_for_tests() -> None:
    """Clear the CPU-forced flag and the cuda-availability cache.

    Test-only. Production code should never call this.
    """
    # Test-only reset of the same toggle used by require_cpu().
    global _cpu_forced  # noqa: PLW0603
    _cpu_forced = False
    _cuda_available.cache_clear()
