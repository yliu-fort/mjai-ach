"""Unit tests for the GPU-assertion contract (AGENTS.md §1 D6).

The contract is "no silent degradation": if GPU is unavailable and the caller
did not opt into CPU, we raise. These tests cover all four branches without
requiring an actual GPU.
"""

from __future__ import annotations

import pytest

from mjai.utils import gpu_assert
from mjai.utils.gpu_assert import GPUUnavailableError


@pytest.fixture(autouse=True)
def _reset_state():
    """Each test starts from a clean module state."""
    gpu_assert.reset_for_tests()
    yield
    gpu_assert.reset_for_tests()


def test_require_cpu_then_resolve_returns_cpu(monkeypatch):
    """Programmatic require_cpu() forces CPU without touching env."""
    gpu_assert.require_cpu()
    info = gpu_assert.resolve_device()
    assert info.device == "cpu"
    assert not info.is_cuda
    assert "requested" in info.reason


def test_env_var_cpu(monkeypatch):
    """MJAI_CPU=1 opts into CPU."""
    monkeypatch.setenv(gpu_assert.CPU_ENV_VAR, "1")
    info = gpu_assert.resolve_device()
    assert info.device == "cpu"


def test_no_gpu_no_opt_in_raises(monkeypatch):
    """GPU unavailable + no opt-in => loud failure, never silent CPU."""
    monkeypatch.delenv(gpu_assert.CPU_ENV_VAR, raising=False)
    monkeypatch.setattr(gpu_assert, "_cuda_available", lambda: False)
    with pytest.raises(GPUUnavailableError, match="GPU"):
        gpu_assert.resolve_device()


def test_gpu_available_returns_cuda(monkeypatch):
    """GPU available + no opt-in => cuda device."""
    monkeypatch.delenv(gpu_assert.CPU_ENV_VAR, raising=False)
    monkeypatch.setattr(gpu_assert, "_cuda_available", lambda: True)
    info = gpu_assert.resolve_device()
    assert info.is_cuda
    assert info.device == "cuda"


def test_preferred_index(monkeypatch):
    """preferred_index selects cuda:<n>."""
    monkeypatch.delenv(gpu_assert.CPU_ENV_VAR, raising=False)
    monkeypatch.setattr(gpu_assert, "_cuda_available", lambda: True)
    info = gpu_assert.resolve_device(preferred_index=2)
    assert info.device == "cuda:2"


def test_cpu_request_overrides_available_gpu(monkeypatch):
    """Even with GPU available, explicit --cpu wins."""
    monkeypatch.setattr(gpu_assert, "_cuda_available", lambda: True)
    gpu_assert.require_cpu()
    info = gpu_assert.resolve_device()
    assert info.device == "cpu"


def test_env_var_truthy_variants(monkeypatch):
    """Env var accepts common truthy spellings."""
    monkeypatch.setattr(gpu_assert, "_cuda_available", lambda: False)
    for val in ("1", "true", "TRUE", "yes"):
        monkeypatch.setenv(gpu_assert.CPU_ENV_VAR, val)
        assert gpu_assert.resolve_device().device == "cpu", f"failed for {val!r}"
