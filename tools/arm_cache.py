"""Content-addressed completion markers for probe arms.

Both probes (``tools/league_probe.py``, ``tools/theta_probe.py``) decide
whether to re-run an arm by looking at its output directory. That directory
name encodes only (game, mode/theta, seed), so every other knob — the step
budget, the eval cadence, the device, any ACH toggle — was invisible to the
cache: raising ``TOTAL_ENV_STEPS`` and re-running the notebook silently
"skipped" arms that had been trained at the OLD budget, and the resulting
figure mixed the two without saying so.

So the ``DONE`` marker records a fingerprint of the arm's resolved
:class:`ExperimentConfig` instead of just existing. :func:`status` compares
the fingerprint the caller is about to run against the one on disk and returns
``hit`` / ``stale`` / ``missing``, with the differing keys for ``stale`` so the
report names the knob that changed rather than saying "cache miss".

Volatile keys are excluded from the fingerprint (:data:`VOLATILE_KEYS`): they
name the output location or control console chatter and do not change what is
computed. Everything else counts, ``device`` included — a CPU result is not a
CUDA result, and the two are not bit-comparable.

Legacy ``DONE`` files (the old literal ``"ok\\n"``) carry no fingerprint. They
are not guessed at: the arm's own ``config.json``, which ``run_experiment``
writes at start, is hashed instead. Only when that is missing too does the arm
report ``legacy`` — known-finished, config unknown.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keys that name where the run goes or how loud it is, not what it computes.
VOLATILE_KEYS = ("out_dir", "verbose", "progress_bar")

DONE_NAME = "DONE"
RUN_CONFIG_NAME = "config.json"

# What to do with an arm whose config no longer matches the marker.
ON_STALE_CHOICES = ("error", "retrain", "skip")


@dataclass(frozen=True)
class ArmStatus:
    """Cache verdict for one arm.

    Attributes:
        state: ``missing`` (never finished) | ``hit`` (same config) |
            ``stale`` (finished under a different config) | ``legacy``
            (finished, but the marker predates fingerprinting and no
            ``config.json`` survives to compare against).
        fingerprint: the fingerprint of the config the caller wants to run.
        recorded: the fingerprint found on disk, if any.
        diff: ``(key, recorded_value, wanted_value)`` per differing key —
            empty unless ``state == "stale"``.
    """

    state: str
    fingerprint: str
    recorded: str | None = None
    diff: tuple[tuple[str, Any, Any], ...] = ()

    @property
    def is_done(self) -> bool:
        """True when the arm is finished and usable as-is."""
        return self.state in ("hit", "legacy")

    def describe(self) -> str:
        """One-line explanation, naming the knobs that changed when stale."""
        if self.state == "hit":
            return f"cached ({self.fingerprint})"
        if self.state == "legacy":
            return "cached (legacy marker — no config fingerprint to verify)"
        if self.state == "missing":
            return "not trained yet"
        changes = ", ".join(f"{k}: {old!r} -> {new!r}" for k, old, new in self.diff)
        return f"STALE ({self.recorded} -> {self.fingerprint}): {changes or 'config changed'}"


def config_digest(config: dict[str, Any]) -> str:
    """Stable 16-hex-char digest of a config dict, minus the volatile keys."""
    material = {k: v for k, v in sorted(config.items()) if k not in VOLATILE_KEYS}
    blob = json.dumps(material, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def fingerprint(cfg: Any) -> str:
    """Digest of an :class:`ExperimentConfig` (any dataclass instance)."""
    return config_digest(dataclasses.asdict(cfg))


def _read_marker(out_dir: Path) -> dict[str, Any] | None:
    """Parse the DONE marker; None when absent, ``{}`` for the legacy form."""
    marker = out_dir / DONE_NAME
    if not marker.is_file():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}  # legacy "ok\n"
    return data if isinstance(data, dict) else {}


def _recorded_config(out_dir: Path, marker: dict[str, Any]) -> dict[str, Any] | None:
    """The config the finished arm ran under: marker first, config.json second."""
    recorded = marker.get("config")
    if isinstance(recorded, dict):
        return recorded
    run_config = out_dir / RUN_CONFIG_NAME
    if run_config.is_file():
        try:
            data = json.loads(run_config.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if isinstance(data, dict):
            return data
    return None


def write_done(out_dir: Path, cfg: Any) -> str:
    """Mark ``out_dir`` finished, recording the config it ran under.

    Returns the fingerprint written.
    """
    digest = fingerprint(cfg)
    payload = {
        "fingerprint": digest,
        "finished_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "volatile_keys": list(VOLATILE_KEYS),
        "config": dataclasses.asdict(cfg),
    }
    (out_dir / DONE_NAME).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return digest


def status(out_dir: Path, cfg: Any) -> ArmStatus:
    """Classify ``out_dir`` against the config the caller intends to run."""
    want = fingerprint(cfg)
    marker = _read_marker(out_dir)
    if marker is None:
        return ArmStatus("missing", want)
    recorded_config = _recorded_config(out_dir, marker)
    if recorded_config is None:
        return ArmStatus("legacy", want)
    recorded = config_digest(recorded_config)
    if recorded == want:
        return ArmStatus("hit", want, recorded)
    return ArmStatus("stale", want, recorded, diff_configs(recorded_config, cfg))


def diff_configs(recorded: dict[str, Any], cfg: Any) -> tuple[tuple[str, Any, Any], ...]:
    """Differing non-volatile keys as ``(key, recorded_value, wanted_value)``."""
    wanted = dataclasses.asdict(cfg)
    keys = (set(recorded) | set(wanted)) - set(VOLATILE_KEYS)
    out = [(k, recorded.get(k), wanted.get(k)) for k in sorted(keys)]
    return tuple((k, a, b) for k, a, b in out if a != b)


def resolve(st: ArmStatus, on_stale: str, out_dir: Path) -> tuple[str, str]:
    """Turn a status + the ``ON_STALE`` policy into ``(action, message)``.

    ``action`` is ``train`` | ``skip`` | ``refuse``. Lives here rather than in
    the notebook so both notebook families share one decision table
    (AGENTS.md §7: notebooks import logic, they do not restate it).

    ``on_stale`` (the notebook's knob) applies only to the ``stale`` state:

      - ``error``   — refuse the arm and say which knob changed. The default:
        the alternatives either destroy a finished run or quietly mix two
        budgets into one figure.
      - ``retrain`` — wipe the arm directory and train it again. Destructive
        by necessity (a second TB event file in the same ``tb/`` interleaves
        two runs into one curve), which is why it is opt-in.
      - ``skip``    — reuse the mismatched result anyway. For "I only changed
        the device / I know what I am doing".
    """
    if on_stale not in ON_STALE_CHOICES:
        raise ValueError(f"bad on_stale {on_stale!r}; want {' | '.join(ON_STALE_CHOICES)}")
    if st.state == "missing":
        return "train", st.describe()
    if st.is_done:
        return "skip", st.describe()
    if on_stale == "skip":
        return "skip", f"{st.describe()}  [ON_STALE='skip': reusing anyway]"
    if on_stale == "retrain":
        clear_arm(out_dir)
        return "train", f"{st.describe()}  [ON_STALE='retrain': cleared and retraining]"
    return "refuse", (
        f"{st.describe()}\n"
        f"      set ON_STALE='retrain' to rebuild it, ON_STALE='skip' to reuse it "
        f"as-is, or delete {out_dir}"
    )


def clear_arm(out_dir: Path) -> None:
    """Delete a stale arm's directory so it can be retrained from scratch.

    Retraining in place is not an option: ``run_experiment`` opens a second
    TensorBoard event file in the same ``tb/`` directory and the reader then
    interleaves two runs' points into one curve.
    """
    import shutil

    if out_dir.exists():
        shutil.rmtree(out_dir)


__all__ = [
    "ON_STALE_CHOICES",
    "VOLATILE_KEYS",
    "ArmStatus",
    "clear_arm",
    "config_digest",
    "diff_configs",
    "fingerprint",
    "resolve",
    "status",
    "write_done",
]
