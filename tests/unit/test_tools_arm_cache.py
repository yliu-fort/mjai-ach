"""Unit tests for tools/arm_cache.py — content-addressed arm completion markers."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import arm_cache  # noqa: E402

from mjai.scripts.experiment import ExperimentConfig  # noqa: E402


def _cfg(**overrides: object) -> ExperimentConfig:
    base = ExperimentConfig(
        game="kuhn",
        algo="ach",
        self_play_mode="mirror",
        policy_kind="mlp",
        out_dir="runs/nb_ab/kuhn_mirror/seed_0",
        total_env_steps=60_000,
        eval_every_env_steps=5_000,
        device="cpu",
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


def test_volatile_keys_do_not_change_the_fingerprint():
    """Where the run goes and how loud it is are not part of what it computes."""
    base = _cfg()
    for key, value in (("out_dir", "runs/elsewhere"), ("verbose", True), ("progress_bar", True)):
        assert arm_cache.fingerprint(dataclasses.replace(base, **{key: value})) == (
            arm_cache.fingerprint(base)
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("total_env_steps", 10_000),
        ("eval_every_env_steps", 1_000),
        ("seed", 3),
        ("theta", 0.5),
        ("device", "cuda"),
        ("learning_rate", 1e-2),
        ("probe_term_grad_norms", True),
    ],
)
def test_substantive_keys_change_the_fingerprint(key: str, value: object):
    """device included: a CPU result is not a CUDA result (documented choice)."""
    base = _cfg(algo="theta", theta=1.0)
    assert arm_cache.fingerprint(dataclasses.replace(base, **{key: value})) != (
        arm_cache.fingerprint(base)
    )


def test_missing_then_hit_roundtrip(tmp_path: Path):
    cfg = _cfg()
    assert arm_cache.status(tmp_path, cfg).state == "missing"
    assert not arm_cache.status(tmp_path, cfg).is_done
    digest = arm_cache.write_done(tmp_path, cfg)
    got = arm_cache.status(tmp_path, cfg)
    assert got.state == "hit" and got.is_done and got.fingerprint == digest
    assert "cached" in got.describe()


def test_stale_names_the_knob_that_changed(tmp_path: Path):
    """The whole point: report the differing key, not just 'cache miss'."""
    arm_cache.write_done(tmp_path, _cfg(total_env_steps=10_000))
    got = arm_cache.status(tmp_path, _cfg(total_env_steps=60_000))
    assert got.state == "stale" and not got.is_done
    assert got.diff == (("total_env_steps", 10_000, 60_000),)
    assert "total_env_steps: 10000 -> 60000" in got.describe()


def test_stale_is_not_triggered_by_a_volatile_key_alone(tmp_path: Path):
    arm_cache.write_done(tmp_path, _cfg(progress_bar=False, out_dir="a"))
    assert arm_cache.status(tmp_path, _cfg(progress_bar=True, out_dir="b")).state == "hit"


def test_legacy_marker_falls_back_to_the_runs_own_config_json(tmp_path: Path):
    """Old arms keep their cache: run_experiment already dumped config.json."""
    cfg = _cfg()
    (tmp_path / "DONE").write_text("ok\n", encoding="utf-8")
    (tmp_path / "config.json").write_text(
        json.dumps(dataclasses.asdict(cfg)), encoding="utf-8"
    )
    assert arm_cache.status(tmp_path, cfg).state == "hit"
    # ...and a legacy arm trained at a different budget is still caught.
    stale = arm_cache.status(tmp_path, _cfg(total_env_steps=1))
    assert stale.state == "stale"
    assert stale.diff == (("total_env_steps", 60_000, 1),)


def test_legacy_marker_without_config_json_reports_legacy_not_hit(tmp_path: Path):
    """Unverifiable is its own state — never silently claimed as a match."""
    (tmp_path / "DONE").write_text("ok\n", encoding="utf-8")
    got = arm_cache.status(tmp_path, _cfg())
    assert got.state == "legacy"
    assert got.is_done  # finished, so usable
    assert "legacy" in got.describe()


def test_corrupt_marker_is_treated_as_legacy_not_as_a_crash(tmp_path: Path):
    (tmp_path / "DONE").write_text("{not json", encoding="utf-8")
    assert arm_cache.status(tmp_path, _cfg()).state == "legacy"


def test_written_marker_is_readable_json_with_the_config(tmp_path: Path):
    cfg = _cfg()
    arm_cache.write_done(tmp_path, cfg)
    data = json.loads((tmp_path / "DONE").read_text(encoding="utf-8"))
    assert data["fingerprint"] == arm_cache.fingerprint(cfg)
    assert data["config"]["total_env_steps"] == 60_000
    assert data["volatile_keys"] == list(arm_cache.VOLATILE_KEYS)
    assert data["finished_at"]


def test_clear_arm_removes_the_directory(tmp_path: Path):
    arm = tmp_path / "seed_0"
    (arm / "tb").mkdir(parents=True)
    (arm / "tb" / "events.out.tfevents.1").write_text("x", encoding="utf-8")
    arm_cache.clear_arm(arm)
    assert not arm.exists()
    arm_cache.clear_arm(arm)  # idempotent


def test_on_stale_choices_are_the_documented_three():
    assert arm_cache.ON_STALE_CHOICES == ("error", "retrain", "skip")


def test_resolve_trains_when_missing_and_skips_when_done(tmp_path: Path):
    cfg = _cfg()
    assert arm_cache.resolve(arm_cache.status(tmp_path, cfg), "error", tmp_path)[0] == "train"
    arm_cache.write_done(tmp_path, cfg)
    assert arm_cache.resolve(arm_cache.status(tmp_path, cfg), "error", tmp_path)[0] == "skip"


def _stale(tmp_path: Path) -> arm_cache.ArmStatus:
    arm = tmp_path / "seed_0"
    arm.mkdir(parents=True, exist_ok=True)
    arm_cache.write_done(arm, _cfg(total_env_steps=10_000))
    return arm_cache.status(arm, _cfg(total_env_steps=60_000))


def test_resolve_stale_error_refuses_without_deleting_anything(tmp_path: Path):
    """The default must never destroy a finished run."""
    st = _stale(tmp_path)
    action, msg = arm_cache.resolve(st, "error", tmp_path / "seed_0")
    assert action == "refuse"
    assert (tmp_path / "seed_0" / "DONE").is_file()
    assert "total_env_steps" in msg and "ON_STALE='retrain'" in msg


def test_resolve_stale_retrain_clears_the_arm(tmp_path: Path):
    st = _stale(tmp_path)
    action, msg = arm_cache.resolve(st, "retrain", tmp_path / "seed_0")
    assert action == "train"
    assert not (tmp_path / "seed_0").exists()  # cleared: no mixed TB event files
    assert "retrain" in msg


def test_resolve_stale_skip_reuses_and_says_so(tmp_path: Path):
    st = _stale(tmp_path)
    action, msg = arm_cache.resolve(st, "skip", tmp_path / "seed_0")
    assert action == "skip"
    assert (tmp_path / "seed_0" / "DONE").is_file()
    assert "reusing anyway" in msg


def test_resolve_rejects_an_unknown_policy(tmp_path: Path):
    with pytest.raises(ValueError, match="bad on_stale"):
        arm_cache.resolve(arm_cache.status(tmp_path, _cfg()), "yolo", tmp_path)


def test_probes_write_and_read_the_same_marker(tmp_path: Path):
    """Both probes must agree on the marker format (theta_probe imports it)."""
    import league_probe
    import theta_probe

    kwargs = {
        "total_env_steps": 60_000,
        "eval_every_env_steps": 5_000,
        "root": tmp_path,
        "device": "cpu",
    }
    league_cfg = league_probe.arm_config("kuhn", "mirror", 0, **kwargs)
    assert league_probe.arm_status("kuhn", "mirror", 0, **kwargs).state == "missing"
    Path(league_cfg.out_dir).mkdir(parents=True)
    arm_cache.write_done(Path(league_cfg.out_dir), league_cfg)
    assert league_probe.arm_status("kuhn", "mirror", 0, **kwargs).state == "hit"
    assert (
        league_probe.arm_status("kuhn", "mirror", 0, **{**kwargs, "total_env_steps": 1}).state
        == "stale"
    )

    theta_cfg = theta_probe.arm_config("kuhn", 0.5, 0, **kwargs)
    assert theta_cfg.algo == "theta" and theta_cfg.theta == 0.5
    Path(theta_cfg.out_dir).mkdir(parents=True)
    arm_cache.write_done(Path(theta_cfg.out_dir), theta_cfg)
    assert theta_probe.arm_status("kuhn", 0.5, 0, **kwargs).state == "hit"
    assert theta_probe.arm_status("kuhn", 0.75, 0, **kwargs).state == "missing"
