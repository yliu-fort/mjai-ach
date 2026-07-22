"""Tests for the mjai-train entry point (F6); no training is ever run."""

from __future__ import annotations

import pytest

from mjai.scripts import train


def test_bare_defaults_warning_names_divergent_fields_and_remedy():
    msg = train._bare_defaults_warning()
    assert "NOT the ACH paper reproduction" in msg
    for field in ("policy_kind", "learning_rate", "entropy_coef"):
        assert field in msg
    assert "--config" in msg
    assert "reproduce_paper" in msg


def test_main_requires_config_or_full_triple():
    with pytest.raises(SystemExit):
        train.main(["--game", "kuhn"])  # missing --algo/--mode


def test_main_warns_on_bare_defaults(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(train, "run_experiment", lambda cfg: cfg.out_dir)
    rc = train.main(
        [
            "--game",
            "kuhn",
            "--algo",
            "ach",
            "--mode",
            "mirror",
            "--cpu",
            "--out",
            str(tmp_path / "run"),
        ]
    )
    assert rc == 0
    assert "*** WARNING: no --config given" in capsys.readouterr().err


def test_main_with_config_does_not_warn(monkeypatch, capsys, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "game: kuhn\nalgo: ach\nself_play_mode: mirror\nout_dir: x\n", encoding="utf-8"
    )
    monkeypatch.setattr(train, "run_experiment", lambda cfg: cfg.out_dir)
    rc = train.main(["--config", str(cfg_path)])
    assert rc == 0
    assert "WARNING" not in capsys.readouterr().err
