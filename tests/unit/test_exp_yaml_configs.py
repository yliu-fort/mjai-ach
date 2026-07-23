"""Every configs/exp/*.yaml must parse into ExperimentConfig (AGENTS.md §5/§9).

Guards against YAML typos and schema drift across the whole experiment matrix
(including the league A/B arms); unknown keys raise TypeError loudly via the
frozen dataclass constructor.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mjai.scripts.experiment import ExperimentConfig, resolve_theta

EXP_DIR = Path(__file__).resolve().parents[2] / "configs" / "exp"
YAML_FILES = sorted(EXP_DIR.glob("*.yaml"))


def test_exp_dir_found_and_nonempty():
    """Guard the guard: a wrong EXP_DIR must fail, not vacuously pass."""
    assert EXP_DIR.is_dir()
    assert len(YAML_FILES) >= 30  # the Phase-1 matrix + reproduction arms


@pytest.mark.parametrize("path", YAML_FILES, ids=[p.name for p in YAML_FILES])
def test_exp_yaml_parses_into_experiment_config(path: Path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path.name}: top-level mapping expected"
    cfg = ExperimentConfig(**data)
    assert cfg.game, f"{path.name}: empty game"
    assert cfg.algo in ("ppo", "ach", "theta"), f"{path.name}: bad algo {cfg.algo!r}"
    # algo/theta must not contradict each other (mjai.scripts.experiment_build).
    resolve_theta(cfg)
    assert cfg.self_play_mode in (
        "mirror",
        "league",
    ), f"{path.name}: bad self_play_mode {cfg.self_play_mode!r}"


def test_league_yamls_carry_explicit_league_knobs():
    """The MLP league arms list every league knob explicitly (sweep-ready)."""
    names = sorted(p.name for p in EXP_DIR.glob("*_mlp_league.yaml"))
    assert names, "no *_mlp_league.yaml found — the A/B arms are missing"
    for name in names:
        data = yaml.safe_load((EXP_DIR / name).read_text(encoding="utf-8"))
        for knob in (
            "league_capacity",
            "league_mix_current_main",
            "league_mix_history",
            "league_mix_exploiter",
            "league_main_exploiter_promo",
            "league_league_exploiter_promo",
            "league_exploiter_share",
            "league_promo_window",
            "league_role_weight_main",
            "league_role_weight_main_exploiter",
            "league_role_weight_league_exploiter",
            "league_reset_mode",
            "league_main_save_every_rounds",
        ):
            assert knob in data, f"{name}: missing league knob {knob!r}"
        assert data["self_play_mode"] == "league"


def test_mlp_ppo_arms_mirror_their_ach_counterparts():
    """The PPO MLP arms must differ from the ACH ones by `algo` + `out_dir` only.

    That is what makes them a control for the theta scan: same scaffolding,
    same budget, same architecture — only the policy term changes
    (notebooks/theta_*.ipynb, docs/audit_report.md B10 addendum).
    """
    ppo_files = sorted(EXP_DIR.glob("*_ppo_mlp_mirror.yaml"))
    assert ppo_files, "no *_ppo_mlp_mirror.yaml found — the theta=0 controls are missing"
    for ppo_path in ppo_files:
        ach_path = EXP_DIR / ppo_path.name.replace("_ppo_mlp_", "_ach_mlp_")
        assert ach_path.is_file(), f"{ppo_path.name}: no ACH counterpart"
        ppo = yaml.safe_load(ppo_path.read_text(encoding="utf-8"))
        ach = yaml.safe_load(ach_path.read_text(encoding="utf-8"))
        differing = {k for k in set(ppo) | set(ach) if ppo.get(k) != ach.get(k)}
        assert differing == {"algo", "out_dir"}, f"{ppo_path.name}: also differs in {differing}"
        assert ppo["algo"] == "ppo" and ach["algo"] == "ach"
