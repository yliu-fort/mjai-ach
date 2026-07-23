"""Unit tests for the checkpoint -> Policy factory (AGENTS.md §5, F1).

Covers: metadata-driven reconstruction of a non-default MLP (architecture +
parameters survive the round trip), tabular checkpoints, legacy checkpoints
whose sidecar lacks hidden_sizes/activation (derived from the run's dumped
config.json), and loud CheckpointLoadError failures for anything missing or
inconsistent — never a silent default and never a raw torch traceback.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from mjai.agents.ckpt_io import CheckpointManifest, write_checkpoint
from mjai.agents.mlp import MLPSharedActorCritic
from mjai.agents.policy_factory import CheckpointLoadError, load_policy_from_checkpoint
from mjai.agents.tabular import TabularPolicy
from mjai.utils import gpu_assert

OBS_SIZE = 5
NUM_ACTIONS = 3


@pytest.fixture(autouse=True)
def _cpu_mode():
    """Force CPU so tests pass without a CUDA device (AGENTS.md §1 D6)."""
    gpu_assert.reset_for_tests()
    gpu_assert.require_cpu()
    yield
    gpu_assert.reset_for_tests()


def _manifest(kind: str = "mlp", game: str = "kuhn", step: int = 1) -> CheckpointManifest:
    return CheckpointManifest(
        game=game,
        game_string="kuhn_poker",
        algo="ach",
        self_play_mode="mirror",
        policy_kind=kind,
        num_actions=NUM_ACTIONS,
        obs_kind="information_state",
        obs_size=OBS_SIZE,
        train_step=step,
    )


def _write_mlp_ckpt(
    directory: Path,
    *,
    hidden: tuple[int, ...] = (32,),
    activation: type[nn.Module] = nn.ReLU,
    with_sidecar: bool = True,
    trunk_layernorm: bool = True,
) -> MLPSharedActorCritic:
    """Write a manifest + MLP weights; returns the policy that was saved."""
    write_checkpoint(directory, _manifest())
    pol = MLPSharedActorCritic(
        obs_size=OBS_SIZE,
        num_actions=NUM_ACTIONS,
        hidden_sizes=hidden,
        activation=activation,
        trunk_layernorm=trunk_layernorm,
        seed=0,
    )
    if with_sidecar:
        pol.save(str(directory / "policy.pt"))
    else:  # legacy checkpoint: weights only, no policy.pt.meta.json
        torch.save(pol.state_dict(), directory / "policy.pt")
    return pol


def _assert_same_policy(a: MLPSharedActorCritic, b: MLPSharedActorCritic) -> None:
    sd_a, sd_b = a.state_dict(), b.state_dict()
    assert list(sd_a) == list(sd_b)
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), key


def test_mlp_roundtrip_preserves_nondefault_architecture(tmp_path):
    """F1: a (32,)+ReLU checkpoint rebuilds as (32,)+ReLU, not the (128,128)+Tanh default."""
    saved = _write_mlp_ckpt(tmp_path / "ckpt", hidden=(32,), activation=nn.ReLU)
    loaded = load_policy_from_checkpoint(tmp_path / "ckpt")
    assert isinstance(loaded, MLPSharedActorCritic)
    widths = [m.out_features for m in loaded.torso if isinstance(m, nn.Linear)]
    assert widths == [32]
    assert isinstance(loaded.torso[1], nn.ReLU)
    _assert_same_policy(saved, loaded)
    # Same forward behavior on a fixed observation.
    obs = [0.1] * OBS_SIZE
    assert loaded.action_logits(obs, [0, 1, 2]) == saved.action_logits(obs, [0, 1, 2])


def test_sidecar_records_activation(tmp_path):
    _write_mlp_ckpt(tmp_path / "ckpt", activation=nn.Tanh)
    meta = json.loads((tmp_path / "ckpt" / "policy.pt.meta.json").read_text(encoding="utf-8"))
    assert meta["activation"] == "tanh"
    assert meta["hidden_sizes"] == [32]


@pytest.mark.parametrize("layernorm", [True, False])
def test_trunk_layernorm_round_trips_via_sidecar(tmp_path, layernorm):
    """Either torso variant reloads exactly — the sidecar records which it was."""
    ckpt = tmp_path / f"ckpt_{layernorm}"
    saved = _write_mlp_ckpt(ckpt, trunk_layernorm=layernorm)
    meta = json.loads((ckpt / "policy.pt.meta.json").read_text(encoding="utf-8"))
    assert meta["trunk_layernorm"] is layernorm
    loaded = load_policy_from_checkpoint(ckpt)
    assert isinstance(loaded, MLPSharedActorCritic)
    assert loaded.trunk_layernorm is layernorm
    _assert_same_policy(saved, loaded)


def test_tabular_roundtrip(tmp_path):
    write_checkpoint(tmp_path / "ckpt", _manifest(kind="tabular"))
    pol = TabularPolicy(num_actions=NUM_ACTIONS, seed=0)
    row = pol.get_logits([0.0] * OBS_SIZE)
    row[1] = 2.5
    pol.save(str(tmp_path / "ckpt" / "policy.json"))
    loaded = load_policy_from_checkpoint(tmp_path / "ckpt")
    assert isinstance(loaded, TabularPolicy)
    assert loaded.action_logits([0.0] * OBS_SIZE, [0, 1, 2]) == [0.0, 2.5, 0.0]


def test_legacy_checkpoint_derives_arch_from_run_config(tmp_path):
    """Old sidecar-less ckpt: hidden_sizes + activation come from the run's config.json.

    A genuinely legacy checkpoint also predates the trunk LayerNorm, so it is
    saved without one; the factory's fallback to ``trunk_layernorm=False`` is
    what makes such a checkpoint still loadable under the new default.
    """
    ckpt = tmp_path / "run" / "checkpoints" / "step_1"
    saved = _write_mlp_ckpt(
        ckpt, hidden=(32,), activation=nn.ReLU, with_sidecar=False, trunk_layernorm=False
    )
    (tmp_path / "run" / "config.json").write_text(
        json.dumps(
            {
                "policy_kind": "mlp",
                "game": "kuhn",
                "hidden_sizes": [32],
                "activation": "relu",
            }
        ),
        encoding="utf-8",
    )
    loaded = load_policy_from_checkpoint(ckpt)
    assert isinstance(loaded, MLPSharedActorCritic)
    assert isinstance(loaded.torso[1], nn.ReLU)
    _assert_same_policy(saved, loaded)


def test_missing_sidecar_without_config_fails_loudly(tmp_path):
    ckpt = tmp_path / "orphan" / "checkpoints" / "step_1"
    _write_mlp_ckpt(ckpt, with_sidecar=False)
    with pytest.raises(CheckpointLoadError, match="hidden sizes"):
        load_policy_from_checkpoint(ckpt)


def test_sidecar_without_activation_and_no_config_fails_loudly(tmp_path):
    """A sidecar that predates the 'activation' key is not silently defaulted."""
    ckpt = tmp_path / "ckpt"
    _write_mlp_ckpt(ckpt, activation=nn.ReLU)
    meta_path = ckpt / "policy.pt.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["activation"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(CheckpointLoadError, match="activation"):
        load_policy_from_checkpoint(ckpt)


def test_conflicting_provenance_fails_loudly(tmp_path):
    ckpt = tmp_path / "run" / "checkpoints" / "step_1"
    _write_mlp_ckpt(ckpt, hidden=(32,))  # sidecar says [32]
    (tmp_path / "run" / "config.json").write_text(
        json.dumps({"policy_kind": "mlp", "game": "kuhn", "hidden_sizes": [64]}),
        encoding="utf-8",
    )
    with pytest.raises(CheckpointLoadError, match="conflicting provenance"):
        load_policy_from_checkpoint(ckpt)


def test_run_config_for_a_different_game_fails_loudly(tmp_path):
    ckpt = tmp_path / "run" / "checkpoints" / "step_1"
    _write_mlp_ckpt(ckpt, with_sidecar=False)
    (tmp_path / "run" / "config.json").write_text(
        json.dumps(
            {"policy_kind": "mlp", "game": "leduc", "hidden_sizes": [32], "activation": "relu"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(CheckpointLoadError, match="conflicting provenance"):
        load_policy_from_checkpoint(ckpt)


def test_missing_manifest_fails_loudly(tmp_path):
    with pytest.raises(CheckpointLoadError, match="manifest"):
        load_policy_from_checkpoint(tmp_path / "nope")


def test_unknown_policy_kind_fails_loudly(tmp_path):
    write_checkpoint(tmp_path / "ckpt", _manifest(kind="lstm"))
    with pytest.raises(CheckpointLoadError, match="policy_kind"):
        load_policy_from_checkpoint(tmp_path / "ckpt")


def test_missing_weight_file_fails_loudly(tmp_path):
    write_checkpoint(tmp_path / "ckpt", _manifest())
    (tmp_path / "ckpt" / "policy.pt.meta.json").write_text(
        json.dumps(
            {
                "obs_size": OBS_SIZE,
                "num_actions": NUM_ACTIONS,
                "hidden_sizes": [32],
                "activation": "relu",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CheckpointLoadError, match="weight file"):
        load_policy_from_checkpoint(tmp_path / "ckpt")


def test_mismatched_weights_wrapped_not_raw_torch_error(tmp_path):
    """Weights from a (16,) net into a (32,) sidecar must surface as CheckpointLoadError."""
    ckpt = tmp_path / "ckpt"
    _write_mlp_ckpt(ckpt, hidden=(32,))
    other = MLPSharedActorCritic(
        obs_size=OBS_SIZE, num_actions=NUM_ACTIONS, hidden_sizes=(16,), seed=1
    )
    torch.save(other.state_dict(), ckpt / "policy.pt")
    with pytest.raises(CheckpointLoadError, match="do not fit"):
        load_policy_from_checkpoint(ckpt)
