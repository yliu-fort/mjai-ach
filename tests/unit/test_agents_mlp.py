"""Unit tests for MLPSharedActorCritic (AGENTS.md §5).

These run on CPU (the dev env has torch+cpu while the cu128 wheel downloads
in the background). The constructor goes through gpu_assert, so we call
``require_cpu()`` first to opt into CPU — mirroring how the CLI's ``--cpu``
flag works in production.
"""

from __future__ import annotations

import math

import pytest
import torch

from mjai.agents.base import entropy_of_probs
from mjai.agents.mlp import MLPSharedActorCritic
from mjai.utils import gpu_assert

OBS = [0.5, -0.2, 0.0, 1.0]


@pytest.fixture(autouse=True)
def _cpu_mode():
    """Force CPU so tests pass without a CUDA device (AGENTS.md §1 D6)."""
    gpu_assert.reset_for_tests()
    gpu_assert.require_cpu()
    yield
    gpu_assert.reset_for_tests()


def test_construction_validates_args():
    with pytest.raises(ValueError, match="obs_size"):
        MLPSharedActorCritic(obs_size=0, num_actions=3)
    with pytest.raises(ValueError, match="num_actions"):
        MLPSharedActorCritic(obs_size=4, num_actions=0)


def test_lands_on_cpu_when_forced():
    m = MLPSharedActorCritic(obs_size=4, num_actions=3)
    assert m.device.type == "cpu"


def test_act_returns_legal_action_and_finite_logprob():
    m = MLPSharedActorCritic(obs_size=4, num_actions=5, seed=0)
    for _ in range(50):
        a, lp = m.act(OBS, legal_actions=[1, 3], eval=False)
        assert a in {1, 3}
        assert math.isfinite(lp)
        assert lp <= 0.0  # log-prob is non-positive


def test_eval_mode_is_deterministic():
    m = MLPSharedActorCritic(obs_size=4, num_actions=3, seed=1)
    m.eval_mode()
    actions = {m.act(OBS, [0, 1, 2], eval=True)[0] for _ in range(5)}
    assert len(actions) == 1  # greedy is deterministic


def test_eval_respects_legal_mask():
    m = MLPSharedActorCritic(obs_size=4, num_actions=5, seed=2)
    m.eval_mode()
    # Even if the global argmax is action 4, force it illegal.
    for _ in range(20):
        a, _ = m.act(OBS, legal_actions=[0, 1, 2], eval=True)
        assert a in {0, 1, 2}


def test_illegal_actions_never_sampled():
    m = MLPSharedActorCritic(obs_size=4, num_actions=6, seed=3)
    for _ in range(500):
        a, _ = m.act(OBS, legal_actions=[2, 4], eval=False)
        assert a in {2, 4}


def test_value_is_scalar_float():
    m = MLPSharedActorCritic(obs_size=4, num_actions=3, seed=0)
    v = m.value(OBS)
    assert isinstance(v, float)
    assert math.isfinite(v)


def test_action_logits_length_matches_legal():
    m = MLPSharedActorCritic(obs_size=4, num_actions=5, seed=0)
    lg = m.action_logits(OBS, legal_actions=[0, 2, 4])
    assert len(lg) == 3


def test_forward_batched_shapes():
    m = MLPSharedActorCritic(obs_size=4, num_actions=3, hidden_sizes=(16, 8), seed=0)
    batch = torch.randn(7, 4)
    logits, value = m(batch)
    assert logits.shape == (7, 3)
    assert value.shape == (7,)


def test_save_load_roundtrip_preserves_actions(tmp_path):
    m = MLPSharedActorCritic(obs_size=4, num_actions=3, hidden_sizes=(8,), seed=42)
    m.eval_mode()
    path = str(tmp_path / "m.pt")
    m.save(path)
    # Reload into a fresh net of the same shape.
    m2 = MLPSharedActorCritic(obs_size=4, num_actions=3, hidden_sizes=(8,), seed=0)
    m2.load(path)
    m2.eval_mode()
    # Same greedy action.
    a1, _ = m.act(OBS, [0, 1, 2], eval=True)
    a2, _ = m2.act(OBS, [0, 1, 2], eval=True)
    assert a1 == a2
    # Same value.
    assert math.isclose(m.value(OBS), m2.value(OBS), abs_tol=1e-5)


def test_gradients_flow_to_all_components():
    """A single SGD step moves both policy and value params."""
    m = MLPSharedActorCritic(obs_size=4, num_actions=3, hidden_sizes=(8,), seed=0)
    opt = torch.optim.SGD(m.parameters(), lr=1e-2)
    obs_t = torch.as_tensor(OBS).unsqueeze(0)
    before_policy = m.policy_head.weight.detach().clone()
    before_value = m.value_head.weight.detach().clone()
    logits, value = m(obs_t)
    legal = torch.tensor([True, True, True])
    masked = logits + torch.where(legal, 0.0, -1e9)
    loss = -torch.log_softmax(masked, dim=-1)[0, 0] + value.pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert not torch.allclose(before_policy, m.policy_head.weight)
    assert not torch.allclose(before_value, m.value_head.weight)


def test_policy_entropy_near_uniform_at_init():
    """Freshly init net ~ uniform => high entropy over a 3-action legal set."""
    m = MLPSharedActorCritic(obs_size=4, num_actions=3, seed=0)
    # Average the entropy over many stochastic obs.
    ents = []
    for _ in range(20):
        obs = torch.randn(4).tolist()
        lg = m.action_logits(obs, [0, 1, 2])
        # softmax + entropy in nats.
        mx = max(lg)
        exps = [math.exp(x - mx) for x in lg]
        s = sum(exps)
        probs = [e / s for e in exps]
        ents.append(entropy_of_probs(probs))
    # Uniform over 3 actions = log(3) ~= 1.099 nats. Allow some slack from init.
    assert max(ents) > 0.5
