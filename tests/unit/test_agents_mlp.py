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


def test_act_with_value_matches_separate_calls_in_eval():
    """Fused ``act_with_value`` (one forward) must equal ``act`` + ``value`` (two forwards).

    Eval mode is deterministic (greedy argmax, no RNG draw), so the fused and
    separate paths are comparable bit-for-bit on the same net. This is the
    correctness contract for the rollout hot-path optimization (AGENTS.md §8):
    the single-forward override may not drift from the canonical act+value.
    """
    m = MLPSharedActorCritic(obs_size=4, num_actions=5, seed=7)
    m.eval_mode()
    legal = [0, 2, 4]
    # Run many observations to cover weight space reasonably.
    for _ in range(20):
        obs = torch.randn(4).tolist()
        a_sep, lp_sep = m.act(obs, legal, eval=True)
        v_sep = m.value(obs)
        a_fused, lp_fused, v_fused = m.act_with_value(obs, legal, eval=True)
        assert a_sep == a_fused
        assert lp_sep == lp_fused  # both 0.0 in eval mode
        assert math.isclose(v_sep, v_fused, abs_tol=1e-6)


def test_snapshot_restore_roundtrip_and_independence():
    """snapshot_state -> restore_state reproduces the policy exactly and is
    independent of later mutations to the source.

    Also verifies the GPU-memory discipline (AGENTS.md §8): snapshot tensors
    live on CPU regardless of the policy's device, so long-lived stores (hub
    history, league pool) never pin GPU memory.
    """
    m = MLPSharedActorCritic(obs_size=4, num_actions=3, hidden_sizes=(8,), seed=11)
    m.eval_mode()
    snap = m.snapshot_state()
    assert snap["kind"] == "nn"
    # Snapshot must hold CPU tensors — the whole point of moving off GPU.
    for v in snap["state_dict"].values():
        assert v.device.type == "cpu"
    # Reload into a fresh net of matching shape and verify identical outputs.
    m2 = MLPSharedActorCritic(obs_size=4, num_actions=3, hidden_sizes=(8,), seed=0)
    m2.restore_state(snap)
    m2.eval_mode()
    for _ in range(10):
        obs = torch.randn(4).tolist()
        assert m.act(obs, [0, 1, 2], eval=True) == m2.act(obs, [0, 1, 2], eval=True)
        assert math.isclose(m.value(obs), m2.value(obs), abs_tol=1e-6)
    # Independence: mutate m via a gradient step; m2 and the snapshot must not move.
    opt = torch.optim.SGD(m.parameters(), lr=1.0)
    obs_t = torch.as_tensor([0.1, 0.2, 0.3, 0.4]).unsqueeze(0)
    logits, value = m(obs_t)
    loss = logits.sum() + value.sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    probe = [0.1, 0.2, 0.3, 0.4]
    a_after, _ = m.act(probe, [0, 1, 2], eval=True)
    a2_after, _ = m2.act(probe, [0, 1, 2], eval=True)
    # m moved (greedy action may or may not change), but m2 must equal what m
    # was BEFORE the step — i.e. m2 must not equal the post-step m on a probe
    # where the step actually shifted the argmax. We assert the strong form:
    # m2's output equals a fresh restore from the (unchanged) snapshot.
    m3 = MLPSharedActorCritic(obs_size=4, num_actions=3, hidden_sizes=(8,), seed=0)
    m3.restore_state(snap)
    m3.eval_mode()
    assert m2.act(probe, [0, 1, 2], eval=True) == m3.act(probe, [0, 1, 2], eval=True)
    assert math.isclose(m2.value(probe), m3.value(probe), abs_tol=1e-6)
    # Silence unused-var lint for a_after/a2_after (kept for debug readability).
    _ = a_after, a2_after


def test_action_logits_batch_matches_per_state_calls():
    """The batched read must answer exactly what the one-state API answers.

    Exact eval materializes the policy through this method, so a divergence
    here would silently change every equilibrium number.
    """
    import numpy as np

    gpu_assert.reset_for_tests()
    gpu_assert.require_cpu()
    try:
        policy = MLPSharedActorCritic(obs_size=4, num_actions=5, hidden_sizes=(16,), seed=0)
        obs_batch = np.array(
            [[0.1, -0.2, 0.3, 0.4], [1.0, 0.0, -1.0, 0.5], [0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        legal_mask = np.array(
            [[True] * 5, [True, False, True, False, True], [False, True, True, True, False]]
        )
        batched = policy.action_logits_batch(obs_batch, legal_mask)
        assert batched.shape == (3, 5)
        for i, row in enumerate(legal_mask):
            legal = [a for a, ok in enumerate(row) if ok]
            one_at_a_time = policy.action_logits(list(obs_batch[i]), legal)
            # Batched and one-row forwards may differ in the last float32 ulp
            # (different BLAS blocking); rel=1e-5 is far tighter than that gap.
            assert [batched[i, a] for a in legal] == pytest.approx(one_at_a_time, rel=1e-5)
            assert all(batched[i, a] == float("-inf") for a in range(5) if a not in legal)
    finally:
        gpu_assert.reset_for_tests()


def test_action_logits_batch_default_impl_matches_the_mlp_override():
    """The Policy ABC's loop fallback and the MLP's batched override agree."""
    import numpy as np

    from mjai.agents.base import Policy

    gpu_assert.reset_for_tests()
    gpu_assert.require_cpu()
    try:
        policy = MLPSharedActorCritic(obs_size=4, num_actions=3, hidden_sizes=(8,), seed=1)
        obs_batch = np.array([[0.5, 0.5, 0.5, 0.5], [-1.0, 2.0, 0.0, 0.25]], dtype=np.float32)
        legal_mask = np.array([[True, True, False], [False, True, True]])
        override = policy.action_logits_batch(obs_batch, legal_mask)
        fallback = Policy.action_logits_batch(policy, obs_batch, legal_mask)
        assert override == pytest.approx(fallback, rel=1e-5, nan_ok=False)
    finally:
        gpu_assert.reset_for_tests()
