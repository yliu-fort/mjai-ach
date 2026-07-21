"""Unit tests for TabularPolicy and the Policy interface (AGENTS.md §5)."""

from __future__ import annotations

import math

import pytest

from mjai.agents.base import entropy_of_probs, masked_softmax
from mjai.agents.tabular import TabularPolicy, _obs_to_key, uniform_tabular

# A canonical tiny observation for fixtures: 3-dim vector.
OBS_A = [1.0, 0.0, 0.0]
OBS_B = [0.0, 1.0, 0.0]


def test_construction_validates_args():
    with pytest.raises(ValueError, match="num_actions"):
        TabularPolicy(num_actions=0)
    with pytest.raises(ValueError, match="temperature"):
        TabularPolicy(num_actions=3, temperature=0.0)


def test_uniform_policy_samples_only_legal_actions():
    p = uniform_tabular(num_actions=5, seed=42)
    for _ in range(50):
        a, lp = p.act(OBS_A, legal_actions=[1, 3], eval=False)
        assert a in {1, 3}
        # Uniform over 2 actions => logprob ≈ log(0.5).
        assert math.isclose(lp, math.log(0.5), abs_tol=1e-6)


def test_uniform_over_full_action_space_when_all_legal():
    p = uniform_tabular(num_actions=4, seed=0)
    counts = [0, 0, 0, 0]
    for _ in range(4000):
        a, _ = p.act(OBS_A, legal_actions=[0, 1, 2, 3], eval=False)
        counts[a] += 1
    # Each action ~1000; allow generous tolerance for RNG.
    for c in counts:
        assert 700 < c < 1300


def test_eval_mode_is_greedy_and_deterministic():
    p = TabularPolicy(num_actions=3, seed=1)
    # Bias action 2 hard.
    p.get_logits(OBS_A)[2] = 10.0
    a1, _ = p.act(OBS_A, legal_actions=[0, 1, 2], eval=True)
    a2, _ = p.act(OBS_A, legal_actions=[0, 1, 2], eval=True)
    assert a1 == a2 == 2


def test_eval_mode_respects_legal_mask():
    p = TabularPolicy(num_actions=3, seed=1)
    p.get_logits(OBS_A)[2] = 10.0  # would be greedy, but illegal here
    a, _ = p.act(OBS_A, legal_actions=[0, 1], eval=True)
    assert a in {0, 1}


def test_illegal_actions_have_zero_probability():
    p = TabularPolicy(num_actions=4, seed=2)
    p.get_logits(OBS_A)[0] = 5.0  # strong preference for illegal action 0
    for _ in range(200):
        a, _ = p.act(OBS_A, legal_actions=[1, 2, 3], eval=False)
        assert a in {1, 2, 3}


def test_distinct_observations_have_distinct_rows():
    p = TabularPolicy(num_actions=3, seed=0)
    p.get_logits(OBS_A)[0] = 9.0
    # OBS_B should be independent.
    assert p.get_logits(OBS_B)[0] == 0.0
    assert p.num_rows() == 2


def test_value_defaults_to_zero_and_is_mutable():
    p = TabularPolicy(num_actions=2, seed=0)
    assert p.value(OBS_A) == 0.0
    # Mutate via the public dict (the way an UpdateRule would).
    p.values[_obs_to_key(OBS_A)] = 0.7
    assert p.value(OBS_A) == 0.7


def test_temperature_controls_exploration():
    """Higher temperature -> more uniform sampling."""
    cold = TabularPolicy(num_actions=3, temperature=0.1, seed=0)
    hot = TabularPolicy(num_actions=3, temperature=10.0, seed=0)
    # Give action 0 a moderate edge in both.
    cold.get_logits(OBS_A)[0] = 1.0
    hot.get_logits(OBS_A)[0] = 1.0
    cold_count = sum(1 for _ in range(2000) if cold.act(OBS_A, [0, 1, 2], eval=False)[0] == 0)
    hot_count = sum(1 for _ in range(2000) if hot.act(OBS_A, [0, 1, 2], eval=False)[0] == 0)
    assert cold_count > hot_count  # cold concentrates on the best action


def test_save_load_roundtrip(tmp_path):
    p = TabularPolicy(num_actions=3, temperature=2.0, seed=0)
    p.get_logits(OBS_A)[0] = 1.5
    p.get_logits(OBS_B)[2] = -0.7
    p.values[_obs_to_key(OBS_A)] = 0.3

    path = str(tmp_path / "ckpt.json")
    p.save(path)

    q = TabularPolicy(num_actions=3, seed=99)
    q.load(path)
    assert q.num_actions == 3
    assert q.temperature == 2.0
    assert q.get_logits(OBS_A)[0] == 1.5
    assert q.get_logits(OBS_B)[2] == -0.7
    assert q.value(OBS_A) == 0.3
    # Behaves identically after reload.
    a_p, _ = p.act(OBS_A, [0, 1, 2], eval=True)
    a_q, _ = q.act(OBS_A, [0, 1, 2], eval=True)
    assert a_p == a_q


def test_save_load_pkl_format(tmp_path):
    p = TabularPolicy(num_actions=2, seed=0)
    p.get_logits(OBS_A)[0] = 1.0
    path = str(tmp_path / "ckpt.pkl")
    p.save(path)
    q = TabularPolicy(num_actions=2, seed=0)
    q.load(path)
    assert q.get_logits(OBS_A)[0] == 1.0


def test_load_rejects_wrong_kind(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"kind": "mlp", "num_actions": 3}')
    p = TabularPolicy(num_actions=3, seed=0)
    with pytest.raises(ValueError, match="tabular"):
        p.load(str(path))


def test_action_logits_length_matches_legal():
    p = TabularPolicy(num_actions=5, seed=0)
    lg = p.action_logits(OBS_A, legal_actions=[1, 3, 4])
    assert len(lg) == 3


def test_empty_legal_actions_raises():
    p = TabularPolicy(num_actions=3, seed=0)
    with pytest.raises(ValueError, match="non-empty"):
        p.act(OBS_A, [], eval=False)


# --- pure functions on base.py ---


def test_masked_softmax_basic():
    logits = [1.0, 2.0, 3.0, 0.0]
    mask = [False, True, True, False]
    probs = masked_softmax(logits, mask)
    assert probs[0] == 0.0 and probs[3] == 0.0
    assert math.isclose(sum(probs), 1.0)
    assert probs[2] > probs[1]  # higher logit -> higher prob


def test_masked_softmax_stability_with_extreme_values():
    logits = [1e6, -1e6, 1e6 - 1]
    mask = [True, True, True]
    probs = masked_softmax(logits, mask)
    assert all(math.isfinite(p) for p in probs)
    assert math.isclose(sum(probs), 1.0)


def test_masked_softmax_no_legal_returns_uniform_over_mask():
    # If somehow all masked False, return uniform (shouldn't happen in practice).
    probs = masked_softmax([1.0, 2.0], [False, False])
    assert probs == [0.5, 0.5]


def test_entropy_uniform_is_log_n():
    n = 4
    probs = [1.0 / n] * n
    assert math.isclose(entropy_of_probs(probs), math.log(n), abs_tol=1e-9)


def test_entropy_degenerate_is_zero():
    assert entropy_of_probs([1.0, 0.0, 0.0]) == 0.0


def test_act_with_value_matches_separate_calls():
    """Fused ``act_with_value`` must equal ``act`` + ``value`` on the same policy.

    Covers both the greedy (eval) path — bit-for-bit identical, no RNG — and the
    stochastic path — same RNG draw, since the override consumes exactly one
    ``rng.random()`` like ``act`` does, and ``value`` consumes none. This is the
    correctness contract for the rollout hot-path optimization (AGENTS.md §8).
    """
    # Greedy path: deterministic, compare exactly.
    p = TabularPolicy(num_actions=4, seed=3)
    p.get_logits(OBS_A)[1] = 5.0  # bias action 1
    p.values[_obs_to_key(OBS_A)] = 0.7
    a_g, lp_g, v_g = p.act_with_value(OBS_A, legal_actions=[0, 1, 2, 3], eval=True)
    a_sep, lp_sep = p.act(OBS_A, [0, 1, 2, 3], eval=True)
    assert a_g == a_sep == 1
    assert lp_g == lp_sep == 0.0
    assert math.isclose(v_g, 0.7, abs_tol=1e-9)
    assert math.isclose(v_g, p.value(OBS_A), abs_tol=1e-9)

    # Stochastic path: two sibling policies, same seed, same biased logits.
    # First sibling: call act() then value(). Second: call act_with_value().
    # Same RNG state at the call => same sample.
    p1 = TabularPolicy(num_actions=3, seed=99)
    p2 = TabularPolicy(num_actions=3, seed=99)
    for p in (p1, p2):
        p.get_logits(OBS_B)[0] = 0.3
        p.get_logits(OBS_B)[1] = 0.3
        p.values[_obs_to_key(OBS_B)] = -1.25
    a1, lp1 = p1.act(OBS_B, [0, 1, 2], eval=False)
    v1 = p1.value(OBS_B)
    a2, lp2, v2 = p2.act_with_value(OBS_B, [0, 1, 2], eval=False)
    assert a1 == a2
    assert math.isclose(lp1, lp2, abs_tol=1e-9)
    assert math.isclose(v1, v2, abs_tol=1e-9)
    assert math.isclose(v2, -1.25, abs_tol=1e-9)
