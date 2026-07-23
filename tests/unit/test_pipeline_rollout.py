"""Unit tests for RolloutWorkerCore (AGENTS.md §5, Step 4).

Validates that the rollout back-end correctly plays both turn-based (Kuhn) and
simultaneous (BRPS) games and produces a well-formed Batch. Uses tabular
policies (no torch) and forces CPU.
"""

from __future__ import annotations

import math

import pytest

from mjai.agents.tabular import TabularPolicy
from mjai.games.loader import load_game
from mjai.pipeline.rollout import RolloutConfig, RolloutWorkerCore
from mjai.utils import gpu_assert


@pytest.fixture(autouse=True)
def _cpu_mode():
    gpu_assert.reset_for_tests()
    gpu_assert.require_cpu()
    yield
    gpu_assert.reset_for_tests()


def test_brps_rollout_produces_one_transition_per_player():
    """BRPS is one-shot simultaneous => one decision point per player per episode."""
    spec = load_game("brps")
    learner = TabularPolicy(num_actions=spec.num_actions, seed=0)
    opponent = TabularPolicy(num_actions=spec.num_actions, seed=1)
    worker = RolloutWorkerCore(spec, learner_player=0, config=RolloutConfig(seed=42))
    batch = worker.run_episode(learner, opponent)
    # 1 episode x 2 simultaneous players = 2 transitions.
    assert batch.size == 2
    assert batch.obs.shape == (2, spec.obs_size)


def test_kuhn_rollout_produces_multiple_steps():
    """Kuhn is sequential; an episode has 2-6 decision points total."""
    spec = load_game("kuhn")
    learner = TabularPolicy(num_actions=spec.num_actions, seed=0)
    opponent = TabularPolicy(num_actions=spec.num_actions, seed=1)
    worker = RolloutWorkerCore(spec, config=RolloutConfig(n_episodes=20, seed=7))
    batch = worker.run_episode(learner, opponent)
    # 20 episodes, each with >= 2 decision points.
    assert batch.size >= 40
    # Every action chosen is legal.
    for i in range(batch.size):
        assert int(batch.actions[i]) in batch.legal_actions[i]


def test_batch_returns_match_terminal_payoff_sign():
    """For these zero-sum games, returns sum to zero across the 2 players."""
    spec = load_game("kuhn")
    learner = TabularPolicy(num_actions=spec.num_actions, seed=3)
    opponent = TabularPolicy(num_actions=spec.num_actions, seed=4)
    worker = RolloutWorkerCore(spec, config=RolloutConfig(seed=99))
    batch = worker.run_episode(learner, opponent)
    # Sum of per-player returns over an episode = 0 (zero-sum). We pooled both
    # players' transitions, so summing returns over the episode = 0.
    assert math.isclose(float(batch.returns.sum()), 0.0, abs_tol=1e-9)


def test_simultaneous_actions_are_independent():
    """In BRPS both players act without seeing the other's choice."""
    spec = load_game("brps")
    # BRPS's initial observation is the same trivial vector for both players;
    # bias the logits on that exact observation so each picks a fixed action.
    obs = spec.obs_tensor(spec.new_state(), 0)
    learner = TabularPolicy(num_actions=3, seed=0)
    opponent = TabularPolicy(num_actions=3, seed=0)
    learner.get_logits(obs)[0] = 100.0  # force action 0
    opponent.get_logits(obs)[1] = 100.0  # force action 1
    worker = RolloutWorkerCore(spec, config=RolloutConfig(seed=1))
    batch = worker.run_episode(learner, opponent)
    actions = sorted(int(a) for a in batch.actions)
    assert actions == [0, 1]  # each played their own choice


def test_mirror_self_play_both_seats_same_policy():
    """Mirror: learner and opponent are the same object; batch pools both seats."""
    spec = load_game("brps")
    p = TabularPolicy(num_actions=spec.num_actions, seed=5)
    worker = RolloutWorkerCore(spec, learner_player=0, config=RolloutConfig(seed=2))
    batch = worker.run_episode(p, p)
    assert batch.size == 2


def test_multi_episode_batch_size_scales():
    spec = load_game("brps")
    p = TabularPolicy(num_actions=spec.num_actions, seed=0)
    worker = RolloutWorkerCore(spec, config=RolloutConfig(n_episodes=10, seed=0))
    batch = worker.run_episode(p, p)
    assert batch.size == 20  # 10 episodes x 2 players


def test_advantage_is_return_minus_value():
    spec = load_game("kuhn")
    learner = TabularPolicy(num_actions=spec.num_actions, seed=0)
    opponent = TabularPolicy(num_actions=spec.num_actions, seed=1)
    worker = RolloutWorkerCore(spec, config=RolloutConfig(seed=3))
    batch = worker.run_episode(learner, opponent)
    for i in range(batch.size):
        expected_adv = float(batch.returns[i]) - float(batch.values[i])
        assert math.isclose(float(batch.advantages[i]), expected_adv, abs_tol=1e-6)


def test_legal_mask_shape_matches_num_actions():
    spec = load_game("leduc")
    learner = TabularPolicy(num_actions=spec.num_actions, seed=0)
    opponent = TabularPolicy(num_actions=spec.num_actions, seed=1)
    worker = RolloutWorkerCore(spec, config=RolloutConfig(n_episodes=5, seed=0))
    batch = worker.run_episode(learner, opponent)
    assert batch.legal_mask.shape[1] == spec.num_actions
    # At least one legal action per row.
    assert batch.legal_mask.any(axis=1).all()


# ---- producer tags + seat shuffle + keep-set counting ----


def _producer_rows(batch, policy):
    """Row indices whose producer is ``policy`` (identity)."""
    idx = next(i for i, p in enumerate(batch.producers) if p is policy)
    return [i for i in range(batch.size) if int(batch.producer_idx[i]) == idx]


def test_runner_tags_every_transition_with_the_acting_policy():
    """Seat-0 learner, turn-based: learner's rows are exactly the seat-0 rows."""
    spec = load_game("kuhn")
    learner = TabularPolicy(num_actions=spec.num_actions, seed=0)
    opponent = TabularPolicy(num_actions=spec.num_actions, seed=1)
    worker = RolloutWorkerCore(spec, config=RolloutConfig(n_episodes=5, seed=11))
    batch = worker.run_episode(learner, opponent)
    assert set(batch.producers) == {learner, opponent}
    learner_rows = _producer_rows(batch, learner)
    assert learner_rows  # non-empty
    assert all(int(batch.players[i]) == 0 for i in learner_rows)
    assert all(int(batch.players[i]) == 1 for i in _producer_rows(batch, opponent))


def test_shuffle_seats_covers_both_perspectives_and_stays_routable():
    """With shuffle on, the learner acts from BOTH seats across episodes —
    and every one of those transitions is still attributed to the learner."""
    spec = load_game("brps")
    learner = TabularPolicy(num_actions=spec.num_actions, seed=0)
    opponent = TabularPolicy(num_actions=spec.num_actions, seed=1)
    worker = RolloutWorkerCore(
        spec, config=RolloutConfig(n_episodes=40, seed=5, shuffle_seats=True)
    )
    batch = worker.run_episode(learner, opponent)
    learner_seats = {int(batch.players[i]) for i in _producer_rows(batch, learner)}
    assert learner_seats == {0, 1}  # both perspectives covered
    # The opponent's rows are exactly the complement — no misattribution.
    learner_rows = set(_producer_rows(batch, learner))
    opponent_rows = set(_producer_rows(batch, opponent))
    assert learner_rows.isdisjoint(opponent_rows)
    assert len(learner_rows) + len(opponent_rows) == batch.size


def test_shuffle_seats_off_keeps_learner_in_base_seat():
    spec = load_game("brps")
    learner = TabularPolicy(num_actions=spec.num_actions, seed=0)
    opponent = TabularPolicy(num_actions=spec.num_actions, seed=1)
    worker = RolloutWorkerCore(spec, config=RolloutConfig(n_episodes=10, seed=5))
    batch = worker.run_episode(learner, opponent)
    assert all(int(batch.players[i]) == 0 for i in _producer_rows(batch, learner))


def test_shuffle_seats_is_deterministic_under_fixed_seed():
    spec = load_game("brps")
    cfg = RolloutConfig(n_episodes=20, seed=9, shuffle_seats=True)

    def _run():
        # Fresh policies per run: a policy's own sampling RNG is stateful.
        learner = TabularPolicy(num_actions=spec.num_actions, seed=0)
        opponent = TabularPolicy(num_actions=spec.num_actions, seed=1)
        return RolloutWorkerCore(spec, config=cfg).run_episode(learner, opponent)

    b1, b2 = _run(), _run()
    assert list(b1.players) == list(b2.players)
    assert list(b1.actions) == list(b2.actions)


def test_keep_set_counts_per_producer_min_rule():
    """Stop only when EVERY kept producer has >= target transitions."""
    spec = load_game("brps")  # 1 decision per seat per episode
    learner = TabularPolicy(num_actions=spec.num_actions, seed=0)
    opponent = TabularPolicy(num_actions=spec.num_actions, seed=1)
    worker = RolloutWorkerCore(spec, config=RolloutConfig(n_episodes=100, seed=3, target_samples=7))
    batch = worker.run_episode(learner, opponent, keep=(learner, opponent))
    assert len(_producer_rows(batch, learner)) >= 7
    assert len(_producer_rows(batch, opponent)) >= 7


def test_keep_single_producer_ignores_the_opponents_rate():
    """keep=(learner,) counts only the learner's transitions — the dose a
    controller that keeps one learner actually trains on (no half-batches)."""
    spec = load_game("brps")
    learner = TabularPolicy(num_actions=spec.num_actions, seed=0)
    opponent = TabularPolicy(num_actions=spec.num_actions, seed=1)
    worker = RolloutWorkerCore(spec, config=RolloutConfig(n_episodes=100, seed=3, target_samples=7))
    batch = worker.run_episode(learner, opponent, keep=(learner,))
    # ~7 learner transitions => ~14 pooled (both seats simulated).
    assert len(_producer_rows(batch, learner)) >= 7
    assert batch.size >= 14


def test_keep_empty_tuple_fails_loudly():
    spec = load_game("brps")
    learner = TabularPolicy(num_actions=spec.num_actions, seed=0)
    worker = RolloutWorkerCore(spec, config=RolloutConfig(seed=0))
    with pytest.raises(ValueError, match="keep"):
        worker.run_episode(learner, learner, keep=())
