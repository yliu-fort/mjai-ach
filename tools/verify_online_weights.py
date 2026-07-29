"""Does the PIPELINE's per-sample weight aggregate to the offline weight it was verified as?

``docs/liars_residual_floor.md`` §8.5 verified the reach-tempered weight offline:
``tools/history_weighting.py`` walks the game tree, computes the per-information-set
mass ``W(I) = sum_{h in I} reach(h)^(1-kappa)`` that a ``reach(h)^-kappa`` sample
weight delivers, and shows it tracks the ideal ``rho(I)^(1-kappa)``. The RL
implementation then computes ``reach(h)`` a completely different way: it
accumulates ``log`` probabilities as the episode is played, from the chance
outcomes ``RolloutWorkerCore._sample_chance`` draws and the behavior log-probs
the policies return.

Those two routes share no code. This checks they agree, on the game the floor
was measured on rather than on the two-step tree a unit test can enumerate:

    E[ sum of weights the rollout emits at I ] / n_episodes  ==  W(I)

Both sides are computed under the SAME fixed policy, so the only thing that can
disagree is the reach. The comparison is restricted to information sets a finite
sample can actually estimate -- an information set with ``rho = 1e-9`` is visited
zero times in any feasible number of episodes and its Monte-Carlo mass is 0
against a large ``W``, which is the phenomenon under study, not an error in it.

Analysis tool; writes nothing under ``src/``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from tools.history_weighting import per_history_weights

from mjai.agents.tabular import TabularPolicy, _obs_to_key
from mjai.games.loader import load_game
from mjai.pipeline.rollout import RolloutConfig, RolloutWorkerCore
from mjai.seqform.tree import build_sequence_form


def tabular_from_behavior(sf, behavior: torch.Tensor, num_actions: int) -> TabularPolicy:
    """A :class:`TabularPolicy` that plays ``behavior`` at every information set.

    Keyed on ``sf.infoset_observation``, which ``mjai.seqform.tree`` fills from
    the same ``GameSpec.obs_tensor`` the rollout observes -- so the row the
    sequence form calls ``I`` and the row the rollout looks up are the same row.
    """
    policy = TabularPolicy(num_actions=num_actions, seed=0)
    for row in range(sf.num_infosets):
        obs = [float(x) for x in sf.infoset_observation[row]]
        probs = behavior[row]
        policy.logits[_obs_to_key(obs)] = [
            math.log(float(p)) if float(p) > 0.0 else -60.0 for p in probs
        ]
    return policy


def empirical_mass(spec, sf, policy: TabularPolicy, *, kappa: float, episodes: int) -> torch.Tensor:
    """Total weight the real rollout worker emits per information set."""
    index = {
        _obs_to_key([float(x) for x in row]): i for i, row in enumerate(sf.infoset_observation)
    }
    worker = RolloutWorkerCore(
        spec,
        config=RolloutConfig(
            n_episodes=episodes,
            target_samples=None,  # play every episode; no early stop
            seed=0,
            sample_weight_kappa=kappa,
        ),
    )
    batch = worker.run_episode(policy, policy)
    mass = torch.zeros(sf.num_infosets, dtype=torch.float64)
    weights = batch.weights
    if weights is None:  # kappa == 0: every sample counts once
        weights = [1.0] * batch.size
    for i in range(batch.size):
        mass[index[_obs_to_key([float(x) for x in batch.obs[i]])]] += float(weights[i])
    return mass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="liars_dice1")
    ap.add_argument("--nash", type=Path, default=Path("runs/nash_liars_dice1_behavior.pt"))
    ap.add_argument("--kappas", type=float, nargs="+", default=[0.0, 0.5, 0.75])
    ap.add_argument("--episodes", type=int, default=200000)
    # Compare only where the Monte-Carlo estimate has converged: an information
    # set expected to be visited fewer than this many times carries a relative
    # standard error above ~1/sqrt(n) and would swamp the agreement being tested.
    ap.add_argument("--min-visits", type=float, default=200.0)
    ap.add_argument("--out", type=Path, default=Path("runs/exact_ach/online_weights.json"))
    args = ap.parse_args()

    spec = load_game(args.game)
    sf = build_sequence_form(spec)
    target = torch.load(args.nash, weights_only=True).to(torch.float64)
    policy = tabular_from_behavior(sf, target, spec.num_actions)

    acc, rho = per_history_weights(spec, sf, target, [k for k in args.kappas if k > 0.0])
    acc[0.0] = rho  # kappa=0 is the visitation itself: W(I) = sum_h reach(h)
    testable = rho * args.episodes >= args.min_visits
    print(
        f"{args.game}: {int(testable.sum())} of {sf.num_infosets} information sets "
        f"expect >= {args.min_visits:g} visits in {args.episodes} episodes"
    )

    results: dict[str, dict[str, float]] = {}
    print(f"\n{'kappa':>6} {'median |err|':>13} {'p95 |err|':>11} {'max |err|':>11}")
    for kappa in args.kappas:
        got = empirical_mass(spec, sf, policy, kappa=kappa, episodes=args.episodes) / args.episodes
        want = acc[kappa]
        err = ((got - want).abs() / want.clamp(min=1e-300))[testable]
        results[str(kappa)] = {
            "median_rel_err": float(err.median()),
            "p95_rel_err": float(err.quantile(0.95)),
            "max_rel_err": float(err.max()),
            "n_compared": int(testable.sum()),
        }
        r = results[str(kappa)]
        print(
            f"{kappa:>6.2f} {r['median_rel_err']:>13.4f} "
            f"{r['p95_rel_err']:>11.4f} {r['max_rel_err']:>11.4f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
