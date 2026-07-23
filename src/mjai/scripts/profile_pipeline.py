"""Profile the pipeline: per-phase, per-op, per-transfer breakdown (AGENTS.md §8).

Answers the three questions worth asking before optimizing anything here:

  1. **Where do the seconds go?** ``--phases`` times rollout / learner update /
     equilibrium eval separately, synchronizing around each so GPU numbers are
     honest rather than "time to enqueue".
  2. **What does one env-step cost?** ``--ops`` counts the aten ops, kernel
     launches and host<->device memcpys behind a single policy call and a
     single learner update.
  3. **Is the device the right one?** ``--device cuda`` vs ``--device cpu``.
     For Phase-1-sized games the rollout asks the policy for ONE decision at a
     time, so a small MLP forward is launch-and-sync overhead: measured 2809
     env-steps/s on CPU vs 441 on CUDA for Liar's Dice.

The legacy cProfile view is still available with ``--cprofile``.

Usage::

    uv run python -m mjai.scripts.profile_pipeline --game liars_dice1 --device cuda
    uv run python -m mjai.scripts.profile_pipeline --game kuhn --cprofile --steps 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_TEMPLATE = "configs/exp/{game}_ach_mlp_mirror.yaml"


def _load_config(game: str, device: str) -> Any:
    """The game's MLP mirror arm, pinned to ``device``."""
    import dataclasses

    import yaml

    from mjai.scripts.experiment import ExperimentConfig

    repo = Path(__file__).resolve().parents[3]
    path = repo / DEFAULT_CONFIG_TEMPLATE.format(game=game)
    if not path.is_file():
        raise FileNotFoundError(f"no MLP config for {game!r}: {path}")
    cfg = ExperimentConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))
    return dataclasses.replace(cfg, device=device, seed=0)


def _build(game: str, device: str) -> tuple[Any, Any, Any]:
    """(trainer, policy, spec) for ``game`` on ``device``."""
    from mjai.algos.controller import Trainer
    from mjai.games.loader import load_game
    from mjai.scripts.experiment_build import (
        build_controller,
        build_policy,
        build_update_rule,
    )

    cfg = _load_config(game, device)
    spec = load_game(game)
    policy = build_policy(spec, cfg, seed=0)
    rule = build_update_rule(policy, cfg, spec)
    controller = build_controller(spec, policy, cfg, rng=random.Random(0))
    return Trainer(policy=policy, update_rule=rule, controller=controller), policy, cfg


def _sync(device: str) -> None:
    if device != "cpu":
        import torch

        torch.cuda.synchronize()


def profile_phases(game: str, device: str, rounds: int) -> dict[str, object]:
    """Wall-clock split across rollout / update / eval, synchronized per phase."""
    from mjai.eval.nash import evaluate_equilibrium
    from mjai.games.loader import load_game

    trainer, policy, cfg = _build(game, device)
    spec = load_game(game)
    trainer.step()  # warm up: CUDA context, allocator, autotune
    _sync(device)

    t_rollout = t_update = 0.0
    samples = 0
    for _ in range(rounds):
        trainer.controller.set_learner(trainer.policy)
        _sync(device)
        t0 = time.perf_counter()
        batch = trainer.controller.collect()
        _sync(device)
        t1 = time.perf_counter()
        trainer.update_rule.step(batch)
        _sync(device)
        t_rollout += t1 - t0
        t_update += time.perf_counter() - t1
        samples += batch.size

    evaluate_equilibrium(spec, policy, estimator=cfg.eval_estimator)  # warm the skeleton cache
    t0 = time.perf_counter()
    evaluate_equilibrium(
        spec, policy, estimator=cfg.eval_estimator, exact_backend=cfg.eval_exact_backend
    )
    t_eval = time.perf_counter() - t0

    train = t_rollout + t_update
    result = {
        "device": device,
        "rounds": rounds,
        "samples": samples,
        "rollout_s": t_rollout,
        "update_s": t_update,
        "eval_s_per_point": t_eval,
        "env_steps_per_s": samples / train if train else 0.0,
    }
    print(f"PHASE SPLIT — {game} on {device}, {rounds} train rounds")
    print(
        f"  rollout        {t_rollout:8.3f}s  {t_rollout / train:6.1%}  "
        f"{t_rollout / rounds * 1e3:8.2f} ms/round"
    )
    print(
        f"  learner update {t_update:8.3f}s  {t_update / train:6.1%}  "
        f"{t_update / rounds * 1e3:8.2f} ms/round"
    )
    print(f"  -> {samples / train:.0f} env-steps/s")
    print(
        f"  one eval point {t_eval:8.3f}s  (estimator={cfg.eval_estimator}, "
        f"backend={cfg.eval_exact_backend})"
    )
    return result


def profile_ops(game: str, device: str, calls: int) -> dict[str, object]:
    """Op / launch / memcpy accounting for one policy call and one update."""
    from torch.profiler import ProfilerActivity, profile

    from mjai.games.loader import load_game

    trainer, policy, _cfg = _build(game, device)
    spec = load_game(game)
    batch = trainer.controller.collect()
    obs = [0.0] * spec.obs_size
    legal = list(range(min(6, spec.num_actions)))
    activities = [ProfilerActivity.CPU]
    if device != "cpu":
        activities.append(ProfilerActivity.CUDA)

    def report(prof: Any, label: str, n: int) -> dict[str, float]:
        events = prof.key_averages()
        memcpy = [e for e in events if "memcpy" in e.key.lower()]
        h2d = sum(e.count for e in memcpy if "htod" in e.key.lower()) / n
        d2h = sum(e.count for e in memcpy if "dtoh" in e.key.lower()) / n
        syncs = sum(e.count for e in events if "synchronize" in e.key.lower()) / n
        ops = sum(e.count for e in events) / n
        print(f"\n{label} (per call, n={n})")
        print(f"  aten op invocations {ops:8.1f}")
        print(f"  memcpy HtoD / DtoH  {h2d:8.2f} / {d2h:.2f}")
        print(f"  device synchronize  {syncs:8.2f}")
        for e in sorted(events, key=lambda ev: -ev.cpu_time_total)[:6]:
            print(f"    {e.key[:44]:<44s} n={e.count / n:6.2f}  cpu={e.cpu_time_total / n:8.1f} us")
        return {"ops": ops, "h2d": h2d, "d2h": d2h, "syncs": syncs}

    with profile(activities=activities, acc_events=True) as prof:
        for _ in range(calls):
            policy.act_with_value(obs, legal, eval=False)
        _sync(device)
    per_step = report(prof, "ONE act_with_value (one env-step's policy call)", calls)

    with profile(activities=activities, acc_events=True) as prof:
        for _ in range(max(calls // 5, 1)):
            trainer.update_rule.step(batch)
        _sync(device)
    per_update = report(prof, "ONE update_rule.step", max(calls // 5, 1))
    return {"act_with_value": per_step, "update": per_update}


def profile_memory(game: str, device: str, rounds: int) -> dict[str, object]:
    """Allocator churn: allocations, cudaMalloc segments, retries, peak."""
    if device == "cpu":
        return {}
    import torch

    trainer, _policy, _cfg = _build(game, device)
    trainer.step()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_stats()
    for _ in range(rounds):
        trainer.step()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    out = {
        "tensor_allocations": after.get("allocation.all.allocated", 0)
        - before.get("allocation.all.allocated", 0),
        "cuda_malloc_segments": after.get("segment.all.allocated", 0)
        - before.get("segment.all.allocated", 0),
        "alloc_retries": after.get("num_alloc_retries", 0) - before.get("num_alloc_retries", 0),
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
    }
    print(f"\nALLOCATOR CHURN over {rounds} train rounds")
    for key, value in out.items():
        print(f"  {key:<24s} {value:12.2f}")
    return out


def run_cprofile(game: str, algo: str, mode: str, steps: int, out_dir: Path) -> None:
    """Legacy whole-run cProfile view (CPU, tabular-scale runs)."""
    import cProfile
    import io
    import pstats

    from mjai.scripts.experiment import ExperimentConfig, run_experiment

    cfg = ExperimentConfig(
        game=game,
        algo=algo,
        self_play_mode=mode,
        n_steps=steps,
        out_dir=str(out_dir / f"profile_{game}_{algo}_{mode}"),
    )
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    run_experiment(cfg)
    pr.disable()
    wall = time.perf_counter() - t0
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(30)
    print(f"\ncProfile: {steps} steps of {game}/{algo}/{mode} in {wall:.2f}s\n{buf.getvalue()}")
    (out_dir / f"profile_{game}_{algo}_{mode}.txt").write_text(buf.getvalue())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile a short mjai training run.")
    parser.add_argument("--game", default="liars_dice1")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | cuda:N")
    parser.add_argument("--rounds", type=int, default=30, help="Train rounds for the phase split.")
    parser.add_argument("--calls", type=int, default=50, help="Policy calls for the op counts.")
    parser.add_argument("--out", default="profiles")
    parser.add_argument("--phases", action="store_true", help="Phase split (default when bare).")
    parser.add_argument("--ops", action="store_true", help="Op / memcpy / sync counts.")
    parser.add_argument("--memory", action="store_true", help="Allocator churn (CUDA only).")
    parser.add_argument("--cprofile", action="store_true", help="Legacy whole-run cProfile.")
    parser.add_argument("--algo", default="ach", help="cProfile mode only.")
    parser.add_argument("--mode", default="mirror", help="cProfile mode only.")
    parser.add_argument("--steps", type=int, default=50, help="cProfile mode only.")
    args = parser.parse_args(argv)

    if args.device == "cpu":
        from mjai.utils import gpu_assert

        gpu_assert.require_cpu()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.cprofile:
        run_cprofile(args.game, args.algo, args.mode, args.steps, out_dir)
        return 0

    everything = not (args.phases or args.ops or args.memory)
    summary: dict[str, object] = {"game": args.game, "device": args.device}
    if everything or args.phases:
        summary["phases"] = profile_phases(args.game, args.device, args.rounds)
    if everything or args.ops:
        summary["ops"] = profile_ops(args.game, args.device, args.calls)
    if everything or args.memory:
        summary["memory"] = profile_memory(args.game, args.device, 10)
    path = out_dir / f"profile_{args.game}_{args.device.replace(':', '')}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
