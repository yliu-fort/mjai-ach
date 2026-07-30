"""Freeze the NN update rules' exact numeric behavior into a golden fixture.

Writes ``tests/unit/data/nn_updates_golden.json``: for a handful of fully
specified scenarios (explicit initial weights + explicit batch, no seeded
init), the per-step :class:`~mjai.algos.transition.UpdateStats` and the exact
post-step parameter tensors.

Why this exists: the two endpoint classes (``NNACHUpdate`` / ``NNPPOUpdate``)
were merged into one theta-parameterized rule. ACH's numerics are what the
paper reproduction rests on (docs/reproduce_report.md), so the merge had to be
bit-exact at ``theta=1``. **The committed fixture is the output of the
PRE-merge two-endpoint code** and is never regenerated — that is what makes it
evidence. Verify the current code still reproduces it::

    uv run python tools/gen_nn_golden.py --check   # must print OK

The check requires every recorded parameter tensor and every recorded stat to
match **to within platform float32 noise** (:data:`GOLDEN_RTOL`); telemetry keys
the fixture does not carry are allowed to be added (the merge did exactly that —
``grad_norm`` became available on the PPO arms, which the old PPO endpoint never
computed). Removing or changing a recorded value fails.

Why a tolerance and not equality (AGENTS.md D19). Storage is exact — the values
are Python doubles, which round-trip float32 losslessly — but *recomputation* is
not: the same float32 graph reduced by a different BLAS kernel lands a few ULP
away. Measured 2026-07-30 on macOS/CPU torch 2.x against this fixture (generated
on another platform): 9-20 of each scenario's 26-41 recorded values differ, worst
case **4.43 float32 ULP** (5.3e-7 relative) on ``explained_variance``, a ratio of
variances and so the most cancellation-prone number recorded; parameters, which
are single-reduction values, stay inside 2.09 ULP. Equality therefore tested the
host's BLAS, not the update rule.

What the tolerance still catches, measured on ``ach_shipped_layernorm`` as the
worst-case relative deviation of any recorded value — all 100x to 1e5x above
:data:`GOLDEN_RTOL`::

    l_th 2.0 -> 0.5             2.1e+0     loss_centered_logits=True   1.1e+0
    eta +0.1%                   1.0e-3     value_coef +0.1%            6.0e-4
    learning_rate +0.1%         1.0e-4

What it does NOT catch, also measured: ``entropy_coef`` 0.01 -> 0.0101 is
invisible (0 differences), because the entropy term's share of a two-step
parameter change is ~1e-6 relative — under the tolerance. The fixture pins the
policy term's *structure* (which is what licenses the theta merge, AGENTS.md
D11), not the calibration of a coefficient. Also blind, and this one is a
pre-existing coverage gap rather than a consequence of the tolerance:
``gate_centered_logits=True`` reproduces every scenario bit for bit, because the
gate mask happens to come out identical under both readings of ambiguity A3.
``test_golden_comparison_rejects_an_algorithmic_change`` pins the comparator's
teeth directly instead of relying on any of these.

Scenario configs are plain :class:`AlgoConfig` kwargs; pre-merge a small
adapter mapped them onto the two old constructors and asserted that every
knob that did not exist yet carried the value the old code hardcoded.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
OUT_PATH = REPO / "tests" / "unit" / "data" / "nn_updates_golden.json"

# Comparison tolerance for the golden check (module docstring; AGENTS.md D19).
# 1e-5 relative is ~84 float32 ULP: ~19x the worst platform disagreement measured
# against this fixture (4.43 ULP) and ~100x below the smallest structural change
# it exists to catch (learning_rate +0.1% -> 1.0e-4). Not tightened further
# because the 4.43 ULP is one host's BLAS; a GPU or a different vendor library
# can reduce in yet another order, and this check runs wherever CI runs.
# The absolute floor only bites on values the fixture records as exactly 0
# (clip_frac / gate_off_frac at an endpoint), where a relative test is undefined;
# it sits below one ULP of the parameter magnitudes (~1e-8), so parameters are
# effectively relative-only.
GOLDEN_RTOL = 1e-5
GOLDEN_ATOL = 1e-9

# Shared ACH knobs: the shipped reproduction defaults (configs/exp/*_ach_mlp_*.yaml
# -- SGD lr=1e-3, alpha=2.0 <=> value_coef=1.0, beta=1e-2, no grad clipping,
# eta=1.0, l_th=2.0, raw-logit gate + raw-logit loss body paired with the
# LayerNorm torso). Paper: Fu et al. ICLR 2022, p27-28.
ACH_SHIPPED: dict[str, Any] = {
    "theta": 1.0,
    "learning_rate": 0.001,
    "value_coef": 1.0,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.0,
    "gae_lambda": 0.95,
    "optimizer": "sgd",
    "eta": 1.0,
    "l_th": 2.0,
    "ratio_eps": 0.5,
    "clip_eps": 0.2,
    "loss_centered_logits": False,
    "centered_mean_legal_only": False,
    "gate_centered_logits": False,
    "normalize_advantages": False,
    "n_epochs": 1,
    "adam_eps": 1e-5,
}

# Reference PPO as it shipped pre-merge: Adam(eps=1e-5), per-batch advantage
# normalization, clip 0.2, grad-norm clip 0.5, one full-batch epoch.
PPO_LEGACY: dict[str, Any] = {
    **ACH_SHIPPED,
    "theta": 0.0,
    "learning_rate": 0.0003,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "optimizer": "adam",
    "normalize_advantages": True,
}


def _scenarios() -> list[dict[str, Any]]:
    """Scenario definitions; every numeric input is explicit (see module docstring)."""
    return [
        {
            "name": "ach_shipped_layernorm",
            "why": "The shipped ACH default: raw-logit gate + raw-logit loss body "
            "on a LayerNorm torso (docs/reproduce_report.md 6.5).",
            "config": dict(ACH_SHIPPED),
            "policy": {"trunk_layernorm": True},
            "n_steps": 3,
        },
        {
            "name": "ach_pre_layernorm_centered",
            "why": "Pre-LayerNorm behavior: both gate and loss body on the "
            "mean-centered logit, no torso LayerNorm.",
            "config": {
                **ACH_SHIPPED,
                "loss_centered_logits": True,
                "gate_centered_logits": True,
            },
            "policy": {"trunk_layernorm": False},
            "n_steps": 3,
        },
        {
            "name": "ach_legal_only_mean",
            "why": "A5 probe path: y_bar averaged over legal actions only, with "
            "a batch that actually has illegal actions.",
            "config": {
                **ACH_SHIPPED,
                "loss_centered_logits": True,
                "gate_centered_logits": True,
                "centered_mean_legal_only": True,
            },
            "policy": {"trunk_layernorm": True},
            "n_steps": 3,
        },
        {
            "name": "ach_grad_clipped",
            "why": "Grad-clipping branch (max_grad_norm > 0) on the ACH side.",
            "config": {**ACH_SHIPPED, "max_grad_norm": 0.5},
            "policy": {"trunk_layernorm": True},
            "n_steps": 2,
        },
        {
            "name": "ppo_legacy_adam",
            "why": "Reference PPO exactly as it shipped pre-merge: Adam(eps=1e-5) "
            "+ advantage normalization + clip 0.2 + grad clip 0.5. Adam carries "
            "optimizer state, so 3 steps also pin the moment buffers.",
            "config": dict(PPO_LEGACY),
            "policy": {"trunk_layernorm": True},
            "n_steps": 3,
        },
        {
            "name": "ppo_on_ach_scaffolding",
            "why": "PPO loss under the ACH scaffolding (SGD, raw advantages, no "
            "grad clip) -- the theta=0 arm of the scan, and the new PPO default.",
            "config": {
                **ACH_SHIPPED,
                "theta": 0.0,
                "normalize_advantages": False,
            },
            "policy": {"trunk_layernorm": True},
            "n_steps": 3,
        },
    ]


# Toy architecture: small enough that the fixture stays a few tens of KB, wide
# enough to exercise every code path (masking, LayerNorm, two heads).
OBS_SIZE = 8
NUM_ACTIONS = 5
HIDDEN = (16,)
BATCH_SIZE = 10


def _init_params(rng: np.random.Generator, *, trunk_layernorm: bool) -> dict[str, list]:
    """Draw explicit initial weights for every parameter of the toy MLP.

    Explicit rather than seeded so the fixture does not depend on torch's
    initializer staying byte-stable across versions.
    """
    from mjai.agents.mlp import ACTIVATIONS, MLPSharedActorCritic

    policy = MLPSharedActorCritic(
        obs_size=OBS_SIZE,
        num_actions=NUM_ACTIONS,
        hidden_sizes=HIDDEN,
        activation=ACTIVATIONS["relu"],
        trunk_layernorm=trunk_layernorm,
        device="cpu",
        seed=0,
    )
    out: dict[str, list] = {}
    for name, p in policy.named_parameters():
        # LayerNorm weight/bias keep their canonical init (1s / 0s); everything
        # else gets a small symmetric random draw.
        if name.endswith("weight") and p.dim() == 1:
            vals = np.ones(p.shape, dtype=np.float32)
        elif name.endswith("bias") and "norm" in name.lower():
            vals = np.zeros(p.shape, dtype=np.float32)
        else:
            vals = rng.uniform(-0.5, 0.5, size=tuple(p.shape)).astype(np.float32)
        out[name] = vals.tolist()
    return out


def _make_batch_spec(rng: np.random.Generator, *, with_illegal: bool) -> dict[str, Any]:
    """Draw an explicit batch: observations, legal mask, actions, and targets."""
    obs = rng.uniform(-1.0, 1.0, size=(BATCH_SIZE, OBS_SIZE)).astype(np.float32)
    legal_mask = np.ones((BATCH_SIZE, NUM_ACTIONS), dtype=bool)
    if with_illegal:
        # Shrink the legal set row by row (never below 2 legal actions), which
        # is what makes the legal-only-mean knob observable.
        for i in range(BATCH_SIZE):
            n_illegal = int(rng.integers(0, NUM_ACTIONS - 1))
            for a in rng.choice(NUM_ACTIONS, size=n_illegal, replace=False):
                legal_mask[i, int(a)] = False
            if not legal_mask[i].any():
                legal_mask[i, 0] = True
    actions = np.array(
        [int(rng.choice(np.nonzero(legal_mask[i])[0])) for i in range(BATCH_SIZE)],
        dtype=np.int64,
    )
    # Behavior-policy probabilities well inside (0, 1): 1/pi_old is the ACH
    # importance weight, so tiny values would make the fixture hypersensitive.
    old_probs = rng.uniform(0.1, 0.9, size=BATCH_SIZE).astype(np.float32)
    logprobs = np.log(old_probs).astype(np.float32)
    values = rng.uniform(-1.0, 1.0, size=BATCH_SIZE).astype(np.float32)
    returns = rng.uniform(-1.0, 1.0, size=BATCH_SIZE).astype(np.float32)
    # Mixed-sign advantages so both branches of the one-sided ACH gate fire.
    advantages = rng.uniform(-1.5, 1.5, size=BATCH_SIZE).astype(np.float32)
    return {
        "obs": obs.tolist(),
        "legal_mask": legal_mask.tolist(),
        "actions": actions.tolist(),
        "logprobs": logprobs.tolist(),
        "values": values.tolist(),
        "returns": returns.tolist(),
        "advantages": advantages.tolist(),
        "num_actions": NUM_ACTIONS,
    }


def build_batch(spec: dict[str, Any]):  # type: ignore[no-untyped-def]
    """Materialize a :class:`Batch` from a fixture's batch spec."""
    from mjai.algos.transition import Batch

    n = len(spec["actions"])
    return Batch(
        obs=np.asarray(spec["obs"], dtype=np.float32),
        legal_actions=[[a for a, ok in enumerate(row) if ok] for row in spec["legal_mask"]],
        actions=np.asarray(spec["actions"], dtype=np.int64),
        logprobs=np.asarray(spec["logprobs"], dtype=np.float32),
        values=np.asarray(spec["values"], dtype=np.float32),
        returns=np.asarray(spec["returns"], dtype=np.float32),
        advantages=np.asarray(spec["advantages"], dtype=np.float32),
        legal_mask=np.asarray(spec["legal_mask"], dtype=bool),
        players=np.zeros((n,), dtype=np.int8),
        num_actions=int(spec["num_actions"]),
    )


def build_policy(policy_spec: dict[str, Any], init_params: dict[str, list]):  # type: ignore[no-untyped-def]
    """Build the toy MLP and load the fixture's explicit initial weights."""
    from mjai.agents.mlp import ACTIVATIONS, MLPSharedActorCritic

    policy = MLPSharedActorCritic(
        obs_size=OBS_SIZE,
        num_actions=NUM_ACTIONS,
        hidden_sizes=HIDDEN,
        activation=ACTIVATIONS["relu"],
        trunk_layernorm=bool(policy_spec["trunk_layernorm"]),
        device="cpu",
        seed=0,
    )
    with torch.no_grad():
        for name, p in policy.named_parameters():
            p.copy_(torch.as_tensor(init_params[name], dtype=torch.float32))
    return policy


def build_rule(policy: Any, config: dict[str, Any]) -> Any:
    """Construct the update rule for a scenario config.

    Scenario configs are plain :class:`AlgoConfig` kwargs. (Pre-merge this
    function was an adapter dispatching on ``theta`` to the two endpoint
    classes; the diff that removed it is the merge itself.)
    """
    from mjai.algos.nn_updates import NNActorCriticUpdate
    from mjai.algos.update_rule import AlgoConfig

    return NNActorCriticUpdate(policy, AlgoConfig(**config))


def _stats_to_dict(stats: Any) -> dict[str, Any]:
    return {
        "policy_loss": float(stats.policy_loss),
        "value_loss": float(stats.value_loss),
        "entropy": float(stats.entropy),
        "approx_kl": float(stats.approx_kl),
        "clip_frac": float(stats.clip_frac),
        "explained_variance": float(stats.explained_variance),
        "extra": {k: float(v) for k, v in sorted(stats.extra.items())},
    }


def run_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    """Execute one scenario and return its observed stats + final parameters."""
    policy = build_policy(scenario["policy"], scenario["init_params"])
    rule = build_rule(policy, scenario["config"])
    batch = build_batch(scenario["batch"])
    stats_seq = [_stats_to_dict(rule.step(batch)) for _ in range(int(scenario["n_steps"]))]
    params = {name: p.detach().cpu().numpy().tolist() for name, p in policy.named_parameters()}
    return {"stats": stats_seq, "params": params}


def build_fixture() -> dict[str, Any]:
    """Assemble the full fixture (inputs + observed outputs)."""
    from mjai.utils import gpu_assert

    gpu_assert.reset_for_tests()
    gpu_assert.require_cpu()

    rng = np.random.default_rng(20260723)
    scenarios = _scenarios()
    for i, sc in enumerate(scenarios):
        sc["init_params"] = _init_params(rng, trunk_layernorm=sc["policy"]["trunk_layernorm"])
        sc["batch"] = _make_batch_spec(rng, with_illegal=(i % 2 == 1))
        sc["expected"] = run_scenario(sc)
    return {
        "_comment": (
            "Golden fixture for the NN actor-critic update rule. Generated by "
            "tools/gen_nn_golden.py; regenerating must be a deliberate act -- "
            "a diff here means the update rule's numerics changed."
        ),
        "arch": {
            "obs_size": OBS_SIZE,
            "num_actions": NUM_ACTIONS,
            "hidden_sizes": list(HIDDEN),
            "activation": "relu",
            "batch_size": BATCH_SIZE,
        },
        "scenarios": scenarios,
    }


def _mismatch(want: float, got: object) -> float | None:
    """Relative deviation if ``got`` is outside tolerance, else None.

    ``None`` for a value that is absent (rather than 0.0) — a missing key is not
    a numeric difference and is reported separately by the caller.
    """
    if not isinstance(got, int | float):
        return math.inf
    if math.isclose(float(got), want, rel_tol=GOLDEN_RTOL, abs_tol=GOLDEN_ATOL):
        return None
    return abs(float(got) - want) / max(abs(want), 1e-30)


def compare_expected(want: dict[str, Any], got: dict[str, Any], label: str = "") -> list[str]:
    """Compare one scenario's recorded outputs against a fresh run.

    Both arguments are the ``{"stats": [...], "params": {...}}`` shape — the
    fixture's ``expected`` block and :func:`run_scenario`'s return value are the
    same shape on purpose, so the ``--check`` CLI and
    ``tests/unit/test_algos_nn_theta.py`` share ONE comparison. Two gates that
    disagree about what "reproduces the fixture" means would let a numerics
    change through whichever is looser (AGENTS.md §4 names both).

    Returns human-readable differences; empty list means the fixture holds.
    Telemetry keys absent from the fixture may be present in the fresh run
    (additive observability is not a numerics change); the reverse is a failure.
    """
    prefix = f"{label} " if label else ""
    diffs = _compare_stats(want["stats"], got["stats"], prefix)
    diffs.extend(_compare_params(want["params"], got["params"], prefix))
    return diffs


def _compare_scalar(
    label: str, key: str, want: float, got_row: dict[str, Any], prefix: str
) -> list[str]:
    """One recorded scalar: absent, or outside tolerance, or fine (empty list).

    ``label`` is for the message, ``key`` is what to look up — they differ
    because the message carries the step index and the ``extra.`` namespace.
    """
    if key not in got_row:
        return [f"{prefix}{label}: missing from the fresh run"]
    observed = got_row[key]
    rel = _mismatch(want, observed)
    if rel is None:
        return []
    return [f"{prefix}{label}: {want} != {observed} (rel {rel:.2e} > {GOLDEN_RTOL:.0e})"]


def _compare_stats(
    want_stats: list[dict[str, Any]], got_stats: list[dict[str, Any]], prefix: str
) -> list[str]:
    """Per-step :class:`UpdateStats`, including the nested ``extra`` telemetry."""
    diffs: list[str] = []
    for i, (a, b) in enumerate(zip(want_stats, got_stats, strict=True)):
        for key, value in a.items():
            if key == "extra":
                for ek, ev in value.items():
                    diffs.extend(_compare_scalar(f"step{i} extra.{ek}", ek, ev, b["extra"], prefix))
            else:
                diffs.extend(_compare_scalar(f"step{i} {key}", key, value, b, prefix))
    return diffs


def _compare_params(
    want_params: dict[str, Any], got_params: dict[str, Any], prefix: str
) -> list[str]:
    """Post-step parameter tensors, elementwise at the same tolerance."""
    diffs: list[str] = []
    for pname, pvals in want_params.items():
        if pname not in got_params:
            diffs.append(f"{prefix}parameter {pname}: missing from the fresh run")
            continue
        a_arr = np.asarray(pvals, dtype=np.float64)
        b_arr = np.asarray(got_params[pname], dtype=np.float64)
        if a_arr.shape != b_arr.shape:
            diffs.append(f"{prefix}parameter {pname}: shape {a_arr.shape} != {b_arr.shape}")
            continue
        if not np.allclose(b_arr, a_arr, rtol=GOLDEN_RTOL, atol=GOLDEN_ATOL):
            rel = float((np.abs(b_arr - a_arr) / np.maximum(np.abs(a_arr), 1e-30)).max())
            diffs.append(
                f"{prefix}parameter {pname} differs after the update "
                f"(max rel {rel:.2e} > {GOLDEN_RTOL:.0e})"
            )
    return diffs


def diff_against_committed(fixture: dict[str, Any], committed: dict[str, Any]) -> list[str]:
    """Scenario-by-scenario :func:`compare_expected` over the whole fixture."""
    diffs: list[str] = []
    fresh_by_name = {sc["name"]: sc for sc in fixture["scenarios"]}
    for want in committed["scenarios"]:
        name = want["name"]
        got = fresh_by_name.get(name)
        if got is None:
            diffs.append(f"{name}: scenario missing from the fresh run")
            continue
        diffs.extend(compare_expected(want["expected"], got["expected"], label=name))
    return diffs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Re-run the scenarios and compare against the committed fixture "
        "instead of overwriting it. Exit 1 on any difference.",
    )
    args = parser.parse_args(argv)

    fixture = build_fixture()
    if args.check:
        if not OUT_PATH.is_file():
            print(f"FAIL: {OUT_PATH} does not exist")
            return 1
        committed = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        diffs = diff_against_committed(fixture, committed)
        if diffs:
            print(f"FAIL: current code does not reproduce {OUT_PATH}")
            for d in diffs[:20]:
                print(f"  {d}")
            return 1
        print(f"OK: current code reproduces {OUT_PATH} ({len(committed['scenarios'])} scenarios)")
        return 0
    if OUT_PATH.is_file():
        print(
            f"REFUSING to overwrite {OUT_PATH}: it is the pre-merge evidence "
            "(see the module docstring). Delete it explicitly if you really "
            "intend to re-freeze the update rule's numerics."
        )
        return 1
    text = json.dumps(fixture, indent=1) + "\n"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(fixture['scenarios'])} scenarios, {len(text)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
