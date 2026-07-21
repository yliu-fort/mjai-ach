"""Dict-backed tabular policy + value (AGENTS.md §1 D5, §3).

Implements the same :class:`mjai.agents.base.Policy` interface as the NN, so the
same Trainer/UpdateRule can drive either. Used for:
  - the small-game Phase-1 experiments (Kuhn, BRPS, Liar's-Dice-1, ...),
  - the unit-test fixtures (deterministic, no torch),
  - validation against CFR / exact Nash (Step 3).

State representation: the observation vector is hashed to a bytes key so the
dict can key on it. Two states with the same observation vector share a row
(this is the imperfect-recall abstraction OpenSpiel's info-state gives us).
"""

from __future__ import annotations

import base64
import copy
import json
import math
import pickle
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mjai.agents.base import Policy, masked_softmax


def _obs_to_key(obs: list[float]) -> bytes:
    """Stable hashable key for an observation vector.

    Tuples-of-floats are hashable but slow to compare and risk float-equality
    surprises; we round to 9 decimals and freeze to bytes so identical
    observations collide and distinct ones (after rounding) do not.
    """
    return b"|".join(f"{round(x, 9):.9f}".encode() for x in obs)


class TabularPolicy(Policy):
    """Per-observation-row logits over the action space + a scalar value.

    The full action space has ``num_actions`` slots; illegal actions are masked
    to ``-inf`` at act/logits time. The ``temperature`` controls exploration
    during stochastic sampling (higher = more uniform).

    Attributes:
        num_actions: size of the (fixed) action space for the game.
        logits: ``{obs_key: [logit_per_action]}``; lazily zero-initialized.
        values: ``{obs_key: float}``; lazily zero-initialized.
        temperature: softmax temperature for stochastic sampling.
        rng: random.Random for reproducible sampling.
    """

    def __init__(
        self,
        num_actions: int,
        *,
        temperature: float = 1.0,
        seed: int | None = None,
        init_logit_std: float = 0.0,
    ) -> None:
        if num_actions <= 0:
            raise ValueError(f"num_actions must be positive, got {num_actions}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.num_actions = num_actions
        self.temperature = temperature
        self.init_logit_std = init_logit_std
        self.rng = random.Random(seed)
        self.logits: dict[bytes, list[float]] = {}
        self.values: dict[bytes, float] = {}

    # ---- row accessors (used by UpdateRules to read/write in place) ----

    def _row(self, table: dict[bytes, list[float]], key: bytes) -> list[float]:
        row = table.get(key)
        if row is None:
            if self.init_logit_std > 0:
                row = [self.rng.gauss(0.0, self.init_logit_std) for _ in range(self.num_actions)]
            else:
                row = [0.0] * self.num_actions
            table[key] = row
        return row

    def get_logits(self, obs: list[float]) -> list[float]:
        """Mutable reference to the logits row for ``obs`` (creates if missing)."""
        return self._row(self.logits, _obs_to_key(obs))

    def get_value(self, obs: list[float]) -> float:
        key = _obs_to_key(obs)
        if key not in self.values:
            self.values[key] = 0.0
        return self.values[key]

    # ---- Policy interface ----

    def action_logits(self, obs: list[float], legal_actions: list[int]) -> list[float]:
        row = self.get_logits(obs)
        return [row[a] for a in legal_actions]

    def _probs_over_full_space(self, obs: list[float], legal_actions: list[int]) -> list[float]:
        row = self.get_logits(obs)
        mask = [a in legal_actions for a in range(self.num_actions)]
        # Apply temperature: divide logits by T before softmax.
        scaled = [lg / self.temperature for lg in row]
        return masked_softmax(scaled, mask)

    def act(
        self,
        obs: list[float],
        legal_actions: list[int],
        *,
        eval: bool = False,
        rng_key: Any = None,
    ) -> tuple[int, float]:
        if not legal_actions:
            raise ValueError("legal_actions must be non-empty")
        probs = self._probs_over_full_space(obs, legal_actions)
        if eval:
            # Greedy: pick the legal action with highest probability.
            best_a = max(legal_actions, key=lambda a: probs[a])
            # Break ties deterministically by smallest action id.
            best_p = probs[best_a]
            for a in legal_actions:
                if abs(probs[a] - best_p) < 1e-12 and a < best_a:
                    best_a = a
            return best_a, 0.0
        # Stochastic: sample from the legal-action categorical.
        legal_probs = [probs[a] for a in legal_actions]
        chosen = self._sample_categorical(legal_actions, legal_probs)
        lp = math.log(probs[chosen] + 1e-30)
        return chosen, lp

    def value(self, obs: list[float]) -> float:
        return self.get_value(obs)

    def act_with_value(
        self,
        obs: list[float],
        legal_actions: list[int],
        *,
        eval: bool = False,
        rng_key: Any = None,
    ) -> tuple[int, float, float]:
        """Fused ``act`` + ``value`` for the rollout hot path (AGENTS.md §8).

        Semantically identical to calling :meth:`act` then :meth:`value`, but
        computes the masked probability vector only once (``act`` alone already
        pays for it; ``value`` is a dict lookup). Consumes exactly one RNG draw
        in the stochastic branch — matching :meth:`act` so the rollout's RNG
        stream is unchanged from the act+value path.
        """
        if not legal_actions:
            raise ValueError("legal_actions must be non-empty")
        probs = self._probs_over_full_space(obs, legal_actions)
        v = self.get_value(obs)
        if eval:
            best_a = max(legal_actions, key=lambda a: probs[a])
            best_p = probs[best_a]
            for a in legal_actions:
                if abs(probs[a] - best_p) < 1e-12 and a < best_a:
                    best_a = a
            return best_a, 0.0, v
        legal_probs = [probs[a] for a in legal_actions]
        chosen = self._sample_categorical(legal_actions, legal_probs)
        lp = math.log(probs[chosen] + 1e-30)
        return chosen, lp, v

    def _sample_categorical(self, actions: list[int], probs: list[float]) -> int:
        r = self.rng.random()
        cum = 0.0
        for a, p in zip(actions, probs, strict=True):
            cum += p
            if r <= cum:
                return a
        return actions[-1]  # float-rounding fallback

    # ---- persistence ----

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "tabular",
            "num_actions": self.num_actions,
            "temperature": self.temperature,
            "logits_b64": _pickle_b64(self.logits),
            "values_b64": _pickle_b64(self.values),
        }
        if p.suffix == ".pkl":
            p.write_bytes(pickle.dumps(payload))
        else:
            p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: str) -> None:
        p = Path(path)
        if p.suffix == ".pkl":
            # Trusted local checkpoint only (written by our own TabularPolicy.save).
            payload = pickle.loads(p.read_bytes())
        else:
            payload = json.loads(p.read_text(encoding="utf-8"))
        if payload.get("kind") != "tabular":
            raise ValueError(f"Not a tabular policy checkpoint: kind={payload.get('kind')!r}")
        self.num_actions = payload["num_actions"]
        self.temperature = payload["temperature"]
        self.logits = _unpickle_b64(payload["logits_b64"])
        self.values = _unpickle_b64(payload["values_b64"])

    # ---- in-memory snapshot / restore (single source of truth for hub+league) ----
    #
    # ParameterHub and LeagueManager call these instead of reaching into our
    # logits/values dicts. The snapshot is a deep copy so the caller may mutate
    # us freely without affecting stored snapshots (AGENTS.md §8 memory safety).

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "kind": "tabular",
            "logits": copy.deepcopy(self.logits),
            "values": copy.deepcopy(self.values),
        }

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        if snapshot.get("kind") != "tabular":
            raise ValueError(
                f"Snapshot kind mismatch: expected 'tabular', got {snapshot.get('kind')!r}"
            )
        self.logits = copy.deepcopy(snapshot["logits"])
        self.values = copy.deepcopy(snapshot["values"])

    # ---- diagnostics ----

    def num_rows(self) -> int:
        return len(self.logits)

    def reset(self) -> None:
        self.logits.clear()
        self.values.clear()


def _pickle_b64(obj: Any) -> str:
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")


def _unpickle_b64(s: str) -> Any:
    # Trusted local checkpoint only (written by our own _pickle_b64).
    return pickle.loads(base64.b64decode(s))


def uniform_tabular(num_actions: int, *, seed: int | None = None) -> TabularPolicy:
    """A tabular policy that plays uniformly at random (baseline / test fixture).

    Built by zeroing all logits; masked_softmax then yields uniform over legal.
    """
    return TabularPolicy(num_actions=num_actions, temperature=1.0, seed=seed)


def average_rows(rows: Iterable[list[float]]) -> list[float]:
    """Element-wise mean of action-logit rows (used by averaging baselines)."""
    rows = list(rows)
    if not rows:
        return []
    n = len(rows)
    return [sum(col) / n for col in zip(*rows, strict=True)]
