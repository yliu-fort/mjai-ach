"""MLP shared actor-critic (AGENTS.md §1 D1, §3).

The neural counterpart to :class:`mjai.agents.tabular.TabularPolicy`: same
:class:`mjai.agents.base.Policy` interface, torch-backed. A shared torso feeds
a policy head (logits over the full action space; illegal actions masked to
-1e9 before softmax) and a scalar value head.

Device handling (AGENTS.md §1 D6): the constructor calls
:func:`mjai.utils.gpu_assert.resolve_device`, which raises unless GPU is
available or CPU was explicitly requested. To construct an MLP for CPU-only
unit tests, call :func:`mjai.utils.gpu_assert.require_cpu` first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mjai.agents.base import Policy
from mjai.utils.gpu_assert import resolve_device

# Logits assigned to illegal actions before softmax. Large-negative (not -inf)
# so gradients stay finite if an illegal action somehow leaks into a loss.
MASK_VALUE = -1e9

# Activation registry keyed by lowercase name (e.g. "relu"). Single source of
# truth shared by the experiment runner (build by config string) and the
# checkpoint factory (rebuild from the sidecar's recorded name). An activation
# missing here fails loudly at reconstruction time — never silently swapped.
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


class MLPSharedActorCritic(nn.Module, Policy):
    """Shared-torso MLP with separate policy and value heads.

    Args:
        obs_size: length of the per-player observation vector (GameSpec.obs_size).
        num_actions: size of the (fixed) action space (GameSpec.num_actions).
        hidden_sizes: widths of the shared torso layers.
        activation: module constructor for hidden activations (default Tanh).
        trunk_layernorm: append a LayerNorm to the end of the torso. This bounds
            the feature scale feeding both heads, which in turn bounds the logit
            scale — giving an absolute meaning to a logit threshold like ACH's
            ``l_th`` without mean-centering. Default False (historical
            architecture); see docs/reproduce_report.md.
        device: explicit device override; if None, resolved via gpu_assert.
        seed: torch RNG seed for reproducible init.
    """

    def __init__(
        self,
        obs_size: int,
        num_actions: int,
        *,
        hidden_sizes: tuple[int, ...] = (128, 128),
        activation: type[nn.Module] = nn.Tanh,
        trunk_layernorm: bool = False,
        device: str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if obs_size <= 0:
            raise ValueError(f"obs_size must be positive, got {obs_size}")
        if num_actions <= 0:
            raise ValueError(f"num_actions must be positive, got {num_actions}")

        self.obs_size = obs_size
        self.num_actions = num_actions
        if seed is not None:
            torch.manual_seed(seed)
        self._init_seed = seed

        # Shared torso.
        layers: list[nn.Module] = []
        last = obs_size
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(activation())
            last = h
        if trunk_layernorm:
            layers.append(nn.LayerNorm(last))
        self.torso = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last, num_actions)
        self.value_head = nn.Linear(last, 1)
        # Kept for save(): the sidecar records the activation name and whether
        # the torso ends in a LayerNorm, so the checkpoint factory can rebuild
        # the exact architecture (F1). A mismatch would fail the weight load.
        self._activation_name = activation.__name__.lower()
        self.trunk_layernorm = trunk_layernorm

        # Device resolution: explicit override wins; else gpu_assert (raises on
        # silent-degradation risk).
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device(resolve_device().device)
        self.to(self.device)

        self._mode_eval = False

    # ---- forward passes ----

    def _forward_features(self, obs: torch.Tensor) -> torch.Tensor:
        feats: torch.Tensor = self.torso(obs)
        return feats

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (policy_logits[A], value[]) for a batched observation tensor."""
        feats = self._forward_features(obs)
        logits = self.policy_head(feats)
        value = self.value_head(feats).squeeze(-1)
        return logits, value

    def _obs_to_tensor(self, obs: list[float]) -> torch.Tensor:
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    # ---- Policy interface ----

    def action_logits(self, obs: list[float], legal_actions: list[int]) -> list[float]:
        with torch.no_grad():
            obs_t = self._obs_to_tensor(obs).unsqueeze(0)  # (1, obs_size)
            logits, _ = self.forward(obs_t)
            return [logits[0, a].item() for a in legal_actions]

    def _masked_log_probs(self, logits: torch.Tensor, legal_actions: list[int]) -> torch.Tensor:
        """Full-space log-probs with illegal actions masked to -inf-probability."""
        mask = torch.full_like(logits, MASK_VALUE)
        for a in legal_actions:
            mask[..., a] = 0.0
        masked = logits + mask
        return torch.log_softmax(masked, dim=-1)

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
        with torch.no_grad():
            obs_t = self._obs_to_tensor(obs).unsqueeze(0)
            logits, _ = self.forward(obs_t)
            log_probs = self._masked_log_probs(logits[0], legal_actions)
            probs = torch.exp(log_probs)
            if eval:
                # Greedy over the legal set only.
                legal_idx = torch.tensor(legal_actions, device=self.device)
                legal_lp = log_probs[legal_idx]
                best = int(legal_idx[torch.argmax(legal_lp)].item())
                return best, 0.0
            # Stochastic sample from the full-space categorical (illegal have ~0 prob).
            action = int(torch.multinomial(probs, num_samples=1).item())
            return action, float(log_probs[action].item())

    def value(self, obs: list[float]) -> float:
        with torch.no_grad():
            obs_t = self._obs_to_tensor(obs).unsqueeze(0)
            _, value = self.forward(obs_t)
            return float(value.item())

    def act_with_value(
        self,
        obs: list[float],
        legal_actions: list[int],
        *,
        eval: bool = False,
        rng_key: Any = None,
    ) -> tuple[int, float, float]:
        """Fused single-forward ``act`` + ``value`` (AGENTS.md §8).

        Hot-path override: the rollout needs (action, logprob, value) at every
        decision point. Calling :meth:`act` then :meth:`value` would do two
        forwards over the same observation; this does one and reuses the logits
        for both the policy sample and the value head. Consumes exactly one
        ``torch.multinomial`` draw in the stochastic branch — matching
        :meth:`act` so the rollout's RNG stream is unchanged.
        """
        if not legal_actions:
            raise ValueError("legal_actions must be non-empty")
        with torch.no_grad():
            obs_t = self._obs_to_tensor(obs).unsqueeze(0)
            logits, value = self.forward(obs_t)
            log_probs = self._masked_log_probs(logits[0], legal_actions)
            probs = torch.exp(log_probs)
            v = float(value.item())
            if eval:
                legal_idx = torch.tensor(legal_actions, device=self.device)
                legal_lp = log_probs[legal_idx]
                best = int(legal_idx[torch.argmax(legal_lp)].item())
                return best, 0.0, v
            action = int(torch.multinomial(probs, num_samples=1).item())
            return action, float(log_probs[action].item()), v

    # ---- train/eval mode hooks ----

    def train(self, mode: bool = True) -> MLPSharedActorCritic:
        """Override nn.Module.train to keep our _mode_eval flag in sync.

        nn.Module.eval() calls ``self.train(False)``, so this signature must
        accept the optional ``mode`` argument.
        """
        super().train(mode)
        self._mode_eval = not mode
        return self

    def eval_mode(self) -> None:
        """Public alias matching the Policy ABC; sets eval mode."""
        self.train(False)

    # ---- persistence ----

    def save(self, path: str) -> None:
        """Save the torch state_dict (+ metadata) to ``path``.

        ``path`` should end in ``.pt``. A sibling ``.pt.meta.json`` is written
        with obs_size/num_actions/hidden_sizes/activation so the checkpoint
        factory can reconstruct the architecture without the caller passing
        it back.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p)
        meta = {
            "obs_size": self.obs_size,
            "num_actions": self.num_actions,
            "hidden_sizes": [
                # Extract hidden widths from the torso (every Linear's out_features).
                m.out_features
                for m in self.torso
                if isinstance(m, nn.Linear)
            ],
            # Registry key (mlp.ACTIVATIONS) for the torso activation. Older
            # sidecars predate this key; the factory then derives it from the
            # run's dumped config.json or fails loudly (no silent default).
            "activation": self._activation_name,
            # Whether the torso ends in a LayerNorm. Older sidecars predate this
            # key; the factory then falls back to the run config and finally to
            # False (the historical architecture), and a wrong guess surfaces as
            # a loud state_dict mismatch rather than silently wrong weights.
            "trunk_layernorm": self.trunk_layernorm,
            "device": str(self.device),
            "init_seed": self._init_seed,
        }
        p.with_suffix(".pt.meta.json").write_text(json.dumps(meta, indent=2))

    def load(self, path: str) -> None:
        p = Path(path)
        state = torch.load(p, map_location=self.device, weights_only=True)
        self.load_state_dict(state)

    # ---- in-memory snapshot / restore (CPU-stored; AGENTS.md §8) ----
    #
    # The IMPALA ParameterHub keeps a bounded history of snapshots; the league
    # pool stores snapshots long-term. If we kept GPU tensors here, every
    # snapshot would pin GPU memory (8x live weights at the hub's default
    # history bound). Instead we clone every parameter to a CPU tensor on
    # snapshot and move it back to the policy's device on restore. Snapshots are
    # independent of ``self`` — later gradient steps do not mutate them.

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "kind": "nn",
            "state_dict": {k: v.detach().to("cpu").clone() for k, v in self.state_dict().items()},
        }

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        if snapshot.get("kind") != "nn":
            raise ValueError(f"Snapshot kind mismatch: expected 'nn', got {snapshot.get('kind')!r}")
        moved = {k: v.to(self.device) for k, v in snapshot["state_dict"].items()}
        self.load_state_dict(moved)
