"""Policy wrapper that routes the value read to an INDEPENDENT critic.

The rollout computes GAE advantages from the value each ``act_with_value`` call
returns. By default that value comes from the policy's own (shared-trunk) value
head. This wrapper swaps in a SEPARATE critic network for that value, so GAE is
computed with the well-trained, non-drifting critic while the policy net keeps
its own (now-unused) value head out of the advantage loop.

Why it exists (AlgoConfig.separate_critic): the shared-trunk n_critic_updates
drifts the policy (training the shared critic moves the trunk -> policy logits),
and on Liar's Dice that drift *raised* exploitability despite a better critic.
A separate net has no such coupling. This wrapper is the clean integration: the
rollout, controller and trainer are unchanged (they just hold a Policy); only
``act_with_value`` / ``value`` / ``__call__`` re-route the value to the critic.

Action selection, logits and persistence delegate to the policy net -- the
critic is a value-only spectator retrained each run, so it is not part of the
snapshot/restore contract. The update rule trains both: the policy via
``parameters()`` (policy net only), the critic via ``critic_net``.
"""

from __future__ import annotations

from typing import Any, Iterator

import torch

from mjai.agents.base import Policy
from mjai.agents.mlp import MLPSharedActorCritic


class PolicyWithCritic(Policy):
    """A policy whose value read (for GAE) comes from a separate critic net."""

    def __init__(self, policy_net: MLPSharedActorCritic, critic_net: MLPSharedActorCritic) -> None:
        self.policy_net = policy_net
        self.critic_net = critic_net

    # ---- identity / introspection (delegate to policy net) ----
    @property
    def obs_size(self) -> int:
        return self.policy_net.obs_size

    @property
    def num_actions(self) -> int:
        return self.policy_net.num_actions

    @property
    def device(self) -> torch.device:
        return self.policy_net.device

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        # The policy optimizer trains the POLICY net only; the critic has its own
        # optimizer in the update rule (via critic_net.parameters()).
        return self.policy_net.parameters()

    def __call__(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # So the update rule's ``self.policy(obs)`` works as for an nn.Module:
        # returns policy logits (policy loss) + CRITIC value (value loss).
        logits = self.policy_net(obs)[0]
        value = self.critic_net(obs)[1]
        return logits, value

    # ---- action API (policy net) ----
    def act(
        self,
        obs: list[float],
        legal_actions: list[int],
        *,
        eval: bool = False,
        rng_key: Any = None,
    ) -> tuple[int, float]:
        return self.policy_net.act(obs, legal_actions, eval=eval, rng_key=rng_key)

    def action_logits(self, obs: list[float], legal_actions: list[int]) -> list[float]:
        return self.policy_net.action_logits(obs, legal_actions)

    def action_logits_batch(self, obs_batch: Any, legal_mask: Any) -> Any:
        return self.policy_net.action_logits_batch(obs_batch, legal_mask)

    # ---- value API (CRITIC net -- the whole point) ----
    def value(self, obs: list[float]) -> float:
        return self.critic_net.value(obs)

    def act_with_value(
        self,
        obs: list[float],
        legal_actions: list[int],
        *,
        eval: bool = False,
        rng_key: Any = None,
    ) -> tuple[int, float, float]:
        # Two forwards (policy for action+logprob, critic for value) -- the cost
        # of decoupling the value from the policy trunk.
        action, logprob = self.policy_net.act(obs, legal_actions, eval=eval, rng_key=rng_key)
        value = self.critic_net.value(obs)
        return action, logprob, value

    # ---- persistence (policy net only; critic is retrained each run) ----
    def save(self, path: str) -> None:
        self.policy_net.save(path)

    def load(self, path: str) -> None:
        self.policy_net.load(path)

    def snapshot_state(self) -> dict[str, Any]:
        return self.policy_net.snapshot_state()

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        self.policy_net.restore_state(snapshot)
