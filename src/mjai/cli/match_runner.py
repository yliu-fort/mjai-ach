"""Match runner: drive one human-vs-policy or policy-vs-policy match (Step 9).

Owns the env-stepping loop and dispatches to the per-game renderer + parser.
Three modes:
  - interactive:   at least one seat is human; render + prompt each human turn.
  - auto_fast:     both seats are policies; print only the final result.
  - auto_step:     both seats are policies; render every step, pause for Enter.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import pyspiel

from mjai.agents.base import Policy
from mjai.agents.tabular import TabularPolicy, uniform_tabular
from mjai.cli.interfaces import GameRenderer, HumanInputParser
from mjai.games.loader import GameSpec

Seat = Policy | str  # a Policy, or the literal "human"


@dataclass
class MatchResult:
    returns: list[float]
    n_steps: int


class MatchRunner:
    """Runs one match between two seats on ``spec``.

    Args:
        spec: the loaded game.
        renderer, parser: per-game renderer/parser instances.
        seats: length-2 list of Seat (Policy or "human"). Seat 0 = "X"/first.
        input_fn: callable that reads a line of stdin (default: input). Tests
            inject a fake.
        output_fn: callable that writes a string to "stdout" (default: print).
            Tests inject a fake.
        rng: for chance-node sampling.
    """

    def __init__(
        self,
        spec: GameSpec,
        renderer: GameRenderer,
        parser: HumanInputParser,
        seats: list[Seat],
        *,
        input_fn: Callable[[], str] = input,
        output_fn: Callable[[str], None] = print,
        rng: random.Random | None = None,
    ) -> None:
        if len(seats) != 2:
            raise ValueError(f"Need exactly 2 seats, got {len(seats)}")
        self.spec = spec
        self.renderer = renderer
        self.parser = parser
        self.seats = seats
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.rng = rng or random.Random()

    def run(self, mode: str = "interactive") -> MatchResult:
        state = self.spec.new_state()
        n_steps = 0
        while not state.is_terminal():
            if state.is_chance_node():
                self._sample_chance(state)
                continue
            if state.is_simultaneous_node():
                joint = self._simultaneous_step(state)
                state.apply_actions(joint)
            else:
                p = state.current_player()
                a = self._one_step(state, p, mode)
                state.apply_action(a)
            n_steps += 1
            if mode == "auto_step" and not state.is_terminal():
                self.output_fn(self.renderer.render(state, observer_player=None))
                self.output_fn("[press Enter to continue]")
                self.input_fn()
        self.output_fn(self.renderer.render_terminal(state))
        return MatchResult(returns=list(state.returns()), n_steps=n_steps)

    def _one_step(self, state: pyspiel.State, player: int, mode: str) -> int:
        seat = self.seats[player]
        legal = list(state.legal_actions(player))
        if mode == "auto_fast":
            return self._policy_action(seat, state, player, legal)
        # interactive or auto_step with a human seat: render first.
        if seat == "human":
            self.output_fn(self.renderer.render(state, observer_player=player))
            return self._human_action(legal, player)
        # policy seat in interactive/auto_step: render the policy's view too.
        self.output_fn(self.renderer.render(state, observer_player=player))
        return self._policy_action(seat, state, player, legal)

    def _simultaneous_step(self, state: pyspiel.State) -> list[int]:
        # Blind entry for humans: prompt each human on their own info only,
        # never revealing the other's simultaneous choice (AGENTS.md §4).
        joint: list[int] = []
        for p in range(self.spec.num_players):
            seat = self.seats[p]
            legal = list(state.legal_actions(p))
            if seat == "human":
                self.output_fn(self.renderer.render(state, observer_player=p))
                joint.append(self._human_action(legal, p))
            else:
                # In simultaneous mode the policy plays "blind" too (no render
                # of the opponent); just act.
                joint.append(self._policy_action(seat, state, p, legal))
        return joint

    def _human_action(self, legal: list[int], player: int) -> int:
        while True:
            raw = self.input_fn_with_prompt(legal, player)
            try:
                return self.parser.parse(raw, legal)
            except ValueError as e:
                self.output_fn(f"  invalid: {e}")

    def input_fn_with_prompt(self, legal: list[int], player: int) -> str:
        prompt = self.parser.prompt(legal, player)
        self.output_fn(prompt)
        return self.input_fn()

    def _policy_action(
        self, seat: Seat, state: pyspiel.State, player: int, legal: list[int]
    ) -> int:
        if isinstance(seat, str):  # the only str value is "human"
            raise RuntimeError("policy seat is 'human'")
        obs = self.spec.obs_tensor(state, player)
        action, _ = seat.act(obs, legal, eval=True)
        return int(action)

    def _sample_chance(self, state: pyspiel.State) -> None:
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes, strict=True)
        idx = self.rng.choices(range(len(actions)), weights=probs, k=1)[0]
        state.apply_action(actions[idx])


def random_policy_for(spec: GameSpec, *, seed: int = 0) -> Policy:
    """A uniform-random tabular policy of the right size; fixture for auto mode."""
    return uniform_tabular(spec.num_actions, seed=seed)


def _tabular_fixture(spec: GameSpec) -> TabularPolicy:
    return TabularPolicy(num_actions=spec.num_actions, seed=0)
