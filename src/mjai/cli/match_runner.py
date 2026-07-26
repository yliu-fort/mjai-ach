"""Match runner: drive one human-vs-policy or policy-vs-policy match (Step 9).

Owns the env-stepping loop and dispatches to the per-game renderer + parser.
Three modes:
  - interactive:   at least one seat is human; render + prompt each human turn.
  - auto_fast:     every seat is a policy; print only the final result.
  - auto_step:     every seat is a policy; render every step, pause for Enter.

**Seat count is the game's, not 2** (AGENTS.md D13). The runner originally
hard-rejected anything but two seats; 3p Kuhn is the Phase-B decision gate of
the pACH programme, so the count now comes from ``spec.num_players`` and the
terminal/announce paths report every seat rather than a winner/loser pair.
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
    """Runs one match between ``spec.num_players`` seats on ``spec``.

    Args:
        spec: the loaded game.
        renderer, parser: per-game renderer/parser instances.
        seats: one Seat (Policy or "human") per player, in seat order. Seat 0 =
            "X"/first. The length must equal ``spec.num_players``; a mismatch
            raises rather than truncating or padding (AGENTS.md §11).
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
        if len(seats) != spec.num_players:
            raise ValueError(
                f"{spec.name} needs exactly {spec.num_players} seats, got {len(seats)}"
            )
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
                # Announce a robot's sequential move so a spectating human sees
                # what happened (MR2). Skipped for human moves (already shown)
                # and for auto_fast (no per-step output by design).
                if mode != "auto_fast" and self.seats[p] != "human":
                    self._announce(state, p, a)
            n_steps += 1
            if mode == "auto_step" and not state.is_terminal():
                # auto_step has no human (enforced in main.py); render_public is
                # the consistent public view (render(None) would fall back to
                # current_player() and leak on imperfect-info games).
                self.output_fn(self.renderer.render_public(state))
                self.output_fn("[press Enter to continue]")
                self.input_fn()
        self.output_fn(self.renderer.render_terminal(state))
        return MatchResult(returns=list(state.returns()), n_steps=n_steps)

    def _one_step(self, state: pyspiel.State, player: int, mode: str) -> int:
        seat = self.seats[player]
        legal = list(state.legal_actions(player))
        if mode == "auto_fast":
            return self._policy_action(seat, state, player, legal)
        # interactive or auto_step.
        if seat == "human":
            # Human acts from their own view (private info filtered to them).
            self.output_fn(self.renderer.render(state, observer_player=player))
            return self._human_action(legal, player)
        # Robot (policy) seat: render the PUBLIC view only. Rendering the
        # robot's own view (observer_player=player) would leak the opponent's
        # private info to any spectating human (MR1, INV-1). The public view
        # shows pot/board/history but no player's private card/die/hand.
        self.output_fn(self.renderer.render_public(state))
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

    def _announce(self, state: pyspiel.State, player: int, action: int) -> None:
        """Print a one-line description of a robot's move (MR2).

        Lets a spectating human follow the game without seeing the robot's
        private info. Uses OpenSpiel's own ``action_to_string`` so the label is
        always correct (no per-game encoding in the runner).
        """
        try:
            desc = state.action_to_string(player, action)
        except Exception:  # pragma: no cover - defensive; pyspiel rarely fails here
            desc = f"action {action}"
        self.output_fn(f"Player {player} (robot): {desc}")

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
