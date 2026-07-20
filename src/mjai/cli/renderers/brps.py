"""Biased RPS renderer: simultaneous one-shot, blind entry (AGENTS.md §4)."""

from __future__ import annotations

import pyspiel

from mjai.cli.interfaces import GameRenderer

_NAMES = {0: "Rock", 1: "Paper", 2: "Scissors"}
_PAYOFF = "Payoff matrix: R=(-25,+25 vs P, +50,-50 vs S), P=(+25,-25 vs R, -5,+5 vs S), S=(-50,+50 vs R, +5,-5 vs P)"


def create() -> GameRenderer:
    return _BRPSRenderer()


class _BRPSRenderer:
    def render(self, state: pyspiel.State, observer_player: int | None) -> str:
        # One-shot simultaneous: pre-action the state is just "choose your move".
        return (
            f"Biased RPS — player {observer_player}, choose your move (blind).\n"
            f"Actions: 0=Rock, 1=Paper, 2=Scissors\n"
            f"{_PAYOFF}"
        )

    def render_terminal(self, state: pyspiel.State) -> str:
        # Joint action was applied; we cannot recover it from the terminal state
        # for a one-shot matrix game, so just report the payoff.
        ret = state.returns()
        return f"Round over. Returns: seat 0 = {ret[0]:+.0f}, seat 1 = {ret[1]:+.0f}."
