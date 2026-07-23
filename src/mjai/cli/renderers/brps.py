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

    def render_public(self, state: pyspiel.State) -> str:
        """Public-only view: just announces a move is pending, reveals nothing.

        BRPS is one-shot simultaneous with no private state, so the public view
        is simply 'moves pending' — no choice is revealed until the joint
        outcome resolves.
        """
        if state.is_terminal():
            return self.render_terminal(state)
        return "Biased RPS — public view.\nMoves pending (choices hidden until resolved)."

    def render_terminal(self, state: pyspiel.State) -> str:
        # Joint action was applied; we cannot recover it from the terminal state
        # for a one-shot matrix game, so just report the payoff.
        ret = state.returns()
        return f"Round over. Returns: seat 0 = {ret[0]:+.0f}, seat 1 = {ret[1]:+.0f}."
