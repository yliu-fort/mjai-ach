"""Sequence-form machinery for the pACH programme (AGENTS.md D12).

Layer: directly above ``mjai.games``, below ``mjai.config``. Imports games,
agents and utils only — never an algorithm, never an evaluator.

What lives here: per-seat sequence enumeration, the realization-plan operator
x(theta), the multilinear terminal-payoff tensor, the exact best-response
operator and NashConv. Everything is float64 torch and autograd-differentiable,
because the research plan's Oracle track differentiates straight through the
exact expectation (研究计划 §3.3).
"""
