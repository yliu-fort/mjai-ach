"""Eval layer: metrics over a trained run.

exploitability / NashConv / exact-Nash (via OpenSpiel), cross-play payoff
matrices, worst-case vs pool, forgetting metric, non-transitivity detection,
training curves (TensorBoard). Each metric is one module exposing a function
taking a run directory (AGENTS.md §4).

May import :mod:`mjai.algos`, :mod:`mjai.agents`, :mod:`mjai.games`,
:mod:`mjai.config`, :mod:`mjai.utils`.
"""
