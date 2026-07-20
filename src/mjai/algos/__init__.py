"""Algorithms layer.

Update rules (``PPOUpdate``, ``ACHUpdate``) consume transition batches and
return a loss + stats; the ``Trainer`` composes a Policy + UpdateRule + a
self-play controller. Adding an algorithm = new ``UpdateRule`` subclass; no
edits to Trainer or pipeline (AGENTS.md §4).

May import :mod:`mjai.agents`, :mod:`mjai.games`, :mod:`mjai.config`,
:mod:`mjai.utils`.
"""
