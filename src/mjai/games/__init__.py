"""Games layer (sibling of :mod:`mjai.agents`).

Wraps :mod:`pyspiel` with canonical game strings and per-game YAML configs.
Thin adapters under ``games/adapters/`` are added only when OpenSpiel's
information-state tensor is not directly usable (AGENTS.md §4).

May import only :mod:`mjai.utils`. Sibling imports of :mod:`mjai.agents` are
allowed but discouraged — prefer depending on the abstract Policy interface.
"""
