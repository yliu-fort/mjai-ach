"""Config layer: frozen dataclasses loaded from YAML via cattrs.

No runtime logic beyond dataclass definitions; loaded by the runner and dumped
into each run directory for reproducibility (AGENTS.md §9). May import only
:mod:`mjai.utils`.
"""
