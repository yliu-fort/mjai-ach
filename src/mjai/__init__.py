"""mjai-ach: IMPALA-style PPO/ACH + league-play research pipeline.

See AGENTS.md for the governance contract. This package follows a strict layered
architecture (enforced by import-linter); import only what you need from the
sub-packages rather than the top level, to keep dependencies explicit.

Public version. Phase 1: tabular + small-MLP on 7 small games.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
