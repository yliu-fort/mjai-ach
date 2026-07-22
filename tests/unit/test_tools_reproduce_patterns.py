"""Unit tests for the --pattern run-dir glob in the reproduce tools (T3/F2).

The tools live outside the package, so they are imported by path insertion,
mirroring how they self-import ``tb_eval`` at runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import compare_with_paper  # noqa: E402
import summarize_reproduce  # noqa: E402


def test_default_pattern_unchanged_mirror_glob():
    """Behavior invariance: defaults reproduce the old hard-coded glob."""
    assert summarize_reproduce.DEFAULT_PATTERN == "*_ach_mlp_mirror"
    assert compare_with_paper.DEFAULT_PATTERN == "*_ach_mlp_mirror"


def test_pattern_suffix_strips_glob_star():
    assert summarize_reproduce.pattern_suffix("*_ach_mlp_league") == "_ach_mlp_league"
    assert compare_with_paper.pattern_suffix("*_ach_mlp_mirror") == "_ach_mlp_mirror"
    assert summarize_reproduce.pattern_suffix("_ach_mlp_mirror") == "_ach_mlp_mirror"


def test_game_from_dirname_default_matches_legacy_replace():
    """``kuhn_ach_mlp_mirror`` -> ``kuhn``, exactly as the old .replace() did."""
    assert summarize_reproduce.game_from_dirname("kuhn_ach_mlp_mirror") == "kuhn"
    assert summarize_reproduce.game_from_dirname("liars_dice1_ach_mlp_mirror") == "liars_dice1"


def test_game_from_dirname_league_pattern():
    league = "*_ach_mlp_league"
    assert summarize_reproduce.game_from_dirname("kuhn_ach_mlp_league", league) == "kuhn"
    assert summarize_reproduce.game_from_dirname("oshi_zumo_ach_mlp_league", league) == "oshi_zumo"


def test_game_from_dirname_unmatched_dir_passes_through():
    """A dir not ending in the suffix keeps its full name (loud, not silent)."""
    assert summarize_reproduce.game_from_dirname("unexpected") == "unexpected"
