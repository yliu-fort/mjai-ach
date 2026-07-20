"""Game registry: discover playable games (AGENTS.md §1 D10, §4)."""

from __future__ import annotations

from dataclasses import dataclass

from mjai.config.game_config import load_all_game_configs


@dataclass(frozen=True)
class GameEntry:
    """One playable game in the CLI's picker."""

    name: str  # short key
    game_string: str  # canonical pyspiel string
    notes: str  # one-line summary for the menu


def list_games() -> list[GameEntry]:
    """All games discovered from configs/games/*.yaml, sorted by name."""
    configs = load_all_game_configs()
    entries = []
    for name, cfg in configs.items():
        # Use the first line of notes as the menu blurb.
        blurb = (cfg.notes.split("\n", 1)[0]).strip() if cfg.notes else ""
        entries.append(GameEntry(name=name, game_string=cfg.game_string, notes=blurb))
    return sorted(entries, key=lambda e: e.name)
