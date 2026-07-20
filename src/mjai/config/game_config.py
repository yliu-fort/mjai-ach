"""Game-config loading (AGENTS.md §9).

Reads ``configs/games/<name>.yaml`` into a frozen :class:`GameConfig` dataclass
and resolves it to a live :class:`mjai.games.loader.GameSpec`. The YAML carries
provenance/notes; the canonical game string lives there too so adding a game is
a one-file change (plus renderer/parser for the CLI per AGENTS.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from mjai.games.loader import (
    GAME_STRINGS,
    GameSpec,
    load_game,
    load_game_by_string,
)

# Default location of game configs, relative to the repo root.
DEFAULT_GAMES_DIR = Path(__file__).resolve().parents[3] / "configs" / "games"


@dataclass(frozen=True)
class GameConfig:
    """The parsed contents of a game YAML."""

    name: str
    game_string: str
    notes: str = ""


def load_game_config(path: str | Path) -> GameConfig:
    """Parse a single game YAML file into a :class:`GameConfig`."""
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    if "name" not in data or "game_string" not in data:
        raise ValueError(f"{path}: YAML must define 'name' and 'game_string'")
    return GameConfig(
        name=str(data["name"]),
        game_string=str(data["game_string"]),
        notes=str(data.get("notes", "")).strip(),
    )


def load_all_game_configs(games_dir: str | Path | None = None) -> dict[str, GameConfig]:
    """Load every game YAML under ``games_dir`` (default configs/games/).

    Returns ``{name: GameConfig}``. Unknown YAML keys are ignored; missing
    required keys raise per :func:`load_game_config`.
    """
    d = Path(games_dir) if games_dir else DEFAULT_GAMES_DIR
    out: dict[str, GameConfig] = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.yaml")):
        cfg = load_game_config(p)
        out[cfg.name] = cfg
    return out


def resolve_to_spec(cfg: GameConfig, **overrides: object) -> GameSpec:
    """Build a :class:`GameSpec` from a :class:`GameConfig`.

    The config's ``game_string`` must match a registered short name (its
    parenthesized form is allowed and parsed); if not, fall back to the
    canonical registered string for that name.
    """
    if cfg.name in GAME_STRINGS:
        return load_game(cfg.name, **overrides)
    # Unregistered game: load directly by its string.
    return load_game_by_string(cfg.game_string)
