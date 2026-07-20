"""Canonical OpenSpiel game loader + :class:`GameSpec` (AGENTS.md §4).

Wraps :func:`pyspiel.load_game` so that every game in the project is constructed
from a single, validated source. The seven Phase-1 games are registered in
:data:`GAME_STRINGS`; a YAML file under ``configs/games/`` may register more.

The loader **auto-selects the observation encoding**: it prefers
``information_state_tensor`` (perfect recall, what CFR/NFSP need) but falls back
to ``observation_tensor`` when the game does not provide one (e.g. tic-tac-toe,
oshi-zumo). This matches OpenSpiel's own ``rl_environment`` choice and keeps the
agent interface uniform across all games.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyspiel

# The seven Phase-1 games (AGENTS.md §1 D8). Keys are the short names used in
# configs and the CLI; values are the canonical pyspiel game strings, verified
# against open-spiel 2.0.1. (A non-biased RPS can still be loaded on the fly
# via ``load_game_by_string("matrix_rps")`` but is not part of the canonical 7.)
GAME_STRINGS: dict[str, str] = {
    "brps": "matrix_brps",
    "kuhn": "kuhn_poker",
    "leduc": "leduc_poker",
    "ttt": "tic_tac_toe",
    "goofspiel5_ii": "goofspiel(imp_info=True,num_cards=5,points_order=descending)",
    "liars_dice1": "liars_dice(numdice=1,dice_sides=6)",
    "oshi_zumo": "oshi_zumo(coins=5,size=3,horizon=20)",
}


class GameLoadError(ValueError):
    """Raised when a game string fails to load or lacks required capabilities."""


@dataclass(frozen=True)
class GameSpec:
    """Static description of a loaded game, sufficient to build agents/policies.

    Attributes:
        name: short key (e.g. ``"kuhn"``).
        game_string: the canonical pyspiel string this was loaded from.
        game: the loaded :class:`pyspiel.Game`.
        num_players, num_actions, max_game_length: from the game.
        obs_kind: which tensor the loader selected — ``"information_state"`` or
            ``"observation"``. Drives :meth:`obs_tensor` / :meth:`obs_size`.
        obs_size: length of the selected per-player observation vector.
        is_simultaneous: True for simultaneous-move games (joint actions).
        is_perfect_info: True if the game has no hidden information.
        is_zero_sum: True for zero-sum games.
    """

    name: str
    game_string: str
    game: pyspiel.Game
    num_players: int
    num_actions: int
    max_game_length: int
    obs_kind: str
    obs_size: int
    is_simultaneous: bool
    is_perfect_info: bool
    is_zero_sum: bool

    def new_state(self) -> pyspiel.State:
        """Fresh initial state."""
        return self.game.new_initial_state()

    def obs_tensor(self, state: pyspiel.State, player: int) -> list[float]:
        """The selected per-player observation vector at ``state``."""
        if self.obs_kind == "information_state":
            vec = state.information_state_tensor(player)
        else:
            vec = state.observation_tensor(player)
        return [float(x) for x in vec]  # coerce; pyspiel returns an untyped list

    def __repr__(self) -> str:
        return (
            f"GameSpec({self.name!r}, A={self.num_actions}, L={self.max_game_length}, "
            f"obs={self.obs_kind}[{self.obs_size}], sim={self.is_simultaneous})"
        )


def _select_obs_kind(game: pyspiel.Game) -> str:
    """Prefer information_state_tensor; fall back to observation_tensor."""
    t = game.get_type()
    if t.provides_information_state_tensor:
        return "information_state"
    if t.provides_observation_tensor:
        return "observation"
    raise GameLoadError(
        "Game provides neither information_state_tensor nor observation_tensor; "
        "cannot build an agent interface. Add a custom adapter (AGENTS.md §4)."
    )


def _probe_obs_size(game: pyspiel.Game, kind: str) -> int:
    """Measure the per-player observation vector length on an initial state.

    Some games only populate the tensor after chance is resolved (e.g. the
    private card is dealt), but the *size* is constant per game; the initial
    state's tensor has the right length regardless.
    """
    state = game.new_initial_state()
    if kind == "information_state":
        return len(state.information_state_tensor(0))
    return len(state.observation_tensor(0))


def _short_name(game_string: str) -> str:
    """Strip any ``(params)`` suffix to get the bare game short name."""
    i = game_string.find("(")
    return game_string[:i] if i >= 0 else game_string


def _parse_params(game_string: str) -> dict[str, Any]:
    """Parse the ``key=value,...`` params baked into a parenthesized game string.

    Returns ``{}`` if there are no parens. Values are converted to int/float/bool
    when they look numeric or boolean; otherwise kept as strings (matching
    pyspiel's own parser for the common cases we use).

    Rationale: ``pyspiel.load_game(name, params_dict)`` **replaces** rather than
    merges with params in the string, so to override one param of a registered
    game while keeping the others we must merge ourselves.
    """
    i = game_string.find("(")
    if i < 0:
        return {}
    body = game_string[i + 1 : game_string.rfind(")")]
    out: dict[str, Any] = {}
    for raw_chunk in body.split(","):
        chunk = raw_chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        k, v = k.strip(), v.strip()
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        elif v.lstrip("-").isdigit():
            out[k] = int(v)
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v  # leave as string (pyspiel accepts bare strings)
    return out


def load_game(name: str, **overrides: Any) -> GameSpec:
    """Load a registered game by short name, with optional parameter overrides.

    Args:
        name: key in :data:`GAME_STRINGS` (e.g. ``"kuhn"``).
        **overrides: forwarded to ``pyspiel.load_game`` as a parameter dict,
            merged on top of any params baked into the registered game string.
            Most games need none; pass them only when overriding a default
            (e.g. ``load_game("oshi_zumo", coins=10)``).

    Raises:
        GameLoadError: if the name is unknown, the string fails to load, or the
            game lacks both observation encodings.
    """
    if name not in GAME_STRINGS:
        known = ", ".join(sorted(GAME_STRINGS))
        raise GameLoadError(f"Unknown game {name!r}. Known: {known}.")
    game_string = GAME_STRINGS[name]
    try:
        if overrides:
            # Merge baked-in params with the caller's overrides (caller wins),
            # then load via short name + merged dict. pyspiel.load_game requires
            # the bare short name when a params dict is supplied.
            merged = {**_parse_params(game_string), **overrides}
            game = pyspiel.load_game(_short_name(game_string), merged)
        else:
            game = pyspiel.load_game(game_string)
    except Exception as e:  # pyspiel raises SpielError, a subclass of Exception
        raise GameLoadError(f"Failed to load {name!r} ({game_string!r}): {e}") from e

    kind = _select_obs_kind(game)
    t = game.get_type()
    # GameType enum members (Dynamics / Information / Utility) are nested classes
    # not exposed at the pyspiel top level in 2.0.x, so compare by .name string.
    return GameSpec(
        name=name,
        game_string=game_string,
        game=game,
        num_players=game.num_players(),
        num_actions=game.num_distinct_actions(),
        max_game_length=game.max_game_length(),
        obs_kind=kind,
        obs_size=_probe_obs_size(game, kind),
        is_simultaneous=t.dynamics.name == "SIMULTANEOUS",
        is_perfect_info=t.information.name == "PERFECT_INFORMATION",
        is_zero_sum=t.utility.name in ("ZERO_SUM", "CONSTANT_SUM"),
    )


def all_game_names() -> list[str]:
    """Sorted list of registered game short names."""
    return sorted(GAME_STRINGS)


def load_game_by_string(game_string: str) -> GameSpec:
    """Load an arbitrary pyspiel game string (not necessarily registered).

    Use for one-off games outside the canonical 7 (e.g. ``"matrix_rps"`` for a
    non-biased RPS sanity check). Registered games should go through
    :func:`load_game` so they appear in the CLI and configs.
    """
    try:
        game = pyspiel.load_game(game_string)
    except Exception as e:
        raise GameLoadError(f"Failed to load {game_string!r}: {e}") from e
    kind = _select_obs_kind(game)
    t = game.get_type()
    return GameSpec(
        name=_short_name(game_string),
        game_string=game_string,
        game=game,
        num_players=game.num_players(),
        num_actions=game.num_distinct_actions(),
        max_game_length=game.max_game_length(),
        obs_kind=kind,
        obs_size=_probe_obs_size(game, kind),
        is_simultaneous=t.dynamics.name == "SIMULTANEOUS",
        is_perfect_info=t.information.name == "PERFECT_INFORMATION",
        is_zero_sum=t.utility.name in ("ZERO_SUM", "CONSTANT_SUM"),
    )
