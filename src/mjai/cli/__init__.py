"""CLI layer (leaf): the ``mjai-play`` interactive match runner.

Menu-driven: select env -> assign seats (human | load policy) -> mode
(interactive | auto) -> run match. Nothing in the repo imports this package;
it is entered only via the ``mjai-play`` console-script (AGENTS.md §2, §10).

Each game ships a ``cli/renderers/<game>.py`` and ``cli/input_parsers/<game>.py``;
simultaneous-move games use blind human entry to preserve information structure.
"""
