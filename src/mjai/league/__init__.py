"""League layer: the self-play *controller*.

``LeagueManager`` + ``OpponentSampler`` (mirror vs PFSP) + ``CheckpointStore``
+ ``ExploiterRole``. Decides which opponent each episode plays against.

MUST NOT import concrete algos (``mjai.algos.ppo``, ``mjai.algos.ach``,
``mjai.algos.trainer``, ``mjai.algos.baselines``) — enforced by import-linter.
It imports only the abstract controller interface in ``mjai.algos.controller``.
This keeps the controller / Trainer / concrete-rule dependency acyclic
(AGENTS.md §2).

May import :mod:`mjai.algos.controller` (interface only), :mod:`mjai.agents`,
:mod:`mjai.config`, :mod:`mjai.utils`.
"""
