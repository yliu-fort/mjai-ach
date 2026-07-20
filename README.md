# mjai-ach

An IMPALA-style **PPO / ACH + league-play** research pipeline for
imperfect-information games.

> **Status — Phase 1.** Tabular + small-MLP, validated on 7 small games
> (Biased RPS, Goofspiel-5 II, Liar's-Dice-1, Oshi-Zumo, Leduc, Kuhn, Tic-Tac-Toe)
> on a home CPU+GPU. All code is written in Phase 1; Phases 2 (4-player Mahjong)
> and 3 (SLURM, 128-core + multi-GPU) are config + tuning only.

See **[AGENTS.md](AGENTS.md)** for the governance contract that binds both
humans and AI working in this repo.

## What this project tests

The [ACH paper](https://openreview.net/forum?id=DTXZqTNV5nW) (ICLR 2022) argues
that an actor-critic with an entropy-regularized advantage objective ("Actor-Critic
Hedge") converges in last-iterate under pure mirror self-play — **no league, no
average-strategy table**. This project asks the follow-up: in games where
latest-vs-latest play visibly cycles, does league play help *empirically* even
though ACH's theory doesn't require it?

The 2×2 experiment matrix (run for each of the 7 games):

|                | Mirror self-play | League (50/30/20) |
|----------------|:----------------:|:-----------------:|
| **PPO**        |        ✓         |         ✓         |
| **ACH**        |        ✓         |         ✓         |

Each cell reports exploitability / NashConv, cross-play payoff matrices,
worst-case win rate vs the history pool, a forgetting metric, non-transitivity
detection, and full training curves (TensorBoard).

## Quick start

```bash
# Python 3.12 + uv required.
uv sync --extra dev               # installs torch (cu128), open-spiel, ray, dev tools
uv run pre-commit install         # commit gates
uv run pre-commit install --hook-type pre-push   # push gates

# Train one cell of the matrix:
uv run mjai-train --config configs/exp/kuhn_ach_mirror.yaml

# Launch the one-click notebook:
uv run jupyter notebook notebooks/phase1_one_click.ipynb

# Play against any saved policy:
uv run mjai-play
```

## Layout

```
src/mjai/   games  agents  algos  league  pipeline  eval  cli  config  utils
configs/    games/  exp/
notebooks/  scripts/  tests/{unit,integration}/  tools/
```

Layering is enforced by `import-linter` (see AGENTS.md §2). Every file is
<500 lines (AST-guarded). GPU is the default; CPU requires `--cpu` or `MJAI_CPU=1`.

## License

MIT.
