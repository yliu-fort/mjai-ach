"""Regenerate notebooks/phase1_one_click.ipynb.

Run: uv run python tools/build_notebook.py
"""

import json
from pathlib import Path

CELLS = []


def md(*lines):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": list(lines)})


def code(*lines):
    CELLS.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": list(lines),
        }
    )


md(
    "# Phase-1 one-click experiment notebook\n",
    "\n",
    "Single parameterized entry point for the 2x2 Phase-1 experiment matrix\n",
    "(7 games x {PPO, ACH} x {mirror, league}). Trains, evaluates, and renders\n",
    "the ACH 2022 ICLR paper's headline figures (Fig 1 BRPS trajectory, Fig 2\n",
    "equilibrium-distance curves) plus a set of league diagnostics.\n",
    "\n",
    "**Output is layered** (AGENTS.md §6): a few core figures are inlined here;\n",
    "per-game details (cross-play heatmaps, per-game training curves, forgetting\n",
    "curves) are written to `runs/<cell>/plots/` and shown as path links.\n",
    "\n",
    "Imports from `mjai.*`; does not reimplement logic.",
)

code(
    "# === Parameters ===\n",
    "# Set RUN_ALL_MATRIX=True to sweep all 28 cells. QUICK=True picks 3 representative\n",
    "# games + short steps for a fast (<5 min on CPU) end-to-end demo of every plot.\n",
    "GAME          = 'kuhn'      # one of: brps, kuhn, leduc, liars_dice1, goofspiel5_ii, oshi_zumo, ttt\n",
    "ALGO          = 'ach'       # 'ppo' | 'ach'\n",
    "SELF_PLAY     = 'mirror'    # 'mirror' | 'league'\n",
    "N_STEPS       = 1000\n",
    "RUN_ALL_MATRIX = False      # True => sweep all 7 games x 2 algos x 2 modes\n",
    "QUICK          = False      # True => 3 games x 200 steps, fast demo\n",
    "\n",
    "import sys, pathlib\n",
    "sys.path.insert(0, str(pathlib.Path.cwd().parent / 'src'))\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "plt.rcParams['figure.dpi'] = 110\n",
    "\n",
    "from mjai.scripts.experiment import ExperimentConfig, run_experiment\n",
    "from mjai.eval.plots import (\n",
    "    plot_brps_trajectory, plot_equilibrium_curves, plot_final_metric_bars,\n",
    "    plot_crossplay_heatmap, plot_forgetting_curve,\n",
    "    load_train_curve, cell_label, safe_float,\n",
    ")\n",
    "from mjai.scripts.evaluate import _load_policy\n",
    "from mjai.agents.ckpt_io import discover_checkpoints\n",
    "from mjai.config.game_config import load_all_game_configs\n",
    "print('Available games:', sorted(load_all_game_configs().keys()))\n",
    "print('QUICK:', QUICK, '| RUN_ALL_MATRIX:', RUN_ALL_MATRIX)",
)

code(
    "# === Build the cell list ===\n",
    "ALL_GAMES = ['brps', 'kuhn', 'leduc', 'liars_dice1', 'goofspiel5_ii', 'oshi_zumo', 'ttt']\n",
    "ALGOS = ['ppo', 'ach']\n",
    "MODES = ['mirror', 'league']\n",
    "\n",
    "if QUICK:\n",
    "    ALL_GAMES = ['brps', 'kuhn', 'oshi_zumo']  # one-shot / sequential / simultaneous-cycling\n",
    "    N_STEPS_LOCAL = 200\n",
    "    EVAL_EVERY = 25\n",
    "    SAVE_EVERY = 50\n",
    "else:\n",
    "    N_STEPS_LOCAL = N_STEPS\n",
    "    EVAL_EVERY = max(N_STEPS // 10, 25)\n",
    "    SAVE_EVERY = max(N_STEPS // 5, 50)\n",
    "\n",
    "if RUN_ALL_MATRIX or QUICK:\n",
    "    cells = [(g, a, m) for g in ALL_GAMES for a in ALGOS for m in MODES]\n",
    "else:\n",
    "    cells = [(GAME, ALGO, SELF_PLAY)]\n",
    "print(f'Will run {len(cells)} experiment(s); {N_STEPS_LOCAL} steps each; eval every {EVAL_EVERY} steps.')",
)

md(
    "## Train\n",
    "\n",
    "Every cell trains with `eval_during_training=True` so we get a `train_curve.json`\n",
    "(required for the training-curve plots below).",
)
code(
    "run_dirs = {}\n",
    "for game, algo, mode in cells:\n",
    "    cfg = ExperimentConfig(\n",
    "        game=game, algo=algo, self_play_mode=mode, policy_kind='tabular',\n",
    "        n_steps=N_STEPS_LOCAL, save_every_steps=SAVE_EVERY, eval_every_steps=EVAL_EVERY,\n",
    "        eval_during_training=True,\n",
    "        out_dir=f'runs/{game}_{algo}_{mode}', seed=0,\n",
    "    )\n",
    "    print(f'\\n=== {game}/{algo}/{mode} ({N_STEPS_LOCAL} steps) ===')\n",
    "    run_dirs[(game, algo, mode)] = run_experiment(cfg)\n",
    "print('\\nAll training complete.')",
)

md(
    "## Core figure 1 - Biased-RPS policy trajectory (paper Fig 1 reproduction)\n",
    "\n",
    "P(R)/P(P)/P(S) over training, with the analytic Nash (1/16, 10/16, 5/16)\n",
    "dashed. Two panels: PPO/mirror (expected to cycle) vs ACH/mirror (expected\n",
    "to converge).",
)
code(
    "fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)\n",
    "for ax, (algo, mode) in zip(axes, [('ppo', 'mirror'), ('ach', 'mirror')]):\n",
    "    key = ('brps', algo, mode)\n",
    "    if key not in run_dirs:\n",
    "        ax.axis('off'); continue\n",
    "    rows = load_train_curve(run_dirs[key])\n",
    "    plots_dir = run_dirs[key] / 'plots'; plots_dir.mkdir(exist_ok=True)\n",
    "    plot_brps_trajectory(rows, title=f'BRPS - {algo}/{mode}',\n",
    "                         save_path=plots_dir / 'fig1_trajectory.png')\n",
    "    plt.show()  # inline the standalone version too\n",
    "axes[0].set_ylabel('action probability')\n",
    "print('Per-cell versions saved to runs/brps_*/plots/fig1_trajectory.png')",
)

md(
    "## Core figure 2 - Equilibrium-distance training curves (paper Fig 2)\n",
    "\n",
    "One panel per game. Within each panel, 4 lines = {PPO, ACH} x {mirror, league}.\n",
    "y-axis is symlog (lower = closer to Nash).",
)
code(
    "games_run = sorted({g for (g, _, _) in run_dirs})\n",
    "n = len(games_run)\n",
    "ncols = min(3, n); nrows = (n + ncols - 1) // ncols\n",
    "fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)\n",
    "for i, game in enumerate(games_run):\n",
    "    ax = axes[i // ncols][i % ncols]\n",
    "    curves = {}\n",
    "    for (g, algo, mode), rd in run_dirs.items():\n",
    "        if g == game:\n",
    "            curves[cell_label(algo, mode)] = load_train_curve(rd)\n",
    "    if curves:\n",
    "        plot_equilibrium_curves(curves, title=game)\n",
    "        plt.show()\n",
    "        # also save per-game\n",
    "        plots_dir = run_dirs.get((game, 'ach', 'mirror'), list(run_dirs.values())[0]) / 'plots'\n",
    "        plots_dir.mkdir(exist_ok=True)\n",
    "        plot_equilibrium_curves(curves, title=game, save_path=plots_dir / f'fig2_{game}.png')\n",
    "        plt.close()\n",
    "for j in range(len(games_run), nrows * ncols):\n",
    "    axes[j // ncols][j % ncols].axis('off')\n",
    "fig.tight_layout()\n",
    "plt.show()",
)

md(
    "## Core figure 3 - Final 2x2 comparison bar chart (paper Tab 1 graphical)\n",
    "\n",
    "Final equilibrium distance per cell, grouped by game. Lower bar = closer to Nash.",
)
code(
    "results = {}\n",
    "metric_name = 'exploitability'\n",
    "for (g, algo, mode), rd in run_dirs.items():\n",
    "    rows = load_train_curve(rd)\n",
    "    if not rows: continue\n",
    "    last = rows[-1]\n",
    "    for k in ('eval/exploitability', 'eval/nash_conv', 'eval/exact_nash_distance'):\n",
    "        if k in last:\n",
    "            results[(g, algo, mode)] = safe_float(last[k])\n",
    "            metric_name = k.removeprefix('eval/')\n",
    "            break\n",
    "fig = plot_final_metric_bars(results, games=games_run, metric_name=metric_name,\n",
    "                             save_path='runs/fig3_final_bars.png')\n",
    "plt.show()",
)

md(
    "## Core figure 4 - Final 2x2 comparison TABLE (paper Tab 1 format)\n",
    "\n",
    "Same data as the bar chart, in a pivot table; best cell per game is highlighted.",
)
code(
    "import pandas as pd\n",
    "df = pd.DataFrame([{'game': g, 'algo': a, 'mode': m, metric_name: v}\n",
    "                   for (g, a, m), v in results.items()])\n",
    "if not df.empty:\n",
    "    pivot = df.pivot_table(index='game', columns=['algo', 'mode'], values=metric_name)\n",
    "    def _hl(s):\n",
    "        is_best = s == s.min()\n",
    "        return ['background-color: #cfc' if b else '' for b in is_best]\n",
    "    display(pivot.style.apply(_hl, axis=1).format('{:.4g}'))  # noqa\n",
    "else:\n",
    "    print('no results')",
)

md(
    "## Detail figures - per-cell cross-play heatmap + forgetting curve\n",
    "\n",
    "These are written to `runs/<cell>/plots/` (AGENTS.md §6). The links below\n",
    "point at each file.",
)
code(
    "from mjai.eval.crossplay import cross_play_matrix, nontransitivity_score\n",
    "from mjai.games.loader import load_game\n",
    "from mjai.pipeline.rollout import RolloutConfig, RolloutWorkerCore\n",
    "\n",
    "links = []\n",
    "for (g, algo, mode), rd in run_dirs.items():\n",
    "    spec = load_game(g)\n",
    "    ckpts = discover_checkpoints(rd / 'checkpoints')\n",
    "    policies, names = [], []\n",
    "    for cdir, _ in ckpts:\n",
    "        try:\n",
    "            policies.append(_load_policy(cdir)[0]); names.append(cdir.name)\n",
    "        except Exception as e:\n",
    "            print(f'  skip {cdir.name}: {e}')\n",
    "    if len(policies) < 2: continue\n",
    "    runner = RolloutWorkerCore(spec, learner_player=0, config=RolloutConfig(n_episodes=20, seed=0))\n",
    "    cpr = cross_play_matrix(spec, policies, runner, n_episodes=20, policy_names=names)\n",
    "    plots_dir = rd / 'plots'; plots_dir.mkdir(exist_ok=True)\n",
    "    hm = plot_crossplay_heatmap(cpr.payoff, names, title=f'{g}/{algo}/{mode} cross-play',\n",
    "                                save_path=plots_dir / 'crossplay.png')\n",
    "    fg = plot_forgetting_curve(\n",
    "        [float(cpr.win_rate[0, j]) for j in range(1, len(policies))], names[1:],\n",
    "        title=f'{g}/{algo}/{mode} - final policy vs earlier checkpoints',\n",
    "        save_path=plots_dir / 'forgetting.png')\n",
    "    links.append((f'{g}/{algo}/{mode}', hm, fg, nontransitivity_score(cpr)))\n",
    "\n",
    "for label, hm, fg, nt in links:\n",
    "    print(f'{label}  (nontransitivity={nt:.3g})')\n",
    "    print(f'  cross-play heatmap: {hm}')\n",
    "    print(f'  forgetting curve:   {fg}')",
)

md(
    "## Interpretation\n",
    "\n",
    "(No automated reasoning - read the figures and judge. Pointers below.)\n",
    "\n",
    "- **Fig 1 (BRPS trajectory)**: does ACH/mirror settle near (1/16, 10/16, 5/16)\n",
    "  while PPO/mirror visibly cycles? That's the paper's headline motivation.\n",
    "- **Fig 2 (equilibrium curves)**: ACH lines should drop below PPO lines on\n",
    "  most games; league vs mirror should be close on small games (Kuhn) and\n",
    "  visibly different on cycling games (Oshi-Zumo, Goofspiel).\n",
    "- **Fig 3/4 (final bars / table)**: per-game ACH <= PPO? league < mirror on\n",
    "  forgetting for the cycling games?\n",
    "- **Detail heatmaps**: visible off-diagonal structure = non-transitive;\n",
    "  near-antisymmetric = strong RPS-style cycling.\n",
    "\n",
    "Artifacts: every figure is also saved under `runs/<cell>/plots/`.",
)

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("notebooks/phase1_one_click.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells)")
