"""Generate notebooks/ab_<game>.ipynb — one one-click A/B validation notebook
per game (config + training + visualization), for manual execution.

Each notebook trains mirror + league arms (paper-faithful ACH protocol from
configs/exp/<game>_ach_mlp_<mode>.yaml), aggregates the eval curves with
mean/min-max bands, renders the comparison figure and league health telemetry,
and prints a per-arm results table. Logic is imported from mjai.* and
tools/league_probe.py — nothing is reimplemented (AGENTS.md §7).

Run: uv run python tools/build_league_notebooks.py
"""

# The generated notebooks carry Chinese prose; allow fullwidth punctuation in
# this file's strings/comments (RUF001/2/3) rather than mangling the text.
# ruff: noqa: RUF001, RUF002, RUF003

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "notebooks"

# Per-game probe defaults (measured on this machine, 2026-07-22) and notes.
GAMES: dict[str, dict[str, object]] = {
    "kuhn": {
        "total": 60_000,
        "eval_every": 5_000,
        "metric": "exploitability (exact)",
        "note": "Turn-based 2p0s; fast (~3 min/arm). At this budget both modes sit "
        "on the same aggressiveness transient — see docs/kuhn_tie_rootcause.md.",
    },
    "brps": {
        "total": 60_000,
        "eval_every": 5_000,
        "metric": "nash_conv (exact) + TV distance to analytic NE",
        "note": "Cheapest cycling game; league's history pool is expected to "
        "suppress mirror's best-response cycling (v1 finding).",
    },
    "leduc": {
        "total": 60_000,
        "eval_every": 5_000,
        "metric": "exploitability (exact)",
        "note": "Turn-based 2p0s; larger tree than kuhn, still exact-evaluable.",
    },
    "liars_dice1": {
        "total": 10_000,
        "eval_every": 2_500,
        "metric": "exploitability (exact)",
        "note": "SLOW: ~2-4 s per 64-sample update — 1e4 steps ≈ 8-15 min/arm "
        "(8 arms ≈ 1-2 h). Raise TOTAL_ENV_STEPS only if you can wait; deeper "
        "runs need resume support or SLURM (Phase 3).",
    },
    "ttt": {
        "total": 60_000,
        "eval_every": 5_000,
        "metric": "nash_conv (sampled estimator) / exploitability (sampled/2)",
        "note": "Exact NashConv costs ~24 s/eval on ttt, so the arm configs use "
        "the sampled estimator (~2 s/eval, conservative; "
        "src/mjai/eval/sampled_nash.py).",
    },
    "goofspiel5_ii": {
        "total": 10_000,
        "eval_every": 1_000,
        "metric": "nash_conv (exact, ~3.4 s/eval)",
        "note": "Simultaneous; training is fast but exact eval is the cost. "
        "Default 1e4 steps keeps a full run ≈ 5-10 min/arm.",
    },
    "oshi_zumo": {
        "total": 20_000,
        "eval_every": 5_000,
        "metric": "nash_conv (sampled estimator)",
        "note": "Exact NashConv is infeasible at this game size (full-tree "
        "traversal hangs); arms use the sampled estimator. Horizon 20 makes "
        "training slower (~2-4 min/arm).",
    },
}

CELLS: list[dict] = []


def md(text: str) -> None:
    CELLS.append(
        {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
    )


def code(text: str) -> None:
    CELLS.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.splitlines(keepends=True),
        }
    )


HEADER_MD = """# A/B 验证 · {game}（mirror vs league）

一键 notebook：配置 → 训练 → 可视化。与 `notebooks/phase1_one_click.ipynb`
互补——这里聚焦**单游戏**的 league-vs-mirror 收敛对比（F2 调查）。

- 协议：论文忠实 ACH（Fu et al. ICLR 2022），配置直接读
  `configs/exp/{game}_ach_mlp_<mode>.yaml`，本 notebook 只覆写
  `seed / out_dir / total_env_steps / eval_every_env_steps`（AGENTS.md §9）。
- 指标：{metric}。
- 输出目录：`runs/nb_ab/{game}/<mode>/seed_N/`（含 `DONE` 标记；**重复执行训练
  cell 会跳过已完成臂**，中断后续跑即可）。
- **SOTA checkpoint**：每臂训练完自动把评估指标最优的 checkpoint 复制为
  `checkpoints/best/`（附 `best.json` 记录指标值与 env_steps）——可直接在
  `mjai-play` 里加载试玩。
- **进度**：训练时每个臂有 tqdm 实时进度条（`SHOW_TQDM`）。
- 注意：{note}
- 口径脚注：league 的 1 env-step 只计收集角色 seat-0 决策（mirror 计两座），
  横比时同名预算的实际剂量不同（docs/league_investigation.md §2.1）。

**直接 Runtime → Run All 即可**；想加深/加宽，改下一格的参数后重跑（已完成的
臂不会被重训）。"""

PARAMS_CODE = """# === Parameters ===
GAME        = "{game}"
MODES       = ["mirror", "league"]
SEEDS       = [0, 1, 2, 3]
TOTAL_ENV_STEPS = {total}   # per-arm budget (probe depth, not the paper's 1e7)
EVAL_EVERY      = {eval_every}
SHOW_TQDM   = True          # per-arm tqdm bar over env-steps
from pathlib import Path
OUT_ROOT = Path("runs/nb_ab") / GAME"""

SETUP_CODE = """# === Setup: import the probe machinery (no logic reimplemented here) ===
import sys
from pathlib import Path

REPO = Path.cwd()
if not (REPO / "tools" / "league_probe.py").is_file():
    REPO = REPO.parent  # tolerate running from notebooks/
sys.path.insert(0, str(REPO / "tools"))

import league_probe  # run_arm / summarize / render_figure / EVAL_TAG_CHAIN
from IPython.display import Image, display

def train_all():
    statuses = []
    for mode in MODES:
        for seed in SEEDS:
            out = league_probe.arm_dir(OUT_ROOT, GAME, mode, seed)
            if (out / "DONE").exists():
                print(f"skip  {mode:6s} seed={seed} (already DONE)", flush=True)
                statuses.append((mode, seed, "cached"))
                continue
            print(f"train {mode:6s} seed={seed} ...", flush=True)
            try:
                league_probe.run_arm(
                    GAME, mode, seed,
                    total_env_steps=TOTAL_ENV_STEPS,
                    eval_every_env_steps=EVAL_EVERY,
                    root=OUT_ROOT,
                    progress_bar=SHOW_TQDM,
                )
                statuses.append((mode, seed, "done"))
            except Exception as e:  # keep going; report at the end
                statuses.append((mode, seed, f"FAILED: {type(e).__name__}: {e}"))
            print(f"      -> {statuses[-1][2]}", flush=True)
    return statuses"""

TRAIN_CODE = """# === Train (long cell: arms run sequentially; safe to re-run) ===
statuses = train_all()
for mode, seed, st in statuses:
    print(f"{mode:6s} seed={seed}: {st}")"""

SUMMARY_CODE = """# === Aggregate curves + per-arm results table (incl. SOTA best ckpt) ===
import json

summary = league_probe.summarize(OUT_ROOT)
for arm, data in summary.items():
    finals = data.get("final_per_seed", {})
    tag = data.get("tag", "?")
    vals = [round(v, 4) for v in finals.values()]
    mean = round(sum(finals.values()) / len(finals), 4) if finals else None
    print(f"{arm:28s} tag={tag:26s} final/seed={vals}  mean={mean}  done={len(data.get('done', []))}/{len(finals)}")
print()
print("SOTA best checkpoints (lowest eval point per run, copied to checkpoints/best):")
for bj in sorted(OUT_ROOT.glob("*_*/seed_*/checkpoints/best/best.json")):
    info = json.loads(bj.read_text(encoding="utf-8"))
    rel = bj.parent.parent.parent.relative_to(OUT_ROOT)
    print(f"  {str(rel):28s} {info['tag']}={info['value']:.4f} @ env_steps={info['env_steps']}")"""

FIGURE_CODE = """# === Comparison figure (mean + min-max band across seeds) ===
fig_path = league_probe.render_figure(summary, OUT_ROOT)
if fig_path:
    display(Image(filename=str(fig_path)))
else:
    print("no curves yet")"""

TELEMETRY_CODE = """# === League health telemetry (league arms only; B7 scalars) ===
import matplotlib.pyplot as plt
from tb_eval import read_many

LEAGUE_TAGS = [
    "league/promotions_total",
    "league/main_snapshots_total",
    "league/pool_size",
]
tb_dirs = sorted(OUT_ROOT.glob(f"{GAME}_league/seed_*/tb"))
if not tb_dirs:
    print("no league arms finished yet")
else:
    fig, axes = plt.subplots(1, len(LEAGUE_TAGS), figsize=(14, 3.6))
    for ax, tag in zip(axes, LEAGUE_TAGS, strict=True):
        for d, curve in read_many(tb_dirs, tag=tag).items():
            if curve:
                ax.plot([s for s, _ in curve], [v for _, v in curve],
                        label=Path(d).parent.name)
        ax.set_title(tag)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    plt.show()"""

FOOTER_MD = """## 解读指南

- **同预算谁低谁好**：曲线为 mean + min–max 带（n={n_seeds} seeds）。cycling 博弈
  （brps / goofspiel5_ii / oshi_zumo）重点看 mirror 是否出现平台期或单 seed
  发散而 league 是否抑制之；收敛型博弈（kuhn / leduc / ttt）在浅预算下可能
  拉不开差距（参见 docs/kuhn_tie_rootcause.md 的盆地效应分析）。
- **league 健康度**：上一格三面板应看到 main 快照按节奏入池、池逐渐填满至
  capacity=16、晋升持续发生（修复 B1–B4 后的预期行为；背景见
  docs/league_health_check.md）。
- **已知口径**：league env-step 只计 seat-0 决策；exploiter 轮次（~2/3 轮）的
  梯度落在 main 上且 ratio 门控实际生效——结论请带此脚注。
- **产物用法**：`checkpoints/best/` 是该臂的 SOTA 快照，`uv run mjai-play`
  的 policy 列表会直接列出它（带 best 字样）；周期快照在 `checkpoints/step_*`。
- 想更接近论文预算：把 `TOTAL_ENV_STEPS` 提到 1e5–1e6 重跑（已完成的臂会被
  跳过；注意更深的预算需要删除对应臂目录才能重训该臂）。"""


def build(game: str, spec: dict[str, object]) -> Path:
    CELLS.clear()
    md(HEADER_MD.format(game=game, metric=spec["metric"], note=spec["note"]))
    code(PARAMS_CODE.format(game=game, total=spec["total"], eval_every=spec["eval_every"]))
    code(SETUP_CODE)
    code(TRAIN_CODE)
    code(SUMMARY_CODE)
    code(FIGURE_CODE)
    code(TELEMETRY_CODE)
    md(FOOTER_MD.format(n_seeds=4))
    nb = {
        "cells": list(CELLS),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (mjai)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = OUT_DIR / f"ab_{game}.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main() -> None:
    for game, spec in GAMES.items():
        print(build(game, spec))


if __name__ == "__main__":
    main()
