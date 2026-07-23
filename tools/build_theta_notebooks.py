"""Generate notebooks/theta_<game>.ipynb — one theta-scan notebook per game.

Each notebook sweeps the PPO<->ACH interpolation weight theta on ONE game
(config + training + visualization), then renders:

  1. every theta's equilibrium-metric curve, overlaid with a min-max band;
  2. final metric vs theta;
  3. a gradient-scale telemetry panel (the scan's main confounder).

Logic is imported from mjai.* and tools/theta_probe.py — nothing is
reimplemented here (AGENTS.md §7).

Run: uv run python tools/build_theta_notebooks.py
"""

# The generated notebooks carry Chinese prose; allow fullwidth punctuation in
# this file's strings/comments (RUF001/2/3) rather than mangling the text.
# ruff: noqa: RUF001, RUF002, RUF003

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "notebooks"

# Per-game scan defaults. Budgets follow tools/build_league_notebooks.py (same
# machine, same measurements) so the two notebook families are comparable.
GAMES: dict[str, dict[str, object]] = {
    "brps": {
        "total": 60_000,
        "eval_every": 5_000,
        "metric": "nash_conv（exact）",
        "metric_note": "BRPS 是**同时博弈**，`mjai.eval.nash` 只对非同时博弈算 "
        "exploitability，所以本图纵轴是 nash_conv（2p0s 下 exploitability = "
        "nash_conv/2，但仓库不做这个改名，图上标的就是实际算出来的量）。",
        "note": "最便宜的 cycling 博弈，PPO 端预期出现绕圈/平台，ACH 端预期收敛到 "
        "(1/16, 10/16, 5/16)——theta 扫描最容易看出差别的一个游戏（约 45–60 min）。",
    },
    "kuhn": {
        "total": 60_000,
        "eval_every": 5_000,
        "metric": "exploitability（exact）",
        "metric_note": "回合制 2p0s，精确 exploitability = nash_conv / 2，由 "
        "OpenSpiel 全树计算。",
        "note": "训练快（约 45–60 min 跑完 5θ×3seed）。浅预算下两端可能拉不开差距，"
        "参见 docs/kuhn_tie_rootcause.md 的盆地效应分析。",
    },
    "liars_dice1": {
        "total": 10_000,
        "eval_every": 2_500,
        "metric": "exploitability（exact）",
        "metric_note": "回合制 2p0s，精确 exploitability。这也是论文复现里唯一没过 "
        "D5 判定的游戏（docs/reproduce_report.md §6）。",
        "note": "实测（本机 GPU）：训练约 97 ms/更新（672 env-steps/s），1e4 "
        "env-steps ≈ 15 s；精确 exploitability 每次约 12 s，本预算下 4–5 个评估点 "
        "≈ 1 min。**单臂约 1–1.5 min，5θ×3seed ≈ 20 min。** 注意 eval 仍是大头，"
        "加大 EVAL_EVERY 比缩短训练更能省时间。",
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


HEADER_MD = """# theta 扫描 · {game}（PPO ⟷ ACH 单参数插值）

一键 notebook：配置 → 训练 → 可视化。与 `notebooks/ab_{game}.ipynb`（mirror vs
league）互补——这里固定 mirror 自博弈，只扫**策略项**的插值权重 theta。

- **框架**：`src/mjai/algos/nn_updates.py::NNActorCriticUpdate` 是唯一的 NN 更新
  规则，策略损失为

      L_policy = (1 - theta) * L_ppo_clip + theta * L_ach

  `theta=0` 就是 PPO 截断代理，`theta=1` 就是论文忠实 ACH（Fu et al. ICLR 2022,
  Algorithm 2 / Eq. 29）。value 项与 entropy 项两端形式相同，**不参与插值**。
- **单因子**：优化器、优势处理、每批更新次数、梯度裁剪、网络结构全部共享，且默认
  取 ACH 侧（SGD 恒定 lr=1e-3、原始 GAE 优势、单次更新、无裁剪、trunk LayerNorm）。
  所以相邻 theta 之间**只差策略项**。若要评估 PPO 自己的最佳实践（Adam / 优势归一化
  / 多 epoch），那是另一个实验：改 `configs/exp/{game}_ppo_mlp_mirror.yaml` 里注明的
  旋钮，且 theta>0 时会打印 `ACHFidelityWarning`。
- **指标**：{metric}。{metric_note}
- **输出目录**：`runs/nb_theta/{game}/theta_<tag>/seed_N/`（含 `DONE` 标记；
  **重复执行训练 cell 会跳过已完成臂**，中断后续跑即可）。
- **注意**：{note}

**直接 Runtime → Run All 即可**；想加深/加宽，改下一格参数后重跑（已完成的臂不会
被重训；要重训某臂需先删掉它的目录）。"""

PARAMS_CODE = """# === Parameters ===
GAME        = "{game}"
THETAS      = [0.0, 0.25, 0.5, 0.75, 1.0]   # 0 = PPO, 1 = ACH
SEEDS       = [0, 1, 2]
TOTAL_ENV_STEPS = {total}   # per-arm budget (probe depth, not the paper's 1e7)
EVAL_EVERY      = {eval_every}
FINAL_FRAC  = 0.1           # "final" = mean over the last 10% of x (D5 convention)
SHOW_TQDM   = True          # per-arm tqdm bar over env-steps
from pathlib import Path
OUT_ROOT = Path("runs/nb_theta")"""

SETUP_CODE = """# === Setup: import the probe machinery (no logic reimplemented here) ===
import sys
from pathlib import Path

REPO = Path.cwd()
if not (REPO / "tools" / "theta_probe.py").is_file():
    REPO = REPO.parent  # tolerate running from notebooks/
sys.path.insert(0, str(REPO / "tools"))

import theta_probe  # run_arm / summarize / render_* helpers
from IPython.display import Image, display

def train_all():
    statuses = []
    for theta in THETAS:
        for seed in SEEDS:
            out = theta_probe.arm_dir(OUT_ROOT, GAME, theta, seed)
            if (out / "DONE").exists():
                print(f"skip  theta={theta:<5g} seed={seed} (already DONE)", flush=True)
                statuses.append((theta, seed, "cached"))
                continue
            print(f"train theta={theta:<5g} seed={seed} ...", flush=True)
            try:
                theta_probe.run_arm(
                    GAME, theta, seed,
                    total_env_steps=TOTAL_ENV_STEPS,
                    eval_every_env_steps=EVAL_EVERY,
                    root=OUT_ROOT,
                    progress_bar=SHOW_TQDM,
                )
                statuses.append((theta, seed, "done"))
            except Exception as e:  # keep going; report at the end
                statuses.append((theta, seed, f"FAILED: {type(e).__name__}: {e}"))
            print(f"      -> {statuses[-1][2]}", flush=True)
    return statuses

print(f"{len(THETAS)} thetas x {len(SEEDS)} seeds = {len(THETAS) * len(SEEDS)} arms")"""

TRAIN_CODE = """# === Train (long cell: arms run sequentially; safe to re-run) ===
statuses = train_all()
for theta, seed, st in statuses:
    print(f"theta={theta:<5g} seed={seed}: {st}")"""

SUMMARY_CODE = """# === Aggregate curves + per-theta results table ===
summary = theta_probe.summarize(OUT_ROOT, GAME, final_frac=FINAL_FRAC)
entry = summary.get(GAME, {})
print(f"metric: {entry.get('metric')}   final = mean over last {FINAL_FRAC:.0%} of x")
print()
for tag, arm in sorted(entry.get("thetas", {}).items(), key=lambda kv: kv[1]["theta"]):
    finals = {s: v for s, v in arm["final_per_seed"].items() if v is not None}
    vals = [round(v, 4) for v in finals.values()]
    mean = round(sum(finals.values()) / len(finals), 4) if finals else None
    print(f"theta={arm['theta']:<5g} final/seed={vals}  mean={mean}  done={len(arm['done'])}/{len(SEEDS)}")"""

FIGURE_CODE = """# === Figure 1: every theta's curve, overlaid (mean + min-max band) ===
fig1 = theta_probe.render_curves(summary, GAME, OUT_ROOT)
display(Image(filename=str(fig1))) if fig1 else print("no curves yet")"""

FINAL_FIG_CODE = """# === Figure 2: final metric vs theta (error bar = min-max across seeds) ===
fig2 = theta_probe.render_theta_final(summary, GAME, OUT_ROOT)
display(Image(filename=str(fig2))) if fig2 else print("no finals yet")"""

TELEMETRY_CODE = """# === Diagnostic: gradient scale / gate activity / clip rate per theta ===
# The two policy terms differ in gradient magnitude by orders of magnitude (the
# ACH term carries an unbounded 1/pi_old), so the EFFECTIVE learning rate varies
# with theta. Read Figure 2 together with this panel before concluding that a
# theta is "better" — a monotone trend here means the ranking is partly a
# step-size effect, not purely a policy-operator effect.
fig3 = theta_probe.render_telemetry(GAME, OUT_ROOT)
display(Image(filename=str(fig3))) if fig3 else print("no telemetry yet")"""

FOOTER_MD = """## 解读指南

- **图 1（叠加曲线）**：同预算下谁低谁好。带宽是 {n_seeds} 个 seed 的 min–max，不是
  置信区间；单 seed 发散会把带撑开，这本身就是信息（cycling 博弈上 PPO 端更容易发生）。
- **图 2（theta–final）**：如果曲线单调下降，说明「越像 ACH 越好」；如果在中间出现
  极小值，说明混合策略项优于两个端点——这是本 notebook 唯一能给出的新结论，也是
  值得进一步查证的地方（先看图 3 排除步长效应）。
- **图 3（遥测）**：`train/grad_norm` 随 theta 的变化幅度决定了图 2 有多少是「策略算子」
  的功劳、多少是「有效步长」的功劳。`gate_off_frac` 只在 theta>0 有值（ACH 门控关闭
  比例），`clip_frac` 只在 theta<1 有值（PPO 截断比例）。
- **口径**：`final` 取 x 轴末 {final_pct} 的均值（docs/reproduce_report.md 的 D5 口径），
  不是最后一个点。要改口径就改参数格的 `FINAL_FRAC`。
- **与论文对照**：只有 `theta=1` 这一臂是论文忠实 ACH，可以和 docs/figs 里的数字化
  曲线比；其余 theta 都是本仓库自定义的插值实验，**不要**拿去对论文。
- **产物用法**：每臂的 `checkpoints/best/` 是该臂的 SOTA 快照，`uv run mjai-play`
  的 policy 列表会直接列出它。"""


def build(game: str, spec: dict[str, object]) -> Path:
    CELLS.clear()
    md(
        HEADER_MD.format(
            game=game,
            metric=spec["metric"],
            metric_note=spec["metric_note"],
            note=spec["note"],
        )
    )
    code(PARAMS_CODE.format(game=game, total=spec["total"], eval_every=spec["eval_every"]))
    code(SETUP_CODE)
    code(TRAIN_CODE)
    code(SUMMARY_CODE)
    code(FIGURE_CODE)
    code(FINAL_FIG_CODE)
    code(TELEMETRY_CODE)
    md(FOOTER_MD.format(n_seeds=3, final_pct="10%"))
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
    out = OUT_DIR / f"theta_{game}.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main() -> None:
    for game, spec in GAMES.items():
        print(build(game, spec))


if __name__ == "__main__":
    main()
