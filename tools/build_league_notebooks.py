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
        "note": "Re-measured 2026-07-23: training is ~97 ms per 64-sample update "
        "(672 env-steps/s), so 1e4 steps ≈ 15 s. The old '2-4 s per update' note "
        "here was wrong — it was charging the exact-exploitability eval to "
        "training. Eval is ~12 s per point since the batched-materialization fix "
        "(was 4-8 min), and is still the dominant cost: budget by EVAL_EVERY, "
        "not by TOTAL_ENV_STEPS.",
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
- **`THETAS` 旋钮**：默认 `[None]`（就是上面那份 ACH 配置，行为与以前完全一致）。
  想同时测 PPO / ACH / 混合损失就填 `[0.0, 0.5, 1.0]`——只有**策略项**随 theta 变，
  脚手架在所有 theta 上都保持 ACH 协议（D11 单因子）。**注意 `0.0` 不等于
  `configs/exp/{game}_ppo_mlp_mirror.yaml`**，参数格里有完整说明。
- **梯度探针**：`PROBE_GRAD_NORMS=True` 时额外记录 PPO / ACH 两项**各自的**梯度
  范数与二者夹角余弦，用来看混合损失到底是谁在推（总 grad_norm 看不出来）。
- 指标：{metric}。
- 输出目录：`runs/nb_ab/{game}/<mode>[_t<theta>]/seed_N/`。**臂是按配置 hash 缓存
  的**，不是按目录名：改了预算 / 设备 / 任何旋钮都会被认出来并告诉你改了哪个键，
  不会拿旧结果冒充（见参数格的 `ON_STALE`）。
- **SOTA checkpoint**：每臂训练完自动把评估指标最优的 checkpoint 复制为
  `checkpoints/best/`（附 `best.json` 记录指标值与 env_steps）——可直接在
  `mjai-play` 里加载试玩。
- **进度**：训练时每个臂有 tqdm 实时进度条（`SHOW_TQDM`）。
- 注意：{note}
- 口径脚注：league 的 1 env-step 只计收集角色 seat-0 决策（mirror 计两座），
  横比时同名预算的实际剂量不同（docs/league_investigation.md §2.1）。

**直接 Runtime → Run All 即可**；想加深/加宽，改下一格的参数后重跑（已完成的
臂不会被重训）。"""

PARAMS_CODE = '''# === Parameters ===
GAME        = "{game}"
MODES       = ["mirror", "league"]
SEEDS       = [0, 1, 2, 3]
TOTAL_ENV_STEPS = {total}   # per-arm budget (probe depth, not the paper's 1e7)
EVAL_EVERY      = {eval_every}
SHOW_TQDM   = True          # per-arm tqdm bar over env-steps

THETAS      = [None]        # arms = MODES x THETAS x SEEDS
"""PPO <-> ACH interpolation weight of the POLICY term (AGENTS.md D11).

    None    use the YAML as shipped (algo: ach). The default, and the only
            value that keeps the historical arm paths, so arms trained before
            this knob existed still count as cache hits.
    0.0     PPO clipped surrogate      0.5  the convex blend      1.0  ACH

Set e.g. [0.0, 0.5, 1.0] to run PPO / mixed / ACH side by side; each theta
gets its own arm directory (<game>_<mode>_t<tag>/), so they do not overwrite
each other and all of them land in one comparison figure -- line STYLE is the
self-play mode, COLOUR is theta.

READ THIS BEFORE COMPARING: THETAS=[0.0] is NOT
configs/exp/<game>_ppo_mlp_mirror.yaml. Only the policy term changes; the
scaffolding stays on the ACH protocol at every theta (constant-LR SGD, raw
GAE advantages, one epoch per batch, no grad clipping, trunk LayerNorm). That
is deliberate -- it makes theta the single factor that differs between arms.
To evaluate PPO's own best practices (Adam / advantage normalization / multi-
epoch) you want the ppo YAML instead, and that is a different experiment.

Cost scales with the product: [None] is the 8 arms this notebook always ran;
[0.0, 1.0] doubles it to 16.
"""

PROBE_GRAD_NORMS = True     # log the PPO and ACH terms' grad norms separately
"""Per-term gradient telemetry: train/grad_norm_{{ppo,ach}}[_scaled] and
train/grad_cos_ppo_ach. The two policy terms differ in gradient magnitude by
orders of magnitude (ACH carries an unbounded 1/pi_old), so theta is NOT the
blend of influence -- these say which term actually drove each update, and the
cosine says whether they were pulling together. Costs two extra backward
passes per update: measured +8.5% on a full train round (Liar's Dice, CPU,
batch 64). The update itself is bit-identical either way, so leaving this on
does not change any result."""

ON_STALE    = "error"       # "error" | "retrain" | "skip"
"""What to do with an arm that finished under a DIFFERENT config.

Arms are cached by a fingerprint of their resolved ExperimentConfig, not by
directory name, so raising TOTAL_ENV_STEPS (or changing the device, the eval
cadence, any ACH knob) is detected instead of being silently "skipped".

    error    refuse that arm and print which knob changed. Nothing is
             deleted, and the other arms still run.
    retrain  DELETE the arm directory and train it again. Required rather
             than merely nice: a second TensorBoard event file in the same
             tb/ interleaves two runs into one curve.
    skip     reuse the mismatched result anyway.

DEVICE is part of the fingerprint, so flipping cpu <-> cuda marks every
finished arm stale. That is the strict reading (a CPU result is not a CUDA
result); use ON_STALE="skip" for one run if you just want the old numbers.
"""

DEVICE      = "cpu"         # "cpu" | "cuda" | None (= whatever the YAML says)
"""CPU is the default on purpose, and it is the FAST option here.

The rollout asks the policy for ONE decision at a time, so a 21->128->13
forward never fills a GPU: it is ~10 host<->device syncs of launch overhead
around a matmul that takes microseconds. Measured on Liar's Dice (RTX 3060 Ti):

    cpu    2809 env-steps/s      one policy call 241 us
    cuda    441 env-steps/s      one policy call 2110 us   (6.4x slower)

Set "cuda" only if you have raised the network width or batch size far enough
that the matmul dominates the launch overhead -- measure before assuming.
"""
from pathlib import Path
OUT_ROOT = Path("runs/nb_ab") / GAME'''

SETUP_CODE = """# === Setup: import the probe machinery (no logic reimplemented here) ===
import sys
from pathlib import Path

REPO = Path.cwd()
if not (REPO / "tools" / "league_probe.py").is_file():
    REPO = REPO.parent  # tolerate running from notebooks/
sys.path.insert(0, str(REPO / "tools"))

import arm_cache      # config-fingerprint cache (hit / stale / missing)
import league_probe   # run_arm / arm_status / summarize / render_figure
import policy_view    # final-policy view (rollout + mjai.eval.policy_table)
from IPython.display import Image, display

def arm_kwargs(theta):
    return dict(
        total_env_steps=TOTAL_ENV_STEPS,
        eval_every_env_steps=EVAL_EVERY,
        root=OUT_ROOT,
        device=DEVICE,
        theta=theta,
        probe_term_grad_norms=PROBE_GRAD_NORMS,
    )

def arms():
    return [(m, t, s) for m in MODES for t in THETAS for s in SEEDS]

def arm_label(mode, theta, seed):
    return f"{mode:6s} theta={'yaml' if theta is None else format(theta, 'g'):<5s} seed={seed}"

def train_all():
    statuses, refused = [], []
    for mode, theta, seed in arms():
        label = arm_label(mode, theta, seed)
        kwargs = arm_kwargs(theta)
        out = league_probe.arm_dir(OUT_ROOT, GAME, mode, seed, theta)
        st = league_probe.arm_status(GAME, mode, seed, **kwargs)
        action, why = arm_cache.resolve(st, ON_STALE, out)
        if action != "train":
            print(f"skip  {label}: {why}", flush=True)
            statuses.append((label, "cached" if action == "skip" else "REFUSED"))
            if action == "refuse":
                refused.append(label)
            continue
        print(f"train {label}: {why}", flush=True)
        try:
            league_probe.run_arm(GAME, mode, seed, progress_bar=SHOW_TQDM, **kwargs)
            statuses.append((label, "done"))
        except Exception as e:  # keep going; report at the end
            statuses.append((label, f"FAILED: {type(e).__name__}: {e}"))
        print(f"      -> {statuses[-1][1]}", flush=True)
    if refused:
        print()
        print("=" * 72)
        print(f"{len(refused)} arm(s) REFUSED: finished under a different config.")
        print("Nothing was deleted. See the per-arm lines above for the changed knob,")
        print('then set ON_STALE="retrain" (rebuilds them) or "skip" (reuses them).')
        print("=" * 72)
    return statuses

print(f"{len(MODES)} modes x {len(THETAS)} thetas x {len(SEEDS)} seeds "
      f"= {len(arms())} arms on {DEVICE}")"""

TRAIN_CODE = """# === Train (long cell: arms run sequentially; safe to re-run) ===
statuses = train_all()
for label, st in statuses:
    print(f"{label}: {st}")"""

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
# games=[GAME] keeps this to ONE panel; the default panel set is all 7 games,
# which in a per-game notebook renders 6 permanently blank panels.
fig_path = league_probe.render_figure(summary, OUT_ROOT, games=[GAME])
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
tb_dirs = sorted(OUT_ROOT.glob(f"{GAME}_league*/seed_*/tb"))
if not tb_dirs:
    print("no league arms finished yet")
else:
    fig, axes = plt.subplots(1, len(LEAGUE_TAGS), figsize=(14, 3.6))
    for ax, tag in zip(axes, LEAGUE_TAGS, strict=True):
        for d, curve in read_many(tb_dirs, tag=tag).items():
            if curve:
                ax.plot([s for s, _ in curve], [v for _, v in curve],
                        label=Path(d).parent.parent.name + "/" + Path(d).parent.name)
        ax.set_title(tag)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    plt.show()"""

GRAD_CODE = """# === Per-term gradient telemetry (PROBE_GRAD_NORMS) ===
# What the total train/grad_norm cannot tell you: which POLICY term drove each
# update, and whether the two were pulling together. Only meaningful when an
# arm ran at an intermediate theta -- at theta=0 there is no ACH gradient and
# at theta=1 no PPO one, and those tags are then absent rather than logged as
# a misleading 0.0.
GRAD_TAGS = [
    "train/grad_norm",
    "train/grad_norm_ppo_scaled",
    "train/grad_norm_ach_scaled",
    "train/grad_cos_ppo_ach",
]
grad_dirs = sorted(OUT_ROOT.glob(f"{GAME}_*/seed_0/tb"))
fig, axes = plt.subplots(1, len(GRAD_TAGS), figsize=(5 * len(GRAD_TAGS), 3.6))
drew = False
for ax, tag in zip(axes, GRAD_TAGS, strict=True):
    for d, curve in sorted(read_many(grad_dirs, tag=tag).items()):
        if not curve:
            continue
        ax.plot([s for s, _ in curve], [v for _, v in curve], lw=1.0,
                label=Path(d).parent.parent.name)
        drew = True
    ax.set_title(tag)
    ax.set_xlabel("update")
    ax.grid(alpha=0.3)
    if "grad_norm" in tag and ax.get_lines():
        ax.set_yscale("log")   # spans orders of magnitude; non-negative
if not drew:
    print("no gradient telemetry yet (train some arms first)")
else:
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    plt.show()
    if all(t is None or t in (0.0, 1.0) for t in THETAS):
        print("note: every arm sits at a theta endpoint, so only one policy term "
              "exists per arm and grad_cos_ppo_ach is empty by construction. "
              "Add an intermediate theta (e.g. 0.5) to populate it.")"""

POLICY_MD = """## 最终策略（每臂学到了什么）

上面的曲线只说「离 Nash 多远」，不说「到底怎么打」。这一格把每臂**训练完的策略**
物化出来直接看。展示形式由游戏规模决定（实测枚举成本，见
`src/mjai/eval/policy_table.py` 的模块 docstring）：

| 规模 | 形式 |
|---|---|
| brps（2 个信息集）、kuhn（12） | 完整表格 + 柱状图；brps 另附与解析 NE (1/16, 10/16, 5/16) 的 TV 距离 |
| leduc（936）、goofspiel5_ii（2124） | infoset × action 热力图 |
| liars_dice1（24576）、ttt（294778） | 动作边缘分布 + **按自博弈访问频率排序的 Top-K 信息集表** |
| oshi_zumo | 全树枚举跑不完（实测 >90 s），只给根节点开局分布 |

访问频率来自用该臂自己的策略自博弈 `POLICY_EPISODES` 局，按 observation 向量
join 回枚举出来的行——所以「常见局面」是真的按到达频率排的，不是按行号。
完整表格无论哪种规模都会写成 CSV 落盘。"""

POLICY_CODE = """# === Final policy per arm ===
POLICY_PICK     = "best"   # "best" (SOTA snapshot) | "last" | "step_N"
POLICY_SEED     = 0        # which seed's arm to show
POLICY_EPISODES = 400      # self-play episodes used to rank info states (0 = skip)
POLICY_TOP_K    = 12       # rows in the printed table

import pandas as pd
from mjai.eval.policy_table import PolicyViewError, brps_nash_gap, root_policy, to_records

policy_arms = [
    (arm_label(mode, theta, POLICY_SEED).strip(),
     league_probe.arm_dir(OUT_ROOT, GAME, mode, POLICY_SEED, theta))
    for mode in MODES for theta in THETAS
]
policy_arms = [(lab, run) for lab, run in policy_arms if (run / "checkpoints").is_dir()]

if not policy_arms:
    print("no trained arms yet")
else:
    fig_path, views, skipped = policy_view.render_arms(
        policy_arms, OUT_ROOT / "figs" / f"policy_{GAME}.png",
        checkpoint=POLICY_PICK, episodes=POLICY_EPISODES, player=0,
    )
    for lab, why in skipped.items():
        print(f"[{lab}] no policy table: {why}")
        try:
            for p, dist in root_policy(dict(policy_arms)[lab], checkpoint=POLICY_PICK).items():
                top = sorted(dist.items(), key=lambda kv: -kv[1])[:5]
                print(f"    root p{p}: " + ", ".join(f"{k}={v:.3f}" for k, v in top))
        except PolicyViewError as e:
            print(f"    (and no policy to load: {e})")
    if fig_path:
        display(Image(filename=str(fig_path)))
    for lab, view in views.items():
        print(f"\\n--- {lab} --- {view.checkpoint}")
        if view.game == "brps":
            probs, tv = brps_nash_gap(view)
            print(f"    seat-0 policy = {probs.round(4)}   TV to analytic NE = {tv:.4f}")
        rows = view.top_rows(POLICY_TOP_K, player=0)
        df = pd.DataFrame(to_records(view, rows=rows)).set_index("info_state")
        display(df.style.format(precision=3, na_rep="-"))
        csv = OUT_ROOT / "figs" / f"policy_{GAME}_{policy_view.slug(lab)}.csv"
        print(f"    full table ({len(view.labels)} rows) -> "
              f"{policy_view.write_csv(view, csv)}")"""

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
- **theta 对照**（`THETAS` 非 `[None]` 时）：图里颜色是 theta、线型是 mode。若
  `theta=0`（PPO 策略项）在 cycling 博弈上出现平台/发散而 `theta=1`（ACH）收敛，
  那就是论文 Fig 1 的动机在本仓库的复现；下一格的 `grad_cos_ppo_ach` 若持续为负，
  说明中间 theta 的两项在互相抵消，这时 theta 的排序更多是步长效应而非算子效应。
- **产物用法**：`checkpoints/best/` 是该臂的 SOTA 快照，`uv run mjai-play`
  的 policy 列表会直接列出它（带 best 字样）；周期快照在 `checkpoints/step_*`。
- 想更接近论文预算：把 `TOTAL_ENV_STEPS` 提到 1e5–1e6 重跑。已完成的臂会因为
  配置 hash 变了而被标成 stale 并逐键报告差异——按提示把 `ON_STALE` 改成
  `"retrain"` 重训，或 `"skip"` 沿用旧结果。"""


def build(game: str, spec: dict[str, object]) -> Path:
    CELLS.clear()
    md(HEADER_MD.format(game=game, metric=spec["metric"], note=spec["note"]))
    code(PARAMS_CODE.format(game=game, total=spec["total"], eval_every=spec["eval_every"]))
    code(SETUP_CODE)
    code(TRAIN_CODE)
    code(SUMMARY_CODE)
    code(FIGURE_CODE)
    code(TELEMETRY_CODE)
    code(GRAD_CODE)
    md(POLICY_MD)
    code(POLICY_CODE)
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
