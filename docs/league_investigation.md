# League vs Mirror A/B 探测报告（F2 第四步）

> 探测对象：同一套论文忠实 ACH 协议（Fu et al. ICLR 2022，Appendix G p25–26
> + H.3 p27–28）下，**mirror 自对弈** 与 **league 自对弈** 的 exploitability
> 收敛差异。
> 规模：{kuhn, brps} × {mirror, league} × seeds 0–3 = **16 臂**，每臂
> 6e4 env-steps（探测深度，仅为论文 1e7 预算的 0.6%）。
> 性质：F2 决策用的**首份调查**，非验收判定；结论只覆盖早期训练动态。
> 运行环境：RTX 3060 Ti（CUDA，`uv run` + `.venv`），2026-07-22 本机顺序执行。

---

## 1. 实验设置

### 1.1 臂与预算

- 16 条独立命令，每条形如
  `uv run python tools/league_probe.py --game <kuhn|brps> --mode <mirror|league> --seed <0..3> --total-env-steps 60000 --eval-every 5000`。
- 输出：`runs/league_probe/<game>_<mode>/seed_<k>/`（`DONE` 标记、`tb/` 事件文件、
  `checkpoints/`、`config.json`、`train_curve.json`）。
- 汇总：`runs/league_probe/summary.json` + `runs/league_probe/figs/ab_exploitability.png`
  （`tools/league_probe.py --summarize`）。
- 评估：每 5e3 env-steps 一次精确评估（首个评估点因按整 episode 记账落在
  ≈5033；brps_mirror 末点同理落在 60032），共 12 点/臂，对象为当前策略
  π=softmax(y)（非平均策略）。

### 1.2 协议来源配置

超参数全部来自 `configs/exp/<game>_ach_mlp_<mode>.yaml`（探测脚本只覆写
`seed / out_dir / total_env_steps / eval_every_env_steps`，AGENTS.md §9 配置纪律）：

| 项 | 取值（两游戏一致） | 出处 |
|---|---|---|
| 网络 | MLP `(128,)` + ReLU，policy/value 双头 | 论文 p25 |
| 优化器 | SGD 恒定 lr=1e-3，无梯度裁剪 | p27 Table 7 |
| value_coef / β / η | 1.0（≡α=2.0）/ 1e-2 / 1.0 | p27–28 |
| 门控 | 优势符号单侧 logit 门控 l_th=2.0 + ratio 门控 ε=0.5 | p24 Algorithm 2 / p28 |
| Batch | target_samples=64（整 episode 聚合） | p28 Table 8 |
| GAE λ / γ | 0.95 / 1.0 | 假设 A1 |
| League 旋钮（仅 league 臂） | capacity=16；mix=(main 0.5, history 0.3, exploiter 0.2)，PFSP；晋升阈值 0.55/0.55、share 0.70、窗口 20；reset=`to_main` | 代码默认值（LeagueConfig/LeagueMix） |

游戏：`kuhn_poker`（回合制 2p0s，可精确 exploitability）；`matrix_brps`
（一次性同时行动，payoff `{0,-25,50; 25,0,-5; -50,5,0}`，解析 NE
`(1/16, 10/16, 5/16)`）——BRPS 是最便宜的 cycling 博弈，顶替精确
NashConv 不可行的 oshi_zumo（`configs/games/brps.yaml` 注记）。

### 1.3 运行实况

- **16/16 臂一次性成功**，无崩溃、无 300 s 超时、无 4e4 降级重跑；每臂
  `DONE` + `tb/` 事件文件逐臂验证通过。
- 单臂墙钟（tb 文件创建间隔，含启动）：kuhn_mirror ≈101–109 s，
  kuhn_league ≈162–163 s，brps_mirror ≈93–117 s，brps_league ≈179–214 s；
  16 臂合计 ≈35 min。league 臂更慢（对手采样 + 三角色轮换 + 池管理）。

---

## 2. 口径注意（横比前必读）

### 2.1 env-step 计量：mirror 计两座，league 只计收集角色 seat-0

`scripts/experiment.py:321`：`env_steps += trainer.step().batch_size`，
1 env-step = 1 个被采样决策点。而 batch 的构成两臂不同：

- **mirror**：batch 汇集**两个座位**的转移（`algos/controller.py:114`，
  `pipeline/rollout.py:67–71`，论文 p24 共享 θ,ω 学两座数据）。
- **league**：batch 只含**本轮收集角色 seat-0** 的转移
  （`league/league_controller.py:100` 的 `batch.for_player(0)`）。

因此同名 6e4 env-steps 的实际剂量不同：以 BRPS（每座位每局 1 决策）计，
mirror ≈ 3e4 episodes 且梯度含两座，league ≈ 6e4 episodes 但梯度只含
seat-0。横向“同预算谁快谁慢”必须带此脚注。

### 2.2 exploiter 轮次的梯度落在 main 上（且 ratio 门控不再 vacuous）

league 每轮按 `[MAIN, MAIN_EXPLOITER, LEAGUE_EXPLOITER]` 轮换
（`league_controller.py:55–59`），即约 **2/3 的轮次（也就是约 2/3 的
env-step 预算）是 exploiter 收集**。Trainer 只持有一个绑定 main 网络的
UpdateRule，`collect()` 返回什么就吃什么（`algos/controller.py:91–100`）：

- exploiter 轮次的 batch 直接进入 **main** 的 ACH 更新；exploiter 自身
  **从不接受梯度**，只在晋升后被 reset 到当前 main 权重（`manager.py`、
  配置 `league_reset_mode=to_main`）。
- 于是 main 在 exploiter 轮次消费的是**过期的自己**采集的 off-policy
  数据：batch 内 behavior logprob 来自收集时的 exploiter（rollout 单前向
  记录，`rollout.py:110–116`），而 main 已继续训练，`ratio =
  exp(new_logp − old_logp) ≠ 1`。paper-faithful 复现里“同步单线程 ⇒
  ratio 恒 1、门控 vacuous”的前提（`nn_updates.py:250–252`，论文 p28
  注记）在这 2/3 轮次中**不成立**，ratio 门控与 1/old_probs 权重是真实
  起作用的。此为设计现状的客观记录，是否合乎预期留待 F2 判定（§6）。

### 2.3 评估指标：kuhn 有 exploitability；brps 只有 NashConv + TV 距离

- `eval/nash.py`：回合制 2p0s（kuhn）同时记录 `eval/exploitability` 与
  `eval/nash_conv`；同时行动博弈（brps）调 OpenSpiel exploitability 会
  抛错（`nash.py:62–66`），只记录 `eval/nash_conv` +
  `eval/exact_nash_distance`（对解析 NE 的 TV 距离）。
- 已实测 kuhn 两臂每点 **nash_conv ≡ 2 × exploitability**（2p0s 约定），
  后文 brps 的 NashConv 可据此与 kuhn 的 exploitability 换算口径对照。
- 工具影响：`tools/league_probe.py --summarize` 只读 `eval/exploitability`
  一个 tag，故 `summary.json` 中 brps 两臂 band 为空、
  `ab_exploitability.png` 的 brps 面板无数据（运行时有空 legend 的
  UserWarning）。brps 数据经只读方式另取，补充图见 §3.3。

---

## 3. 结果

### 3.1 最终值（末评估点，6e4 步）

**kuhn：exploitability**（↓ 越低越好）

| seed | mirror | league |
|---|---|---|
| 0 | 0.32744 | 0.31141 |
| 1 | 0.32268 | 0.33656 |
| 2 | 0.32012 | 0.32134 |
| 3 | 0.32354 | 0.33386 |
| **mean** | **0.32345** | **0.32579** |
| [min, max] | [0.32012, 0.32744] | [0.31141, 0.33656] |

**brps：NashConv**（↓；≡ 2×exploitability 口径；mirror 末点 @60032）

| seed | mirror | league |
|---|---|---|
| 0 | 7.045 | 5.399 |
| 1 | 5.861 | 2.112 |
| 2 | 6.583 | 2.197 |
| 3 | **50.000**（崩溃） | 1.935 |
| **mean** | **17.372**（剔除 seed_3：6.496） | **2.911** |
| [min, max] | [5.861, 50.000] | [1.935, 5.399] |

**brps：对解析 NE 的 TV 距离**（↓，几何口径）

| seed | mirror | league |
|---|---|---|
| 0 | 0.1858 | 0.1377 |
| 1 | 0.2193 | 0.1564 |
| 2 | 0.0506 | 0.1432 |
| 3 | **0.9375**（崩溃） | 0.1847 |
| **mean** | **0.3483**（剔除 seed_3：0.1519） | **0.1555** |
| [min, max] | [0.0506, 0.9375] | [0.1377, 0.1847] |

### 3.2 轨迹里程碑（4 seeds 均值）

| env-steps | kuhn mirror expl | kuhn league expl | brps mirror NC | brps league NC |
|---|---|---|---|---|
| 10k | 0.3887 | **0.3702** | 15.24 | **3.70** |
| 20k | **0.3305** | 0.3391 | 15.18 | 5.30 |
| 30k | **0.3242** | 0.3351 | 15.97 | 3.68 |
| 45k | **0.3240** | 0.3301 | 15.48 | **2.87** |
| 60k | **0.3235** | 0.3258 | 19.72 | 2.91 |

### 3.3 图

- `runs/league_probe/figs/ab_exploitability.png`：管线产物，双面板；仅 kuhn
  面板有数据（brps tag 缺口，§2.3）。
- `runs/league_probe/figs/ab_exploitability_supplement.png`：本调查以只读
  方式补绘的三面板图——kuhn exploitability、brps NashConv（log y）、
  brps TV 距离，mean + min–max 带（n=4），镜像管线图样式。

---

## 4. 收敛对比：第一结论

### 4.1 kuhn：6e4 预算内打平，差异在噪声内

终值 mean 几乎重合（mirror 0.32345 vs league 0.32579），且 seed 区间大幅
重叠。轨迹上 league 在 10k 处略快（0.3702 vs 0.3887），mirror 在
20k–45k 略领先，60k 处 league 再度追平——全程无统计意义的分离。
league 的 seed 间散布更宽（range 宽 0.0251 vs mirror 0.0073），与对手
混合采样 + exploiter 轮次注入的额外方差一致。注意 league 有 ~2/3 预算
花在 exploiter 轮次（§2.2），main 的有效 on-policy 剂量其实低于 mirror，
打成平手本身值得记录。

### 4.2 brps：league 显著更低更稳；mirror 出现 1/4 崩溃 seed

- **mirror 不收敛**：NashConv 均值全程横在 15–20，末点反而升至 19.72；
  seed_3 收敛到近纯策略（NashConv 50.0 ≈ 双方 BR 激励上限，TV 0.9375），
  余下 3 seeds 也停在 5.9–7.0。
- **league 持续下行**：NashConv 15k 后即落入 2–5 区间并总体走低
  （45k 2.87 → 60k 2.91），终值 {5.40, 2.11, 2.20, 1.94}，即使剔除
  mirror 的崩溃 seed 也低约 2.2×（2.911 vs 6.496），且无崩溃。
- **两指标在非崩溃 seed 上分歧**：TV 距离剔除 seed_3 后 mirror mean
  0.1519 ≈ league 0.1555（几何上同样接近 NE），但 NashConv 差 2 倍以上。
  解读：payoff 量级 ±50 的博弈里，偏离 NE 的**方向**比**距离**更致命——
  mirror 未崩溃 seed 停在高激励方向上，league 的策略落在低激励区域。
  激励加权口径（NashConv）下 league 明确占优，几何口径（TV）只是打平。

### 4.3 cycling 博弈是否呈现差异：是，且方向符合预期

BRPS 正是为暴露 latest-vs-latest 自对弈的 best-response cycling 而设
（`configs/games/brps.yaml`）。探测期内 mirror 呈现教科书式症状：均值
平台期 + 单 seed 灾难性崩溃；league 的历史池/PFSP 混合在 6e4 步内即表现
出抑制 cycling 的作用（更低、更稳、持续下行）。kuhn（回合制、收敛型）
上两臂无差异，说明该差异是博弈动力学驱动的，而非 league 的普遍
加速/减速。

---

## 5. 局限性

1. **探测深度**：6e4 步 = 论文预算的 0.6%。kuhn 终值 ~0.32 距 1e7 复现的
   0.0205（`docs/reproduce_report.md`）尚远，本文所有“谁快谁慢”仅描述
   早期动态，不得外推到渐近行为。
2. **横比口径**：同名 env-step 两臂剂量不同（§2.1），league 还有 ~2/3
   轮次是 exploiter 收集且梯度落在 main（§2.2）——“同预算”结论须带此
   双重脚注。
3. **oshi_zumo 缺席**：其规模下精确 NashConv 不可行（brps 配置注记，
   2026-07-22），cycling 结论仅靠 BRPS 一个代理博弈支撑。
4. **样本量**：n=4 seeds；kuhn 的平手与 brps 的崩溃率（1/4）都只是粗估。
5. **工具缺口**：`--summarize` 只认 `eval/exploitability`，brps 的 band 在
   `summary.json`/管线图中为空（本报告以只读提取 + 补充图覆盖）；brps
   末评估点因 episode 记账落在 60032 而非 60000。
6. **指标分歧**：brps 上 NashConv 与 TV 距离在非崩溃 seed 间结论不同
   （§4.2），单一指标会误导，两个都得报。

---

## 6. 后续建议

1. **brps 加深 + 加密**：预算提到 3e5–1e6 步确认 league 是否持续下行、
   mirror 平台期是否打破；seeds 扩到 8 以估计 mirror 崩溃率（当前 1/4）。
2. **kuhn 加 seeds**：同预算 8 seeds 复核平手结论；或在 2e5+ 步看 league
   早期优势（10k 处）是否随池子成熟重现。
3. **修 probe 的 tag 回退**：`--summarize` 按
   `eval/exploitability → eval/nash_conv → eval/exact_nash_distance` 回退
   （`eval/plots.py:113–114` 已有此约定），单独一个 commit（AGENTS.md
   §10 一次一事）。
4. **核查 exploiter→main 的 off-policy 路径**（§2.2）：确认 exploiter 轮次
   激活的 ratio 门控与 1/old_probs 权重是否符合设计预期；量化 main 更新中
   off-policy 数据占比（~2/3）对收敛的影响。这是 F2 下
   一步最值得先回答的问题。
5. **league 健康指标**：探测脚本增记 exploiter 胜率/晋升次数/池龄分布，
   便于把“league 更稳”归因到具体机制。
