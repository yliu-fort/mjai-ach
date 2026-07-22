# Kuhn 平局根因彻底分析（league vs mirror @ 6e4 env-steps）

> 任务：回答「kuhn 为什么拉不开差距」（首份 A/B 报告 `docs/league_investigation.md`
> 留下的问题）。方法：只读源码复核 + 现有 `runs/league_probe` 数据重解析 +
> 仪表化诊断对照臂（`runs/league_probe_diag/`，不改 `src/` 与 `configs/`）。
> 所有数字均来自实际运行/读取的数据；推测性解读均显式标注。
> 运行环境：RTX 3060 Ti（CUDA，`uv run`），2026-07-23 本机顺序执行。

---

## 0. 裁决摘要（根因排序）

| 排名 | 根因 | 一句话 |
|---|---|---|
| **R1（主因）** | **假设 (c)：6e4 预算内两臂都在同一条「激进度瞬态」上，且 exploitability 在该区域是浅盆地（盆地底 ≈ 0.31–0.33）** | 6e4 的 kuhn 不测量「谁更接近 Nash」，只测量「谁更深地滑进 always-bet 盆地」；机制差异对该指标在该预算内不可见 |
| **R2（放大器）** | **league 的多样性机械在探测预算内实际未激活（3 个「死机械」）** | 默认配置下 league 的对手分布 ≡ 100% current main（D0/seed_0 实测 1652/1652 轮），两臂「没有设计上那么不同」 |
| **R3（掩盖因子）** | **假设 (d)：统计功效不足 + league 的方差是镜像的 ~15 倍（SD 口径 4 倍）** | n=4 时可检测下限 \|Δ\|≈0.017，观测差 0.0023；league 的方差来自「晋升级联」这一双峰事件 |

被**排除**的推断：假设 (b) 的「打平 ⇒ league 单位剂量效率更高」——指标在盆地底无分辨力，且 league 在瞬态上的位置其实**落后**于 mirror（§5.2）。
**部分证实**：假设 (e)（off-policy 在该预算内对终值无害，但机制是门控阻尼+盆地平坦，不是「梯度方向碰巧一致」，§5.3）；假设 (a) 的弱形式（该预算内多样性无收益，D2 直接证实，§5.1）；其强形式（收敛型小博弈天然无效）在本预算内**不可裁决**。

---

## 1. 决定性事实：6e4 的 kuhn 是一条「aggressiveness 盆地」

### 1.1 两个锚点的精确 exploitability（OpenSpiel 精确计算，本调查实测）

| 策略 | exploitability | w_p1（BR0 对 P1 的提取值） | w_p0（BR1 对 P0 的提取值） | v0（自对弈局值） |
|---|---|---|---|---|
| 均匀随机 | 0.45833 | 0.50000 | 0.41667 | +0.12500 |
| MLP 初始网络（seed 0） | 0.46695 | — | — | — |
| **always-bet/call（12 态全押）** | **0.33333（精确 = 1/3）** | 0.33333 | 0.33333 | +0.14375 |
| Nash（参考） | 0 | −1/18 | +1/18 | −1/18 |

### 1.2 两臂终态策略表（12 信息态 P(bet/call)，逐 checkpoint 实测）

kuhn 的 Nash 要求**面对下注用 J 弃牌（P(call)=0）**、P0 首开 J 以 α≤1/3 下注、Q 过牌。实测 60k 终态：

| 臂 | mean P(bet)（12 态均值） | P0 首开 J（Nash ≤1/3） | **J 面对下注跟注（Nash = 0）** | 终态 entropy |
|---|---|---|---|---|
| mirror（seed_0） | 0.964 | 0.93 | P0 侧 0.95 / P1 侧 0.96 | 0.15–0.30（4 seeds） |
| league（seed_0） | 0.751 | 0.74 | P0 侧 0.73 / P1 侧 0.70 | 0.51–0.59（4 seeds） |

瞬态演进（seed_0，checkpoints 实测）：

| 阶段 | mirror meanPbet / expl | league meanPbet / expl |
|---|---|---|
| ~5k | 0.612 / 0.3917 | 0.680 / 0.3620 |
| ~30k | 0.846 / 0.3318 | 0.746 / 0.3284 |
| 60k | 0.964 / 0.3274 | 0.751 / 0.3114 |

两臂都在**单调冲向 always-bet**（Nash 的反方向维度上），exploitability 从
0.46 滑向盆地底 0.31–0.33。mirror 冲得更深（0.96，已略过盆地最优点），
league 停在 ~0.75。

### 1.3 几何口径佐证：两臂在策略空间里都**远离**了 Nash

到解析 Nash（α=1/3）的 12 态 mean-L1 距离（本调查实测）：

| 臂 | L1-to-Nash |
|---|---|
| 均匀随机（未训练） | **0.389** |
| league（4 seeds） | 0.413 / 0.413 / 0.422 / 0.425 |
| mirror（4 seeds） | 0.506 / 0.495 / 0.484 / 0.494 |

**训练 6e4 步后，两臂在策略空间上都比随机初始化离 Nash 更远**——exploitability
的下降（0.46→0.32）与到 Nash 的距离**方向相反**。这就是「盆地」的定量含义：
该预算段内 exploitability 只反映激进度，不反映均衡接近度。渐近参照：
mirror 同协议 1e7 步达 0.0205（`docs/reproduce_report.md`），其必经路径是
**逆转**「J 面对下注跟注」（0.96→0）——即先走出盆地。

### 1.4 逐侧弱点分解（60k 终态，OpenSpiel BR 值，seed_0/seed_1）

| 臂 | w_p1（P1 侧弱点 ↓） | w_p0（P0 侧弱点 ↓） | v0_profile |
|---|---|---|---|
| mirror | 0.3225 / 0.3136 | 0.3324 / 0.3318 | +0.009 / +0.009 |
| league | 0.3059 / 0.3479 | 0.3170 / 0.3252 | +0.157 / +0.202 |

注意 league 的 v0_profile 显著偏正（P0 每局净胜自家 P1 ~0.16–0.20）——
profile 内部更不对称，但 exploitability 对两座取平均后**掩盖**了这一差异。

---

## 2. league 机械在探测预算内的实际形态（源码复核 + 仪表化实测）

### 2.1 三个「死机械」（逐条给出证据）

1. **历史池从未获得 main 快照**。`main_save_every_steps = cfg.save_every_steps = 1000`
   （`src/mjai/scripts/experiment.py:229`，YAML `save_every_steps: 1000`），按
   **main 轮次**计数（`manager.py:94-98`）。探测期总轮次 ~1652、main 轮次
   ~551 < 1000 → 全预算内 **0 次 main 快照**（D0 全部 4 seeds 实测
   `pool_max.main = 0`）。30% history 桶恒为空 → 回退 current main
   （`opponent_sampler.py:97-99`）。
2. **PFSP 的 win_rates 从未被写入**。`CheckpointStore.update_win_rate`
   （`checkpoint_store.py:123`）在全仓库无任何调用方（grep 实证）→ 所有
   win-rate 恒为默认 0.5 → PFSP 退化为**桶内均匀采样**。
3. **`build_controller.copy_weights` 对 MLP 是静默 no-op**（`experiment.py:210-220`）：
   仅当 src/dst 同时有 `.logits`/`.values` 属性（tabular 形态）才拷贝；
   `MLPSharedActorCritic` 两者皆无。运行时验证：构造两个不同 seed 的 MLP 调用同
   逻辑后参数不变（param-abs-sum 255.7202 → 255.7202）。后果：**exploiter 从未被
   热启动、晋升后也从未被 reset**——整个运行期是两个冻结的随机初始化网络。
   晋升后 KL 不归零（D0/seed_1：晋升轮 kl=0.008 → 下一同角色轮 kl=0.092，
   全程 183 次晋升无一次回落）直接证实 reset 未生效。这违反 AGENTS.md
   「禁止静默 fallback」的精神（本次只读，未改）。

### 2.2 因此，默认配置的 league @6e4 实际是

- 对手分布：**≡ 100% current main**。D0/seed_0：1652/1652 轮对手为
  current_main（bucket 抽取 549/338/214 ≈ 0.50/0.31/0.19，符合 mix，但
  history/exploiter 桶全部落空回退）。seed_1–3 有 11–14% 轮次对手为池中
  **冻结随机网络**（晋升级联产物，见 §4），仍无任何「历史 main / 真 exploiter」。
- 梯度剂量结构（实测）：league 1652 轮 × 36.3 样本（**仅 seat-0**，
  `league_controller.py:100`）；mirror 929 轮 × 64.6 样本（双座）。每 6e4
  env-steps：league 更新 1652 次（mirror 的 1.78 倍），其中 main 轮 551 次
  = 20007 样本 on-policy（**恰好 1/3**），exploiter 轮 1101 次 = 40000 样本
  off-policy（**恰好 2/3**——复核了首份报告 §2.2 的占比估计，精确成立）。
- **只有 6/12 信息态拿到直接梯度**（seat-0 = P0 侧）：batch 只含 P0 转移，
  P1 输入从不出现在任何损失里。但实测 P1 侧弱点（w_p1 0.306–0.348）与
  mirror 的直接训练侧（0.314–0.323）相当——共享躯干 + value 耦合使 P1 行为
  随 P1 输入的「邻近结构」被动改善（机制性解读，标注为推测）。
- exploiter 轮的**优势信号用的是冻结随机 value 头**（rollout 记录的是
  exploiter 自己的 V），符号/幅度相对 main 的 baseline 有系统性失真风险；
  且 ratio 门控真实生效（见下）。

### 2.3 off-policy 的定量画像（TB `train/approx_kl` 按轮次 mod 3 分桶 + D0 门控率）

| 角色 | approx_kl 均值（4 seeds） | gate_off_frac（D0，4 seeds） |
|---|---|---|
| main（同步采集） | ~1e-10（精确 on-policy） | **0.000**（l_th 门控全程未触发） |
| main_exploiter 轮 | 0.095 / 0.164 / 0.154 / 0.146 | 0.109 / 0.361 / 0.446 / 0.388 |
| league_exploiter 轮 | 0.167 / 0.137 / 0.094 / 0.131 | 0.457 / 0.326 / 0.114 / 0.264 |

- 首份报告 §2.2「ratio 门控在 exploiter 轮真实生效」**定量证实**：exploiter 轮
  11–46% 样本被门掉（c=0），main 轮 0%。
- 门控带宽推断（标注为推断）：对近均匀行为策略（old_prob≈0.5），ratio∈(0.5,1.5)
  ⇔ main 概率 ∈ (0.25, 0.75)——与 league 的 P0 押注概率停在 0.70–0.83、entropy
  停在 0.51–0.59 一致；mirror 则顶到 l_th 门控上限（P(bet)≈0.98 ≈ 中心化 logit
  2.0 对应的 0.982）。

---

## 3. 诊断对照臂（`runs/league_probe_diag/`，仪表化运行器 `run_diag.py`）

方法：monkeypatch 纯观察者 wrapper（不消耗任何 RNG）+ `dataclasses.replace`
覆写配置，训练循环是真实的 `run_experiment`。**D0（默认配置）与原始
`runs/league_probe/kuhn_league/seed_{0..3}` 的 12 个评估点逐位一致**（4/4 seeds
验证），证明观察者无扰且运行可复现。每臂 6e4 步、eval 每 5e3。

| 臂 | 配置改动 | seeds | 终值 exploitability | 关键观测 |
|---|---|---|---|---|
| D0 | 无（≡ 原始 league 臂） | 0–3 | 0.31141 / 0.33656 / 0.32134 / 0.33386 | 见 §2；晋升数 0 / 183 / 33 / 89 |
| D1 | mix=(1,0,0)（配置层面零多样性） | 0–1 | **0.31141** / 0.33371 | seed_0 与 D0 **完全相同**（池本就恒空，逐点相等）；seed_1 有 224 次晋升但按 mix 永不采样，与 D0 差 0.003（噪声内） |
| D2 | save_every_steps=25（**真实激活历史池**） | 0–1 | 0.31074 / 0.33797 | seed_0：291/1651 轮（17.6%）对手为历史 main 快照（pool:main#0..20），0 次晋升；终值与里程碑与 D0 几乎重合（10k 0.36235 vs 0.36197；45k 同为 0.31984） |

诊断臂终值与参照（60k）：

```
mirror 4 seeds : 0.32744 0.32268 0.32012 0.32354   mean 0.32345  SD 0.00303
league 4 seeds : 0.31141 0.33656 0.32134 0.33386   mean 0.32579  SD 0.01166
D1 (1,0,0)     : 0.31141 0.33371
D2 (真历史池)  : 0.31074 0.33797
```

全部落在同一盆地底 0.31–0.34。**无论把多样性关掉（D1）还是真正打开（D2），
kuhn @6e4 的终值都不动。**

---

## 4. league 方差来源：晋升级联是双峰事件（逐 seed 实测）

| seed | 晋升次数（me/le） | 池对手轮次占比 | 池构成（max） | 终值 |
|---|---|---|---|---|
| 0 | **0** | 0.0% | 空 | **0.31141** |
| 1 | 43/140 | 13.3% | 16 个冻结随机 le | 0.33656 |
| 2 | 33/0 | 11.1% | 16 个冻结随机 me | 0.32134 |
| 3 | 41/48 | 14.0% | 7 me + 16 le（冻结随机） | 0.33386 |

晋升信号 `won := mean(batch.advantages) > 0`（`league_controller.py:108-115`）
使用的是**冻结随机 value 头**算出的 GAE——其偏置在初始化时被随机确定，于是
「晋升/不晋升」成为逐 seed 的双峰事件；池成员全部是**冻结随机网络的快照**
（reset 未生效，§2.1-3）。这是 league 终值 SD 4 倍于 mirror（0.0117 vs
0.0030，方差比 ~14.8）的主要候选机制（证据：seed_0 无级联且终值最优；
3 个有级联的 seed 落在 0.321–0.337；n=4，标注为候选机制而非定论）。
mirror 无此机制。

统计功效（实测终值计算）：Welch t = −0.389（df≈3.4，双侧 p≈0.72）；
合并 Cohen's d ≈ 0.28。n=4/臂、α=0.05、80% 功效的可检测下限 d≈2.0，
即 \|Δmean\|≳0.017——观测差 0.0023 低于可检测下限约 7 倍。

---

## 5. 逐假设裁决

### 5.1 假设 (a)：信息态太少，池机制天然无效 —— 弱形式证实，强形式不可裁决

- **证实（弱，本预算内）**：D2 把历史池真正激活（17.6% 轮次对历史 main）
  后终值不动（0.31074 vs 0.31141）；D1 与 D0 在池恒空的 seed_0 上逐点相等。
  该预算内多样性对 kuhn 无可见收益。
- **不可裁决（强）**：默认配置下多样性机械从未激活（§2.1），且 R1 的盆地
  会掩盖任何机制差异。「收敛型小博弈上 league 天然无效」需要 ≥1e6 预算
  （走出盆地）才能判定。

### 5.2 假设 (b)：剂量差异 ⇒ league 单位剂量效率更高 —— 事实确认，推断被排除

- 事实：league main 的 on-policy 剂量 = 1/3（20007/60007 样本），直接梯度
  只覆盖 6/12 信息态，但更新次数是 mirror 的 1.78 倍（1652 vs 929）。
- 推断不成立：终值打平是因为**盆地底无分辨力**——league 在瞬态上的位置
  其实**落后**于 mirror（meanPbet 0.75 vs 0.96；entropy 0.51–0.59 vs
  0.15–0.30），却拿到同一分数。「效率」在该预算内不可定义，更不可比较。

### 5.3 假设 (e)：exploiter 轮 off-policy 数据无害（梯度方向碰巧一致）—— 部分证实

- 证实：2/3 预算吃 off-policy 数据（KL 0.09–0.17），终值仍打平。
- 机制更正：「无害」的实现方式是 **ratio 门控阻尼**（11–46% 样本被门掉、
  剩余被 [0.25,0.75] 带宽限幅 + 1/old_probs 重权）**叠加盆地平坦性**，而非
  梯度方向一致。代价可测：league 的瞬态推进被显著拖慢（§5.2）。且 exploiter
  轮的优势信号基于冻结随机 value 头，方向失真风险是结构性的——「无害」
  是预算与指标相关的局部现象，不是普遍许可。

### 5.4 假设 (c)：两臂同处一条瞬态 —— 证实（主因）

证据链：§1.2 策略表（同向冲向 always-bet）；§1.1 always-bet 精确 1/3 ≈
盆地底；§1.3 L1-to-Nash 显示两臂都在**远离** Nash（exploitability 却在降）；
10k 处 league 略快、20k–45k mirror 略领先的轨迹交叉正是「同一曲线上不同
速度」的形态（逐 seed 差在 ±0.03 内震荡）。6e4 = 论文预算 0.6%，距
「学弃牌」的盆地出口尚远。

### 5.5 假设 (d)：方差掩盖真实但小的差异 —— 证实（作为掩盖因子）

league SD 0.0117 vs mirror 0.0030；可检测下限 0.017 ≫ 观测差 0.0023（§4）。
即便存在真实小差异，当前设计与 n=4 也无法分辨。league 的额外方差溯源到
晋升级联双峰性 + 冻结 exploiter 的随机初始化 KL/门控剖面（§4）。

---

## 6. 机制链总结（一句话版）

6e4 的 kuhn 把两臂放在同一条「冲向 always-bet」的瞬态上（R1），exploitability
在盆地底对机制差异失明；而 league 的多样性机械因配置失配 + 一个静默的
copy_weights no-op 从未真正上线（R2），两臂实际差异只剩「seat-0 半树 batch
+ 2/3 冻结 bot 采集的门控 off-policy 数据」——这点差异恰好也被盆地吞掉；
残存的终值差（0.0023）远低于 league 晋升级联制造的方差可检测下限（R3）。

---

## 7. 对 league 在收敛型博弈上定位的建议

1. **不要用 kuhn @≤6e4 做 league 的判别实验**（盆地效应）。cycling 敏感性
   用 brps（首份报告已显示 league 显著占优）；kuhn 若要判别，预算需 ≥1e6
   （参照 mirror 复现：2–4e6 进入 ~0.05 平台、1e7 达 0.0205），n≥8。
2. **修 `build_controller.copy_weights`**：改用 `Policy.snapshot_state()` /
   `restore_state()` 的泛型实现（当前对 MLP 静默 no-op，违反 AGENTS.md
   「禁止静默 fallback」）。修复后 exploiter 才会成为「可重置的 main 副本」，
   晋升/重置/池构成全部改变，本报告的 league 侧数值需重测。单独一个 commit
   （AGENTS.md §10）。
3. **`main_save_every_steps` 与预算解耦**：当前硬接 `save_every_steps=1000`，
   在 6e4 探测预算下恒不触发。建议按 env-step 口径配置（如 total 的 5%），
   或在探测配置里显式调小（本调查 D2 用 25 验证可行）。
4. **PFSP 要么接线要么声明**：`update_win_rate` 无调用方 → 实际为均匀采样。
   接上胜率统计，或在文档/配置里把采样声明为 uniform，消除名实不符。
5. **探测脚本增记 league 健康指标**：晋升次数、池构成、per-role KL、
   门控率、updates 数、双座样本数（本调查的 `league_log.jsonl` 可作模板；
   首份报告 §6.4/§6.5 已提出，本调查证实其必要性——没有这些指标，
   §2.1 的三个死机械在首份报告中不可见）。

---

## 8. 局限性

1. D1/D2 各 2 seeds，只覆盖 mix=(1,0,0) 与「激活历史池」两个对照点；
   mix=(0,·,·) 臂未跑（在本预算内会因桶空恒回退 current main，信息量低）。
2. 所有结论均为 6e4 预算（论文 0.6%）内的早期动态，不对渐近行为作任何断言；
   「league 在收敛型博弈的渐近价值」仍然开放。
3. §2.3 的门控带宽 [0.25,0.75] 是从门控公式 + 观测概率的推断，非逐样本日志。
4. §4 的「晋升级联 → 终值方差」是 4 seeds 上的候选机制，方向一致但样本小。
5. P1 侧「共享躯干被动改善」为机制性推测（实测现象：未训练侧弱点与训练侧
   相当，但归因未做消融）。

---

## 9. 附录：关键命令与产物

```bash
# 现有数据重解析（per-seed 曲线、里程碑、按角色 KL、功效统计）
uv run python runs/league_probe_diag/analyze_existing.py        # -> existing_analysis.json
# 逐玩家/逐侧 NashConv 分解 + 12 信息态策略表
uv run python runs/league_probe_diag/per_player_decomp.py
uv run python runs/league_probe_diag/per_side_weakness.py       # -> per_side_weakness.json
# 仪表化诊断臂（D0 默认 / D1 mix=(1,0,0) / D2 save_every=25）
uv run python runs/league_probe_diag/run_diag.py --arm D0 --seed 0   # … seeds 0-3
uv run python runs/league_probe_diag/run_diag.py --arm D1 --seed 0   # … seeds 0-1
uv run python runs/league_probe_diag/run_diag.py --arm D2 --seed 0   # … seeds 0-1
uv run python runs/league_probe_diag/aggregate_diag.py          # -> diag_digest.json
```

产物：`runs/league_probe_diag/{D0,D1,D2}/seed_*/`（train_curve.json、tb/、
checkpoints/、league_log.jsonl、DONE）、`existing_analysis.json`、
`diag_digest.json`、`per_side_weakness.json`、四个分析/运行脚本
（`analyze_existing.py`、`per_player_decomp.py`、`per_side_weakness.py`、
`run_diag.py`、`aggregate_diag.py`）。
