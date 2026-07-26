# Liar's Dice 0.18-floor 消融（Phase A：β-sweep）

> 问题：ACH 在 Liar's Dice 上的 exploitability 贴在 ~0.18–0.20 下不去（论文 0.171）。
> 根本原因是**熵正则**（β=1e-2 把不动点推向软均衡），还是**优化/架构误差**？
>
> **结论（决定性）：不是熵正则。** 把 β 从 1e-2 一路降到 0，地板和策略熵都不动。
> 地板是 ACH 更新机制本身（门控 + 1/π_old + 非法 logit 漂移）封顶了策略锐化，
> 与 β 无关。→ 进入 Phase B，门控 l_th 成为首要去验证的封顶因素。

---

## 1. 实验设计

β-sweep：β ∈ {1e-2, 3e-3, 1e-3, 1e-4, 0}，LayerNorm + raw-logit 臂（论文忠实 mirror +
D16 平均追踪器），seed 0，跑满 1e7 env-steps，CPU。其余超参全同 anchor。β=1e-2 是已提交
的 anchor run；另 4 臂为 `configs/exp/liars_dice1_ach_mlp_beta_*.yaml`。

判定逻辑：若 current exploitability 随 β→0 **趋于 0**，地板是熵正则的软均衡点
（ACH 停在自己目标函数的地板上，无 bug，论文也在那）；若**卡在 ~0.18 不动**，地板是
优化误差，Phase B 接手。

---

## 2. 结果（seed 0，1e7 env-steps，exploitability = NashConv/2）

| β | current (tail-10%) | uniform-avg | best-iterate | tail 策略熵 |
|---|---|---|---|---|
| 1e-2 | 0.2053 | 0.1839 | 0.1805 | 1.038 |
| 3e-3 | 0.2044 | 0.1879 | 0.1772 | 1.111 |
| 1e-3 | 0.2012 | 0.1918 | 0.1838 | 1.039 |
| 1e-4 | 0.1985 | 0.1819 | 0.1856 | 1.001 |
| 0 | 0.2060 | 0.1882 | 0.1865 | 1.010 |

![β floor](figs/liars_beta_floor.png)

- **current exploitability 随 β 几乎不变**：β=1e-2 → 0.205，β=0 → 0.206（比值 1.03×）。
  best-iterate（~0.177–0.187）和 uniform-avg（~0.18–0.19）同样 β-不变。
- **策略熵也 β-不变**：tail 熵全程 ~1.0，β=0（无熵奖励）和 β=1e-2 一样停在 ~1.0。
  （早期有弱 β 效应：β=1e-2 首 3 点熵 1.26 vs β=0 1.19，确认 β 已接线生效，只是被地板吞掉。）

---

## 3. 判定

**地板不是熵正则。** 把 β 完全关掉（β=0），exploitability 仍 0.206、熵仍 1.01——与
β=1e-2 不可区分。这同时排除了两种熵解释：

1. ~~"β=1e-2 的软均衡 exploitability 就是 0.18"~~ —— 若如此，β=0 应让地板掉向 0；它没掉。
2. ~~"熵奖励把策略撑在软平台"~~ —— 若如此，β=0 应让策略锐化、熵下降；熵仍 ~1.0。

**真正机制：策略被钉在熵 ~1.0 的软平台上，且这个钉子与 β 无关。** β 只加熵、不减熵；
即使取消熵奖励（β=0），策略也锐化不下去——说明封顶的是 ACH 更新机制本身。与
`docs/reproduce_report.md` §6.2 一致：**门控（l_th=2.0）+ 非法动作 logit 漂移**把最优
合法动作的正优势更新误门掉（10M 步时 55.7% 信息态被误判饱和），策略无法锐化，停在
高熵软平台。β 在门控绑定之后才起作用，所以毫无杠杆。

> 副产物：Liar's Dice 的"平均 vs 当前"几乎无差距（0.21 vs 0.18，0.05 O，见
> `docs/average_policy_anchor.md`）也是同一现象——当前策略在软平台上窄幅振荡，
> 平均只能削掉这点振荡。地板是硬上限，不是振荡。

---

## 4. Phase B 计划（β-sweep 重新排序后的优先级）

熵-不变性把矛头从"1/π_old"（§6.7 原首要假设）转向**门控 l_th**：若门控封顶锐化，
抬高 l_th 应让熵下降、地板下降。逐臂单因子（固定 β=1e-2，LayerNorm 臂），跑满 1e7：

| 臂 | 测的假设 | 优先级（Phase A 后） |
|---|---|---|
| **B4. l_th sweep {1, 2, 4, 8}** | 门控阈值是否就是锐化封顶；熵-不变性最直接的预测 | **最高** |
| B1. 1/π_old 截断（floor π_old∈{0.01,0.05}） | §6.7：稀有动作 π_old→0 → 梯度爆炸封顶锐化 | 高 |
| B2. legalmean（§6.2 非法漂移）+ B4 组合 | 非法 logit 污染门控均值；与 l_th 是否叠加 | 高 |
| B3. 评论家：独立/多次 critic 更新；seqform BR 当 oracle 教师 | explained_variance≈0.15，GAE 优势噪声 | 中 |
| B5. 容量 {128,256,512} | 24576 信息态是否超出 128-MLP 表达 | 中 |

**B4 是 Phase A 的直接推论**：熵-不变性说"有东西在熵 ~1.0 处封顶"，门控是最自然的候选
（centered logit 触 l_th=2.0 即停更新）。若 l_th=8 让熵掉到 ~0.5 且 exploitability 掉到
~0.1 以下，门控就是元凶；组合 B2（legalmean）看是否把非法漂移的污染也一并解掉。

## 5. 方法论与限制

- **单 seed**：β-sweep 的结论（地板 β-不变）是 5 个 β 点一致的**趋势**，比单点排序稳健；
  仍建议 Phase B 的决赛臂补 2–3 seed。
- **β=0 未爆**：无熵奖励下 1/π_old 没有数值崩溃（exploitability 0.206、无 NaN）——
  说明地板不是 β=0 引发的失稳，而是β-无关的结构性封顶（弱化了"1/π_old 爆炸"作为
  地板主因的嫌疑，把它降到 B1 而非首位）。
- 全程报 **exploitability**（= NashConv/2），见 memory `exploitability-vs-nashconv-units`。

## 6. 代码与产物

- 配置：`configs/exp/liars_dice1_ach_mlp_beta_{3e-3,1e-3,1e-4,0}.yaml`（β=1e-2 =
  `liars_dice1_ach_mlp_anchor.yaml`）。
- 分析：`tools/beta_floor_sweep.py` → `docs/figs/liars_beta_floor.png`。
- run 数据：`runs/ab_beta/liars_beta*_seed0/train_curve.json`（gitignored）。
