# Liar's Dice 0.18-floor 消融：根因 = 无界 1/π_old

> 问题：ACH 在 Liar's Dice 上的 exploitability 贴在 ~0.18–0.20 下不去（论文 0.171）。
>
> **根因（两阶段消融定位）：无界的 1/π_old 重要性权重。** 它逼着 ACH 门控（l_th）保持
> 紧——一放松（l_th≥8）策略锐化到 π_old→0、1/π_old 爆炸、logit 失稳崩溃；而紧门控把
> 策略锐化封顶在熵 ~1.0，这个封顶就是 ~0.20 地板。门控是直接封顶，1/π_old 是深层根因。
> 直接印证 `docs/reproduce_report.md` §6.7 的首要开放假设。

两阶段：
- **Phase A（β-sweep）**：地板 β-不变 → 排除熵正则。
- **Phase B4（l_th sweep）**：门控确实是锐化封顶（熵随 l_th 单调降），但放松门控要么
  无益（地板鲁棒）、要么崩溃（l_th≥8 → 1/π_old 爆炸）→ 1/π_old 才是逼出门控紧、进而
  逼出地板的深层因素。

---

## 1. Phase A — β-sweep：排除熵正则

β ∈ {1e-2, 3e-3, 1e-3, 1e-4, 0}，LayerNorm + raw-logit 臂，seed 0，1e7 env-steps，CPU。
β=1e-2 是已提交 anchor run；另 4 臂为 `configs/exp/liars_dice1_ach_mlp_beta_*.yaml`。
全部报 exploitability = NashConv/2（仓库/论文口径）。

| β | current (tail) | uniform-avg | best-iter | tail 策略熵 |
|---|---|---|---|---|
| 1e-2 | 0.2053 | 0.1839 | 0.1805 | 1.038 |
| 3e-3 | 0.2044 | 0.1879 | 0.1772 | 1.111 |
| 1e-3 | 0.2012 | 0.1918 | 0.1838 | 1.039 |
| 1e-4 | 0.1985 | 0.1819 | 0.1856 | 1.001 |
| 0 | 0.2060 | 0.1882 | 0.1865 | 1.010 |

![β floor](figs/liars_beta_floor.png)

- **current exploitability 随 β 几乎不变**（β=1e-2 → 0.205，β=0 → 0.206，比值 1.03×）；
  best-iterate、uniform-avg 同样 β-不变。
- **策略熵也 β-不变**（全程 ~1.0；β=0 无熵奖励仍停在 ~1.0）。
- 判定：地板**不是熵正则**。β 只加熵、不减熵；即使取消熵奖励，策略也锐化不下去——
  封顶的是 ACH 更新机制本身。β 在该封顶绑定之后才起作用，毫无杠杆。

## 2. Phase B4 — l_th sweep：门控是锐化封顶，1/π_old 是深层根因

Phase A 的"熵-不变性"最直接预测：**门控 l_th 封顶了锐化**（centered logit 触 l_th 即停
更新）。扫 l_th ∈ {1, 2(anchor), 4, 8, 1e6(=no gate)}，β=1e-2 固定，seed 0，1e7。配置
`configs/exp/liars_dice1_ach_mlp_lth_*.yaml`。

| l_th | current (tail) | uniform-avg | best-iter | tail 策略熵 | 状态 |
|---|---|---|---|---|---|
| 1 | 0.455 | 0.452 | 0.447 | 1.588 | 完整（门更紧 → 更软 → 更差）|
| 2 | **0.205** | 0.184 | 0.181 | 1.038 | 完整（anchor，最佳稳定点）|
| 4 | 0.238 | 0.226 | 0.184 | 0.927 | 完整（轻度锐化，地板并未变好）|
| 8 | 0.66 | 0.62 | 0.53 | 0.101 | **崩溃 @1.2e6**（1/π_old 爆炸 → illegal action）|
| 1e6 (no gate) | 0.49 | 0.54 | 0.44 | 0.112 | **崩溃 @5e5**（同上）|

![l_th floor](figs/liars_lth_floor.png)

三条结论：

1. **门控就是锐化封顶（Phase A 预测确认）**：策略熵随 l_th **单调下降** 1.59 → 1.04 →
   0.93 → 0.10 → 0.11。Phase A 的"策略钉在熵 ~1.0"正是 l_th=2 门控在绑定。
2. **但放松门控救不了地板**：l_th=4 只把熵从 1.04 降到 0.93，地板反而略升（0.205→0.238）；
   地板对中等 l_th 鲁棒。l_th=2 是稳定点中的最佳。
3. **激进放松直接崩**：l_th=8 / no-gate 把熵压到 ~0.1（策略剧烈锐化），随后 π_old 在稀有
   动作上 →0，无界的 1/π_old 项 `η·y·c·A/π_old` 爆炸（lth8 崩前 pterm_max=258 vs
   baseline ~10），logit 失稳，rollout 选出 illegal action（`bidnum=-1`）崩溃。

→ **门控的"紧"是被 1/π_old 稳定性逼出来的**：不紧就崩。所以深层根因是 1/π_old，门控是为
保 1/π_old 稳定而不得不紧的"刹车"，副作用就是熵 ~1.0 的封顶 = ~0.20 地板。

## 3. 根因链

```
无界 1/π_old（§6.7）
   └─► 门控必须保持紧（l_th≈2；放松到 ≥8 则 π_old→0 → 1/π_old 爆炸 → illegal action 崩）
          └─► 紧门控把策略锐化封顶在熵 ~1.0
                 └─► 熵 ~1.0 的软平台 = exploitability ~0.20 地板
                        └─► 平均策略也只比当前低 0.05 O（窄幅振荡，见 average_policy_anchor.md）
```

门控是地板的**直接**封顶；**1/π_old 是逼出门控紧、进而逼出地板的深层根因。** 这把
`reproduce_report.md` §6.7 列为"最贴合失败特征"的 1/π_old 假设，从推测升级为定位结论：
唯一能让熵突破 ~1.0 的设置（l_th≥8）恰恰触发 1/π_old 崩溃。

## 4. 修复方向（未跑，下一实验）

根因是 1/π_old，所以修复是**解耦门控与 1/π_old 稳定性**：

- **B1 + B4 组合臂**：先给 1/π_old 加下界（floor π_old∈{0.01,0.05}，即 importance
  weight 截断），再抬高 l_th（{4,8}）。预测：截断 1/π_old 后可以安全锐化（熵降到 ~0.3–0.5
  而不崩），exploitability 掉到 ~0.1 以下。这才是"破地板"的正面实验。
- 注意 B1 单独（floor 1/π_old 但仍 l_th=2）可能不够——门控本身还在 ~1.0 封顶；需 B1+B4
  联用。B2（legalmean，§6.2 非法漂移）可与 B1+B4 叠加看是否进一步改善。

> 这是"测修复"，超出"研究根因"本身——故未自动启动，留作下一步决策。

## 5. 方法论与限制

- **单 seed**：β-sweep（5 点趋势）与 l_th 熵单调性都鲁棒；决赛修复臂（B1+B4）需 2–3 seed。
- **崩溃臂的数据仍可用**：l_th=8/no-gate 崩前已显示熵→~0.1（锐化发生）和 pterm→258
  （1/π_old 爆炸），正是定根因的关键证据。
- 全程报 **exploitability**（= NashConv/2），见 memory `exploitability-vs-nashconv-units`。

## 6. 代码与产物

- Phase A 配置：`configs/exp/liars_dice1_ach_mlp_beta_{3e-3,1e-3,1e-4,0}.yaml`。
- Phase B 配置：`configs/exp/liars_dice1_ach_mlp_lth_{1,4,8,1e6}.yaml`（l_th=2 = anchor）。
- 分析：`tools/beta_floor_sweep.py`、`tools/lth_floor_sweep.py` →
  `docs/figs/liars_beta_floor.png`、`docs/figs/liars_lth_floor.png`。
- run 数据：`runs/ab_beta/`、`runs/ab_lth/`（gitignored）。
