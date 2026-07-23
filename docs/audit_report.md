# mjai-ach 只读静态审计报告：NN PPO/ACH 最佳实践 & ACH 论文保真

> 审计对象：NN 版 PPO / ACH 实现及其数据管线。
> 论文基准：Fu et al., ICLR 2022（OpenReview `DTXZqTNV5nW`），事实以
> `docs/paper_spec_ach.md` 为准（页码 = 本地 `ach_paper.pdf` PDF 页码）；关键处
> 已对 `docs/ach_paper_fulltext.txt` 原文复核（Algorithm 2 在 p24；H.3 在 p27–28）。
> 本报告只读代码，未做任何修改。所有结论标注 file:line 或论文页码。

---

## 1. 摘要

- **PPO 结论**：骨架正确（GAE、clip 0.2、grad-norm 0.5、动作 mask、价值 MSE、
  entropy bonus 都在），但为与 ACH 公平对比做了多处**有意偏离** 37-details
  （单 epoch 全 batch、无 minibatch shuffle、无 LR 退火），已显式标注；另有
  **无意偏离**：非正交初始化、Adam eps 未调、KL 早停恒不触发、`target_kl`/
  `n_epochs`/`beta` 为死参数。整体是"可用的简化 PPO"，不是 37-details 完整版。
- **ACH 结论**：当前 `NNACHUpdate` **不是论文 ACH**。它是 NeuRD 风格的
  对称双侧 logit 门控 + 优势归一化 + Adam，与论文 Algorithm 2/Eq.29（p24）的
  优势符号依赖单侧门控、无归一化、SGD 恒定 lr 相差显著。门控形态、归一化、
  优化器三处叠加，有效学习动力学与论文不是同一个算法。
- **最重的 3 个发现**：
  1. **门控错误**（nn_updates.py:194）：`within=(y_a.abs()<=l_th)` 是对称双侧
     门控，论文是符号依赖单侧门控 + ratio 门控（p24）；对称门控会把"反方向
     越界"样本的纠正梯度一并清零，样本只能靠熵项缓慢救回（准冻死）。
  2. **超参系统性偏离且硬编码**：l_th=1.0（论文 2.0）、Adam（论文 SGD lr=1e-3）、
     优势归一化（论文无）、value_coef=0.5（论文 α=2.0），且 eta/l_th 在代码里
     不在 YAML（违反 AGENTS.md §9）。
  3. **实验配置层无法复现论文**：`ExperimentConfig` 没有 hidden_sizes/activation
     字段（论文结构 (128,)+ReLU 配不出来），`build_policy` 对 mlp 强制
     `require_cpu()`（experiment.py:74，与 D6 冲突），且唯一的 ACH YAML 是
     tabular 配置；`n_steps`/评估节奏单位是 round 而非论文的 training steps。

---

## 2. 发现清单表

| ID | 区域 | 分类 | 严重度 | 证据 (file:line) | 论文页码 | 建议修复 |
|---|---|---|---|---|---|---|
| F1 | ACH 门控 | BUG（相对论文） | 高 | nn_updates.py:190–195（`within` 对称双侧门控、无 ratio 门控） | p24 Alg.2/Eq.29 | 改为符号依赖单侧门控 + ratio 门控（见 §5 施工单 W1） |
| F2 | 优势归一化 | 偏离论文 | 高 | nn_updates.py:156–158 → `_normalize_advantages` :67–73 | p24 无归一化 | ACH 路径去掉归一化；PPO 保留（W2） |
| F3 | 优化器 | 偏离论文 | 高 | nn_updates.py:55（Adam） | p27 Table 7（SGD 恒定 lr=1e-3） | ACH 用 SGD(constant 1e-3)（W3） |
| F4 | l_th / eta 硬编码且值错 | 治理(§9)+偏离 | 高 | nn_updates.py:134（l_th=1.0）、:137（eta=1.0） | p28 Table 8（l_th=2.0）、p27（η=1.0） | l_th 改 2.0；两者移入 YAML（W4） |
| F5 | value_coef 不符且不可配 | 偏离论文 | 中 | nn_updates.py:208；update_rule.py:36（0.5）；experiment.py:83（不传） | p27 Table 7（α=2.0） | 暴露到 YAML；论文等效 value_coef=1.0（W5） |
| F6 | 损失用减均值 logit | 实现选择（歧义 A3） | 中 | nn_updates.py:190–191（`centered` 进损失） | p24（Alg.2 写原始 y；正文说减均值） | 保留减均值并在 docstring 标注 A3；梯度差异见 §3.B6 |
| F7 | n_epochs / beta 死参数 | BUG（接口说谎） | 中 | nn_updates.py:118–120,129；:173 `for _ in range(1)` | p24（single mini-batch 单次更新） | 删除或接线；docstring 同步（W7） |
| F8 | target_kl 恒不触发 | 低 | nn_updates.py:225–226（单 epoch 循环内 break 无意义） | — | 单 epoch 下移除或改到 round 级（W7） |
| F9 | docstring 自相矛盾 | 文档错误 | 中 | nn_updates.py:11–14（"REINFORCE+baseline，no clipping"）vs :182（"NeuRD direct-logit loss"）；NNACHUpdate :273–274（"just REINFORCE + entropy regularizer"） | — | 改写时统一为论文 Eq.29 描述（W8） |
| F10 | batch 大小与论文不符 | 偏离论文 | 中 | experiment.py:43（episodes_per_round=50）；rollout 全量进 batch | p28 Table 8（batch 64） | 按样本数 64 收集（W6） |
| F11 | n_steps / eval 单位语义 | 偏离论文+文档 | 中 | experiment.py:166（step=round）、:175（eval_every 单位 round） | p25（每 1e5 training steps）、p26（总长 1e7） | 引入 env-step 计数器（W9） |
| F12 | discount / pool_both_players 死配置 | BUG（配置无效） | 低 | rollout.py:33,38 定义后无人消费；update_rule.py:39–40 同死 | — | 删除或接线（W10） |
| F13 | gae_lambda 未从 AlgoConfig 接线 | 低 | experiment.py:110（RolloutConfig 只传 n_episodes/seed） | — | 显式传 gae_lambda=0.95（W10） |
| F14 | MLP 结构不可配 + 强制 CPU | 治理(D6 张力)+偏离 | 高 | experiment.py:74（require_cpu）、:77（默认 (128,128)+Tanh）；ExperimentConfig :35–62 无结构字段；mlp.py:48–49 | p25（1×128 FC+ReLU 双头） | 加 hidden_sizes/activation/device 字段；移除强制 require_cpu（W11） |
| F15 | 权重初始化非正交 | 偏离最佳实践 | 低 | mlp.py:66–74（PyTorch 默认初始化） | —（37-details #8） | PPO 侧可加正交初始化；ACH 复现为对齐论文可不做，需声明 |
| F16 | Adam eps 未调 | 偏离最佳实践 | 低 | nn_updates.py:55（默认 1e-8） | —（37-details #3 建议 1e-5） | PPO 侧设 eps=1e-5 |
| F17 | eval 吞异常 | 治理(no-silent-fallback 精神) | 中 | nash.py:128–132；experiment.py:253（`contextlib.suppress(Exception)`） | — | 至少记 warning；训练曲线缺列要有痕（W12） |
| F18 | Transition.reward 语义错误 | 文档/数据 | 低 | rollout.py:131（每步 reward=终局 return，非即时奖励） | — | 改注释或填即时奖励（下游未用，低危） |
| F19 | 复现配置缺失 | 治理 | 中 | configs/exp/kuhn_ach_mirror.yaml 是 tabular；无 mlp ACH 复现配置 | p25 | 新增三博弈 mlp ACH 配置（W13） |
| F20 | 测试锁定现状而非论文 | 测试缺口 | 中 | test_algos_nn_updates.py:105–117（锁定"归一化后 A=0 → loss=0"）；无门控测试、无 SGD 测试 | — | 改写时同步重写（W14） |
| F21 | GPU 决定论未设置 | 可复现性 | 低 | 全仓无 `torch.use_deterministic_algorithms`/cudnn 设置；mlp.py:62 全局 manual_seed | — | 记录限制；需要时加 deterministic flag |
| F22 | TabularACHUpdate 名为 ACH 实为 CFR+ wrapper | 文档 | 低 | tabular_updates.py:174–202（docstring 已自述） | — | 与 NN 改写无关；保留但注意命名误导 |

---

## 3. 逐条详述

### A. 数据与优势估计

**A1（Q1：advantages/returns 怎么算）**
`rollout.py:_assign_returns`（147–178）按玩家分组做**每玩家 GAE(λ)**：
- λ 来自 `RolloutConfig.gae_lambda`（rollout.py:34，默认 0.95），但
  `experiment.py:110` 构造 `RolloutConfig` 时**不传 gae_lambda** → 永远用默认
  0.95，且与 `AlgoConfig.gae_lambda`（update_rule.py:40，本身也是死字段）不连通（F13）。
- γ 实际**硬编码为 1.0**：`delta = 0.0 + next_value - t.value`（rollout.py:173）
  无 γ 因子；`RolloutConfig.discount`（:33）定义后无人消费（F12）。
- 终局 bootstrap 正确：最后一跳 `delta = r − V(s)`（:175–176），终局后
  next_value=0，不会 done 后再加 V ✓。
- `batch.returns` = 该玩家终局收益 `returns[player]`（:168）——即论文的采样
  回报 G ✓（零和：player1 的 r 是 −r0，正确）。
- γ=1.0 对这三个短episode、纯终局奖励的博弈与论文假设 A1（spec §4）兼容；
  但"γ 可配置"目前是假象。

**A2（Q2：双方样本进 batch；ratio 是否恒 1）**
- mirror 模式：`MirrorSelfPlay.collect` 用同一个 policy 对象打两个座位
  （controller.py:131），`run_episode` 无条件 pool 双方 transitions
  （rollout.py:82–88；`pool_both_players` 标志从未被消费，F12）。
  → 双方样本同批训练 ✓，与论文"both players use the same θ and ω"
  （p24，fulltext 1748 行）一致。
- 行为策略=当前 learner：进程内 `Trainer.step` 先 collect 后 update
  （controller.py:91–95），采集与更新之间权重不变 → `batch.logprobs` 就是
  当前策略的 logprob，**ratio 恒为 1** ✓，与论文 p28 单线程注记
  （"π(a|s;θ) is always identical to π_old(a|s)"，fulltext 1982–83）一致。
  `_ray.py`/`LocalIMPALARunner`（_ray.py:131–133）也是同步 collect，
  无真实异步；论文 IMPALA 异步在本仓库 Phase 1 不存在 → ratio clip 门控
  与论文同样 vacuous。

**A3（Q3：每 round 样本量；n_steps/eval 单位）**
- `episodes_per_round=50`（experiment.py:43；kuhn_ach_mirror.yaml:7）。
  用均匀随机策略实测平均每局决策数（含双方，uv run 只读验证）：
  Kuhn ≈ 2.24、Leduc ≈ 3.65、Liar's Dice ≈ 3.35 → 每 round batch ≈
  **112 / 183 / 168 样本**，是论文 batch=64（p28 Table 8）的 ~2–3 倍（F10）。
- `n_steps` 语义 = **train round 数**（experiment.py:166 的循环变量；每 round =
  50 局采集 + 1 次全 batch 更新）。不是环境步、不是更新次数（虽然当前
  1 update/round 使"round 数=更新次数"恰好成立）。
- `eval_every_steps` 单位同样是 round（experiment.py:175）。
- 与论文对应（按 spec A2 环境步口径）：每 round ≈ 112–183 环境决策步，
  论文每 1e5 步评估 ≈ 每 ~550–900 rounds；总长 1e7 步 ≈ 5.5万–9万 rounds。
  kuhn_ach_mirror.yaml 的 n_steps=1000 仅约 1.1e5 环境步（论文总长的 ~1%），
  eval_every=200 rounds ≈ 2.2e4 环境步（比论文密 ~4.5 倍）。F11。

### B. ACH 保真（对照 spec §1.2 / p24）

**B4（Q4：门控）** —— 确认代码现状：
```python
centered = logits - logits.mean(dim=-1, keepdim=True)   # :190
y_a = centered.gather(...)                              # :191 采样动作的减均值 logit
within = (y_a.abs() <= self.l_th).float()               # :194 对称双侧门控
ach_loss = -(self.eta * y_a * adv * within / (old_probs + 1e-8)).mean()  # :195
```
与论文差异三点：
1. **双侧 vs 单侧**：论文 A≥0 只查 `y−ȳ < l_th`（下侧越界不门），A<0 只查
   `y−ȳ > −l_th`；仓库 `|y_a|≤l_th` 两侧都门。
2. **无 ratio 门控**：论文还有 `1{ratio<1+ε}`/`1{ratio>1−ε}`。单线程下该门控
   vacuous（p28），所以这条**行为上无差异**，但结构上缺失，未来真异步时会
   偏离。
3. **冻死分析**：被门样本的 policy 梯度整项为 0。论文单侧门控下，"回拉"方向
   永不 blocked（门只挡"继续越界"方向）；仓库对称门控下，若某样本的 y_a
   在**与优势推力相反的一侧**越界（如 A>0 想把 logit 推上去，但 y_a 已
   ≪ −l_th），纠正梯度也被清零 → 该样本只剩 entropy 项（系数 1e-2）缓慢把
   logit 拉回门内。不是严格永久冻死（entropy 仍在、门控每步按当前 y_a 重估），
   但 policy-gradient 的自我纠正通道被切断，饱和状态下恢复极慢。这与仓库自己
   在 tabular_updates.py:36–39 注释里描述的"frozen"失效模式同构。

**B5（Q5：优势归一化）**：`step` 无条件调用 `_normalize_advantages`
（nn_updates.py:156–158）。论文无归一化（p24 损失里 A 原样进入）。
影响：归一化把每批优势的尺度压到单位方差，等效于给 η=1.0 乘上一个
**逐批自适应的 lr 缩放**（Kuhn 收益 ±1、Leduc 可达 ±14 时差异巨大），还改变
正负样本的相对权重（减均值）。这破坏了论文 η(s) 作为 Hedge 学习率的语义，
是 ACH 保真上仅次于门控的重大偏离（F2）。注意 `_normalize_advantages` 在
std<1e-8 时仍减均值（:72），而 `test_constant_advantages_ach_policy_loss_is_zero`
（tests :105–117）恰好把这个行为锁成了测试（F20）。

**B6（Q6：减均值进损失）**：确认仓库把 `centered` 用于损失本体（:190–195），
论文 Algorithm 2 损失写原始 `y(a|s;θ)`、均值只出现在门控（p24；spec 歧义 A3
已裁定实现取减均值）。梯度差异：原始 y 时样本梯度只流向动作 a 的 logit
（∂L/∂y_a = −c·η·A/π_old）；减均值后扩散为 `g_b − mean(g)` 形式，同状态其它
动作的 logit 受到反向耦合（softmax 不变但参数动力学不同）。保留该选择时须在
docstring 显式引用 A3。

**B7（Q7：熵/价值系数挂载）**：
- 熵：仓库 `- entropy_coef * entropy`（:209），entropy=−Σπlogπ（:203–205）
  → 损失项 = +entropy_coef·Σπlogπ，与 Eq.29 的 `β·Σπ logπ`（β=1e-2，p28
  Table 8）**形式一致** ✓，默认值 0.01（update_rule.py:37）也吻合 ✓。
- 价值：仓库 `value_coef * mean((V−G)²)`（:202,:208），论文 `α/2·(V−G)²`，
  α=2.0（p27 Table 7）→ 论文等效逐样本系数 1.0；仓库默认 0.5 → 等效 α=1.0，
  差 2 倍，且 `experiment.py:83` 不传 value_coef → YAML 无法配（F5）。
- 结构与论文"value 与 policy 损失合并、所有参数同时更新"（p27 末段，
  fulltext 1968–70）一致 ✓（单一 loss.backward + optimizer.step，:211–214）。

**B8（Q8：优化器）**：仓库 Adam(lr=config.learning_rate)（nn_updates.py:55）；
论文 OpenSpiel 实验 **SGD 恒定 lr=1e-3**（p27，fulltext 1956 行"We used
stochastic gradient descent with a constant learning rate"；Table 7 lr=1e-3）。
Adam 的自适应二阶矩与 NeuRD/ACH 的 logit 空间动力学叠加后不是同一算法（F3）。
`ExperimentConfig.learning_rate` 默认 1e-4（experiment.py:58）也不是 1e-3。

**B9（Q9：已知硬编码核实）**：
- `eta=1.0` 硬编码 nn_updates.py:137（值与论文一致，但位置违反 §9）。
- `l_th=1.0` 硬编码 nn_updates.py:134，**论文 l_th=2.0**（p28 Table 8，
  fulltext 1980 行）→ 值错 + 位置错（F4）。:130–133 的注释说"sweep found
  l_th=2.0 allows too much saturation"——这是本仓库自有的 sweep 结论，不是
  论文结论，改写时应以论文 2.0 为默认。
- `beta`（:120,:129）赋值后从未使用——熵项用的是 `config.entropy_coef`（:209）。
  死参数（F7）。
- `n_epochs`（:118,:127）存了不用——循环体写死 `for _ in range(1)`（:173）。
  死参数（F7）。
- docstring 矛盾：模块头 :11–14 说 ACH 是"REINFORCE policy gradient with the
  learned critic as baseline + entropy bonus, no clipping"；`NNACHUpdate`
  docstring :273–274 说"just REINFORCE + entropy regularizer"；而实际实现
  :182 注释自述是"NeuRD direct-logit loss"。三者描述的是三个不同算法（F9）。

**B10（Q10：theta 统一接口去留）**：
技术上 `policy_loss = (1−θ)·ppo + θ·ach`（:197）的插值在改论文门控后**仍
可写**（两个 loss 项插值永远合法），但统一设计的真正约束不在 loss 而在
**共享脚手架**：当前 PPO/ACH 被迫共享 Adam、优势归一化、单 epoch、target_kl
（:98–105 docstring 自述这是设计目标）。论文忠实 ACH 要求 SGD+无归一化+单
更新，而 37-details PPO 要求 Adam+归一化+多 epoch——**两者在脚手架层面不可
调和**，继续共享意味着至少一端失真。
建议（满足"单一 ACH 实现"约束）：**拆分端点、保留基类**——
`_NNUpdateBase` 只留 batch→tensor、masked logp、熵/价值损失组装、grad clip、
stats 等无算法语义的部分；`NNPPOUpdate` 与论文版 `NNACHUpdate` 各自拥有自己的
step()、优化器与归一化策略；删除 theta 插值（或保留为独立实验类，默认不
暴露）。PPO-vs-ACH 的可比性改在**实验层**保证（同游戏、同 rollout、同
样本量、同评估），而非约束 loss 内部。AGENTS.md D4 的"no PPO clipping"措辞
与论文矛盾（论文 ACH 本就含 PPO 式 ratio clip 门控，只是单线程 vacuous，
p24/p28）——按用户决定，改写时修订 D4。

### C. PPO 最佳实践（对照 Huang et al. 2022, "The 37 Implementation Details"）

| # | 细节 | 现状 | 判定 |
|---|---|---|---|
| 1 | 正交初始化+增益 | 无，PyTorch 默认（mlp.py:66–74） | 无意偏离（F15） |
| 2 | Adam eps=1e-5 | 默认 1e-8（nn_updates.py:55） | 无意偏离（F16） |
| 3 | LR 退火 | 无 | 偏离（ACH 论文本身恒定 lr，PPO 侧若对齐 37-details 需加；可标注有意） |
| 4 | GAE(λ) | 有，每玩家 GAE，γ=1 硬编码、λ=0.95（rollout.py:147–178） | 符合（γ 不可配见 F12/F13） |
| 5 | minibatch shuffle + 多 epoch | 无；单 epoch 全 batch 硬编码（nn_updates.py:173），无 shuffle | **有意偏离**（:160–164 docstring 声明为 apples-to-apples；恰好也与 ACH 论文 single-mini-batch 一致） |
| 6 | 优势归一化 | 有，逐批（:156–158） | 符合（对 PPO） |
| 7 | value loss 无 clip | 无 clip（:202） | 符合（37-details 默认不 clip） |
| 8 | clip_eps=0.2 | 0.2（:117, :180） | 符合 |
| 9 | grad norm 0.5 | 0.5（:213, update_rule.py:38） | 符合 |
| 10 | KL 早停 target_kl | 0.04 存在但单 epoch 下永不触发（:225） | 形存实亡（F8） |
| 11 | 动作 mask | −1e9 mask（mlp.py:28；nn_updates.py:63–65） | 符合（论文未提 mask，spec A5 声明按合法集 softmax） |
| 12 | entropy/value 系数 | 0.01 / 0.5（update_rule.py:36–37） | 符合常规 |
| 13 | 独立 V 与 π 网络 vs 共享 | 共享 torso 双头（mlp.py:72–74） | 与论文结构一致（p25 共享躯干，fulltext 1747–48"share a large portion of parameters"）✓；37-details 无强制 |

### D. 治理与工程质量

**D12（Q12）**：
- §9 违反：`eta`/`l_th` 是代码内魔法数字（nn_updates.py:134,137），不进 YAML
  （F4）；`value_coef`/`max_grad_norm`/`gae_lambda` 也不可从 exp YAML 配
  （experiment.py:83,110）（F5/F13）。
- D4「无 PPO 裁剪」与论文矛盾：论文 ACH 含 ratio clip 门控（p24），只是单线程
  vacuous（p28）。按用户拍板，改写时修订 D4，此处仅记录。
- 分层/import：抽查干净——`nn_updates`→`agents.mlp`/`algos.*`（向下 ✓）；
  `league_controller` 只 import `algos.controller`/`transition`（league_controller.py:23–27），
  无具体 algo import ✓；`eval.nash`→`algos.baselines`（向下 ✓）；未发现
  上指 import。
- 文件行数：全部 < 500（最大 tabular_updates.py 369、experiment.py 319、
  nn_updates.py 311）✓。

**D13（Q13）**：
- `_PolicyAdapter`（nash.py:38–56）：用 `information_state_tensor`（:45，与训练
  obs 一致，loader.py:88–98 优先 info-state ✓）+ 合法动作集上 softmax（:50–55）
  = **当前策略 π=softmax(y)**，正是论文口径（p25/p27 报告当前策略）✓。
  不是 greedy，是完整混合策略，exploitability 语义正确 ✓。
- `evaluate_equilibrium` 两处 `contextlib.suppress(Exception)`（nash.py:128,131）
  + `experiment.py:253`：评估失败时静默缺列，训练曲线无声缺数据。指标计算
  失败不同于运行时 fallback，但当前写法连 warning 都没有，违背
  AGENTS.md「fail loudly」精神（F17）。建议至少 `warnings.warn` 或在行里
  写入 `eval/error` 字段。

**D14（Q14）**：
- 种子链：`run_experiment` 设 `random.seed`+`np.random.seed`（experiment.py:148–149）；
  rollout 用独立 `random.Random(seed)`（rollout.py:73）；MLP 构造时
  `torch.manual_seed(seed)`（mlp.py:62，**全局** torch 种子——同进程多个
  policy 会互相重置 RNG 流，当前单 policy 实验无碍）；动作采样走
  `torch.multinomial`（mlp.py:140,179），吃全局 torch RNG。总体"种子贯穿"
  成立但脆弱。
- GPU 决定论：无 `torch.use_deterministic_algorithms`、无 cudnn 设置 → GPU 上
  不保证逐位复现（F21）。
- 结构不匹配：`MLPSharedActorCritic` 默认 `(128,128)`+Tanh（mlp.py:48–49），
  论文是 `(128,)`+ReLU（p25）。**YAML 层无法配**：`ExperimentConfig`
  （experiment.py:35–62）无 hidden_sizes/activation 字段，`build_policy`
  （:77）写死默认。且 `build_policy` 对 mlp 强制 `gpu_assert.require_cpu()`
  （:74）——注释自称"notebook/smoke default"，但 ExperimentConfig 没有
  任何覆盖路径，等于所有 mlp 实验静默 CPU，与 D6「训练默认 GPU、无静默
  降级」直接冲突（F14，高）。

---

## 4. 「改写到论文忠实 ACH」施工单（精确到函数/行）

按用户决定：**只改 `NNACHUpdate`，不新增第二个 ACH 类**；PPO 端点保留。

- **W1 门控改写**（nn_updates.py `NNPolicyGradientUpdate.step`，:182–195）：
  替换为论文 Algorithm 2（p24）——
  ```
  centered = logits - logits.mean(-1, keepdim=True)
  y_a = centered.gather(动作)          # 门控用减均值
  ratio = exp(new_logp - old_logp)
  gate_pos = (y_a < l_th) & (ratio < 1+eps)    # A>=0 样本
  gate_neg = (y_a > -l_th) & (ratio > 1-eps)   # A<0 样本
  c = where(adv >= 0, gate_pos, gate_neg).float()
  policy_loss = -(eta * y_loss * c * adv / (old_probs + eps0)).mean()
  ```
  `y_loss` 取 `centered`（沿用 A3 裁定）或原始 logit（若改采 Algorithm 2 字面），
  二选一并在 docstring 标注。
- **W2 去归一化**（:156–158）：ACH 端点跳过 `_normalize_advantages`；
  PPO 端点保留。拆分端点 step() 后自然解决（见 B10）。
- **W3 优化器**：ACH 用 `torch.optim.SGD(params, lr=1e-3)`（恒定，无动量——
  论文未提动量，fulltext 1956）；PPO 留 Adam（顺手 eps=1e-5，F16）。
  位置：`_NNUpdateBase.__init__` :55 拆到各端点。
- **W4 超参入 YAML**：`l_th`（默认 2.0）、`eta`（默认 1.0）、`ratio_eps`
  （默认如 0.5，单线程 vacuous）加入 `AlgoConfig`/exp YAML；删除
  nn_updates.py:134,137 硬编码。AGENTS.md §9。
- **W5 value_coef**：exp YAML 暴露；论文等效 α=2.0 ⇔ 现公式下
  `value_coef=1.0`（或改公式为 `α/2·MSE` 并设 α=2.0）。
- **W6 batch=64**：`episodes_per_round` 语义改为"收集至 ≥64 样本即停"
  （rollout.py `run_episode` 支持按样本数截断）或设 episodes_per_round
  使期望样本 ≈64（Kuhn ≈28 局、Leduc ≈17 局、Liar's ≈19 局）；推荐前者。
  保持每批 1 次更新（已是 :173 行为 ✓）。
- **W7 死参数清理**：删 `beta`（:120,129）、`n_epochs`（:118,127）、
  ACH 侧 `target_kl`（:119,225）；同步构造签名 :111–121、:243–267、:279–300。
- **W8 文档**：模块 docstring（:1–15）与 `NNACHUpdate` docstring（:270–277）
  重写为 Eq.29 描述；删除"REINFORCE"表述；标注 A3/A5 假设；同步
  `update_rule.py` 模块头 :12–15 的陈旧描述。
- **W9 步数口径**：`run_experiment`（experiment.py:166）增加 env-step 计数
  （累计 batch.size），`eval_every_env_steps`（默认 1e5）、`total_env_steps`
  （默认 1e7）与 round 口径并存；dump 进 config.json。
- **W10 死配置**：删除或接线 `RolloutConfig.discount`（rollout.py:33）、
  `pool_both_players`（:38）、`AlgoConfig.discount/gae_lambda`
  （update_rule.py:39–40）；`experiment.py:110` 显式传 gae_lambda=0.95。
- **W11 结构可配**：`ExperimentConfig` 加 `hidden_sizes: list[int]=[128]`、
  `activation: str="relu"`、`device: str|null`；`build_policy`（experiment.py:65–78）
  接线并**移除强制 require_cpu**（:74），改由 `--cpu`/配置显式请求（D6）。
- **W12 eval 失败要有痕**：nash.py:128,131 与 experiment.py:253 的 suppress
  内至少 `warnings.warn` + 行内写 `eval/error`。
- **W13 复现配置**：新增 `configs/exp/{kuhn,leduc,liars_dice1}_ach_mlp_mirror.yaml`
  （policy_kind=mlp、hidden_sizes=[128]、relu、SGD 1e-3、batch 64、β=1e-2、
  l_th=2.0、η=1.0、value_coef=1.0、8 seeds 批量入口）。三游戏 YAML
  （kuhn.yaml:6、leduc.yaml:5、liars_dice1.yaml:6）已验证为 OpenSpiel 默认
  参数 ✓（uv run 实测：liars_dice 默认即 numdice=1/dice_sides=6；
  info-state 11/30/21 与 notes 一致）。
- **W14 测试**：重写 `test_algos_nn_updates.py`——
  :105–117 的"常数优势→loss 0"依赖归一化，改写后 ACH 不再成立（常数非零
  优势在论文 ACH 下 loss≠0）；新增：单侧门控单测（构造 y_a 越界样本验证
  梯度方向性）、ratio 门控单测、SGD 优化器类型断言、l_th=2.0 默认断言、
  与 tabular CFR+ 在 Kuhn 上的 exploitability 收敛对照（慢测试）。
- **W15 AGENTS.md**：修订 D4 表述为"ACH 按论文 Algorithm 2 实现（含单侧
  logit 门控与 vacuous ratio 门控；无 PPO clipped-surrogate 作为策略损失）"。

## 5. 无法判断 / 需实验裁决

- **U1 减均值进损失（A3）**：论文正文与 Algorithm 2 表述不一致，两种读法的
  动力学差异（单动作 vs 全动作梯度扩散）只能由 Kuhn/Leduc 对照实验裁决。
- **U2 对称门控的实际伤害程度**：理论上会延缓/阻断越界样本恢复，但在
  l_th=1.0 + entropy 1e-2 下是否可观测地劣于论文门控，需 A/B 实验。
- **U3 "training steps"口径（spec A2）**：环境步 vs 更新次数，需先按环境步
  复现、对比 Fig 10 曲线横轴再裁决。
- **U4 Adam vs SGD 在小博弈上的敏感度**：论文用 SGD，但小博弈上 Adam 是否
  反而更快（仓库注释暗示本地 sweep 影响过 l_th 选择）未知；复现应先用论文
  SGD 出基线，再允许偏差实验。
- **U5 γ/λ（spec A1）**：论文 H.3 未给；现 γ=1.0/λ=0.95 是合理默认，若曲线
  不吻合再做 λ∈{0.95,1.0} 敏感性。
- **U6 仓库注释声称的本地 sweep 结论**（nn_updates.py:130–133 "sweep found
  l_th=2.0 allows too much saturation"）：仓库内未找到该 sweep 的产物/配置，
  无法验证；改写后以论文 l_th=2.0 为准重跑。

---

## 6. 附录（2026-07-23）：B10 的修订 —— theta 统一接口回归

> 本节**修订 B10 的结论**。原判定为「删除 theta 插值」，理由是 PPO 与 ACH 在
> **脚手架层**不可调和（论文 ACH 要 SGD + 不归一化 + 单更新；37-details PPO 要
> Adam + 归一化 + 多 epoch），继续共享意味着至少一端失真。原文同时留了口子：
> 「或保留为独立实验类，默认不暴露」。

### 6.1 为什么改判

B10 的分析没错，但它假设了「共享脚手架 ⇒ 有一端被迫失真」。实际上失真只发生在
**脚手架的取值被写死**的时候。把脚手架旋钮化之后：

- 每个 PPO 最佳实践（Adam、优势归一化、多 epoch、梯度裁剪）成为 `AlgoConfig`
  上的独立字段，可单独开关、单独 A/B；
- 默认值一律取 **ACH 协议侧**，且**在所有 theta 上生效**——于是 `theta=0` 的臂
  与 `theta=1` 的臂之间**只差策略项**，PPO-vs-ACH 第一次成为真正的单因子对比；
- 需要 37-details 版 PPO 时显式打开旋钮，这本身是另一个实验（对脚手架做 A/B，
  而不是对 theta 做 A/B）；
- theta>0 时打开论文不用的旋钮会发出 `ACHFidelityWarning`，避免「看起来像复现、
  其实不是」的静默偏差（§11 不得静默降级）。

### 6.2 论文忠实度如何被保护

D4 的实质约束是「ACH 的策略损失里没有 PPO 截断代理」。统一规则在 `theta=1` 时
**根本不构造 PPO 项**（短路，不是乘 0），`theta=0` 时同样不构造 ACH 项，因此两个
端点在数值上与专用实现完全一致。

这一点不是靠论证，而是靠**证据**：`tests/unit/data/nn_updates_golden.json` 由
**合并前**的双端点代码（commit `32a25e3` 的 `NNACHUpdate` / `NNPPOUpdate`）生成，
覆盖 6 个场景（shipped ACH、pre-LayerNorm 居中、legal-only 均值、带梯度裁剪、
legacy PPO+Adam、theta=0 on ACH 脚手架），记录每步 stats 与**更新后的全部参数张量**。
合并后 `tools/gen_nn_golden.py --check` 必须复现它：

- 6/6 场景的**更新后参数逐比特相同**；
- 所有既有 stats 逐比特相同；
- 唯一差异是两个 PPO 场景**新增** `grad_norm` 遥测键（旧 PPO 端点从未计算过），
  属纯增量可观测性，不改数值。

该 fixture **不再重新生成**（生成器在文件已存在时拒绝覆写）。它一旦需要变更，
就等于宣告更新规则的数值变了，必须是一次有意的、被审阅的动作。

### 6.3 遗留的真实代价（不掩盖）

- **默认 PPO 变了**：`algo: ppo` 现在默认 SGD + 原始优势 + 单 epoch，而不是
  Adam + 归一化。这是「PPO 侧默认跟随 ACH」的直接后果，好处是可比性，代价是
  `algo: ppo` 不再等于文献里的 reference PPO。恢复方式写在
  `configs/exp/<game>_ppo_mlp_mirror.yaml` 的头部注释里。
- **梯度尺度随 theta 变化**：ACH 项带无界的 `1/pi_old` 且用未归一化优势，PPO 项是
  O(1)。凸组合的梯度范数因此随 theta 强烈变化，**有效步长不是常数**。仓库不做任何
  "配平"（那会改掉两个算法），而是在每次更新记录 `train/grad_norm`，并在
  `notebooks/theta_*.ipynb` 的诊断面板里画出来——theta–exploitability 曲线的解读
  必须带上这一条。
- **tabular 路径不参与**：`policy_kind: tabular` 的 PPO/ACH 仍是两套离散实现
  （D5，ACH 侧是 CFR+ 包装），`algo: theta` 在 tabular 下直接报错。
