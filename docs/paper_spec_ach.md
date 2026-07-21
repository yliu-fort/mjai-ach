# ACH 论文规格书（ground truth）

> 本文档是审计与复现的**事实基准**：只记录论文里写了什么，每条标注页码。
> 不做仓库代码评价（见 `audit_report.md`）。
>
> 论文：Fu et al., *Actor-Critic Policy Optimization in a Large-Scale
> Imperfect-Information Game*, ICLR 2022, OpenReview `DTXZqTNV5nW`。
> 本地存档：`docs/ach_paper.pdf`（经 Internet Archive 快照取得；
> OpenReview 原链有反爬验证），全文文本 `docs/ach_paper_fulltext.txt`（28 页）。
> 下文页码 = PDF 页码（1-based）。

---

## 1. 算法：NW-CFR（理论版）与 ACH（实用版）

### 1.1 NW-CFR — Algorithm 1（p5）

- policy net `y(a|s;θ)` 逼近累积加权优势 `Ra_t(s,a) = Σ_k f^{μ_k}_p(s) r^c_k(s,a)`；
  每轮策略 `π_t = softmax(η(s)·y)`。
- Q net 用采样回报 `G` 的平方误差训练；优势估计
  `A^{π_t}(s,a) = Q(s,a;ω_t) − Σ_b π_t(b|s)Q(s,b;ω_t)`。
- policy 损失（Eq. 2）：对**采样到的状态** s，向目标
  `y(a|s;θ_{t−1}) + (1/M)Σ_i 1{s∈τ_i} A^{π_t}(s,a)` 做 MSE 回归；
  未采样状态目标增量为 0。
- 定理 1：NW-CFR ≈ 加权 CFR + Hedge，平均策略 exploitability 上界
  `ε ≤ |S|Δ√(ln|A|/(2T)) + ΔΣ_s (w_h−w_l)/w_h`（p5, Eq. 3）。

### 1.2 ACH — Algorithm 2 + Eq. 29（p24，复现的真正对象）

实用改动（相对 NW-CFR）：当前策略加熵正则、只用采样状态、不算平均策略、
IMPALA 式 actor/learner 解耦、mini-batch 训练、PPO 式 importance-ratio
clipping 处理异步；行为策略 `μ_{p,t} = π_{p,t}`；双方共用同一组 θ, ω；
`η(s)` 折进学习目标里，所以 `π(a|s) = softmax(y(a|s;θ))` 直接得到（p24）。

**逐样本损失**（mini-batch 内对每个样本 `[a, s, A(s,a), G, π_old(a|s)]`）：

```
若 A(s,a) ≥ 0:  c = 1{ π(a|s;θ)/π_old(a|s) < 1+ε } · 1{ y(a|s;θ) − ȳ(·|s;θ) <  l_th }
若 A(s,a) < 0:  c = 1{ π(a|s;θ)/π_old(a|s) > 1−ε } · 1{ y(a|s;θ) − ȳ(·|s;θ) > −l_th }

L_ACH = −c · η(s) · y(a|s;θ)/π_old(a|s) · A(s,a)
        + α/2 · [V(s;ω) − G]²
        + β · Σ_a π(a|s;θ) log π(a|s;θ)          （= −β·H，即熵奖励）
```
（p24 Algorithm 2 与 Eq. 29）

要点：
- **门控 c 是优势符号依赖的、单侧的**：A≥0 时只在"未触上界/未超 ratio 上限"
  时保留样本；A<0 时对称地只看下界。被门掉的样本梯度为 0，但**反方向回拉
  永远允许**（与 PPO clipped surrogate 的门控逻辑同构）。
- policy 梯度项里的 y 是 policy net 输出（logit）；均值 ȳ 只在门控中出现。
  原文 "the mean ȳ(·|s;θ) is subtracted from the policy output, which is then
  clipped within a range [−l_th, l_th]"（p24）。→ 见「歧义 A3」。
- **每次迭代只对单个 mini-batch 更新一次**（"we update θ and ω once using a
  single mini-batch at each iteration"，p24，出现两次）。
- value 与 policy 损失合并、所有参数**同时**更新（p27 末段；与 OpenSpiel 自带
  A2C/RPG/NeuRD 的"critic 单独更新 32 次"不同）。
- 优势 `A(s,a)` 用 **GAE(λ)** 估计，仅对采样到的状态-动作（p24）。
- 价值目标 `G` 为采样回报（p24）。

---

## 2. 复现范围：附录 G 的 OpenSpiel 小博弈实验（p25–26）

| 项 | 论文设定 | 出处 |
|---|---|---|
| 游戏 | Kuhn poker、Leduc poker、Liar's Dice（OpenSpiel 默认参数） | p25 |
| 对比方法 | A2C / RPG / NeuRD / ACH（**本次复现只跑 ACH**，见用户决定 3） | p25 |
| 网络结构 | OpenSpiel 标准：**1 层 128 单元 FC + ReLU，再接两个独立线性头**（policy、value） | p25 |
| 训练环境 | 单线程、2.24GHz CPU（actor 与 learner 串行） | p25, p28 |
| 评估 | OpenSpiel **精确** exploitability，**每 1e5 training steps 一次** | p25 |
| 训练总长 | x 轴到 **1e7 training steps**（Fig 10） | p26 |
| 重复 | **8 次独立运行**，曲线报 mean，阴影报 range（min–max） | p26 |
| 报告对象 | **当前策略**（非平均策略） | p25, p27 |
| y 轴范围 | Kuhn 0–0.5；Leduc 0–2.5；Liar's Dice 0–1.2（Fig 10 坐标） | p26 |

**p28 关键注释**：单线程下 actor 与 learner 串行执行，因此 Algorithm 2 中
`π(a|s;θ)` 恒等于 `π_old(a|s)` —— **ratio 恒为 1，ratio-clipping 门控在该实验中
恒真（vacuous）**，唯一生效的门控是单侧 logit 阈值。H.2 对同步 FHP 实验同样
注明不需要 ratio clip（p27）。

### Fig 10 曲线形态（目视，精确数值在对比阶段数字化）

- Kuhn：ACH 约 2e6 steps 内降到 ~0.05–0.1，随后平稳；RPG ~0.1–0.15；
  NeuRD ~0.2–0.25；A2C ~0.3+ 几乎不降。
- Leduc：ACH 降到 ~0.2–0.4；RPG ~0.4–0.5；NeuRD ~0.5–0.7；A2C ~1.5–2。
- Liar's Dice：ACH 降到 ~0.1–0.25；其余 0.6–0.9。论文称 ACH 优势在该游戏最显著。
- Table 4（head-to-head，1e7 steps 处取 agent，10,000 场对局）**不在复现范围**。

---

## 3. 超参数（附录 H.3，p27–28）

**ACH（OpenSpiel 实验最终值）**：

| 参数 | 值 | 搜索范围 | 出处 |
|---|---|---|---|
| 优化器 | **SGD，恒定学习率**（非 Adam） | — | p27 |
| Learning rate | **1e-3** | {1e-3, 5e-3} | p27 Table 7 |
| Value loss coef α | **2.0** | {1.0, 2.0} | p27 Table 7 |
| Hedge 系数 η(s) | **1.0** | {1.0, 1e-1, 1e-2} | p27 Table 7 |
| Batch size | **64**（样本数） | — | p28 Table 8 |
| 熵系数 β | **1e-2** | — | p28 Table 8 |
| Logit 阈值 l_th | **2.0** | — | p28 Table 8 |
| Ratio clip ε | 任意（单线程下 vacuous，见 §2） | — | p28 |

对照：Mahjong 实验用 Adam(lr 2.5e-4)、ε=0.5、l_th=6.0、batch 8192、
GAE λ=0.95、γ=0.995、α=0.5、β=1e-2、η=1.0（p26 Table 5）；
FHP 用 lr 1e-4、α=2.0、β=3e-2、l_th=2.0、λ=0.95、γ=0.995、reward ×0.002（p27 Table 6）。

---

## 4. 论文未写明、需假设的项（复现时显式声明）

- **A1（γ/λ）**：H.3 未给折扣因子 γ 与 GAE λ。Mahjong/FHP 均为 λ=0.95、
  γ=0.995，据此沿用；三个小博弈奖励只在终局、序列极短，γ∈{0.995,1.0}
  差异可忽略。若结果不吻合，做 λ∈{0.95,1.0} 敏感性检查。
- **A2（"training steps"定义）**：按**环境步数**（frames）理解；每 1e5 步评估
  一次、总长 1e7。若复现曲线横轴错位一个数量级，再改按更新次数理解。
- **A3（损失中 y 是否减均值）**：Algorithm 2 的损失写原始 `y(a|s;θ)`，正文说
  "均值从 policy output 中减去"。减均值后 softmax 不变，但梯度会从单动作
  扩散到全动作（`g_a − mean_b g_b`）。实现取**减均值**（与正文表述一致），
  并在审计报告中标注此歧义。
- **A4（OpenSpiel 版本）**：论文用 ~2021 年 OpenSpiel；三个游戏默认参数
  多年未变，风险低。复现固定用仓库 `uv.lock` 里的版本，游戏串：
  `kuhn_poker`、`leduc_poker`、`liars_dice`（1 骰，默认）。
- **A5（合法性 mask）**：论文未提动作 mask；OpenSpiel 实现均按合法动作
  集合 softmax。复现同样 mask。

## 5. 已核实的代码注释引用（p24）

仓库 `nn_updates.py` 注释引用的三处均为论文原文，属实：
- "Algorithm 2" 存在（ACH 伪代码，p24）✓
- "we only update once using a single mini-batch at each iteration" ✓（p24 原文）
- logit 减均值 + `[−l_th, l_th]` 阈值 ✓（p24 Algorithm 2 门控与正文）

但论文门控是**单侧、符号依赖**的（§1.2），与仓库当前的对称门控不同——
差异细节见 `audit_report.md`。
