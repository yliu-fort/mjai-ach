# BRPS 上 ACH-MLP 为什么"直接不收敛"：logit 步长被网络放大 131 倍，第一次更新就出局

> **一句话结论**：这不是"收敛慢"、也不是 liar's dice 那种"收敛到门控允许的最好策略"。BRPS 上
> ACH-MLP 的**第一次更新**就把被采样动作的 logit 推动 **10.06**（`l_th=2` 盒子的 5 倍），
> 策略在 ~1000 env-steps 内退化为纯策略；而在 mirror 自对弈的对称零和博弈里，纯策略两座位的
> 收益恒为 0 ⇒ `A ≡ 0` ⇒ 单边门控落进 `A ≥ 0` 分支、要求 `y < l_th`（假）⇒ **梯度恒等于 0**。
> 熵梯度在顶点同样为 0。于是纯策略顶点是**精确的吸收态**（实测 `|grad| = 0.00e+00` 持续 2×10⁴
> 次更新），NashConv 停在 50 或 100 —— **理论最大值**。
>
> 放大倍数有闭式：`dy = lr · η · A / π_old · ‖∂y/∂θ‖²`，而 `trunk_layernorm=True`
> （`docs/reproduce_report.md` §6.5 的偏离）把 `‖f‖²` **钉死**为 `hidden_size`，所以
> `‖∂y/∂θ‖² ≈ hidden_size + 3 = 131`。BRPS 的 ±50 支付把这个放大变成致命值：
> `1e-3 × 131 × 25 × 3 ≈ 10`。
>
> | 层级 | 命题 | 证据 |
> |---|---|---|
> | **实测** | 放大倍数 = 131.1×（`‖f‖²` = 128.10 = hidden_size），单步 `dy` 10.06 vs 表格式预测 0.077 | §2（`tools/brps_logit_step.py`） |
> | **实测** | 剂量-反应单调：放大 1.0 → 1.54 → 19.3 → 131 时 NashConv 3.03 → 4.79 → 6.47 → **66.7** | §6（RL 3e5×3 seeds） |
> | **实测** | 纯策略顶点梯度**恰好为 0**（不是"很小"）；门控与熵项在顶点同时失效 | §3（`tools/brps_noise.py`） |
> | **实测** | 逃逸只能靠稀有样本的 `1/π_old`（实测 iw_max 3143、pterm_max 1.85e4）⇒ 顶点间跳跃，4/30 run 直接 **NaN 崩溃** | §4 |
> | **结构** | 即使修掉放大，paper 超参下**精确无噪声算子也不收敛**：旋转 6.66e-3/步 vs 收缩 1.9e-6/步（比 2.9e-4），振幅锁在盒壁 `spread → 2·l_th = 4` | §5（`tools/brps_operator.py`） |
> | **界限** | β 和 `l_th` 在 BRPS 上**都救不了 RL**（NashConv 66.7 / 53.3），但在精确算子里 β=0.1 → 0.023、`l_th=log10/2` → 0.002 —— 两者治的是 §5 的病，不是 §2 的病 | §5–§6 |
> | **范围** | 8 个游戏里 BRPS 的首步位移 19.65 是第二名（leduc 5.11）的 4 倍；这是 D8 里唯一支付量级 ±50 的博弈 | §7 |
>
> 度量口径：exploitability = NashConv/2（memory `exploitability-vs-nashconv-units`）；BRPS 是
> 同时行动博弈，只能用 `eval/nash_conv`（`src/mjai/eval/nash.py:289`）。NashConv = 2·max_a A_a，
> 纯 Rock 给 50、纯 Scissors 给 100。

---

## 0. 被测对象，以及两个必须先说清楚的前提

**BRPS 是三个 logit 的问题。** `matrix_brps` 的 `information_state_tensor` 是 `[0.]` —— 一个恒为
零的特征，horizon 1，`Dynamics.SIMULTANEOUS`。所以 MLP 的输入是常量，网络退化成"偏置 + torso
输出的一个固定方向"上的 3 个 logit：**函数逼近能力不可能是瓶颈**，"MLP 不行"必须由别的东西解释。
本文给出的解释是：MLP 不改变能表达什么，它改变**一次更新走多远**。

**tabular 臂不能当对照。** `configs/exp/brps_ach_mirror.yaml` 的 `algo: ach` 走的是
`TabularACHUpdate`，而它是一个 **CFR+ wrapper**（`src/mjai/algos/tabular_updates.py:190`，
docstring 自述"每轮全树遍历"）—— 不是 ACH。所以"tabular 收敛、MLP 不收敛"**不是**函数逼近的
证据；MLP 路径是本仓库唯一真正的 ACH（AGENTS.md D4/D11）。本文合法的对照是同脚手架的
`theta=0`（PPO 端点）与"关掉放大"的诸臂。

顺带一个已经写在仓库里的旁证：tabular 路径**显式**做了 per-batch advantage 归一化并把 logit
clamp 到 ±10，注释直接点名 BRPS 的 ±50 会 saturate 并 "collapse to a deterministic one"
（`tabular_updates.py:96`、`:37`）。NN 路径按论文忠实取 `normalize_advantages=False`、**无
clamp**，只有 `l_th` 门控 —— 而门控在更新**之前**求值，拦不住一次 10 单位的位移（§2）。

**同时行动路径本身没有 bug**（花任何算力之前先排掉的）：`RolloutWorkerCore` 在 `apply_actions`
之前记录两个座位、`returns[p]` 逐座位归属、horizon-1 下 GAE 退化成 `A = r − V`
（`src/mjai/pipeline/rollout.py:391`），mirror 下两座位同 producer 一起进 batch。没有符号或归属
错误。

---

## 1. 现象：不是"没收敛到 Nash"，是**在前 1000 步内死掉**

`configs/exp/brps_ach_mlp_mirror.yaml`，3e5 env-steps，3 seeds（`runs/brps_probe/paper/`）：

| seed | 末点 π | NashConv（全程） | entropy | grad_norm | 结局 |
|---|---|---|---|---|---|
| 0 | (2e-6, 0, 0.999998) | **100.0**（13/13 个评估点全是 100） | 0.000 | 5e-6 | 6.5e4 步 **NaN 崩溃** |
| 1 | (1.0, 0, 0) | **50.0**（60/60 个点） | 0.000 | 2.2e-7 | 冻死 |
| 2 | (1.0, 0, 0) | 50.0（85% 的点；其余 99.9） | 0.000 | **0.0** | 冻死 |

加密采样（eval 每 100 env-steps）给出时间线，`runs/brps_probe_dense/`：

| env-steps | P_R | P_P | P_S | NashConv | gate_off | grad_norm | pterm_max | iw_max |
|---|---|---|---|---|---|---|---|---|
| 128（**第 2 次更新**） | 0.020 | **0.979** | 0.001 | 7.79 | 0.00 | 281 | 57.1 | 2.66 |
| 256 | 0.0001 | 0.999 | 0.001 | 9.98 | 0.97 | 443 | 321 | 49.9 |
| 320–832 | — | 0.999 | — | 9.98 | **1.00** | 3.3→0.29 | **0** | 1.00 |
| 960 | 0.000 | 0.015 | **0.985** | **97.8** | 0.97 | 978 | **1.85e4** | **1089** |
| 1536+ | 0.000 | 0.0003 | 0.9997 | **99.95** | 0.00 | →1e-3 | →4e-3 | 1.00 |

读法：**两次更新之内**策略就从均匀跳到 0.979 集中；随后 gate_off=1.00、pterm_max=0 —— 完全冻结；
960 步处一个稀有样本（iw 1089）产生 1.85e4 的项，把策略一脚踢到另一个顶点；再冻结。全程 NashConv
∈ {50, 100}，即**最坏可能值**。

---

## 2. 机制一：logit 空间的有效步长被放大 131 倍（`tools/brps_logit_step.py`）

精确算子（`tools/brps_operator.py`）与采样算子（`tools/brps_noise.py`）都在动**一张 logit 表**，
步长是教科书式的 `dy = lr · η · A / π_old`。管线动的是**参数**，logit 是
`y = W_π f(θ) + b_π`，链式法则把同一个损失梯度映回 logit 空间时多出一个因子 —— 该 logit 自身
参数梯度的平方范数（网络在该输入处的 NTK 对角）：

```
dy_a = lr · η · A / π_old(a) · ‖∂y_a/∂θ‖²
```

`trunk_layernorm=True` 把 torso 输出归一化到零均值单位方差，于是 **`‖f‖² = hidden_size` 恒成立**
（与权重无关），头部贡献就是 `hidden_size + 1`。实测（advantage=25 = Rock vs Paper，lr=1e-3，
取最稀有动作，熵与 critic 关闭以只测策略项的传递函数）：

| arm | π_old | ‖f‖² | NTK | dy（表格式预测） | dy（MLP 实测） | 放大 | 更新后 y |
|---|---|---|---|---|---|---|---|
| h=1, LN on | 0.218 | 0.01 | 1.54 | 0.115 | 0.177 | 1.54× | 0.16 |
| h=8, LN on | 0.236 | 7.98 | 17.4 | 0.106 | 1.31 | 12.4× | 1.19 |
| h=32, LN on | 0.240 | 31.9 | 36.0 | 0.104 | 3.72 | 35.7× | 3.36 |
| **h=128, LN on（配置臂）** | 0.326 | **128.10** | 131.2 | **0.077** | **10.06** | **131.1×** | **9.60** |
| h=128, LN **off** | 0.309 | 18.1 | 19.3 | 0.081 | 1.56 | 19.3× | 1.38 |
| h=512, LN on | 0.216 | 511.6 | 515.0 | 0.116 | 59.6 | 514.7× | 59.5 |

**配置臂的有效 logit 学习率是 0.131，不是 1e-3**；要还原表格式步长需要 `lr = 7.63e-6`。
一次更新把 logit 送到 9.60 —— `l_th=2` 盒壁的 **4.8 倍**。门控在更新**之前**求值，所以盒子
在结构上无法拦住这一步；它只能在下一步宣布"已经越界"。

`‖f‖²` 一列是承重的：LayerNorm 不是无关的实现细节，它**取消了唯一可能让特征范数保持小值的机制**，
把放大倍数固定成网络宽度。这个偏离当初是为了复现 liar's dice 曲线而采纳的（§6.5），在那里
`|A| ≤ 1`，同样的 131× 不致命。

---

## 3. 机制二：纯策略顶点是**精确的**吸收态

对称零和 + mirror 自对弈 ⇒ 两座位同策略 ⇒ `V = π^T M π = 0` 恒成立，`A_a = (Mπ)_a`。策略变成
确定性 `π = e_k` 时：

- 两座位都打 `k`，收益 `M[k,k] = 0`，于是**所有样本的 `A` 恰好是 0**；
- `A = 0` 落进 `nn_losses.ach_policy_loss` 的 `advantages >= 0` 分支，门控条件是 `y_a < l_th`
  —— 越界的 logit 使它为假 ⇒ `c = 0`；
- 熵梯度 `β π_j (log π_j + H)` 在顶点上每一项都 → 0（`π=1` 时 `log π + H = 0`，`π→0` 时前因子
  → 0）。

**没有任何一项提供恢复力。** 采样算子实测（`tools/brps_noise.py`，从 `y=(0,20,0)` 出发，2×10⁴
次更新）：

```
saturated y=(0,20,0)  →  pi=[0.0, 1.0, 0.0]  spread=20.000  gate_off=1.00  |grad|=0.00e+00
y=(0,3,0)（盒子内）    →  pi=[0.029, 0.481, 0.490]  spread=2.829  gate_off=0.00  |grad|=3.41
```

梯度是**精确的 0**，不是"很小"。这正是 `tabular_updates.py:190` 的 docstring 早已写下的那句
"once mirror self-play goes deterministic, payoffs vanish, advantages hit zero, and regrets
freeze" —— 那段话是为 tabular 的旧实现写的，本文测到它在 NN 路径上同样成立，而 NN 路径没有
tabular 后来加的 clamp 与归一化。

## 4. 机制三：唯一的逃逸通道是无界的 `1/π_old`，它也是 NaN 的来源

`iw_clip=None` 是论文忠实设置，没有任何东西约束 `1/π_old`。冻结在 `π_k ≈ 1−1e-3` 时，每隔 ~15
次更新会采到一个稀有动作，其 iw ~ 10³（实测 iw_max **3143**、pterm_max **1.85e4**），一步把策略
踢到另一个顶点 —— 于是"顶点间跳跃"，NashConv 始终 ≈ 50/100。更极端时 float32 溢出：

```
RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
  (mlp.py:259, torch.multinomial)
```

30 个 RL run 里 **4 个**这样崩掉（paper/seed0、beta0.1/seed0、lth1.15/seed0 等）。也就是说 BRPS
上这个臂不仅不收敛，还**会带着 NaN 中途死掉**。

---

## 5. 拿掉放大之后：paper 超参下的**精确算子**仍然不收敛

BRPS 的期望 ACH 算子只有三维（`tools/brps_operator.py`；反对称支付 ⇒ `V ≡ 0`，访问权重 ρ ≡ 1，
所以 `docs/ach_operator_theory.md` 的规范漂移与 leak 公式在这里都退化 —— 全混合 NE 时
`Σ_a A_a = 0`、`k = n` ⇒ `leak = 0`。**BRPS 的失败与 liar's dice 无关，是另一套机制**）：

```
y ← y + lr·( η·c·(Mπ)  −  β·π·(log π + H) )
```

从均匀初始化迭代，paper 超参（lr=1e-3, β=1e-2, l_th=2, η=1）：

- 局部特征值（向量场，每步）：旋转 **Im = 6.66e-3**（周期 943 步），收缩 **Re = −1.92e-6**
  （时间常数 5.2e5 步 ≈ 3.3e7 env-steps）。比值 **2.9e-4** —— 每转一圈半径只缩 0.18%。
- β 拥有**全部**收缩：β=0 时 Re = −1.2e-10（数值零，纯旋转 = Poincaré 复现）；Re 随 β 线性
  （1e-2→1.9e-6，0.1→2.06e-5，0.5→1.0e-4，2.5→4.07e-4）。
- 全局行为不是慢速内旋，而是**向外**：按"圈数"对齐后

| 步长臂 | 100 圈 | 200 圈 | 300 圈 | 400 圈 | 结论 |
|---|---|---|---|---|---|
| lr=1e-2 | tv 0.120, spread 4.05 | 0.125 | 0.125 | 0.125, **spread 4.04** | 极限环**贴在盒壁 2·l_th=4** |
| **lr=1e-3（paper）** | 0.066 | 0.073 | 0.083 | **0.096（仍在增大）** | 向外，朝盒壁去 |
| lr=1e-4 | 0.059 | 0.051 | 0.045 | **0.038（在缩）** | 收缩 |
| lr=1e-3, 支付/50 | 0.017 | 0.0128 | 0.0128 | **0.0128, expl 0.12** | 真不动点 |

即：**离散步长 `lr·η·|A|` 的向外漂移与 β 的收缩在 BRPS 的 paper 值上处在不稳定的一侧**，所以
即便噪声、critic、函数逼近全部拿掉，ACH 的最后迭代在 BRPS 上也只到一个极限环，振幅由 `l_th`
盒子锁定。两个能治它的旋钮（在精确算子里，156k 次更新预算内）：

| 臂 | 末点 expl | TV | 说明 |
|---|---|---|---|
| paper | 4.17（振幅 0.5–10） | 0.066 | 极限环 |
| **β=0.1** | **0.023** | 0.0025 | 收缩率 ×10，被捕获；残余是 QRE 熵偏差 |
| β=0.5 / 2.5 | 0.118 / 0.718 | — | 过头：熵偏差反弹 ⇒ β 有内点最优 |
| **l_th=1.1513=log10/2** | **0.0023** | 0.0003 | 恰好装得下 NE 的最紧盒子；夹住轨道又不偏置不动点 |
| l_th=0.5 / 1 | 4.23 / 2.10 | 0.15 / 0.21 | 盒子装不下 NE（需要 spread log10=2.303）⇒ 冻死在盒壁 |
| l_th=8 / 无门控 | 29.7 / 18.2 | 0.42 / 0.26 | 盒子是**天花板**：l_th=2 时盒内最坏 expl 已达 **47.8** |

最后一行是与 liar's dice 的**符号反转**：那里盒子是地板（禁止 Nash 需要的锐化），这里盒子是
天花板，而且 `l_th=2` 的天花板高到允许 π=(0.018, 0.018, 0.965)、expl 47.8 —— 它根本不阻止坍缩。

---

## 6. RL 判别臂（3e5 env-steps × 3 seeds，`runs/brps_probe/`）

`amp` = §2 实测的 logit 放大倍数。`NC_tail` = 各 seed 末 20% 评估点的 NashConv 均值再跨 seed 平均。

| arm | 改动 | amp | crash | **NC_tail** | NC_min | π_max | entropy | 判读 |
|---|---|---|---|---|---|---|---|---|
| paper | — | 131 | 1/3 | **66.7** | 66.7 | 1.0000 | 0.000 | 死在顶点 |
| beta0.1 | `entropy_coef=0.1` | 131 | 1/3 | **66.7** | 66.7 | 1.0000 | 0.000 | **无效**（治的是 §5 不是 §2）✓预测 |
| normadv | `normalize_advantages` | 131 | 1/3 | **66.7** | 66.7 | 1.0000 | 0.000 | **无效**（见下） |
| lth1.15 | `l_th=log10/2` | 131 | 1/3 | 53.3 | 53.3 | 0.9996 | 0.003 | 几乎无效（盒子拦不住 10 单位的一步） |
| batch1024 | `target_samples=1024` | 131 | 0 | 26.7 | 3.98 | 0.966 | 0.098 | 基本无效 ⇒ **不是方差问题** |
| iwclip10 | `iw_clip=10` | 131 | 0 | 16.9 | 5.49 | 0.846 | 0.452 | 部分缓解（掐掉稀有样本的踢击） |
| ppo | `theta=0`（同脚手架） | 131 | 0 | 12.3 | 6.89 | 0.9986 | 0.011 | **也坍缩**，但 ratio clip 把伤害压在 ~10 |
| no_ln | `trunk_layernorm=False` | 19.3 | 0 | 6.47 | 0.83 | 0.618 | 0.846 | 活 |
| h1 | `hidden_sizes=[1]` | 1.54 | 0 | 4.79 | 0.54 | 0.652 | 0.788 | 活 |
| **lr_matched** | `lr=7.63e-6 = 1e-3/131` | 1.0 | 0 | **3.03** | **0.47** | 0.618 | 0.826 | **回到 §5 精确算子的极限环** |

三点读法：

1. **剂量-反应单调**：`NC_tail` 3.03 → 4.79 → 6.47 → 66.7 严格随 amp 1.0 → 1.54 → 19.3 → 131
   递增。把 lr 除以实测的 131，或把网络从 128 宽缩到 1，效果**相同** —— 这是"放大倍数是机制"
   而不是"学习率碰巧太大"的判据。
2. **β 与 `l_th` 在 RL 上全部失效**，尽管它们在精确算子里分别把 expl 打到 0.023 / 0.002。这正是
   §2 与 §5 是**两个不同病灶**的证据：它们治的是极限环，治不了一步越界。
3. **`normalize_advantages` 也救不了**，尽管它把 `|A|` 压到 ~1。机制上自洽：归一化让步长与
   advantage 的**绝对**大小脱钩，于是策略越接近均衡（advantage 越小）步长相对越大，噪声被放大
   到单位尺度继续推 —— 这解释了为什么 tabular 路径在归一化之外**还**需要 ±10 的 logit clamp
   （`tabular_updates.py:37`）：归一化单独是不够的。
4. **PPO（`theta=0`）同样坍缩**（π_max 0.9986、entropy 0.011），说明 131× 放大是**脚手架**的
   性质、两个 theta 端点共享；而"梯度恰好为 0 的吸收态 + 无界 `1/π_old` + NaN"是 ACH 策略项
   （y 线性损失 × `1/π_old` × 单边门控）特有的，它把 PPO 的 NC≈10 变成 NC=50/100。

---

## 7. 为什么是 BRPS：8 个游戏的首步位移

首步位移 `D₁ = lr·(1+‖f‖²)·|A|_max·n`（均匀初始化 `π=1/n`，`|A|_max` 取博弈的最大效用，因此是
**上界**）：

| 游戏 | min/max utility | \|A\|_max | D₁ | vs `l_th=2` |
|---|---|---|---|---|
| **brps** | −50 / 50 | **50** | **19.65** | **9.8×** |
| leduc | −13 / 13 | 13 | 5.11 | 2.6× |
| liars_dice1 | −1 / 1 | 1 | 1.70 | 0.9× |
| ttt | −1 / 1 | 1 | 1.18 | 0.6× |
| kuhn3 | −2 / 4 | 4 | 1.05 | 0.5× |
| oshi_zumo | −1 / 1 | 1 | 0.79 | 0.4× |
| goofspiel5_ii | −1 / 1 | 1 | 0.66 | 0.3× |
| kuhn | −2 / 2 | 2 | 0.52 | 0.3× |

BRPS 是 D8 里唯一支付量级 ±50 的博弈，D₁ 是第二名的 4 倍、门控盒子的 10 倍。这就是为什么
`docs/reproduce_report.md` 的三个复现游戏（kuhn/leduc/liars）没有暴露这个机制，而 BRPS 一碰就死。
论文本身没有跑 BRPS —— `configs/exp/brps_ach_mlp_*.yaml` 是本仓库把论文协议（p25–28 的 lr=1e-3、
β=1e-2、l_th=2、batch 64、1×128+LayerNorm）**照搬**到一个支付量级大 50 倍的博弈上的结果。

---

## 8. 局限

1. **未跑完整 1e7 协议**。预算是 3e5 × 3 seeds；但故障发生在前 10³ 步、且是吸收态，更长预算只会
   让"死着"更久。精确算子这一级用的是等价 156k–2e6 次更新（1e7–1.3e8 env-steps），已覆盖协议。
2. **NaN 崩溃的确切溢出点未定位**。只知道它来自无界 `1/π_old`（iw_max 实测 3143）与 float32
   logit；没有做逐步 tracing。
3. **§7 的 D₁ 用 `|A|_max`**，是上界不是典型值 —— leduc 的 5.11 并未致命（它的 advantage 很少
   取到 ±13）。所以这张表说明"BRPS 突出"，不构成一个 sharp threshold。
4. **`h1` 臂同时改了容量与放大**，两者在 BRPS 上不可分（输入恒为 `[0.]`，容量本来就不是瓶颈，
   §0），但严格说它不是纯粹的单因子臂；`lr_matched` 才是。
5. **没有改任何默认值**。本文只诊断。要不要把 `iw_clip`、logit clamp、或按 `|A|`/`‖f‖²` 归一的
   学习率变成 BRPS 配置的默认，是一次单独的改动（AGENTS.md §11 一次一个关注点）。
6. **一个下游后果未处理**：`notebooks/phase1_one_click.ipynb` 的 BRPS×MLP 单元格与
   `configs/exp/brps_ach_mlp_league.yaml` 报告的是一个死策略的数字。本文不改它们，但任何引用
   BRPS MLP 结果的地方都应先读本文。

## 9. 复现

```bash
uv run python -m tools.brps_operator --iters 156000                      # §5 精确算子 + 盒子闭式
uv run python -m tools.brps_noise --updates 15625 --seeds 0 1 2          # §3 采样算子
uv run python -m tools.brps_logit_step                                   # §2 放大倍数表
uv run python tools/ab_factor_probe.py --config brps_ach_mlp_mirror \
    --label paper --overrides '{}' --seed 0 --root runs/brps_probe \
    --total-env-steps 300000 --eval-every 5000                           # §6 RL 臂
```

判别臂的两个 yaml：`configs/exp/brps_ach_mlp_mirror_lr_matched.yaml`（放大匹配的 lr）、
`configs/exp/brps_ach_mlp_mirror_h1.yaml`（hidden=1）。其余臂由 `--overrides` 给出，解析后的完整
配置写在各 run 目录的 `config.json` 里（AGENTS.md §9）。
