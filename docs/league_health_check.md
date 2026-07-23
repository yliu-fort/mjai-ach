# League 健康度与参数合理性检查报告

> 检查对象：首次 league 训练（`runs/league_probe/`，{kuhn, brps} × {mirror, league}
> × seeds 0–3，6e4 env-steps/臂）背后的 league 机制实现与参数。
> 方法：**只读源码** + 运行期验证（monkeypatch 插桩重放、受控消融），
> src/ 与 configs/ 零改动；所有数字来自实际读取/运行（命令见附录 A）。
> 产物：`docs/league_health_check.md`（本文件）与 `runs/league_health/`。
> 日期：2026-07-22（同 probe 运行日）。

---

## 0. 四问速答

| # | 问题 | 结论 | 证据强度 |
|---|---|---|---|
| 1 | 晋升是否被触发过？ | **被触发过，且是 spam**：brps 4 臂各晋升 235/227/198/227 次，kuhn 4 臂 0/183/33/89 次。但**胜率信号是坏的**（不测量胜负，实测与真实胜率脱节），晋升由随机初始化的 critic 偏置驱动；主快照 0 次（cadence 失效），池里全是冻结随机网的克隆 | 强（8/8 臂插桩重放） |
| 2 | 池动态健康吗？ | **不健康**：`main_save_every_steps` 实际生效 1000（main 轮次），首个主快照需 ~9.6e4 env-steps > 6e4 预算 → 8/8 臂**主快照为零**；brps 的池在 ~200 轮内被晋升产生的 16 个冻结随机克隆填满；PFSP 双重失效（实测 = 均匀采样） | 强 |
| 3 | off-policy 路径是 feature 还是 bug？ | **判定：bug（带意外正则化副作用）**。论文"同步 ⇒ ratio 门控 vacuous"前提在 2/3 轮次被破坏（实测 KL 0.10–0.79、15–95% 样本被 ratio 门控淘汰）；消融证明 brps 的稳定化来自"冻结随机探索者注入"，不需要任何 league 部件即可复现 | 强（门控实测 + 3 seeds 消融） |
| 4 | 参数合理性 | 逐旋钮评级见 §5：probe 规模下半数旋钮"不适用"（池空/信号坏）；`window`/`share`/轮换比/cadence **需改**；`mix`/`capacity` 修复后可用 | 中-强 |

**一句话总结**：probe 的 league 臂里，league 机制（池/PFSP/晋升）**没有一项按设计工作**；
brps 上"league 更稳"的探针结论应改写为"**2/3 预算换成冻结随机探索者数据 + 单侧
ratio 门控信任域之后更稳**"——这是两个 bug 的偶合副作用，不能作为 F2 采纳 league 的依据。

---

## 1. 证据来源与方法

1. **只读源码**：`src/mjai/league/` 四文件、`scripts/experiment.py`（league 接线）、
   `algos/nn_updates.py`（ACH 门控）、`pipeline/rollout.py`、`algos/transition.py`、
   `configs/exp/{kuhn,brps}_ach_mlp_league.yaml`。
2. **probe 产物直读**：8 个 league 臂的 `config.json`/`train_curve.json`/checkpoints/TB。
   TB 全量 tag：`eval/*` + `train/{policy_loss,value_loss,entropy,approx_kl,clip_frac,
   explained_variance}`——**无任何 league 遥测**（连已算出的 `gate_off_frac` 都被
   `experiment.py:_log_stats` 丢弃）。
3. **运行期验证**（脚本均在 `runs/league_health/`，`uv run python`）：
   - `diag_static.py`：copy_weights / PFSP 的运行时验证（见 §2.2、§3.3）。
   - `diag_instrumented.py --game {kuhn,brps} --steps 60000 --seed {0..3}`：
     **8/8 league 臂全量插桩重放**（晋升计数、wired vs 真实胜率、逐角色门控统计）。
     汇总固化于 `runs/league_health/instr_summary.json`。
   - `diag_ablation.py --seed {0,1,2}`：brps 2e4 步消融（§4.3）。
   - 确定性校验：消融 mirror 臂在 2e4 步的 NashConv 与 probe mirror 臂
     逐位一致（4.683/8.593/1.842 vs 4.683/8.593/1.842），证明重放/消融与
     原始 16 臂动力学一致。

---

## 2. Q1：晋升机制在 probe 中是否被触发过？

### 2.1 实况：触发过，且频繁；但池里没有一个主快照

8/8 league 臂插桩重放（probe 同配置同 seed，6e4 步）：

| 臂 | 晋升次数 | wired 胜率信号 | 真实胜率（seat-0 正回报占比） | 真实平均回报 | 主快照入池数 |
|---|---|---|---|---|---|
| kuhn seed_0 | **0** | 10.8% | 43.8% | −0.130 | 0 |
| kuhn seed_1 | **183** | 83.1% | 42.0% | −0.167 | 0 |
| kuhn seed_2 | **33** | 36.5% | 45.4% | — | 0 |
| kuhn seed_3 | **89** | 61.2% | 45.4% | — | 0 |
| brps seed_0 | **235** | 91.0% | 34.0% | −3.65 | 0 |
| brps seed_1 | **227** | 100.0% | 33.0% | −4.53 | 0 |
| brps seed_2 | **198** | 92.2% | 28.3% | — | 0 |
| brps seed_3 | **227** | 90.0% | 32.0% | — | 0 |

- 首次晋升可早至 `_main_steps=3`（约第 9 轮；brps seed_0 与 kuhn seed_3 均实测）——
  此时池还完全为空，靠 §6-B6 的 `-1` 伪成员通道完成"击败 70% 的池"。
- brps seed_0：池在 ~200 轮内填满 16 个成员并保持满员；235 次入池记录中
  **173 次 league_exploiter + 62 次 main_exploiter，0 次 main**。
  成员全是同一（两个）冻结随机网的克隆（见 §2.2），晋升后 eviction 因
  `win_rates` 恒空而打分为 −inf，淘汰次序实质任意。

### 2.2 根因①：胜率信号不测量胜负（解析 + 实测双重证据）

`league_controller.py:113-114`：`mean_adv = batch.advantages.mean()`（**双座位合并**
的 batch），`won = mean_adv > 0`。

解析（brps，每座位每局 1 个决策点，GAE 单步）：零和 ⇒ r₀+r₁=0，逐 episodes 合并后

```
mean_adv_pooled = mean( (r₀−V_e) + (r₁−V_m) ) / 2 = −(V_e + V_m)/2
won ⟺ V_exploiter + V_main < 0
```

即信号的符号 = **两个 critic 误差之和的符号**，与谁赢无关。更糟的是 main 的 critic
在 exploiter 轮次被 value loss 训练去拟合** exploiter 的（负）回报**（约 2/3 的更新
都在喂这个目标），把 `V_m` 系统性地压负 → brps 中 `won` 实测 90–100%。kuhn 中
符号方向由两个冻结随机 critic 的初始化决定，实测 10.8%–83.1% 逐 seed 乱跳，
而四个 seed 的真实胜率都在 42–45%。**该指标在任何游戏、任何 seed 下都不跟踪
真实胜率**（真实胜率 28–45% 全程稳定，信号 10.8%–100% 全程饱和/饿死）。

根因②（决定"重置"语义）：`experiment.py:210-220` 的 `copy_weights` 闭包要求
`hasattr(src,"logits") and hasattr(src,"values")`——`MLPSharedActorCritic` 只有
`policy_head/value_head/torso`，**对 MLP 是静默空操作**（`diag_static.py` 运行期
证实：拷贝后 exploiter 权重逐位不变、与 main 不相等；而 `_clone` 走的
snapshot/restore 路径正常）。后果：

- 两个 exploiter 从构造起就是**独立随机初始化的冻结网**，全程不变（它们从不接受
  梯度，`reset to_main` 又是空操作）；
- 晋升存入池的快照 = 冻结随机网的克隆（快照路径是好的）。

所以在 probe 里，"exploiter"这个名字名不副实：它们是**两个冻结的随机探索者**。

### 2.3 根因③：主快照 cadence 实际生效值 = 1000 main 轮次（probe 内 0 次）

- 接线：`experiment.py:229` `main_save_every_steps=cfg.save_every_steps` →
  YAML `save_every_steps: 1000`（已读两份 league YAML 确认）。**不是** LeagueConfig
  默认 200（`manager.py:36`）。
- 单位错位：该值计数的是 **main 轮次**（`manager.py:96-98`，`record_main_round`
  每个 MAIN collect 加一），不是 env-steps。
- probe 总量：brps 1875 轮（main 轮 625）、kuhn 1651–1660 轮（main 轮 ~550–554）。
  首次主快照在第 1000 main 轮 ≈ 3000 总轮 ≈ **9.6e4（brps）/ 1.09e5（kuhn）env-steps
  > 6e4 预算** ⇒ 8/8 臂主快照 0 次（插桩实测 `pool_adds` 无 main 记录）。

### 2.4 若要在 probe 规模"健康地可触发"（建议，未改代码）

前提是先修 §2.2 的两个 bug（信号改 seat-0 逐 episode 胜率；copy_weights 改走
snapshot/restore），然后：

- **信号粒度**：每轮聚合成"本轮 32 episodes 的真实胜率"（se≈0.088），窗口 20 轮
  = 640 episodes（se≈0.0198），阈值 0.55 ≈ +2.53σ，误晋升率 ~0.6%/次检查——
  阈值本身在正确信号下可用，**当前逐个"轮次布尔"喂法则不可用**：
  P(Bin(10,.5)≥6)=0.377、P(Bin(20,.5)≥11)=0.412，纯噪声也会频繁过阈。
- **cadence**：probe 规模建议 `main_save_every_steps ≈ 20–50`（main 轮），
  约 2–5k env-steps 一个快照，池在 ~10–40% 进度处填满 16 员、池龄跨度
  ~1.5–7.7e4 env；现状 1000 = 池恒空，200（默认）= 2–3 员勉强。
- **share**：修复前建议把 live main 从 share 分母中剔除（§6-B6）。

---

## 3. Q2：池动态

### 3.1 实际生效节奏 × capacity=16：probe 内全是病态

- 主快照节奏：0 次/6e4 步（§2.3）。**池的历史多样性为零**——唯一的入池来源是
  坏信号驱动的晋升 spam（§2.1），成员全是冻结随机克隆，"太新/太旧"的权衡
  在 probe 里根本不存在：brps 是"化石垃圾填满"，kuhn seed_0 是"全空"。
- capacity=16 本身在 probe 从未被合法触达（被 spam 填满不算）；FIFO-evict-main-
  first 的逻辑没有 main 可 evict 时退化为"−inf 打分任意淘汰 exploiter"。

### 3.2 反事实算术（推测，标注为推算）

以 brps 每 main 轮 ≈ 96 env-steps（3 轮 × 32）计：

| cadence（main 轮） | 6e4 probe | 1e7 规模（≈104k main 轮） |
|---|---|---|
| 1000（现状） | 0 快照 | ~104 快照；16 员池龄跨度 ≈ 1.54e6 env（~15% 训练历史）——可用 |
| 200（LeagueConfig 默认） | 3 快照（main 轮 200/400/600） | ~520 快照；池龄跨度 ≈ 3.1e5 env（~3%）——偏新但可接受 |
| 20–50（§2.4 建议） | 13–31 快照，池成熟 | 过密，无必要 |

结论：cadence 数值在 1e7 下两个现值都还能用；**坏的是单位语义（轮次冒充
steps）与 probe 尺度的失配**。

### 3.3 PFSP ε=0.05：双重失效（运行期实测）

- 失效一：`manager.py:89` 永远传 `learner_member_id=None` → `opponent_sampler.py:121-123`
  对所有候选返回默认 0.5 → 权重恒为 1/(0+0.05)=20 → **桶内均匀**。
  `diag_static.py` 实测：8 个 winrate 0.50–0.85 递增的成员，抽样频率
  0.121–0.130（均匀期望 0.125）；若正确传入 learner id，频率应为
  0.374/0.183/…/0.045（实测与理论值逐位吻合——采样器本身没写错，是接线没喂数据）。
- 失效二：`CheckpointStore.update_win_rate` **在生产代码中无任何调用者**
  （全仓 grep，仅测试调用）→ `win_rates` 恒空 → 即使修了失效一也永远走 0.5 默认；
  同时 exploiter eviction 打分恒 −inf（§3.1）。
- 附带：就算两处都修好，"未测量 ⇒ 0.5 ⇒ 权重最大"的设计让新成员被优先采样
  （PFSP-attractive），这是合理的新成员探测行为，值得保留。

---

## 4. Q3：§2.2 off-policy 路径裁决

### 4.1 实测：门控在 exploiter 轮次真实咬合，且咬合很深

插桩逐角色统计（每次 NNActorCriticUpdate.step 前向重算 ratio）：

| 臂 | 角色 | ratio 落出 (0.5,1.5) 占比 | gate_off_frac | approx_kl |
|---|---|---|---|---|
| brps seed_0 | main | **0.0%** | 0.085 | ~2e-9 |
| brps seed_0 | main_exploiter | **95.5%** | 0.268 | 0.676 |
| brps seed_0 | league_exploiter | **89.7%** | 0.427 | 0.747 |
| brps seed_1 | main_exploiter | 93.5% | 0.573 | 0.788 |
| kuhn seed_0 | main | **0.0%** | 0.000 | ~1e-10 |
| kuhn seed_0 | main_exploiter | 15.4% | 0.109 | 0.095 |
| kuhn seed_0 | league_exploiter | 65.4% | 0.457 | 0.167 |
| kuhn seed_3 | main_exploiter | 53.1% | 0.387 | 0.146 |

- main 轮次：ratio 恒 1（KL ~1e-9），门控 vacuous——**论文 p28 前提成立** ✓。
- exploiter 轮次（占 2/3 预算）：behavior 是冻结随机网，KL 0.10–0.79 nats，
  15–95% 样本落出 ratio 门控区间。门控是**单侧**的：正优势样本要求
  ratio<1.5、负优势要求 ratio>0.5，于是"main 已偏离随机策略的方向"被精确
  封锁，"尚未偏离的方向"放行——构成一个以随机行为策略为锚的**硬信任域**。
  此外 value loss 在这些轮次把 main 的 critic 朝 exploiter 的回报拟合（§2.2）。
- `1/old_probs` 权重：按 ACH 形式它是对"全动作求和"的精确 IS（E_{a~π_old}
  [f(a)/π_old(a)] = Σf(a)），任意 behavior 下无偏；真正出问题的是 ratio 门控
  按样本淘汰破坏求和完整性 + 优势来自 exploiter 的 critic/回报。实测
  mean 1/old_probs ≈ 3.0（brps）/2.0（kuhn）≈ 动作数，随机锚下幅值良性。

### 4.2 论文假设对照

论文 ACH（Algorithm 2 / p28 注记）的收敛论证（hedging/势能博弈框架）建立在
**同步单线程 ⇒ π=π_old ⇒ ratio≡1、门控 vacuous** 之上。league 接线让 2/3 的
更新消费"过期自己/随机探索者"的跨策略数据：ratio≠1 恒成立、门控按单侧规则
淘汰 15–95% 样本、价值目标来自另一个智能体的回报。**这不是论文分析过的算法
变体，其收敛保证不能援引论文**；它也不是 IMPALA 式异步（那里 ratio 门控正是为
off-policy 设计的），因为这里的 behavior 不是"稍旧的自己"而是"永不训练的
随机网 + 被 spam 的池"。

### 4.3 消融：brps 的稳定化不需要任何 league 部件

`diag_ablation.py`：brps 2e4 步，3 seeds。`randinject` = 自写 30 行控制器：
三角色轮换 [mirror-seat0, 冻结随机网A vs main, 冻结随机网B vs main]，
**无池、无晋升、无采样器**——恰好复现 probe league 臂的有效动力学。

| seed | mirror @2e4 | randinject @2e4 | probe league @2e4（对照） |
|---|---|---|---|
| 0 | 4.683 | **1.635** | 2.018 |
| 1 | 8.593 | **2.059** | 7.047 |
| 2 | 1.842 | **3.442** | 6.290 |
| 3 | (50.0，崩溃) | — | 5.827 |

randinject 三臂均值 2.38，无崩溃迹象；与 league 臂同向且同量级地低于 mirror。
机制解释：mirror 的崩溃模式是自我强化的近纯策略坍缩（seed_3 NC=50）；随机锚
信任域在 π_main(a) > 1.5·π_rand(a) 时切断正反馈，从动力学上禁止坍缩。

### 4.4 裁决

**该路径是 bug，不是 feature。** 分两层：

1. **实现层**：路径的实际形态由两个 bug 决定（冻结随机 exploiter = copy_weights
   空操作；晋升 spam/池内容 = 坏胜率信号）。它产生的"稳定化"是偶合副作用。
2. **设计层**：即使修好 bug，"exploiter 轮次梯度落 main"在论文假设下无收敛
   论证（§4.2）；2/3 预算 off-policy 在 kuhn 上还能与 mirror 打平已属侥幸
   （§4.1 kuhn 门控咬得也不轻）。

brps mirror 崩溃 vs league 稳定的证据链完整闭合为：**随机探索数据 + 单侧
ratio 门控 = 意外正则化**。若想要这个效果，应以显式机制实现（如均匀混合探索、
行为策略锚定信任域、或对 exploiter 数据做规范 IS），而不是继承 bug。

---

## 5. Q4：参数合理性评级表

评级：✅ 适用 / ⚠️ 勉强 / ❌ 需改 / ⛔ 不适用（机制未生效，数值无意义）。

| 旋钮 | 当前值 | probe 规模（6e4） | 1e7 规模 | 理由 |
|---|---|---|---|---|
| mix 0.5/0.3/0.2 | 0.5/0.3/0.2 | ⛔（池空 ⇒ 退化为 100% live main；brps 的 20% exploiter 桶抽到的是随机克隆） | ⚠️（修复后可用的起步值；0.5 偏保守，利于稳定） | 依赖池与晋升先修好才有意义 |
| promo 阈值 0.55/0.55 | 0.55 | ⛔（信号坏，阈值无意义） | ⚠️→✅（正确信号下 = +2.53σ/窗口，可用；建议固化"每轮聚合真实胜率再入窗"） | 先修信号；数值本身合理 |
| share 0.70 | 0.70 | ❌ | ❌→✅ | `-1` 伪成员计入分母，空池 1/1 即可晋升（§6-B6）；剔除后 0.70 合理 |
| promo_window 20 | 20（**轮次**，非文档所称 episodes） | ❌（单位语义错；信号坏） | ⚠️（修复后 20 轮×32 eps = 640 eps/窗，统计量充足） | 文档与单位需改；ME/LE 最小条目 10 vs 3 不对称 |
| capacity 16 | 16 | ⛔（从未被合法填满） | ✅（2p 小游戏够用；FIFO-evict-main 合理） | 需先修 eviction 打分（win_rates 恒 −inf） |
| reset_mode to_main | to_main | ⛔（对 MLP 是空操作） | ⚠️ | **实现必须修**（走 snapshot/restore）；设计上"晋升即复制 main"使 exploiter 瞬间 wr≈0.5，靠近再晋升阈值，天然 churn——修复后建议重新评估 vs `random` |
| 角色轮换 1:2 | [MAIN, ME, LE] | ❌（2/3 预算 off-policy；brps 靠它侥幸稳定，kuhn 靠它损失 2/3 on-policy 剂量） | ❌ | 论文无此设计；建议 main 占比 ≥1/2，或对 exploiter 数据显式 IS/丢弃；任何保留形式都需要 §4 级别的论证 |
| main_save_every_steps | 接线 1000 / 默认 200（**main 轮次**） | ❌（1000 ⇒ 0 快照；200 ⇒ 2–3 快照） | ⚠️（1000 ⇒ 池龄 ~15% 历史可用；200 ⇒ ~3% 偏新） | 单位语义必须改（建议直接以 env-steps 表达）；probe 规模建议 20–50 |
| PFSP ε=0.05 | 0.05 | ⛔（双重失效，§3.3） | ⛔→✅ | 修好接线（传 learner id + 周期性测量写回）后默认 0.5 新成员优先是合理设计 |

### 缺失遥测（建议新增 TB tag，不改代码）

按 §6 的 D9 单 writer 约定，由 runner/experiment 层统一记录：

- `league/pool_size`、`league/pool_size_by_role/{main,main_exploiter,league_exploiter}`
- `league/promotions_total`、`league/promotions/{main_exploiter,league_exploiter}`（累计）
- `league/exploiter_true_winrate/{role}`（每轮 seat-0 正回报 episode 占比）、
  `league/exploiter_mean_return/{role}`
- `league/win_signal_wired/{role}`（现信号的均值；修复前后对照用）
- `league/pool_age_oldest_rounds`、`league/pool_age_mean_rounds`（以 main 轮/env-step 计）
- `league/offpolicy/approx_kl_by_role/{role}`、`league/offpolicy/ratio_culled_frac_by_role/{role}`
- `train/gate_off_frac`（`NNActorCriticUpdate` 已算在 `stats.extra`，`_log_stats` 目前丢弃）
- `league/opponent_bucket/{main,history,exploiter}` 命中率（验证 mix 是否生效）

---

## 6. bug / 设计缺陷清单（按严重度）

| # | 严重度 | 位置 | 缺陷 | 证据 |
|---|---|---|---|---|
| B1 | **Critical** | `experiment.py:210-220` | `copy_weights` 对 MLP 静默空操作（检查 `.logits/.values`，MLP 无此属性）⇒ exploiter 永不热启动、reset 永不生效；违反 AGENTS.md"禁止静默退化" | `diag_static.py` 运行期证实；`_clone`（snapshot/restore）正常 |
| B2 | **Critical** | `league_controller.py:113-114` | 胜率信号 = 双座位合并优势均值 > 0，解析上等价于"两 critic 误差之和 < 0"，不测量胜负；brps 90–100% 假胜（晋升 spam），kuhn 10.8–83.1% 逐 seed 乱跳；与真实胜率（28–45%）全程脱节 | §2.1–2.2，8/8 臂实测 |
| B3 | **High** | `experiment.py:229` + `manager.py:96-98` | `main_save_every_steps` 复用 `save_every_steps=1000` 且计数单位是 main 轮次 ⇒ probe 内主快照 0 次、池历史多样性为零；命名（steps）与单位（rounds）不符 | §2.3 |
| B4 | **High** | `manager.py:89` + `checkpoint_store.py:123` | PFSP 双重失效：`learner_member_id` 恒 None ⇒ 均匀采样；`update_win_rate` 生产零调用 ⇒ win_rates 恒空、exploiter eviction 打分恒 −inf | §3.3 实测 + 全仓 grep |
| B5 | **High（设计）** | `league_controller.py:88-100` + Trainer | exploiter 轮次（2/3 预算）梯度落 main：ratio 门控单侧淘汰 15–95% 样本、value 目标来自 exploiter 回报；论文同步假设被破坏，收敛保证不可援引；probe 的稳定化经消融证实为偶合正则化 | §4 全节 |
| B6 | **Medium** | `manager.py:112,119-135` | LE 晋升把 live main 记为 `-1` 伪成员计入 share 分母 ⇒ 空池时 1/1≥0.70 即可晋升（第 ~9 轮即发生） | §2.1 首次晋升 main_steps=3 |
| B7 | **Medium** | `experiment.py:_log_stats` + 全 league 模块 | 零 league 遥测入 TB；`gate_off_frac` 已算出但被丢弃；池/晋升状态不落盘 ⇒ 事后无法审计（本次只能靠全量重放） | TB tag 全量列举 |
| B8 | **Low** | `manager.py:105,126` vs YAML 注释 | `promo_window` 单位是轮次但文档/注释称 episodes；ME 最小条目 window//2=10 与 LE 的 3 不对称 | 源码对照 |
| B9 | **Low（设计）** | Trainer 单 UpdateRule 架构 | exploiter 从不训练 ⇒ 只有 main 退化时它才能真赢——晋升机制实质是"main 退化探测器"，与 AlphaStar 式 exploiter（持续训练主动攻击）不同；是否合乎设计意图需 F2 明确 | 源码 + §4.4 |

---

## 7. 局限与诚实声明

1. 原始 16 臂磁盘产物**不含**任何 league 遥测，§2.1 的晋升计数来自同配置同 seed
   的**独立插桩重放**；端到端确定性已由消融 mirror 臂逐位复刻 probe 值佐证，
   但严格说这是"等价重跑"而非"原臂日志"。
2. `instr_<game>.json` 每游戏只保留最后一个 seed 的逐轮明细（同路径覆盖）；
   全部 8 臂汇总数字以运行时 stdout 为准，已固化 `instr_summary.json`。
3. 消融仅 brps × 3 seeds × 2e4 步，量级小；kuhn seed 1–3、brps seed 1–3 的
   逐轮门控统计未逐一展开（摘要数字齐全）。
4. §3.2 的 1e7 推算为算术外推（每 main 轮 env-steps 按 probe 实测值），未实测。
5. git status 显示 src/、configs/ 存在**本检查开始之前**的未提交修改
   （oshi_zumo 配置、experiment.py 等）；本检查对其零改动（全部写入仅限
   `docs/league_health_check.md` 与 `runs/league_health/`）。

---

## 附录 A：关键命令

```bash
# 静态机制运行期验证（copy_weights 空操作 / PFSP 均匀）
uv run python runs/league_health/diag_static.py

# 8/8 league 臂插桩重放（晋升计数、wired vs 真实胜率、逐角色门控统计）
uv run python runs/league_health/diag_instrumented.py --game brps --steps 60000 --seed 0
uv run python runs/league_health/diag_instrumented.py --game brps --steps 60000 --seed 1
uv run python runs/league_health/diag_instrumented.py --game brps --steps 60000 --seed 2
uv run python runs/league_health/diag_instrumented.py --game brps --steps 60000 --seed 3
uv run python runs/league_health/diag_instrumented.py --game kuhn --steps 60000 --seed 0
uv run python runs/league_health/diag_instrumented.py --game kuhn --steps 60000 --seed 1
uv run python runs/league_health/diag_instrumented.py --game kuhn --steps 60000 --seed 2
uv run python runs/league_health/diag_instrumented.py --game kuhn --steps 60000 --seed 3

# brps 消融：mirror vs 随机注入器（无 league 部件），2e4 步 × 3 seeds
uv run python runs/league_health/diag_ablation.py --seed 0 --steps 20000
uv run python runs/league_health/diag_ablation.py --seed 1 --steps 20000
uv run python runs/league_health/diag_ablation.py --seed 2 --steps 20000

# probe 产物直读（config/curve/checkpoints/TB tags）
ls runs/league_probe/*_league/seed_*/checkpoints
uv run python -c "… EventAccumulator('runs/league_probe/brps_league/seed_0/tb') …"
```

产物清单：`runs/league_health/diag_static.py`、`diag_instrumented.py`、
`diag_ablation.py`、`instr_summary.json`（8 臂汇总）、`instr_{kuhn,brps}.json`
（末 seed 逐轮明细）、`instr_{kuhn,brps}/seed_*/`（重放臂 tb/checkpoints）、
`ablation/brps_{mirror,randinject}/seed_{0,1,2}/`（消融臂曲线）。
