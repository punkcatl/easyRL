# Go2 Locomotion 训练日志

每轮训练的结果分析、诊断和迭代策略记录。

---

## Round 1

**配置：**
```python
n_steps=24, batch_size=128, epochs=5, lr=3e-4, n_iterations=3000
alive_bonus=0.1, feet_air_time_threshold=0.2
reward: lin_vel_tracking=1.0, action_rate_penalty=-0.01, torque_penalty=-0.0002
```

**收敛曲线（每50iter avg reward）：**

```
Reward
 0.6 |                           * peak=0.61 (iter 1991)
 0.0 |─────────────────────────────────────────────────── 0 线
-0.5 |                      ━━━━━━━━━━━━━━━━  (2000-3000 平台)
-1.0 |              ━━━━━━━━  (1000-2000 快速上升)
-2.0 |      ━━━━━━━  (300-1000 缓慢上升)
-3.0 | ━━━━  (0-300 随机探索)
     0    500   1000  1500  2000  2500  3000
```

**关键指标：**

| 指标 | 值 |
|------|---|
| 初始 reward（iter 1-50） | -3.11 |
| 最终 reward（last 50） | -0.41 |
| 峰值 reward | +0.61（iter 1991） |
| last 100 std | 0.149（仍在震荡） |
| 总提升 | +1.90 |

**阶段分析：**

| 阶段 | avg reward | std | 状态 |
|------|-----------|-----|------|
| 0-300 | -2.66 | 0.78 | 随机探索 |
| 300-600 | -1.76 | 0.39 | 开始学习 |
| 600-1000 | -1.21 | 0.16 | 快速上升 |
| 1000-1500 | -0.86 | 0.26 | 持续改善 |
| 1500-2000 | -0.59 | 0.30 | 接近平台 |
| 2000-2500 | -0.52 | 0.23 | 平台期 |
| 2500-3000 | -0.45 | 0.17 | 收敛但负值 |

**问题诊断：**

1. **reward 未突破 0** — 惩罚项（action_rate -0.01、torque -0.0002）在正常行走时累积量超过速度跟踪奖励
2. **震荡明显** — batch_size=128 太小，每次更新梯度方差大；Iter 2000 峰值后退化，好策略被高方差梯度破坏
3. **平台期过早** — 2000 iter 后 std 收窄但 reward 不再上升，局部最优
4. **batch < data 修复** — n_steps×num_envs=768 vs batch_size=128，每次 6 个 mini-batch，比 Round 0 的 1 个好，但数据量仍偏少

**迭代策略：**

- 增大 batch_size（128→512），减少梯度方差
- 增大 n_steps（24→48），数据量翻倍
- 降低 lr（3e-4→1e-4），保护好策略
- 增加 epochs（5→8），充分利用每批数据
- 减小惩罚权重（action_rate -0.01→-0.005，torque -0.0002→-0.0001）
- 增强正向信号（lin_vel_tracking 1.0→1.5，alive_bonus 0.1→0.2）
- 延长训练（3000→6000 iter）

---

## 训练策略：续训 vs 重开

**结论：reward 改动大时重开，reward 微调时续训（Round 3+）。**

续训（加载上一轮 checkpoint 继续训练）可以跳过早期探索阶段，直接从已有策略基础上提升。
但有一个关键约束：**Critic 的 value 估计是基于当前 reward 定义的**。
如果 reward 权重变动较大，Critic 估计会偏差，GAE 优势计算错误，策略更新方向偏移，
需要 300~500 iter 重新校准，这段成本会抵消续训节省的时间。

| 场景 | 策略 | 理由 |
|------|------|------|
| 只改 PPO 超参（lr, batch_size） | 续训 ✅ | 网络结构和 reward 不变，直接收益 |
| reward 权重小幅调整（±20%以内） | 续训 ✅ | 轻微影响，Critic 快速校准 |
| reward 权重大幅调整（>50%）| 重开 ⚠️ | Critic 校准成本高，续训优势消失 |
| 加减 reward 项 | 重开 ❌ | reward 空间变化，旧 value 估计无效 |
| 改观测/动作空间 | 重开 ❌ | 网络结构变了，不能续训 |

**各轮决策：**
- Round 1→2：reward 大幅调整（lin_vel 1.0→1.5，action_rate -0.01→-0.005），重开
- Round 2→3+：只做小幅微调，改用续训，同时重置 Adam optimizer 状态

**实现：** `train_teacher.py` 支持 `--resume path/to/checkpoint.pth` 参数，
加载网络权重但重置 optimizer，避免旧梯度动量影响新 reward 下的更新方向。

---

## Round 2

**配置：**
```python
n_steps=48, batch_size=512, epochs=8, lr=1e-4, n_iterations=6000
alive_bonus=0.2, feet_air_time_threshold=0.2
reward: lin_vel_tracking=1.5, action_rate_penalty=-0.005, torque_penalty=-0.0001
```

**目标：** reward 稳定突破 0，最终收敛到 +0.5 以上

**关键指标：**

| 指标 | 值 |
|------|---|
| 初始 reward（iter 1-50） | -2.63 |
| 最终 reward（last 50） | +14.94 |
| 峰值 reward | +23.62（iter 4678） |
| last 100 std | 1.610（震荡较大） |
| Round 1 vs Round 2 提升 | +15.35 |

**阶段分析：**

| 阶段 | avg reward | std | 状态 |
|------|-----------|-----|------|
| 0-500 | -0.11 | 1.98 | 快速突破 0 |
| 500-1000 | +2.50 | 2.80 | 震荡上升 |
| 1000-3000 | +8.00 | 1.95 | 平台期 |
| 3000-4000 | +15.43 | 3.38 | 二次突破 |
| 4000-6000 | +14.80 | 2.40 | 高位收敛 |

**可视化结果：**
机器狗四足站立，**基本不动**。诊断发现策略学到了"站着不动"的局部最优：

```
Per-step avg reward (实测):
  ang_vel_tracking: +0.44  ← 主导！站着不动 yaw=0，误差为 0，满分
  lin_vel_tracking: +0.03  ← 几乎没有贡献
  feet_air_time:    +0.00  ← 从未抬脚
```

**根本原因：** `lin_vel_x ∈ [-1.0, 1.0]` 允许采样到接近 0 的速度命令，站着不动就能得高分。reward 被钻漏洞。

**Round 2 → Round 3 决策：** reward 设计有重大漏洞，必须重开（不续训）。

---

## Round 3

**配置：**
```python
# 命令空间：强制非零前进
lin_vel_x: [0.5, 1.0]        # 禁止静止和后退
command_resample_interval: 100 # 更频繁切换，增加泛化

# Reward：大幅提高线速度权重，降低角速度权重
lin_vel_tracking: 3.0          # 1.5 -> 3.0
ang_vel_tracking: 0.2          # 0.5 -> 0.2
feet_air_time_reward: 2.0      # 1.0 -> 2.0，鼓励真实步态
low_speed_penalty: -0.5        # 新增：站着不动时额外惩罚
tracking_sigma: 0.15           # 0.25 -> 0.15，更严格
feet_air_time_threshold: 0.1   # 0.2 -> 0.1，更容易触发

# 物理
action_scale: 0.35             # 0.25 -> 0.35，更大关节摆幅

# PPO
lr: 3e-4, epochs: 10, n_iterations: 5000
```

**目标：** avg vx > 0.5 m/s，feet_air_time_reward 有正贡献（说明有真实步态），reward 突破靠速度跟踪而非站立不动。

**最终结果：**

| 指标 | 值 |
|------|---|
| 初始 reward（iter 1-50） | +2.63（R1/R2 均为负，R3 显著改善）|
| 最终 reward（last 50） | +11.79 |
| 峰值 reward | +45.31（iter 3780）|
| last 100 std | 2.44（震荡仍大）|

**阶段分析：**

| 阶段 | avg reward | std | 状态 |
|------|-----------|-----|------|
| 0-500 | +9.09 | 4.53 | 快速启动（比R2快10x）|
| 500-2000 | +15.16 | 2.52 | 稳步上升 |
| 2000-4000 | +23.27 | 7.0 | 高位震荡，峰值45 |
| 4000-5000 | +19.96 | 6.84 | 退化 |

**行为诊断（运行300步实测）：**
```
avg vx:  0.036 m/s（目标 1.0，仍然几乎不动）
feet air time: 0.000s（没有抬脚）

Reward 主导项：
  joint_pos_penalty:  -1.04/step  ← 主导惩罚！
  low_speed_penalty:  -0.46/step
  collision_penalty:  -0.32/step  ← 仍在碰撞
  lin_vel_tracking:   +0.04/step  ← 几乎没贡献
```

**问题诊断：**

每轮都在堆惩罚项，但每个惩罚项都给策略提供了新的"偷懒"方向：
- Round 1：站着不动 → ang_vel=0 → 满分
- Round 2：改大 lin_vel → 还是站着，偶然突破
- Round 3：加 joint_pos_penalty → 关节不动 = 减少惩罚，又是新局部最优

**根本原因：正向 reward 信号太弱，负向惩罚太强。**
`joint_pos_penalty` 单步 -1.04，而 `lin_vel_tracking` 最大只有 +3.0/step。
策略永远在"减少惩罚"而不是"追求速度"。

**Round 3 → Round 4 决策：** reward 设计有结构性问题，必须重开。
核心原则：**先让策略能走起来，惩罚项稍后再加**。

---

## Round 4

**核心原则：正向信号主导，惩罚项轻量，curriculum 基于实际速度**

### 第一次尝试（失败，只跑了100 iter）

命令 Curriculum 从 `[-0.1, 0.1]` 开始，触发条件是 `total_reward > lin_vel_weight * 0.8 = 4.0`。
但站着不动时 cmd ≈ 0 → lin_vel_tracking ≈ exp(0)*5.0=5.0 → 轻松超过阈值。
结果：50 iter 时 curriculum 已扩展到 [-0.5, 0.5]，策略从未真正走动。

**根本原因：** Curriculum 用 total reward 判断，而非实际行走速度。初始命令包含零，站着就是满分。

### 第二次修正（当前配置）

**修复3个问题：**

1. **Curriculum 触发条件**：改为 `tracking_ratio = actual_vx / commanded_vx > 0.5`（只有真走才扩展）
2. **初始命令范围**：`lin_vel_x: [0.3, 0.6]`（强制非零前进，杜绝站立漏洞）
3. **low_speed_penalty**：从 -0.5 → -2.0，且改为渐进式 `max(0, 1 - speed/cmd)` 而非二值判断

**配置：**
```python
# 命令范围：强制非零前进，Curriculum 基于实际速度
command_range: {"lin_vel_x": [0.3, 0.6], "lin_vel_y": [-0.1, 0.1], "ang_vel_yaw": [-0.3, 0.3]}
command_limit: {"lin_vel_x": [-1.0, 1.5], "lin_vel_y": [-0.5, 0.5], "ang_vel_yaw": [-1.0, 1.0]}
cmd_curriculum_threshold: 0.5   # tracking_ratio > 50% 才扩展
rel_standing_envs: 0.0         # Round 4 不留站立环境

# Reward
lin_vel_tracking:         5.0
ang_vel_tracking:         0.5
feet_air_time_reward:     2.0
alive_bonus:              0.5
low_speed_penalty:        -2.0   # -0.5->-2.0，渐进式，确保站着净负
action_rate_penalty:      -0.01
joint_pos_penalty:        -0.05
torque_penalty:           -0.0001
flat_orientation_penalty: -1.0
lin_vel_z_penalty:        -2.0
ang_vel_xy_penalty:       -0.05
collision_penalty:        -1.0
joint_acc_penalty:        -2.5e-7
```

**Reward 验证（站着 vs 走路）：**
- 站着不动 (cmd=0.5): lin_vel≈0.94 + ang_vel=0.5 + alive=0.5 - low_speed=2.0 = **-0.06/step**
- 正确行走 (cmd=0.5): lin_vel=5.0 + ang_vel=0.5 + alive=0.5 + feet_air>0 = **6.0+/step**
- 差距 6+ 倍，策略不可能卡在站立局部最优

**目标：** avg vx > 0.3 m/s，tracking_ratio > 0.5，feet_air_time_reward > 0。

### Round 4b 实际结果（900 iter 后终止）

```
Iter  50: reward=4.72,  track_ratio=0.14
Iter 200: reward=15.06, track_ratio=0.02
Iter 450: reward=16.16, track_ratio=0.05
Iter 650: reward=17.27, track_ratio=0.08
Iter 900: reward=13.74, track_ratio=0.04
```

**结论：失败。** Reward 涨到 15-17 但 tracking_ratio 始终 ≈ 0.02-0.08（机器人几乎不走）。
策略再次找到不走路但拿高分的漏洞：
- sigma=0.15 时 cmd=0.5 站着仍得 exp(-0.25/0.15)×5.0 = 0.94
- ang_vel_tracking 站着满分 = 0.5
- alive_bonus = 0.5/step 无条件奖励存活
- 合计站着能拿 ~1.9/step，加上减少惩罚后策略收敛到 reward≈16 但 velocity≈0

**教训：** sigma 太小 + alive_bonus + 缺乏线性前进梯度 = 站立依然是强局部最优。必须同时修复。

---

## Round 5

**核心原则：三专家共识全票通过 9 项同时实施**

基于 Round 1-4 全部历史数据，由 Reward 设计专家、PPO 超参专家、环境/探索专家独立分析后交叉评审达成共识。

### 同时实施的 9 项变更

| # | 变更 | 旧值 | 新值 | 预期影响 |
|---|------|------|------|----------|
| 1 | 观测归一化 RunningMeanStd | 无 | 每 rollout 更新 | 收敛 +30-50% |
| 2 | forward_progress 奖励 + 移除 alive_bonus | alive=0.5 | forward_progress=2.0, termination=-10 | 打破站立陷阱 |
| 3 | 初始状态随机化 | 固定站立 | joint±0.2, height[0.30,0.38], vel±0.3 | 多样性 +2-3x |
| 4 | 128 envs async + epochs 5 | 32 sync, epochs=10 | 128 async 8workers, epochs=5 | 稳定性 +20-30%, 速度 +3-5x |
| 5 | 脚部接触 body→geom | calf body 碰撞 | foot geom ("FL","FR","RL","RR") | 步态信号修正 |
| 6 | tracking_sigma 0.15→0.25 | 0.15 | 0.25 | 梯度 +2x |
| 7 | PD kp/kd + action_scale | kp=20, kd=0.5, scale=0.35 | kp=[20,35,35,...], kd=1.0, scale=0.25 | 足够推地力矩 |
| 8 | 奖励权重重构 | 见 Round 4 | lin_vel=3, feet=1.5+门控, orientation=-0.5, 移除 joint_pos/low_speed/alive | 信号纯净 |
| 9 | DR 课程化 | 全量 DR 从 iter 0 | Phase1(0-500)无DR, Phase2(500-1500)轻, Phase3(1500+)全 | 早期 +2-3x |

### 配置

```python
# Environment
num_envs: 128, vec_env_type: "async", num_workers: 8
action_scale: 0.25
kp: [20,35,35, 20,35,35, 20,35,35, 20,35,35]
kd: [1.0]*12
obs_normalize: True
init_state_randomize: True

# Reward
lin_vel_tracking:       3.0     # exp kernel, sigma=0.25
ang_vel_tracking:       0.5
forward_progress:       2.0     # clip(vel_in_cmd_dir, -0.5, 2.0)
feet_air_time_reward:   1.5     # gated by clip(body_speed/0.3, 0, 1)
base_height_reward:     1.0     # exp kernel around 0.34m, sigma=0.01
termination_penalty:    -10.0   # one-time on terminated
flat_orientation:       -0.5
lin_vel_z:              -2.0
ang_vel_xy:             -0.05
action_rate:            -0.01
torque:                 -0.00005
joint_acc:              -2.5e-7
collision:              -1.0

# PPO
epochs: 5, entropy_coef: 0.02, batch_size: 512, n_steps: 48, lr: 3e-4

# DR Curriculum
Phase 1 (iter 0-500):    无 DR
Phase 2 (iter 500-1500): friction[0.8,1.1], mass[0.95,1.05], force[0,1]
Phase 3 (iter 1500+):    friction[0.5,1.25], mass[0.8,1.2], force[0,3]

# Termination
min_body_height: 0.20, max_body_height: 0.45
```

### 验证标准（200 iter 内）

- `mean_episode_velocity > 0.1 m/s` → 站立陷阱已破
- `tracking_ratio > 0` → 机器人在追踪命令方向
- `feet_air_time > 0` → 有抬脚动作
- 如果 200 iter 后 velocity ≈ 0 → 存在更深层 bug，需 debug 而非调参

### Round 5 实际结果（1050 iter 后终止）

```
Iter   50: reward=2.43,  track=0.05 | vx: [0.30, 0.60]  | no-DR
Iter  150: reward=3.34,  track=0.47 | vx: [-0.20, 1.10] | no-DR   ← 首次走起来！
Iter  350: reward=3.10,  track=0.39 | vx: [-1.00, 1.50] | no-DR   ← curriculum 满范围
Iter  500: reward=3.57,  track=0.37 | vx: [-1.00, 1.50] | no-DR   ← 稳步提升
Iter  700: reward=4.20,  track=0.34 | vx: [-1.00, 1.50] | light-DR
Iter  750: reward=5.69,  track=0.13 | vx: [-1.00, 1.50] | light-DR ← exploit 开始
Iter  850: reward=9.51,  track=0.05 | vx: [-1.00, 1.50] | light-DR ← 崩溃
Iter 1050: reward=10.73, track=-0.01 | vx: [-1.00, 1.50] | light-DR ← 完全不走了
```

**结论：部分成功，最终失败。** 
前 500 iter 是项目首次实现真正行走（tracking 0.3-0.5），但在 light-DR 阶段策略再次发现 exploit。

**根因分析（三个同时存在的漏洞）：**
1. `base_height_reward=1.0` 站着就满分（无速度门控）
2. `forward_progress` 站着 = 0（不是负值），不够惩罚
3. `lin_vel_tracking=3.0` + sigma=0.25，cmd 范围扩展到含零区域后，站着仍能拿高分
4. Curriculum 扩展太快（一次 > threshold 就扩展），策略还没稳定就被推到更难范围

**教训：**
- 所有"无条件正奖励"（height, ang_vel）都是站立 exploit 的温床
- forward_progress 必须让站着为负，不能为零
- Curriculum 需要连续稳定才能扩展（加 stable_count）

---

## Round 6

**核心原则：forward_progress 绝对主导 + 所有正向奖励 speed-gated + 站着净负**

### 修复 3 个问题：

1. **forward_progress** 带 baseline subtraction: `clip(vel_in_cmd, -0.5, 2.0) - 0.3*cmd_norm`
   → 站着时 forward_progress = -0.3*cmd_norm（负的！）
2. **base_height_reward** speed-gated: `exp(error) * clip(body_speed/0.3, 0, 1)`
   → 站着时 = 0
3. **lin_vel_tracking 降权 3.0→1.0**，forward_progress **升权 2.0→4.0**
   → forward_progress 成为绝对主导驱动信号
4. **Curriculum 稳定条件**：需连续 10 次 tracking > threshold 才扩展

### 配置变更（vs Round 5）：

```python
# Reward
lin_vel_tracking:       1.0     # 3.0->1.0 (精调阶段用，学步阶段不该主导)
ang_vel_tracking:       0.3     # 0.5->0.3
forward_progress:       4.0     # 2.0->4.0 (绝对主导, baseline subtracted)
# forward_progress formula: (clip(vel_in_cmd_dir, -0.5, 2.0) - 0.3*cmd_norm) * 4.0
# base_height_reward: gated by clip(body_speed/0.3, 0, 1)
# Curriculum: stable_count >= 10 before expansion
```

### Reward 验证：

| 场景 | standing | walking 70% | 差距 |
|------|----------|-------------|------|
| 全范围平均 | **-0.13** | **+2.67** | **2.80** |
| cmd=0.5 | +0.07 | +3.5 | +3.4 |
| cmd=1.0 | -0.88 | +3.2 | +4.1 |
| cmd=1.5 | -1.50 | +2.8 | +4.3 |

站着在大多数命令下是净负的！只有 cmd≈0 时站着勉强为正（1.3），但初始范围不含零。

### Round 6 实际结果（5000 iter 完整跑完）

```
Phase 1 (iter 0-500):    reward 1.0→1.8, track 0.37→0.53  | no-DR, curriculum扩展中
Phase 2 (iter 500-1500): reward 1.8→2.0, track 0.44→0.58  | light-DR, 满范围稳定跟踪 ★最佳阶段
Phase 3 (iter 1500-2000): reward 2.0→2.2, track 0.4→0.6   | full-DR初期, peak=2.79 @iter2019
Phase 4 (iter 2000-3500): reward 2.2→1.6, track 0.4→0.2   | full-DR, 缓慢退化
Phase 5 (iter 3500-5000): reward 1.6→-0.3, track 0.2→0.0  | 策略崩塌
```

**最佳 checkpoint: iter 1500** (tracking≈0.55, reward≈2.0, 进入full-DR前的巅峰)

**结论：部分成功 — 项目首次实现 2000+ iter 持续行走，但后期崩塌。**

**根因分析（后期崩塌）：**
- lr=3e-4 恒定过大：full-DR 下梯度方差增大，大 lr 破坏已学到的好策略
- 无 entropy 衰减：后期探索过强，策略被噪声破坏
- PPO 典型的"遗忘性崩塌"：梯度更新幅度 > 策略改善幅度 → 净退化

**关键经验：**
1. 前 1500 iter（no-DR + light-DR）训练非常成功，reward 设计终于正确
2. Full-DR 切入后需要降 lr 保护策略
3. 最佳策略在 iter 1500 左右，应该从此处续训并加 lr 衰减

---

## Round 7

**核心原则：从最佳 checkpoint 续训 + lr/entropy 线性衰减防止崩塌**

### 策略：

从 Round 6 iter 1500 checkpoint 续训（tracking≈0.55 的最佳状态），加入：
1. **LR 线性衰减**: 3e-4 → 1e-5（over 3000 iter）
2. **Entropy 线性衰减**: 0.02 → 0.005
3. **总 iter 减少到 3000**（续训不需要那么多）
4. DR curriculum 同样从头开始（iter 0-500 no-DR 让续训网络重新适应）

### 配置：

```python
lr: 3e-4 → 1e-5 (linear decay)
entropy_coef: 0.02 → 0.005 (linear decay)
n_iterations: 3000
resume: teacher_iter1500.pth
```

### Round 7 实际结果（3000 iter 完成）

```
Iter   50: reward=3.20, track=1.33 | vx: [-0.10,1.00] | lr=3.0e-4 | no-DR  ← 续训策略太强
Iter  500: reward=3.91, track=0.63 | vx: [-1.00,1.50] | lr=2.5e-4          ← curriculum 已满
Iter 1000: reward=9.68, track=0.06 | vx: [-1.00,1.50] | lr=2.0e-4 | light-DR ← exploit!
Iter 2000: reward=9.81, track=0.01 | vx: [-1.00,1.50] | lr=1.1e-4 | full-DR
Iter 3000: reward=10.1, track=0.02 | vx: [-1.00,1.50] | lr=1.0e-5 | full-DR  ← lr衰减无效
```

**结论：失败。** lr 衰减没用——exploit 在 lr 还高的早期就已经形成，衰减只是锁住了坏策略。

### 重大发现：Round 6 iter1500 的 "成功" 被高估

**实际评估 Round 6 iter1500 checkpoint：**
```
Command: vx=0.5 m/s, 跑 200 steps
Avg vx: -0.06 m/s（几乎不走！）
策略在振荡，偶尔 ratio=0.64，大多数时候 ratio≈0 或负
```

**根因：** 训练中的 tracking_ratio 是 128 envs 瞬时值的平均，不代表持续前进。
策略学到的是"随机振荡"而非"稳定行走"——在 128 envs 中总有一些 env 在某一步恰好 vx>0，拉高了平均值。

**这意味着之前所有轮次中看到的 tracking_ratio 0.3-0.7 都可能是假象！**

### 下一步方向（Round 8 需要根本性改变）

tracking_ratio 作为监控指标是不可靠的（未 clip 的负值抵消正值）。需要：
1. tracking_ratio 改为 clip([0, 1.5])，加 pct_positive 指标
2. 每 200 iter 跑一个 deterministic eval episode，打印真实 avg_vx
3. feet_air_time 必须有上限 clip（否则倒地悬空即可刷分）

---

## Round 8

**核心原则：修复 feet_air_time exploit + 可靠监控指标 + trot schedule**

### 发现的新 Bug（Round 8 第一次尝试就崩溃）

**去掉 speed gate 后暴露了 feet_air_time 的无上限 exploit：**
- 策略学到：让机器人跌倒/弹跳，四脚悬空
- air_time 无上限累积（每脚可达 2s+），reward = 4脚 × 2s × 1.0 = 8/step
- 这就是 reward=16 的来源！speed gate 之前掩盖了这个问题

### 所有改动（vs Round 7）

```python
# go2_reward.py
feet_air_time: clip per foot to [0, 0.3]  # 加上限，每脚最多 0.3s
gait_schedule: * upright_gate             # 倒地时不给 trot 奖励
# speed gate 完全移除（不再需要）

# config.py
tracking_sigma: 0.25->0.1               # 更尖锐，减少站立拿分
base_height_sigma: 0.01->0.05           # 更平滑，早期训练友好
action_rate_penalty: -0.01->-0.1        # 抑制振荡（关键）
gait_schedule: 0.5                      # 新增 trot 步态引导

# train_teacher.py
tracking_ratio: clip到[0,1.5]，加pct30指标  # 修复虚假高值
eval: 每200 iter 跑1个 det episode打印avg_vx  # 真实指标
```

### 验证结果（Round 8 第二次，iter 0-350）

```
Iter  50: reward=0.51, track=0.10 pct30= 7%  ← 无暴涨，健康起点
Iter 200: reward=0.86, track=0.16 pct30=19%
  [EVAL] iter200: avg_vx=0.041 m/s (低但在增长)
Iter 350: reward=1.06, track=0.30 pct30=48%  ← 接近一半 envs 真的在走！
```

**这是 9 轮以来第一次看到 pct30 持续上升且 reward 不暴涨的健康曲线。**

### Round 8 最终结果（5000 iter 完整）

```
EVAL vx (cmd=0.5):
  iter  200: 0.041 m/s
  iter  400: 0.130 m/s  ← 突破
  iter  600: 0.153 m/s
  iter  800: 0.170 m/s
  iter 1000: 0.171 m/s  ← 进入 full-DR
  iter 2000: 0.182 m/s  ← full-DR 下持续提升
  iter 4000: 0.175 m/s
  iter 5000: 0.174 m/s  ← 平台期

Reward: 0.9→1.38（全程单调上升，无崩塌）✅
pct30:  ~45-55%（稳定，但未达到连续10次>50% 触发 curriculum 扩展）
Curriculum: 全程锁在 [0.30, 0.60]（未扩展）
```

**结论：项目首次实现全程稳定训练（5000 iter 无崩塌）！**
- 机器狗能稳定前进 ~0.17 m/s（命令 0.5 m/s 的 34%）
- 可视化：前冲+趴地，有前进但步态不协调

**瓶颈：** curriculum 扩展阈值 pct30>50% 连续 10 次 几乎够但没够，策略卡在小速度范围。

**Round 9 方向：**
1. 从 Round 8 final checkpoint 续训
2. 降低 curriculum 扩展阈值 0.5→0.35（让策略接触更大速度范围）
3. 提高 forward_progress 权重 3.0→4.0（增强快走激励）
4. lr 从 3e-5 继续衰减到 1e-5（保持 Round 8 末期的低 lr 继续精化）

---

## Round 9

**续训 R8 + 降低 curriculum 阈值 → eval vx 退步**

配置：从 R8 final续训，curriculum threshold 0.5→0.35，forward_progress 3→4，lr 3e-5→5e-6

结果：curriculum 扩展到 [-0.20, 1.10]，但 eval vx 从 0.173 降到 0.160
- 扩展命令范围后策略被大命令稀释，梯度方向变混乱
- 结论：**在能稳定走出 0.3 m/s 之前不能扩展 curriculum**

---

## Round 10

**策略：提高初始命令范围 [0.4, 0.8]，强迫策略学快走，收紧 curriculum 阈值到 0.6**

关键变化：
- `command_range: [0.3, 0.6] → [0.4, 0.8]`（从更高速度开始）
- `cmd_curriculum_threshold: 0.35 → 0.6`（更难扩展，先在当前范围练好）
- 从 R8 final 续训，lr 1e-4→1e-5（稍高 lr 让策略能适应新速度范围）

结果：eval vx 0.179-0.185 m/s，和 R8 持平无突破。策略天花板在"lurching"行为——前冲倒地平均速度就是 0.18。

---

## Round 11

**核心：杀死 lurching，强制 stable trot（专家分析 5 项修复）**

新 reward：
```python
forward_progress: 1.5 (tracking fraction = clip(vel/cmd, 0, 1.2), 超速无奖励)
gait_schedule:    2.0 (+ moving gate)
gait_symmetry:    1.5 (对角反相, + moving gate) [新增]
all_feet_contact_penalty: -1.5 (四脚着地扣分) [新增]
ang_vel_xy_penalty: -0.3 (6x，惩罚摔倒翻滚)
lin_vel_z_penalty:  -4.0 (2x)
```

实际结果（5000 iter，约 iter 1500 左右）：
```
iter 100: track=0.63 pct30=61%  ← 最快早期收敛
iter 200: eval_vx=0.251 m/s (85 steps)  ← 历史最高速度！但85步摔倒
iter 400: eval_vx=0.086 m/s (500 steps)  ← 学稳定性后速度降
iter 800: eval_vx=0.061 m/s
iter 1200: eval_vx=0.065 m/s  ← 低速平台
```

**新问题：探索-利用差距**
training track=0.5-0.64（噪声随机走路），deterministic eval vx 只有 0.04-0.07。
根因：`ang_vel_xy_penalty=-0.3` 过强，确定性 mean action 学到"不动=最小化翻滚惩罚"。

**Round 12 方向：** `ang_vel_xy_penalty` 降回 -0.05，其余 R11 设计保留。

---

## Round 12（最佳轮次）

恢复 `ang_vel_xy_penalty=-0.05`，修复 curriculum 只扩上界，其余 R11 设计保留。

**结果（5000 iter）：**
```
eval vx 稳定在 0.188-0.200 m/s，全程 500 steps 存活
reward: 2.6→3.1 单调上升，full-DR 无崩塌
pct30: 55-72%，track: 0.5-0.75
curriculum 扩展到 [0.30, 0.60] 后未再扩展
```
**这是项目最稳定的一轮。** 但速度平台在 ~0.19 m/s。

速度诊断（R12 checkpoint）：
```
cmd=0.3: avg_vx=0.180 m/s (60%)
cmd=0.5: avg_vx=0.191 m/s (38%)
cmd=1.0: avg_vx=0.177 m/s (18%)
```
**发现：策略速度固定在 ~0.19 m/s，与命令无关。** 这是 tracking_fraction reward 的副作用——策略只需要达到 cmd 的一定比例即可，没有动力走更快。

---

## Round 13

从 R12 续训，命令范围 [0.4, 1.0]，希望强迫更快。结果 eval vx 退步到 0.166，高速命令稀释了梯度。

---

## Round 14

**核心改变：`lin_vel_tracking` 权重 1.0→3.0，`forward_progress` 改回 raw velocity**

`tracking_fraction` reward 让策略只需追上命令的某个比例——没有绝对速度激励。
改回 `forward_progress = raw_vel * 1.5 + lin_vel_tracking=3.0 * exp_kernel`：
- 越快越好（forward_progress）
- 同时向命令速度收敛（lin_vel_tracking）

命令范围 [0.3, 0.8]，从头训练（权重变化大，续训 Critic 会出错）。

结果：eval vx 0.188-0.209 m/s，比 R12 提升但仍有天花板。速度诊断：
```
cmd=0.5: 0.209 m/s  cmd=0.8: 0.238 m/s  cmd=1.0: 0.237 m/s
```
策略速度上限 ~0.24 m/s——物理瓶颈，步幅太小（action_scale=0.25）。

---

## ⭐ Round 15 — 重大突破

**关键发现：`action_scale 0.25 → 0.35` 是真正的突破点**

之前 14 轮都在用 action_scale=0.25（小步幅），机器人腿部摆幅物理上不足以产生高速推进力。
Round 15 将 action_scale 恢复到 0.35，立竿见影。

### 配置（在 Round 14 基础上）

```python
action_scale: 0.25 → 0.35    # 关键变化
command_range: [0.3, 0.8]
lin_vel_tracking: 3.0
forward_progress: 1.5 (raw velocity)
gait_schedule: 2.0, gait_symmetry: 1.5, all_feet_contact_penalty: -1.5
```

### 训练曲线

```
Iter  50: reward=1.27, pct30=56%
Iter 100: reward=2.47, pct30=66%  ← 收敛加速
Iter 150: reward=2.85, pct30=70%  ← curriculum [0.30, 1.20]
Iter 200: reward=2.97, track=0.75 pct30=86% | [EVAL] 0.671 m/s ← 历史突破！
Iter 400: reward=3.45, pct30=95%  | [EVAL] 0.682 m/s
Iter 600: reward=3.78, pct30=97%  | [EVAL] 0.763 m/s ← 峰值
Iter 750: reward=3.93, pct30=100% | track=0.83
Iter 900: reward=4.03, track=0.97  ← 接近完美跟踪
Iter 5000: reward=4.90, pct30=100% | [EVAL] 0.614 m/s (full-DR 稳定)
Peak reward: 5.15 @iter4663
```

### 最终速度能力（deterministic eval，500 steps 全程存活）

```
cmd=0.3 m/s: 实际 0.579 m/s (193%，策略跑得比命令快)
cmd=0.5 m/s: 实际 0.614 m/s (123%)
cmd=0.8 m/s: 实际 0.803 m/s (100% ← 完美跟踪！)
cmd=1.0 m/s: 实际 0.967 m/s (97%)
cmd=1.5 m/s: 实际 1.002 m/s (67%，速度上限约 1.0 m/s)
```

### 可视化结果

cmd=1.0 m/s，500 steps 全程存活，截图可见：
- 四足对角步态（trot）：FL+RR / FR+RL 交替抬起
- 腿部大幅摆动，明显动态推进
- 身体平稳，无摔倒

### 与历史最佳对比

| 指标 | Round 8（之前最好）| Round 15（突破）|
|------|-------------------|----------------|
| eval vx | 0.18 m/s | **0.61-0.97 m/s** |
| pct30 | 45-55% | **99-100%** |
| track | 0.34-0.36 | **0.95-1.04** |
| 速度跟踪 | 与命令无关（固定速） | **接近完美跟踪** |

**突破原因总结：**
1. `action_scale=0.35`：更大步幅，物理上能产生足够推进力
2. `all_feet_contact_penalty=-1.5`：彻底消灭四脚着地的 lurching
3. `gait_symmetry=1.5`：引导对角步态形成
4. `lin_vel_tracking=3.0`：强化命令速度跟踪信号
5. Curriculum 只扩上界：避免零速命令导致确定性策略不动

Checkpoint: `results/teacher_round15.pth`

---

## 后续 TODO

### 高优先级

1. **更高速度（1.5+ m/s）**
   - 当前上限约 1.0 m/s，受限于步频和步幅
   - 方向：进一步提高 action_scale（0.35→0.45），或增大 kp（thigh/calf: 35→50）
   - 可能需要提高控制频率（50Hz→100Hz，decimation 4→2）

2. **转弯能力验证与优化**
   - 目前只测了直线，ang_vel_yaw 命令未充分测试
   - 需要跑 ang_vel=±1.0 的 eval，看转弯是否流畅
   - 可能需要对 ang_vel_tracking 加更多训练样本

3. **扰动鲁棒性测试**
   - full-DR 下测试外力扰动（push_force=5N, 10N）
   - 使用 `env.apply_force()` 接口做 benchmark

### 中优先级

4. **步态质量优化**
   - 当前是 trot（对角步态），但可能有轻微跳跃（hopping）
   - 加 air_time_variance 惩罚，强化四腿对称性
   - 加 foot_clearance 奖励，确保抬脚高度足够

5. **能耗优化**
   - 加 energy_penalty（joint_vel × torque）
   - 实际硬件部署前需要限制能耗，避免电机过热

6. **侧向行走和斜向速度**
   - 当前 lin_vel_y 命令未充分训练
   - 在 curriculum 扩展中同步增大 y 方向范围

### 低优先级（基础完善后）

7. **Student Policy 蒸馏**（真正的 Sim-to-Real 准备）
   - 用 Teacher policy 采集数据，训练 Student（无 privileged obs，只用历史观测）
   - Student 才是最终部署到真实 Go2 的策略

8. **地形课程**
   - 加入 rough/slope/stairs 地形
   - 目前只在平地训练，真实地面有凹凸

9. **Reward 精化**
   - 把训练好的 Teacher 策略作为起点，精化步态质量
   - 参考 Walk These Ways 的多模式步态（walk/trot/bound）
