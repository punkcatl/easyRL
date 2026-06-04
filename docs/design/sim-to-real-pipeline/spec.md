# Sim-to-Real Pipeline Design Spec

## 1. Overview

A complete sim-to-real locomotion pipeline: PPO trains a Teacher policy with privileged information under Curriculum Domain Randomization, then a Student with RMA (Rapid Motor Adaptation) module distills the Teacher via behavior cloning. Validated through sim-to-sim transfer tests.

Environments: Ant-v4, Hopper-v4, Humanoid-v4, Pusher-v4.

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      Phase 1: Teacher Training                    │
│                                                                    │
│  MuJoCo (Ant/Hopper/Humanoid/Pusher)                              │
│       + Curriculum DR (friction/mass/ext_force/actuator)           │
│       + Vectorized Env (parallel sampling)                         │
│                         ↓                                          │
│  PPO (Gaussian Policy + obs normalization + reward scaling)        │
│  Input: proprioception + privileged info [μ, mass, F_ext, act]    │
│  Output: joint actions                                             │
└───────────────────────────────┬────────────────────────────────────┘
                                │ save Teacher weights + collect trajectories
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Phase 2: Student Distillation                 │
│                                                                    │
│  Adaptation Module: [obs_{t-49}, ..., obs_t] → latent z (16-dim) │
│  Base Policy:       [obs_t, z] → action                           │
│  Loss:              L2(action_student, action_teacher)             │
│                                                                    │
│  Training data: Teacher trajectories under DR environments         │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Phase 3: Sim-to-Sim Validation               │
│                                                                    │
│  Groups:                                                           │
│    A. Baseline (no DR)                                             │
│    B. DR only (Teacher without privileged input at test time)      │
│    C. Full pipeline (DR + Teacher-Student + RMA)                   │
│                                                                    │
│  Test domains:                                                     │
│    - In-distribution: fixed params within training range           │
│    - Out-of-distribution: params 30% beyond training range         │
│    - Perturbation: random impulse forces during execution          │
│                                                                    │
│  Metrics: survival steps / forward velocity / cost of transport /  │
│           perturbation recovery time                               │
└──────────────────────────────────────────────────────────────────┘
```

## 3. Module Structure

```
applications/sim_to_real/
├── __init__.py
├── config.py                       ← all hyperparameters
├── envs/
│   ├── __init__.py
│   ├── domain_randomization.py     ← DR wrapper (Curriculum + layered params)
│   └── vectorized_env.py           ← multi-env parallel sampling
├── agent/
│   ├── __init__.py
│   ├── ppo_continuous.py           ← PPO (obs norm + reward scaling + GAE)
│   ├── teacher.py                  ← Teacher policy (proprio + privileged)
│   └── student.py                  ← Student (Base Policy + Adaptation Module)
├── train_teacher.py                ← Phase 1 entry
├── train_student.py                ← Phase 2 entry (BC distillation)
├── evaluate.py                     ← Phase 3 entry (sim-to-sim validation)
├── results/                        ← training outputs
└── docs/
    └── theory.md                   ← bilingual design rationale
```

## 4. Domain Randomization Parameters

### 4.1 Parameter Categories

| Category | Parameter | Initial Range | Final Range (Curriculum) |
|---|---|---|---|
| Dynamics | body mass scale | [0.95, 1.05] | [0.7, 1.3] |
| Dynamics | body inertia scale | [0.95, 1.05] | [0.7, 1.3] |
| Contact | ground friction μ | [0.9, 1.1] | [0.5, 1.5] |
| External | random push force (x/y/z) | [0, 5] N | [0, 50] N |
| External | push pulse interval | every 200 steps | every 100 steps |
| Actuator | torque gain scale | [0.95, 1.05] | [0.8, 1.2] |
| Actuator | action delay | 0-1 steps | 0-3 steps |

### 4.2 Curriculum Strategy

Linear growth from episode 0 to episode N/2, then hold at final range. Implemented as:

```
progress = min(episode / (total_episodes / 2), 1.0)
current_range = initial_range + progress * (final_range - initial_range)
```

### 4.3 Implementation

DR is applied as a gymnasium Wrapper. On each `reset()`:
1. Compute curriculum progress
2. Sample parameters from current range
3. Modify MuJoCo model XML attributes (mass, friction, damping, gain)
4. Store sampled values as privileged info for Teacher

Actuator delay is implemented as a FIFO action buffer.

## 5. PPO Agent (Enhanced for Locomotion)

### 5.1 Enhancements over Vanilla PPO

| Feature | Purpose |
|---|---|
| Observation normalization (running mean/std) | MuJoCo obs dimensions vary in magnitude |
| Reward scaling (running std) | Prevent critic value explosion |
| Vectorized environments | Parallel sampling for speed |
| Gradient clipping (max_grad_norm=0.5) | Training stability |
| GAE (λ=0.95) | Advantage estimation |

### 5.2 Network Architecture

Both Teacher and Student Base Policy use:

```
Actor:  input → 256 → 256 → action_dim (Gaussian: mean + log_std)
Critic: input → 256 → 256 → 1
```

### 5.3 Key Hyperparameters

```
lr: 3e-4
gamma: 0.99
gae_lambda: 0.95
clip_eps: 0.2
epochs: 10
batch_size: 4096
num_envs: 16 (vectorized)
n_steps_per_update: 2048 (per env)
```

## 6. Teacher Design

### 6.1 Observation Space (Ant-v4 example)

| Component | Dim | Content |
|---|---|---|
| Proprioception | 27 | qpos, qvel (MuJoCo default obs) |
| Privileged: friction | 1 | current μ value |
| Privileged: mass | 1 | mass scale factor |
| Privileged: ext_force | 3 | F_x, F_y, F_z |
| Privileged: actuator | 2 | gain scale, delay steps |
| **Total** | **34** | |

### 6.2 Per-Environment Dimensions

| Environment | Proprio Dim | Privileged Dim | Total | Action Dim |
|---|---|---|---|---|
| Ant-v4 | 27 | 7 | 34 | 8 |
| Hopper-v4 | 11 | 7 | 18 | 3 |
| Humanoid-v4 | 376 | 7 | 383 | 17 |
| Pusher-v4 | 23 | 7 | 30 | 7 |

### 6.3 Training

Standard PPO with Curriculum DR. Train until convergence on each environment (track average reward plateau).

## 7. Student Design (RMA)

### 7.1 Adaptation Module

```
Input:  50 frames of proprioception concatenated
        e.g., Ant: 50 × 27 = 1350 dim
Network: Linear(1350, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128, 16)
Output: latent z (16 dim)
```

The latent z implicitly encodes environment parameters (friction, mass, etc.) by observing how the robot responds to its actions over time (implicit system identification).

### 7.2 Base Policy

```
Input:  obs_t (proprio) + z (16 dim)
        e.g., Ant: 27 + 16 = 43 dim
Network: Linear(43, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, action_dim)
Output: action (deterministic at deployment)
```

### 7.3 Distillation Training

1. Load trained Teacher
2. Run Teacher in DR environments, collect (obs_history, obs_t, privileged_info, action_teacher) tuples
3. Train Student end-to-end:
   - Adaptation Module: obs_history → z
   - Base Policy: (obs_t, z) → action_student
   - Loss: MSE(action_student, action_teacher)
4. No RL reward needed — pure supervised learning

### 7.4 Data Collection

Collect ~1M transitions from Teacher across diverse DR settings. Store in replay buffer or save to disk.

## 8. Sim-to-Sim Validation

### 8.1 Test Configurations

| Test Domain | Description | Purpose |
|---|---|---|
| Nominal | Standard MuJoCo defaults | Sanity check |
| In-distribution | Fixed params within DR range (e.g., mass=1.2, μ=0.7) | Generalization within training |
| Out-of-distribution | Params 30% beyond DR range (e.g., mass=1.6, μ=0.35) | Robustness to unseen domains |
| Perturbation | Nominal params + 50N impulse every 50 steps | Recovery ability |

### 8.2 Metrics

| Metric | Definition | Unit |
|---|---|---|
| Survival steps | Steps before falling (terminated) | steps |
| Forward velocity | Average x-velocity while alive | m/s |
| Cost of transport | Energy / (mass × distance) | dimensionless |
| Recovery time | Steps from perturbation to stable gait | steps |

### 8.3 Comparison Groups

| Group | Description |
|---|---|
| A. Baseline | PPO trained without DR, standard env |
| B. DR only | PPO trained with Curriculum DR, tested without privileged info |
| C. Full pipeline | Teacher (DR) + Student (RMA) distillation |

### 8.4 Expected Results

```
              Nominal    In-Dist    Out-of-Dist    Perturbation
Baseline      ✓ good     △ degrades  ✗ fails       ✗ fails
DR only       ✓ good     ✓ good      △ degrades    △ partial
Full pipeline ✓ good     ✓ good      ✓ robust      ✓ recovers
```

### 8.5 Outputs

- Comparison table (survival rate, avg speed per test domain)
- Survival curves (steps vs. parameter deviation)
- Training curves (reward vs. episode for each group)
- Velocity profile under perturbation (time series)

## 9. Dependencies

- `gymnasium[mujoco]`: MuJoCo environments (already installed)
- `torch`: neural networks
- `numpy`, `matplotlib`: data and visualization

## 10. Deliverables

1. Working Curriculum DR wrapper for 4 MuJoCo environments
2. Enhanced PPO with obs norm + reward scaling + vectorized env
3. Trained Teacher policies (one per environment)
4. Student + RMA Adaptation Module (distilled from Teacher)
5. Sim-to-sim validation results (tables + plots)
6. Bilingual theory documentation

---

# Sim-to-Real Pipeline 设计规范

## 1. 概述

完整的 sim-to-real locomotion pipeline：PPO 在 Curriculum Domain Randomization 下训练带有特权信息的 Teacher 策略，随后通过行为克隆将 Teacher 蒸馏到带有 RMA（快速电机适应）模块的 Student。通过 sim-to-sim 迁移测试验证效果。

环境：Ant-v4、Hopper-v4、Humanoid-v4、Pusher-v4。

## 2. 架构

```
┌──────────────────────────────────────────────────────────────────┐
│                      阶段 1: Teacher 训练                          │
│                                                                    │
│  MuJoCo (Ant/Hopper/Humanoid/Pusher)                              │
│       + Curriculum DR (摩擦/质量/外力/执行器)                       │
│       + Vectorized Env (并行采样)                                  │
│                         ↓                                          │
│  PPO (Gaussian Policy + obs normalization + reward scaling)        │
│  输入: 本体感知 + 特权信息 [μ, mass_scale, F_ext, actuator]        │
│  输出: 关节动作                                                    │
└───────────────────────────────┬────────────────────────────────────┘
                                │ 保存 Teacher 权重 + 采集轨迹
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                      阶段 2: Student 蒸馏                          │
│                                                                    │
│  Adaptation Module: [obs_{t-49}, ..., obs_t] → latent z (16维)    │
│  Base Policy:       [obs_t, z] → action                           │
│  Loss:              L2(action_student, action_teacher)             │
│                                                                    │
│  训练数据: Teacher 在 DR 环境中产生的轨迹                           │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                      阶段 3: Sim-to-Sim 验证                       │
│                                                                    │
│  对比组:                                                           │
│    A. Baseline (无 DR)                                             │
│    B. DR only (Teacher 去掉特权输入直接测试)                        │
│    C. Full pipeline (DR + Teacher-Student + RMA)                   │
│                                                                    │
│  测试域:                                                           │
│    - 域内: 参数在训练范围内的固定值                                  │
│    - 域外: 参数超出训练范围 30%                                     │
│    - 扰动: 运行中施加随机脉冲力                                    │
│                                                                    │
│  指标: 存活步数 / 前进速度 / 能耗 / 扰动恢复时间                    │
└──────────────────────────────────────────────────────────────────┘
```

## 3. 模块结构

```
applications/sim_to_real/
├── __init__.py
├── config.py                       ← 所有超参数
├── envs/
│   ├── __init__.py
│   ├── domain_randomization.py     ← DR wrapper (Curriculum + 分层参数)
│   └── vectorized_env.py           ← 多环境并行采样
├── agent/
│   ├── __init__.py
│   ├── ppo_continuous.py           ← PPO (obs norm + reward scaling + GAE)
│   ├── teacher.py                  ← Teacher 策略 (本体 + 特权信息)
│   └── student.py                  ← Student (Base Policy + Adaptation Module)
├── train_teacher.py                ← 阶段 1 入口
├── train_student.py                ← 阶段 2 入口 (BC 蒸馏)
├── evaluate.py                     ← 阶段 3 入口 (sim-to-sim 验证)
├── results/                        ← 训练输出
└── docs/
    └── theory.md                   ← 双语设计原理文档
```

## 4. Domain Randomization 参数

### 4.1 参数类别

| 类别 | 参数 | 初始范围 | 最终范围 (Curriculum) |
|---|---|---|---|
| 动力学 | body mass scale | [0.95, 1.05] | [0.7, 1.3] |
| 动力学 | body inertia scale | [0.95, 1.05] | [0.7, 1.3] |
| 接触 | 地面摩擦 μ | [0.9, 1.1] | [0.5, 1.5] |
| 外力 | 随机推力 (x/y/z) | [0, 5] N | [0, 50] N |
| 外力 | 推力脉冲间隔 | 每 200 步 | 每 100 步 |
| 执行器 | 力矩增益 scale | [0.95, 1.05] | [0.8, 1.2] |
| 执行器 | 动作延迟 | 0-1 步 | 0-3 步 |

### 4.2 Curriculum 策略

从 episode 0 到 total_episodes/2 线性增长，之后保持最终范围：

```
progress = min(episode / (total_episodes / 2), 1.0)
current_range = initial_range + progress * (final_range - initial_range)
```

### 4.3 实现方式

DR 实现为 gymnasium Wrapper。每次 `reset()` 时：
1. 计算 curriculum 进度
2. 从当前范围采样参数
3. 修改 MuJoCo 模型属性（mass, friction, damping, gain）
4. 将采样值存为特权信息传给 Teacher

执行器延迟通过 FIFO 动作缓冲区实现。

## 5. PPO Agent（增强版）

### 5.1 相比 Vanilla PPO 的增强

| 功能 | 作用 |
|---|---|
| Observation normalization (running mean/std) | MuJoCo 各维度量纲差异大 |
| Reward scaling (running std) | 防止 critic 值爆炸 |
| Vectorized environments | 并行采样加速 |
| Gradient clipping (max_grad_norm=0.5) | 训练稳定性 |
| GAE (λ=0.95) | 优势估计 |

### 5.2 网络结构

Teacher 和 Student Base Policy 均使用：

```
Actor:  input → 256 → 256 → action_dim (Gaussian: mean + log_std)
Critic: input → 256 → 256 → 1
```

### 5.3 关键超参数

```
lr: 3e-4
gamma: 0.99
gae_lambda: 0.95
clip_eps: 0.2
epochs: 10
batch_size: 4096
num_envs: 16 (vectorized)
n_steps_per_update: 2048 (per env)
```

## 6. Teacher 设计

### 6.1 观测空间（以 Ant-v4 为例）

| 组成部分 | 维度 | 内容 |
|---|---|---|
| 本体感知 | 27 | qpos, qvel (MuJoCo 默认 obs) |
| 特权: 摩擦 | 1 | 当前 μ 值 |
| 特权: 质量 | 1 | mass scale factor |
| 特权: 外力 | 3 | F_x, F_y, F_z |
| 特权: 执行器 | 2 | gain scale, delay steps |
| **总计** | **34** | |

### 6.2 各环境维度

| 环境 | 本体感知维度 | 特权维度 | 总计 | 动作维度 |
|---|---|---|---|---|
| Ant-v4 | 27 | 7 | 34 | 8 |
| Hopper-v4 | 11 | 7 | 18 | 3 |
| Humanoid-v4 | 376 | 7 | 383 | 17 |
| Pusher-v4 | 23 | 7 | 30 | 7 |

### 6.3 训练

标准 PPO + Curriculum DR。每个环境独立训练至收敛（监控平均 reward 稳定）。

## 7. Student 设计（RMA）

### 7.1 Adaptation Module

```
输入:  50 帧本体感知拼接
       例如 Ant: 50 × 27 = 1350 维
网络:  Linear(1350, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128, 16)
输出:  latent z (16 维)
```

latent z 通过观察机器人对动作的响应历史，隐式编码环境参数（摩擦、质量等），本质是在线系统辨识。

### 7.2 Base Policy

```
输入:  obs_t (本体感知) + z (16 维)
       例如 Ant: 27 + 16 = 43 维
网络:  Linear(43, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, action_dim)
输出:  action (部署时使用确定性输出)
```

### 7.3 蒸馏训练

1. 加载训练好的 Teacher
2. Teacher 在 DR 环境中运行，采集 (obs_history, obs_t, privileged_info, action_teacher) 元组
3. 端到端训练 Student:
   - Adaptation Module: obs_history → z
   - Base Policy: (obs_t, z) → action_student
   - Loss: MSE(action_student, action_teacher)
4. 无需 RL reward — 纯监督学习

### 7.4 数据采集

从 Teacher 在多样化 DR 设置下采集约 100 万条 transition。存储在 replay buffer 或落盘。

## 8. Sim-to-Sim 验证

### 8.1 测试配置

| 测试域 | 描述 | 目的 |
|---|---|---|
| Nominal | MuJoCo 标准默认参数 | 基础正确性检查 |
| 域内 (In-dist) | DR 范围内的固定参数 (如 mass=1.2, μ=0.7) | 训练范围内泛化能力 |
| 域外 (OOD) | 超出 DR 范围 30% (如 mass=1.6, μ=0.35) | 未见域的鲁棒性 |
| 扰动 | 标准参数 + 每 50 步施加 50N 脉冲力 | 恢复能力 |

### 8.2 指标

| 指标 | 定义 | 单位 |
|---|---|---|
| 存活步数 | 摔倒（terminated）前的步数 | steps |
| 前进速度 | 存活期间平均 x 方向速度 | m/s |
| 能耗 | energy / (mass × distance) | 无量纲 |
| 恢复时间 | 从扰动到恢复稳定步态的步数 | steps |

### 8.3 对比组

| 组别 | 描述 |
|---|---|
| A. Baseline | 无 DR，标准环境训练的 PPO |
| B. DR only | Curriculum DR 训练的 PPO，测试时去掉特权输入 |
| C. Full pipeline | Teacher (DR) + Student (RMA) 蒸馏 |

### 8.4 预期结果

```
              Nominal    域内        域外(+30%)    扰动(50N)
Baseline      ✓ 好       △ 下降      ✗ 摔倒       ✗ 摔倒
DR only       ✓ 好       ✓ 好        △ 下降       △ 部分恢复
Full pipeline ✓ 好       ✓ 好        ✓ 鲁棒       ✓ 恢复
```

### 8.5 输出物

- 对比表（各测试域的存活率、平均速度）
- 存活曲线（步数 vs 参数偏移程度）
- 训练曲线（reward vs episode，三组对比）
- 扰动下的速度时序图

## 9. 依赖

- `gymnasium[mujoco]`: MuJoCo 环境（已安装）
- `torch`: 网络训练
- `numpy`, `matplotlib`: 数据处理与可视化

## 10. 交付物

1. 4 个 MuJoCo 环境的 Curriculum DR wrapper
2. 增强版 PPO（obs norm + reward scaling + vectorized env）
3. 训练好的 Teacher 策略（每环境一个）
4. Student + RMA Adaptation Module（从 Teacher 蒸馏）
5. Sim-to-sim 验证结果（表格 + 图表）
6. 双语设计原理文档
