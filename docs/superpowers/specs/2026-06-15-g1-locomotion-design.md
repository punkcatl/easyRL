# G1 Locomotion Project Design

Unitree G1 humanoid robot RL locomotion training using Isaac Lab + rsl_rl,
with custom Teacher-Student RMA distillation and ONNX deployment.

---

## Overview

| Item | Value |
|------|-------|
| Robot | Unitree G1 (G1_MINIMAL, 29 DOF) |
| Simulator | Isaac Lab 2.3.0 + Isaac Sim 5.1.0, GPU parallel |
| Hardware | NVIDIA RTX A4000 16GB, 12 cores, 128GB RAM |
| RL Framework | rsl_rl 3.0.1 (Phase 1) |
| Env Style | Manager-Based (inherit Isaac Lab G1 configs) |
| Location | `applications/g1_locomotion/` in easyRL repo |
| Training Goal | Lower-body walking (flat → rough terrain) |
| Pipeline | Phase 1 Teacher → Phase 2 Student RMA → Phase 3 ONNX |

---

## Project Structure

```
applications/g1_locomotion/
├── config/
│   ├── __init__.py
│   ├── flat_env_cfg.py          # inherit Isaac Lab G1FlatEnvCfg, override reward/cmd
│   ├── rough_env_cfg.py         # inherit G1RoughEnvCfg, terrain curriculum
│   └── ppo_cfg.py               # rsl_rl PPO runner config
├── mdp/
│   ├── __init__.py
│   └── rewards.py               # custom reward terms (not in framework)
├── scripts/
│   ├── train_teacher.py         # Phase 1 training entry
│   ├── play.py                  # visualization inference
│   └── collect_teacher_data.py  # Phase 2 data collection
├── student/
│   ├── __init__.py
│   ├── networks.py              # AdaptationModule + StudentPolicy
│   ├── train_student.py         # Phase 2 BC training
│   └── evaluate.py              # Sim2Sim validation
├── export/
│   ├── export_onnx.py           # Phase 3 ONNX export
│   └── benchmark.py             # inference latency test
├── results/                     # checkpoints, logs, tensorboard
├── training_log.md              # per-round iteration log
├── project_design.md            # symlink or copy of this doc
└── __init__.py
```

Key decisions:
- `config/` and `mdp/` = Phase 1 (framework-based)
- `student/` and `export/` = Phase 2/3 (custom code)
- `scripts/` = training entry points, calling rsl_rl runner
- TensorBoard logs auto-written to `results/`

---

## Phase 1: Teacher Training (rsl_rl + Isaac Lab)

### What Isaac Lab Provides (A mode — out-of-box)

| Component | Provided by |
|-----------|-------------|
| G1 robot model | `G1_MINIMAL_CFG` (USD asset, joint defs) |
| Environment | `LocomotionVelocityRoughEnvCfg` (obs/action/termination) |
| Reward library | `track_lin_vel_xy_exp`, `feet_air_time_positive_biped`, etc. |
| Domain Randomization | push_robot, base_mass, friction, external_force (events) |
| Terrain system | flat / rough / stairs, auto curriculum |
| PPO training | rsl_rl runner (asymmetric AC, privileged obs to critic) |
| GPU parallel | 1024-4096 envs out-of-box |
| Obs normalization | built-in |

### Our Customization

**Environment Config (flat, initial):**

| Config | Isaac Lab Default | Our Override |
|--------|------------------|-------------|
| num_envs | 4096 | 1024 (A4000 VRAM) |
| episode_length_s | 20s | 20s |
| command range vx | [0, 1.0] | [0.3, 0.6] → curriculum expand |
| reward weights | official defaults | iterative tuning |
| terrain | flat | flat first, rough later |

**Reward Design (initial, will iterate):**

| Reward Term | Source | Weight | Purpose |
|-------------|--------|--------|---------|
| track_lin_vel_xy_exp | framework | 1.5 | linear velocity tracking |
| track_ang_vel_z_exp | framework | 1.0 | yaw tracking |
| feet_air_time_positive_biped | framework | 0.5 | biped foot air time |
| feet_slide | framework | -0.1 | foot sliding penalty |
| flat_orientation_l2 | framework | -1.0 | stay upright |
| lin_vel_z_l2 | framework | -0.2 | suppress bouncing |
| action_rate_l2 | framework | -0.005 | action smoothness |
| dof_acc_l2 | framework | -1e-7 | joint acceleration |
| dof_torques_l2 | framework | -2e-6 | torque |
| joint_deviation_arms | framework | -0.1 | arms at default |
| joint_deviation_hip | framework | -0.1 | hip deviation |
| dof_pos_limits | framework | -1.0 | joint limits |
| termination_penalty | framework | -200.0 | fall penalty |

**PPO Config:**

| Parameter | Value |
|-----------|-------|
| actor_hidden_dims | [256, 128, 128] |
| critic_hidden_dims | [256, 128, 128] |
| num_steps_per_env | 24 |
| max_iterations | 1500 (flat) / 3000 (rough) |
| learning_rate | 1e-3, adaptive schedule |
| num_envs | 1024 |
| activation | elu |
| clip_param | 0.2 |
| entropy_coef | 0.008 |
| gamma | 0.99 |
| lam | 0.95 |

**Training Command:**

```bash
conda activate env_isaaclab
python scripts/train_teacher.py --task G1-Flat-v0 --num_envs 1024
```

**Estimated Training Time:** 1024 envs × 1500 iter on A4000 ≈ 20-40 minutes.

---

## Phase 2: Teacher-Student Distillation (Custom)

### Step 1: Data Collection

Use frozen teacher in DR environment for deterministic inference:

| Config | Value |
|--------|-------|
| Collection envs | 1024 envs, full DR |
| Collection steps | 500k transitions |
| Record | (obs_history, obs_current, teacher_action) |
| obs_history | last 50 steps × obs_dim, flattened |
| Storage | .npz (memory-mapped) |

### Step 2: BC Training

```
obs_history (50 × obs_dim)
       │
       ▼
┌─────────────────────────────────────┐
│  AdaptationModule                   │
│  MLP: (50*obs_dim) → 256 → 128 → 16│
└──────────────────┬──────────────────┘
                   │ latent z (16D)
obs_current        │
       │           │
       └─────┬─────┘
             ▼
┌─────────────────────────────────────────┐
│  StudentPolicy                          │
│  MLP: (obs_dim + 16) → 256 → 128 → action_dim │
└──────────────────┬──────────────────────┘
                   │ action_pred
                   ▼
         MSE loss vs teacher_action
```

| Parameter | Value |
|-----------|-------|
| latent_dim | 16 |
| student_lr | 1e-3 |
| epochs | 200 |
| batch_size | 256 |
| val_ratio | 0.1 |
| early_stop_patience | 15 |
| optimizer | Adam |

### Step 3: Sim2Sim Validation

Run student in clean environment (no DR), report:
- avg_reward vs teacher baseline
- survival rate (episodes lasting full 20s)
- tracking ratio (actual speed / command speed)

Pass criteria: reward degradation < 10%, survival > 95%.

---

## Phase 3: ONNX Export + Benchmark

**Export Architecture:**

```
┌───────────────────────────────────────┐
│  StudentONNXWrapper                   │
│  = AdaptationModule + StudentPolicy   │
│                                       │
│  inputs:  obs_history (50*obs_dim,)   │
│           obs_current (obs_dim,)      │
│  output:  action      (action_dim,)   │
└──────────────────┬────────────────────┘
                   │ torch.onnx.export (opset 17)
                   ▼
          student_g1.onnx
```

**Verification:**

| Metric | Requirement |
|--------|-------------|
| Accuracy | max\|PyTorch - ONNX\| < 1e-4 |
| Latency | avg < 20ms (50Hz control budget) |
| Model size | record, no hard limit |

**Benchmark Output:**

```
Model: student_g1.onnx
Size: X.X MB
Input: obs_history (50*obs_dim,) + obs_current (obs_dim,)
Output: action (action_dim,)
Accuracy: max_diff = X.XXe-X ✓
Latency (CPU, 1000 runs): avg=X.Xms, p95=X.Xms, p99=X.Xms ✓
```

---

## Comparison with Go2 Project

| Dimension | Go2 | G1 |
|-----------|-----|-----|
| Robot | Quadruped (12 DOF) | Humanoid (29 DOF) |
| Simulator | MuJoCo CPU | Isaac Lab GPU |
| Parallel scale | 128 envs | 1024 envs |
| Training time | 4-8 hours / round | 20-40 min / round |
| RL framework | Custom PPO | rsl_rl (industrial) |
| Environment | Custom Go2Env | Inherit Isaac Lab Manager-Based |
| Reward | Custom reward computer | Framework mdp functions + custom |
| DR | Custom DomainRandomizer | Framework events (config-based) |
| Teacher-Student | Custom | Custom (symmetric with Go2) |
| ONNX deploy | Custom | Custom |

**Portfolio Narrative:**

Go2 = full-stack from scratch, proves deep understanding of underlying algorithms.
G1 = industrial framework for efficient iteration, proves engineering delivery capability.
Both share Teacher-Student RMA deployment pipeline — one hand-written, one framework-based,
demonstrating adaptability across different engineering trade-offs.

---

## Iteration Workflow

```
1. Modify config/ reward weights or command range
2. Run train_teacher.py (20-40 min)
3. TensorBoard curves + play.py visualization
4. Diagnose issues, record in training_log.md
5. Repeat until flat-ground stable walking
6. Switch to rough terrain, continue iteration
7. Phase 2/3 distillation and deployment
```

---

## Training Log Format

Each round records (same depth as Go2 training_log.md):
- Config changes from previous round
- Convergence curve (TensorBoard screenshot or ASCII)
- Key metrics (reward, survival, tracking ratio)
- Phase analysis (what worked, what didn't)
- Diagnosis and next-round plan
- TensorBoard log path for detailed curves

---
---

# G1 Locomotion 项目设计文档

使用 Isaac Lab + rsl_rl 训练 Unitree G1 人形机器人 RL 运动控制，
包含自定义 Teacher-Student RMA 蒸馏和 ONNX 部署。

---

## 概览

| 项目 | 值 |
|------|---|
| 本体 | Unitree G1 (G1_MINIMAL, 29 DOF) |
| 仿真器 | Isaac Lab 2.3.0 + Isaac Sim 5.1.0，GPU 并行 |
| 硬件 | NVIDIA RTX A4000 16GB，12 核，128GB RAM |
| RL 框架 | rsl_rl 3.0.1（Phase 1） |
| 环境风格 | Manager-Based（继承 Isaac Lab G1 配置） |
| 位置 | easyRL 仓库 `applications/g1_locomotion/` |
| 训练目标 | 下半身行走（平地 → 复杂地形） |
| 流水线 | Phase 1 Teacher → Phase 2 Student RMA → Phase 3 ONNX |

---

## 项目结构

```
applications/g1_locomotion/
├── config/
│   ├── __init__.py
│   ├── flat_env_cfg.py          # 继承 Isaac Lab G1FlatEnvCfg，覆盖 reward/cmd
│   ├── rough_env_cfg.py         # 继承 G1RoughEnvCfg，地形 curriculum
│   └── ppo_cfg.py               # rsl_rl PPO runner 配置
├── mdp/
│   ├── __init__.py
│   └── rewards.py               # 自定义 reward terms（框架没有的）
├── scripts/
│   ├── train_teacher.py         # Phase 1 训练入口
│   ├── play.py                  # 可视化推理
│   └── collect_teacher_data.py  # Phase 2 数据采集
├── student/
│   ├── __init__.py
│   ├── networks.py              # AdaptationModule + StudentPolicy
│   ├── train_student.py         # Phase 2 BC 训练
│   └── evaluate.py              # Sim2Sim 验证
├── export/
│   ├── export_onnx.py           # Phase 3 ONNX 导出
│   └── benchmark.py             # 推理延迟测试
├── results/                     # checkpoint、日志、tensorboard
├── training_log.md              # 逐轮迭代记录
├── project_design.md            # 本文档的链接或副本
└── __init__.py
```

关键决策：
- `config/` 和 `mdp/` = Phase 1（基于框架的部分）
- `student/` 和 `export/` = Phase 2/3（自定义代码）
- `scripts/` = 训练入口，调用 rsl_rl runner
- TensorBoard 日志自动写入 `results/`

---

## Phase 1：Teacher 训练（rsl_rl + Isaac Lab）

### Isaac Lab 开箱提供的（A 模式）

| 组件 | 提供方 |
|------|--------|
| G1 机器人模型 | `G1_MINIMAL_CFG`（USD 资产，关节定义） |
| 环境封装 | `LocomotionVelocityRoughEnvCfg`（obs/action/termination 全套） |
| Reward 库 | `track_lin_vel_xy_exp`、`feet_air_time_positive_biped` 等现成 reward terms |
| Domain Randomization | push_robot、base_mass、friction、external_force（events 配置化） |
| 地形系统 | flat / rough / stairs，自动 curriculum |
| PPO 训练 | rsl_rl runner（含 asymmetric AC，privileged obs 给 critic） |
| GPU 并行 | 开箱 1024-4096 envs |
| Obs 归一化 | 内置 |

### 我们的定制

**环境配置（平地，初始版）：**

| 配置 | Isaac Lab 默认 | 我们的覆盖 |
|------|---------------|-----------|
| num_envs | 4096 | 1024（A4000 显存适配） |
| episode_length_s | 20s | 20s |
| command range vx | [0, 1.0] | [0.3, 0.6] → curriculum 扩展 |
| reward 权重 | 官方默认 | 多轮迭代调优 |
| 地形 | flat | 先 flat，后 rough |

**Reward 设计（初始版，会迭代）：**

| Reward 项 | 来源 | 权重 | 作用 |
|-----------|------|------|------|
| track_lin_vel_xy_exp | 框架 | 1.5 | 线速度跟踪 |
| track_ang_vel_z_exp | 框架 | 1.0 | 偏航跟踪 |
| feet_air_time_positive_biped | 框架 | 0.5 | 双足抬脚时间 |
| feet_slide | 框架 | -0.1 | 脚底滑动惩罚 |
| flat_orientation_l2 | 框架 | -1.0 | 保持直立 |
| lin_vel_z_l2 | 框架 | -0.2 | 抑制弹跳 |
| action_rate_l2 | 框架 | -0.005 | 动作平滑 |
| dof_acc_l2 | 框架 | -1e-7 | 关节加速度 |
| dof_torques_l2 | 框架 | -2e-6 | 力矩 |
| joint_deviation_arms | 框架 | -0.1 | 手臂保持默认 |
| joint_deviation_hip | 框架 | -0.1 | hip 偏差 |
| dof_pos_limits | 框架 | -1.0 | 关节极限 |
| termination_penalty | 框架 | -200.0 | 摔倒惩罚 |

**PPO 配置：**

| 参数 | 值 |
|------|---|
| actor_hidden_dims | [256, 128, 128] |
| critic_hidden_dims | [256, 128, 128] |
| num_steps_per_env | 24 |
| max_iterations | 1500（flat）/ 3000（rough） |
| learning_rate | 1e-3，adaptive schedule |
| num_envs | 1024 |
| activation | elu |
| clip_param | 0.2 |
| entropy_coef | 0.008 |
| gamma | 0.99 |
| lam | 0.95 |

**训练命令：**

```bash
conda activate env_isaaclab
python scripts/train_teacher.py --task G1-Flat-v0 --num_envs 1024
```

**预估训练时间：** 1024 envs × 1500 iter，A4000 上约 20-40 分钟。

---

## Phase 2：Teacher-Student 蒸馏（自定义）

### Step 1：数据采集

用 frozen teacher 在带 DR 的环境中确定性推理：

| 配置 | 值 |
|------|---|
| 采集环境 | 1024 envs，full DR |
| 采集步数 | 500k transitions |
| 记录内容 | (obs_history, obs_current, teacher_action) |
| obs_history | last 50 steps × obs_dim，flatten |
| 存储格式 | .npz（内存映射，避免 OOM） |

### Step 2：BC 训练

```
obs_history (50 × obs_dim)
       │
       ▼
┌─────────────────────────────────────┐
│  AdaptationModule                   │
│  MLP: (50*obs_dim) → 256 → 128 → 16│
└──────────────────┬──────────────────┘
                   │ latent z (16D)
obs_current        │
       │           │
       └─────┬─────┘
             ▼
┌─────────────────────────────────────────┐
│  StudentPolicy                          │
│  MLP: (obs_dim + 16) → 256 → 128 → action_dim │
└──────────────────┬──────────────────────┘
                   │ action_pred
                   ▼
         MSE loss vs teacher_action
```

| 参数 | 值 |
|------|---|
| latent_dim | 16 |
| student_lr | 1e-3 |
| epochs | 200 |
| batch_size | 256 |
| val_ratio | 0.1 |
| early_stop_patience | 15 |
| optimizer | Adam |

### Step 3：Sim2Sim 验证

在无 DR 的干净环境中运行 Student，报告：
- avg_reward vs teacher baseline
- 存活率（episode 能走满 20s 的比例）
- tracking ratio（实际速度 / 命令速度）

通过标准：reward 退化 < 10%，存活率 > 95%。

---

## Phase 3：ONNX 导出 + Benchmark

**导出架构：**

```
┌───────────────────────────────────────┐
│  StudentONNXWrapper                   │
│  = AdaptationModule + StudentPolicy   │
│                                       │
│  inputs:  obs_history (50*obs_dim,)   │
│           obs_current (obs_dim,)      │
│  output:  action      (action_dim,)   │
└──────────────────┬────────────────────┘
                   │ torch.onnx.export (opset 17)
                   ▼
          student_g1.onnx
```

**验证标准：**

| 指标 | 要求 |
|------|------|
| 精度 | max\|PyTorch - ONNX\| < 1e-4 |
| 延迟 | avg < 20ms（50Hz 控制频率预算） |
| 模型大小 | 记录，不设硬限 |

**Benchmark 输出：**

```
Model: student_g1.onnx
Size: X.X MB
Input: obs_history (50*obs_dim,) + obs_current (obs_dim,)
Output: action (action_dim,)
Accuracy: max_diff = X.XXe-X ✓
Latency (CPU, 1000 runs): avg=X.Xms, p95=X.Xms, p99=X.Xms ✓
```

---

## 与 Go2 项目的对比

| 维度 | Go2 | G1 |
|------|-----|-----|
| 本体 | 四足 (12 DOF) | 人形 (29 DOF) |
| 仿真 | MuJoCo CPU | Isaac Lab GPU |
| 并行规模 | 128 envs | 1024 envs |
| 训练时间 | 4-8 小时 / round | 20-40 分钟 / round |
| RL 框架 | 自实现 PPO | rsl_rl（工业级） |
| 环境 | 自写 Go2Env | 继承 Isaac Lab Manager-Based |
| Reward | 自写 reward computer | 框架 mdp functions + 自定义 |
| DR | 自写 DomainRandomizer | 框架 events（配置化） |
| Teacher-Student | 自己写 | 自己写（与 Go2 对称） |
| ONNX 部署 | 自己写 | 自己写 |

**Portfolio 叙事：**

Go2 = 从零实现全栈，证明对底层算法的深度理解。
G1 = 使用工业级框架高效迭代，证明工程落地能力。
两个项目共享 Teacher-Student RMA 部署路线 —— 一个手写一个基于框架，
展示在不同工程选型下的适应能力。

---

## 迭代工作流

```
1. 修改 config/ 中的 reward 权重或 command 范围
2. 运行 train_teacher.py（20-40 分钟）
3. TensorBoard 看曲线 + play.py 可视化
4. 诊断问题，记录到 training_log.md
5. 重复，直到平地稳定行走
6. 切换 rough terrain，继续迭代
7. Phase 2/3 蒸馏部署
```

---

## 训练日志格式

每轮记录（深度与 Go2 training_log.md 一致）：
- 相对上一轮的配置变更
- 收敛曲线（TensorBoard 截图或 ASCII）
- 关键指标（reward、存活率、tracking ratio）
- 阶段分析（什么有效、什么无效）
- 诊断和下一轮计划
- TensorBoard 日志路径（用于查看详细曲线）
