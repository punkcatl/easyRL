# Sim-to-Real Pipeline: Curriculum DR + Teacher-Student + RMA

A complete sim-to-real transfer pipeline for MuJoCo locomotion tasks. The pipeline trains a Teacher policy with privileged information under progressively harder domain randomization, then distills it into a deployable Student via Rapid Motor Adaptation (RMA).

## Architecture

```
Phase 1: Teacher Training (PPO + Curriculum DR)
┌─────────────────────────────────────────────────────────┐
│  MuJoCo Env (randomized mass/friction/force/gain/delay) │
│                    ↓                                    │
│  obs + privileged_info → Teacher (PPO) → action         │
└─────────────────────────────────────────────────────────┘

Phase 2: Student Distillation (Behavior Cloning)
┌─────────────────────────────────────────────────────────┐
│  obs_history → Adaptation Module → latent z             │
│                                       ↓                 │
│                          obs + z → Base Policy → action │
│                                       ↓                 │
│                    MSE Loss vs Teacher action            │
└─────────────────────────────────────────────────────────┘

Deployment: Student only (no privileged info needed)
┌─────────────────────────────────────────────────────────┐
│  real obs_history → Adaptation Module → z               │
│         real obs + z → Base Policy → action             │
└─────────────────────────────────────────────────────────┘
```

## Requirements

```bash
pip install gymnasium[mujoco] torch numpy tqdm matplotlib
```

## Usage

### Phase 1: Train Teacher

```bash
# Train Teacher with Curriculum Domain Randomization
python train_teacher.py --env Ant-v4

# Train baseline without DR (for comparison)
python train_teacher.py --env Ant-v4 --no-dr

# Other supported envs
python train_teacher.py --env Hopper-v4
python train_teacher.py --env Humanoid-v4
```

### Phase 2: Train Student (RMA Distillation)

```bash
# Distill Teacher into Student via Behavior Cloning
python train_student.py --env Ant-v4
```

### Evaluation

```bash
# Compare Teacher vs Student vs Baseline across perturbation levels
python evaluate.py --env Ant-v4

# Evaluate with out-of-distribution parameters
python evaluate.py --env Ant-v4 --ood
```

### Supported Environments

| Environment | obs_dim | action_dim | Difficulty |
|---|---|---|---|
| `Ant-v4` | 27 | 8 | Medium |
| `Hopper-v4` | 11 | 3 | Low |
| `Humanoid-v4` | 376 | 17 | High |
| `Pusher-v4` | 23 | 7 | Low |

## Module Structure

```
sim_to_real/
├── __init__.py
├── config.py                    <- hyperparameters & env configs
├── train_teacher.py             <- Phase 1: PPO + Curriculum DR
├── train_student.py             <- Phase 2: BC distillation
├── evaluate.py                  <- comparison evaluation
├── envs/
│   ├── domain_randomization.py  <- CurriculumDR wrapper (mass/friction/force/gain/delay)
│   └── vectorized_env.py       <- vectorized env helper
├── agent/
│   ├── ppo_continuous.py        <- continuous PPO with GAE
│   ├── teacher.py               <- Teacher (PPO + privileged info)
│   └── student.py               <- Student (Adaptation Module + Base Policy)
└── results/                     <- saved models and curves
```

## Key Concepts

- **Curriculum Domain Randomization**: randomization ranges grow linearly from narrow (easy) to wide (hard) over the first 50% of training
- **Privileged Information**: Teacher observes friction, mass scale, external force, actuator gain, delay — unavailable in real deployment
- **RMA (Rapid Motor Adaptation)**: Student infers a latent vector z from observation history, implicitly performing online system identification
- **Behavior Cloning**: Student learns to imitate Teacher's actions given only proprioceptive history

## Configuration

All hyperparameters in `config.py`:

- **Domain Randomization**: initial/final ranges for mass, friction, force, gain, delay
- **Curriculum**: `curriculum_end_fraction` controls when ranges reach maximum
- **PPO**: lr, gamma, GAE lambda, clip_eps, batch_size, hidden_dim
- **Student**: history_length=50, latent_dim=16, distill_dataset_size=1M

---

# Sim-to-Real 管线：课程式域随机化 + Teacher-Student + RMA

一套完整的 MuJoCo 运动任务仿真到真实迁移管线。先用特权信息在渐进式域随机化下训练 Teacher 策略，再通过 RMA（快速运动自适应）蒸馏为可部署的 Student。

## 架构

```
阶段 1: Teacher 训练 (PPO + 课程式域随机化)
┌─────────────────────────────────────────────────────────┐
│  MuJoCo 环境 (随机化质量/摩擦/外力/增益/延迟)            │
│                    ↓                                    │
│  obs + 特权信息 → Teacher (PPO) → action                │
└─────────────────────────────────────────────────────────┘

阶段 2: Student 蒸馏 (行为克隆)
┌─────────────────────────────────────────────────────────┐
│  obs_history → 自适应模块 → 隐向量 z                     │
│                                 ↓                       │
│                    obs + z → 基础策略 → action           │
│                                 ↓                       │
│                  MSE Loss vs Teacher action              │
└─────────────────────────────────────────────────────────┘

部署: 仅需 Student (无需特权信息)
┌─────────────────────────────────────────────────────────┐
│  真实 obs_history → 自适应模块 → z                       │
│       真实 obs + z → 基础策略 → action                   │
└─────────────────────────────────────────────────────────┘
```

## 依赖安装

```bash
pip install gymnasium[mujoco] torch numpy tqdm matplotlib
```

## 使用方法

### 阶段 1: 训练 Teacher

```bash
# 使用课程式域随机化训练 Teacher
python train_teacher.py --env Ant-v4

# 训练无 DR 的基线（用于对比）
python train_teacher.py --env Ant-v4 --no-dr
```

### 阶段 2: 训练 Student (RMA 蒸馏)

```bash
# 通过行为克隆将 Teacher 蒸馏为 Student
python train_student.py --env Ant-v4
```

### 评估

```bash
# 对比 Teacher / Student / Baseline 在不同扰动下的表现
python evaluate.py --env Ant-v4

# 使用超出训练分布的参数评估
python evaluate.py --env Ant-v4 --ood
```

### 支持的环境

| 环境 | obs_dim | action_dim | 难度 |
|---|---|---|---|
| `Ant-v4` | 27 | 8 | 中 |
| `Hopper-v4` | 11 | 3 | 低 |
| `Humanoid-v4` | 376 | 17 | 高 |
| `Pusher-v4` | 23 | 7 | 低 |

## 核心概念

- **课程式域随机化**: 随机化范围在训练前 50% 内从窄（简单）线性增长到宽（困难）
- **特权信息**: Teacher 可观测摩擦系数、质量缩放、外力、执行器增益、延迟——真实部署时不可用
- **RMA (快速运动自适应)**: Student 从观测历史推断隐向量 z，隐式完成在线系统辨识
- **行为克隆**: Student 仅凭本体感知历史学习模仿 Teacher 的动作

## 配置

所有超参数集中在 `config.py`：

- **域随机化**: 质量、摩擦、外力、增益、延迟的初始/最终范围
- **课程**: `curriculum_end_fraction` 控制范围何时达到最大值
- **PPO**: lr, gamma, GAE lambda, clip_eps, batch_size, hidden_dim
- **Student**: history_length=50, latent_dim=16, distill_dataset_size=1M
