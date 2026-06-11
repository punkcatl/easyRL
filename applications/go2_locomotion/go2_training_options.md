# Go2 Locomotion Training: Two Paths

There are two ways to train a Go2 locomotion policy. Choose based on your goal.

## Quick Reference

| | easyRL go2_locomotion | unitree_rl_lab |
|--|--|--|
| Simulator | MuJoCo (CPU) | Isaac Lab (GPU) |
| Parallel envs | 32 | 4096+ |
| Training time | ~8 hours | ~15-30 minutes |
| GPU utilization | ~10% | ~90% |
| Purpose | Understand the algorithm | Get a working policy fast |
| Code style | Clean, minimal, self-implemented | Industrial-grade, production |
| Dependencies | MuJoCo, PyTorch, Gymnasium | Isaac Lab, rsl_rl, Isaac Sim |

## Path 1: easyRL go2_locomotion (this repo)

**Goal:** Understand how Go2 locomotion RL works from first principles.

The implementation in `applications/go2_locomotion/` is a clean educational re-implementation of the Unitree/RSL pipeline. Every component is written from scratch and easy to read:

```
applications/go2_locomotion/
├── envs/go2_env.py          # MuJoCo env, PD control, 48D obs
├── agent/networks.py        # AsymmetricActorCritic from scratch
├── agent/ppo.py             # PPO + GAE, ~100 lines
├── dr/domain_randomization.py
├── train_teacher.py         # Phase 1: PPO + DR
├── train_student.py         # Phase 2: RMA distillation
├── evaluate.py              # Sim2Sim evaluation
├── benchmark.py             # Motion Test Suite
└── export_onnx.py
```

**When to use this path:**
- Learning the algorithm (PPO, GAE, Asymmetric AC, RMA)
- Understanding why each design choice was made
- Debugging reward functions and observing behavior
- Interview preparation — you can explain every line

**Run:**
```bash
python applications/go2_locomotion/train_teacher.py
python applications/go2_locomotion/evaluate.py --mode teacher --render
```

## Path 2: unitree_rl_lab (external repo)

**Goal:** Train a high-quality Go2 policy quickly using the actual Unitree stack.

`unitree_rl_lab` is Unitree's official open-source training framework, built on Isaac Lab (NVIDIA's GPU-accelerated physics simulator). It runs 4096+ environments in parallel on GPU, completing training in minutes instead of hours.

**Repository:** `https://github.com/unitreerobotics/unitree_rl_lab`

**Supported robots:** Go2, G1, H1, H1-2

**Available tasks:**
```
Go2-Velocity-Flat-v0       # flat ground, same task as our easyRL version
Go2-Velocity-Rough-v0      # rough terrain with curriculum
G1-Velocity-Flat-v0        # G1 humanoid walking
H1-Velocity-Flat-v0        # H1 humanoid walking
```

**Install (requires NVIDIA GPU):**
```bash
# 1. Install Isaac Lab 2.3 (follow official Isaac Lab install guide)
# 2. Clone unitree_rl_lab
git clone https://github.com/unitreerobotics/unitree_rl_lab
cd unitree_rl_lab
pip install -e .

# 3. Train Go2 flat velocity tracking
python train.py --task Go2-Velocity-Flat-v0
```

**When to use this path:**
- You need a trained policy quickly
- You want to see what state-of-the-art performance looks like
- Benchmarking your easyRL implementation against the industrial baseline

## Relationship Between the Two

The easyRL implementation was designed by studying unitree_rl_lab and RSL's legged_gym. The core algorithms are identical:

```
Both use:
  PPO + GAE + clipped surrogate
  Asymmetric Actor-Critic (obs -> Actor, obs+privileged -> Critic)
  Domain Randomization (friction, mass, motor gains, external push)
  PD position control  target = 0.25 * action + default_angle
  48D observation space  (same fields)
  12D action space       (same joints)
  Teacher-Student RMA    (history -> latent -> action)

Difference:
  easyRL:          MuJoCo CPU, 32 envs,   hours
  unitree_rl_lab:  Isaac Lab GPU, 4096+ envs, minutes
```

The gap is entirely in simulation infrastructure, not algorithm design. Understanding easyRL means you understand unitree_rl_lab's core logic.

---

# Go2 运动控制训练：两条路线

训练 Go2 locomotion 策略有两种方式，根据目标选择。

## 快速对比

| | easyRL go2_locomotion | unitree_rl_lab |
|--|--|--|
| 仿真器 | MuJoCo（CPU） | Isaac Lab（GPU） |
| 并行环境数 | 32 | 4096+ |
| 训练时间 | ~8 小时 | ~15-30 分钟 |
| GPU 利用率 | ~10% | ~90% |
| 目的 | 理解算法原理 | 快速得到可用策略 |
| 代码风格 | 简洁、极简、自实现 | 工业级、生产级 |
| 依赖 | MuJoCo、PyTorch、Gymnasium | Isaac Lab、rsl_rl、Isaac Sim |

## 路线一：easyRL go2_locomotion（本仓库）

**目标：** 从第一原理理解 Go2 locomotion RL 的工作机制。

`applications/go2_locomotion/` 中的实现是 Unitree/RSL 完整管线的简洁教学复现版，每个组件都从零实现，代码易读：

```
applications/go2_locomotion/
├── envs/go2_env.py          # MuJoCo 环境，PD 控制，48D 观测
├── agent/networks.py        # AsymmetricActorCritic 自实现
├── agent/ppo.py             # PPO + GAE，约 100 行
├── dr/domain_randomization.py
├── train_teacher.py         # Phase 1: PPO + DR
├── train_student.py         # Phase 2: RMA 蒸馏
├── evaluate.py              # Sim2Sim 评估
├── benchmark.py             # Motion Test Suite
└── export_onnx.py
```

**适用场景：**
- 学习算法（PPO、GAE、Asymmetric AC、RMA）
- 理解每个设计决策背后的原因
- 调试 reward 函数并观察行为变化
- 面试准备——能解释每一行代码

**运行：**
```bash
python applications/go2_locomotion/train_teacher.py
python applications/go2_locomotion/evaluate.py --mode teacher --render
```

## 路线二：unitree_rl_lab（外部仓库）

**目标：** 用宇树官方技术栈快速训练高质量 Go2 策略。

`unitree_rl_lab` 是宇树官方开源的训练框架，基于 Isaac Lab（NVIDIA GPU 加速物理仿真器）。在 GPU 上并行运行 4096+ 个环境，训练时间从小时级压缩到分钟级。

**仓库地址：** `https://github.com/unitreerobotics/unitree_rl_lab`

**支持机器人：** Go2、G1、H1、H1-2

**可用任务：**
```
Go2-Velocity-Flat-v0       # 平地速度跟踪（与 easyRL 版本任务相同）
Go2-Velocity-Rough-v0      # 粗糙地形 + Curriculum
G1-Velocity-Flat-v0        # G1 人形行走
H1-Velocity-Flat-v0        # H1 人形行走
```

**安装（需要 NVIDIA GPU）：**
```bash
# 1. 安装 Isaac Lab 2.3（按官方 Isaac Lab 安装指引操作）
# 2. 克隆 unitree_rl_lab
git clone https://github.com/unitreerobotics/unitree_rl_lab
cd unitree_rl_lab
pip install -e .

# 3. 训练 Go2 平地速度跟踪
python train.py --task Go2-Velocity-Flat-v0
```

**适用场景：**
- 需要快速得到一个训练好的策略
- 想看工业级的训练效果
- 将 easyRL 实现与工业基线进行对比

## 两者的关系

easyRL 的实现参考了 unitree_rl_lab 和 RSL 的 legged_gym 设计，核心算法完全一致：

```
两者共同使用：
  PPO + GAE + clipped surrogate
  Asymmetric Actor-Critic（obs -> Actor，obs+privileged -> Critic）
  Domain Randomization（摩擦、质量、电机增益、外力推扰）
  PD 位置控制  target = 0.25 * action + default_angle
  48D 观测空间（字段完全相同）
  12D 动作空间（关节完全相同）
  Teacher-Student RMA（history -> latent -> action）

区别：
  easyRL:          MuJoCo CPU，32 envs，小时级
  unitree_rl_lab:  Isaac Lab GPU，4096+ envs，分钟级
```

差距完全在仿真基础设施层面，算法设计完全一致。理解了 easyRL，就理解了 unitree_rl_lab 的核心逻辑。
