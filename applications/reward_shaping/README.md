# Reward Shaping

A comprehensive reward shaping tutorial with comparison experiments. Covers 4 core methods across MuJoCo and highway-env environments.

## Module Structure

```
applications/reward_shaping/
├── __init__.py
├── config.py                    ← experiment hyperparameters
├── rewards/
│   ├── sparse.py                ← sparse reward wrapper
│   ├── dense.py                 ← dense reward wrapper
│   ├── potential_based.py       ← potential-based shaping (Ng 1999)
│   └── multi_objective.py       ← multi-objective weighted reward
├── hacking/
│   ├── ant_rolling.py           ← Ant rolling hack + fix
│   ├── hopper_jumping.py        ← Hopper jumping hack + fix
│   ├── humanoid_sliding.py      ← Humanoid sliding hack + fix
│   ├── highway_lane_spam.py     ← lane-change spam hack + fix
│   └── highway_parking.py       ← zero-speed parking hack + fix
├── experiments/
│   ├── run_sparse_vs_dense.py   ← Experiment 1
│   ├── run_potential_shaping.py ← Experiment 2
│   ├── run_multi_objective.py   ← Experiment 3
│   ├── run_hacking_cases.py     ← Experiment 4
│   └── plot_comparison.py       ← unified plotting
├── results/                     ← training outputs + figures
└── docs/
    └── theory.md                ← theory + interview tutorial
```

## Quick Start

### Dependencies

```bash
pip install gymnasium[mujoco] highway-env torch numpy matplotlib
```

### Run Experiments

```bash
# Experiment 1: Sparse vs Dense reward comparison
python experiments/run_sparse_vs_dense.py

# Experiment 2: Potential-based shaping
python experiments/run_potential_shaping.py

# Experiment 3: Multi-objective weight sensitivity
python experiments/run_multi_objective.py

# Experiment 4: Reward hacking cases (5 cases)
python experiments/run_hacking_cases.py

# Generate all comparison plots
python experiments/plot_comparison.py
```

### Use Reward Wrappers Independently

```python
import gymnasium as gym
from rewards.sparse import SparseRewardWrapper
from rewards.dense import DenseRewardWrapper
from rewards.potential_based import PotentialShapingWrapper, mujoco_x_potential
from rewards.multi_objective import MultiObjectiveRewardWrapper

# Sparse reward on Ant
env = gym.make("Ant-v4")
env = SparseRewardWrapper(env, env_type="mujoco", threshold=100.0)

# Dense reward on highway
env = gym.make("highway-v0")
env = DenseRewardWrapper(env, env_type="highway")

# Potential-based shaping (stackable on top of other wrappers)
env = gym.make("Ant-v4")
env = SparseRewardWrapper(env, env_type="mujoco")
env = PotentialShapingWrapper(env, mujoco_x_potential, gamma=0.99)

# Multi-objective with custom weights
env = gym.make("Ant-v4")
env = MultiObjectiveRewardWrapper(env, w_speed=2.0, w_alive=0.5, w_energy=0.01, w_posture=0.5)
```

## Experiments Overview

| Experiment | Question | Environments |
|------------|----------|--------------|
| 1. Sparse vs Dense | How does reward density affect convergence? | Ant-v4, highway-v0 |
| 2. Potential Shaping | Can shaping accelerate sparse reward learning? | Ant-v4, highway-v0 |
| 3. Multi-objective | How sensitive is behavior to weight choices? | Ant-v4 |
| 4. Reward Hacking | What happens when reward design is flawed? | Ant/Hopper/Humanoid, highway |

## PPO Configuration

All experiments use the same PPO agent (`algorithms/ppo/`) for fair comparison:

```
lr: 3e-4, gamma: 0.99, gae_lambda: 0.95, clip_eps: 0.2, epochs: 10
hidden_dim: 256 (MuJoCo) / 128 (highway)
```

---

# Reward Shaping

全面的 reward shaping 教程，包含对比实验。覆盖 4 个核心方法，在 MuJoCo 和 highway-env 环境上实验。

## 模块结构

```
applications/reward_shaping/
├── __init__.py
├── config.py                    ← 实验超参数
├── rewards/
│   ├── sparse.py                ← 稀疏奖励包装器
│   ├── dense.py                 ← 密集奖励包装器
│   ├── potential_based.py       ← 势函数 shaping（Ng 1999）
│   └── multi_objective.py       ← 多目标加权奖励
├── hacking/
│   ├── ant_rolling.py           ← Ant 翻滚 hack + 修复
│   ├── hopper_jumping.py        ← Hopper 高跳 hack + 修复
│   ├── humanoid_sliding.py      ← Humanoid 滑行 hack + 修复
│   ├── highway_lane_spam.py     ← 疯狂换道 hack + 修复
│   └── highway_parking.py       ← 零速停车 hack + 修复
├── experiments/
│   ├── run_sparse_vs_dense.py   ← 实验 1
│   ├── run_potential_shaping.py ← 实验 2
│   ├── run_multi_objective.py   ← 实验 3
│   ├── run_hacking_cases.py     ← 实验 4
│   └── plot_comparison.py       ← 统一绘图
├── results/                     ← 训练输出 + 图表
└── docs/
    └── theory.md                ← 理论 + 面试教程
```

## 快速开始

### 依赖

```bash
pip install gymnasium[mujoco] highway-env torch numpy matplotlib
```

### 运行实验

```bash
# 实验 1：稀疏 vs 密集奖励对比
python experiments/run_sparse_vs_dense.py

# 实验 2：势函数 shaping
python experiments/run_potential_shaping.py

# 实验 3：多目标权重敏感性
python experiments/run_multi_objective.py

# 实验 4：Reward hacking 案例（5 个）
python experiments/run_hacking_cases.py

# 生成所有对比图
python experiments/plot_comparison.py
```

### 独立使用 Reward 包装器

```python
import gymnasium as gym
from rewards.sparse import SparseRewardWrapper
from rewards.dense import DenseRewardWrapper
from rewards.potential_based import PotentialShapingWrapper, mujoco_x_potential
from rewards.multi_objective import MultiObjectiveRewardWrapper

# Ant 上的稀疏奖励
env = gym.make("Ant-v4")
env = SparseRewardWrapper(env, env_type="mujoco", threshold=100.0)

# highway 上的密集奖励
env = gym.make("highway-v0")
env = DenseRewardWrapper(env, env_type="highway")

# 势函数 shaping（可叠加在其他包装器之上）
env = gym.make("Ant-v4")
env = SparseRewardWrapper(env, env_type="mujoco")
env = PotentialShapingWrapper(env, mujoco_x_potential, gamma=0.99)

# 自定义权重的多目标
env = gym.make("Ant-v4")
env = MultiObjectiveRewardWrapper(env, w_speed=2.0, w_alive=0.5, w_energy=0.01, w_posture=0.5)
```

## 实验概览

| 实验 | 研究问题 | 环境 |
|------|----------|------|
| 1. Sparse vs Dense | 奖励密度如何影响收敛？ | Ant-v4, highway-v0 |
| 2. Potential Shaping | Shaping 能加速稀疏奖励学习吗？ | Ant-v4, highway-v0 |
| 3. Multi-objective | 行为对权重选择有多敏感？ | Ant-v4 |
| 4. Reward Hacking | 奖励设计有缺陷时会发生什么？ | Ant/Hopper/Humanoid, highway |

## PPO 配置

所有实验使用相同的 PPO agent（`algorithms/ppo/`）以确保公平对比：

```
lr: 3e-4, gamma: 0.99, gae_lambda: 0.95, clip_eps: 0.2, epochs: 10
hidden_dim: 256 (MuJoCo) / 128 (highway)
```
