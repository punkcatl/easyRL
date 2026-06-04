# Reward Shaping Design Spec

## 1. Overview

A comprehensive reward shaping tutorial with comparison experiments. Covers 4 core methods (sparse vs dense, potential-based shaping, multi-objective weighted, reward hacking) across MuJoCo (Ant/Hopper/Humanoid) and highway-env. Includes theory documentation with interview-focused content.

## 2. Module Structure

```
applications/reward_shaping/
├── __init__.py
├── config.py                           ← experiment hyperparameters
├── rewards/
│   ├── __init__.py
│   ├── sparse.py                       ← sparse reward implementations
│   ├── dense.py                        ← dense reward implementations
│   ├── potential_based.py              ← potential function shaping
│   └── multi_objective.py              ← multi-objective weighted reward
├── hacking/
│   ├── __init__.py
│   ├── ant_rolling.py                  ← Ant rolling reproduction + fix
│   ├── hopper_jumping.py              ← Hopper high-jump reproduction + fix
│   ├── humanoid_sliding.py            ← Humanoid sliding reproduction + fix
│   ├── highway_lane_spam.py           ← lane-change spam reproduction + fix
│   └── highway_parking.py            ← parking (zero-speed) reproduction + fix
├── experiments/
│   ├── run_sparse_vs_dense.py          ← Experiment 1
│   ├── run_potential_shaping.py        ← Experiment 2
│   ├── run_multi_objective.py          ← Experiment 3 (weight sensitivity)
│   ├── run_hacking_cases.py            ← Experiment 4 (5 cases)
│   └── plot_comparison.py             ← unified plotting
├── results/                            ← training outputs + figures
└── docs/
    └── theory.md                       ← tutorial (theory + tuning + interview)
```

## 3. Experiment Matrix

```
                       MuJoCo (Ant/Hopper)        highway-env
Exp 1: Sparse vs Dense      ✓                        ✓
Exp 2: Potential-based       ✓                        ✓
Exp 3: Multi-objective       ✓                        ✓
Exp 4: Reward Hacking        ✓ (3 cases)              ✓ (2 cases)
```

## 4. Experiment 1: Sparse vs Dense

### 4.1 MuJoCo (Ant-v4)

- **Sparse:** reward = +1 only when x_position > 100, else 0
- **Dense:** reward = Δx per step (forward displacement increment)

### 4.2 highway-env

- **Sparse:** reward = +1 at destination, -1 on collision, else 0
- **Dense:** reward = speed / max_speed per step

### 4.3 Output

Training curves comparison showing convergence speed difference. Same PPO hyperparameters, only reward differs.

## 5. Experiment 2: Potential-based Shaping

### 5.1 Theory (Ng 1999)

Shaping reward that preserves optimal policy:

```
F(s, s') = γΦ(s') - Φ(s)
```

Where Φ(s) is an arbitrary potential function. The shaped reward is:

```
r_shaped = r_original + F(s, s')
```

Theorem: For any MDP M, the optimal policy under (r + F) is identical to the optimal policy under r, provided F has this form.

### 5.2 MuJoCo (Ant-v4)

- Φ(s) = x_position (higher potential closer to goal)
- shaped_reward = sparse_reward + γΦ(s') - Φ(s)
- Compare: sparse only vs sparse + shaping

### 5.3 highway-env

- Φ(s) = speed / max_speed (higher potential at higher speed)
- Compare: sparse only vs sparse + shaping

### 5.4 Output

- Show shaping accelerates convergence
- Show final policy is equivalent (same performance at convergence)
- Visualize the potential function landscape

## 6. Experiment 3: Multi-objective Weighted

### 6.1 MuJoCo (Ant-v4)

```python
reward = (w_speed * forward_velocity
        + w_alive * alive_bonus
        + w_energy * (-energy_cost)
        + w_posture * (-posture_penalty))
```

Default weights: w_speed=1.0, w_alive=0.5, w_energy=0.01, w_posture=0.1

### 6.2 highway-env

```python
reward = (w_speed * speed_reward
        + w_collision * collision_penalty
        + w_comfort * (-jerk_penalty)
        + w_lane * lane_keeping_reward)
```

Default weights: w_speed=1.0, w_collision=-10.0, w_comfort=0.1, w_lane=0.5

### 6.3 Weight Sensitivity Analysis

For each weight, sweep values while keeping others fixed:
- w_speed ∈ [0.1, 0.5, 1.0, 2.0, 5.0]
- w_collision ∈ [-1, -5, -10, -20, -50]
- etc.

### 6.4 Output

- Weight sensitivity heatmap (reward vs weight value)
- Pareto front visualization (speed vs safety trade-off)
- Recommended weight ranges per scenario

## 7. Experiment 4: Reward Hacking Cases

### 7.1 MuJoCo Cases

| Case | Broken Reward | Symptom | Fix |
|---|---|---|---|
| Ant rolling | `r = forward_velocity` | Rolls instead of walks | Add `posture_penalty = (z - z_target)²` |
| Hopper jumping | `r = large_alive_bonus + small_speed` | Jumps in place | Reduce alive_bonus, increase speed weight |
| Humanoid sliding | `r = forward_velocity` | Slides on belly | Add `min_height` constraint as penalty |

### 7.2 highway-env Cases

| Case | Broken Reward | Symptom | Fix |
|---|---|---|---|
| Lane-change spam | `r = ... + lane_change_bonus` | Oscillates left/right | Change to penalty or add cooldown |
| Zero-speed parking | `r = -10*collision + speed` | Stops moving (v=0) | Add `min_speed_penalty` term |

### 7.3 Per-Case Structure

Each case follows this pattern:
1. **Reproduce:** Train with broken reward, show the undesired behavior
2. **Analyze:** Explain why the agent exploits this reward
3. **Fix:** Modify reward design, retrain
4. **Compare:** Before/after training curves + behavior visualization

## 8. PPO Configuration

Use the same PPO agent across all experiments for fair comparison. Based on existing `algorithms/ppo/` with continuous action support.

Key hyperparameters (consistent across experiments):
```
lr: 3e-4
gamma: 0.99
gae_lambda: 0.95
clip_eps: 0.2
epochs: 10
hidden_dim: 128 (highway) / 256 (MuJoCo)
```

## 9. Tutorial Document (theory.md)

### Outline

1. **Why Reward Design is the Hardest Part of RL**
   - The reward hypothesis
   - Alignment problem in miniature

2. **Sparse vs Dense: Trade-off Analysis**
   - Exploration difficulty with sparse rewards
   - Signal-to-noise with dense rewards
   - When to choose which

3. **Potential-based Shaping Theory**
   - Ng 1999 theorem full proof
   - How to design good potential functions
   - Limitations and practical considerations

4. **Multi-objective: Industrial Practice**
   - Weight tuning methodology
   - Normalization across objectives
   - Pareto optimality concepts

5. **Reward Hacking: Patterns and Prevention**
   - Common failure modes (Goodhart's Law in RL)
   - Design principles to avoid hacking
   - Debugging checklist

6. **Interview FAQ**
   - High-frequency questions with reference answers
   - Common follow-ups and how to handle them

## 10. Dependencies

- `gymnasium[mujoco]`: MuJoCo environments
- `highway-env`: driving environments
- `torch`: PPO training
- `numpy`, `matplotlib`: data and plotting

## 11. Deliverables

1. Reward implementations (sparse, dense, potential, multi-objective)
2. 5 reward hacking reproductions with fixes
3. Training curves and comparison plots for all experiments
4. Weight sensitivity analysis and Pareto visualization
5. Bilingual theory + interview tutorial document

---

# Reward Shaping 设计规范

## 1. 概述

全面的 reward shaping 教程，包含对比实验。覆盖 4 个核心方法（稀疏 vs 密集、势函数 shaping、多目标加权、reward hacking），在 MuJoCo（Ant/Hopper/Humanoid）和 highway-env 上实验。附带面试导向的理论文档。

## 2. 模块结构

```
applications/reward_shaping/
├── __init__.py
├── config.py                           ← 实验超参数
├── rewards/
│   ├── __init__.py
│   ├── sparse.py                       ← 稀疏 reward 实现
│   ├── dense.py                        ← 密集 reward 实现
│   ├── potential_based.py              ← 势函数 shaping
│   └── multi_objective.py              ← 多目标加权
├── hacking/
│   ├── __init__.py
│   ├── ant_rolling.py                  ← Ant 翻滚复现 + 修复
│   ├── hopper_jumping.py              ← Hopper 高跳复现 + 修复
│   ├── humanoid_sliding.py            ← Humanoid 滑行复现 + 修复
│   ├── highway_lane_spam.py           ← 疯狂换道复现 + 修复
│   └── highway_parking.py            ← 停车不动复现 + 修复
├── experiments/
│   ├── run_sparse_vs_dense.py          ← 实验 1
│   ├── run_potential_shaping.py        ← 实验 2
│   ├── run_multi_objective.py          ← 实验 3 (权重敏感性)
│   ├── run_hacking_cases.py            ← 实验 4 (5 个案例)
│   └── plot_comparison.py             ← 统一绘图
├── results/                            ← 训练输出 + 图表
└── docs/
    └── theory.md                       ← 教程 (理论 + 调参 + 面试要点)
```

## 3. 实验矩阵

```
                       MuJoCo (Ant/Hopper)        highway-env
实验 1: Sparse vs Dense      ✓                        ✓
实验 2: Potential-based       ✓                        ✓
实验 3: Multi-objective       ✓                        ✓
实验 4: Reward Hacking        ✓ (3 案例)               ✓ (2 案例)
```

## 4. 实验 1：Sparse vs Dense

### 4.1 MuJoCo (Ant-v4)

- **Sparse:** 只在 x_position > 100 时给 +1，其余 0
- **Dense:** 每步给 Δx（前进位移增量）

### 4.2 highway-env

- **Sparse:** 到达终点 +1，碰撞 -1，其余 0
- **Dense:** 每步给 speed / max_speed

### 4.3 输出

训练曲线对比，展示收敛速度差异。相同 PPO 超参数，仅 reward 不同。

## 5. 实验 2：Potential-based Shaping

### 5.1 理论（Ng 1999）

保持最优策略不变的 shaping reward 形式：

```
F(s, s') = γΦ(s') - Φ(s)
```

Φ(s) 为任意势函数。Shaped reward：

```
r_shaped = r_original + F(s, s')
```

定理：对任意 MDP M，(r + F) 下的最优策略等同于 r 下的最优策略，只要 F 满足上述形式。

### 5.2 MuJoCo (Ant-v4)

- Φ(s) = x_position（离目标越近势越高）
- shaped_reward = sparse_reward + γΦ(s') - Φ(s)
- 对比：纯 sparse vs sparse + shaping

### 5.3 highway-env

- Φ(s) = speed / max_speed（越快势越高）
- 对比：纯 sparse vs sparse + shaping

### 5.4 输出

- 展示 shaping 加速收敛
- 展示最终策略等价（收敛后性能相同）
- 可视化势函数

## 6. 实验 3：Multi-objective Weighted

### 6.1 MuJoCo (Ant-v4)

```python
reward = (w_speed * forward_velocity
        + w_alive * alive_bonus
        + w_energy * (-energy_cost)
        + w_posture * (-posture_penalty))
```

默认权重：w_speed=1.0, w_alive=0.5, w_energy=0.01, w_posture=0.1

### 6.2 highway-env

```python
reward = (w_speed * speed_reward
        + w_collision * collision_penalty
        + w_comfort * (-jerk_penalty)
        + w_lane * lane_keeping_reward)
```

默认权重：w_speed=1.0, w_collision=-10.0, w_comfort=0.1, w_lane=0.5

### 6.3 权重敏感性分析

固定其他权重，逐个扫描：
- w_speed ∈ [0.1, 0.5, 1.0, 2.0, 5.0]
- w_collision ∈ [-1, -5, -10, -20, -50]
- 等等

### 6.4 输出

- 权重敏感性热力图
- Pareto 前沿可视化（速度 vs 安全 trade-off）
- 各场景推荐权重范围

## 7. 实验 4：Reward Hacking 案例

### 7.1 MuJoCo 案例

| 案例 | 有问题的 reward | 现象 | 修复方案 |
|---|---|---|---|
| Ant 翻滚 | `r = forward_velocity` | 翻滚比走路快 | 加 `posture_penalty = (z - z_target)²` |
| Hopper 高跳 | `r = large_alive + small_speed` | 原地跳 | 降 alive_bonus，升 speed 权重 |
| Humanoid 滑行 | `r = forward_velocity` | 趴着滑 | 加 `min_height` 约束惩罚 |

### 7.2 highway-env 案例

| 案例 | 有问题的 reward | 现象 | 修复方案 |
|---|---|---|---|
| 疯狂换道 | `r = ... + lane_change_bonus` | 左右抖动 | 改为惩罚或加冷却时间 |
| 停车不动 | `r = -10*collision + speed` | v=0 不动 | 加 `min_speed_penalty` |

### 7.3 每个案例的结构

1. **复现:** 用有问题的 reward 训练，展示异常行为
2. **分析:** 解释为什么 agent 会利用这个 reward
3. **修复:** 修改 reward 设计，重新训练
4. **对比:** 修复前后的训练曲线 + 行为可视化

## 8. PPO 配置

所有实验使用相同 PPO agent（公平对比），基于现有 `algorithms/ppo/`。

关键超参数（全实验一致）：
```
lr: 3e-4
gamma: 0.99
gae_lambda: 0.95
clip_eps: 0.2
epochs: 10
hidden_dim: 128 (highway) / 256 (MuJoCo)
```

## 9. 教程文档（theory.md）

### 大纲

1. **为什么 Reward Design 是 RL 最难的部分**
   - 奖励假设
   - 微型对齐问题

2. **Sparse vs Dense：Trade-off 分析**
   - 稀疏 reward 的探索困难
   - 密集 reward 的信噪比问题
   - 何时选择哪种

3. **Potential-based Shaping 理论推导**
   - Ng 1999 定理完整证明
   - 如何设计好的势函数
   - 局限性与实际考量

4. **Multi-objective：工业实践**
   - 权重调节方法论
   - 跨目标归一化
   - Pareto 最优概念

5. **Reward Hacking：模式与防范**
   - 常见失败模式（RL 中的 Goodhart 定律）
   - 避免 hacking 的设计原则
   - 调试清单

6. **面试高频问题 + 参考回答**

## 10. 依赖

- `gymnasium[mujoco]`：MuJoCo 环境
- `highway-env`：驾驶环境
- `torch`：PPO 训练
- `numpy`, `matplotlib`：数据与绘图

## 11. 交付物

1. Reward 实现（sparse、dense、potential、multi-objective）
2. 5 个 reward hacking 案例复现与修复
3. 全部实验的训练曲线与对比图
4. 权重敏感性分析与 Pareto 可视化
5. 双语理论 + 面试教程文档
