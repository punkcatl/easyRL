# DQN Training Flow: One Update Step Explained

## Overview

This document explains what happens inside `DQNAgent.update()` — a single training step. Each call samples a batch from the replay buffer, computes a loss, and updates the Q-network parameters.

**Example parameters used throughout this document** (from CartPole environment):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| batch_size | 64 | number of samples per training step |
| state_dim | 4 | CartPole state: position, velocity, pole angle, angular velocity |
| action_dim | 2 | CartPole actions: push left (0) or push right (1) |
| hidden_dim | 128 | neurons per hidden layer |
| tau | 0.005 | soft update rate (0.5% new + 99.5% old) |
| max_norm | 10.0 | gradient clipping threshold |

These are specific to our implementation. In general, replace 64 with `batch_size`, 4 with `state_dim`, 2 with `action_dim`, etc.

## Complete Data Flow

```
Step 1: SAMPLE from replay buffer
    buffer.sample(batch_size)
    -> states, actions, rewards, next_states, dones
         |
         v
Step 2: FORWARD PASS
    Q-network:
        states (batch_size, state_dim) -> q_net -> all_q (batch_size, action_dim)
        all_q.gather(actions) -> q_values (batch_size, 1)

    Target-network (no_grad):
        next_states -> target_net -> next_q (batch_size, action_dim)
        next_q.max(dim=1) -> max_next_q (batch_size, 1)

    TD target:
        targets = rewards + gamma * max_next_q * (1 - dones)
         |
         v
Step 3: COMPUTE LOSS
    loss = SmoothL1Loss(q_values, targets)
    batch_size samples -> averaged into one scalar
         |
         v
Step 4: BACKWARD PASS
    loss.backward()
    PyTorch walks backward through the computation graph:
    loss -> SmoothL1 -> q_values -> gather -> q_net output -> q_net params
    Result: each parameter's .grad is filled with its gradient
         |
         v
Step 5: GRADIENT CLIPPING
    clip_grad_norm_(params, max_norm=10.0)
    total_norm = sqrt(sum of all grad^2)
    If total_norm > max_norm: scale all grads by (max_norm / total_norm)
    Direction preserved, magnitude capped
         |
         v
Step 6: OPTIMIZER STEP
    optimizer.step()
    Adam reads each param's .grad, combines with:
      - momentum (smoothed gradient direction)
      - adaptive scale (per-parameter learning rate)
    Updates all 6 parameter tensors (3 layers x weight + bias)
         |
         v
Step 7: SOFT UPDATE TARGET NETWORK
    for each (param, target_param):
      target_param = tau * param + (1 - tau) * target_param
    Uses .data to bypass gradient tracking
```

## Key Concepts Explained

### Why Two Networks?

The Q-network is being updated every step — its output is a moving target. If we use the same network to both predict Q-values and compute TD targets, the target keeps shifting as we update, making training unstable (like chasing your own tail).

The target network provides a stable reference point. It changes slowly (tau=0.005 means only 0.5% per step via soft update), so the TD target stays relatively fixed during training.

### The Computation Graph

`loss` is not just a number. It carries a full record of how it was computed:

```
q_net.parameters
    -> Linear(state) -> ReLU -> Linear -> ReLU -> Linear
        -> all Q-values
            -> .gather(actions) -> q_values
                -> SmoothL1Loss(q_values, targets) -> loss
```

`backward()` traces this chain in reverse, applying the chain rule at each step to compute how much each parameter contributed to the loss. This is automatic differentiation — you never write gradient formulas manually.

### Why zero_grad() Before backward()?

PyTorch **accumulates** gradients by default. If you call `backward()` twice without clearing, the second round's gradients get **added** to the first round's. This is useful in some advanced scenarios (gradient accumulation across mini-batches), but in standard training you want fresh gradients each step.

### Adam vs Simple Gradient Descent

Simple SGD: `param = param - lr * grad`

Adam improves on this in two ways:

1. **Momentum (1st moment)**: Instead of using raw gradient, uses exponential moving average of past gradients. Smooths out noise, keeps consistent direction.

2. **Adaptive learning rate (2nd moment)**: Parameters with historically large gradients get smaller steps; parameters with small gradients get larger steps. Each parameter effectively has its own learning rate.

Result: faster convergence, less sensitive to learning rate choice.

### Why .data for Soft Update?

Every PyTorch tensor with `requires_grad=True` records operations for the backward pass. The soft update formula is a mathematical operation on parameters, but it's NOT part of the training loss computation. If recorded:
- It would pollute the gradient computation on the next `backward()` call
- Intermediate tensors would stay in memory waiting for a `backward()` that never comes

Using `.data` extracts the raw numbers without any tracking — a pure numerical assignment.

## Tensor Shapes at Each Stage

Using CartPole as example: batch_size=64, state_dim=4, action_dim=2.

| Variable | Shape | Why |
|----------|-------|-----|
| `states_t` | (batch_size, state_dim) | each sample is a state vector |
| `actions_t` | (batch_size, 1) | one action index per sample, unsqueezed for gather |
| `rewards_t` | (batch_size, 1) | one reward per sample, unsqueezed to match q_values |
| `dones_t` | (batch_size, 1) | one done flag per sample, unsqueezed to match q_values |
| `q_net(states_t)` | (batch_size, action_dim) | Q-value for every action |
| `q_values` | (batch_size, 1) | after gather: only the chosen action's Q-value |
| `max_next_q_values` | (batch_size, 1) | best Q-value at next state (from target net) |
| `targets` | (batch_size, 1) | TD target for each sample |
| `loss` | scalar | single number (mean over batch) |

## Network Parameters

Our Q-network has 3 linear layers (2 hidden + 1 output):

| Layer | Weight shape | Bias shape | Parameters |
|-------|-------------|-----------|------------|
| Linear 1 | (state_dim, hidden_dim) | (hidden_dim,) | 4x128 + 128 = 640 |
| Linear 2 | (hidden_dim, hidden_dim) | (hidden_dim,) | 128x128 + 128 = 16,512 |
| Linear 3 | (hidden_dim, action_dim) | (action_dim,) | 128x2 + 2 = 258 |
| **Total** | | | **17,410** |

All 6 tensors (3 weights + 3 biases) are updated by `optimizer.step()` each training step.

## What Gets Updated vs What Stays Fixed

| Component | Updated? | How |
|-----------|----------|-----|
| Q-network (q_net) | Yes, every step | optimizer.step() modifies weights via gradient descent |
| Target network (target_net) | Yes, every step (slowly) | soft update: tau% new + (1-tau)% old |
| Replay buffer | No (read-only during update) | only store_transition() adds to it |
| Optimizer internal state | Yes | Adam maintains momentum and scale for each parameter |
| epsilon | No (during update) | typically decayed in the training loop, not here |

---

# DQN 训练流程：一次更新步骤详解

## 概述

本文档解释 `DQNAgent.update()` 内部发生了什么——即一次训练步骤。每次调用从经验回放池中采样一个批次，计算损失，并更新 Q 网络参数。

**本文档使用的示例参数**（基于 CartPole 环境）：

| 参数 | 值 | 含义 |
|------|-----|------|
| batch_size | 64 | 每次训练采样数量 |
| state_dim | 4 | CartPole 状态：位置、速度、杆角度、角速度 |
| action_dim | 2 | CartPole 动作：向左推(0)或向右推(1) |
| hidden_dim | 128 | 每个隐藏层的神经元数 |
| tau | 0.005 | 软更新速率（0.5% 新参数 + 99.5% 旧参数） |
| max_norm | 10.0 | 梯度裁剪阈值 |

这些是我们代码中的具体配置。通用场景中，用 `batch_size` 替换 64，用 `state_dim` 替换 4，用 `action_dim` 替换 2，以此类推。

## 完整数据流

```
第1步：从经验回放池采样
    buffer.sample(batch_size)
    -> states, actions, rewards, next_states, dones
         |
         v
第2步：前向传播
    Q网络:
        states (batch_size, state_dim) -> q_net -> all_q (batch_size, action_dim)
        all_q.gather(actions) -> q_values (batch_size, 1)

    目标网络 (no_grad):
        next_states -> target_net -> next_q (batch_size, action_dim)
        next_q.max(dim=1) -> max_next_q (batch_size, 1)

    TD目标:
        targets = rewards + gamma * max_next_q * (1 - dones)
         |
         v
第3步：计算损失
    loss = SmoothL1Loss(q_values, targets)
    batch_size个样本 -> 平均成一个标量
         |
         v
第4步：反向传播
    loss.backward()
    PyTorch沿计算图反向走:
    loss -> SmoothL1 -> q_values -> gather -> q_net输出 -> q_net参数
    结果: 每个参数的.grad被填上对应的梯度值
         |
         v
第5步：梯度裁剪
    clip_grad_norm_(params, max_norm=10.0)
    total_norm = sqrt(所有grad的平方之和)
    如果 total_norm > max_norm: 所有grad乘以 (max_norm / total_norm)
    方向不变，大小被限制
         |
         v
第6步：优化器更新参数
    optimizer.step()
    Adam读取每个参数的.grad，结合:
      - 动量（平滑的梯度方向）
      - 自适应缩放（每个参数各自的学习率）
    更新全部6个参数张量（3层 x weight + bias）
         |
         v
第7步：软更新目标网络
    对每对 (param, target_param):
      target_param = tau * param + (1 - tau) * target_param
    使用.data绕过梯度追踪
```

## 核心概念详解

### 为什么需要两个网络？

Q 网络每步都在更新——它的输出是移动的靶子。如果用同一个网络既预测 Q 值又计算 TD 目标，目标会随着更新不断偏移，训练不稳定（好比追着自己的尾巴跑）。

目标网络提供一个稳定的参考点。它变化很慢（tau=0.005 即每步只混入 0.5% 的新参数），所以 TD 目标在训练过程中相对固定。

### 计算图

`loss` 不只是一个数字，它携带了一整棵计算记录：

```
q_net的参数
    -> Linear(state) -> ReLU -> Linear -> ReLU -> Linear
        -> 所有Q值
            -> .gather(actions) -> q_values
                -> SmoothL1Loss(q_values, targets) -> loss
```

`backward()` 沿这条链反向走，在每一步应用链式法则，计算每个参数对 loss 的贡献有多大。这就是自动微分——你不需要手动写梯度公式。

### 为什么 backward() 前要 zero_grad()？

PyTorch 默认**累加**梯度。如果不清零就调 `backward()`，这一轮的梯度会**加到**上一轮的梯度上。这在某些高级场景有用（跨小批次的梯度累积），但标准训练中每步都要新鲜的梯度。

### Adam vs 简单梯度下降

简单 SGD：`参数 = 参数 - 学习率 * 梯度`

Adam 在此基础上做了两点改进：

1. **动量（一阶矩）**：不直接用原始梯度，而是用历史梯度的指数移动平均。平滑噪声，保持方向一致性。

2. **自适应学习率（二阶矩）**：历史梯度一直很大的参数步子小一点，梯度一直很小的参数步子大一点。每个参数实际上有各自的学习率。

效果：收敛更快，对学习率的选择不那么敏感。

### 为什么软更新要用 .data？

每个 `requires_grad=True` 的 PyTorch 张量都会记录操作，为反向传播做准备。软更新公式是对参数的数学运算，但它**不是**训练损失计算的一部分。如果被记录：
- 下一次 `backward()` 时梯度会意外流过这条路径，污染 Q 网络的梯度计算
- 中间张量会留在内存中等待一个永远不会来的 `backward()`

使用 `.data` 提取纯数值，不带任何追踪——一次干净的数值赋值。

## 各阶段的张量形状

以 CartPole 为例：batch_size=64, state_dim=4, action_dim=2。

| 变量 | 形状 | 原因 |
|------|------|------|
| `states_t` | (batch_size, state_dim) | 每个样本是一个状态向量 |
| `actions_t` | (batch_size, 1) | 每个样本一个动作索引，unsqueeze后供gather使用 |
| `rewards_t` | (batch_size, 1) | 每个样本一个奖励，unsqueeze后和q_values对齐 |
| `dones_t` | (batch_size, 1) | 每个样本一个结束标志，unsqueeze后和q_values对齐 |
| `q_net(states_t)` | (batch_size, action_dim) | 每个动作的Q值 |
| `q_values` | (batch_size, 1) | gather后：只保留实际选择的动作的Q值 |
| `max_next_q_values` | (batch_size, 1) | 下一状态的最大Q值（来自目标网络） |
| `targets` | (batch_size, 1) | 每个样本的TD目标 |
| `loss` | 标量 | 单个数字（batch内所有样本的平均） |

## 网络参数

我们的 Q 网络有 3 个线性层（2 个隐藏层 + 1 个输出层）：

| 层 | 权重形状 | 偏置形状 | 参数数量 |
|----|---------|---------|---------|
| Linear 1 | (state_dim, hidden_dim) | (hidden_dim,) | 4x128 + 128 = 640 |
| Linear 2 | (hidden_dim, hidden_dim) | (hidden_dim,) | 128x128 + 128 = 16,512 |
| Linear 3 | (hidden_dim, action_dim) | (action_dim,) | 128x2 + 2 = 258 |
| **总计** | | | **17,410** |

全部 6 个张量（3 个权重 + 3 个偏置）每次训练步骤都被 `optimizer.step()` 更新。

## 什么被更新了 vs 什么保持不变

| 组件 | 是否更新？ | 方式 |
|------|-----------|------|
| Q网络 (q_net) | 是，每步 | optimizer.step() 通过梯度下降修改权重 |
| 目标网络 (target_net) | 是，每步（缓慢） | 软更新：tau% 新参数 + (1-tau)% 旧参数 |
| 经验回放池 | 否（update时只读） | 只有 store_transition() 往里加数据 |
| 优化器内部状态 | 是 | Adam 维护每个参数的动量和缩放系数 |
| epsilon | 否（update内不变） | 通常在训练循环中衰减，不在这里 |
