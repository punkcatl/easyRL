# Loss Functions & Target Network Update Strategies

## MSE Loss vs Smooth L1 Loss

### MSE Loss (Mean Squared Error)

$$L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

- Gradient: $\frac{\partial L}{\partial \hat{y}} = -2(y - \hat{y})$
- When error is large, gradient is proportionally large — can cause gradient explosion
- Penalizes large errors quadratically

### Smooth L1 Loss (Huber Loss)

$$L = \begin{cases} \frac{1}{2}(y - \hat{y})^2, & |y - \hat{y}| < 1 \\ |y - \hat{y}| - \frac{1}{2}, & \text{otherwise} \end{cases}$$

- When error < 1: behaves like MSE (smooth, good convergence near zero)
- When error >= 1: behaves like L1 (gradient capped at 1, prevents explosion)
- Combines the best of both worlds

### Comparison

| Property | MSE | Smooth L1 |
|----------|-----|-----------|
| Gradient at large error | Unbounded (proportional to error) | Bounded (constant = 1) |
| Gradient at small error | Smooth, → 0 as error → 0 | Smooth, → 0 as error → 0 |
| Sensitivity to outliers | High | Low |
| Training stability in RL | Poor (TD targets are noisy) | Good |
| Convergence precision | Good near optimum | Good near optimum |

**Why Smooth L1 is preferred in DQN:**

TD targets ($r + \gamma \max Q'$) are noisy estimates that fluctuate during training. MSE amplifies these large errors into huge gradients, destabilizing learning. Smooth L1 caps the gradient at 1 for large errors, acting as built-in gradient clipping at the loss level.

## Hard Update vs Soft Update

### Hard Update

$$\theta^{-} \leftarrow \theta \quad \text{(every } N \text{ steps)}$$

- Copy all parameters from Q-network to target network every N steps
- Between copies, target network is completely frozen
- Introduces a hyperparameter N (update frequency)

### Soft Update (Polyak Averaging)

$$\theta^{-} \leftarrow \tau \theta + (1 - \tau) \theta^{-} \quad \text{(every step)}$$

- Blend a small fraction ($\tau$, typically 0.001~0.01) of Q-network into target network each step
- Target network continuously tracks Q-network but with significant lag
- Introduces a hyperparameter $\tau$ (blending rate)

### Comparison

| Property | Hard Update | Soft Update |
|----------|-------------|-------------|
| Target stability | Stable between copies, sudden jump at copy | Always slowly changing, no jumps |
| Hyperparameter | N (update frequency) | $\tau$ (blending rate) |
| Sensitivity | Sensitive to N choice | Robust ($\tau$ in 0.001~0.01 works well) |
| Staleness | Target drifts stale before next copy | Always near-current |
| Implementation | `target.load_state_dict(q.state_dict())` | `p_tgt = τ * p + (1-τ) * p_tgt` |
| Origin | DQN (2015) | DDPG (2016), now standard |

### Behavior Over Time

```
Hard Update (N=100):

Target params: ████████████████░░░░░░░░░░░░░░░░████████████████
                ^copy          frozen            ^copy

Soft Update (τ=0.005):

Target params: ─────────────────────────────────────────────────
               (smooth, continuous change every step)
```

**Why soft update is generally preferred:**

1. No sudden target jumps → smoother loss landscape
2. One less hyperparameter to tune carefully
3. Target is never "stale" — always reflects recent learning
4. Works well across different environments without adjustment

---

# 损失函数与目标网络更新策略

## MSE 损失 vs Smooth L1 损失

### MSE 损失（均方误差）

$$L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

- 梯度：$\frac{\partial L}{\partial \hat{y}} = -2(y - \hat{y})$
- 误差大时，梯度也大——可能导致梯度爆炸
- 对大误差施加二次惩罚

### Smooth L1 损失（Huber 损失）

$$L = \begin{cases} \frac{1}{2}(y - \hat{y})^2, & |y - \hat{y}| < 1 \\ |y - \hat{y}| - \frac{1}{2}, & \text{otherwise} \end{cases}$$

- 误差 < 1 时：表现像 MSE（平滑，接近零时收敛好）
- 误差 >= 1 时：表现像 L1（梯度恒为 1，防止爆炸）
- 结合了两者的优点

### 对比

| 特性 | MSE | Smooth L1 |
|------|-----|-----------|
| 大误差时的梯度 | 无界（与误差成正比） | 有界（恒为 1） |
| 小误差时的梯度 | 平滑，误差→0 时梯度→0 | 平滑，误差→0 时梯度→0 |
| 对异常值的敏感性 | 高 | 低 |
| RL 中的训练稳定性 | 差（TD 目标有噪声） | 好 |
| 收敛精度 | 接近最优时好 | 接近最优时好 |

**为什么 DQN 中优先使用 Smooth L1：**

TD 目标（$r + \gamma \max Q'$）是有噪声的估计值，训练过程中波动大。MSE 会把这些大误差放大为巨大的梯度，破坏学习稳定性。Smooth L1 在大误差时将梯度限制为 1，相当于在损失函数层面内置了梯度裁剪。

## 硬更新 vs 软更新

### 硬更新

$$\theta^{-} \leftarrow \theta \quad \text{（每 } N \text{ 步）}$$

- 每隔 N 步，把 Q 网络的全部参数拷贝到目标网络
- 两次拷贝之间，目标网络完全冻结
- 引入超参数 N（更新频率）

### 软更新（Polyak 平均）

$$\theta^{-} \leftarrow \tau \theta + (1 - \tau) \theta^{-} \quad \text{（每步）}$$

- 每步将 Q 网络的一小部分（$\tau$，通常 0.001~0.01）混入目标网络
- 目标网络持续跟踪 Q 网络，但有显著滞后
- 引入超参数 $\tau$（混合比率）

### 对比

| 特性 | 硬更新 | 软更新 |
|------|--------|--------|
| 目标稳定性 | 两次拷贝间稳定，拷贝时突变 | 始终缓慢变化，无突变 |
| 超参数 | N（更新频率） | $\tau$（混合比率） |
| 敏感度 | 对 N 的选择敏感 | 鲁棒（0.001~0.01 范围内都好使） |
| 过时程度 | 拷贝前目标会变得过时 | 始终接近最新 |
| 实现方式 | `target.load_state_dict(q.state_dict())` | `p_tgt = τ * p + (1-τ) * p_tgt` |
| 起源 | DQN (2015) | DDPG (2016)，现为主流 |

### 随时间变化的行为

```
硬更新 (N=100):

目标参数: ████████████████░░░░░░░░░░░░░░░░████████████████
           ^拷贝          冻结期            ^拷贝

软更新 (τ=0.005):

目标参数: ─────────────────────────────────────────────────
          （平滑、连续，每步都变化）
```

**为什么通常优先使用软更新：**

1. 没有突然的目标跳变 → 更平滑的损失面
2. 少调一个敏感的超参数
3. 目标永远不会"过时"——始终反映最近的学习成果
4. 不同环境下无需调整即可良好工作
