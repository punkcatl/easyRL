# Reward Shaping: Theory, Practice, and Interview Guide

## 1. Why Reward Design is the Hardest Part of RL

The reward hypothesis states that all goals can be described as maximization of expected cumulative reward. In practice, translating human intent into a scalar signal is where most RL projects fail.

Key challenges:
- **Specification vs optimization gap**: The agent optimizes what you wrote, not what you meant
- **Sparse signal**: Correct behavior may only produce reward at episode end
- **Reward scale sensitivity**: PPO/SAC performance varies dramatically with reward magnitude
- **Multi-objective conflicts**: Safety, efficiency, and comfort cannot all be maximized simultaneously

This is the alignment problem in miniature -- the same fundamental difficulty that arises in AI safety research.

## 2. Sparse vs Dense Rewards

### Sparse Rewards

Definition: reward is non-zero only at specific events (goal reached, collision, timeout).

```
r(s, a) = +1  if s is goal state
         = 0   otherwise
```

Pros:
- Easy to specify correctly (less prone to hacking)
- Clear success criterion

Cons:
- Exploration is extremely difficult (reward signal is rare)
- Credit assignment over long horizons is hard
- May require millions of episodes to learn

### Dense Rewards

Definition: reward provides signal at every timestep.

```
r(s, a) = delta_x  (forward progress per step)
```

Pros:
- Fast convergence (gradient signal at every step)
- Works with shorter episodes

Cons:
- Prone to reward hacking (agent finds shortcuts)
- Harder to specify correctly
- May create local optima that trap the agent

### When to Use Which

| Scenario | Recommendation |
|----------|----------------|
| Simple task, short horizon | Dense |
| Complex task, human can specify progress | Dense with constraints |
| Hard to define intermediate progress | Sparse + exploration bonus |
| Safety-critical | Sparse (less hacking risk) |

## 3. Potential-based Reward Shaping

### The Theorem (Ng, Harada, Russell 1999)

For any MDP M with reward R, define a shaping reward:

```
F(s, s') = gamma * Phi(s') - Phi(s)
```

where Phi: S -> R is an arbitrary potential function.

**Theorem**: The optimal policy under R' = R + F is identical to the optimal policy under R.

### Proof Sketch

Consider the value function under shaped reward:

```
V'(s) = E[ sum_{t=0}^{inf} gamma^t * (R(s_t, a_t, s_{t+1}) + gamma*Phi(s_{t+1}) - Phi(s_t)) ]
       = E[ sum_{t=0}^{inf} gamma^t * R(s_t, a_t, s_{t+1}) ] + E[ sum_{t=0}^{inf} gamma^{t+1}*Phi(s_{t+1}) - gamma^t*Phi(s_t) ]
```

The second term telescopes:

```
= V(s) + E[ lim_{T->inf} gamma^{T+1}*Phi(s_{T+1}) - Phi(s_0) ]
= V(s) - Phi(s_0)    (assuming gamma < 1 and Phi bounded)
```

Since Phi(s_0) is constant for a given start state, argmax_pi V'(s) = argmax_pi V(s).

### Designing Good Potential Functions

Guidelines:
- Phi should be higher for states closer to the goal
- Phi should be smooth (avoid discontinuities that create artifacts)
- Phi should be cheap to compute (called every step)
- Domain knowledge helps: distance-to-goal, normalized speed, task progress

Examples:
```python
# MuJoCo locomotion: x-position as potential
Phi(s) = x_position

# Highway driving: speed as potential
Phi(s) = speed / max_speed

# Navigation: negative distance to goal
Phi(s) = -distance_to_goal
```

### Limitations

- Only guarantees optimal policy equivalence, not convergence speed
- Bad potential functions can slow learning (misleading gradient)
- Does not apply to non-stationary rewards or multi-agent settings without modification

## 4. Multi-objective Reward Design

### Weighted Sum Approach

```
R_total = w_1 * R_1 + w_2 * R_2 + ... + w_n * R_n
```

Each component captures one objective (speed, safety, energy, comfort).

### Weight Tuning Methodology

1. **Normalize each component** to similar scales (e.g., [0, 1] or [-1, 1])
2. **Start with equal weights**, observe behavior
3. **Adjust one weight at a time**, keep others fixed
4. **Log component values separately** to diagnose which objective dominates

### Normalization Strategies

| Method | Formula | When to use |
|--------|---------|-------------|
| Min-max | (r - r_min) / (r_max - r_min) | Known bounds |
| Running statistics | (r - mu) / sigma | Unknown bounds |
| Clipping | clip(r, -C, C) | Outlier protection |

### Pareto Optimality

When objectives conflict, no single solution is best on all axes. The Pareto front shows the set of non-dominated solutions.

In practice:
- Sweep weights to trace the Pareto front
- Present trade-off curves to stakeholders
- Choose operating point based on domain requirements

## 5. Reward Hacking: Patterns and Prevention

### What is Reward Hacking?

The agent finds an unintended strategy that achieves high reward without performing the desired task. This is Goodhart's Law applied to RL: "When a measure becomes a target, it ceases to be a good measure."

### Common Patterns

| Pattern | Example | Root Cause |
|---------|---------|------------|
| Shortcut | Ant rolls instead of walks | Missing posture constraint |
| Loophole | Agent stops to avoid penalty | Penalty dominates reward |
| Oscillation | Lane-change spam | Positive reward for transitions |
| Degenerate equilibrium | Zero-speed parking | Risk avoidance > task reward |
| Specification gaming | Jumping in place | alive_bonus >> speed_reward |

### Prevention Principles

1. **Constrain the solution space**: Add penalties for obviously wrong behavior
2. **Balance reward magnitudes**: No single term should dominate by >10x
3. **Reward outcomes, not actions**: Prefer goal-reaching over process rewards
4. **Test with adversarial thinking**: Ask "how would a literal optimizer exploit this?"
5. **Log behavior metrics separately**: Monitor speed, height, collisions independently

### Debugging Checklist

- [ ] Is any reward component >10x larger than others?
- [ ] Can the agent get reward without making task progress?
- [ ] Are there states where doing nothing yields positive reward?
- [ ] Does the agent's behavior match what you intended?
- [ ] Have you visualized the actual trajectories (not just the return curve)?

## 6. Interview FAQ

**Q: What is the difference between reward shaping and reward engineering?**

A: Reward engineering is the general practice of designing reward functions. Reward shaping specifically refers to adding an auxiliary reward F(s,s') to accelerate learning. Potential-based shaping (Ng 1999) is the only form guaranteed to preserve the optimal policy.

**Q: How do you handle sparse rewards in practice?**

A: Common approaches: (1) Potential-based shaping to add dense guidance without changing optimal policy, (2) Curiosity-driven exploration (ICM, RND), (3) Hindsight Experience Replay (HER) for goal-conditioned tasks, (4) Curriculum learning starting from easier variants.

**Q: What is the Ng 1999 theorem and why does it matter?**

A: It proves that shaping reward F(s,s') = gamma*Phi(s') - Phi(s) preserves the optimal policy for any potential function Phi. This matters because it lets you add dense reward signals to accelerate training without worrying about introducing suboptimal behavior.

**Q: How do you detect reward hacking?**

A: Monitor behavior metrics independently from reward. If return is high but task metrics (distance traveled, collision rate) are poor, the agent is hacking. Always visualize trajectories -- reward curves alone are insufficient.

**Q: How do you tune multi-objective reward weights?**

A: (1) Normalize all components to similar scales, (2) Start with equal weights, (3) Sweep one weight at a time while fixing others, (4) Log each component separately to see trade-offs. In industry, this often requires iterating with domain experts who understand acceptable trade-offs.

**Q: Can potential-based shaping hurt learning?**

A: Yes. While it preserves the optimal policy, a poorly chosen potential function can create misleading value landscapes that slow convergence. For example, if Phi increases away from the goal, the shaping signal fights the true reward.

**Q: What is the connection between reward hacking and AI alignment?**

A: Reward hacking in RL is a microcosm of the alignment problem. The agent optimizes the literal reward function, not human intent. This gap -- between what we specify and what we mean -- is the central challenge of building aligned AI systems.

---

# Reward Shaping：理论、实践与面试指南

## 1. 为什么 Reward Design 是 RL 最难的部分

奖励假设（Reward Hypothesis）声称所有目标都能表达为期望累积奖励的最大化。但在实践中，将人类意图转化为标量信号是多数 RL 项目失败的根源。

核心挑战：
- **规范与优化的鸿沟**：Agent 优化的是你写的东西，而非你想要的
- **稀疏信号**：正确行为可能仅在 episode 结束时才产生奖励
- **奖励尺度敏感性**：PPO/SAC 性能随奖励量级变化剧烈
- **多目标冲突**：安全、效率、舒适不可能同时最大化

这是对齐问题（Alignment Problem）的微缩版 -- 与 AI 安全研究中的核心难题本质相同。

## 2. Sparse vs Dense Rewards

### 稀疏奖励

定义：仅在特定事件时奖励非零（到达目标、碰撞、超时）。

```
r(s, a) = +1  if s 是目标状态
         = 0   其他情况
```

优点：
- 容易正确定义（不易被 hack）
- 成功准则明确

缺点：
- 探索极度困难（奖励信号稀少）
- 长视野信用分配困难
- 可能需要数百万 episode 才能学到

### 密集奖励

定义：每个 timestep 都提供信号。

```
r(s, a) = delta_x  (每步前进距离)
```

优点：
- 收敛快（每步都有梯度信号）
- 短 episode 也能工作

缺点：
- 容易被 reward hacking（agent 找捷径）
- 更难正确定义
- 可能产生困住 agent 的局部最优

### 选择指南

| 场景 | 建议 |
|------|------|
| 简单任务，短视野 | Dense |
| 复杂任务，能定义中间进度 | Dense + 约束 |
| 难以定义中间进度 | Sparse + 探索奖励 |
| 安全关键 | Sparse（hacking 风险低） |

## 3. Potential-based Reward Shaping

### 定理（Ng, Harada, Russell 1999）

对任意 MDP M（奖励为 R），定义 shaping reward：

```
F(s, s') = gamma * Phi(s') - Phi(s)
```

其中 Phi: S -> R 是任意势函数。

**定理**：R' = R + F 下的最优策略与 R 下的最优策略相同。

### 证明概要

考虑 shaped reward 下的价值函数：

```
V'(s) = E[ sum_{t=0}^{inf} gamma^t * (R(s_t, a_t, s_{t+1}) + gamma*Phi(s_{t+1}) - Phi(s_t)) ]
       = E[ sum_{t=0}^{inf} gamma^t * R(s_t, a_t, s_{t+1}) ] + E[ sum_{t=0}^{inf} gamma^{t+1}*Phi(s_{t+1}) - gamma^t*Phi(s_t) ]
```

第二项可以 telescope（逐项消去）：

```
= V(s) + E[ lim_{T->inf} gamma^{T+1}*Phi(s_{T+1}) - Phi(s_0) ]
= V(s) - Phi(s_0)    (假设 gamma < 1 且 Phi 有界)
```

由于 Phi(s_0) 对给定初始状态为常数，argmax_pi V'(s) = argmax_pi V(s)。

### 设计好的势函数

准则：
- Phi 对越接近目标的状态值越高
- Phi 应平滑（避免产生伪影的不连续性）
- Phi 应计算代价低（每步都要调用）
- 领域知识有帮助：到目标的距离、归一化速度、任务进度

示例：
```python
# MuJoCo 运动：x 坐标作为势
Phi(s) = x_position

# 高速公路驾驶：速度作为势
Phi(s) = speed / max_speed

# 导航：负的到目标距离
Phi(s) = -distance_to_goal
```

### 局限性

- 只保证最优策略等价，不保证收敛速度
- 糟糕的势函数可能减慢学习（误导梯度）
- 不适用于非平稳奖励或多智能体设定（需修改）

## 4. 多目标 Reward 设计

### 加权求和方法

```
R_total = w_1 * R_1 + w_2 * R_2 + ... + w_n * R_n
```

每个分量捕捉一个目标（速度、安全、能耗、舒适）。

### 权重调节方法论

1. **归一化各分量**至相近尺度（如 [0, 1] 或 [-1, 1]）
2. **从等权重开始**，观察行为
3. **每次只调一个权重**，固定其他
4. **分别记录各分量值**，诊断哪个目标占主导

### 归一化策略

| 方法 | 公式 | 适用场景 |
|------|------|----------|
| Min-max | (r - r_min) / (r_max - r_min) | 已知上下界 |
| 运行统计 | (r - mu) / sigma | 未知上下界 |
| 截断 | clip(r, -C, C) | 异常值保护 |

### Pareto 最优

当目标冲突时，不存在所有轴上都最优的单一解。Pareto 前沿展示非支配解的集合。

实践中：
- 扫描权重以描绘 Pareto 前沿
- 将 trade-off 曲线展示给利益相关者
- 根据领域需求选择工作点

## 5. Reward Hacking：模式与防范

### 什么是 Reward Hacking？

Agent 找到一种非预期策略，获得高奖励但未执行期望任务。这是 Goodhart 定律在 RL 中的应用："当一个度量变成目标时，它就不再是好的度量。"

### 常见模式

| 模式 | 示例 | 根因 |
|------|------|------|
| 走捷径 | Ant 翻滚而非行走 | 缺少姿态约束 |
| 钻空子 | Agent 停止以避免惩罚 | 惩罚项主导奖励 |
| 震荡 | 疯狂换道 | 对状态转换给正奖励 |
| 退化均衡 | 零速停车 | 风险规避 > 任务奖励 |
| 规范博弈 | 原地跳 | alive_bonus >> speed_reward |

### 防范原则

1. **约束解空间**：对明显错误行为加惩罚
2. **平衡奖励量级**：任何单项不应超过其他项 10 倍以上
3. **奖励结果而非动作**：优先使用到达目标的奖励
4. **对抗性思考**：问自己"一个字面量优化器会如何利用这个？"
5. **分别记录行为指标**：独立监控速度、高度、碰撞次数

### 调试清单

- [ ] 是否有某个奖励分量比其他大 10 倍以上？
- [ ] Agent 能否在不推进任务的情况下获得奖励？
- [ ] 是否存在什么都不做就能得到正奖励的状态？
- [ ] Agent 的行为是否符合你的意图？
- [ ] 你是否可视化了实际轨迹（而非仅看 return 曲线）？

## 6. 面试高频问题

**Q: Reward shaping 和 reward engineering 有什么区别？**

A: Reward engineering 是设计奖励函数的通用实践。Reward shaping 特指添加辅助奖励 F(s,s') 以加速学习。Potential-based shaping（Ng 1999）是唯一保证保持最优策略不变的形式。

**Q: 实践中如何处理稀疏奖励？**

A: 常见方法：(1) Potential-based shaping 添加密集引导而不改变最优策略，(2) 好奇心驱动探索（ICM, RND），(3) Hindsight Experience Replay（HER）用于目标条件任务，(4) 课程学习从简单变体开始。

**Q: Ng 1999 定理是什么，为什么重要？**

A: 它证明了 shaping reward F(s,s') = gamma*Phi(s') - Phi(s) 对任意势函数 Phi 都保持最优策略不变。重要性在于：可以添加密集奖励信号加速训练，无需担心引入次优行为。

**Q: 如何检测 reward hacking？**

A: 独立于奖励监控行为指标。如果 return 很高但任务指标（行进距离、碰撞率）很差，说明 agent 在 hacking。始终可视化轨迹 -- 仅看奖励曲线是不够的。

**Q: 如何调节多目标 reward 权重？**

A: (1) 将所有分量归一化到相近尺度，(2) 从等权重开始，(3) 固定其他权重逐个扫描，(4) 分别记录各分量以观察 trade-off。在工业界，这通常需要与领域专家反复迭代以确定可接受的 trade-off。

**Q: Potential-based shaping 会损害学习吗？**

A: 会。虽然它保持最优策略不变，但选择不好的势函数可能创建误导性的价值景观从而减慢收敛。例如，如果 Phi 随远离目标而增大，shaping 信号就会与真实奖励对抗。

**Q: Reward hacking 与 AI alignment 有什么联系？**

A: RL 中的 reward hacking 是对齐问题的微缩版。Agent 优化的是字面奖励函数，而非人类意图。我们定义的与我们想要的之间的鸿沟，正是构建对齐 AI 系统的核心挑战。
