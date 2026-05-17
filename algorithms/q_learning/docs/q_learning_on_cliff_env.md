# Q-Learning: Finding Optimal Path in Cliff Environment by Updating Q-Table

## 1. What is the Q-Table

A `num_states × num_actions` table. `Q(s, a)` represents "the total expected return if we take action a in state s and follow the optimal policy thereafter." Initialized to all zeros (knowing nothing).

## 2. How Each Step Updates

The agent takes action a0 in state s0, the environment returns reward r and next state s1, then update:

```
Q(s0, a0) ← Q(s0, a0) + α * [r + γ * max Q(s1, ·) - Q(s0, a0)]
```

Breaking it down:
- `r + γ * max Q(s1, ·)` — **TD Target**: actual reward received + best estimated value of next state
- `Q(s0, a0)` — current estimate
- The difference between them is the **TD Error** (how accurate the estimate was)
- `α` (learning rate) controls how much to correct each time

Intuition: Each step uses "real experience" to correct "previous guesses" — if overestimated, adjust down; if underestimated, adjust up.

## 3. Exploration vs. Exploitation (ε-greedy)

- With probability ε=0.1, **randomly select an action** (explore the unknown)
- With probability 1-ε=0.9, **select the action with highest Q-value** (exploit known knowledge)

Without exploration, the agent might forever follow the first path that happens to avoid the cliff, missing better paths.

## 4. Convergence Process

| Training Phase | Q-Table State | Agent Behavior |
|---------|---------|-----------|
| Early | All zeros, random walking | Frequently falls off cliff, return -100+ |
| Middle | Large negative values near cliff | Learns to avoid cliff |
| Late | Optimal path has highest Q-values | Stably follows shortest path |

The key mechanism is **value back-propagation**: cells near the goal learn good Q-values first, then these values gradually propagate to farther cells through `γ * max Q(s1, ·)`, spreading like a wave from the goal to the start.

## 5. Extracting the Optimal Path

After training, the Q-table converges. Taking `argmax Q(s, ·)` for each state gives the optimal action, chaining them together yields the optimal path:

```
Start(36) → Right → Right → ... → Right → Down → Goal(47)
```

## Summary

Q-Learning lets the agent learn by trial and error. Each step corrects the Q-table using "actual reward + best estimate of the future." After enough episodes, the Q-table converges, and selecting the highest Q-value action at each state gives the optimal policy.

---

# Q-Learning: 通过更新Q表得到悬崖环境下的最优路径

## 1. Q表是什么

一张 `状态数 × 动作数` 的表，`Q(s, a)` 表示"在状态 s 执行动作 a，之后一直按最优策略走，能拿到多少总回报"。初始全为 0（什么都不知道）。

## 2. 每一步怎么更新

agent 在状态 s0 执行动作 a0，环境返回奖励 r 和下一个状态 s1，然后更新：

```
Q(s0, a0) ← Q(s0, a0) + α * [r + γ * max Q(s1, ·) - Q(s0, a0)]
```

拆解来看：
- `r + γ * max Q(s1, ·)` — **TD目标**：实际拿到的奖励 + 下一个状态的最优估计价值
- `Q(s0, a0)` — 当前的估计
- 两者之差就是 **TD误差**（估得准不准）
- `α`（学习率）控制每次修正多少

直觉：每次走一步，用"真实经验"修正"之前的猜测"，猜高了就往下调，猜低了就往上调。

## 3. 探索与利用（ε-greedy）

- 以 ε=0.1 的概率**随机选动作**（探索未知）
- 以 1-ε=0.9 的概率**选Q值最大的动作**（利用已知）

如果不探索，agent 可能永远走第一条碰巧不掉崖的路径，错过更优路径。

## 4. 收敛过程

| 训练阶段 | Q表状态 | agent行为 |
|---------|---------|-----------|
| 初期 | 全是0，乱走 | 频繁掉悬崖，回报 -100+ |
| 中期 | 悬崖附近被写入大负值 | 学会避开悬崖 |
| 后期 | 最优路径的Q值最大 | 稳定走最短路径 |

关键机制是**反向传播价值**：终点旁边的格子先学到好的Q值，然后这个好的值通过 `γ * max Q(s1, ·)` 逐步传给更远的格子，像波浪一样从终点向起点扩散。

## 5. 读取最优路径

训练结束后，Q表收敛。对每个状态取 `argmax Q(s, ·)` 就是最优动作，串起来就是最优路径：

```
起点(36) → 右 → 右 → ... → 右 → 下 → 终点(47)
```

## 总结

Q-Learning 让 agent 反复试错，每一步用"实际奖励 + 对未来的最优估计"来修正Q表，经过足够多轮次后Q表收敛，每个状态选Q值最大的动作就是最优策略。
