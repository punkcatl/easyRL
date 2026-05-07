# Q-Learning

## Intuition

Q-Learning maintains a table that stores the "quality" of each state-action pair. At every step, the agent observes the outcome of its action and nudges the table entry toward a better estimate — specifically, toward the immediate reward plus the best future value it can get from the next state. Over many episodes, the table converges to the optimal values, and the agent simply picks the action with the highest Q-value.

## Core Formula

**Q-value update (TD learning):**

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

Where:
- $\alpha$ — learning rate
- $\gamma$ — discount factor
- $r$ — immediate reward
- $s'$ — next state
- $\max_{a'} Q(s', a')$ — best possible value from next state

**Epsilon-greedy action selection:**

$$a = \begin{cases} \text{random action} & \text{with probability } \epsilon \\ \arg\max_a Q(s, a) & \text{with probability } 1 - \epsilon \end{cases}$$

## Formula-to-Code Mapping

| Formula | Code location |
|---------|---------------|
| $Q(s,a)$ table initialization | `agent.py:L13` — `self.q_table = np.zeros((n_states, n_actions))` |
| $\epsilon$-greedy selection | `agent.py:L17-L20` — random vs argmax branch |
| TD target: $r + \gamma \max_{a'} Q(s', a')$ | `agent.py:L27` — `target = reward + self.gamma * np.max(self.q_table[next_state])` |
| Terminal state target: just $r$ | `agent.py:L25` — `target = reward` |
| Q-value update with learning rate $\alpha$ | `agent.py:L28` — `self.q_table[state, action] += self.lr * (target - self.q_table[state, action])` |

## Deep Dive (Optional)

**Why does this converge?**

Q-Learning is an off-policy TD(0) method. The key insight is the contraction property of the Bellman optimality operator:

$$\mathcal{T}^* Q(s,a) = \mathbb{E}\left[r + \gamma \max_{a'} Q(s', a')\right]$$

Under the Robbins-Monro conditions (every state-action visited infinitely often, learning rate decays appropriately), the Q-table converges to $Q^*$ — the optimal action-value function. The update at `agent.py:L28` is a stochastic approximation of applying this operator.

**Off-policy nature:**

The agent can follow any exploratory policy (epsilon-greedy) while learning about the optimal policy (greedy w.r.t. Q). This is because the update uses $\max_{a'}$ regardless of what action was actually taken next.

---

# Q-Learning（中文版）

## 直觉

Q-Learning 维护一个表格，记录每个"状态-动作"对的"质量值"。每一步，智能体观察到动作结果后，把表格中的对应条目往更准确的方向调整——具体来说，调整的目标是"即时奖励 + 下一个状态能获得的最佳未来价值"。经过足够多的回合，表格会收敛到最优值，智能体只需选择 Q 值最高的动作即可。

## 核心公式

**Q 值更新（TD 学习）：**

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

其中：
- $\alpha$ — 学习率
- $\gamma$ — 折扣因子
- $r$ — 即时奖励
- $s'$ — 下一个状态
- $\max_{a'} Q(s', a')$ — 下一个状态能获得的最大价值

**Epsilon-greedy 动作选择：**

$$a = \begin{cases} \text{随机动作} & \text{概率 } \epsilon \\ \arg\max_a Q(s, a) & \text{概率 } 1 - \epsilon \end{cases}$$

## 公式与代码对应

| 公式 | 代码位置 |
|------|---------|
| $Q(s,a)$ 表初始化 | `agent.py:L13` — `self.q_table = np.zeros((n_states, n_actions))` |
| $\epsilon$-greedy 选择 | `agent.py:L17-L20` — 随机 vs argmax 分支 |
| TD 目标: $r + \gamma \max_{a'} Q(s', a')$ | `agent.py:L27` — `target = reward + self.gamma * np.max(self.q_table[next_state])` |
| 终止状态目标: 仅 $r$ | `agent.py:L25` — `target = reward` |
| 带学习率 $\alpha$ 的 Q 值更新 | `agent.py:L28` — `self.q_table[state, action] += self.lr * (target - self.q_table[state, action])` |

## 深入推导（选读）

**为什么会收敛？**

Q-Learning 是一种离策略（off-policy）TD(0) 方法。关键在于贝尔曼最优算子的压缩性质：

$$\mathcal{T}^* Q(s,a) = \mathbb{E}\left[r + \gamma \max_{a'} Q(s', a')\right]$$

在 Robbins-Monro 条件下（每个状态-动作对被无限次访问、学习率适当衰减），Q 表收敛到 $Q^*$——最优动作价值函数。`agent.py:L28` 的更新就是对这个算子的随机近似。

**离策略的本质：**

智能体可以用任意探索策略（epsilon-greedy）收集数据，同时学习最优策略（关于 Q 的贪心策略）。这是因为更新使用了 $\max_{a'}$，与实际采取的下一个动作无关。
