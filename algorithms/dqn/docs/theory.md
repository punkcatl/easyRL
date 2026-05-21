# Deep Q-Network (DQN)

## Intuition

DQN solves the problem of Q-Learning in large or continuous state spaces — you can't maintain a table with millions of entries. Instead, a neural network approximates the Q-function: given a state, it outputs Q-values for all actions. Two key tricks make training stable: (1) a replay buffer breaks correlation between consecutive samples, and (2) a target network provides a stable training target that doesn't shift with every update.

## Core Formula

**Loss function (Huber loss / Smooth L1):**

$$L(\theta) = \mathbb{E}\left[ \text{SmoothL1}\left( r + \gamma \max_{a'} Q(s', a'; \theta^{-}) - Q(s, a; \theta) \right) \right]$$

Where:
- $\theta$ — parameters of the Q-network (being trained)
- $\theta^{-}$ — parameters of the target network (updated via soft update from $\theta$)

**Target network soft update (every step):**

$$\theta^{-} \leftarrow \tau \theta + (1 - \tau) \theta^{-}$$

## Formula-to-Code Mapping

| Formula | Code location |
|---------|---------------|
| $Q(s, a; \theta)$ — Q-network forward pass | `agent.py:L10-L25` — `QNetwork` class |
| Experience replay buffer | `agent.py:L28-L49` — `ReplayBuffer` class |
| $\epsilon$-greedy with neural net | `agent.py:L83-L92` — `take_action` |
| $Q(s, a; \theta)$ for selected actions | `agent.py:L116` — `q_net(states_t).gather(1, actions_t)` |
| $\max_{a'} Q(s', a'; \theta^{-})$ target computation | `agent.py:L123` — `target_net(next_states_t).max(1, keepdim=True)[0]` |
| TD target: $r + \gamma \max Q' \cdot (1 - \text{done})$ | `agent.py:L125` — `rewards_t + self.gamma * max_next_q_values * (1 - dones_t)` |
| Smooth L1 loss $L(\theta)$ | `agent.py:L127` — `nn.SmoothL1Loss()(q_values, targets)` |
| Gradient descent step | `agent.py:L130-L133` — zero_grad, backward, clip_grad, step |
| Target network soft update $\theta^{-} \leftarrow \tau\theta + (1-\tau)\theta^{-}$ | `agent.py:L138-L142` — `target_param.data.copy_(tau * param + (1-tau) * target_param)` |

## Deep Dive (Optional)

**Why experience replay?**

Neural networks assume i.i.d. training data, but consecutive RL transitions are highly correlated (state $s_{t+1}$ is similar to $s_t$). Sampling random mini-batches from a large buffer breaks this correlation, leading to more stable gradients.

**Why a target network?**

Without it, both the prediction and the target shift simultaneously — like a dog chasing its own tail. A soft update ($\tau = 0.005$) lets $\theta^{-}$ track $\theta$ slowly, keeping the target stable enough for the Q-network to learn meaningful updates without the sudden jumps of a hard copy.

**Q-value overestimation in DQN:**

Standard DQN systematically overestimates Q-values. The mechanism:

1. The TD target is $y = r + \gamma \max_{a'} Q(s', a'; \theta^{-})$. The max operation over noisy estimates always picks the one that's overestimated by chance — like picking the "highest score" from a noisy exam always selects a lucky outlier.
2. The Q-network is trained to approximate $y$. Since $y$ includes an immediate reward $r$ plus the max-biased future estimate, the Q-network ends up **higher** than the target network's own outputs.
3. Soft update blends Q-network into target network: $\theta^{-} \leftarrow \tau\theta + (1-\tau)\theta^{-}$. Since Q > target, this **raises** the target network.
4. The raised target network feeds into the next round's max, producing an even higher $y$.
5. Positive feedback loop: overestimation accumulates over training.

Soft update makes the accumulation **smooth** (no oscillation), but does not prevent it. The only fix is to decouple action selection from value evaluation.

**Double DQN (the fix):**

Use the Q-network to select the best action, but the target network to evaluate it:

$$y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^{-})$$

The two networks have independent noise, so it's unlikely that both overestimate the same action simultaneously. This breaks the positive feedback loop.

The code difference is just one line — both DQN and Double DQN have two networks:

```python
# Standard DQN: target network selects AND evaluates (max does both)
max_next_q = self.target_net(next_states_t).max(1, keepdim=True)[0]

# Double DQN: Q-network selects, target network evaluates
best_actions = self.q_net(next_states_t).argmax(1, keepdim=True)
max_next_q = self.target_net(next_states_t).gather(1, best_actions)
```

In standard DQN, `.max()` binds selection and evaluation together — if the target network overestimates an action, it both picks it and gives it a high score (self-confirming). Double DQN breaks this by letting one network pick and the other score.

Not implemented in our code, but a common next step.

---

# 深度 Q 网络 (DQN)（中文版）

## 直觉

DQN 解决了 Q-Learning 在大规模或连续状态空间中的问题——你不可能维护一张几百万条目的表格。取而代之的是，用一个神经网络来逼近 Q 函数：输入状态，输出所有动作的 Q 值。两个关键技巧使训练稳定：（1）经验回放缓冲区打破了连续样本之间的相关性，（2）目标网络提供了一个不会随每次更新而变化的稳定训练目标。

## 核心公式

**损失函数（Huber 损失 / Smooth L1）：**

$$L(\theta) = \mathbb{E}\left[ \text{SmoothL1}\left( r + \gamma \max_{a'} Q(s', a'; \theta^{-}) - Q(s, a; \theta) \right) \right]$$

其中：
- $\theta$ — Q 网络的参数（正在训练）
- $\theta^{-}$ — 目标网络的参数（通过软更新从 $\theta$ 同步）

**目标网络软更新（每步）：**

$$\theta^{-} \leftarrow \tau \theta + (1 - \tau) \theta^{-}$$

## 公式与代码对应

| 公式 | 代码位置 |
|------|---------|
| $Q(s, a; \theta)$ — Q 网络前向传播 | `agent.py:L10-L25` — `QNetwork` 类 |
| 经验回放缓冲区 | `agent.py:L28-L49` — `ReplayBuffer` 类 |
| 带神经网络的 $\epsilon$-greedy | `agent.py:L83-L92` — `take_action` |
| 当前选中动作的 $Q(s, a; \theta)$ | `agent.py:L116` — `q_net(states_t).gather(1, actions_t)` |
| $\max_{a'} Q(s', a'; \theta^{-})$ 目标计算 | `agent.py:L123` — `target_net(next_states_t).max(1, keepdim=True)[0]` |
| TD 目标: $r + \gamma \max Q' \cdot (1 - \text{done})$ | `agent.py:L125` — `rewards_t + self.gamma * max_next_q_values * (1 - dones_t)` |
| Smooth L1 损失 $L(\theta)$ | `agent.py:L127` — `nn.SmoothL1Loss()(q_values, targets)` |
| 梯度下降步 | `agent.py:L130-L133` — zero_grad, backward, clip_grad, step |
| 目标网络软更新 $\theta^{-} \leftarrow \tau\theta + (1-\tau)\theta^{-}$ | `agent.py:L138-L142` — `target_param.data.copy_(tau * param + (1-tau) * target_param)` |

## 深入推导（选读）

**为什么需要经验回放？**

神经网络假设训练数据是独立同分布的，但连续的 RL 转移高度相关（$s_{t+1}$ 和 $s_t$ 很相似）。从大缓冲区中随机采样小批量数据打破了这种相关性，使梯度更稳定。

**为什么需要目标网络？**

没有它，预测值和目标值同时变化——就像狗追自己的尾巴。软更新（$\tau = 0.005$）让 $\theta^{-}$ 缓慢跟踪 $\theta$，使目标保持足够稳定，让 Q 网络能学到有意义的更新，同时避免硬拷贝带来的突变。

**DQN 的 Q 值高估缺陷：**

标准 DQN 会系统性地高估 Q 值。机制如下：

1. TD 目标是 $y = r + \gamma \max_{a'} Q(s', a'; \theta^{-})$。max 操作对有噪声的估计取最大值，总是选到碰巧被高估的那个——就像从一场有随机误差的考试中选"最高分"那道，大概率选到了运气好的异常值。
2. Q 网络被训练去逼近 $y$。由于 $y$ 包含即时奖励 $r$ 加上 max 偏高的未来估计，Q 网络最终会**高于**目标网络自身的输出值。
3. 软更新将 Q 网络混入目标网络：$\theta^{-} \leftarrow \tau\theta + (1-\tau)\theta^{-}$。由于 Q > target，这实际上**抬高**了目标网络。
4. 被抬高的目标网络进入下一轮 max 计算，产生更高的 $y$。
5. 正反馈循环：高估随训练不断累积。

软更新让这个累积过程**平滑**（不会震荡发散），但并不阻止高估。唯一的解决方案是将动作选择和价值评估解耦。

**Double DQN（解决方案）：**

用 Q 网络选择最优动作，但用目标网络评估其价值：

$$y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^{-})$$

两个网络有独立的噪声，不太可能同时高估同一个动作，从而打破正反馈循环。

代码层面只差一行——DQN 和 Double DQN 都有两个网络：

```python
# 标准 DQN：目标网络既选动作又评估（max 一步搞定）
max_next_q = self.target_net(next_states_t).max(1, keepdim=True)[0]

# Double DQN：Q 网络选动作，目标网络评估
best_actions = self.q_net(next_states_t).argmax(1, keepdim=True)
max_next_q = self.target_net(next_states_t).gather(1, best_actions)
```

标准 DQN 中 `.max()` 把选择和评估绑在一起——如果目标网络碰巧高估了某个动作，它既选中了它又给它打高分（自己确认自己）。Double DQN 让一个网络选、另一个网络打分，打破了这种自我确认。

我们的代码中没有实现，但这是常见的进阶方向。
