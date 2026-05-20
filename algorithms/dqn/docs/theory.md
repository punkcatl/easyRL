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
| $Q(s, a; \theta)$ — Q-network forward pass | `agent.py:L10-L24` — `QNetwork` class |
| Experience replay buffer | `agent.py:L27-L48` — `ReplayBuffer` class |
| $\epsilon$-greedy with neural net | `agent.py:L83-L89` — `select_action` |
| $Q(s, a; \theta)$ for selected actions | `agent.py:L107` — `q_net(states_t).gather(1, actions_t)` |
| $\max_{a'} Q(s', a'; \theta^{-})$ target computation | `agent.py:L111` — `target_net(next_states_t).max(1, keepdim=True)[0]` |
| TD target: $r + \gamma \max Q' \cdot (1 - \text{done})$ | `agent.py:L112` — `rewards_t + self.gamma * next_q_values * (1 - dones_t)` |
| Smooth L1 loss $L(\theta)$ | `agent.py:L114` — `nn.SmoothL1Loss()(q_values, targets)` |
| Gradient descent step | `agent.py:L116-L119` — zero_grad, backward, clip_grad, step |
| Target network soft update $\theta^{-} \leftarrow \tau\theta + (1-\tau)\theta^{-}$ | `agent.py:L122-L123` — `target_param.data.copy_(tau * param + (1-tau) * target_param)` |

## Deep Dive (Optional)

**Why experience replay?**

Neural networks assume i.i.d. training data, but consecutive RL transitions are highly correlated (state $s_{t+1}$ is similar to $s_t$). Sampling random mini-batches from a large buffer breaks this correlation, leading to more stable gradients.

**Why a target network?**

Without it, both the prediction and the target shift simultaneously — like a dog chasing its own tail. A soft update ($\tau = 0.005$) lets $\theta^{-}$ track $\theta$ slowly, keeping the target stable enough for the Q-network to learn meaningful updates without the sudden jumps of a hard copy.

**From DQN to Double DQN:**

Standard DQN overestimates Q-values because $\max$ is applied over noisy estimates. Double DQN decouples action selection from evaluation:

$$y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^{-})$$

This is not implemented in our code but is a common next step.

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
| $Q(s, a; \theta)$ — Q 网络前向传播 | `agent.py:L10-L24` — `QNetwork` 类 |
| 经验回放缓冲区 | `agent.py:L27-L48` — `ReplayBuffer` 类 |
| 带神经网络的 $\epsilon$-greedy | `agent.py:L83-L89` — `select_action` |
| 当前选中动作的 $Q(s, a; \theta)$ | `agent.py:L107` — `q_net(states_t).gather(1, actions_t)` |
| $\max_{a'} Q(s', a'; \theta^{-})$ 目标计算 | `agent.py:L111` — `target_net(next_states_t).max(1, keepdim=True)[0]` |
| TD 目标: $r + \gamma \max Q' \cdot (1 - \text{done})$ | `agent.py:L112` — `rewards_t + self.gamma * next_q_values * (1 - dones_t)` |
| Smooth L1 损失 $L(\theta)$ | `agent.py:L114` — `nn.SmoothL1Loss()(q_values, targets)` |
| 梯度下降步 | `agent.py:L116-L119` — zero_grad, backward, clip_grad, step |
| 目标网络软更新 $\theta^{-} \leftarrow \tau\theta + (1-\tau)\theta^{-}$ | `agent.py:L122-L123` — `target_param.data.copy_(tau * param + (1-tau) * target_param)` |

## 深入推导（选读）

**为什么需要经验回放？**

神经网络假设训练数据是独立同分布的，但连续的 RL 转移高度相关（$s_{t+1}$ 和 $s_t$ 很相似）。从大缓冲区中随机采样小批量数据打破了这种相关性，使梯度更稳定。

**为什么需要目标网络？**

没有它，预测值和目标值同时变化——就像狗追自己的尾巴。软更新（$\tau = 0.005$）让 $\theta^{-}$ 缓慢跟踪 $\theta$，使目标保持足够稳定，让 Q 网络能学到有意义的更新，同时避免硬拷贝带来的突变。

**从 DQN 到 Double DQN：**

标准 DQN 会高估 Q 值，因为 $\max$ 作用在有噪声的估计上。Double DQN 将动作选择和评估解耦：

$$y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^{-})$$

我们的代码中没有实现，但这是常见的进阶方向。
