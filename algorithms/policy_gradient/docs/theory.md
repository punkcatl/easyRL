# REINFORCE (Policy Gradient)

## Intuition

Instead of learning a value function and deriving a policy from it (like Q-Learning/DQN), REINFORCE directly optimizes the policy itself. The idea: if an action led to high total return, increase its probability; if it led to low return, decrease it. The policy is a neural network that outputs action probabilities, and we adjust its weights using gradient ascent on expected return.

## Core Formula

**Policy gradient theorem:**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

**Discounted return:**

$$G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$$

**Loss function (negative for gradient ascent):**

$$L = -\sum_{t=0}^{T} \log \pi_\theta(a_t|s_t) \cdot G_t$$

## Formula-to-Code Mapping

| Formula | Code location |
|---------|---------------|
| $\pi_\theta$ — policy network (state → action probs) | `agent.py:L8-L23` — `PolicyNetwork` with Softmax output |
| Sampling $a \sim \pi_\theta(\cdot|s)$ | `agent.py:L53-L54` — `Categorical(probs)` then `dist.sample()` |
| Storing $\log \pi_\theta(a_t|s_t)$ | `agent.py:L55` — `self.log_probs.append(dist.log_prob(action))` |
| Computing $G_t$ (discounted returns) | `agent.py:L66-L69` — reverse loop accumulating `G = r + gamma * G` |
| Normalizing returns | `agent.py:L74` — `(returns - mean) / (std + eps)` |
| Loss: $-\log\pi \cdot G_t$ | `agent.py:L79` — `loss += -log_prob * G` |
| Gradient update | `agent.py:L82-L84` — zero_grad, backward, step |

## Deep Dive (Optional)

**Why does this work? (Intuition behind the gradient)**

Consider $\nabla_\theta \log \pi_\theta(a|s) \cdot G$:
- $\nabla_\theta \log \pi_\theta(a|s)$ points in the direction that increases the probability of action $a$
- Multiplying by $G$ scales this direction: large positive $G$ → strongly increase probability; negative $G$ → decrease probability

This is the "likelihood ratio trick" — we can estimate the gradient of an expectation without differentiating through the environment dynamics.

**Why normalize returns?**

Raw returns can have large variance. Subtracting the mean creates a baseline effect: actions with above-average returns get reinforced, below-average get suppressed. This doesn't bias the gradient (zero-mean baseline) but significantly reduces variance.

**Limitations of REINFORCE:**

1. High variance — uses full episode returns, leading to noisy gradients
2. On-policy only — data from old policies cannot be reused
3. No value function — cannot bootstrap from partial episodes

PPO addresses all three of these issues.

---

# REINFORCE 策略梯度（中文版）

## 直觉

与 Q-Learning/DQN 学习价值函数再推导策略不同，REINFORCE 直接优化策略本身。核心思想：如果某个动作导致了高回报，就增加其概率；如果导致了低回报，就降低其概率。策略是一个输出动作概率的神经网络，我们用期望回报的梯度上升来调整其权重。

## 核心公式

**策略梯度定理：**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

**折扣回报：**

$$G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$$

**损失函数（取负号以实现梯度上升）：**

$$L = -\sum_{t=0}^{T} \log \pi_\theta(a_t|s_t) \cdot G_t$$

## 公式与代码对应

| 公式 | 代码位置 |
|------|---------|
| $\pi_\theta$ — 策略网络（状态 → 动作概率） | `agent.py:L8-L23` — 带 Softmax 输出的 `PolicyNetwork` |
| 采样 $a \sim \pi_\theta(\cdot|s)$ | `agent.py:L53-L54` — `Categorical(probs)` 然后 `dist.sample()` |
| 存储 $\log \pi_\theta(a_t|s_t)$ | `agent.py:L55` — `self.log_probs.append(dist.log_prob(action))` |
| 计算 $G_t$（折扣回报） | `agent.py:L66-L69` — 反向循环累加 `G = r + gamma * G` |
| 回报归一化 | `agent.py:L74` — `(returns - mean) / (std + eps)` |
| 损失: $-\log\pi \cdot G_t$ | `agent.py:L79` — `loss += -log_prob * G` |
| 梯度更新 | `agent.py:L82-L84` — zero_grad, backward, step |

## 深入推导（选读）

**为什么有效？（梯度背后的直觉）**

考虑 $\nabla_\theta \log \pi_\theta(a|s) \cdot G$：
- $\nabla_\theta \log \pi_\theta(a|s)$ 指向增加动作 $a$ 概率的方向
- 乘以 $G$ 缩放这个方向：大的正 $G$ → 强烈增加概率；负 $G$ → 降低概率

这就是"似然比技巧"——我们可以估计期望的梯度而无需对环境动力学求导。

**为什么要归一化回报？**

原始回报方差很大。减去均值产生基线效果：高于平均回报的动作被强化，低于平均的被抑制。这不会偏置梯度（零均值基线）但显著降低方差。

**REINFORCE 的局限性：**

1. 高方差——使用完整回合回报，导致梯度噪声大
2. 仅支持在策略（on-policy）——旧策略的数据不能重用
3. 没有价值函数——不能从部分回合中自举

PPO 解决了以上三个问题。
