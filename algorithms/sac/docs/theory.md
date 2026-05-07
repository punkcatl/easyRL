# Soft Actor-Critic (SAC)

## Intuition

SAC extends RL to continuous action spaces with a unique twist: it maximizes both expected return AND policy entropy. This means the agent seeks not just the highest-reward behavior, but the most *random* highest-reward behavior. Why? Because maintaining randomness prevents premature convergence, improves exploration, and makes the policy robust to perturbations. SAC uses twin Q-networks (to avoid overestimation), a separate policy network, and automatic entropy tuning.

## Core Formula

**Maximum entropy objective:**

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}\left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$

**Soft Bellman equation (Q-target):**

$$Q_{target}(s, a) = r + \gamma \left( \min_{i=1,2} Q_{\theta_i'}(s', a') - \alpha \log \pi(a'|s') \right)$$

**Policy loss (maximize Q while maximizing entropy):**

$$L_\pi = \mathbb{E}\left[ \alpha \log \pi(a|s) - \min_{i=1,2} Q_{\theta_i}(s, a) \right]$$

**Automatic entropy tuning:**

$$L_\alpha = -\alpha \cdot \mathbb{E}\left[ \log \pi(a|s) + \mathcal{H}_{target} \right]$$

**Soft target network update:**

$$\theta' \leftarrow \tau \theta + (1 - \tau) \theta'$$

## Formula-to-Code Mapping

| Formula | Code location |
|---------|---------------|
| Gaussian policy $\pi_\theta$ (mean + log_std) | `agent.py:L11-L40` — `GaussianPolicy` with tanh squashing |
| Reparameterization trick: $a = \tanh(\mu + \sigma \cdot \epsilon)$ | `agent.py:L35-L36` — `dist.rsample()` then `torch.tanh(x)` |
| Log prob with tanh correction | `agent.py:L38-L39` — subtracts $\log(1 - \tanh^2(x))$ |
| Twin Q-networks $Q_{\theta_1}, Q_{\theta_2}$ | `agent.py:L116-L117` — `self.q1`, `self.q2` |
| Target networks $Q_{\theta_1'}, Q_{\theta_2'}$ | `agent.py:L118-L119` — `self.q1_target`, `self.q2_target` |
| $\min Q$ for target (avoid overestimation) | `agent.py:L166` — `torch.min(q1_next, q2_next) - alpha * next_log_probs` |
| Soft Bellman target: $r + \gamma(1-d)(\min Q' - \alpha\log\pi)$ | `agent.py:L167` — `rewards_t + gamma * (1-dones_t) * q_next` |
| Q-network MSE loss | `agent.py:L170` — `MSELoss()(q1(s,a), q_target)` |
| Policy loss: $\alpha\log\pi - \min Q$ | `agent.py:L185` — `(self.alpha * log_probs - q_new).mean()` |
| Automatic $\alpha$ update | `agent.py:L193` — `-(log_alpha * (log_probs + target_entropy).detach()).mean()` |
| Target entropy $\mathcal{H}_{target} = -\dim(A)$ | `agent.py:L128` — `self.target_entropy = -action_dim` |
| Soft update: $\theta' \leftarrow \tau\theta + (1-\tau)\theta'$ | `agent.py:L200-L203` — `tau * param + (1-tau) * target_param` |

## Deep Dive (Optional)

**Why maximum entropy?**

Standard RL finds ONE optimal path. Maximum entropy RL finds ALL near-optimal paths and keeps them available. Benefits:
1. Better exploration — the policy doesn't collapse to a single trajectory
2. Robustness — if the environment changes slightly, alternative good actions are still available
3. Transfer — a diverse policy adapts faster to new tasks

**Why twin Q-networks?**

Single Q-network overestimates values (same problem as DQN). Using two networks and taking the minimum is a conservative estimate that prevents the policy from exploiting overestimation errors.

**Reparameterization trick:**

To backpropagate through the sampling process, SAC uses:
$$a = \tanh(\mu_\theta(s) + \sigma_\theta(s) \cdot \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)$$

The noise $\epsilon$ is external to the computation graph, so gradients flow through $\mu$ and $\sigma$.

**Tanh squashing and log-prob correction:**

Since $a = \tanh(x)$, the log-probability must be corrected by the change of variables formula:
$$\log \pi(a|s) = \log p(x|s) - \sum_i \log(1 - a_i^2)$$

This correction appears at `agent.py:L38`.

**Automatic entropy tuning:**

Instead of manually setting $\alpha$, SAC learns it by maintaining a target entropy $\mathcal{H}_{target} = -\dim(A)$. If the policy is too deterministic (entropy below target), $\alpha$ increases to encourage more randomness; if too random, $\alpha$ decreases.

---

# 软演员-评论家 SAC（中文版）

## 直觉

SAC 将 RL 扩展到连续动作空间，并有一个独特之处：它同时最大化期望回报和策略熵。这意味着智能体不仅寻找最高奖励的行为，还寻找最高奖励行为中最*随机*的那个。为什么？因为保持随机性可以防止过早收敛、改善探索、使策略对扰动更鲁棒。SAC 使用双 Q 网络（避免高估）、独立策略网络和自动熵调节。

## 核心公式

**最大熵目标：**

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}\left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$

**软贝尔曼方程（Q 目标）：**

$$Q_{target}(s, a) = r + \gamma \left( \min_{i=1,2} Q_{\theta_i'}(s', a') - \alpha \log \pi(a'|s') \right)$$

**策略损失（最大化 Q 同时最大化熵）：**

$$L_\pi = \mathbb{E}\left[ \alpha \log \pi(a|s) - \min_{i=1,2} Q_{\theta_i}(s, a) \right]$$

**自动熵调节：**

$$L_\alpha = -\alpha \cdot \mathbb{E}\left[ \log \pi(a|s) + \mathcal{H}_{target} \right]$$

**软目标网络更新：**

$$\theta' \leftarrow \tau \theta + (1 - \tau) \theta'$$

## 公式与代码对应

| 公式 | 代码位置 |
|------|---------|
| 高斯策略 $\pi_\theta$（均值 + log标准差） | `agent.py:L11-L40` — 带 tanh 压缩的 `GaussianPolicy` |
| 重参数化技巧: $a = \tanh(\mu + \sigma \cdot \epsilon)$ | `agent.py:L35-L36` — `dist.rsample()` 然后 `torch.tanh(x)` |
| 带 tanh 修正的 log prob | `agent.py:L38-L39` — 减去 $\log(1 - \tanh^2(x))$ |
| 双 Q 网络 $Q_{\theta_1}, Q_{\theta_2}$ | `agent.py:L116-L117` — `self.q1`, `self.q2` |
| 目标网络 $Q_{\theta_1'}, Q_{\theta_2'}$ | `agent.py:L118-L119` — `self.q1_target`, `self.q2_target` |
| 目标中取 $\min Q$（避免高估） | `agent.py:L166` — `torch.min(q1_next, q2_next) - alpha * next_log_probs` |
| 软贝尔曼目标: $r + \gamma(1-d)(\min Q' - \alpha\log\pi)$ | `agent.py:L167` — `rewards_t + gamma * (1-dones_t) * q_next` |
| Q 网络 MSE 损失 | `agent.py:L170` — `MSELoss()(q1(s,a), q_target)` |
| 策略损失: $\alpha\log\pi - \min Q$ | `agent.py:L185` — `(self.alpha * log_probs - q_new).mean()` |
| 自动 $\alpha$ 更新 | `agent.py:L193` — `-(log_alpha * (log_probs + target_entropy).detach()).mean()` |
| 目标熵 $\mathcal{H}_{target} = -\dim(A)$ | `agent.py:L128` — `self.target_entropy = -action_dim` |
| 软更新: $\theta' \leftarrow \tau\theta + (1-\tau)\theta'$ | `agent.py:L200-L203` — `tau * param + (1-tau) * target_param` |

## 深入推导（选读）

**为什么要最大熵？**

标准 RL 找到一条最优路径。最大熵 RL 找到所有近似最优路径并保持它们可用。好处：
1. 更好的探索——策略不会坍缩到单一轨迹
2. 鲁棒性——如果环境略有变化，替代的好动作仍然可用
3. 迁移——多样化策略更快适应新任务

**为什么用双 Q 网络？**

单个 Q 网络会高估价值（与 DQN 相同的问题）。使用两个网络取最小值是保守估计，防止策略利用高估误差。

**重参数化技巧：**

为了让梯度能通过采样过程反向传播，SAC 使用：
$$a = \tanh(\mu_\theta(s) + \sigma_\theta(s) \cdot \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)$$

噪声 $\epsilon$ 在计算图之外，所以梯度可以流过 $\mu$ 和 $\sigma$。

**Tanh 压缩与 log-prob 修正：**

由于 $a = \tanh(x)$，对数概率必须用换元公式修正：
$$\log \pi(a|s) = \log p(x|s) - \sum_i \log(1 - a_i^2)$$

这个修正出现在 `agent.py:L38`。

**自动熵调节：**

SAC 不手动设置 $\alpha$，而是通过维护目标熵 $\mathcal{H}_{target} = -\dim(A)$ 来学习它。如果策略太确定（熵低于目标），$\alpha$ 增大以鼓励更多随机性；如果太随机，$\alpha$ 减小。
