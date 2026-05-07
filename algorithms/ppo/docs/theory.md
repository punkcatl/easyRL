# Proximal Policy Optimization (PPO)

## Intuition

PPO improves on REINFORCE in three ways: (1) it uses an Actor-Critic architecture — a value function reduces gradient variance by providing a baseline, (2) it reuses collected data for multiple gradient steps instead of discarding after one update, and (3) it clips the policy update to prevent catastrophically large changes. The result is a stable, sample-efficient algorithm that is the industry standard for most RL applications including autonomous driving.

## Core Formula

**Clipped surrogate objective:**

$$L^{CLIP}(\theta) = \mathbb{E}\left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

Where the probability ratio is:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

**Generalized Advantage Estimation (GAE):**

$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**Total loss:**

$$L = L^{CLIP}_{actor} + c_1 L^{VF}_{critic} - c_2 H[\pi_\theta]$$

## Formula-to-Code Mapping

| Formula | Code location |
|---------|---------------|
| Actor-Critic shared network | `agent.py:L8-L24` — `ActorCritic` with shared layers, separate actor/critic heads |
| Action sampling + log prob | `agent.py:L42-L46` — `Categorical` distribution sampling |
| TD error $\delta_t = r + \gamma V(s') - V(s)$ | `agent.py:L60` — `delta = rewards[t] + gamma * values_ext[t+1] * (1-dones[t]) - values_ext[t]` |
| GAE: $\hat{A}_t = \delta_t + \gamma\lambda\hat{A}_{t+1}$ | `agent.py:L61` — `gae = delta + gamma * gae_lambda * (1-dones[t]) * gae` |
| Returns = advantages + values | `agent.py:L65` — `returns_t = advantages_t + values_t` |
| Advantage normalization | `agent.py:L68` — `(advantages - mean) / (std + eps)` |
| Ratio $r_t(\theta) = \exp(\log\pi_{new} - \log\pi_{old})$ | `agent.py:L93` — `ratio = torch.exp(new_log_probs - batch_old_log_probs)` |
| Unclipped objective: $r_t \cdot \hat{A}$ | `agent.py:L96` — `surr1 = ratio * batch_advantages` |
| Clipped objective | `agent.py:L97` — `surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * batch_advantages` |
| Actor loss: $-\min(\text{surr1}, \text{surr2})$ | `agent.py:L98` — `-torch.min(surr1, surr2).mean()` |
| Critic loss: MSE on returns | `agent.py:L101` — `MSELoss()(new_values, batch_returns)` |
| Total loss with entropy bonus | `agent.py:L104` — `actor_loss + 0.5 * critic_loss - 0.01 * entropy` |
| Multiple epochs over same data | `agent.py:L73` — `for _ in range(self.epochs)` |

## Deep Dive (Optional)

**Why clip?**

Without clipping, a single large policy update can collapse performance irreversibly. The clip mechanism creates a "trust region" — if the new policy deviates too far from the old one (ratio $> 1+\epsilon$ or $< 1-\epsilon$), the gradient is zero in that direction. This guarantees monotonic improvement in practice.

**How does clipping work directionally?**

- If $\hat{A}_t > 0$ (good action): we want to increase $r_t$, but clip at $1+\epsilon$ prevents going too far
- If $\hat{A}_t < 0$ (bad action): we want to decrease $r_t$, but clip at $1-\epsilon$ prevents going too far

**Why GAE?**

GAE interpolates between:
- $\lambda = 0$: TD(0) advantage — low variance, high bias (only uses one-step lookahead)
- $\lambda = 1$: Monte Carlo advantage — high variance, low bias (uses full episode return)

$\lambda = 0.95$ (as in our code) gives a good bias-variance tradeoff.

**Why entropy bonus?**

The term $-c_2 H[\pi_\theta]$ encourages exploration by penalizing overly deterministic policies. Without it, the policy can converge prematurely to a suboptimal deterministic action.

---

# 近端策略优化 PPO（中文版）

## 直觉

PPO 在三个方面改进了 REINFORCE：（1）使用 Actor-Critic 架构——价值函数作为基线减少梯度方差，（2）重复利用收集的数据进行多次梯度更新而非一次后丢弃，（3）裁剪策略更新幅度防止灾难性的大变化。结果是一个稳定、样本高效的算法，是包括自动驾驶在内的大多数 RL 应用的工业标准。

## 核心公式

**裁剪替代目标函数：**

$$L^{CLIP}(\theta) = \mathbb{E}\left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中概率比为：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

**广义优势估计 (GAE)：**

$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**总损失：**

$$L = L^{CLIP}_{actor} + c_1 L^{VF}_{critic} - c_2 H[\pi_\theta]$$

## 公式与代码对应

| 公式 | 代码位置 |
|------|---------|
| Actor-Critic 共享网络 | `agent.py:L8-L24` — `ActorCritic` 共享层 + 分离的 actor/critic 头 |
| 动作采样 + log prob | `agent.py:L42-L46` — `Categorical` 分布采样 |
| TD 误差 $\delta_t = r + \gamma V(s') - V(s)$ | `agent.py:L60` — `delta = rewards[t] + gamma * values_ext[t+1] * (1-dones[t]) - values_ext[t]` |
| GAE: $\hat{A}_t = \delta_t + \gamma\lambda\hat{A}_{t+1}$ | `agent.py:L61` — `gae = delta + gamma * gae_lambda * (1-dones[t]) * gae` |
| 回报 = 优势 + 价值 | `agent.py:L65` — `returns_t = advantages_t + values_t` |
| 优势归一化 | `agent.py:L68` — `(advantages - mean) / (std + eps)` |
| 比率 $r_t(\theta) = \exp(\log\pi_{new} - \log\pi_{old})$ | `agent.py:L93` — `ratio = torch.exp(new_log_probs - batch_old_log_probs)` |
| 未裁剪目标: $r_t \cdot \hat{A}$ | `agent.py:L96` — `surr1 = ratio * batch_advantages` |
| 裁剪目标 | `agent.py:L97` — `surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * batch_advantages` |
| Actor 损失: $-\min(\text{surr1}, \text{surr2})$ | `agent.py:L98` — `-torch.min(surr1, surr2).mean()` |
| Critic 损失: 回报的 MSE | `agent.py:L101` — `MSELoss()(new_values, batch_returns)` |
| 带熵奖励的总损失 | `agent.py:L104` — `actor_loss + 0.5 * critic_loss - 0.01 * entropy` |
| 对同一批数据多次迭代 | `agent.py:L73` — `for _ in range(self.epochs)` |

## 深入推导（选读）

**为什么要裁剪？**

没有裁剪，一次过大的策略更新可能不可逆地崩溃性能。裁剪机制创建了一个"信任域"——如果新策略偏离旧策略太远（比率 $> 1+\epsilon$ 或 $< 1-\epsilon$），该方向的梯度为零。这在实践中保证了单调改进。

**裁剪的方向性：**

- 若 $\hat{A}_t > 0$（好动作）：我们想增大 $r_t$，但在 $1+\epsilon$ 处裁剪防止走太远
- 若 $\hat{A}_t < 0$（坏动作）：我们想减小 $r_t$，但在 $1-\epsilon$ 处裁剪防止走太远

**为什么用 GAE？**

GAE 在两者之间插值：
- $\lambda = 0$：TD(0) 优势——低方差，高偏差（仅用一步前瞻）
- $\lambda = 1$：蒙特卡洛优势——高方差，低偏差（用完整回合回报）

$\lambda = 0.95$（如代码中）提供了良好的偏差-方差折中。

**为什么加熵奖励？**

$-c_2 H[\pi_\theta]$ 项通过惩罚过于确定性的策略来鼓励探索。没有它，策略可能过早收敛到次优的确定性动作。
