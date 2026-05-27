# Policy Gradient Formula Summary

## Formula

$$
\nabla_\theta J(\theta)
=
E_{s \sim \nu^{\pi_\theta}}
\left[
E_{a \sim \pi_\theta(\cdot|s)}
\left[
Q^{\pi_\theta}(s,a)\nabla_\theta \ln \pi_\theta(a|s)
\right]
\right]
$$

## Core Meaning

This formula tells us how to change the policy parameters $\theta$ so that the expected return $J(\theta)$ increases.

At a high level, it says: under the current policy, increase the probability of actions that lead to higher long-term return, and decrease the probability of actions that lead to lower return.

## Meaning of Each Term

- $J(\theta)$: the objective of the policy, usually the expected return from the start state.
- $\nabla_\theta J(\theta)$: the gradient of the objective with respect to the policy parameters, namely the update direction.
- $\pi_\theta(a|s)$: the probability that the current policy selects action $a$ in state $s$.
- $\nu^{\pi_\theta}(s)$: the state visitation distribution induced by the current policy.
- $Q^{\pi_\theta}(s,a)$: the expected long-term return after taking action $a$ in state $s$ and then following the current policy.
- $\nabla_\theta \ln \pi_\theta(a|s)$: how the parameters should move if we want to change the probability of selecting action $a$ in state $s$.

## Inner and Outer Expectations

### Inner Expectation Over Actions

$$
E_{a \sim \pi_\theta(\cdot|s)}[f(s,a)] = \sum_a \pi_\theta(a|s) f(s,a)
$$

In this formula, the quantity being averaged is

$$
f(s,a)=Q^{\pi_\theta}(s,a)\nabla_\theta \ln \pi_\theta(a|s)
$$

This means that for a fixed state $s$, we average the gradient contribution of each possible action using the current policy's action probabilities as weights.

The point of weighted averaging is that actions are not equally likely. Actions the current policy chooses more often should contribute more to the overall update direction.

### Outer Expectation Over States

$$
E_{s \sim \nu^{\pi_\theta}}[g(s)] = \sum_s \nu^{\pi_\theta}(s) g(s)
$$

Here,

$$
g(s)=E_{a \sim \pi_\theta(\cdot|s)}\left[Q^{\pi_\theta}(s,a)\nabla_\theta \ln \pi_\theta(a|s)\right]
$$

This means we average over states according to how often the current policy actually visits them. Frequently visited states matter more than states that are rarely reached.

### Does the Outer Expectation Ignore Value Maximization?

No. The value information is already contained in

$$
g(s)=E_{a \sim \pi_\theta(\cdot|s)}\left[Q^{\pi_\theta}(s,a)\nabla_\theta \ln \pi_\theta(a|s)\right]
$$

The outer expectation does not average bare state probabilities. It averages these state-level gradient contributions, and each contribution already includes action values through $Q^{\pi_\theta}(s,a)$.

What the outer expectation adds is state importance weighting: states that are visited more often under the current policy should have more influence on the overall update.

The state visitation distribution is not optimized as an independent control variable. Instead, it changes indirectly when the policy changes. In other words, policy gradient directly adjusts action probabilities, and the resulting change in trajectories then changes which states are visited more often.

## Why Expectation and Gradient Appear Together

The objective itself is an expectation:

$$
J(\theta)=E[\text{return}]
$$

So its gradient must describe how to improve return in an average sense, not just for one sampled trajectory.

In reinforcement learning, states, actions, and returns are all random under the current policy and environment dynamics. A single term such as

$$
Q^{\pi_\theta}(s,a)\nabla_\theta \ln \pi_\theta(a|s)
$$

is only a sample-level gradient contribution. The expectation combines many such possible contributions into one overall update direction.

So the role of the gradient is to tell us how parameters should move, while the role of the expectation is to make that direction correspond to long-run average improvement rather than one noisy outcome.

## Why the Weighted Average Matters

The formula optimizes the expected return of the current policy in actual interaction with the environment. Because both states and actions occur with non-uniform probabilities, the average must respect those probabilities.

If we replaced the weighted average with a simple unweighted average, rare states or rare actions would influence the update as much as common ones, which would not match the real behavior distribution of the policy.

## Meaning of $\nabla_\theta \ln \pi_\theta(a|s)$

This term does not say whether the action is good or bad. Instead, it says how the parameters should move if we want to increase or decrease the probability of action $a$ in state $s$.

- $Q^{\pi_\theta}(s,a)$ answers: is this action worth encouraging?
- $\nabla_\theta \ln \pi_\theta(a|s)$ answers: if we want to encourage or suppress it, how should the parameters move?

Their product therefore means: use the action value to scale the parameter-change direction that affects the probability of that action.

## Log-Derivative Identity

A key identity used in the derivation is

$$
\nabla_\theta \pi_\theta(a|s)=\pi_\theta(a|s)\nabla_\theta \ln \pi_\theta(a|s)
$$

It follows from the chain rule:

$$
\nabla_\theta \ln \pi_\theta(a|s)=\frac{1}{\pi_\theta(a|s)}\nabla_\theta \pi_\theta(a|s)
$$

Multiplying both sides by $\pi_\theta(a|s)$ yields the identity above.

This identity lets the gradient be rewritten into an expectation form, which is why it can be estimated from samples.

## Relationship to REINFORCE

In theory, the formula uses $Q^{\pi_\theta}(s,a)$. In REINFORCE, this is replaced by a Monte Carlo return:

$$
G_t = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}
$$

So the sampled policy gradient becomes

$$
\nabla_\theta J(\theta) \approx \sum_t G_t \nabla_\theta \ln \pi_\theta(a_t|s_t)
$$

That is the practical bridge from the theorem to code.

## One-Sentence Summary

The policy gradient formula says: under the current policy's real state and action distribution, increase the probability of actions with higher long-term return by updating the parameters in the direction that makes those actions more likely.

---

# 策略梯度公式总结

## 公式

$$
\nabla_\theta J(\theta)
=
E_{s \sim \nu^{\pi_\theta}}
\left[
E_{a \sim \pi_\theta(\cdot|s)}
\left[
Q^{\pi_\theta}(s,a)\nabla_\theta \ln \pi_\theta(a|s)
\right]
\right]
$$

## 核心含义

这个公式告诉我们：为了让策略的期望回报 $J(\theta)$ 变大，策略参数 $\theta$ 应该朝哪个方向更新。

从直觉上看，它表达的是：在当前策略下，让长期回报更高的动作概率变大，让长期回报更低的动作概率变小。

## 各项含义

- $J(\theta)$：策略的优化目标，通常是从初始状态出发的期望回报。
- $\nabla_\theta J(\theta)$：目标函数对策略参数的梯度，也就是参数更新方向。
- $\pi_\theta(a|s)$：当前策略在状态 $s$ 下选择动作 $a$ 的概率。
- $\nu^{\pi_\theta}(s)$：当前策略诱导出的状态访问分布。
- $Q^{\pi_\theta}(s,a)$：在状态 $s$ 下执行动作 $a$ 后，再继续按照当前策略行动时的长期期望回报。
- $\nabla_\theta \ln \pi_\theta(a|s)$：如果想改变状态 $s$ 下选择动作 $a$ 的概率，参数应该如何调整。

## 内层与外层期望

### 内层动作期望

$$
E_{a \sim \pi_\theta(\cdot|s)}[f(s,a)] = \sum_a \pi_\theta(a|s) f(s,a)
$$

在这个公式里，被平均的量是

$$
f(s,a)=Q^{\pi_\theta}(s,a)\nabla_\theta \ln \pi_\theta(a|s)
$$

这表示：在固定状态 $s$ 下，把每个动作对梯度的贡献按当前策略的动作概率做加权平均。

之所以要做加权平均，是因为动作并不是等概率出现的。当前策略更常选择的动作，应该对整体更新方向有更大的影响。

### 外层状态期望

$$
E_{s \sim \nu^{\pi_\theta}}[g(s)] = \sum_s \nu^{\pi_\theta}(s) g(s)
$$

其中，

$$
g(s)=E_{a \sim \pi_\theta(\cdot|s)}\left[Q^{\pi_\theta}(s,a)\nabla_\theta \ln \pi_\theta(a|s)\right]
$$

这表示：按照当前策略实际访问各个状态的频率，对不同状态下的平均梯度贡献再做一次加权平均。经常访问到的状态比很少访问到的状态更重要。

### 外层期望会忽略价值最大化吗？

不会。价值信息已经包含在

$$
g(s)=E_{a \sim \pi_\theta(\cdot|s)}\left[Q^{\pi_\theta}(s,a)\nabla_\theta \ln \pi_\theta(a|s)\right]
$$

里面了。

外层期望并不是在平均一个“纯状态概率”，而是在平均每个状态对应的梯度贡献；而这个梯度贡献已经通过 $Q^{\pi_\theta}(s,a)$ 把动作价值考虑进去了。

外层额外做的事情，是给不同状态分配整体重要性权重：当前策略下更常访问到的状态，应该对总更新方向有更大的影响。

状态访问分布并不是一个被单独直接优化的控制量。它会随着策略变化而间接变化。也就是说，策略梯度直接调整的是动作概率，而轨迹分布随之改变，最终会进一步改变状态访问频率。

## 为什么期望要和梯度结合

目标函数本身就是一个期望：

$$
J(\theta)=E[\text{return}]
$$

因此，它的梯度描述的也必须是“平均意义下如何让回报变大”，而不是只针对某一条采样轨迹。

在强化学习里，状态、动作和回报在当前策略与环境动力学下都是随机的。像

$$
Q^{\pi_\theta}(s,a)\nabla_\theta \ln \pi_\theta(a|s)
$$

这样的项，只是一次样本对应的梯度贡献。求期望的作用，就是把许多可能出现的样本贡献整合成一个总体更新方向。

所以，梯度负责回答“参数该往哪改”，期望负责保证这个方向对应的是长期平均上的改进，而不是某一次带噪声的结果。

## 为什么加权平均重要

这个公式优化的是当前策略在真实环境交互中的期望回报。由于状态和动作都不是均匀出现的，所以平均时必须按照它们真实出现的概率分布来加权。

如果改成简单平均，那么极少出现的状态或动作会和高频状态或动作拥有相同影响力，这就不再对应当前策略的真实行为分布。

## $\nabla_\theta \ln \pi_\theta(a|s)$ 的含义

这一项本身并不判断动作好坏。它表达的是：如果想提高或降低状态 $s$ 下动作 $a$ 的选择概率，参数 $\theta$ 应该往哪个方向调整。

- $Q^{\pi_\theta}(s,a)$ 回答的是：这个动作值不值得鼓励。
- $\nabla_\theta \ln \pi_\theta(a|s)$ 回答的是：如果要鼓励或抑制它，参数该怎么改。

因此，两者相乘的含义就是：用动作价值去缩放那个“能够改变该动作概率”的参数调整方向。

## 对数导数恒等式

推导中会用到一个关键恒等式：

$$
\nabla_\theta \pi_\theta(a|s)=\pi_\theta(a|s)\nabla_\theta \ln \pi_\theta(a|s)
$$

它来自链式法则：

$$
\nabla_\theta \ln \pi_\theta(a|s)=\frac{1}{\pi_\theta(a|s)}\nabla_\theta \pi_\theta(a|s)
$$

两边同时乘上 $\pi_\theta(a|s)$，就得到上面的恒等式。

这个恒等式的作用是把梯度改写成“期望”形式，因此才能用采样数据来估计。

## 与 REINFORCE 的关系

理论公式里使用的是 $Q^{\pi_\theta}(s,a)$。在 REINFORCE 中，通常用蒙特卡洛回报来替代它：

$$
G_t = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}
$$

于是采样后的策略梯度写成

$$
\nabla_\theta J(\theta) \approx \sum_t G_t \nabla_\theta \ln \pi_\theta(a_t|s_t)
$$

这就是从理论公式走到实际代码实现的桥梁。

## 一句话总结

策略梯度公式表达的是：在当前策略真实经历到的状态和动作分布下，提高长期回报更高的动作的选择概率，也就是沿着能让这些动作更容易被选中的参数方向去更新网络。
