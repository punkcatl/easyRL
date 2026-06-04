# Value-Based vs Policy-Based Methods

## Overview

This note summarizes the key differences between value-based reinforcement learning methods and policy-based methods, using DQN and REINFORCE as the main examples.

In short:

- Value-based methods learn how good each action is.
- Policy-based methods learn how likely each action should be.

## Quick Comparison

| Item | DQN / Value-Based | REINFORCE / Policy-Based |
|---|---|---|
| Learns | $Q(s,a)$ | $\pi(a\|s)$ |
| Network output | action values | action probabilities |
| Action selection | `argmax` or $\epsilon$-greedy | sample from distribution |
| Main update idea | fit value target | increase probability of high-return actions |

## What Each Method Learns

### Value-Based Methods

Value-based methods approximate an action-value function:

$$
Q(s, a)
$$

It means: if the agent takes action $a$ in state $s$, what long-term return should it expect?

So the network output is usually a vector of action values:

$$
[Q(s,a_1), Q(s,a_2), \dots, Q(s,a_n)]
$$

These outputs are scores, not probabilities. They do not need to be non-negative, and they do not need to sum to 1.

The policy is then derived from the values, for example by choosing:

$$
a^* = \arg\max_a Q(s,a)
$$

### Policy-Based Methods

Policy-based methods learn the policy directly:

$$
\pi(a|s)
$$

It means: in state $s$, what is the probability of choosing action $a$?

So the network output is usually a probability distribution over actions:

$$
[\pi(a_1|s), \pi(a_2|s), \dots, \pi(a_n|s)]
$$

These outputs must form a valid probability distribution, so for discrete actions the network usually ends with `Softmax`.

## Why One Outputs Values and the Other Outputs Probabilities

### Value-Based

The model is not directly learning how to act. It learns how good each action is, and the policy is derived afterward by comparing those values.

So the network should output action values.

### Policy-Based

The model is directly learning the action-selection rule itself. In discrete action spaces, that rule is naturally written as a probability distribution.

So the network should output action probabilities.

## Action Selection Difference

### Value-Based

Typical choice rule:

$$
a^* = \arg\max_a Q(s,a)
$$

This is usually deterministic unless exploration such as $\epsilon$-greedy is added.

### Policy-Based

Typical choice rule:

$$
a \sim \pi(\cdot|s)
$$

This means the action is sampled from the policy distribution.

For example, if the output probabilities are:

$$
[0.7, 0.2, 0.1]
$$

then action 1 is sampled more often, while actions 2 and 3 remain possible.

## How Softmax Converts Scores Into Probabilities

Suppose a policy network produces raw scores (logits):

$$
[z_1, z_2, \dots, z_n]
$$

Softmax converts them into probabilities using:

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

This does two things:

- exponentiation makes all outputs positive
- normalization makes them sum to 1

Example:

$$
[2.0, 1.0, 0.1]
$$

After exponentiation:

$$
[e^{2.0}, e^{1.0}, e^{0.1}] \approx [7.39, 2.72, 1.11]
$$

After normalization:

$$
[0.659, 0.242, 0.099]
$$

So Softmax turns arbitrary real-valued scores into a valid probability distribution.

## Typical Output Examples

- Value output: `[2.3, 1.1, 0.7]`
- Policy output: `[0.62, 0.25, 0.13]`

## Suggested Reading Order

For learning and review, this order is more useful than reading the files randomly:

1. `hands_on_rl/ch09_policy_gradient/code/policy_gradient.py`
	Start with the teaching-style REINFORCE implementation. It is the most direct bridge between the policy gradient formula and code.
2. `algorithms/policy_gradient/agent.py`
	Then compare it with the project version to see the more engineering-oriented implementation style.
3. `algorithms/dqn/agent.py`
	Read the DQN agent after that to contrast value-based output with policy-based output.
4. `docs/policy_gradient_formula_summary.md`
	Return to the formula summary when you want to connect the code back to the mathematical objective.

## One-Sentence Summary

Value-based methods learn a scoring function over actions, while policy-based methods learn the action distribution itself.

---

# 价值方法与策略方法对比

## 概览

这份笔记总结了基于价值的强化学习方法和基于策略的方法之间的关键区别，并以 DQN 和 REINFORCE 作为主要例子。

一句话概括：

- 价值方法学习每个动作“有多好”
- 策略方法学习每个动作“应该以多大概率被选中”

## 快速对照

| 对比项 | DQN / 价值方法 | REINFORCE / 策略方法 |
|---|---|---|
| 学习对象 | $Q(s,a)$ | $\pi(a\|s)$ |
| 网络输出 | 动作价值 | 动作概率 |
| 动作选择 | `argmax` 或 $\epsilon$-greedy | 从分布中采样 |
| 更新核心 | 拟合价值目标 | 提高高回报动作的概率 |

## 两类方法各自学什么

### 基于价值的方法

价值方法近似的是动作价值函数：

$$
Q(s, a)
$$

它表示：在状态 $s$ 下执行动作 $a$，未来的长期回报期望是多少。

所以网络输出通常是一个动作价值向量：

$$
[Q(s,a_1), Q(s,a_2), \dots, Q(s,a_n)]
$$

这些输出是分数，不是概率。它们不要求非负，也不要求和为 1。

策略通常是从这些价值中推导出来的，例如直接选择：

$$
a^* = \arg\max_a Q(s,a)
$$

### 基于策略的方法

策略方法直接学习策略本身：

$$
\pi(a|s)
$$

它表示：在状态 $s$ 下选择动作 $a$ 的概率是多少。

所以网络输出通常是一个动作概率分布：

$$
[\pi(a_1|s), \pi(a_2|s), \dots, \pi(a_n|s)]
$$

这些输出必须构成合法的概率分布，因此离散动作策略网络最后通常要接 `Softmax`。

## 为什么一个输出价值，一个输出概率

### 价值方法

模型并不是直接学习“怎么选动作”，而是在学习“每个动作有多好”，再根据这些价值推导出动作选择。

因此网络自然应该输出动作价值。

### 策略方法

模型直接学习动作选择规则本身。而在离散动作空间里，这个规则最自然的表达形式就是概率分布。

因此网络自然应该输出动作概率。

## 动作选择方式的差异

### 价值方法

典型选择规则：

$$
a^* = \arg\max_a Q(s,a)
$$

这通常是确定性的，除非额外加入 $\epsilon$-greedy 之类的探索机制。

### 策略方法

典型选择规则：

$$
a \sim \pi(\cdot|s)
$$

也就是从策略分布中采样动作。

例如，如果输出概率是：

$$
[0.7, 0.2, 0.1]
$$

那么动作 1 会更常被选中，但动作 2 和动作 3 仍然可能被采样。

## Softmax 如何把分数变成概率

假设策略网络先输出原始分数（logits）：

$$
[z_1, z_2, \dots, z_n]
$$

Softmax 用下面这个公式把它们变成概率：

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

它做了两件事：

- 通过指数运算把所有输出变成正数
- 通过归一化让它们的总和变成 1

例如：

$$
[2.0, 1.0, 0.1]
$$

指数运算后：

$$
[e^{2.0}, e^{1.0}, e^{0.1}] \approx [7.39, 2.72, 1.11]
$$

归一化后：

$$
[0.659, 0.242, 0.099]
$$

所以 Softmax 可以把任意实数分数转成合法的概率分布。

## 典型输出示例

- 价值输出：`[2.3, 1.1, 0.7]`
- 策略输出：`[0.62, 0.25, 0.13]`

## 建议阅读顺序

如果是为了学习和回看，比起随机看文件，下面这个顺序更合适：

1. `hands_on_rl/ch09_policy_gradient/code/policy_gradient.py`
	先看教学版 REINFORCE 实现。它和策略梯度公式的对应关系最直接。
2. `algorithms/policy_gradient/agent.py`
	再对比项目里的工程版实现，观察更偏工程化的写法。
3. `algorithms/dqn/agent.py`
	接着看 DQN 智能体，把 value-based 的输出和 policy-based 的输出直接对照起来。
4. `docs/policy_gradient_formula_summary.md`
	最后回到公式总结，把代码和数学目标重新对应起来。

## 一句话总结

价值方法学习的是动作评分函数，策略方法学习的是动作分布本身。
