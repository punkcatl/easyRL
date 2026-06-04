# RL Core Concepts

A growing collection of concept explanations. Organized by category.

## 1. RL Foundation

### MDP (Markov Decision Process)

The mathematical framework underlying all RL — defines the "game rules":

```
MDP = (S, A, P, R, γ)

S: State space    — all possible states (e.g., all car positions/speeds)
A: Action space   — all possible actions (e.g., left/right/accelerate)
P: Transition     — P(s'|s,a) probability of reaching s' from (s,a)
R: Reward         — R(s,a) immediate reward for taking action a in state s
γ: Discount       — how much to value future vs present (0~1)
```

**Markov property**: The future depends only on the current state, not on history:

```
P(sₜ₊₁ | sₜ, aₜ, sₜ₋₁, aₜ₋₁, ...) = P(sₜ₊₁ | sₜ, aₜ)
```

This is why RL agents only need the current state to make decisions — no memory of past states needed (in theory).

**Why it matters**: Every RL algorithm is solving an MDP. The "environment" in code (gym/highway-env) IS the MDP — it defines states, actions, transitions, and rewards. The "agent" is trying to find the optimal policy π* that maximizes expected return within this MDP.

### Episode / Step / Trajectory

- **Step (timestep)**: One decision cycle — observe state, choose action, get reward, transition to next state
- **Episode**: A complete run from initial state to terminal state (e.g., one full game of highway driving until crash or timeout)
- **Trajectory (τ)**: The full sequence of (s₀, a₀, r₀, s₁, a₁, r₁, ..., sₜ) in one episode

```
Episode = [step₀, step₁, step₂, ..., stepₜ]
Step = (state, action, reward, next_state, done)
Trajectory = the same data viewed as one continuous sequence
```

### Reward vs Return (Gₜ)

- **Reward (r)**: Immediate feedback at one step — "how good was this moment"
- **Return (Gₜ)**: Discounted cumulative future reward from step t onward — "how good is the future from here"

```
Gₜ = rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + ...

Example (γ=0.99): rewards = [1, 1, 1, 1]
G₀ = 1 + 0.99 + 0.98 + 0.97 = 3.94  ← "total future value from step 0"
G₃ = 1                                ← "only immediate reward left"
```

RL optimizes Return (long-term), not Reward (short-term).

### Policy (π)

A mapping from state to action — the agent's "brain":

- **Deterministic**: π(s) = a — one fixed action per state (DQN's argmax)
- **Stochastic**: π(a|s) = probability — a distribution over actions (REINFORCE/PPO)

```
Deterministic: state "car ahead" → always brake
Stochastic:    state "car ahead" → 70% brake, 20% change lane, 10% keep
```

Stochastic enables exploration during training. A well-trained policy becomes near-deterministic (one action dominates).

### Value Function V(s) vs Action-Value Q(s,a)

- **V(s)**: "How good is it to be in state s" — expected return from s following policy π
- **Q(s,a)**: "How good is it to take action a in state s" — expected return from (s,a) following policy π

```
V(s) = E[Gₜ | sₜ = s]           ← average over all possible actions
Q(s,a) = E[Gₜ | sₜ = s, aₜ = a] ← specific action chosen

Relationship: V(s) = Σ π(a|s) · Q(s,a)  ← weighted average of Q over policy
```

DQN learns Q(s,a). Actor-Critic's Critic learns V(s).

### Advantage A(s,a)

"How much better is action a compared to the average action in state s":

```
A(s,a) = Q(s,a) - V(s)

If A > 0: this action is better than average → encourage
If A < 0: this action is worse than average → discourage
If A = 0: this action is exactly average
```

Used in A2C/PPO. Reduces variance compared to using raw return Gₜ.

## 2. Learning Methods

### Bellman Equation

The recursive relationship that connects values across time steps:

```
V(s) = E[r + γ · V(s')]     ← "value now = reward now + discounted value next"
Q(s,a) = E[r + γ · max Q(s',a')]  ← DQN's version
```

This is why we can learn values step-by-step (TD learning) instead of waiting for the whole episode (Monte Carlo).

### TD (Temporal Difference) vs Monte Carlo

Two ways to estimate return:

| | Monte Carlo | TD |
|---|---|---|
| When to update | After episode ends | Every step |
| Target | Actual full return Gₜ | r + γ·V(s') (estimate) |
| Bias | None (uses true return) | Has bias (uses estimated V) |
| Variance | High (full trajectory is noisy) | Low (one step is stable) |
| Example | REINFORCE | Actor-Critic, DQN |

```
MC target:  Gₜ = r₀ + γr₁ + γ²r₂ + ... (wait until end)
TD target:  r + γ·V(s')                   (estimate the rest)
```

### Exploration vs Exploitation

The fundamental RL dilemma:

- **Exploration**: Try new actions to discover potentially better strategies
- **Exploitation**: Use what you already know works to maximize reward

| Method | How it explores |
|--------|----------------|
| ε-greedy (DQN) | Random action with probability ε |
| Stochastic policy (REINFORCE/PPO) | Sample from probability distribution |
| Entropy bonus (PPO) | Penalize overly certain policies |
| Maximum entropy (SAC) | Explicitly maximize randomness alongside reward |

Too much exploration → slow learning. Too little → stuck in local optimum.

## 3. Math & Statistics

### Tensor

A multi-dimensional array — the unified name:

```
Scalar (0-dim tensor):  3.14
Vector (1-dim tensor):  [1, 2, 3]
Matrix (2-dim tensor):  [[1,2], [3,4]]
3-dim tensor:           [[[...], [...]], [[...]]]
```

In PyTorch, `torch.tensor` differs from numpy `ndarray` in that it:
- Can run on GPU
- Supports automatic differentiation (needed for backprop)

### Standard Deviation (std)

Measures how "spread out" data is from the mean.

```
Data: [2, 4, 6, 8, 10]
1. Mean: 6
2. Differences from mean: [-4, -2, 0, 2, 4]
3. Squared: [16, 4, 0, 4, 16]
4. Mean of squares (variance): 8
5. Square root (std): √8 ≈ 2.83
```

Variance = std². Variance is for math (squared units, easy to manipulate); std is for intuition (same units as data).

### Z-Score Normalization

`y = (x - mean) / std` — transforms data to mean=0, std=1.

This is mathematically guaranteed:
- Subtract mean → mean becomes 0
- Divide by std → std becomes 1

### Entropy

Measures how "random" a probability distribution is:

```
H(π) = -Σ π(a|s) · log π(a|s)

Uniform [0.2, 0.2, 0.2, 0.2, 0.2]: H = 1.61  ← maximum randomness
Peaked  [0.01, 0.01, 0.96, 0.01, 0.01]: H = 0.24  ← very certain
Deterministic [0, 0, 1, 0, 0]: H = 0  ← no randomness
```

PPO adds entropy bonus to loss to prevent premature convergence (keep exploring).

### KL Divergence

Measures how different two probability distributions are:

```
KL(π_new || π_old) = Σ π_new(a) · log(π_new(a) / π_old(a))

KL = 0: identical distributions
KL > 0: distributions differ (larger = more different)
```

TRPO uses KL to constrain policy updates. PPO uses clip as a simpler alternative.

## 4. Neural Network Basics

### Logits

The raw output of a network before normalization. Has no independent meaning.

```python
logits = self.fc2(x)          # e.g. [2.0, 1.0, -1.0] ← logits
probs = softmax(logits)       # [0.67, 0.24, 0.09]   ← probabilities
```

You can't say "2.0 represents something specific". Only after softmax does it become "67% probability of choosing this action". The name comes from log-odds in statistics but just think of it as "raw scores before softmax".

### Backpropagation

The algorithm to compute gradients (∂loss/∂θ) for all network parameters efficiently using the chain rule:

```
Forward:  input → layer1 → layer2 → ... → output → loss
Backward: loss → ∂loss/∂output → ∂loss/∂layer2 → ... → ∂loss/∂layer1
```

Each layer passes gradients to the previous layer (chain rule). `loss.backward()` does this automatically in PyTorch.

### Activation Functions (ReLU, Tanh, Sigmoid)

Non-linear functions between layers. Without them, stacking layers is useless (multiple linear transforms = one linear transform).

```
ReLU:    max(0, x)     — simple, fast, most common. Problem: dead neurons (x<0 → gradient=0)
Tanh:    [-1, 1]       — centered, used in Actor output for continuous actions
Sigmoid: [0, 1]        — used for probabilities, rarely in hidden layers now
```

### Batch / Epoch

- **Batch**: A subset of data used for one gradient update (e.g., 64 transitions from replay buffer)
- **Epoch**: One complete pass through all available data

```
DQN:   sample a batch (64) from buffer → one update. No "epoch" concept.
PPO:   collect N steps → divide into batches → train for K epochs on same data
```

Larger batch → more stable gradient, slower per-update. Smaller batch → noisier, faster iteration.

### Overfitting

Model memorizes training data but fails on new data. In RL context:

- DQN overfits to replay buffer → performs well on seen states, poorly on new ones
- Too-large network on simple task → memorizes specific trajectories instead of learning general strategy

Signs: training reward high, eval reward low/unstable.

### Hyperparameters vs Parameters

| | Parameters | Hyperparameters |
|---|---|---|
| Examples | Network weights W, bias b | lr, gamma, hidden_dim |
| Who decides | Learned automatically during training | Set manually before training |
| When determined | Continuously updated during training | Fixed before training starts |

Why "hyper": hyper = "higher level". They are "parameters that control how parameters learn":

```
hyperparameter lr = 1e-3
    ↓ controls
parameter W update magnitude
    ↓ determines
network output
```

They don't participate in `loss.backward()`, are not updated by gradients, but determine the behavior of the entire training process — hence "parameters above parameters".

## 5. Training Techniques

### Experience Replay

Store past transitions (s, a, r, s', done) in a buffer, randomly sample batches for training.

Benefits:
- Break correlation between consecutive samples (sequential data hurts neural networks)
- Reuse data multiple times (sample efficiency)
- Smooth out training (mix old and new experiences)

Used by: DQN, DDPG, TD3, SAC. NOT used by: REINFORCE, PPO (on-policy methods).

### Target Network

A delayed copy of the main network, updated slowly:

```
Main Q-net:   θ  ← updated every step
Target Q-net: θ⁻ ← soft update: θ⁻ = τ·θ + (1-τ)·θ⁻
```

Without it: Q target changes every step → chasing a moving target → training diverges.
With it: Q target moves slowly → stable optimization target.

Used by: DQN, DDPG, TD3, SAC.

### GAE (Generalized Advantage Estimation)

A method to compute Advantage that balances bias vs variance via parameter λ:

```
λ = 0: A = r + γV(s') - V(s)          ← low variance, high bias (TD)
λ = 1: A = Gₜ - V(s)                  ← high variance, low bias (MC)
λ = 0.95: weighted blend of both       ← practical sweet spot
```

Used in PPO. The `gae_lambda` hyperparameter controls this tradeoff.

### Gradient Clipping

Limit gradient magnitude to prevent training explosion:

```
If ||gradient|| > max_norm:
    gradient = gradient × (max_norm / ||gradient||)
```

Direction preserved, magnitude capped. Used when single-sample gradients can be extremely large (REINFORCE, RNNs).

---

# RL 核心概念

概念解释汇总，按类别组织，持续更新。

## 1. RL 基础框架

### MDP（马尔可夫决策过程）

所有 RL 算法的数学基础——定义了"游戏规则"：

```
MDP = (S, A, P, R, γ)

S: 状态空间    — 所有可能的状态（如所有车辆位置/速度组合）
A: 动作空间    — 所有可能的动作（如左转/右转/加速）
P: 状态转移    — P(s'|s,a) 在状态 s 做动作 a 后到达 s' 的概率
R: 奖励函数    — R(s,a) 在状态 s 做动作 a 获得的即时奖励
γ: 折扣因子    — 未来奖励相对当前的权重 (0~1)
```

**马尔可夫性质**：未来只取决于当前状态，与历史无关：

```
P(sₜ₊₁ | sₜ, aₜ, sₜ₋₁, aₜ₋₁, ...) = P(sₜ₊₁ | sₜ, aₜ)
```

所以 RL agent 只需要当前状态就能做决策——理论上不需要记忆历史状态。

**为什么重要**：每个 RL 算法都在求解一个 MDP。代码中的"环境"（gym/highway-env）就是 MDP——它定义了状态、动作、转移和奖励。"智能体"要在这个 MDP 中找到最优策略 π* 来最大化期望回报。

### Episode / Step / Trajectory（回合 / 步 / 轨迹）

- **Step（时间步）**：一次决策循环——观察状态、选动作、拿奖励、进入下一状态
- **Episode（回合）**：从初始状态到终止状态的完整一轮（如一次高速公路驾驶直到撞车或超时）
- **Trajectory（轨迹）**：一个 episode 中完整的 (s₀, a₀, r₀, s₁, a₁, r₁, ..., sₜ) 序列

```
Episode = [step₀, step₁, step₂, ..., stepₜ]
Step = (state, action, reward, next_state, done)
Trajectory = 同样的数据看作一条连续序列
```

### Reward vs Return（即时奖励 vs 回报）

- **Reward (r)**：单步的即时反馈——"这一刻有多好"
- **Return (Gₜ)**：从第 t 步开始的折扣累计未来奖励——"从这里开始未来有多好"

```
Gₜ = rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + ...

例子 (γ=0.99): rewards = [1, 1, 1, 1]
G₀ = 1 + 0.99 + 0.98 + 0.97 = 3.94  ← "从第0步开始的总未来价值"
G₃ = 1                                ← "只剩即时奖励"
```

RL 优化的是 Return（长期），不是 Reward（短期）。

### Policy π（策略）

从状态到动作的映射——agent 的"大脑"：

- **确定性策略**：π(s) = a — 每个状态对应一个固定动作（DQN 的 argmax）
- **随机策略**：π(a|s) = 概率 — 动作上的概率分布（REINFORCE/PPO）

```
确定性: 状态"前方有车" → 一定刹车
随机:   状态"前方有车" → 70%刹车, 20%换道, 10%保持
```

随机策略使训练时能探索。训练充分后策略趋于确定性（一个动作概率占主导）。

### Value V(s) vs Q(s,a)（状态价值 vs 动作价值）

- **V(s)**："处于状态 s 有多好" — 从 s 出发遵循策略 π 的期望回报
- **Q(s,a)**："在状态 s 做动作 a 有多好" — 从 (s,a) 出发遵循策略 π 的期望回报

```
V(s) = E[Gₜ | sₜ = s]           ← 对所有可能动作取平均
Q(s,a) = E[Gₜ | sₜ = s, aₜ = a] ← 指定了具体动作

关系: V(s) = Σ π(a|s) · Q(s,a)  ← Q 按策略的加权平均
```

DQN 学 Q(s,a)。Actor-Critic 的 Critic 学 V(s)。

### Advantage A(s,a)（优势函数）

"动作 a 比该状态下的平均动作好多少"：

```
A(s,a) = Q(s,a) - V(s)

A > 0: 这个动作优于平均 → 鼓励
A < 0: 这个动作劣于平均 → 抑制
A = 0: 这个动作恰好是平均水平
```

用于 A2C/PPO。相比直接用 Gₜ，方差更小。

## 2. 学习方法

### Bellman 方程

连接相邻时间步价值的递推关系：

```
V(s) = E[r + γ · V(s')]           ← "当前价值 = 当前奖励 + 折扣后的下一步价值"
Q(s,a) = E[r + γ · max Q(s',a')]  ← DQN 版本
```

有了它才能逐步学习价值（TD 学习），而不用等整个 episode 结束（Monte Carlo）。

### TD vs Monte Carlo（时序差分 vs 蒙特卡洛）

两种估计回报的方式：

| | Monte Carlo | TD |
|---|---|---|
| 何时更新 | episode 结束后 | 每步都更新 |
| 目标值 | 真实完整回报 Gₜ | r + γ·V(s')（估计值）|
| 偏差 | 无（用的是真实回报）| 有偏（用了估计的 V）|
| 方差 | 高（整条轨迹噪声大）| 低（单步稳定）|
| 例子 | REINFORCE | Actor-Critic, DQN |

```
MC 目标:  Gₜ = r₀ + γr₁ + γ²r₂ + ... （等到结束）
TD 目标:  r + γ·V(s')                  （估计剩余部分）
```

### Exploration vs Exploitation（探索 vs 利用）

RL 的根本困境：

- **探索**：尝试新动作，发现可能更好的策略
- **利用**：用已知有效的策略最大化奖励

| 方法 | 如何探索 |
|------|---------|
| ε-greedy (DQN) | 以概率 ε 随机选动作 |
| 随机策略 (REINFORCE/PPO) | 从概率分布中采样 |
| Entropy bonus (PPO) | 惩罚过于确定的策略 |
| 最大熵 (SAC) | 显式最大化随机性的同时最大化奖励 |

探索太多 → 学习慢。探索太少 → 卡在局部最优。

## 3. 数学 / 统计工具

### 张量 (Tensor)

多维数组的统称：

```
标量 (0维张量):  3.14
向量 (1维张量):  [1, 2, 3]
矩阵 (2维张量):  [[1,2], [3,4]]
3维张量:        [[[...], [...]], [[...]]]
```

PyTorch 的 `torch.tensor` 和 numpy 的 `ndarray` 本质一样（多维数组），区别是：
- 能放到 GPU 上加速
- 能自动求导（反向传播需要）

### 标准差 (Standard Deviation)

衡量数据离均值有多"散"。

```
数据: [2, 4, 6, 8, 10]
1. 均值: 6
2. 与均值的差: [-4, -2, 0, 2, 4]
3. 平方: [16, 4, 0, 4, 16]
4. 平方的均值（方差）: 8
5. 开根号（标准差）: √8 ≈ 2.83
```

方差 = 标准差²。方差用于数学推导（单位是平方，便于计算）；标准差用于直觉理解（和原数据同单位）。

### Z-Score 标准化

`y = (x - mean) / std` — 将数据变换为均值=0、标准差=1。

这是数学上必然的：
- 减自己的均值 → 均值归零
- 除以自己的标准差 → 标准差归一

### 熵 (Entropy)

衡量概率分布有多"随机"：

```
H(π) = -Σ π(a|s) · log π(a|s)

均匀 [0.2, 0.2, 0.2, 0.2, 0.2]: H = 1.61  ← 最大随机性
集中 [0.01, 0.01, 0.96, 0.01, 0.01]: H = 0.24  ← 非常确定
确定 [0, 0, 1, 0, 0]: H = 0  ← 零随机性
```

PPO 在 loss 中加 entropy bonus 防止过早收敛（保持探索）。

### KL 散度 (KL Divergence)

衡量两个概率分布有多不同：

```
KL(π_new || π_old) = Σ π_new(a) · log(π_new(a) / π_old(a))

KL = 0: 分布完全相同
KL > 0: 分布有差异（越大差异越大）
```

TRPO 用 KL 约束策略更新幅度。PPO 用 clip 作为更简单的替代。

## 4. 神经网络基础

### Logits

网络归一化之前的原始输出值，没有独立含义。

```python
logits = self.fc2(x)          # 比如 [2.0, 1.0, -1.0] ← logits
probs = softmax(logits)       # [0.67, 0.24, 0.09]   ← 概率
```

不能说 2.0 代表什么具体量。只有经过 softmax 变成概率后，才有"选这个动作的概率是 67%"这样的含义。名字来自统计学的 log-odds，简单理解为"softmax 之前的原始分数"。

### 反向传播 (Backpropagation)

用链式法则高效计算所有参数梯度 (∂loss/∂θ) 的算法：

```
前向: input → layer1 → layer2 → ... → output → loss
反向: loss → ∂loss/∂output → ∂loss/∂layer2 → ... → ∂loss/∂layer1
```

每层把梯度传给前一层（链式法则）。PyTorch 中 `loss.backward()` 自动完成。

### 激活函数 (ReLU, Tanh, Sigmoid)

层之间的非线性函数。没有它们，多层叠加等于一层（多个线性变换 = 一个线性变换）。

```
ReLU:    max(0, x)     — 简单高效，最常用。问题：死神经元（x<0 时梯度=0）
Tanh:    [-1, 1]       — 中心对称，连续动作 Actor 输出常用
Sigmoid: [0, 1]        — 用于概率输出，现在很少在隐藏层用
```

### Batch / Epoch（批次 / 轮次）

- **Batch**：一次梯度更新用的数据子集（如从 replay buffer 中采样 64 条经验）
- **Epoch**：所有可用数据完整过一遍

```
DQN:  从 buffer 采一个 batch (64) → 更新一次。没有 "epoch" 概念。
PPO:  收集 N 步 → 分成多个 batch → 在同一批数据上训练 K 个 epoch
```

Batch 越大 → 梯度越稳定，但单次更新越慢。Batch 越小 → 噪声大，迭代快。

### 过拟合 (Overfitting)

模型记住了训练数据但对新数据表现差。在 RL 中：

- DQN 过拟合 replay buffer → 在见过的状态上表现好，新状态上差
- 网络太大 + 任务简单 → 记住了特定轨迹而非学到通用策略

表现：训练 reward 高，eval reward 低/不稳定。

### 超参数 vs 参数

| | 参数 (parameters) | 超参数 (hyperparameters) |
|---|---|---|
| 例子 | 网络权重 W、偏置 b | lr、gamma、hidden_dim |
| 谁决定的 | 训练过程自动学出来 | 人手动设定 |
| 何时确定 | 训练中不断更新 | 训练前就固定 |

为什么叫"超"：hyper = "更高层的"。它是"控制参数如何学习的参数"：

```
超参数 lr = 1e-3
    ↓ 控制
参数 W 的更新幅度
    ↓ 决定
网络输出
```

它们不参与 `loss.backward()`，不被梯度更新，但决定了整个训练过程的行为——所以叫"超越参数的参数"。

## 5. 训练技巧

### 经验回放 (Experience Replay)

存储过去的经验 (s, a, r, s', done) 到 buffer，随机采样 batch 用于训练。

好处：
- 打破连续样本间的相关性（序列数据对神经网络不利）
- 数据复用多次（提高样本效率）
- 平滑训练（混合新旧经验）

使用：DQN、DDPG、TD3、SAC。不使用：REINFORCE、PPO（on-policy 方法）。

### 目标网络 (Target Network)

主网络的延迟副本，缓慢更新：

```
主 Q 网络:   θ  ← 每步更新
目标 Q 网络: θ⁻ ← 软更新: θ⁻ = τ·θ + (1-τ)·θ⁻
```

没有它：Q 目标每步都变 → 追逐移动靶 → 训练发散。
有了它：Q 目标缓慢移动 → 优化目标稳定。

使用：DQN、DDPG、TD3、SAC。

### GAE（广义优势估计）

一种平衡偏差与方差的优势计算方法，通过参数 λ 控制：

```
λ = 0: A = r + γV(s') - V(s)          ← 低方差，高偏差（TD）
λ = 1: A = Gₜ - V(s)                  ← 高方差，低偏差（MC）
λ = 0.95: 两者的加权混合              ← 实践中的最佳平衡点
```

用于 PPO。`gae_lambda` 超参数控制这个权衡。

### 梯度裁剪 (Gradient Clipping)

限制梯度大小防止训练爆炸：

```
如果 ||梯度|| > max_norm:
    梯度 = 梯度 × (max_norm / ||梯度||)
```

方向不变，幅度受限。用于单样本梯度可能极大的场景（REINFORCE、RNN）。
