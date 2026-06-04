# RL Hyperparameter Tuning Guide

## Part 1: General Tuning Workflow

### Pre-Tuning Diagnostic Checklist

Before spending time on hyperparameters, confirm it's actually a tuning problem:

| Check | How | If Failed |
|-------|-----|-----------|
| Environment works | Random agent gets non-zero reward sometimes | Fix env setup, check action/observation space |
| Observation is correct | Print obs shape & range, verify matches docs | Fix preprocessing, normalization |
| Reward signal exists | Plot reward over episodes for random policy | Reward function is broken, not a tuning issue |
| Loss is decreasing | Check loss curve in TensorBoard | Bug in training loop, not hyperparameters |
| Gradients are flowing | Check grad norms are non-zero, non-exploding | Fix network init, check for detached tensors |
| Buffer is filling | Verify training starts after sufficient samples | Adjust min buffer size or episode count |
| Actions are valid | Agent outputs are within env action space | Fix action clipping/scaling |

**Rule: if loss doesn't decrease at all, it's a bug, not a tuning problem.**

### General Tuning Priority

Tune in this order — later items rarely matter if earlier ones are wrong:

```
1. Reward Design     — is the signal learnable? (see Part 3)
2. Learning Rate     — most impactful single param
3. Exploration       — epsilon / entropy / noise (algorithm-dependent)
4. Network Size      — must match problem complexity
5. Batch & Buffer    — sample efficiency and stability
6. Everything Else   — gamma, tau, clip range, etc.
```

### Symptom Lookup Table

| Symptom | Likely Cause | First Fix |
|---------|-------------|-----------|
| Reward stays at 0 / doesn't improve | LR too high/low, or exploration too fast/slow | Check LR with known-good env (CartPole) first |
| Reward improves then collapses | LR too high, target net lag too large, or PPO clip too wide | Halve LR, reduce tau/increase target update freq |
| Loss explodes (NaN/Inf) | LR too high, no gradient clipping, bad reward scale | Add gradient clipping, normalize rewards |
| Loss decreases but reward doesn't | Value overestimation, or network memorizing buffer | Use Double DQN / clipped critics, check buffer diversity |
| High variance in reward | Small buffer, no target net, or too much exploration noise | Increase buffer, verify target net, reduce noise |
| Agent gets stuck in one action | Exploration died too fast, or entropy too low | Slow epsilon decay, increase entropy coefficient |
| Training very slow | LR too low, batch too large, network too small | Increase LR, reduce batch, increase hidden dim |
| Policy oscillates between strategies | LR too high for policy, or PPO epochs too many | Reduce policy LR, fewer PPO epochs, smaller clip |

### General Parameter Details

#### Learning Rate (`lr`)

**Principle**: controls step size in parameter space. Too high → oscillation/divergence. Too low → slow/stuck.

| Range | Use Case |
|-------|----------|
| 1e-4 | Complex environments, fine-tuning, pixel inputs |
| 3e-4 | Default for most algorithms (PPO, SAC, TD3) |
| 1e-3 | Simple environments, value-based methods (DQN) |
| 3e-3 | Very simple tasks or initial exploration |

**Diagnosis:**

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Loss oscillates wildly | LR too high | Halve it |
| Loss barely moves after 1000+ updates | LR too low | Double it |
| Works initially, then diverges | LR too high for later training | Use scheduler (linear decay / cosine) |

**Coupling**: when you double batch size, increase LR by ~1.4-2x. (Linear scaling rule says 2x; sqrt scaling says ~1.4x. In RL, sqrt scaling is often more stable — 1.5x is a practical middle ground.)

#### Discount Factor (`gamma`)

**Principle**: effective horizon ≈ 1 / (1 - gamma). Agent considers ~H future steps.

| gamma | Horizon | When to Use |
|-------|---------|-------------|
| 0.9 | ~10 steps | Very short episodes, immediate reward tasks |
| 0.95 | ~20 steps | Short episodes |
| 0.99 | ~100 steps | Most tasks (default) |
| 0.995 | ~200 steps | Long episodes |
| 0.999 | ~1000 steps | Very long episodes, sparse rewards |

**Rule of thumb**: set gamma so that horizon > episode length. For highway-env (~30-50 steps), 0.99 is fine.

#### Network Architecture

| Problem | Input Dim | Recommended Architecture |
|---------|-----------|--------------------------|
| Simple (CartPole) | < 10 | 2 layers, 64 units |
| Medium (Highway-env) | 10-100 | 2 layers, 128-256 units |
| Complex (continuous control) | 10-100 | 2 layers, 256 units |
| Pixel-based | image | CNN encoder + 2 FC layers, 512 units |

**Signs of wrong size:**
- Too small: loss plateaus high, reward never improves (underfitting)
- Too large: loss fine but reward noisy (overfitting to buffer)

**Tips:**
- Use ReLU for value-based methods, Tanh for policy output in continuous action spaces
- Layer normalization helps in continuous control
- Orthogonal initialization is slightly better than default

#### Gradient Clipping (`max_grad_norm`)

| Value | Effect |
|-------|--------|
| 0.5 | Very conservative, prevents all spikes but may slow learning |
| 1.0 | Standard default (PPO, A2C) |
| 10.0 | Permissive, only catches extreme cases |
| None | No clipping — risky with high LR or sparse rewards |

**When to use**: always for policy gradient methods. Optional for DQN (Huber loss already clips implicitly).

#### Batch Size

| Size | Pros | Cons |
|------|------|------|
| 32 | Frequent updates, fast early learning | High gradient variance |
| 64 | Good balance for value-based methods | — |
| 128-256 | Standard for PPO/A2C (needs enough trajectories) | — |
| 2048+ | Stable policy gradients (PPO large-scale) | Needs many env steps per update |

**For on-policy** (PPO/A2C): total rollout buffer = num_envs × n_steps. This is then split into minibatches (e.g., 8192 total / 64 minibatch_size = 128 minibatches per epoch). Larger total buffer → more stable gradient estimate.

**For off-policy** (DQN/SAC/TD3): batch sampled from buffer. 256 is typical default.

---

## Part 2: Algorithm-Specific Guides

---

### REINFORCE

#### Key Parameters

| Parameter | Typical Range | Role |
|-----------|---------------|------|
| `lr` | 1e-4 ~ 3e-3 | Policy network learning rate |
| `gamma` | 0.99 ~ 0.999 | Discount factor |
| `baseline` | True/False | Whether to subtract baseline (mean return) to reduce variance |
| `entropy_bonus` | 0.0 ~ 0.1 | Coefficient for entropy regularization |
| `max_grad_norm` | 0.5 ~ 5.0 | Gradient clipping threshold |

#### Typical Network

```
Input (state_dim)
  → Linear(state_dim, 128) → ReLU
  → Linear(128, 128) → ReLU
  → Linear(128, action_dim) → Softmax
```

#### Common Problems

| Problem | Cause | Solution |
|---------|-------|----------|
| Extreme variance in returns | No baseline | Enable baseline subtraction |
| Policy collapses to one action | Entropy too low, LR too high | Add entropy bonus (0.01-0.05), reduce LR |
| Very slow convergence | High variance, small batch | Use more episodes per update, add baseline |
| Gradient explodes | Long episodes with large rewards | Normalize returns, clip gradients |

#### Quick Start

```python
config = {
    "lr": 1e-3,
    "gamma": 0.99,
    "baseline": True,
    "entropy_bonus": 0.01,
    "max_grad_norm": 1.0,
    "hidden_dim": 128,
}
```

---

### DQN

#### Key Parameters

| Parameter | Typical Range | Role |
|-----------|---------------|------|
| `lr` | 1e-4 ~ 1e-3 | Q-network learning rate |
| `gamma` | 0.9 ~ 0.999 | Discount factor |
| `epsilon_start` | 0.5 ~ 1.0 | Initial exploration rate |
| `epsilon_end` | 0.01 ~ 0.1 | Minimum exploration rate |
| `epsilon_decay` | 0.99 ~ 0.999 | Per-episode decay multiplier |
| `buffer_size` | 10000 ~ 1000000 | Replay buffer capacity |
| `batch_size` | 32 ~ 256 | Samples per update |
| `tau` | 0.001 ~ 0.01 | Target network soft update rate |
| `hidden_dim` | 64 ~ 512 | Network width |

**Epsilon schedule**: episodes to reach epsilon_end ≈ `log(epsilon_end / epsilon_start) / log(epsilon_decay)`

| epsilon_decay | Episodes to reach 0.01 | Character |
|---------------|------------------------|-----------|
| 0.99 | ~460 | Fast — risky if under-explored |
| 0.995 | ~920 | Balanced |
| 0.999 | ~4600 | Thorough exploration |

#### Variants

| Variant | What It Changes | When to Use |
|---------|----------------|-------------|
| Double DQN | Decouples action selection from evaluation | Always (nearly free, reduces overestimation) |
| Dueling DQN | Splits Q into V + A streams | When some states are clearly good/bad regardless of action |
| Prioritized Replay | Samples high-TD-error transitions more often | Sparse reward, but adds complexity |
| N-step Returns | Uses n-step bootstrapping | Faster propagation, but biased with off-policy |

#### Typical Network

```
Input (state_dim)
  → Linear(state_dim, 128) → ReLU
  → Linear(128, 128) → ReLU
  → Linear(128, action_dim)  # raw Q-values, no activation
```

Dueling variant:
```
Input (state_dim)
  → Linear(state_dim, 128) → ReLU
  ├→ Linear(128, 1)           # V(s)
  └→ Linear(128, action_dim)  # A(s,a)
  → Q = V + (A - mean(A))
```

#### Common Problems

| Problem | Cause | Solution |
|---------|-------|----------|
| Reward stays at 0 | Epsilon decays before buffer fills | Increase epsilon_decay, delay training start |
| Reward improves then collapses | LR too high, or tau too large | Halve LR, reduce tau to 0.001 |
| Loss decreases but reward doesn't | Q-value overestimation | Use Double DQN, reduce LR |
| High reward variance | Buffer too small | Increase buffer_size to 100k+ |
| Agent fixates on one action | Epsilon decayed too fast | Slow decay, raise epsilon_end |

#### Quick Start

```python
config = {
    "lr": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "buffer_size": 50000,
    "batch_size": 64,
    "hidden_dim": 128,
    "tau": 0.005,
    "double_dqn": True,
}
```

---

### A2C (Advantage Actor-Critic)

#### Key Parameters

| Parameter | Typical Range | Role |
|-----------|---------------|------|
| `lr` | 1e-4 ~ 1e-3 | Shared or separate LR for actor/critic |
| `gamma` | 0.99 ~ 0.999 | Discount factor |
| `n_steps` | 5 ~ 128 | Steps before bootstrapping (rollout length) |
| `entropy_coef` | 0.001 ~ 0.1 | Entropy regularization weight |
| `value_loss_coef` | 0.25 ~ 1.0 | Weight of value loss in total loss |
| `max_grad_norm` | 0.5 ~ 5.0 | Gradient clipping |
| `num_envs` | 4 ~ 32 | Parallel environments for variance reduction |

#### Typical Network

Shared backbone:
```
Input (state_dim)
  → Linear(state_dim, 128) → ReLU
  → Linear(128, 128) → ReLU
  ├→ Linear(128, action_dim) → Softmax    # policy head
  └→ Linear(128, 1)                        # value head
```

#### Common Problems

| Problem | Cause | Solution |
|---------|-------|----------|
| Policy collapses early | Entropy coef too low | Increase to 0.05-0.1 |
| Value loss dominates training | value_loss_coef too high | Reduce to 0.25 |
| High variance, unstable | Too few envs or n_steps too short | Increase num_envs, increase n_steps |
| Slow learning | LR too low or n_steps too long (high bias from infrequent bootstrapping) | Increase LR, reduce n_steps |
| Premature convergence | Entropy dies off | Use entropy coef schedule (start high, decay) |

#### Quick Start

```python
config = {
    "lr": 7e-4,
    "gamma": 0.99,
    "n_steps": 5,
    "entropy_coef": 0.01,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,
    "num_envs": 16,
    "hidden_dim": 128,
}
```

---

### PPO (Proximal Policy Optimization)

#### Key Parameters

| Parameter | Typical Range | Role |
|-----------|---------------|------|
| `lr` | 1e-4 ~ 3e-4 | Learning rate (often linearly decayed) |
| `gamma` | 0.99 ~ 0.999 | Discount factor |
| `gae_lambda` | 0.9 ~ 0.99 | GAE bias-variance tradeoff |
| `clip_range` | 0.1 ~ 0.3 | PPO clipping epsilon |
| `n_epochs` | 3 ~ 10 | Passes over collected rollout |
| `n_steps` | 128 ~ 2048 | Rollout length per env |
| `batch_size` | 32 ~ 512 | Minibatch size within epoch |
| `entropy_coef` | 0.0 ~ 0.05 | Entropy bonus |
| `value_loss_coef` | 0.5 ~ 1.0 | Value function loss weight |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `num_envs` | 4 ~ 128 | Parallel environments |

**GAE lambda** controls bias-variance in advantage estimation:
- lambda=0: 1-step TD advantage, A = r + gamma*V(s') - V(s) (low variance, high bias)
- lambda=1: MC advantage, uses full episode return minus baseline (high variance, low bias)
- lambda=0.95: good default — blends multi-step returns

**Clip range** limits policy change per update:
- 0.1: conservative updates (more stable)
- 0.2: standard (default)
- 0.3: aggressive updates (faster but less stable)

#### Typical Network

```
Input (state_dim)
  → Linear(state_dim, 256) → Tanh
  → Linear(256, 256) → Tanh
  ├→ Linear(256, action_dim)   # policy (+ log_std for continuous)
  └→ Linear(256, 1)            # value

# For continuous actions:
# policy outputs mean, log_std is a learnable parameter
```

#### Common Problems

| Problem | Cause | Solution |
|---------|-------|----------|
| Policy doesn't improve | LR too low, or clip too tight | Increase LR to 3e-4, widen clip to 0.2 |
| Training unstable / reward collapses | Too many epochs, clip too wide | Reduce n_epochs to 3-4, clip to 0.1 |
| Policy oscillates | LR too high, batch too small | Reduce LR, increase n_steps × num_envs |
| Entropy drops to zero | Entropy coef too low | Increase to 0.01-0.05 |
| Value function inaccurate | Separate networks diverging | Increase value_loss_coef, or share backbone |
| Sample inefficient | n_steps too short, few envs | Increase rollout length and num_envs |

#### Quick Start

```python
config = {
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "n_epochs": 10,
    "n_steps": 2048,
    "batch_size": 64,
    "entropy_coef": 0.01,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,
    "num_envs": 4,
    "hidden_dim": 256,
}
```

---

### TD3 (Twin Delayed DDPG)

#### Key Parameters

| Parameter | Typical Range | Role |
|-----------|---------------|------|
| `actor_lr` | 1e-4 ~ 3e-4 | Actor learning rate |
| `critic_lr` | 1e-4 ~ 3e-4 | Critic learning rate (often = actor_lr) |
| `gamma` | 0.98 ~ 0.99 | Discount factor |
| `tau` | 0.005 | Target network soft update |
| `policy_delay` | 2 | Update actor every N critic updates |
| `noise_std` | 0.1 ~ 0.3 | Exploration noise (Gaussian) |
| `target_noise_std` | 0.2 | Target policy smoothing noise |
| `target_noise_clip` | 0.5 | Clipping range for target noise |
| `buffer_size` | 100000 ~ 1000000 | Replay buffer |
| `batch_size` | 100 ~ 256 | Samples per update |
| `learning_starts` | 10000 ~ 25000 | Random exploration steps before training |

**TD3 innovations over DDPG:**
1. Twin critics (take minimum → reduces overestimation)
2. Delayed policy updates (policy_delay=2 → more stable)
3. Target policy smoothing (add noise to target action → smoother Q)

#### Typical Network

```
# Actor
Input (state_dim)
  → Linear(state_dim, 256) → ReLU
  → Linear(256, 256) → ReLU
  → Linear(256, action_dim) → Tanh  # bounded actions

# Critic (x2, twin critics)
Input (state_dim + action_dim)
  → Linear(state_dim + action_dim, 256) → ReLU
  → Linear(256, 256) → ReLU
  → Linear(256, 1)
```

#### Common Problems

| Problem | Cause | Solution |
|---------|-------|----------|
| Agent doesn't explore | Noise too low, or learning_starts too short | Increase noise_std, increase learning_starts |
| Q-values diverge | LR too high, no twin critics | Ensure twin critic, reduce LR |
| Jerky/noisy actions | Noise too high at test time | Use deterministic policy at evaluation (no noise) |
| Slow convergence | Policy delay too high, LR too low | Try policy_delay=2 (default), increase LR |
| Performance plateaus | Buffer too small, or exploration noise decays | Increase buffer, keep noise constant during training |

#### Quick Start

```python
config = {
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "policy_delay": 2,
    "noise_std": 0.1,
    "target_noise_std": 0.2,
    "target_noise_clip": 0.5,
    "buffer_size": 1000000,
    "batch_size": 256,
    "learning_starts": 25000,
    "hidden_dim": 256,
}
```

---

### SAC (Soft Actor-Critic)

#### Key Parameters

| Parameter | Typical Range | Role |
|-----------|---------------|------|
| `actor_lr` | 1e-4 ~ 3e-4 | Actor learning rate |
| `critic_lr` | 1e-4 ~ 3e-4 | Critic learning rate |
| `alpha_lr` | 1e-4 ~ 3e-4 | Temperature (alpha) learning rate |
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.005 | Target network soft update |
| `alpha` | 0.2 (if fixed) | Entropy temperature |
| `auto_alpha` | True | Whether to auto-tune alpha |
| `target_entropy` | -dim(A) | Target entropy for auto-tuning (continuous) |
| `buffer_size` | 100000 ~ 1000000 | Replay buffer |
| `batch_size` | 256 | Samples per update |
| `learning_starts` | 5000 ~ 25000 | Random steps before training |

**Alpha (entropy temperature)**:
- High alpha → more exploration (policy is more random)
- Low alpha → more exploitation (policy is more deterministic)
- Auto-tuning (recommended): adjusts alpha to maintain target entropy

**Target entropy**:
- Continuous: -dim(action_space) is the standard heuristic (from SAC paper)
- Discrete: -log(1/|A|) × ratio (ratio ~0.98) — common heuristic from SAC-Discrete implementations, not in original paper

#### Typical Network

```
# Actor (outputs distribution)
Input (state_dim)
  → Linear(state_dim, 256) → ReLU
  → Linear(256, 256) → ReLU
  ├→ Linear(256, action_dim)  # mean
  └→ Linear(256, action_dim)  # log_std (clamped to [-20, 2])
  → sample via reparameterization trick
  → Tanh squashing for bounded actions

# Critic (x2, twin critics)
Input (state_dim + action_dim)
  → Linear(state_dim + action_dim, 256) → ReLU
  → Linear(256, 256) → ReLU
  → Linear(256, 1)
```

#### Common Problems

| Problem | Cause | Solution |
|---------|-------|----------|
| Too much exploration, never converges | Alpha too high | Enable auto-tuning, or reduce fixed alpha |
| Premature convergence | Alpha too low or drops too fast | Increase target entropy, or raise alpha |
| Q-values diverge to large numbers | LR too high, no reward scaling | Reduce LR, normalize/clip rewards |
| Actions saturate at bounds | Tanh squashing + high alpha | Lower target entropy, check log_std bounds |
| Unstable training early on | learning_starts too low | Increase to 10k-25k for buffer diversity |
| NaN in log probabilities | log_std unbounded | Clamp log_std to [-20, 2] |

#### Quick Start

```python
config = {
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "auto_alpha": True,
    "target_entropy": "auto",  # -dim(action_space)
    "buffer_size": 1000000,
    "batch_size": 256,
    "learning_starts": 10000,
    "hidden_dim": 256,
}
```

---

## Part 3: Reward Shaping Guide

### When to Reshape Reward (Instead of Tuning)

Tuning hyperparameters can't fix a fundamentally unlearnable reward signal. Consider reward shaping when:

| Signal | Meaning |
|--------|---------|
| Random agent gets 0 reward for 99%+ of episodes | Reward is too sparse — agent can't discover signal through random exploration |
| Optimal behavior requires a long sequence with no intermediate signal | Need intermediate rewards to guide learning |
| Reward has extreme scale (±10000) | Normalize or clip rewards |
| Multiple competing objectives | Need to balance reward components explicitly |

### Reward Design Principles

1. **Dense > Sparse**: provide signal at every step when possible
2. **Potential-based shaping**: F(s,s') = gamma × phi(s') - phi(s) guarantees same optimal policy (Ng et al. 1999; strictly holds for infinite horizon — for episodic tasks, set phi(terminal)=0)
3. **Don't reward the means, reward the ends**: reward "being close to goal" not "moving toward goal" (the latter can be exploited)
4. **Scale matters**: keep reward roughly in [-1, 1] or normalize dynamically
5. **Avoid reward hacking**: if the agent can get reward without doing what you want, it will

### Highway-Env Example

Default highway-env reward (simplified):

```python
reward = speed_reward * (1 - crashed)
# Problem: no penalty for near-misses, no reward for lane centering
```

Improved shaping:

```python
reward = (
    0.4 * normalized_speed          # encourage forward progress
    + 0.3 * lane_centering          # stay in lane center
    + 0.2 * headway_reward          # maintain safe following distance
    - 1.0 * crashed                 # heavy crash penalty
    - 0.1 * abs(steering_change)    # penalize jerky driving
)
```

### Common Reward Shaping Patterns

| Pattern | Formula | Use Case |
|---------|---------|----------|
| Distance-based | -distance_to_goal | Navigation tasks |
| Progress-based | distance_prev - distance_curr | Encourage movement toward goal |
| Time penalty | -0.01 per step | Encourage efficiency |
| Survival bonus | +1 per step alive | Balance-type tasks |
| Potential-based | gamma × phi(s') - phi(s) | Safe shaping that preserves optimal policy |

### References

- [Stable Baselines3 — PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Stable Baselines3 — SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [Stable Baselines3 — TD3](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html)
- [CleanRL — Implementation Details](https://docs.cleanrl.dev/)
- [Andrychowicz et al. — What Matters in On-Policy RL](https://arxiv.org/abs/2006.05990)
- [Henderson et al. — Deep RL that Matters](https://arxiv.org/abs/1709.06560)

---

# RL 超参数调优指南

## 第一部分：通用调参流程

### 调参前诊断清单

花时间调参之前，先确认这确实是调参问题：

| 检查项 | 方法 | 如果失败 |
|--------|------|----------|
| 环境正常 | 随机智能体偶尔能拿到非零奖励 | 修复环境设置，检查动作/观察空间 |
| 观察值正确 | 打印 obs 形状和范围，核对文档 | 修复预处理、归一化 |
| 奖励信号存在 | 画出随机策略的奖励曲线 | 奖励函数有问题，不是调参问题 |
| 损失在下降 | 查看 TensorBoard 的 loss 曲线 | 训练循环有 bug，不是超参数问题 |
| 梯度正常 | 检查梯度范数非零且不爆炸 | 修复网络初始化，检查是否有 detached tensors |
| Buffer 在填充 | 确认训练在积累足够样本后才开始 | 调整最小 buffer 大小或回合数 |
| 动作有效 | 智能体输出在环境动作空间内 | 修复动作裁剪/缩放 |

**原则：如果 loss 完全不下降，那是 bug，不是调参问题。**

### 通用调参优先级

按以下顺序调——后面的参数在前面错了时毫无意义：

```
1. 奖励设计     — 信号是否可学习？（见第三部分）
2. 学习率       — 单一影响最大的参数
3. 探索机制     — epsilon / 熵 / 噪声（取决于算法）
4. 网络大小     — 必须匹配问题复杂度
5. 批次与缓冲   — 样本效率和稳定性
6. 其他参数     — gamma, tau, clip range 等
```

### 症状速查表

| 症状 | 可能原因 | 第一步 |
|------|----------|--------|
| 奖励一直为 0 / 不涨 | LR 太高/太低，或探索太快/太慢 | 先在已知环境（CartPole）上验证 LR |
| 奖励先涨后崩 | LR 太大，目标网络滞后太大，或 PPO clip 太宽 | LR 减半，减小 tau / 增加 target update 频率 |
| Loss 爆炸 (NaN/Inf) | LR 太大，没有梯度裁剪，奖励尺度异常 | 加梯度裁剪，归一化奖励 |
| Loss 下降但奖励不涨 | Q 值高估，或网络在记忆 buffer | 用 Double DQN / 裁剪 critics，检查 buffer 多样性 |
| 奖励方差很大 | Buffer 太小，没有目标网络，或探索噪声太大 | 增大 buffer，验证目标网络，减小噪声 |
| 智能体卡在一个动作 | 探索衰减太快，或熵太低 | 放慢 epsilon 衰减，增大 entropy 系数 |
| 训练非常慢 | LR 太低，batch 太大，网络太小 | 增大 LR，减小 batch，增大 hidden dim |
| 策略在不同策略间震荡 | 策略 LR 太高，或 PPO epochs 太多 | 降低策略 LR，减少 PPO epochs，缩小 clip |

### 通用参数详解

#### 学习率 (`lr`)

**原理**：控制参数空间中的步长。太大 → 震荡/发散。太小 → 学习过慢/卡住。

| 范围 | 适用场景 |
|------|----------|
| 1e-4 | 复杂环境、微调、像素输入 |
| 3e-4 | 大多数算法的默认值（PPO、SAC、TD3） |
| 1e-3 | 简单环境、基于值的方法（DQN） |
| 3e-3 | 极简任务或初始探索 |

**诊断：**

| 症状 | 诊断 | 解决 |
|------|------|------|
| 损失剧烈震荡 | LR 太大 | 减半 |
| 1000+ 次更新后 loss 几乎不动 | LR 太小 | 翻倍 |
| 初期有效后来发散 | LR 在训练后期太大 | 使用调度器（线性衰减/余弦） |

**耦合关系**：batch size 翻倍时，LR 增加约 1.4-2 倍。（线性缩放规则为 2 倍；sqrt 缩放为 ~1.4 倍。RL 中 sqrt 缩放通常更稳定，1.5 倍是实用折中。）

#### 折扣因子 (`gamma`)

**原理**：有效视野 ≈ 1 / (1 - gamma)。智能体考虑约 H 步未来奖励。

| gamma | 视野 | 何时使用 |
|-------|------|----------|
| 0.9 | ~10 步 | 极短回合、即时奖励任务 |
| 0.95 | ~20 步 | 短回合 |
| 0.99 | ~100 步 | 大多数任务（默认） |
| 0.995 | ~200 步 | 长回合 |
| 0.999 | ~1000 步 | 超长回合、稀疏奖励 |

**经验法则**：设 gamma 使视野 > 回合长度。highway-env（约 30-50 步）用 0.99 即可。

#### 网络结构

| 问题 | 输入维度 | 推荐架构 |
|------|----------|----------|
| 简单 (CartPole) | < 10 | 2 层, 64 单元 |
| 中等 (Highway-env) | 10-100 | 2 层, 128-256 单元 |
| 复杂 (连续控制) | 10-100 | 2 层, 256 单元 |
| 像素输入 | 图像 | CNN 编码器 + 2 层 FC, 512 单元 |

**判断是否合适：**
- 太小：loss 卡在高位，奖励不涨（欠拟合）
- 太大：loss 看着正常但奖励很抖（对 buffer 过拟合）

**技巧：**
- 基于值的方法用 ReLU，连续动作空间的策略输出用 Tanh
- 连续控制中 Layer Normalization 有帮助
- 正交初始化略优于默认初始化

#### 梯度裁剪 (`max_grad_norm`)

| 值 | 效果 |
|----|------|
| 0.5 | 非常保守，防止所有尖峰但可能减慢学习 |
| 1.0 | 标准默认值（PPO, A2C） |
| 10.0 | 宽松，只拦截极端情况 |
| 无 | 不裁剪 — 高 LR 或稀疏奖励下有风险 |

**何时使用**：策略梯度方法必须用。DQN 可选（Huber loss 隐式裁剪了）。

#### 批次大小

| 大小 | 优点 | 缺点 |
|------|------|------|
| 32 | 频繁更新，早期学习快 | 梯度方差大 |
| 64 | 基于值方法的好选择 | — |
| 128-256 | PPO/A2C 标准（需足够轨迹） | — |
| 2048+ | 稳定的策略梯度（大规模 PPO） | 每次更新需要很多环境步 |

**On-policy**（PPO/A2C）：总 rollout buffer = num_envs × n_steps，然后被切分为 minibatch（如 8192 总样本 / 64 minibatch_size = 每 epoch 128 个 minibatch）。总 buffer 越大 → 梯度估计越稳定。

**Off-policy**（DQN/SAC/TD3）：从 buffer 采样。256 是典型默认值。

---

## 第二部分：分算法指南

---

### REINFORCE

#### 关键参数

| 参数 | 常见范围 | 作用 |
|------|----------|------|
| `lr` | 1e-4 ~ 3e-3 | 策略网络学习率 |
| `gamma` | 0.99 ~ 0.999 | 折扣因子 |
| `baseline` | True/False | 是否减去基线（平均回报）来降低方差 |
| `entropy_bonus` | 0.0 ~ 0.1 | 熵正则化系数 |
| `max_grad_norm` | 0.5 ~ 5.0 | 梯度裁剪阈值 |

#### 典型网络结构

```
Input (state_dim)
  → Linear(state_dim, 128) → ReLU
  → Linear(128, 128) → ReLU
  → Linear(128, action_dim) → Softmax
```

#### 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 回报方差极大 | 没有基线 | 启用基线减去 |
| 策略坍缩到单一动作 | 熵太低，LR 太大 | 加熵奖励 (0.01-0.05)，降低 LR |
| 收敛极慢 | 高方差，小批次 | 每次更新用更多轮，加基线 |
| 梯度爆炸 | 长回合 + 大奖励值 | 归一化回报，裁剪梯度 |

#### Quick Start

```python
config = {
    "lr": 1e-3,
    "gamma": 0.99,
    "baseline": True,
    "entropy_bonus": 0.01,
    "max_grad_norm": 1.0,
    "hidden_dim": 128,
}
```

---

### DQN

#### 关键参数

| 参数 | 常见范围 | 作用 |
|------|----------|------|
| `lr` | 1e-4 ~ 1e-3 | Q 网络学习率 |
| `gamma` | 0.9 ~ 0.999 | 折扣因子 |
| `epsilon_start` | 0.5 ~ 1.0 | 初始探索率 |
| `epsilon_end` | 0.01 ~ 0.1 | 最小探索率 |
| `epsilon_decay` | 0.99 ~ 0.999 | 逐轮衰减乘数 |
| `buffer_size` | 10000 ~ 1000000 | 经验回放池容量 |
| `batch_size` | 32 ~ 256 | 每次更新采样数 |
| `tau` | 0.001 ~ 0.01 | 目标网络软更新速率 |
| `hidden_dim` | 64 ~ 512 | 网络宽度 |

**Epsilon 衰减时间表**：到达 epsilon_end 所需轮数 ≈ `log(epsilon_end / epsilon_start) / log(epsilon_decay)`

| epsilon_decay | 到 0.01 的轮数 | 特点 |
|---------------|----------------|------|
| 0.99 | ~460 | 快 — 探索不充分有风险 |
| 0.995 | ~920 | 平衡 |
| 0.999 | ~4600 | 充分探索 |

#### 变体

| 变体 | 改变了什么 | 何时使用 |
|------|-----------|----------|
| Double DQN | 动作选择与评估解耦 | 总是用（几乎无成本，减少高估） |
| Dueling DQN | 将 Q 拆分为 V + A 两路 | 某些状态明显好/坏时 |
| 优先级回放 | 高 TD-error 转换被更频繁采样 | 稀疏奖励，但增加复杂度 |
| N-step Returns | 使用 n 步自举 | 加速传播，但 off-policy 下有偏 |

#### 典型网络结构

```
Input (state_dim)
  → Linear(state_dim, 128) → ReLU
  → Linear(128, 128) → ReLU
  → Linear(128, action_dim)  # 原始 Q 值，无激活函数
```

Dueling 变体：
```
Input (state_dim)
  → Linear(state_dim, 128) → ReLU
  ├→ Linear(128, 1)           # V(s)
  └→ Linear(128, action_dim)  # A(s,a)
  → Q = V + (A - mean(A))
```

#### 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 奖励一直为 0 | Epsilon 在 buffer 填满前衰减完 | 增大 epsilon_decay，延迟训练开始 |
| 奖励先涨后崩 | LR 太大，或 tau 太大 | LR 减半，tau 降到 0.001 |
| Loss 下降但奖励不涨 | Q 值高估 | 用 Double DQN，减小 LR |
| 奖励方差高 | Buffer 太小 | 增大 buffer_size 到 100k+ |
| 智能体只执行一个动作 | Epsilon 衰减太快 | 减慢衰减，提高 epsilon_end |

#### Quick Start

```python
config = {
    "lr": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "buffer_size": 50000,
    "batch_size": 64,
    "hidden_dim": 128,
    "tau": 0.005,
    "double_dqn": True,
}
```

---

### A2C（优势演员-评论家）

#### 关键参数

| 参数 | 常见范围 | 作用 |
|------|----------|------|
| `lr` | 1e-4 ~ 1e-3 | 演员/评论家共享或独立学习率 |
| `gamma` | 0.99 ~ 0.999 | 折扣因子 |
| `n_steps` | 5 ~ 128 | 自举前的步数（rollout 长度） |
| `entropy_coef` | 0.001 ~ 0.1 | 熵正则化权重 |
| `value_loss_coef` | 0.25 ~ 1.0 | 价值损失在总损失中的权重 |
| `max_grad_norm` | 0.5 ~ 5.0 | 梯度裁剪 |
| `num_envs` | 4 ~ 32 | 并行环境数（降低方差） |

#### 典型网络结构

共享主干：
```
Input (state_dim)
  → Linear(state_dim, 128) → ReLU
  → Linear(128, 128) → ReLU
  ├→ Linear(128, action_dim) → Softmax    # 策略头
  └→ Linear(128, 1)                        # 价值头
```

#### 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 策略早期坍缩 | 熵系数太低 | 增加到 0.05-0.1 |
| 价值损失主导训练 | value_loss_coef 太大 | 减到 0.25 |
| 方差大、不稳定 | 环境太少或 n_steps 太短 | 增加 num_envs，增大 n_steps |
| 学习慢 | LR 太低或 n_steps 太长（自举不频繁导致高偏差） | 增大 LR，减小 n_steps |
| 过早收敛 | 熵消失 | 使用熵系数调度（高起始，逐渐衰减） |

#### Quick Start

```python
config = {
    "lr": 7e-4,
    "gamma": 0.99,
    "n_steps": 5,
    "entropy_coef": 0.01,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,
    "num_envs": 16,
    "hidden_dim": 128,
}
```

---

### PPO（近端策略优化）

#### 关键参数

| 参数 | 常见范围 | 作用 |
|------|----------|------|
| `lr` | 1e-4 ~ 3e-4 | 学习率（通常线性衰减） |
| `gamma` | 0.99 ~ 0.999 | 折扣因子 |
| `gae_lambda` | 0.9 ~ 0.99 | GAE 偏差-方差权衡 |
| `clip_range` | 0.1 ~ 0.3 | PPO 裁剪 epsilon |
| `n_epochs` | 3 ~ 10 | 对收集的 rollout 遍历次数 |
| `n_steps` | 128 ~ 2048 | 每个环境的 rollout 长度 |
| `batch_size` | 32 ~ 512 | epoch 内的小批次大小 |
| `entropy_coef` | 0.0 ~ 0.05 | 熵奖励 |
| `value_loss_coef` | 0.5 ~ 1.0 | 价值函数损失权重 |
| `max_grad_norm` | 0.5 | 梯度裁剪 |
| `num_envs` | 4 ~ 128 | 并行环境数 |

**GAE lambda** 控制优势估计的偏差-方差：
- lambda=0：1-step TD 优势，A = r + gamma*V(s') - V(s)（低方差、高偏差）
- lambda=1：MC 优势，用完整回合回报减去基线（高方差、低偏差）
- lambda=0.95：好的默认值 — 混合多步回报

**Clip range** 限制每次更新的策略变化幅度：
- 0.1：保守更新（更稳定）
- 0.2：标准（默认）
- 0.3：激进更新（更快但不太稳定）

#### 典型网络结构

```
Input (state_dim)
  → Linear(state_dim, 256) → Tanh
  → Linear(256, 256) → Tanh
  ├→ Linear(256, action_dim)   # 策略 (连续时 + log_std)
  └→ Linear(256, 1)            # 价值

# 连续动作：
# 策略输出均值，log_std 是可学习参数
```

#### 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 策略不提升 | LR 太低，或 clip 太紧 | 增大 LR 到 3e-4，放宽 clip 到 0.2 |
| 训练不稳定 / 奖励崩塌 | epochs 太多，clip 太宽 | 减少 n_epochs 到 3-4，clip 到 0.1 |
| 策略震荡 | LR 太高，batch 太小 | 降低 LR，增加 n_steps × num_envs |
| 熵降到零 | 熵系数太低 | 增加到 0.01-0.05 |
| 价值函数不准 | 独立网络发散 | 增加 value_loss_coef，或共享主干 |
| 样本效率低 | n_steps 太短，环境太少 | 增大 rollout 长度和 num_envs |

#### Quick Start

```python
config = {
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "n_epochs": 10,
    "n_steps": 2048,
    "batch_size": 64,
    "entropy_coef": 0.01,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,
    "num_envs": 4,
    "hidden_dim": 256,
}
```

---

### TD3（Twin Delayed DDPG）

#### 关键参数

| 参数 | 常见范围 | 作用 |
|------|----------|------|
| `actor_lr` | 1e-4 ~ 3e-4 | 演员学习率 |
| `critic_lr` | 1e-4 ~ 3e-4 | 评论家学习率（通常 = actor_lr） |
| `gamma` | 0.98 ~ 0.99 | 折扣因子 |
| `tau` | 0.005 | 目标网络软更新 |
| `policy_delay` | 2 | 每 N 次 critic 更新后更新一次 actor |
| `noise_std` | 0.1 ~ 0.3 | 探索噪声（高斯） |
| `target_noise_std` | 0.2 | 目标策略平滑噪声 |
| `target_noise_clip` | 0.5 | 目标噪声裁剪范围 |
| `buffer_size` | 100000 ~ 1000000 | 经验回放池 |
| `batch_size` | 100 ~ 256 | 每次更新采样数 |
| `learning_starts` | 10000 ~ 25000 | 训练前随机探索步数 |

**TD3 相对 DDPG 的三个改进：**
1. 双 critic（取最小值 → 减少高估）
2. 延迟策略更新（policy_delay=2 → 更稳定）
3. 目标策略平滑（给目标动作加噪声 → Q 更平滑）

#### 典型网络结构

```
# Actor
Input (state_dim)
  → Linear(state_dim, 256) → ReLU
  → Linear(256, 256) → ReLU
  → Linear(256, action_dim) → Tanh  # 有界动作

# Critic (x2, 双 critic)
Input (state_dim + action_dim)
  → Linear(state_dim + action_dim, 256) → ReLU
  → Linear(256, 256) → ReLU
  → Linear(256, 1)
```

#### 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 智能体不探索 | 噪声太小，或 learning_starts 太短 | 增大 noise_std，增加 learning_starts |
| Q 值发散 | LR 太大，没有双 critic | 确保有双 critic，降低 LR |
| 动作抖动/有噪声 | 测试时噪声太大 | 评估时用确定性策略（不加噪声） |
| 收敛慢 | policy_delay 太高，LR 太低 | 试 policy_delay=2（默认），增大 LR |
| 性能停滞 | Buffer 太小，或探索噪声衰减了 | 增大 buffer，训练中保持噪声恒定 |

#### Quick Start

```python
config = {
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "policy_delay": 2,
    "noise_std": 0.1,
    "target_noise_std": 0.2,
    "target_noise_clip": 0.5,
    "buffer_size": 1000000,
    "batch_size": 256,
    "learning_starts": 25000,
    "hidden_dim": 256,
}
```

---

### SAC（软演员-评论家）

#### 关键参数

| 参数 | 常见范围 | 作用 |
|------|----------|------|
| `actor_lr` | 1e-4 ~ 3e-4 | 演员学习率 |
| `critic_lr` | 1e-4 ~ 3e-4 | 评论家学习率 |
| `alpha_lr` | 1e-4 ~ 3e-4 | 温度参数 (alpha) 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `tau` | 0.005 | 目标网络软更新 |
| `alpha` | 0.2（固定时） | 熵温度 |
| `auto_alpha` | True | 是否自动调节 alpha |
| `target_entropy` | -dim(A) | 自动调节的目标熵（连续） |
| `buffer_size` | 100000 ~ 1000000 | 经验回放池 |
| `batch_size` | 256 | 每次更新采样数 |
| `learning_starts` | 5000 ~ 25000 | 训练前随机步数 |

**Alpha（熵温度）**：
- Alpha 高 → 更多探索（策略更随机）
- Alpha 低 → 更多利用（策略更确定）
- 自动调节（推荐）：调整 alpha 以维持目标熵

**目标熵**：
- 连续：-dim(action_space) 是标准启发式（来自 SAC 论文）
- 离散：-log(1/|A|) × ratio（ratio ~0.98）— SAC-Discrete 实现中的常用启发式，非原论文内容

#### 典型网络结构

```
# Actor（输出分布）
Input (state_dim)
  → Linear(state_dim, 256) → ReLU
  → Linear(256, 256) → ReLU
  ├→ Linear(256, action_dim)  # 均值
  └→ Linear(256, action_dim)  # log_std（裁剪到 [-20, 2]）
  → 通过重参数化技巧采样
  → Tanh 压缩得到有界动作

# Critic (x2, 双 critic)
Input (state_dim + action_dim)
  → Linear(state_dim + action_dim, 256) → ReLU
  → Linear(256, 256) → ReLU
  → Linear(256, 1)
```

#### 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 探索过多，不收敛 | Alpha 太高 | 启用自动调节，或降低固定 alpha |
| 过早收敛 | Alpha 太低或下降太快 | 增大目标熵，或提高 alpha |
| Q 值发散到很大数值 | LR 太高，没有奖励缩放 | 降低 LR，归一化/裁剪奖励 |
| 动作饱和在边界 | Tanh 压缩 + 高 alpha | 降低目标熵，检查 log_std 边界 |
| 早期训练不稳定 | learning_starts 太低 | 增加到 10k-25k 以保证 buffer 多样性 |
| log 概率出现 NaN | log_std 无界 | 将 log_std 裁剪到 [-20, 2] |

#### Quick Start

```python
config = {
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "auto_alpha": True,
    "target_entropy": "auto",  # -dim(action_space)
    "buffer_size": 1000000,
    "batch_size": 256,
    "learning_starts": 10000,
    "hidden_dim": 256,
}
```

---

## 第三部分：Reward Shaping 指南

### 什么时候该改 Reward（而不是调参数）

调超参数无法修复一个根本不可学的奖励信号。以下情况考虑 reward shaping：

| 信号 | 含义 |
|------|------|
| 随机智能体 99%+ 的回合奖励为 0 | 奖励太稀疏 — 智能体通过随机探索无法发现信号 |
| 最优行为需要长序列且无中间信号 | 需要中间奖励引导学习 |
| 奖励尺度极端（±10000） | 归一化或裁剪奖励 |
| 多个相互竞争的目标 | 需要显式平衡奖励分量 |

### 奖励设计原则

1. **稠密优于稀疏**：尽可能每步提供信号
2. **基于势函数的塑形**：F(s,s') = gamma × phi(s') - phi(s) 保证最优策略不变（Ng et al. 1999；严格来说仅在无限 horizon 下成立 — 有限回合任务中需设 phi(terminal)=0）
3. **奖励目的而非手段**：奖励"离目标近"而非"朝目标移动"（后者可被利用）
4. **尺度很重要**：保持奖励大致在 [-1, 1]，或动态归一化
5. **避免奖励黑客**：如果智能体能在不完成任务的情况下获得奖励，它一定会这么做

### Highway-Env 示例

默认 highway-env 奖励（简化）：

```python
reward = speed_reward * (1 - crashed)
# 问题：没有对险些碰撞的惩罚，没有车道居中奖励
```

改进的塑形：

```python
reward = (
    0.4 * normalized_speed          # 鼓励前进
    + 0.3 * lane_centering          # 保持车道居中
    + 0.2 * headway_reward          # 保持安全跟车距离
    - 1.0 * crashed                 # 碰撞重罚
    - 0.1 * abs(steering_change)    # 惩罚急打方向盘
)
```

### 常见 Reward Shaping 模式

| 模式 | 公式 | 适用场景 |
|------|------|----------|
| 基于距离 | -distance_to_goal | 导航任务 |
| 基于进步 | distance_prev - distance_curr | 鼓励朝目标移动 |
| 时间惩罚 | -0.01 每步 | 鼓励效率 |
| 存活奖励 | +1 每步存活 | 平衡类任务 |
| 基于势函数 | gamma × phi(s') - phi(s) | 安全塑形，保持最优策略不变 |

### 参考来源

- [Stable Baselines3 — PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Stable Baselines3 — SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [Stable Baselines3 — TD3](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html)
- [CleanRL — Implementation Details](https://docs.cleanrl.dev/)
- [Andrychowicz et al. — What Matters in On-Policy RL](https://arxiv.org/abs/2006.05990)
- [Henderson et al. — Deep RL that Matters](https://arxiv.org/abs/1709.06560)
