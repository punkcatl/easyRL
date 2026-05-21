# DQN Hyperparameter Tuning Guide

## Parameter Overview

| Parameter | Current Value | Typical Range | Role |
|-----------|--------------|---------------|------|
| `lr` | 1e-3 | 1e-4 ~ 3e-3 | Network weight update step size |
| `gamma` | 0.99 | 0.9 ~ 0.999 | Future reward discount |
| `epsilon_start` | 1.0 | 0.5 ~ 1.0 | Initial exploration rate |
| `epsilon_end` | 0.01 | 0.01 ~ 0.1 | Minimum exploration rate |
| `epsilon_decay` | 0.995 | 0.99 ~ 0.999 | Exploration decay speed |
| `buffer_size` | 50000 | 10000 ~ 1000000 | Replay buffer capacity |
| `batch_size` | 64 | 32 ~ 256 | Samples per update |
| `hidden_dim` | 128 | 64 ~ 512 | Network width |
| `tau` | 0.005 | 0.001 ~ 0.01 | Target network soft update rate |

## Tuning Priority

Not all parameters are equally important. Tune in this order:

1. **Learning rate** — most impactful, wrong value means no learning at all
2. **Epsilon decay** — controls explore/exploit balance
3. **Network size** (hidden_dim) — must match problem complexity
4. **Batch size & buffer size** — affects sample efficiency and stability
5. **Gamma, tau** — usually safe at default values

## Per-Parameter Tuning Advice

### Learning Rate (`lr`)

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Loss oscillates wildly, reward never improves | lr too high | Reduce to 1e-4 |
| Loss decreases very slowly, reward flat for hundreds of episodes | lr too low | Increase to 3e-3 |
| Training works initially then diverges | lr too high for later training | Use learning rate scheduler |

Rule of thumb: start with 1e-3, if unstable halve it, if too slow double it.

### Gamma (`gamma`)

| Value | Effect | When to use |
|-------|--------|-------------|
| 0.9 | Short-sighted, ~10 steps horizon | Simple tasks, short episodes |
| 0.99 | Long-sighted, ~100 steps horizon | Most tasks (default) |
| 0.999 | Very long-sighted, ~1000 steps | Long episodes with sparse rewards |

Effective horizon ≈ 1 / (1 - gamma). For highway-env with episodes of ~30-50 steps, 0.99 is appropriate.

### Epsilon Decay (`epsilon_start`, `epsilon_end`, `epsilon_decay`)

```
Episodes to reach epsilon_end ≈ log(epsilon_end / epsilon_start) / log(epsilon_decay)

Current: log(0.01 / 1.0) / log(0.995) ≈ 920 episodes
```

| Decay Speed | epsilon_decay | Episodes to 0.01 | Behavior |
|-------------|--------------|-------------------|----------|
| Fast | 0.99 | ~460 | Exploit early, risky if environment not explored |
| Medium | 0.995 | ~920 | Good balance (current) |
| Slow | 0.999 | ~4600 | Thorough exploration, needs more episodes |

Tips:
- If reward stays flat early on: exploration is fine, problem is elsewhere
- If agent gets stuck in suboptimal behavior: slow down decay or increase epsilon_end
- `epsilon_end=0.01` means 1% random actions even after convergence — prevents total stagnation

### Buffer Size (`buffer_size`)

| Size | Pros | Cons |
|------|------|------|
| Small (5000) | Adapts quickly to new policy | Unstable, forgets good past experiences |
| Medium (50000) | Good balance | — |
| Large (500000+) | Very stable training | Slow to adapt, high memory usage |

Rule of thumb: buffer should hold 10-50x more transitions than one episode length. For 30-step episodes, 50000 is generous.

Warning: if buffer too large relative to training time, most samples are from random (early) policy — agent learns noise.

### Batch Size (`batch_size`)

| Size | Pros | Cons |
|------|------|------|
| 32 | More frequent updates, faster early learning | Higher gradient variance |
| 64 | Good balance (current) | — |
| 256 | Stable gradients | Slower per-episode training, needs larger buffer |

Batch size and learning rate are coupled: if you double batch_size, consider increasing lr by ~1.5x.

### Hidden Dim (`hidden_dim`)

| Problem Complexity | Recommended | Example |
|-------------------|-------------|---------|
| Simple (CartPole, GridWorld) | 64 | 4-dim state, 2 actions |
| Medium (Highway-env) | 128 | 25-dim state, 5 actions |
| Complex (Atari pixel input) | 256-512 | High-dim state, many actions |

Signs of wrong size:
- Too small: loss plateaus high, reward never improves — network can't represent Q-function
- Too large: loss looks fine but reward is noisy — overfitting to buffer samples

### Tau (`tau`)

| Value | Target net lag | Stability | Speed |
|-------|---------------|-----------|-------|
| 0.001 | ~1000 steps | Very stable | Slow |
| 0.005 | ~200 steps | Stable (current) | Moderate |
| 0.01 | ~100 steps | Less stable | Fast |
| 1.0 | Hard update every step | Unstable | — |

Smaller tau = more stable but slower convergence. Only adjust if training is unstable (reduce tau) or too slow (increase tau).

## Common Problems and Solutions

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| Reward stays at 0 or negative | Epsilon decays too fast before buffer fills | Increase epsilon_decay to 0.999, or delay training until buffer has enough samples |
| Reward improves then collapses | lr too high, or tau too large | Halve lr, reduce tau to 0.001 |
| Loss decreases but reward doesn't improve | Q-values overestimated | Reduce lr, try Double DQN |
| Training very slow | lr too low, or batch_size too large | Increase lr, reduce batch_size |
| High variance in reward | Buffer too small, or no target network | Increase buffer_size, check tau isn't 1.0 |

## Sanity Checks Before Tuning

Before spending time on hyperparameters, verify:

1. **Is the environment working?** — Random agent should get non-zero reward sometimes
2. **Is the buffer filling?** — Training only starts after `buffer_size >= batch_size`
3. **Is loss decreasing?** — If not, it's a bug, not a tuning problem
4. **Is epsilon actually decaying?** — Check TensorBoard epsilon curve

## Quick Start Recipe

For a new environment, start with:

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
}
```

Then adjust in order:
1. Run 200 episodes — is loss decreasing? If not → fix lr
2. Run 500 episodes — is reward trending up? If not → check epsilon decay and hidden_dim
3. If training is unstable → reduce lr, reduce tau
4. If training is too slow → increase lr, increase batch_size

---

# DQN 超参数调优指南

## 参数总览

| 参数 | 当前值 | 常见范围 | 作用 |
|------|--------|----------|------|
| `lr` | 1e-3 | 1e-4 ~ 3e-3 | 网络权重更新步长 |
| `gamma` | 0.99 | 0.9 ~ 0.999 | 未来奖励折扣因子 |
| `epsilon_start` | 1.0 | 0.5 ~ 1.0 | 初始探索率 |
| `epsilon_end` | 0.01 | 0.01 ~ 0.1 | 最小探索率 |
| `epsilon_decay` | 0.995 | 0.99 ~ 0.999 | 探索率衰减速度 |
| `buffer_size` | 50000 | 10000 ~ 1000000 | 经验回放池容量 |
| `batch_size` | 64 | 32 ~ 256 | 每次更新采样数 |
| `hidden_dim` | 128 | 64 ~ 512 | 网络宽度 |
| `tau` | 0.005 | 0.001 ~ 0.01 | 目标网络软更新速率 |

## 调参优先级

并非所有参数都同等重要，按以下顺序调：

1. **学习率** — 影响最大，错了就完全学不动
2. **Epsilon 衰减** — 控制探索/利用平衡
3. **网络大小**（hidden_dim） — 必须匹配问题复杂度
4. **Batch size 和 buffer size** — 影响样本效率和稳定性
5. **Gamma、tau** — 通常默认值就够

## 逐参数调优建议

### 学习率 (`lr`)

| 症状 | 诊断 | 解决 |
|------|------|------|
| 损失剧烈震荡，奖励始终不涨 | lr 太大 | 降到 1e-4 |
| 损失下降极慢，奖励几百轮不动 | lr 太小 | 升到 3e-3 |
| 初期有效后来发散 | lr 在训练后期太大 | 使用学习率调度器 |

经验法则：从 1e-3 开始，不稳定就减半，太慢就翻倍。

### Gamma (`gamma`)

| 值 | 效果 | 适用场景 |
|----|------|----------|
| 0.9 | 短视，约看 10 步 | 简单任务、短回合 |
| 0.99 | 长视，约看 100 步 | 大多数任务（默认） |
| 0.999 | 极长视，约看 1000 步 | 长回合、稀疏奖励 |

有效视野 ≈ 1 / (1 - gamma)。highway-env 每轮约 30-50 步，0.99 合适。

### Epsilon 衰减 (`epsilon_start`, `epsilon_end`, `epsilon_decay`)

```
到达 epsilon_end 所需轮数 ≈ log(epsilon_end / epsilon_start) / log(epsilon_decay)

当前: log(0.01 / 1.0) / log(0.995) ≈ 920 轮
```

| 衰减速度 | epsilon_decay | 到 0.01 的轮数 | 行为 |
|----------|--------------|----------------|------|
| 快 | 0.99 | ~460 | 早期就开始利用，如果环境没探索够会卡住 |
| 中 | 0.995 | ~920 | 好的平衡（当前） |
| 慢 | 0.999 | ~4600 | 充分探索，需要更多轮数 |

技巧：
- 奖励早期一直平坦：探索没问题，问题在别处
- 智能体卡在次优行为：放慢衰减或提高 epsilon_end
- `epsilon_end=0.01` 表示收敛后仍有 1% 随机动作 — 防止完全僵化

### 经验回放池大小 (`buffer_size`)

| 大小 | 优点 | 缺点 |
|------|------|------|
| 小 (5000) | 快速适应新策略 | 不稳定，遗忘好的历史经验 |
| 中 (50000) | 好的平衡 | — |
| 大 (500000+) | 训练非常稳定 | 适应慢，内存占用高 |

经验法则：buffer 应存放单轮步数的 10-50 倍。对 30 步的回合来说，50000 绰绰有余。

注意：如果 buffer 相对训练时长太大，大部分样本来自早期随机策略 — 智能体在学噪声。

### 批次大小 (`batch_size`)

| 大小 | 优点 | 缺点 |
|------|------|------|
| 32 | 更新更频繁，早期学习快 | 梯度方差大 |
| 64 | 好的平衡（当前） | — |
| 256 | 梯度稳定 | 每轮训练慢，需要更大 buffer |

Batch size 和学习率是耦合的：batch_size 翻倍时，lr 可以提高约 1.5 倍。

### 隐藏层维度 (`hidden_dim`)

| 问题复杂度 | 推荐 | 例子 |
|-----------|------|------|
| 简单 (CartPole, GridWorld) | 64 | 4 维状态，2 个动作 |
| 中等 (Highway-env) | 128 | 25 维状态，5 个动作 |
| 复杂 (Atari 像素输入) | 256-512 | 高维状态，多动作 |

判断网络大小是否合适：
- 太小：loss 卡在高位，奖励不涨 — 网络表达能力不够
- 太大：loss 看着正常但奖励很抖 — 对 buffer 样本过拟合

### Tau (`tau`)

| 值 | 目标网络滞后 | 稳定性 | 速度 |
|----|-------------|--------|------|
| 0.001 | ~1000 步 | 非常稳定 | 慢 |
| 0.005 | ~200 步 | 稳定（当前） | 适中 |
| 0.01 | ~100 步 | 较不稳定 | 快 |
| 1.0 | 每步硬更新 | 不稳定 | — |

tau 越小 = 越稳定但收敛越慢。只在训练不稳定（减小 tau）或太慢（增大 tau）时调整。

## 常见问题与解决方案

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 奖励一直为 0 或负数 | epsilon 在 buffer 填满前衰减太快 | 增大 epsilon_decay 到 0.999，或延迟训练直到 buffer 有足够样本 |
| 奖励先涨后崩 | lr 太大，或 tau 太大 | lr 减半，tau 降到 0.001 |
| 损失下降但奖励不涨 | Q 值被高估 | 减小 lr，尝试 Double DQN |
| 训练非常慢 | lr 太低，或 batch_size 太大 | 增大 lr，减小 batch_size |
| 奖励方差很大 | buffer 太小，或没有目标网络 | 增大 buffer_size，检查 tau 是否为 1.0 |

## 调参前的健全性检查

花时间调参之前，先确认：

1. **环境是否正常？** — 随机智能体偶尔能拿到非零奖励
2. **Buffer 是否在填充？** — 训练只在 `buffer_size >= batch_size` 后才开始
3. **损失是否在下降？** — 如果不是，这是 bug 不是调参问题
4. **Epsilon 是否在衰减？** — 查看 TensorBoard 的 epsilon 曲线

## 快速起步配方

面对一个新环境，先用以下默认值：

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
}
```

然后按顺序调整：
1. 跑 200 轮 — loss 在下降吗？如果不是 → 调 lr
2. 跑 500 轮 — 奖励有上升趋势吗？如果不是 → 检查 epsilon decay 和 hidden_dim
3. 如果训练不稳定 → 减小 lr，减小 tau
4. 如果训练太慢 → 增大 lr，增大 batch_size
