# Target Network Update Strategies

## Hard Update

$$\theta^{-} \leftarrow \theta \quad \text{(every } N \text{ steps)}$$

- Copy all parameters from Q-network to target network every N steps
- Between copies, target network is completely frozen
- Introduces a hyperparameter N (update frequency)

## Soft Update (Polyak Averaging)

$$\theta^{-} \leftarrow \tau \theta + (1 - \tau) \theta^{-} \quad \text{(every step)}$$

- Blend a small fraction ($\tau$, typically 0.001~0.01) of Q-network into target network each step
- Target network continuously tracks Q-network but with significant lag
- Introduces a hyperparameter $\tau$ (blending rate)

## Comparison

| Property | Hard Update | Soft Update |
|----------|-------------|-------------|
| Target stability | Stable between copies, sudden jump at copy | Always slowly changing, no jumps |
| Hyperparameter | N (update frequency) | $\tau$ (blending rate) |
| Sensitivity | Sensitive to N choice | Robust ($\tau$ in 0.001~0.01 works well) |
| Staleness | Target drifts stale before next copy | Always near-current |
| Implementation | `target.load_state_dict(q.state_dict())` | `p_tgt = τ * p + (1-τ) * p_tgt` |
| Origin | DQN (2015) | DDPG (2016), now standard |

## Behavior Over Time

```
Hard Update (N=100):

Target params: ████████████████░░░░░░░░░░░░░░░░████████████████
                ^copy          frozen            ^copy

Soft Update (τ=0.005):

Target params: ─────────────────────────────────────────────────
               (smooth, continuous change every step)
```

## Will the Two Networks Ever Be Equal Under Soft Update?

At initialization, they are identical (`load_state_dict` copies all parameters). Once training starts, Q-network changes via `optimizer.step()` every step, while the target network only absorbs 0.5% of those changes. After that, they **never become equal again** — unless training fully converges (gradients reach zero and Q-network stops changing), at which point the target network exponentially approaches the Q-network.

**How large is the gap?**

The target network is essentially an exponential moving average of the Q-network's historical parameters, lagging by approximately `1/tau` steps. With tau=0.005, the target network roughly reflects the Q-network from ~200 steps ago.

- Early training: Q-network changes rapidly, gap is large
- Late training: Q-network converges, gap shrinks

**Does the gap hurt training?**

The gap is the **design intent**, not a side effect:

| tau value | Gap size | Effect |
|-----------|----------|--------|
| tau = 1.0 | No gap (equivalent to no target network) | Unstable: TD target fluctuates with Q-network |
| tau = 0.005 | ~200 steps lag | Sweet spot: stable enough without being stale |
| tau = 0.0001 | ~10000 steps lag | Too stale: learning from outdated information, slow convergence |

The gap provides training stability — the target network acts like a "referee who doesn't change their mind easily", preventing the Q-network from chasing its own shifting predictions.

Hard update has the same principle: the target is frozen between copies (gap exists), then the gap suddenly drops to zero at copy time. Soft update keeps the gap smooth and continuous.

## Why Soft Update is Generally Preferred

1. No sudden target jumps → smoother loss landscape
2. One less hyperparameter to tune carefully
3. Target is never "stale" — always reflects recent learning
4. Works well across different environments without adjustment

---

# 目标网络更新策略

## 硬更新

$$\theta^{-} \leftarrow \theta \quad \text{（每 } N \text{ 步）}$$

- 每隔 N 步，把 Q 网络的全部参数拷贝到目标网络
- 两次拷贝之间，目标网络完全冻结
- 引入超参数 N（更新频率）

## 软更新（Polyak 平均）

$$\theta^{-} \leftarrow \tau \theta + (1 - \tau) \theta^{-} \quad \text{（每步）}$$

- 每步将 Q 网络的一小部分（$\tau$，通常 0.001~0.01）混入目标网络
- 目标网络持续跟踪 Q 网络，但有显著滞后
- 引入超参数 $\tau$（混合比率）

## 对比

| 特性 | 硬更新 | 软更新 |
|------|--------|--------|
| 目标稳定性 | 两次拷贝间稳定，拷贝时突变 | 始终缓慢变化，无突变 |
| 超参数 | N（更新频率） | $\tau$（混合比率） |
| 敏感度 | 对 N 的选择敏感 | 鲁棒（0.001~0.01 范围内都好使） |
| 过时程度 | 拷贝前目标会变得过时 | 始终接近最新 |
| 实现方式 | `target.load_state_dict(q.state_dict())` | `p_tgt = τ * p + (1-τ) * p_tgt` |
| 起源 | DQN (2015) | DDPG (2016)，现为主流 |

## 随时间变化的行为

```
硬更新 (N=100):

目标参数: ████████████████░░░░░░░░░░░░░░░░████████████████
           ^拷贝          冻结期            ^拷贝

软更新 (τ=0.005):

目标参数: ─────────────────────────────────────────────────
          （平滑、连续，每步都变化）
```

## 软更新下两个网络会相等吗？

初始化时两者完全相等（`load_state_dict` 复制过去）。训练开始后，Q 网络每步被 `optimizer.step()` 修改，目标网络只混入 0.5%，从此**永远不会再相等**——除非训练完全收敛（梯度为零，Q 网络不再变化），目标网络才会指数级逼近 Q 网络。

**差异有多大？**

目标网络本质上是 Q 网络历史参数的指数移动平均，大约滞后 `1/tau` 步。当 tau=0.005 时，目标网络大致反映 ~200 步之前的 Q 网络。

- 训练初期：Q 网络变化快，差异大
- 训练后期：Q 网络趋于收敛，差异逐渐缩小

**差异会影响训练吗？**

差异是**设计目的**，不是副作用：

| tau 值 | 差异程度 | 效果 |
|--------|---------|------|
| tau = 1.0 | 无差异（相当于没有目标网络） | 不稳定：TD 目标随 Q 网络同步抖动 |
| tau = 0.005 | 滞后约 200 步 | 平衡点：目标够稳定又不太过时 |
| tau = 0.0001 | 滞后约 10000 步 | 太过时：学的是旧信息，收敛慢 |

差异提供了训练稳定性——目标网络像一个"不轻易改变主意的裁判"，不会因为 Q 网络一次更新就跟着剧烈变化。

硬更新也是同样道理：两次拷贝之间目标网络冻结（差异存在），拷贝瞬间差异突然归零。软更新的差异始终平滑存在。

## 为什么通常优先使用软更新

1. 没有突然的目标跳变 → 更平滑的损失面
2. 少调一个敏感的超参数
3. 目标永远不会"过时"——始终反映最近的学习成果
4. 不同环境下无需调整即可良好工作
