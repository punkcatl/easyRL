# GAE (Generalized Advantage Estimation) Derivation

## 1. Starting Point: TD Error

The single-step TD error is defined as:

$$\delta_t = \underbrace{r_t + \gamma V(s_{t+1})}_{\text{TD target: revised estimate of } V(s_t) \text{ after one step}} - \underbrace{V(s_t)}_{\text{baseline: estimate before acting}}$$

$\delta_t$ is a **biased estimate** of the advantage function $A(s_t, a_t)$ — it only looks one step into the future. The bias comes from using $V(s_{t+1})$ to approximate all future returns beyond $t+1$; when $V$ is inaccurate, this approximation introduces error.

## 2. Multi-Step Advantage Estimates

Instead of looking one step ahead, we can look $n$ steps:

$$\begin{aligned}
&\hat{A}_t^{(1)} = \delta_t                                       &&= r_t + \gamma V(s_{t+1}) - V(s_t) \\
&\hat{A}_t^{(2)} = \delta_t + \gamma\delta_{t+1}                  &&= r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t) \\
&\hat{A}_t^{(3)} = \delta_t + \gamma\delta_{t+1} + \gamma^2\delta_{t+2} &&= r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 V(s_{t+3}) - V(s_t)
\end{aligned}$$

General form:

$$\hat{A}_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k \delta_{t+k}$$

<details>
<summary>Verification by expanding δ terms (telescoping cancellation of V terms)</summary>

$$\hat{A}_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k \delta_{t+k} = \sum_{k=0}^{n-1} \gamma^k \left[ r_{t+k} + \gamma V(s_{t+k+1}) - V(s_{t+k}) \right]$$

Split the sum into reward terms and value terms:

$$= \underbrace{\sum_{k=0}^{n-1} \gamma^k r_{t+k}}_{\text{discounted rewards}} + \underbrace{\sum_{k=0}^{n-1} \gamma^k \left[\gamma V(s_{t+k+1}) - V(s_{t+k})\right]}_{\text{value terms (will telescope)}}$$

Expand the second sum (the value terms) for $n=3$ as an example. Note that $\gamma^k \cdot \gamma V(s_{t+k+1}) = \gamma^{k+1} V(s_{t+k+1})$:

$$\begin{aligned}
&k=0: \quad \gamma^1 V(s_{t+1}) - \gamma^0 V(s_t)     &&= +\gamma V(s_{t+1})   &&- V(s_t) \\
&k=1: \quad \gamma^2 V(s_{t+2}) - \gamma^1 V(s_{t+1}) &&= +\gamma^2 V(s_{t+2}) &&- \gamma V(s_{t+1}) \\
&k=2: \quad \gamma^3 V(s_{t+3}) - \gamma^2 V(s_{t+2}) &&= +\gamma^3 V(s_{t+3}) &&- \gamma^2 V(s_{t+2})
\end{aligned}$$

Each row's positive term cancels with the next row's negative term (**telescoping**), leaving only the very first negative ($-V(s_t)$) and the very last positive ($\gamma^3 V(s_{t+3})$):

$$= \gamma^3 V(s_{t+3}) - V(s_t) = \gamma^n V(s_{t+n}) - V(s_t)$$

The same telescoping holds for arbitrary $n$:

$$\sum_{k=0}^{n-1} \gamma^k \left[\gamma V(s_{t+k+1}) - V(s_{t+k})\right] = \gamma^n V(s_{t+n}) - V(s_t)$$

</details>

Conclusion: $\hat{A}_t^{(n)} = G_t^{(n)} - V(s_t)$, where $G_t^{(n)} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})$ is the $n$-step return, and $V(s_t)$ is the baseline (current estimate of state value).

## 3. The Bias-Variance Trade-off

| $n$ | Property |
|-----|----------|
| $n=1$ | Low variance, high bias (only one step; error is large when $V$ is inaccurate) |
| $n=\infty$ | High variance, low bias (Monte Carlo — waits for the full trajectory, noisy) |

The core trade-off is **bootstrap vs. real rewards**. "Bootstrap" means using the learned $V$ to substitute for unobserved future returns (e.g., $\gamma V(s_{t+1})$ in n=1). More bootstrap → fewer random variables → low variance, but $V$ is approximate so bias creeps in. More real reward steps → no approximation → low bias, but each $r_t$ is random (affected by stochastic policy and environment transitions), so noise accumulates.

**GAE's insight: don't pick a single $n$ — take an exponentially weighted average over all $n$.** Use $n=1, 2, 3, \ldots$ simultaneously, but give smaller $n$ higher weight, with the weight decaying as $\lambda^{n-1}$. Why decay? Because larger $n$ accumulates more real reward steps, and each step introduces noise — so larger $n$ has higher variance. Downweighting them suppresses variance while still retaining some low-bias signal. Tuning $\lambda$ controls this trade-off.

## 4. GAE Definition: Exponential Weighting

$$\hat{A}_t^{\text{GAE}} = (1-\lambda) \left[ \hat{A}_t^{(1)} + \lambda \hat{A}_t^{(2)} + \lambda^2 \hat{A}_t^{(3)} + \cdots \right]$$

where $\lambda \in [0, 1]$ is the decay weight, and $(1-\lambda)$ normalizes so weights sum to 1. The weight assigned to the $n$-th term $\hat{A}_t^{(n)}$ is $(1-\lambda)\lambda^{n-1}$, so $n=1$ gets weight $(1-\lambda)$, $n=2$ gets $(1-\lambda)\lambda$, etc.:

$$\text{Sum of weights} = (1-\lambda)(1 + \lambda + \lambda^2 + \cdots) = (1-\lambda) \cdot \frac{1}{1-\lambda} = 1 \quad \checkmark$$

Here we used the geometric series formula: when $|\lambda| < 1$, $1 + \lambda + \lambda^2 + \cdots = \frac{1}{1-\lambda}$ (proof: let $S$ = the sum, then $S - \lambda S = 1$, so $S = \frac{1}{1-\lambda}$).

Substituting $\hat{A}_t^{(n)}$ definitions:

$$\hat{A}_t^{\text{GAE}} = (1-\lambda)\left[\delta_t + \lambda(\delta_t + \gamma\delta_{t+1}) + \lambda^2(\delta_t + \gamma\delta_{t+1} + \gamma^2\delta_{t+2}) + \cdots\right]$$

By expanding and collecting terms by $\delta_{t+k}$, this simplifies to:

$$\boxed{\hat{A}_t^{\text{GAE}} = \sum_{k=0}^{\infty} (\gamma\lambda)^k \delta_{t+k} = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots}$$

## 5. Deriving the Recursive Formula

From the summation form:

$$\begin{aligned}
&\hat{A}_t     = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots \\
&\hat{A}_{t+1} = \delta_{t+1} + (\gamma\lambda)\delta_{t+2} + \cdots
\end{aligned}$$

Observe:

$$\hat{A}_t = \delta_t + (\gamma\lambda)\left[\delta_{t+1} + (\gamma\lambda)\delta_{t+2} + \cdots\right] = \delta_t + (\gamma\lambda) \hat{A}_{t+1}$$

**The recursive formula:**

$$\boxed{\hat{A}_t = \delta_t + \gamma\lambda \cdot \hat{A}_{t+1}}$$

Boundary condition: after the trajectory ends, there is no future advantage, so $\hat{A}_{T+1} = 0$. This gives $\hat{A}_T = \delta_T + \gamma\lambda \cdot 0 = \delta_T$.

## 6. Implementation

Compute backwards from $T$ to $0$. Input `td_delta` is a tensor of shape `(T,)` where each element is $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$:

```python
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:           # from T to 0
        advantage = delta + gamma * lmbda * advantage   # A_t = delta_t + gamma*lambda * A_{t+1}
        advantage_list.append(advantage)
    advantage_list.reverse()               # flip back to chronological order
    return torch.tensor(advantage_list, dtype=torch.float)
```

Why iterate backwards? Because $\hat{A}_t$ depends on $\hat{A}_{t+1}$ — you can't compute earlier values without knowing later ones first. Starting from the end, the boundary condition $\hat{A}_{T+1} = 0$ (no future after the trajectory ends) gives you the first known value. Initial `advantage = 0` is this boundary: the variable `advantage` represents "the next step's advantage", and the first iteration processes step $T$ whose "next step" is $T+1$ — the trajectory has ended, no future TD errors exist, so it's $0$. Each subsequent iteration moves one step earlier.

**Note:** This implementation assumes a complete trajectory. If the episode is truncated (not terminated), the boundary should be $\hat{A}_{T+1} = \gamma\lambda \cdot V(s_{T+1})$ instead of $0$, since future returns still exist beyond the truncation point.

## 7. Verifying the Two Extremes

**$\lambda = 0$:**

$$\hat{A}_t = \delta_t + 0 \cdot \hat{A}_{t+1} = \delta_t$$

Degenerates to single-step TD error. High bias (relies on $V$ accuracy), low variance.

**$\lambda = 1$:**

$$\begin{aligned}
&\hat{A}_t = \delta_t + \gamma \hat{A}_{t+1} \\
&\phantom{\hat{A}_t} = \delta_t + \gamma\delta_{t+1} + \gamma^2\delta_{t+2} + \cdots \\
&\phantom{\hat{A}_t} = \sum_{k=0}^{\infty}\gamma^k\delta_{t+k} = G_t - V(s_t)
\end{aligned}$$

The last step uses the result from Section 2: when $n \to \infty$, $\hat{A}_t^{(\infty)} = G_t - V(s_t)$ (the terminal bootstrap $\gamma^\infty V(s_\infty) \to 0$).

Degenerates to Monte Carlo advantage estimate. Low bias, high variance.

## 8. Intuition

| $\lambda$ | Formula | Name | Bias | Variance |
|-----------|---------|------|------|----------|
| $0$ | $\hat{A}_t = \delta_t$ | Single-step TD | High | Low |
| $0.95$ | $\hat{A}_t = \sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k}$ | GAE (balanced) | Medium | Medium |
| $1$ | $\hat{A}_t = G_t - V(s_t)$ | Monte Carlo | Low | High |

With $\lambda = 0.95$: nearby deltas have high weight ($0.95^0=1,\ 0.95^1=0.95,\ 0.95^2\approx0.90, \ldots$), distant ones decay exponentially. The estimate primarily trusts the next few TD errors but doesn't completely ignore distant ones.

## 9. Role in PPO / Actor-Critic

$$\begin{aligned}
&\text{Collect episode} \\
&\longrightarrow \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \\
&\longrightarrow \hat{A}_t = \sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k} \\
&\longrightarrow \text{Update policy with } \hat{A}_t
\end{aligned}$$

- $\hat{A}_t > 0$: this action is better than average $\rightarrow$ increase its probability
- $\hat{A}_t < 0$: this action is worse than average $\rightarrow$ decrease its probability

---

# GAE（广义优势估计）推导

## 1. 起点：TD 误差

单步 TD 误差定义：

$$\delta_t = \underbrace{r_t + \gamma V(s_{t+1})}_{\text{TD target：走一步后对 } V(s_t) \text{ 的修正}} - \underbrace{V(s_t)}_{\text{baseline：走之前的估计}}$$

$\delta_t$ 是对优势函数 $A(s_t, a_t)$ 的一个**有偏估计**——它只看了一步未来。偏差来源于用 $V(s_{t+1})$ 近似 $t+1$ 之后的所有未来回报；当 $V$ 不准时，这个近似就会引入误差。

## 2. 多步优势估计

如果不只看一步，而是看 $n$ 步：

$$\begin{aligned}
&\hat{A}_t^{(1)} = \delta_t                                       &&= r_t + \gamma V(s_{t+1}) - V(s_t) \\
&\hat{A}_t^{(2)} = \delta_t + \gamma\delta_{t+1}                  &&= r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t) \\
&\hat{A}_t^{(3)} = \delta_t + \gamma\delta_{t+1} + \gamma^2\delta_{t+2} &&= r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 V(s_{t+3}) - V(s_t)
\end{aligned}$$

一般形式：

$$\hat{A}_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k \delta_{t+k}$$

<details>
<summary>展开验证（相邻 V 项望远镜式消去）</summary>

$$\hat{A}_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k \delta_{t+k} = \sum_{k=0}^{n-1} \gamma^k \left[ r_{t+k} + \gamma V(s_{t+k+1}) - V(s_{t+k}) \right]$$

将求和拆为奖励项和价值项：

$$= \underbrace{\sum_{k=0}^{n-1} \gamma^k r_{t+k}}_{\text{折扣奖励}} + \underbrace{\sum_{k=0}^{n-1} \gamma^k \left[\gamma V(s_{t+k+1}) - V(s_{t+k})\right]}_{\text{价值项（将会望远镜消去）}}$$

以 $n=3$ 为例，展开第二个求和（价值项）。注意 $\gamma^k \cdot \gamma V(s_{t+k+1}) = \gamma^{k+1} V(s_{t+k+1})$：

$$\begin{aligned}
&k=0: \quad \gamma^1 V(s_{t+1}) - \gamma^0 V(s_t)     &&= +\gamma V(s_{t+1})   &&- V(s_t) \\
&k=1: \quad \gamma^2 V(s_{t+2}) - \gamma^1 V(s_{t+1}) &&= +\gamma^2 V(s_{t+2}) &&- \gamma V(s_{t+1}) \\
&k=2: \quad \gamma^3 V(s_{t+3}) - \gamma^2 V(s_{t+2}) &&= +\gamma^3 V(s_{t+3}) &&- \gamma^2 V(s_{t+2})
\end{aligned}$$

每一行的正项与下一行的负项抵消（**望远镜式消去**），最终只剩第一个负项（$-V(s_t)$）和最后一个正项（$\gamma^3 V(s_{t+3})$）：

$$= \gamma^3 V(s_{t+3}) - V(s_t) = \gamma^n V(s_{t+n}) - V(s_t)$$

对任意 $n$ 同理：

$$\sum_{k=0}^{n-1} \gamma^k \left[\gamma V(s_{t+k+1}) - V(s_{t+k})\right] = \gamma^n V(s_{t+n}) - V(s_t)$$

</details>

结论：$\hat{A}_t^{(n)} = G_t^{(n)} - V(s_t)$，其中 $G_t^{(n)} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})$ 是 $n$ 步回报，$V(s_t)$ 是基线（当前状态的价值估计）。

## 3. 偏差-方差权衡

| $n$ | 特点 |
|-----|------|
| $n=1$ | 低方差、高偏差（只看一步，$V$ 不准时误差大） |
| $n=\infty$ | 高方差、低偏差（Monte Carlo，等整条轨迹，噪声大） |

核心权衡是 **bootstrap vs. 真实奖励**。"Bootstrap"指用学到的 $V$ 代替未观测到的未来回报（如 n=1 时的 $\gamma V(s_{t+1})$）。Bootstrap 越多 → 随机变量越少 → 方差低，但 $V$ 是近似值所以引入偏差。真实奖励步数越多 → 不做近似 → 偏差低，但每个 $r_t$ 都是随机变量（受随机策略和环境转移概率影响），噪声会累积。

**GAE 的思路：不选一个 $n$，而是对所有 $n$ 做指数加权平均。** 同时使用 $n=1, 2, 3, \ldots$ 的所有估计，但小 $n$ 权重高，越远的 $n$ 权重按 $\lambda^{n-1}$ 指数衰减。为什么要衰减？因为越大的 $n$ 累积了越多步真实奖励，每一步都引入噪声，所以方差越大。降低它们的权重可以压制方差，同时仍保留一些低偏差信号。调节 $\lambda$ 即控制这一取舍。

## 4. GAE 定义：指数加权

$$\hat{A}_t^{\text{GAE}} = (1-\lambda)\left[\hat{A}_t^{(1)} + \lambda\hat{A}_t^{(2)} + \lambda^2\hat{A}_t^{(3)} + \cdots\right]$$

其中 $\lambda \in [0, 1]$ 是衰减权重，$(1-\lambda)$ 是归一化系数（保证权重和为 1）。第 $n$ 项 $\hat{A}_t^{(n)}$ 的权重为 $(1-\lambda)\lambda^{n-1}$，所以 $n=1$ 的权重是 $(1-\lambda)$，$n=2$ 的权重是 $(1-\lambda)\lambda$，以此类推：

$$\text{权重之和} = (1-\lambda)(1 + \lambda + \lambda^2 + \cdots) = (1-\lambda) \cdot \frac{1}{1-\lambda} = 1 \quad \checkmark$$

这里用了等比级数公式：当 $|\lambda| < 1$ 时，$1 + \lambda + \lambda^2 + \cdots = \frac{1}{1-\lambda}$（证明：设 $S$ = 该求和，则 $S - \lambda S = 1$，故 $S = \frac{1}{1-\lambda}$）。

代入 $\hat{A}_t^{(n)}$ 的定义：

$$\hat{A}_t^{\text{GAE}} = (1-\lambda)\left[\delta_t + \lambda(\delta_t + \gamma\delta_{t+1}) + \lambda^2(\delta_t + \gamma\delta_{t+1} + \gamma^2\delta_{t+2}) + \cdots\right]$$

展开并按 $\delta_{t+k}$ 收集同类项，可化简为：

$$\boxed{\hat{A}_t^{\text{GAE}} = \sum_{k=0}^{\infty} (\gamma\lambda)^k \delta_{t+k} = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots}$$

## 5. 得到递推公式

从求和式：

$$\begin{aligned}
&\hat{A}_t     = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots \\
&\hat{A}_{t+1} = \delta_{t+1} + (\gamma\lambda)\delta_{t+2} + \cdots
\end{aligned}$$

观察到：

$$\hat{A}_t = \delta_t + (\gamma\lambda)\left[\delta_{t+1} + (\gamma\lambda)\delta_{t+2} + \cdots\right] = \delta_t + (\gamma\lambda)\hat{A}_{t+1}$$

**递推公式：**

$$\boxed{\hat{A}_t = \delta_t + \gamma\lambda \cdot \hat{A}_{t+1}}$$

边界条件：轨迹结束后不存在未来优势，即 $\hat{A}_{T+1} = 0$。因此 $\hat{A}_T = \delta_T + \gamma\lambda \cdot 0 = \delta_T$。

## 6. 代码实现

从后往前递推。输入 `td_delta` 是 shape 为 `(T,)` 的 tensor，每个元素是 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$：

```python
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:           # 从 T 到 0
        advantage = delta + gamma * lmbda * advantage   # A_t = delta_t + gamma*lambda * A_{t+1}
        advantage_list.append(advantage)
    advantage_list.reverse()               # 翻转回正序
    return torch.tensor(advantage_list, dtype=torch.float)
```

为什么倒着遍历？因为 $\hat{A}_t$ 依赖 $\hat{A}_{t+1}$——不先算出后面的值，就无法算前面的。从末尾开始，边界条件 $\hat{A}_{T+1} = 0$（轨迹结束后不存在未来）提供了第一个已知值。`advantage` 变量代表"下一步的优势"，第一次循环处理的是最后一步 $T$，它的"下一步"是 $T+1$——轨迹已结束，没有未来的 TD 误差，所以是 $0$。此后每次迭代往前推一步。

**注意：** 此实现假设轨迹是完整的（terminated）。如果 episode 是被截断的（truncated），边界条件应为 $\hat{A}_{T+1} = \gamma\lambda \cdot V(s_{T+1})$ 而非 $0$，因为截断点之后仍有未来回报。

## 7. 两个极端验证

**$\lambda = 0$ 时：**

$$\hat{A}_t = \delta_t + 0 \cdot \hat{A}_{t+1} = \delta_t$$

退化为单步 TD 误差。偏差高（依赖 $V$ 的准确性），方差低。

**$\lambda = 1$ 时：**

$$\begin{aligned}
&\hat{A}_t = \delta_t + \gamma\hat{A}_{t+1} \\
&\phantom{\hat{A}_t} = \delta_t + \gamma\delta_{t+1} + \gamma^2\delta_{t+2} + \cdots \\
&\phantom{\hat{A}_t} = \sum_{k=0}^{\infty}\gamma^k\delta_{t+k} = G_t - V(s_t)
\end{aligned}$$

最后一步利用了第 2 节的结论：当 $n \to \infty$ 时，$\hat{A}_t^{(\infty)} = G_t - V(s_t)$（末端 bootstrap 项 $\gamma^\infty V(s_\infty) \to 0$）。

退化为 Monte Carlo 优势估计。偏差低，方差高。

## 8. 直觉

| $\lambda$ | 公式 | 名称 | 偏差 | 方差 |
|-----------|------|------|------|------|
| $0$ | $\hat{A}_t = \delta_t$ | 单步 TD | 高 | 低 |
| $0.95$ | $\hat{A}_t = \sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k}$ | GAE（折中） | 中 | 中 |
| $1$ | $\hat{A}_t = G_t - V(s_t)$ | Monte Carlo | 低 | 高 |

$\lambda = 0.95$ 时：近处的 $\delta$ 权重大（$0.95^0=1,\ 0.95^1=0.95,\ 0.95^2\approx0.90, \ldots$），远处的指数衰减。估计主要相信近几步的 TD 信息，远处的不太信但也不完全忽略。

## 9. 在 PPO / Actor-Critic 中的位置

$$\begin{aligned}
&\text{采集 episode} \\
&\longrightarrow \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \\
&\longrightarrow \hat{A}_t = \sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k} \\
&\longrightarrow \text{用 } \hat{A}_t \text{ 更新策略}
\end{aligned}$$

- $\hat{A}_t > 0$：这个动作比平均好 $\rightarrow$ 增大其概率
- $\hat{A}_t < 0$：这个动作比平均差 $\rightarrow$ 减小其概率
