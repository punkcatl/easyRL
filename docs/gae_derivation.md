# GAE (Generalized Advantage Estimation) Derivation

## 1. Starting Point: TD Error

The single-step TD error is defined as:

```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

delta_t is a **biased estimate** of the advantage function A(s_t, a_t) — it only looks one step into the future.

## 2. Multi-Step Advantage Estimates

Instead of looking one step ahead, we can look n steps:

```
A_t^(1) = delta_t
         = r_t + gamma*V(s_{t+1}) - V(s_t)

A_t^(2) = delta_t + gamma*delta_{t+1}
         = r_t + gamma*r_{t+1} + gamma^2*V(s_{t+2}) - V(s_t)

A_t^(3) = delta_t + gamma*delta_{t+1} + gamma^2*delta_{t+2}
         = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + gamma^3*V(s_{t+3}) - V(s_t)

General form:
A_t^(n) = sum_{k=0}^{n-1} gamma^k * delta_{t+k}
```

Verification by expanding delta terms (telescoping cancellation of V terms):

```
A_t^(n) = sum_{k=0}^{n-1} gamma^k * [r_{t+k} + gamma*V(s_{t+k+1}) - V(s_{t+k})]
        = r_t + gamma*r_{t+1} + ... + gamma^(n-1)*r_{t+n-1} + gamma^n*V(s_{t+n}) - V(s_t)
```

This equals the n-step return G_t^(n) minus V(s_t).

## 3. The Bias-Variance Trade-off

| n | Property |
|---|----------|
| n=1 | Low variance, high bias (only one step; error is large when V is inaccurate) |
| n=infinity | High variance, low bias (Monte Carlo — waits for the full trajectory, noisy) |

**GAE's insight: don't pick a single n — take an exponentially weighted average over all n.**

## 4. GAE Definition: Exponential Weighting

```
A_t^GAE = (1 - lambda) * [A_t^(1) + lambda*A_t^(2) + lambda^2*A_t^(3) + ...]
```

where lambda in [0, 1] is the decay weight, and (1 - lambda) normalizes so weights sum to 1:

```
Weights: (1-lambda) * [1, lambda, lambda^2, ...]
Sum:     (1-lambda) * (1 + lambda + lambda^2 + ...) = (1-lambda) * 1/(1-lambda) = 1  ✓
```

Substituting A_t^(n) definitions:

```
A_t^GAE = (1-lambda) * [delta_t + lambda*(delta_t + gamma*delta_{t+1}) + lambda^2*(delta_t + gamma*delta_{t+1} + gamma^2*delta_{t+2}) + ...]
```

## 5. Key Derivation: Simplification to Closed Form

Collect terms by delta_t, delta_{t+1}, delta_{t+2}, ...:

**Coefficient of delta_t** (appears in every A_t^(n) for n=1,2,3,...):

```
(1-lambda) * (1 + lambda + lambda^2 + ...) = (1-lambda) * 1/(1-lambda) = 1
```

**Coefficient of delta_{t+1}** (appears in A_t^(n) for n>=2, multiplied by gamma):

delta_{t+1} first appears in A_t^(2) (weighted by lambda^1 in the outer sum), then in A_t^(3) (weighted by lambda^2), etc. So its geometric series starts at lambda, not 1:

```
(1-lambda) * gamma * (lambda + lambda^2 + lambda^3 + ...)
= (1-lambda) * gamma * lambda/(1-lambda)
= gamma * lambda
```

**Coefficient of delta_{t+2}** (appears in A_t^(n) for n>=3, multiplied by gamma^2):

```
(1-lambda) * gamma^2 * (lambda^2 + lambda^3 + ...)
= (1-lambda) * gamma^2 * lambda^2/(1-lambda)
= (gamma * lambda)^2
```

**General pattern: coefficient of delta_{t+k} is (gamma * lambda)^k**

Therefore:

```
A_t^GAE = sum_{k=0}^{infinity} (gamma * lambda)^k * delta_{t+k}

        = delta_t + (gamma*lambda)*delta_{t+1} + (gamma*lambda)^2*delta_{t+2} + ...
```

## 6. Deriving the Recursive Formula

From the summation form:

```
A_t   = delta_t + (gamma*lambda)*delta_{t+1} + (gamma*lambda)^2*delta_{t+2} + ...
A_{t+1} =              delta_{t+1} + (gamma*lambda)*delta_{t+2} + ...
```

Observe:

```
A_t = delta_t + (gamma*lambda) * [delta_{t+1} + (gamma*lambda)*delta_{t+2} + ...]
    = delta_t + (gamma*lambda) * A_{t+1}
```

**The recursive formula:**

```
A_t = delta_t + gamma * lambda * A_{t+1}
```

Boundary condition: after the trajectory ends, there is no future advantage, so A_{T+1} = 0. This gives A_T = delta_T + gamma * lambda * 0 = delta_T.

## 7. Implementation

Compute backwards from T to 0:

```python
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:           # from T to 0
        advantage = gamma * lmbda * advantage + delta   # A_t = delta_t + gamma*lambda * A_{t+1}
        advantage_list.append(advantage)
    advantage_list.reverse()               # flip back to chronological order
    return torch.tensor(advantage_list, dtype=torch.float)
```

Initial `advantage = 0` serves as the boundary condition (after the trajectory ends, A = 0). Each iteration applies the recursive formula one step backward.

## 8. Verifying the Two Extremes

**lambda = 0:**

```
A_t = delta_t + 0 * A_{t+1} = delta_t
```

Degenerates to single-step TD error. High bias (relies on V accuracy), low variance.

**lambda = 1:**

```
A_t = delta_t + gamma * A_{t+1}
    = delta_t + gamma*delta_{t+1} + gamma^2*delta_{t+2} + ...
    = sum gamma^k * delta_{t+k}
    = (r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...) - V(s_t)
    = G_t - V(s_t)
```

Degenerates to Monte Carlo advantage estimate. Low bias, high variance.

## 9. Intuition

```
lambda=0                lambda=0.95             lambda=1
  |                       |                       |
  v                       v                       v
A_t = delta_t      A_t = sum(gamma*lambda)^k    A_t = G_t - V(s_t)
                         * delta_{t+k}
Single-step TD         GAE (balanced)           Monte Carlo
High bias/Low var      Trade-off               Low bias/High var
```

With lambda = 0.95: nearby deltas have high weight (0.95^0=1, 0.95^1=0.95, 0.95^2=0.9...), distant ones decay exponentially. The estimate primarily trusts the next few TD errors but doesn't completely ignore distant ones.

## 10. Role in PPO / Actor-Critic

```
Collect an episode
    |
    v
Compute TD errors: delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
    |
    v
Compute GAE advantage: A_t = sum (gamma*lambda)^k * delta_{t+k}    <-- this formula
    |
    v
Update policy: actor loss uses A_t as the advantage signal
```

- A_t > 0: this action is better than average -> increase its probability
- A_t < 0: this action is worse than average -> decrease its probability

---

# GAE（广义优势估计）推导

## 1. 起点：TD 误差

单步 TD 误差定义：

```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

delta_t 是对优势函数 A(s_t, a_t) 的一个**有偏估计**——它只看了一步未来。

## 2. 多步优势估计

如果不只看一步，而是看 n 步：

```
A_t^(1) = delta_t
         = r_t + gamma*V(s_{t+1}) - V(s_t)

A_t^(2) = delta_t + gamma*delta_{t+1}
         = r_t + gamma*r_{t+1} + gamma^2*V(s_{t+2}) - V(s_t)

A_t^(3) = delta_t + gamma*delta_{t+1} + gamma^2*delta_{t+2}
         = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + gamma^3*V(s_{t+3}) - V(s_t)

一般形式：
A_t^(n) = sum_{k=0}^{n-1} gamma^k * delta_{t+k}
```

展开验证（相邻 V 项望远镜式消去）：

```
A_t^(n) = sum_{k=0}^{n-1} gamma^k * [r_{t+k} + gamma*V(s_{t+k+1}) - V(s_{t+k})]
        = r_t + gamma*r_{t+1} + ... + gamma^(n-1)*r_{t+n-1} + gamma^n*V(s_{t+n}) - V(s_t)
```

这等于 n 步回报 G_t^(n) 减去 V(s_t)。

## 3. 偏差-方差权衡

| n | 特点 |
|---|------|
| n=1 | 低方差、高偏差（只看一步，V 不准时误差大） |
| n=无穷 | 高方差、低偏差（Monte Carlo，等整条轨迹，噪声大） |

**GAE 的思路：不选一个 n，而是对所有 n 做指数加权平均。**

## 4. GAE 定义：指数加权

```
A_t^GAE = (1 - lambda) * [A_t^(1) + lambda*A_t^(2) + lambda^2*A_t^(3) + ...]
```

其中 lambda 属于 [0, 1] 是衰减权重，(1 - lambda) 是归一化系数（保证权重和为 1）：

```
权重: (1-lambda) * [1, lambda, lambda^2, ...]
求和: (1-lambda) * (1 + lambda + lambda^2 + ...) = (1-lambda) * 1/(1-lambda) = 1  ✓
```

代入 A_t^(n) 的定义：

```
A_t^GAE = (1-lambda) * [delta_t + lambda*(delta_t + gamma*delta_{t+1}) + lambda^2*(delta_t + gamma*delta_{t+1} + gamma^2*delta_{t+2}) + ...]
```

## 5. 关键推导：化简为封闭形式

按 delta_t, delta_{t+1}, delta_{t+2}, ... 收集同类项：

**delta_t 的系数**（出现在每一个 A_t^(n) 中，n=1,2,3,...）：

```
(1-lambda) * (1 + lambda + lambda^2 + ...) = (1-lambda) * 1/(1-lambda) = 1
```

**delta_{t+1} 的系数**（出现在 A_t^(n) 中 n>=2，乘以 gamma）：

delta_{t+1} 首次出现在 A_t^(2) 中（外层求和权重为 lambda^1），然后在 A_t^(3) 中（权重 lambda^2），以此类推。所以等比级数从 lambda 开始，不是从 1 开始：

```
(1-lambda) * gamma * (lambda + lambda^2 + lambda^3 + ...)
= (1-lambda) * gamma * lambda/(1-lambda)
= gamma * lambda
```

**delta_{t+2} 的系数**（出现在 A_t^(n) 中 n>=3，乘以 gamma^2）：

```
(1-lambda) * gamma^2 * (lambda^2 + lambda^3 + ...)
= (1-lambda) * gamma^2 * lambda^2/(1-lambda)
= (gamma * lambda)^2
```

**一般规律：delta_{t+k} 的系数为 (gamma * lambda)^k**

因此：

```
A_t^GAE = sum_{k=0}^{infinity} (gamma * lambda)^k * delta_{t+k}

        = delta_t + (gamma*lambda)*delta_{t+1} + (gamma*lambda)^2*delta_{t+2} + ...
```

## 6. 得到递推公式

从求和式：

```
A_t   = delta_t + (gamma*lambda)*delta_{t+1} + (gamma*lambda)^2*delta_{t+2} + ...
A_{t+1} =              delta_{t+1} + (gamma*lambda)*delta_{t+2} + ...
```

观察到：

```
A_t = delta_t + (gamma*lambda) * [delta_{t+1} + (gamma*lambda)*delta_{t+2} + ...]
    = delta_t + (gamma*lambda) * A_{t+1}
```

**递推公式：**

```
A_t = delta_t + gamma * lambda * A_{t+1}
```

边界条件：轨迹结束后不存在未来优势，即 A_{T+1} = 0。因此 A_T = delta_T + gamma * lambda * 0 = delta_T。

## 7. 代码实现

从后往前递推：

```python
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:           # 从 T 到 0
        advantage = gamma * lmbda * advantage + delta   # A_t = delta_t + gamma*lambda * A_{t+1}
        advantage_list.append(advantage)
    advantage_list.reverse()               # 翻转回正序
    return torch.tensor(advantage_list, dtype=torch.float)
```

初始 `advantage = 0` 相当于边界条件（轨迹结束后 A = 0），每步往前套递推公式。

## 8. 两个极端验证

**lambda = 0 时：**

```
A_t = delta_t + 0 * A_{t+1} = delta_t
```

退化为单步 TD 误差。偏差高（依赖 V 的准确性），方差低。

**lambda = 1 时：**

```
A_t = delta_t + gamma * A_{t+1}
    = delta_t + gamma*delta_{t+1} + gamma^2*delta_{t+2} + ...
    = sum gamma^k * delta_{t+k}
    = (r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...) - V(s_t)
    = G_t - V(s_t)
```

退化为 Monte Carlo 优势估计。偏差低，方差高。

## 9. 直觉

```
lambda=0                lambda=0.95             lambda=1
  |                       |                       |
  v                       v                       v
A_t = delta_t      A_t = sum(gamma*lambda)^k    A_t = G_t - V(s_t)
                         * delta_{t+k}
单步 TD                 GAE（折中）              Monte Carlo
高偏差/低方差           平衡                    低偏差/高方差
```

lambda = 0.95 时：近处的 delta 权重大（0.95^0=1, 0.95^1=0.95, 0.95^2=0.9...），远处的指数衰减。估计主要相信近几步的 TD 信息，远处的不太信但也不完全忽略。

## 10. 在 PPO / Actor-Critic 中的位置

```
采集一个 episode
    |
    v
算 TD error: delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
    |
    v
算 GAE advantage: A_t = sum (gamma*lambda)^k * delta_{t+k}    <-- 就是这个公式
    |
    v
用 A_t 更新 policy（actor loss 中乘以 A_t）
```

- A_t > 0：这个动作比平均好 -> 增大其概率
- A_t < 0：这个动作比平均差 -> 减小其概率
