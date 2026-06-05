# RL+MPC Design Rationale

## 1. Why RL+MPC

Traditional autonomous driving stacks separate planning from control. RL excels at high-level decision-making under uncertainty but produces noisy low-level commands. MPC excels at smooth, constraint-respecting trajectory tracking but needs reference targets.

Combining them:
- PPO handles the combinatorial decision problem (when to change lanes, speed up/down)
- MPC handles continuous control with physical constraints (jerk limits, steering rate)
- The interface between them is clean: (v_ref, y_ref) reference signals

This separation also means the MPC can guarantee constraint satisfaction regardless of what PPO decides, improving safety.

## 2. PPO for Discrete Decisions

### Action Space Design

5 discrete actions match highway-env's DiscreteMetaAction convention:

| Index | Action | Effect |
|-------|--------|--------|
| 0 | LANE_LEFT | Move y_ref to left lane center |
| 1 | IDLE | Maintain current references |
| 2 | LANE_RIGHT | Move y_ref to right lane center |
| 3 | FASTER | Increase v_ref by delta_v |
| 4 | SLOWER | Decrease v_ref by delta_v |

### Network Architecture

Two-layer MLP with ReLU activations:
- PolicyNet: obs(25) -> 128 -> 128 -> 5 (softmax)
- ValueNet: obs(25) -> 128 -> 128 -> 1

### PPO Update

Standard clipped PPO with GAE advantage estimation:
- Clip ratio epsilon = 0.2
- GAE lambda = 0.95
- 10 update epochs per episode

## 3. Longitudinal MPC

### Triple Integrator Model

State: x = [s, v, a]^T (position, velocity, acceleration)
Control: u = j (jerk)

Continuous dynamics:
```
ds/dt = v
dv/dt = a
da/dt = j
```

Exact discretization (dt = 0.1s):
```
x(k+1) = Ad * x(k) + Bd * u(k)

Ad = [1   dt   0.5*dt^2]     Bd = [(1/6)*dt^3]
     [0   1    dt       ]          [0.5*dt^2  ]
     [0   0    1        ]          [dt         ]
```

### Cost Function

```
J = sum_{k=0}^{N} Q_v * (v(k) - v_ref)^2 + Q_a * a(k)^2 + R_j * j(k)^2
```

- Q_v = 10: velocity tracking (primary objective)
- Q_a = 1: penalize large accelerations for comfort
- R_j = 0.1: penalize jerk for smoothness

### Constraints

| Variable | Min | Max | Rationale |
|----------|-----|-----|-----------|
| v (velocity) | 0 m/s | 40 m/s | Physical speed limits |
| a (acceleration) | -4 m/s^2 | 2 m/s^2 | Comfort + tire friction |
| j (jerk) | -5 m/s^3 | 5 m/s^3 | Passenger comfort |

### Solver

CasADi + IPOPT with warm starting. Horizon N=20 steps (2.0s lookahead). The QP structure of the linear model makes this very fast to solve.

## 4. Lateral MPC

### Kinematic Bicycle Model

```
dx/dt = v * cos(psi)
dy/dt = v * sin(psi)
dpsi/dt = v / L * tan(delta)
```

Where L = 2.5m (wheelbase), delta = front wheel steering angle.

This is a nonlinear model handled natively by CasADi's NLP solver (no linearization needed).

### Cost Function

```
J = sum_{k=0}^{N} Q_y * (y(k) - y_ref)^2 + Q_psi * (psi(k) - psi_ref)^2 + R_delta * delta(k)^2
```

- Q_y = 10: lateral position tracking
- Q_psi = 5: heading alignment
- R_delta = 1: steering effort penalty

### Constraints

| Variable | Min | Max | Rationale |
|----------|-----|-----|-----------|
| delta (steering) | -0.5 rad | 0.5 rad | Physical steering limits |
| delta_dot (rate) | -0.3 rad/s | 0.3 rad/s | Actuator bandwidth |

Steering rate constraint is enforced between consecutive time steps to prevent sudden steering inputs.

### Solver

CasADi + IPOPT, N=15 steps (1.5s lookahead). Shorter horizon than longitudinal because lateral dynamics are faster. Warm starting from previous solution for real-time performance.

## 5. Action Mapping Strategy

The action mapper maintains persistent (v_ref, y_ref) state:
- FASTER/SLOWER: increment/decrement v_ref by delta_v (5 m/s), clamp to [0, 40]
- LANE_LEFT/RIGHT: query actual lane geometry for target y position
- IDLE: no change to references

Using actual lane center positions (from highway-env road network) rather than fixed offsets ensures correct behavior on curved roads and non-uniform lane widths.

## 6. Action Normalization

highway-env ContinuousAction expects inputs in [-1, 1]. The MPC outputs physical units:
- Steering: delta in [-0.5, 0.5] rad -> normalized by delta_max
- Acceleration: a_des in [-4, 2] m/s^2 -> normalized by |a_min|

---

# RL+MPC 设计原理

## 1. 为什么结合 RL 和 MPC

传统自动驾驶架构将规划与控制分开。RL 擅长不确定性下的高层决策，但底层指令噪声大。MPC 擅长平滑、满足约束的轨迹跟踪，但需要参考目标。

结合两者的优势：
- PPO 处理组合决策问题（何时换道、加减速）
- MPC 处理带物理约束的连续控制（jerk 限制、转向速率）
- 两者之间的接口简洁：(v_ref, y_ref) 参考信号

这种分离还意味着无论 PPO 做出什么决策，MPC 都能保证约束满足，提高安全性。

## 2. PPO 离散决策

### 动作空间设计

5 个离散动作，与 highway-env 的 DiscreteMetaAction 对应：

| 编号 | 动作 | 效果 |
|------|------|------|
| 0 | LANE_LEFT | y_ref 移至左车道中心 |
| 1 | IDLE | 保持当前参考值 |
| 2 | LANE_RIGHT | y_ref 移至右车道中心 |
| 3 | FASTER | v_ref 增加 delta_v |
| 4 | SLOWER | v_ref 减小 delta_v |

### 网络结构

两层 MLP，ReLU 激活：
- PolicyNet: obs(25) -> 128 -> 128 -> 5 (softmax)
- ValueNet: obs(25) -> 128 -> 128 -> 1

### PPO 更新

标准 clipped PPO + GAE 优势估计：
- Clip 比率 epsilon = 0.2
- GAE lambda = 0.95
- 每个 episode 训练 10 个 epoch

## 3. 纵向 MPC

### 三阶积分器模型

状态: x = [s, v, a]^T（位移、速度、加速度）
控制: u = j（jerk）

连续动力学：
```
ds/dt = v
dv/dt = a
da/dt = j
```

精确离散化（dt = 0.1s）：
```
x(k+1) = Ad * x(k) + Bd * u(k)

Ad = [1   dt   0.5*dt^2]     Bd = [(1/6)*dt^3]
     [0   1    dt       ]          [0.5*dt^2  ]
     [0   0    1        ]          [dt         ]
```

### 代价函数

```
J = sum_{k=0}^{N} Q_v * (v(k) - v_ref)^2 + Q_a * a(k)^2 + R_j * j(k)^2
```

- Q_v = 10: 速度跟踪（主目标）
- Q_a = 1: 惩罚大加速度以保证舒适性
- R_j = 0.1: 惩罚 jerk 以保证平滑性

### 约束

| 变量 | 最小值 | 最大值 | 理由 |
|------|--------|--------|------|
| v（速度） | 0 m/s | 40 m/s | 物理速度限制 |
| a（加速度） | -4 m/s^2 | 2 m/s^2 | 舒适性 + 轮胎摩擦 |
| j（jerk） | -5 m/s^3 | 5 m/s^3 | 乘客舒适性 |

### 求解器

CasADi + IPOPT，带热启动。预测步长 N=20（2.0s 前瞻）。线性模型的 QP 结构使求解非常快速。

## 4. 横向 MPC

### 运动学自行车模型

```
dx/dt = v * cos(psi)
dy/dt = v * sin(psi)
dpsi/dt = v / L * tan(delta)
```

其中 L = 2.5m（轴距），delta = 前轮转角。

这是非线性模型，由 CasADi 的 NLP 求解器原生处理（无需线性化）。

### 代价函数

```
J = sum_{k=0}^{N} Q_y * (y(k) - y_ref)^2 + Q_psi * (psi(k) - psi_ref)^2 + R_delta * delta(k)^2
```

- Q_y = 10: 横向位置跟踪
- Q_psi = 5: 航向对齐
- R_delta = 1: 转向能量惩罚

### 约束

| 变量 | 最小值 | 最大值 | 理由 |
|------|--------|--------|------|
| delta（转向角） | -0.5 rad | 0.5 rad | 物理转向限制 |
| delta_dot（转向速率） | -0.3 rad/s | 0.3 rad/s | 执行器带宽 |

转向速率约束在相邻时间步之间强制执行，防止突然的转向输入。

### 求解器

CasADi + IPOPT，N=15 步（1.5s 前瞻）。比纵向更短的预测窗口，因为横向动力学更快。利用前一步解的热启动保证实时性能。

## 5. 动作映射策略

动作映射器维护持久的 (v_ref, y_ref) 状态：
- FASTER/SLOWER: v_ref 增减 delta_v（5 m/s），截断到 [0, 40]
- LANE_LEFT/RIGHT: 查询实际车道几何获取目标 y 位置
- IDLE: 参考值不变

使用实际车道中心位置（来自 highway-env 道路网络）而非固定偏移，确保在弯道和非均匀车道宽度上行为正确。

## 6. 动作归一化

highway-env ContinuousAction 期望输入在 [-1, 1] 范围内。MPC 输出物理单位：
- 转向: delta 在 [-0.5, 0.5] rad -> 除以 delta_max 归一化
- 加速度: a_des 在 [-4, 2] m/s^2 -> 除以 |a_min| 归一化
