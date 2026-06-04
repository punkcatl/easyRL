# RL+MPC Demo Design Spec

## 1. Overview

A demonstration module combining PPO decision-making with MPC control for autonomous driving scenarios. PPO outputs discrete high-level decisions (accelerate, decelerate, lane change), which are mapped to reference targets and tracked by longitudinal/lateral MPC controllers.

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    highway-env                            │
│  (highway/merge/roundabout/intersection/racetrack)       │
└────────────────────────┬────────────────────────────────┘
                         │ obs (Kinematics)
                         ▼
┌─────────────────────────────────────────────────────────┐
│              PPO Decision Layer (Discrete)                │
│  Output: FASTER / SLOWER / LANE_LEFT / LANE_RIGHT / IDLE │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Action Mapper                                │
│  FASTER → v_ref += Δv     LANE_LEFT → y_ref = left lane  │
└──────────┬──────────────────────────┬───────────────────┘
           │                          │
           ▼                          ▼
┌──────────────────────┐   ┌──────────────────────┐
│  Longitudinal MPC    │   │  Lateral MPC          │
│  Model: triple       │   │  Model: kinematic     │
│         integrator   │   │         bicycle       │
│  Solver: CasADi      │   │  Solver: CasADi       │
│  Output: a_des       │   │  Output: δ            │
└──────────┬───────────┘   └──────────┬───────────┘
           │                          │
           ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│              env.step([δ, a_des])                         │
└─────────────────────────────────────────────────────────┘
```

## 3. Module Structure

```
applications/vehicle_control/rl_mpc/
├── envs/
│   ├── env_wrapper.py          ← unified highway-env wrapper (7 scenarios)
│   └── carla_wrapper.py        ← CARLA interface stub (abstract base + placeholder)
├── controller/
│   ├── lon_mpc.py              ← longitudinal MPC (CasADi, triple integrator)
│   ├── lat_mpc.py              ← lateral MPC (CasADi, kinematic bicycle)
│   └── action_mapper.py        ← discrete action → (v_ref, y_ref) mapping
├── agent/
│   └── ppo_decision.py         ← PPO decision layer (discrete, reuse algorithms/ppo/)
├── train.py                    ← training entry point
├── eval.py                     ← evaluation + visualization
├── config.py                   ← centralized hyperparameters
└── docs/
    └── theory.md               ← design rationale document
```

## 4. Environment Wrapper

### 4.1 Supported Scenarios

| Environment | Scenario | Complexity |
|---|---|---|
| `highway-v0` | Straight highway, overtaking/following | Low |
| `merge-v0` | On-ramp merging | Medium |
| `roundabout-v0` | Roundabout multi-vehicle | Medium-High |
| `intersection-v0` | T/cross intersection | High |
| `intersection-v1` | Intersection variant | High |
| `racetrack-v0` | Track with turns | Medium |
| `racetrack-large-v0` | Larger track | Medium-High |

### 4.2 Configuration

- Observation: Kinematics (relative coordinates, 5 vehicles), features: `[x, y, vx, vy, heading]`
- Action: ContinuousAction (steering + acceleration), controlled by MPC output
- Unified interface: `reset() → obs`, `step(action) → obs, reward, done, info`
- Scenario selection via config parameter

### 4.3 CARLA Interface (Reserved)

Define abstract base class `BaseEnvWrapper` with the same interface. `carla_wrapper.py` contains a stub implementation with `NotImplementedError` for future CARLA integration.

## 5. PPO Decision Layer

### 5.1 Action Space

5 discrete actions (matching highway-env DiscreteMetaAction convention):

| Index | Action | Meaning |
|---|---|---|
| 0 | LANE_LEFT | Set y_ref to left lane center |
| 1 | IDLE | Maintain current v_ref and y_ref |
| 2 | LANE_RIGHT | Set y_ref to right lane center |
| 3 | FASTER | Increase v_ref |
| 4 | SLOWER | Decrease v_ref |

### 5.2 Network

Reuse the discrete PPO architecture from `hands_on_rl/ch12_PPO/ppo.py`:
- PolicyNet: state_dim → 128 → action_dim (softmax)
- ValueNet: state_dim → 128 → 1

### 5.3 Reward

Use highway-env's default reward (speed reward + collision penalty + lane keeping bonus). Can be customized per scenario in config.

## 6. Action Mapper

```python
FASTER     → v_ref = clip(v_current + 5, 0, v_max)
SLOWER     → v_ref = clip(v_current - 5, 0, v_max)
LANE_LEFT  → y_ref = current_lane_center - lane_width
LANE_RIGHT → y_ref = current_lane_center + lane_width
IDLE       → v_ref, y_ref unchanged
```

Parameters (`Δv`, `v_max`, `lane_width`) configurable in `config.py`.

## 7. Longitudinal MPC

### 7.1 Model: Triple Integrator

State: `x = [s, v, a]ᵀ` (position, velocity, acceleration)
Control: `u = j` (jerk)
Output: `a_des = x[2]`

Continuous time:
```
ṡ = v
v̇ = a
ȧ = j
```

Exact discretization (dt = 0.1s):
```
x(k+1) = Ad · x(k) + Bd · u(k)

Ad = [1   dt   0.5·dt²]     Bd = [(1/6)·dt³]
     [0   1    dt      ]          [0.5·dt²  ]
     [0   0    1       ]          [dt        ]
```

### 7.2 Constraints

| Variable | Min | Max |
|---|---|---|
| a (acceleration) | -4 m/s² | 2 m/s² |
| j (jerk) | -5 m/s³ | 5 m/s³ |
| v (velocity) | 0 m/s | 40 m/s |

### 7.3 Cost Function

```
J = Σ (v(k) - v_ref)² · Q_v + a(k)² · Q_a + j(k)² · R_j
```

### 7.4 Solver

CasADi with IPOPT, horizon N=20, dt=0.1s.

## 8. Lateral MPC

### 8.1 Model: Kinematic Bicycle

```
ẋ = v · cos(ψ)
ẏ = v · sin(ψ)
ψ̇ = v / L · tan(δ)
```

State: `[x, y, ψ]` (position x, position y, heading)
Control: `δ` (front wheel steering angle)
Parameter: `L` (wheelbase), `v` (from longitudinal state, treated as parameter)

### 8.2 Constraints

| Variable | Min | Max |
|---|---|---|
| δ (steering angle) | -0.5 rad | 0.5 rad |
| δ̇ (steering rate) | -0.3 rad/s | 0.3 rad/s |

### 8.3 Cost Function

```
J = Σ (y(k) - y_ref)² · Q_y + (ψ(k) - ψ_ref)² · Q_ψ + δ(k)² · R_δ
```

### 8.4 Solver

CasADi with IPOPT, horizon N=15, dt=0.1s. Nonlinear model handled natively by CasADi (no manual linearization needed).

## 9. Training Pipeline

1. Initialize highway-env with selected scenario
2. PPO observes state, outputs discrete action
3. Action Mapper converts to (v_ref, y_ref)
4. Longitudinal MPC computes a_des to track v_ref
5. Lateral MPC computes δ to track y_ref
6. env.step([δ, a_des])
7. Collect reward, update PPO

PPO update uses GAE (λ=0.95), clip ratio ε=0.2, 10 epochs per episode.

## 10. Dependencies

- `gymnasium` + `highway-env`: simulation environments
- `casadi`: MPC solver (pip install casadi)
- `torch`: PPO neural networks
- `numpy`, `matplotlib`: data processing and visualization

## 11. Deliverables

1. Working RL+MPC closed-loop on all 7 highway-env scenarios
2. Training curves showing PPO convergence
3. Visualization of MPC tracking performance (v vs v_ref, y vs y_ref)
4. CARLA wrapper stub ready for future integration
5. Theory documentation explaining the design rationale

---

# RL+MPC Demo 设计规范

## 1. 概述

一个结合 PPO 决策与 MPC 控制的自动驾驶演示模块。PPO 输出离散高层决策（加速、减速、换道），映射为参考目标后由纵向/横向 MPC 控制器跟踪执行。

## 2. 架构

```
┌─────────────────────────────────────────────────────────┐
│                    highway-env                            │
│  (highway/merge/roundabout/intersection/racetrack)       │
└────────────────────────┬────────────────────────────────┘
                         │ obs (Kinematics)
                         ▼
┌─────────────────────────────────────────────────────────┐
│              PPO 决策层（离散动作）                         │
│  输出: FASTER / SLOWER / LANE_LEFT / LANE_RIGHT / IDLE   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              动作映射器 (Action Mapper)                    │
│  FASTER → v_ref += Δv     LANE_LEFT → y_ref = 左车道中心  │
└──────────┬──────────────────────────┬───────────────────┘
           │                          │
           ▼                          ▼
┌──────────────────────┐   ┌──────────────────────┐
│   纵向 MPC (CasADi)   │   │   横向 MPC (CasADi)   │
│   模型: 三阶积分器     │   │   模型: 运动学自行车   │
│   输出: a_des          │   │   输出: δ (转向角)     │
└──────────┬───────────┘   └──────────┬───────────┘
           │                          │
           ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│              env.step([δ, a_des])                         │
└─────────────────────────────────────────────────────────┘
```

## 3. 模块结构

```
applications/vehicle_control/rl_mpc/
├── envs/
│   ├── env_wrapper.py          ← highway-env 统一封装（7 个场景切换）
│   └── carla_wrapper.py        ← CARLA 接口预留（抽象基类 + stub）
├── controller/
│   ├── lon_mpc.py              ← 纵向 MPC（CasADi，三阶积分器）
│   ├── lat_mpc.py              ← 横向 MPC（CasADi，运动学自行车模型）
│   └── action_mapper.py        ← 离散动作 → (v_ref, y_ref) 映射
├── agent/
│   └── ppo_decision.py         ← PPO 决策层（离散，复用 algorithms/ppo/）
├── train.py                    ← 训练入口
├── eval.py                     ← 评估 + 可视化
├── config.py                   ← 超参数集中管理
└── docs/
    └── theory.md               ← 设计原理文档
```

## 4. 环境封装

### 4.1 支持的场景

| 环境 | 场景 | 复杂度 |
|---|---|---|
| `highway-v0` | 高速直道，超车/跟车 | 低 |
| `merge-v0` | 匝道汇入 | 中 |
| `roundabout-v0` | 环岛多车交汇 | 中高 |
| `intersection-v0` | 十字/丁字路口 | 高 |
| `intersection-v1` | 路口变体 | 高 |
| `racetrack-v0` | 带弯道的赛道 | 中 |
| `racetrack-large-v0` | 大赛道 | 中高 |

### 4.2 配置

- 观测: Kinematics（相对坐标，5 辆车），特征: `[x, y, vx, vy, heading]`
- 动作: ContinuousAction（转向 + 加速度），由 MPC 输出控制
- 统一接口: `reset() → obs`, `step(action) → obs, reward, done, info`
- 通过 config 参数切换场景

### 4.3 CARLA 接口（预留）

定义抽象基类 `BaseEnvWrapper`，接口与 highway-env wrapper 一致。`carla_wrapper.py` 中包含 stub 实现（`NotImplementedError`），供后续 CARLA 集成使用。

## 5. PPO 决策层

### 5.1 动作空间

5 个离散动作（与 highway-env DiscreteMetaAction 一致）:

| 编号 | 动作 | 含义 |
|---|---|---|
| 0 | LANE_LEFT | y_ref 设为左车道中心 |
| 1 | IDLE | 保持当前 v_ref 和 y_ref |
| 2 | LANE_RIGHT | y_ref 设为右车道中心 |
| 3 | FASTER | 提高 v_ref |
| 4 | SLOWER | 降低 v_ref |

### 5.2 网络结构

复用 `hands_on_rl/ch12_PPO/ppo.py` 的离散 PPO 架构:
- PolicyNet: state_dim → 128 → action_dim (softmax)
- ValueNet: state_dim → 128 → 1

### 5.3 奖励

使用 highway-env 默认奖励（速度奖励 + 碰撞惩罚 + 车道保持奖励）。可在 config 中按场景自定义。

## 6. 动作映射器

```python
FASTER     → v_ref = clip(v_current + 5, 0, v_max)
SLOWER     → v_ref = clip(v_current - 5, 0, v_max)
LANE_LEFT  → y_ref = current_lane_center - lane_width
LANE_RIGHT → y_ref = current_lane_center + lane_width
IDLE       → v_ref, y_ref 不变
```

参数（`Δv`、`v_max`、`lane_width`）在 `config.py` 中可配置。

## 7. 纵向 MPC

### 7.1 模型：三阶积分器

状态: `x = [s, v, a]ᵀ`（位移、速度、加速度）
控制: `u = j`（jerk）
输出: `a_des = x[2]`

连续时间:
```
ṡ = v
v̇ = a
ȧ = j
```

精确离散化（dt = 0.1s）:
```
x(k+1) = Ad · x(k) + Bd · u(k)

Ad = [1   dt   0.5·dt²]     Bd = [(1/6)·dt³]
     [0   1    dt      ]          [0.5·dt²  ]
     [0   0    1       ]          [dt        ]
```

### 7.2 约束

| 变量 | 最小值 | 最大值 |
|---|---|---|
| a（加速度） | -4 m/s² | 2 m/s² |
| j（jerk） | -5 m/s³ | 5 m/s³ |
| v（速度） | 0 m/s | 40 m/s |

### 7.3 代价函数

```
J = Σ (v(k) - v_ref)² · Q_v + a(k)² · Q_a + j(k)² · R_j
```

### 7.4 求解器

CasADi + IPOPT，预测 horizon N=20，dt=0.1s。

## 8. 横向 MPC

### 8.1 模型：运动学自行车

```
ẋ = v · cos(ψ)
ẏ = v · sin(ψ)
ψ̇ = v / L · tan(δ)
```

状态: `[x, y, ψ]`（x 坐标、y 坐标、航向角）
控制: `δ`（前轮转角）
参数: `L`（轴距），`v`（来自纵向状态，作为参数传入）

### 8.2 约束

| 变量 | 最小值 | 最大值 |
|---|---|---|
| δ（转向角） | -0.5 rad | 0.5 rad |
| δ̇（转向速率） | -0.3 rad/s | 0.3 rad/s |

### 8.3 代价函数

```
J = Σ (y(k) - y_ref)² · Q_y + (ψ(k) - ψ_ref)² · Q_ψ + δ(k)² · R_δ
```

### 8.4 求解器

CasADi + IPOPT，预测 horizon N=15，dt=0.1s。CasADi 原生支持非线性模型，无需手动线性化。

## 9. 训练流程

1. 初始化 highway-env（选定场景）
2. PPO 观测状态，输出离散动作
3. Action Mapper 转换为 (v_ref, y_ref)
4. 纵向 MPC 计算 a_des 跟踪 v_ref
5. 横向 MPC 计算 δ 跟踪 y_ref
6. env.step([δ, a_des])
7. 收集奖励，更新 PPO

PPO 更新使用 GAE（λ=0.95），clip 比率 ε=0.2，每个 episode 训练 10 个 epoch。

## 10. 依赖

- `gymnasium` + `highway-env`: 仿真环境
- `casadi`: MPC 求解器（pip install casadi）
- `torch`: PPO 神经网络
- `numpy`, `matplotlib`: 数据处理与可视化

## 11. 交付物

1. 7 个 highway-env 场景上可运行的 RL+MPC 闭环
2. 训练曲线展示 PPO 收敛过程
3. MPC 跟踪性能可视化（v vs v_ref, y vs y_ref）
4. CARLA wrapper stub 可供后续集成
5. 设计原理文档
